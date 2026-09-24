"""Canonical customer experiment: generate PD data, train, evaluate and report."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
import yaml

from credit_rl.config import SimulationConfig
from credit_rl.evaluation.policy_engine import evaluate_policy
from credit_rl.evaluation.policy_statistics import summarize, paired_comparisons
from credit_rl.evaluation.scenarios import make_scenarios, assert_disjoint
from credit_rl.policies.registry import baseline_specs, ppo_spec
from credit_rl.policies.training import train_agent
from credit_rl.risk.longitudinal import LongitudinalPDModel
from credit_rl.risk.settings import load_settings
from .common import write_manifest
from .pd_training import run as train_pd


def settings_for(profile, config_dir):
    config_dir = Path(config_dir)
    config = SimulationConfig.from_yaml(config_dir / "simulation.yaml")
    profile_config = yaml.safe_load((config_dir / "main_evaluation.yaml").read_text())[profile]
    settings = yaml.safe_load((config_dir / "policy_evaluation.yaml").read_text())
    pd_settings = load_settings(config_dir / "pd_model.yaml", config)
    p = profile_config
    if p["pd_customers"] is not None:
        pd_settings["split"] = dict.fromkeys(pd_settings["split"], p["pd_customers"])
    pd_settings["models"]["boosting_iterations"] = p["pd_boosting_iterations"]
    pd_settings["evaluation"]["bootstrap_repetitions"] = p["bootstrap_repetitions"]
    for role in ("train", "validation", "test"):
        settings["population"][role] = p[role + "_customers"]
    settings["ppo"]["seeds"] = p["ppo_seeds"]
    for key in ("total_timesteps", "n_steps", "batch_size", "validation_every"):
        settings["ppo"][key] = p[key]
    settings["evaluation"]["bootstrap_repetitions"] = p["bootstrap_repetitions"]
    settings["evaluation"]["macro_scenarios"] = ["baseline", "severe_stress"]
    # Fixed declared parameters; this compact protocol does not run pilot tuning.
    return config, settings, pd_settings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(output):
    output = Path(output)
    summary = pd.read_csv(output / "results/summary.csv")
    required = {"Static", "PDThreshold", "MyopicEconomic", "PPO", "AlwaysDecrease20"}
    if set(summary.policy) != required or set(summary.scenario) != {"baseline", "severe_stress"}:
        raise ValueError("Incomplete policy/scenario panel")
    for column in ("net_economic_value", "revenue", "credit_loss", "default_rate", "mean_limit", "mean_utilization"):
        if not np.isfinite(summary[column]).all():
            raise ValueError(f"Nonfinite {column}")
    if not summary.default_rate.between(0, 1).all():
        raise ValueError("Invalid default rates")
    for name in ("paired_comparisons.csv", "episode_metrics.csv", "pd/metrics.csv"):
        if not (output / "results" / name).is_file():
            raise ValueError(f"Missing {name}")
    return summary


def run(profile="smoke", output=None, config_dir="configs", stage="all"):
    output = Path(output or f"outputs/main/{profile}")
    config, settings, pd_settings = settings_for(profile, config_dir)
    for name in ("results", "models", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    identity_data = dict(simulation=asdict(config), policy=settings, pd=pd_settings)
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        raise ValueError("Inputs changed; use a fresh --output directory")
    if stage == "figures":
        from .report_assets import report
        verify(output)
        report(output)
        return output
    source_hashes = {str(p.relative_to(Path(__file__).parents[1])): digest(p)
                     for p in sorted(Path(__file__).parents[1].rglob("*.py"))}
    def scientific_sources(hashes):
        # Rendering/orchestration edits do not invalidate fitted estimators;
        # expanded experiment parameters are already part of identity above.
        return {key: value for key, value in hashes.items()
                if not key.replace("\\", "/").startswith("experiments/")
                and "reporting.py" not in key}
    if manifest.exists() and scientific_sources(json.loads(manifest.read_text()).get("package_sha256", {})) != scientific_sources(source_hashes):
        raise ValueError("Source changed; use a fresh --output directory")
    write_manifest(manifest, config, identity_data, identity=identity, profile=profile,
                   experiment="main_customer_evaluation", package_sha256=source_hashes)
    pd_path = output / "models/pd/logistic_calibrated.joblib"
    with threadpool_limits(limits=1):
        if not pd_path.exists() or not (output / "results/pd/manifest.json").exists():
            train_pd(config, pd_settings, output)
        risk = LongitudinalPDModel.load(pd_path)
        train = make_scenarios(config, settings, "train", "markov")
        validation = make_scenarios(config, settings, "validation", "baseline")
        test = make_scenarios(config, settings, "test", "baseline")
        assert_disjoint(train, validation, test)
        (output / "results/population_ids.json").write_text(json.dumps({
            role: [s.customer_id for s in group] for role, group in
            (("train", train), ("validation", validation), ("test", test))}, indent=2))
        specs = [s for s in baseline_specs(config, settings, include_oracle=False)
                 if s.name in ("Static", "PDThreshold", "MyopicEconomic", "AlwaysDecrease20")]
        hashes = {"pd": digest(pd_path)}
        for seed in settings["ppo"]["seeds"]:
            folder = output / "models" / f"ppo_{seed}"
            path = folder / "selected.zip"
            if not (folder / "metadata.json").exists():
                train_agent(config, risk, settings, train, validation, seed=seed, destination=folder)
            hashes[f"ppo_{seed}"] = digest(path)
            specs.append(ppo_spec(PPO.load(path, device="cpu"), seed, f"models/ppo_{seed}/selected.zip"))
        old_hashes = output / "results/model_hashes.json"
        if old_hashes.exists() and json.loads(old_hashes.read_text()) != hashes:
            raise ValueError("Frozen model artifact changed")
        old_hashes.write_text(json.dumps(hashes, indent=2))
        if stage == "train":
            return output
        episodes, histories = [], []
        for macro in settings["evaluation"]["macro_scenarios"]:
            scenarios = make_scenarios(config, settings, "test", macro)
            for spec in specs:
                e, history, _ = evaluate_policy(spec, scenarios, config, risk, settings, macro)
                episodes.append(e)
                histories.append(history)
                print(f"{macro} {spec.name} seed={spec.seed}: {e.net_economic_value.mean():.2f} EUR", flush=True)
        episodes = pd.concat(episodes, ignore_index=True)
        episodes.to_csv(output / "results/episode_metrics.csv", index=False)
        pd.concat(histories, ignore_index=True).to_csv(output / "results/trajectories.csv.gz", index=False)
        summary, seeds = summarize(episodes, settings)
        summary.to_csv(output / "results/summary.csv", index=False)
        seeds.to_csv(output / "results/seed_metrics.csv", index=False)
        paired, _ = paired_comparisons(episodes, settings)
        paired.to_csv(output / "results/paired_comparisons.csv", index=False)
        verify(output)
        from .report_assets import report
        report(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard"], default="smoke")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--stage", choices=["all", "train", "evaluate", "figures"], default="all")
    args = parser.parse_args()
    run(args.profile, args.output, args.config_dir, args.stage)


if __name__ == "__main__":
    main()
