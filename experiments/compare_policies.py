"""Controlled policy benchmark. Stages separate training decisions from final holdouts."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits

from credit_rl.config import SimulationConfig
from credit_rl.risk.longitudinal import LongitudinalPDModel
from credit_rl.evaluation.scenarios import make_scenarios, assert_disjoint
from credit_rl.evaluation.policy_engine import evaluate_policy
from credit_rl.evaluation.policy_statistics import summarize, paired_comparisons, robustness
from credit_rl.policies.registry import baseline_specs, ppo_spec
from credit_rl.policies.training import train_agent
from .common import write_manifest


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def run_id(config, settings):
    return hashlib.sha256(json.dumps(dict(simulation=asdict(config), settings=settings,
        training_sha256=hashlib.sha256(Path(inspect.getfile(train_agent)).read_bytes()).hexdigest(),
        pd_sha256=hashlib.sha256(Path(settings["pd_model"]).read_bytes()).hexdigest()), sort_keys=True).encode()).hexdigest()


def audit(config, settings, pd_model, results):
    scenarios = make_scenarios(config, settings, "train", "markov", customers=300)
    specs = baseline_specs(config, settings, include_oracle=False)
    frames = []
    for spec in [s for s in specs if s.name in ("Static", "Random", "MyopicEconomic")]:
        _, history, _ = evaluate_policy(spec, scenarios, config, pd_model, settings, "audit_train")
        frames.append(history[history.month > 0])
    data = pd.concat(frames, ignore_index=True)
    columns = ["reward_interest_income", "reward_fee_income", "reward_credit_loss", "reward_funding_cost",
               "reward_capital_cost", "reward_constraint_penalty", "reward"]
    description = data[columns].describe(percentiles=[.01, .05, .5, .95, .99]).T
    description.to_csv(results/"reward_audit.csv")
    data[columns+["policy", "customer_id", "month"]].to_csv(results/"reward_audit_transitions.csv.gz", index=False)
    if not np.isfinite(data[columns]).all().all():
        raise ValueError("Nonfinite reward audit")
    print(f"Reward audit: {len(data)} transitions; raw EUR; fixed kEUR PPO scale={settings['ppo']['reward_scale']}", flush=True)
    print(description[["mean", "std", "1%", "50%", "99%"]].to_string(), flush=True)


def tune_thresholds(config, settings, pd_model, validation, results):
    rows = []
    for low, high in settings["heuristics"]["threshold_candidates"]:
        for spec in baseline_specs(config, settings, (low, high), include_oracle=False):
            if spec.name not in ("PDThreshold", "UtilizationPD"):
                continue
            e, _, _ = evaluate_policy(spec, validation, config, pd_model, settings, "validation", keep_history=False)
            rows.append(dict(policy=spec.name, low=low, high=high, score=float(e.discounted_reward.mean())))
    frame = pd.DataFrame(rows)
    frame.to_csv(results/"threshold_validation.csv", index=False)
    # One common threshold pair for both rules; mean validation objective, no test access.
    scores = frame.groupby(["low", "high"], sort=True).score.mean()
    low, high = scores.idxmax()
    return [float(low), float(high)]


def train(config, settings, pd_model, results, models, identity):
    train_set = make_scenarios(config, settings, "train", "markov")
    validation = make_scenarios(config, settings, "validation", "baseline")
    assert_disjoint(train_set, validation)
    selection_file = results/"selection.json"
    if selection_file.exists():
        selection = json.loads(selection_file.read_text())
        if selection["run_id"] != identity:
            raise ValueError("Existing training selection has different inputs; use another --output")
    else:
        thresholds = tune_thresholds(config, settings, pd_model, validation, results)
        pilots = []
        for index, candidate in enumerate(settings["ppo"]["candidates"]):
            metadata = train_agent(config, pd_model, settings, train_set, validation,
                seed=17, destination=models/f"pilot_{index}", overrides=candidate,
                total_timesteps=settings["ppo"]["pilot_timesteps"])
            pilots.append(dict(candidate=index, score=metadata["validation_score"], **candidate))
        pd.DataFrame(pilots).to_csv(results/"pilot_validation.csv", index=False)
        selected = max(pilots, key=lambda row: row["score"])["candidate"]
        selection = dict(run_id=identity, thresholds=thresholds,
                         ppo_overrides=settings["ppo"]["candidates"][selected], selected_candidate=selected)
        json_write(selection_file, selection)
    for without_pd in (False, True):
        for seed in settings["ppo"]["seeds"]:
            destination = models/f"{'without_pd' if without_pd else 'with_pd'}_{seed}"
            meta_path = destination/"metadata.json"
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text())
                if metadata.get("run_id") != identity:
                    raise ValueError("Saved agent input mismatch; use another --output")
                print(f"Reusing completed {destination.name}", flush=True)
                continue
            metadata = train_agent(config, pd_model, settings, train_set, validation,
                seed=seed, destination=destination, without_pd=without_pd, overrides=selection["ppo_overrides"])
            metadata["run_id"] = identity
            json_write(meta_path, metadata)


def load_specs(config, settings, results, models, identity):
    selection = json.loads((results/"selection.json").read_text())
    if selection["run_id"] != identity:
        raise ValueError("Evaluation configuration differs from training")
    specs = baseline_specs(config, settings, selection["thresholds"])
    for without_pd in (False, True):
        for seed in settings["ppo"]["seeds"]:
            folder = models/f"{'without_pd' if without_pd else 'with_pd'}_{seed}"
            metadata = json.loads((folder/"metadata.json").read_text())
            if metadata.get("run_id") != identity:
                raise ValueError("Agent provenance mismatch")
            path = folder/"selected.zip"
            model = PPO.load(path, device="cpu")
            specs.append(ppo_spec(model, seed, path, without_pd))
    return specs


def evaluate(config, settings, pd_model, results, models, identity):
    specs = load_specs(config, settings, results, models, identity)
    groups = [make_scenarios(config, settings, name, "baseline") for name in ("train", "validation", "test", "oot")]
    assert_disjoint(*groups)
    json_write(results/"population_ids.json", {name: [s.customer_id for s in group]
        for name, group in zip(("train", "validation", "test", "oot"), groups)})
    trace_dir = results/"trajectories"
    trace_dir.mkdir(exist_ok=True)
    all_episodes, benchmarks = [], []
    for macro in settings["evaluation"]["macro_scenarios"] + ["oot_markov", "baseline_biased_pd"]:
        partition = "oot" if macro == "oot_markov" else "test"
        path_name = "markov" if macro == "oot_markov" else "baseline" if macro == "baseline_biased_pd" else macro
        scenarios = make_scenarios(config, settings, partition, path_name)
        for spec in specs:
            # Sensitivity isolates the actor's PD channel; economics uses the unmodified risk model.
            current = replace(spec, pd_multiplier=settings["evaluation"]["sensitivity_pd_multiplier"]) if macro == "baseline_biased_pd" else spec
            if macro == "baseline_biased_pd" and spec.name not in ("PPO", "PDThreshold", "MyopicEconomic"):
                continue
            key = f"{macro}__{spec.name}__{spec.seed}"
            e, history, performance = evaluate_policy(current, scenarios, config, pd_model, settings, macro)
            e.to_csv(trace_dir/f"{key}.episodes.csv", index=False)
            history.to_csv(trace_dir/f"{key}.csv.gz", index=False)
            all_episodes.append(e)
            benchmarks.append(dict(policy=spec.name, seed=spec.seed, scenario=macro, **performance))
            print(f"Evaluated {key}: value={e.net_economic_value.mean():.1f} default={e.defaulted.mean():.3f}", flush=True)
    episodes = pd.concat(all_episodes, ignore_index=True)
    episodes.to_csv(results/"episode_metrics.csv", index=False)
    summary, seed_metrics = summarize(episodes, settings)
    summary.to_csv(results/"summary.csv", index=False)
    seed_metrics.to_csv(results/"seed_metrics.csv", index=False)
    paired, distributions = paired_comparisons(episodes, settings)
    paired.to_csv(results/"paired_comparisons.csv", index=False)
    distributions.to_csv(results/"paired_distributions.csv.gz", index=False)
    robustness(summary).to_csv(results/"robustness.csv", index=False)
    pd.DataFrame(benchmarks).to_csv(results/"evaluation_benchmark.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/policy_evaluation.yaml"))
    parser.add_argument("--simulation", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--stage", choices=["audit", "smoke", "train", "evaluate", "figures", "all"], default="all")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    config = SimulationConfig.from_yaml(args.simulation)
    pd_model = LongitudinalPDModel.load(settings["pd_model"])
    results, models, figures = [args.output/name/"policy_evaluation" for name in ("results", "models", "figures")]
    for folder in (results, models, figures):
        folder.mkdir(parents=True, exist_ok=True)
    identity = run_id(config, settings)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.stage in ("audit", "all"):
            audit(config, settings, pd_model, results)
        if args.stage == "smoke":
            train_set = make_scenarios(config, settings, "train", "markov", customers=30)
            val_set = make_scenarios(config, settings, "validation", "baseline", customers=10)
            train_agent(config, pd_model, settings, train_set, val_set, seed=7, destination=models/"smoke", total_timesteps=1024)
        if args.stage in ("train", "all"):
            if not (results/"reward_audit.csv").exists():
                audit(config, settings, pd_model, results)
            train(config, settings, pd_model, results, models, identity)
        if args.stage in ("evaluate", "all"):
            evaluate(config, settings, pd_model, results, models, identity)
        if args.stage in ("figures", "all"):
            from credit_rl.evaluation.policy_reporting import report
            report(config, settings, pd_model, results, models, figures, identity)
    write_manifest(results/f"manifest_{args.stage}.json", config, settings, run_id=identity,
                   pd_sha256=hashlib.sha256(Path(settings["pd_model"]).read_bytes()).hexdigest(), stage=args.stage)


if __name__ == "__main__":
    main()
