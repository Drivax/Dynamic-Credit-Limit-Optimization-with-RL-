"""Paired static/increase/reduce trajectories, latent-risk and macro diagnostics."""

import argparse
import os
from dataclasses import replace
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.evaluation.metrics import summarize_trajectories
from credit_rl.policies.baselines import ConstantPolicy, RiskThresholdPolicy, StaticPolicy
from credit_rl.risk.pd_model import SnapshotPDModel
from credit_rl.risk.training import train_risk_model
from credit_rl.simulation.customer import MacroState, initialize_customer
from credit_rl.simulation.simulator import simulate_customer
from credit_rl.simulation.synthetic_snapshot import generate_synthetic_portfolio
from .common import load_run_config, write_manifest


def plot_customer(history: pd.DataFrame, title: str, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    fields = [("credit_limit", "Credit limit (EUR)"), ("balance", "Balance (EUR)"),
              ("utilization", "Utilization"), ("predicted_pd", "Predicted monthly PD"),
              ("cumulative_reward", "Cumulative reward (EUR)"),
              ("months_delinquent", "Consecutive delinquent months")]
    for name, group in history.groupby("policy", sort=False):
        group = group.sort_values("month").copy()
        group["cumulative_reward"] = group.reward.cumsum()
        for ax, (field, label) in zip(axes.flat, fields):
            ax.plot(group.month, group[field], label=name, linewidth=1.6)
            if group.iloc[-1].defaulted:
                ax.scatter(group.month.iloc[-1], group[field].iloc[-1], marker="x", s=55)
            ax.set(xlabel="Month", ylabel=label)
            ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title + "\nCrosses mark terminal default; lines stop at default/horizon")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(config: SimulationConfig, settings: dict, output: Path) -> dict:
    results = output / "results" / "trajectory_sanity"
    figures = output / "figures" / "trajectory_sanity"
    models = output / "models" / "trajectory_sanity"
    for folder in (results, figures, models):
        folder.mkdir(parents=True, exist_ok=True)
    seed = settings["seed"]
    # Snapshot training and evaluation populations use distinct seeds.
    training = generate_synthetic_portfolio(settings["risk_training_clients"], seed=seed)
    classifier, risk_report = train_risk_model(training, random_state=seed)
    pd_model = SnapshotPDModel(classifier, config.pd)
    joblib.dump(classifier, models / "snapshot_pd.joblib")
    risk_report["validation_predictions"].to_csv(results / "snapshot_pd_validation.csv", index=False)
    risk_report["feature_importance"].to_csv(results / "snapshot_pd_importance.csv", index=False)
    portfolio = generate_synthetic_portfolio(settings["customers"], seed=seed + 1)
    policies = {
        "static": StaticPolicy(config.environment),
        "increase_10pct": ConstantPolicy(config.environment.action_multipliers.index(1.1)),
        "reduce_10pct": ConstantPolicy(config.environment.action_multipliers.index(0.9)),
        "risk_threshold": RiskThresholdPolicy(config.environment),
    }
    frames = []
    for customer in range(len(portfolio)):
        for name, policy in policies.items():
            env = CreditLimitEnv(portfolio, pd_model, config)
            frame = simulate_customer(env, policy, seed=seed + 1000 + customer,
                                      options={"customer_index": customer}).to_dataframe()
            frame["policy"], frame["episode_id"] = name, customer
            frames.append(frame)
    trajectories = pd.concat(frames, ignore_index=True)
    trajectories.to_csv(results / "trajectories.csv", index=False)
    summary = summarize_trajectories(trajectories)
    summary.to_csv(results / "policy_summary.csv", index=False)
    for customer in range(min(3, len(portfolio))):
        plot_customer(trajectories[trajectories.episode_id == customer], f"Customer {customer}: paired policies",
                      figures / f"customer_{customer}.png")

    state, traits = initialize_customer(portfolio.iloc[0], "controlled", config, np.random.default_rng(seed))
    latent_frames = []
    for name, customer_traits in (("reference", traits), ("weak_latent", replace(traits,
            creditworthiness=traits.creditworthiness - 2, payment_propensity=traits.payment_propensity / 2))):
        frame = simulate_customer(CreditLimitEnv(pd_model=pd_model, config=config), StaticPolicy(config.environment),
            seed=seed + 3000, options={"initial_state": state, "traits": customer_traits}).to_dataframe()
        frame["policy"], frame["episode_id"] = name, 0
        latent_frames.append(frame)
    latent_history = pd.concat(latent_frames, ignore_index=True)
    latent_history.to_csv(results / "latent_risk_trajectories.csv", index=False)
    plot_customer(latent_history, "Same observed initial customer, different hidden traits", figures / "latent_risk.png")

    # Lock macro regimes for an explicit stress intervention. Both transitions
    # are disabled; all other coefficients, customers and random shocks match.
    locked = replace(config, dynamics=replace(config.dynamics, normal_to_stress=0, stress_to_normal=0))
    stress_rows = []
    for trial in range(settings["stress_trials"]):
        for name, macro in (("normal", MacroState.NORMAL), ("stress", MacroState.STRESS)):
            frame = simulate_customer(CreditLimitEnv(pd_model=pd_model, config=locked), StaticPolicy(config.environment),
                seed=seed + 4000 + trial,
                options={"initial_state": state, "traits": traits, "macro_state": macro}).to_dataframe()
            stress_rows.append({"regime": name, "trial": trial, "defaulted": bool(frame.defaulted.iloc[-1]),
                "return": float(frame.reward.sum()), "months": len(frame) - 1,
                "first_payment_ratio": frame.payment_ratio.iloc[1],
                "first_delinquent": frame.delinquency_status.iloc[1]})
    stress_data = pd.DataFrame(stress_rows)
    stress_data.to_csv(results / "macro_trials.csv", index=False)
    stress_summary = stress_data.groupby("regime").agg(
        trials=("trial", "size"), default_incidence=("defaulted", "mean"),
        mean_return=("return", "mean"), first_payment_ratio=("first_payment_ratio", "mean"),
        first_delinquency_rate=("first_delinquent", "mean"))
    stress_summary.to_csv(results / "macro_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    axes[0].bar(stress_summary.index, stress_summary.default_incidence)
    axes[0].set(ylabel="Default incidence by horizon", ylim=(0, 1))
    axes[1].bar(stress_summary.index, stress_summary.first_payment_ratio)
    axes[1].set(ylabel="Mean first-month payment ratio", ylim=(0, 1))
    fig.suptitle("Paired locked-regime diagnostics; one initial customer, repeated shocks")
    fig.savefig(figures / "macro_stress.png", dpi=150)
    plt.close(fig)
    write_manifest(results / "manifest.json", config, settings,
                   pd_model="Snapshot GradientBoosting; domain shift is not calibrated",
                   snapshot_validation=risk_report["metrics"],
                   controlled_initial_state=replace(state, macro_state=MacroState.NORMAL).__dict__,
                   controlled_traits=traits.__dict__,
                   note="Controlled traits are experiment metadata, never policy input or env info/history.")
    print(summary.to_string(index=False))
    print(stress_summary.to_string())
    return {"trajectories": trajectories, "policy_summary": summary, "macro_summary": stress_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--experiments", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(SimulationConfig.from_yaml(args.config), load_run_config(args.experiments), args.output)


if __name__ == "__main__":
    main()
