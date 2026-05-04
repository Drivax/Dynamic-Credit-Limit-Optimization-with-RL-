from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from credit_limit_rl.config import PortfolioConfig
from credit_limit_rl.data import generate_synthetic_portfolio
from credit_limit_rl.env import ACTION_ADJUSTMENTS, ACTION_LABELS, CreditLimitEnv, build_observation, simulate_credit_decision
from credit_limit_rl.risk import train_risk_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_policy(
    policy_name: str,
    portfolio: pd.DataFrame,
    risk_model,
    config: PortfolioConfig,
    model: PPO | None = None,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    rng = np.random.default_rng(123)
    rows: list[dict[str, float | int | str]] = []

    for _, client in portfolio.iterrows():
        if model is None:
            action_index = 2
        else:
            action_index, _ = model.predict(build_observation(client), deterministic=True)
            action_index = int(action_index)

        outcome = simulate_credit_decision(client, float(ACTION_ADJUSTMENTS[action_index]), risk_model, config, rng)
        rows.append(
            {
                "policy": policy_name,
                "action_index": action_index,
                "action_label": ACTION_LABELS[action_index],
                "adjustment": float(ACTION_ADJUSTMENTS[action_index]),
                "reward": float(outcome["reward"]),
                "defaulted": int(outcome["defaulted"]),
                "predicted_pd": float(outcome["predicted_pd"]),
                "new_limit": float(outcome["new_limit"]),
                "monthly_interest": float(outcome["monthly_interest"]),
                "fee_income": float(outcome["fee_income"]),
                "expected_loss": float(outcome["expected_loss"]),
                "rwa_cost": float(outcome["rwa_cost"]),
                "constraint_penalty": float(outcome["constraint_penalty"]),
            }
        )

    details = pd.DataFrame(rows)
    summary = {
        "policy": policy_name,
        "avg_reward": float(details["reward"].mean()),
        "median_reward": float(details["reward"].median()),
        "reward_std": float(details["reward"].std(ddof=0)),
        "reward_p05": float(details["reward"].quantile(0.05)),
        "reward_p95": float(details["reward"].quantile(0.95)),
        "default_rate": float(details["defaulted"].mean()),
        "avg_predicted_pd": float(details["predicted_pd"].mean()),
        "avg_limit": float(details["new_limit"].mean()),
        "avg_adjustment": float(details["adjustment"].mean()),
        "avg_monthly_interest": float(details["monthly_interest"].mean()),
        "avg_fee_income": float(details["fee_income"].mean()),
        "avg_expected_loss": float(details["expected_loss"].mean()),
        "avg_rwa_cost": float(details["rwa_cost"].mean()),
        "avg_constraint_penalty": float(details["constraint_penalty"].mean()),
        "portfolio_reward": float(details["reward"].sum()),
    }
    return summary, details


def save_risk_charts(validation_predictions: pd.DataFrame, feature_importance: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    chart_paths: dict[str, str] = {}
    y_true = validation_predictions["default_next_month"].to_numpy()
    y_score = validation_predictions["predicted_pd"].to_numpy()

    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    roc_path = output_dir / "risk_roc_curve.png"
    plt.figure(figsize=(7.0, 5.0))
    plt.plot(fpr, tpr, label="Gradient Boosting")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title("Risk Model ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=180)
    plt.close()
    chart_paths["risk_roc_curve"] = str(roc_path)

    pr_path = output_dir / "risk_precision_recall_curve.png"
    plt.figure(figsize=(7.0, 5.0))
    plt.plot(recall, precision, label="Gradient Boosting")
    plt.title("Risk Model Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path, dpi=180)
    plt.close()
    chart_paths["risk_precision_recall_curve"] = str(pr_path)

    importance_path = output_dir / "risk_feature_importance.png"
    top_features = feature_importance.head(8).iloc[::-1]
    plt.figure(figsize=(8.0, 5.5))
    plt.barh(top_features["feature"], top_features["importance"], color="#1f77b4")
    plt.title("Top Risk Drivers")
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(importance_path, dpi=180)
    plt.close()
    chart_paths["risk_feature_importance"] = str(importance_path)

    return chart_paths


def save_policy_charts(
    policy_summaries: pd.DataFrame,
    ppo_details: pd.DataFrame,
    output_dir: Path,
    config: PortfolioConfig,
) -> dict[str, str]:
    chart_paths: dict[str, str] = {}

    comparison_path = output_dir / "policy_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    axes[0].bar(policy_summaries["policy"], policy_summaries["avg_reward"], color=["#6c757d", "#2a9d8f"])
    axes[0].set_title("Average Reward")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(policy_summaries["policy"], policy_summaries["default_rate"] * 100.0, color=["#6c757d", "#e76f51"])
    axes[1].set_title("Default Rate")
    axes[1].set_ylabel("Percent")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(policy_summaries["policy"], policy_summaries["avg_limit"], color=["#6c757d", "#457b9d"])
    axes[2].set_title("Average Granted Limit")
    axes[2].set_ylabel("EUR")
    axes[2].tick_params(axis="x", rotation=15)

    fig.suptitle("Policy Backtest Comparison")
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)
    chart_paths["policy_comparison"] = str(comparison_path)

    action_path = output_dir / "ppo_action_distribution.png"
    action_counts = (
        ppo_details["action_label"]
        .value_counts()
        .reindex(ACTION_LABELS, fill_value=0)
    )
    plt.figure(figsize=(7.5, 5.0))
    plt.bar(action_counts.index, action_counts.values, color="#264653")
    plt.title("PPO Action Distribution on Test Portfolio")
    plt.xlabel("Action")
    plt.ylabel("Clients")
    plt.tight_layout()
    plt.savefig(action_path, dpi=180)
    plt.close()
    chart_paths["ppo_action_distribution"] = str(action_path)

    waterfall_path = output_dir / "ppo_reward_components.png"
    component_means = ppo_details[
        ["monthly_interest", "fee_income", "expected_loss", "rwa_cost", "constraint_penalty"]
    ].mean()
    signed_components = pd.Series(
        {
            "Interest": component_means["monthly_interest"],
            "Fees": component_means["fee_income"],
            "Expected loss": -config.lambda_default * component_means["expected_loss"],
            "RWA cost": -config.lambda_rwa * component_means["rwa_cost"],
            "Risk penalty": -component_means["constraint_penalty"],
        }
    )
    plt.figure(figsize=(8.0, 5.0))
    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in signed_components.values]
    plt.bar(signed_components.index, signed_components.values, color=colors)
    plt.title("Average PPO Reward Components")
    plt.ylabel("EUR per client-step")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=180)
    plt.close()
    chart_paths["ppo_reward_components"] = str(waterfall_path)

    return chart_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a dynamic credit limit RL policy.")
    parser.add_argument("--clients", type=int, default=50_000, help="Number of synthetic clients to generate.")
    parser.add_argument("--train-timesteps", type=int, default=200_000, help="PPO training timesteps.")
    parser.add_argument("--episode-length", type=int, default=512, help="Episode length for the custom environment.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Folder where trained artifacts are saved.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Folder where charts and performance snapshots are saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    output_dir = args.output_dir
    results_dir = args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.clients:,} synthetic clients...")
    portfolio = generate_synthetic_portfolio(n_clients=args.clients, seed=args.seed)
    split_index = int(len(portfolio) * 0.8)
    train_portfolio = portfolio.iloc[:split_index].reset_index(drop=True)
    test_portfolio = portfolio.iloc[split_index:].reset_index(drop=True)

    risk_model, risk_report = train_risk_model(train_portfolio, random_state=args.seed)
    risk_metrics = risk_report["metrics"]
    print("Risk model metrics:")
    print(json.dumps(risk_metrics, indent=2))

    config = PortfolioConfig()
    train_env = DummyVecEnv([
        lambda: CreditLimitEnv(
            portfolio=train_portfolio,
            risk_model=risk_model,
            config=config,
            episode_length=args.episode_length,
            seed=args.seed,
        )
    ])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        gamma=0.98,
        learning_rate=2.5e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        ent_coef=0.02,
        clip_range=0.2,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
    )
    print(f"Training PPO for {args.train_timesteps:,} timesteps...")
    model.learn(total_timesteps=args.train_timesteps, progress_bar=False)

    model_path = output_dir / "ppo_credit_limit"
    risk_model_path = output_dir / "risk_model.joblib"
    sample_path = output_dir / "sample_clients.csv"
    metrics_path = output_dir / "metrics.json"
    policy_details_path = results_dir / "policy_decisions.csv"
    feature_importance_path = results_dir / "risk_feature_importance.csv"
    validation_predictions_path = results_dir / "risk_validation_predictions.csv"

    model.save(str(model_path))
    joblib.dump(risk_model, risk_model_path)
    test_portfolio.head(1_000).to_csv(sample_path, index=False)
    risk_report["feature_importance"].to_csv(feature_importance_path, index=False)
    risk_report["validation_predictions"].to_csv(validation_predictions_path, index=False)

    static_summary, static_details = evaluate_policy("static_maintain", test_portfolio, risk_model, config, model=None)
    ppo_summary, ppo_details = evaluate_policy("ppo_dynamic", test_portfolio, risk_model, config, model=model)
    evaluation = [static_summary, ppo_summary]
    policy_details = pd.concat([static_details, ppo_details], ignore_index=True)
    policy_details.to_csv(policy_details_path, index=False)

    risk_chart_paths = save_risk_charts(risk_report["validation_predictions"], risk_report["feature_importance"], results_dir)
    policy_chart_paths = save_policy_charts(pd.DataFrame(evaluation), ppo_details, results_dir, config)
    reward_lift = float(ppo_summary["portfolio_reward"] - static_summary["portfolio_reward"])
    reward_lift_pct = float(reward_lift / abs(static_summary["portfolio_reward"])) if static_summary["portfolio_reward"] else 0.0
    elapsed_seconds = float(time.perf_counter() - start_time)
    metrics = {
        "run_config": {
            "clients": args.clients,
            "train_timesteps": args.train_timesteps,
            "episode_length": args.episode_length,
            "seed": args.seed,
            "train_runtime_seconds": elapsed_seconds,
        },
        "risk_model": risk_metrics,
        "policy_comparison": evaluation,
        "policy_lift": {
            "portfolio_reward_delta": reward_lift,
            "portfolio_reward_delta_pct": reward_lift_pct,
            "default_rate_delta": float(ppo_summary["default_rate"] - static_summary["default_rate"]),
            "avg_limit_delta": float(ppo_summary["avg_limit"] - static_summary["avg_limit"]),
        },
        "ppo_action_distribution": ppo_details["action_label"].value_counts().reindex(ACTION_LABELS, fill_value=0).to_dict(),
        "generated_files": {
            "policy_decisions_csv": policy_details_path.as_posix(),
            "risk_feature_importance_csv": feature_importance_path.as_posix(),
            "risk_validation_predictions_csv": validation_predictions_path.as_posix(),
            **{key: Path(value).as_posix() for key, value in risk_chart_paths.items()},
            **{key: Path(value).as_posix() for key, value in policy_chart_paths.items()},
        },
        "actions": dict(zip(ACTION_LABELS, ACTION_ADJUSTMENTS.tolist())),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Policy comparison:")
    print(pd.DataFrame(evaluation).to_string(index=False))
    print(f"Artifacts saved in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
