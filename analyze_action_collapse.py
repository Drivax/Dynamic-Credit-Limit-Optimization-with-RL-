"""
Diagnostic script to analyze PPO action collapse and reward signal issues.

This script:
1. Inspects the trained policy's action distribution across the portfolio
2. Compares state values across all 5 action bins
3. Checks if rewards are properly scaled and triggering penalties
4. Compares against multiple baseline strategies
5. Visualizes pathologies
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from credit_limit_rl.config import PortfolioConfig
from credit_limit_rl.data import generate_synthetic_portfolio
from credit_limit_rl.env import ACTION_ADJUSTMENTS, ACTION_LABELS, CreditLimitEnv, build_observation, simulate_credit_decision
from credit_limit_rl.risk import train_risk_model


def load_model_and_assets(artifacts_dir: Path) -> tuple[PPO, pd.DataFrame, Any, PortfolioConfig]:
    """Load trained PPO model, portfolio, risk model, and config."""
    model_path = artifacts_dir / "ppo_credit_limit.zip"
    portfolio_path = artifacts_dir / "sample_clients.csv"
    risk_model_path = artifacts_dir / "risk_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not portfolio_path.exists():
        raise FileNotFoundError(f"Portfolio not found: {portfolio_path}")
    if not risk_model_path.exists():
        raise FileNotFoundError(f"Risk model not found: {risk_model_path}")

    model = PPO.load(str(model_path), device="cpu")
    portfolio = pd.read_csv(portfolio_path)
    risk_model = joblib.load(str(risk_model_path))
    config = PortfolioConfig()

    print(f"Loaded model, portfolio ({len(portfolio)} clients), risk model, and config.")
    return model, portfolio, risk_model, config


def compute_action_values(
    model: PPO,
    client_obs: np.ndarray,
) -> np.ndarray:
    """
    Extract value estimates for each action from the policy's value head.
    
    For PPO with discrete actions, we can:
    1. Get the current state value (V-function)
    2. Get action logits (π_logits) and compute Q-function approximation
    
    Returns:
        Array of shape (n_actions,) with action values.
    """
    # Get the policy network features and action logits
    with model.policy.predict_net.zero_grad() if hasattr(model.policy, 'predict_net') else DummyVecEnv([]):
        features = model.policy.extract_features(client_obs.reshape(1, -1))
        action_logits = model.policy.action_net(features)
        state_value = model.policy.value_net(features)

    action_logits_np = action_logits.detach().cpu().numpy().flatten()
    state_value_np = state_value.detach().cpu().numpy().flatten()[0]
    
    # Q(s, a) ≈ V(s) + policy_gradient signal strength
    # For PPO, action logits are not direct Q-values, but they indicate relative preferences
    action_probs = np.exp(action_logits_np) / np.sum(np.exp(action_logits_np))
    
    return action_logits_np, state_value_np, action_probs


def analyze_action_distribution(
    model: PPO,
    portfolio: pd.DataFrame,
    risk_model: Any,
    config: PortfolioConfig,
    sample_size: int = 1000,
) -> pd.DataFrame:
    """
    Analyze PPO's action distribution across the portfolio.
    
    Returns a dataframe with action selections and diagnostics.
    """
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(portfolio), size=min(sample_size, len(portfolio)), replace=False)
    sample_portfolio = portfolio.iloc[sample_indices].reset_index(drop=True)

    rows = []
    for idx, (_, client) in enumerate(sample_portfolio.iterrows()):
        obs = build_observation(client)
        action_idx, _ = model.predict(obs, deterministic=True)
        action_idx = int(action_idx)

        # Get action preferences (logits)
        try:
            action_logits, state_value, action_probs = compute_action_values(model, obs)
        except Exception as e:
            print(f"Warning: Could not compute action values for client {idx}: {e}")
            action_logits = np.array([0.0] * 5)
            state_value = 0.0
            action_probs = np.ones(5) / 5

        # Simulate outcome
        outcome = simulate_credit_decision(client, float(ACTION_ADJUSTMENTS[action_idx]), risk_model, config, rng)

        rows.append({
            "client_idx": idx,
            "action_idx": action_idx,
            "action_label": ACTION_LABELS[action_idx],
            "state_value": state_value,
            "logit_-20%": action_logits[0],
            "logit_-10%": action_logits[1],
            "logit_0%": action_logits[2],
            "logit_+10%": action_logits[3],
            "logit_+20%": action_logits[4],
            "prob_-20%": action_probs[0],
            "prob_-10%": action_probs[1],
            "prob_0%": action_probs[2],
            "prob_+10%": action_probs[3],
            "prob_+20%": action_probs[4],
            "logit_variance": np.var(action_logits),
            "logit_max_min_gap": np.max(action_logits) - np.min(action_logits),
            "internal_score": client["internal_score"],
            "predicted_pd": outcome["predicted_pd"],
            "monthly_interest": outcome["monthly_interest"],
            "reward": outcome["reward"],
            "expected_loss": outcome["expected_loss"],
            "rwa_cost": outcome["rwa_cost"],
            "constraint_penalty": outcome["constraint_penalty"],
        })

    return pd.DataFrame(rows)


def categorize_by_risk(portfolio_analysis: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Segment the portfolio by risk level and analyze action distribution within each segment."""
    pd_quartiles = pd.qcut(portfolio_analysis["predicted_pd"], q=4, labels=["Low", "Med-Low", "Med-High", "High"], duplicates="drop")

    segments = {}
    for label in ["Low", "Med-Low", "Med-High", "High"]:
        mask = pd_quartiles == label
        if mask.any():
            segments[label] = portfolio_analysis[mask]

    return segments


def evaluate_baseline_strategies(
    portfolio: pd.DataFrame,
    risk_model: Any,
    config: PortfolioConfig,
) -> dict[str, dict[str, float]]:
    """
    Evaluate multiple baseline strategies beyond "static maintain".
    
    - static_maintain: always action 2 (0%)
    - increment_low_risk: action 4 (+20%) for clients with PD < median, else action 2
    - decrement_high_risk: action 0 (-20%) for clients with PD > median, else action 2
    - conservative: action 1 (-10%) for all
    - aggressive: action 3 (+10%) for all
    """
    rng = np.random.default_rng(123)
    
    strategies = {
        "static_maintain": lambda x: 2,
        "conservative": lambda x: 1,
        "aggressive": lambda x: 3,
        "decrement_high_risk": lambda x: 0 if x["predicted_pd"] > 0.08 else 2,
        "increment_low_risk": lambda x: 4 if x["predicted_pd"] < 0.08 else 2,
    }

    results = {}

    for strategy_name, policy_fn in strategies.items():
        total_reward = 0.0
        default_count = 0
        total_pd = 0.0
        total_limit = 0.0

        for _, client in portfolio.iterrows():
            # Compute action based on strategy
            if strategy_name == "static_maintain" or strategy_name == "conservative" or strategy_name == "aggressive":
                action_idx = policy_fn(client)
            else:
                # First compute predicted_pd
                from credit_limit_rl.env import _risk_feature_frame
                risk_frame = _risk_feature_frame(client, client["current_limit"], client["monthly_spend"])
                predicted_pd = float(risk_model.predict_proba(risk_frame)[0, 1])
                temp_client = client.copy()
                temp_client["predicted_pd"] = predicted_pd
                action_idx = policy_fn(temp_client)

            outcome = simulate_credit_decision(client, float(ACTION_ADJUSTMENTS[action_idx]), risk_model, config, rng)
            total_reward += outcome["reward"]
            default_count += outcome["defaulted"]
            total_pd += outcome["predicted_pd"]
            total_limit += outcome["new_limit"]

        results[strategy_name] = {
            "avg_reward": total_reward / len(portfolio),
            "portfolio_reward": total_reward,
            "default_rate": 100.0 * default_count / len(portfolio),
            "avg_pd": total_pd / len(portfolio),
            "avg_limit": total_limit / len(portfolio),
        }

    return results


def plot_diagnostics(portfolio_analysis: pd.DataFrame, output_dir: Path) -> None:
    """Create diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Action distribution histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    action_counts = portfolio_analysis["action_label"].value_counts().reindex(ACTION_LABELS, fill_value=0)
    action_counts.plot(kind="bar", ax=ax, color="#1f77b4")
    ax.set_title("PPO Action Distribution (Diagnosis: Collapse to +20%?)")
    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "action_distribution.png", dpi=180)
    plt.close()

    # 2. Logit variance across portfolio
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        portfolio_analysis["predicted_pd"],
        portfolio_analysis["logit_variance"],
        alpha=0.5,
        s=20,
    )
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Logit Variance (Action Preference Spread)")
    ax.set_title("Logit Variance vs Risk (Low variance = Collapse)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "logit_variance_vs_risk.png", dpi=180)
    plt.close()

    # 3. Action logits by risk segment
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    segments = categorize_by_risk(portfolio_analysis)
    segment_names = sorted(segments.keys())

    for ax, seg_name in zip(axes.flatten(), segment_names):
        seg_data = segments[seg_name]
        logit_cols = [f"logit_{label}" for label in ACTION_LABELS]
        logit_means = seg_data[logit_cols].mean()
        logit_means.index = ACTION_LABELS

        logit_means.plot(kind="bar", ax=ax, color="#1f77b4")
        ax.set_title(f"Mean Action Logits - {seg_name} Risk (n={len(seg_data)})")
        ax.set_ylabel("Logit Value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "logits_by_risk_segment.png", dpi=180)
    plt.close()

    # 4. Reward components by action taken
    fig, ax = plt.subplots(figsize=(10, 6))
    reward_by_action = portfolio_analysis.groupby("action_label")[["monthly_interest", "reward", "expected_loss", "rwa_cost"]].mean()
    reward_by_action.reindex(ACTION_LABELS).plot(kind="bar", ax=ax)
    ax.set_title("Average Reward Components by Action Taken")
    ax.set_ylabel("Amount (EUR)")
    ax.set_xlabel("Action")
    ax.legend(loc="best", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "reward_components_by_action.png", dpi=180)
    plt.close()

    # 5. Reward distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(portfolio_analysis["reward"], bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(portfolio_analysis["reward"].mean(), color="red", linestyle="--", label=f"Mean: {portfolio_analysis['reward'].mean():.2f}")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution (Check for saturation or collapse)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_distribution.png", dpi=180)
    plt.close()

    print(f"Diagnostic plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze PPO action collapse and reward signals.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Artifacts directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnostics"), help="Output directory for plots")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of clients to sample for analysis")
    args = parser.parse_args()

    # Load model and data
    model, portfolio, risk_model, config = load_model_and_assets(args.artifacts_dir)

    print("\n" + "=" * 80)
    print("ACTION DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Analyze action distribution
    portfolio_analysis = analyze_action_distribution(model, portfolio, risk_model, config, sample_size=args.sample_size)

    # Summary statistics
    action_dist = portfolio_analysis["action_label"].value_counts(normalize=True) * 100
    print("\nAction Distribution (%):")
    for label in ACTION_LABELS:
        pct = action_dist.get(label, 0.0)
        print(f"  {label:>5}: {pct:6.2f}%")

    print("\nLogit Variance Statistics:")
    print(f"  Mean logit variance: {portfolio_analysis['logit_variance'].mean():.6f}")
    print(f"  Median logit variance: {portfolio_analysis['logit_variance'].median():.6f}")
    print(f"  Std of logit variance: {portfolio_analysis['logit_variance'].std():.6f}")
    print(f"  Min logit variance: {portfolio_analysis['logit_variance'].min():.6f}")
    print(f"  Max logit variance: {portfolio_analysis['logit_variance'].max():.6f}")

    print("\nLogit Max-Min Gap (Action Preference Spread):")
    print(f"  Mean: {portfolio_analysis['logit_max_min_gap'].mean():.4f}")
    print(f"  Median: {portfolio_analysis['logit_max_min_gap'].median():.4f}")

    # Risk segment analysis
    print("\n" + "=" * 80)
    print("ACTION DISTRIBUTION BY RISK SEGMENT")
    print("=" * 80)

    segments = categorize_by_risk(portfolio_analysis)
    for seg_name in sorted(segments.keys()):
        seg_data = segments[seg_name]
        print(f"\n{seg_name} Risk (n={len(seg_data)}, PD range: {seg_data['predicted_pd'].min():.4f}-{seg_data['predicted_pd'].max():.4f}):")
        action_dist_seg = seg_data["action_label"].value_counts(normalize=True) * 100
        for label in ACTION_LABELS:
            pct = action_dist_seg.get(label, 0.0)
            print(f"  {label:>5}: {pct:6.2f}%")

    # Reward signal analysis
    print("\n" + "=" * 80)
    print("REWARD SIGNAL ANALYSIS")
    print("=" * 80)

    print("\nOverall Reward Statistics:")
    print(f"  Mean reward: {portfolio_analysis['reward'].mean():.2f}")
    print(f"  Median reward: {portfolio_analysis['reward'].median():.2f}")
    print(f"  Std reward: {portfolio_analysis['reward'].std():.2f}")
    print(f"  Min reward: {portfolio_analysis['reward'].min():.2f}")
    print(f"  Max reward: {portfolio_analysis['reward'].max():.2f}")

    print("\nAverage Reward Components:")
    print(f"  Interest income: {portfolio_analysis['monthly_interest'].mean():.2f}")
    print(f"  Expected loss: {portfolio_analysis['expected_loss'].mean():.2f}")
    print(f"  RWA cost: {portfolio_analysis['rwa_cost'].mean():.2f}")
    print(f"  Constraint penalty: {portfolio_analysis['constraint_penalty'].mean():.2f}")

    print("\nReward by Action Taken:")
    for label in ACTION_LABELS:
        mask = portfolio_analysis["action_label"] == label
        if mask.any():
            mean_reward = portfolio_analysis[mask]["reward"].mean()
            count = mask.sum()
            print(f"  {label:>5}: {mean_reward:8.2f} (n={count})")

    # Baseline comparison
    print("\n" + "=" * 80)
    print("BASELINE STRATEGY COMPARISON")
    print("=" * 80)

    baseline_results = evaluate_baseline_strategies(portfolio, risk_model, config)
    comparison_df = pd.DataFrame(baseline_results).T
    print("\n" + comparison_df.to_string())

    # Save diagnostic plots
    plot_diagnostics(portfolio_analysis, args.output_dir)

    # Save analysis results as CSV
    portfolio_analysis.to_csv(args.output_dir / "portfolio_analysis.csv", index=False)
    comparison_df.to_csv(args.output_dir / "baseline_comparison.csv")

    print("\n" + "=" * 80)
    print("Analysis complete. Outputs saved to:", args.output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
