"""
Retrain with optimized reward weights to encourage risk-aware policies.

Key insight: Current weights (lambda_default=5.0, lambda_rwa=1.5) are too weak.
High-risk clients should get actions like -20% or -10%, not +20%.
Low-risk clients should discriminate between +10% and +20%.

Solution: Increase penalties for default risk and capital costs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from credit_limit_rl.config import PortfolioConfig
from credit_limit_rl.data import generate_synthetic_portfolio
from credit_limit_rl.env import CreditLimitEnv, ACTION_LABELS, ACTION_ADJUSTMENTS, simulate_credit_decision
from credit_limit_rl.evaluation import evaluate_policy
from credit_limit_rl.risk import train_risk_model


def create_optimized_config(
    lambda_default: float,
    lambda_rwa: float,
) -> PortfolioConfig:
    """Create config with optimized reward weights."""
    return PortfolioConfig(
        min_limit=500.0,
        max_limit=15000.0,
        annual_percentage_rate=0.18,
        fee_rate=0.012,
        loss_given_default=0.55,
        rwa_factor=0.08,
        lambda_default=lambda_default,
        lambda_rwa=lambda_rwa,
        max_pd_threshold=0.12,
        portfolio_pd_threshold=0.10,
    )


def align_batch_size(n_steps: int, requested_batch_size: int) -> int:
    """Return the largest batch size <= request that evenly divides the rollout size."""
    n_steps = int(n_steps)
    requested_batch_size = max(1, min(int(requested_batch_size), n_steps))
    if n_steps % requested_batch_size == 0:
        return requested_batch_size
    for candidate in range(requested_batch_size - 1, 0, -1):
        if n_steps % candidate == 0:
            return candidate
    return 1


def build_ppo_model(
    train_env: DummyVecEnv,
    seed: int,
    ppo_params: dict,
) -> PPO:
    """Build PPO model."""
    n_steps = int(ppo_params["n_steps"])
    batch_size = align_batch_size(n_steps, int(ppo_params["batch_size"]))

    return PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        gamma=float(ppo_params["gamma"]),
        learning_rate=float(ppo_params["learning_rate"]),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(ppo_params["n_epochs"]),
        ent_coef=float(ppo_params["ent_coef"]),
        clip_range=float(ppo_params["clip_range"]),
        policy_kwargs={"net_arch": list(ppo_params["net_arch"])},
        seed=seed,
    )


def load_best_params(path: Path) -> dict:
    """Load HPO parameters."""
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Retrain with optimized reward weights.")
    parser.add_argument("--clients", type=int, default=10000, help="Number of clients.")
    parser.add_argument("--train-timesteps", type=int, default=100000, help="Training timesteps.")
    parser.add_argument("--episode-length", type=int, default=256, help="Episode length.")
    parser.add_argument("--lambda-default", type=float, default=12.0, help="Penalty for expected loss.")
    parser.add_argument("--lambda-rwa", type=float, default=4.0, help="Penalty for capital cost.")
    parser.add_argument("--ent-coef", type=float, default=0.10, help="Entropy coefficient.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # Load existing assets
    artifacts_dir = Path("artifacts")
    risk_model_path = artifacts_dir / "risk_model.joblib"
    
    print("Loading portfolio and risk model...")
    portfolio = generate_synthetic_portfolio(n_clients=args.clients, seed=args.seed)
    split_idx = int(len(portfolio) * 0.8)
    train_portfolio = portfolio.iloc[:split_idx].reset_index(drop=True)
    test_portfolio = portfolio.iloc[split_idx:].reset_index(drop=True)
    risk_model = joblib.load(str(risk_model_path))

    # Create optimized config
    config = create_optimized_config(args.lambda_default, args.lambda_rwa)
    
    print(f"\nOptimized reward configuration:")
    print(f"  lambda_default: {config.lambda_default} (was 5.0)")
    print(f"  lambda_rwa: {config.lambda_rwa} (was 1.5)")

    # Load PPO params with high entropy
    hpo_params_path = artifacts_dir / "hpo" / "best_params.json"
    if hpo_params_path.exists():
        params = load_best_params(hpo_params_path)
        ppo_params = {
            "gamma": float(params.get("gamma", 0.95)),
            "learning_rate": float(params.get("learning_rate", 0.000658628931758311)),
            "n_steps": int(round(float(params.get("n_steps", 1536)))),
            "batch_size": int(round(float(params.get("batch_size", 192)))),
            "n_epochs": int(round(float(params.get("n_epochs", 11)))),
            "ent_coef": float(params.get("ent_coef", 0.02)),
            "clip_range": float(params.get("clip_range", 0.2)),
            "net_arch": [
                int(round(float(params.get("net_arch_size", 192)))),
                int(round(float(params.get("net_arch_size", 192)))),
            ],
        }
    else:
        ppo_params = {
            "gamma": 0.95,
            "learning_rate": 0.000658628931758311,
            "n_steps": 1536,
            "batch_size": 192,
            "n_epochs": 11,
            "ent_coef": 0.02,
            "clip_range": 0.2,
            "net_arch": [192, 192],
        }
    
    # Override entropy
    ppo_params["ent_coef"] = args.ent_coef

    print(f"\nTraining configuration:")
    print(f"  Clients: {args.clients}")
    print(f"  Timesteps: {args.train_timesteps}")
    print(f"  Episode length: {args.episode_length}")
    print(f"  Entropy coef: {ppo_params['ent_coef']}")

    # Analyze reward distribution with new config BEFORE training
    print(f"\nAnalyzing reward function with new weights...")
    rng = np.random.default_rng(123)
    sample = test_portfolio.sample(n=min(100, len(test_portfolio)), random_state=42)
    
    for group_label, group_df in [
        ("HIGH-RISK (internal_score < median)", sample[sample["internal_score"] < sample["internal_score"].median()]),
        ("LOW-RISK (internal_score > median)", sample[sample["internal_score"] > sample["internal_score"].median()]),
    ]:
        print(f"\n  {group_label}:")
        action_rewards = {label: [] for label in ACTION_LABELS}
        
        for _, client in group_df.iterrows():
            for action_idx, label in enumerate(ACTION_LABELS):
                outcome = simulate_credit_decision(client, float(ACTION_ADJUSTMENTS[action_idx]), risk_model, config, rng)
                action_rewards[label].append(outcome["reward"])
        
        for label in ACTION_LABELS:
            mean_reward = np.mean(action_rewards[label])
            print(f"    {label}: {mean_reward:8.1f}", end="")
        
        best_action = max(action_rewards, key=lambda x: np.mean(action_rewards[x]))
        print(f" --> Optimal: {best_action}")

    # Train model
    print(f"\nTraining PPO with optimized reward function...")
    train_env = DummyVecEnv([
        lambda: CreditLimitEnv(
            portfolio=train_portfolio,
            risk_model=risk_model,
            config=config,
            episode_length=args.episode_length,
            seed=args.seed,
        )
    ])

    model = build_ppo_model(train_env, args.seed, ppo_params)
    model.learn(total_timesteps=args.train_timesteps, progress_bar=False)

    # Save model
    model_name = f"ppo_optimized_weights"
    model.save(str(artifacts_dir / model_name))
    print(f"Model saved to {artifacts_dir}/{model_name}.zip")

    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    static_summary, static_details = evaluate_policy("static_maintain", test_portfolio, risk_model, config, model=None)
    ppo_summary, ppo_details = evaluate_policy("ppo_optimized", test_portfolio, risk_model, config, model=model)
    
    # Action distribution
    print("\nAction Distribution (Optimized Model):")
    action_dist = ppo_details["action_label"].value_counts(normalize=True) * 100
    for label in ACTION_LABELS:
        pct = action_dist.get(label, 0.0)
        print(f"  {label:>5}: {pct:6.2f}%")

    # Performance
    print(f"\n{'Policy':<20} {'Avg Reward':>15} {'Default Rate':>15} {'Avg Limit':>15}")
    print("-"*65)
    print(f"{'Static maintain':<20} {static_summary['avg_reward']:>15.2f} {static_summary['default_rate']*100:>14.2f}% {static_summary['avg_limit']:>15.2f}")
    print(f"{'PPO optimized':<20} {ppo_summary['avg_reward']:>15.2f} {ppo_summary['default_rate']*100:>14.2f}% {ppo_summary['avg_limit']:>15.2f}")
    
    reward_improvement = ppo_summary["avg_reward"] - static_summary["avg_reward"]
    print(f"\nReward improvement: {reward_improvement:+.2f} ({reward_improvement/abs(static_summary['avg_reward'])*100:+.2f}%)")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    ppo_details.to_csv(results_dir / "policy_optimized_weights.csv", index=False)
    
    diagnostics = {
        "config": {
            "lambda_default": config.lambda_default,
            "lambda_rwa": config.lambda_rwa,
            "entropy_coef": ppo_params["ent_coef"],
        },
        "policy_comparison": {
            "ppo_avg_reward": float(ppo_summary["avg_reward"]),
            "ppo_default_rate": float(ppo_summary["default_rate"]),
            "ppo_avg_limit": float(ppo_summary["avg_limit"]),
        },
        "action_distribution": ppo_details["action_label"].value_counts().reindex(ACTION_LABELS, fill_value=0).to_dict(),
    }
    
    with open(results_dir / "optimized_weights_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)


if __name__ == "__main__":
    main()
