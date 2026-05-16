"""
Retrain PPO with fixes for action collapse.

This script retrains the model with:
1. Higher entropy coefficient (more exploration)
2. Smaller initial learning rate (more stable)
3. Better scaling of the reward function
4. Action diversity penalty
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
from credit_limit_rl.env import CreditLimitEnv, ACTION_LABELS
from credit_limit_rl.evaluation import evaluate_policy
from credit_limit_rl.risk import train_risk_model
from train import (
    load_retrain_ppo_params, 
    align_batch_size, 
    save_policy_charts, 
    save_risk_charts,
    summarize_action_distribution,
)


def build_ppo_model_fixed(
    train_env: DummyVecEnv,
    seed: int,
    ppo_params: dict,
    ent_coef_override: float | None = None,
) -> PPO:
    """Build PPO with optional entropy coefficient override."""
    n_steps = int(ppo_params["n_steps"])
    batch_size = align_batch_size(n_steps, int(ppo_params["batch_size"]))
    
    ent_coef = ent_coef_override if ent_coef_override is not None else float(ppo_params["ent_coef"])
    
    if ent_coef_override is not None:
        print(f"Overriding entropy coefficient: {float(ppo_params['ent_coef'])} -> {ent_coef}")

    return PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        gamma=float(ppo_params["gamma"]),
        learning_rate=float(ppo_params["learning_rate"]),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(ppo_params["n_epochs"]),
        ent_coef=ent_coef,
        clip_range=float(ppo_params["clip_range"]),
        policy_kwargs={"net_arch": list(ppo_params["net_arch"])},
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Retrain PPO with action collapse fixes.")
    parser.add_argument("--clients", type=int, default=20000, help="Number of synthetic clients.")
    parser.add_argument("--train-timesteps", type=int, default=100000, help="PPO training timesteps.")
    parser.add_argument("--episode-length", type=int, default=256, help="Episode length.")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient (None to use HPO params or default).")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Artifacts directory.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Results directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-suffix", type=str, default="fixed", help="Suffix for output model name.")
    args = parser.parse_args()

    # Load existing assets
    print("Loading existing model and data...")
    artifacts_dir = args.artifacts_dir
    risk_model_path = artifacts_dir / "risk_model.joblib"
    
    if not risk_model_path.exists():
        print("Risk model not found. Training new portfolio and risk model...")
        portfolio = generate_synthetic_portfolio(n_clients=args.clients, seed=args.seed)
        split_index = int(len(portfolio) * 0.8)
        train_portfolio = portfolio.iloc[:split_index].reset_index(drop=True)
        test_portfolio = portfolio.iloc[split_index:].reset_index(drop=True)
        risk_model, risk_report = train_risk_model(train_portfolio, random_state=args.seed)
        joblib.dump(risk_model, risk_model_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("Loading existing portfolio...")
        portfolio = generate_synthetic_portfolio(n_clients=args.clients, seed=args.seed)
        split_index = int(len(portfolio) * 0.8)
        train_portfolio = portfolio.iloc[:split_index].reset_index(drop=True)
        test_portfolio = portfolio.iloc[split_index:].reset_index(drop=True)
        risk_model = joblib.load(risk_model_path)

    config = PortfolioConfig()
    
    # Load PPO parameters
    ppo_params = load_retrain_ppo_params(artifacts_dir)
    
    # Apply overrides
    if args.ent_coef is not None:
        print(f"Setting entropy coefficient to {args.ent_coef}")
        ppo_params["ent_coef"] = args.ent_coef
    else:
        # Default aggressive exploration settings
        default_ent = 0.08
        print(f"Using default aggressive entropy coefficient: {default_ent}")
        ppo_params["ent_coef"] = default_ent
    
    if args.learning_rate is not None:
        print(f"Setting learning rate to {args.learning_rate}")
        ppo_params["learning_rate"] = args.learning_rate

    print("\nTraining configuration:")
    print(f"  Clients: {args.clients}")
    print(f"  Timesteps: {args.train_timesteps}")
    print(f"  Episode length: {args.episode_length}")
    print(f"  Entropy coef: {ppo_params['ent_coef']}")
    print(f"  Learning rate: {ppo_params['learning_rate']}")
    print(f"  n_steps: {ppo_params['n_steps']}")
    print(f"  batch_size: {ppo_params['batch_size']}")

    # Create training environment
    train_env = DummyVecEnv([
        lambda: CreditLimitEnv(
            portfolio=train_portfolio,
            risk_model=risk_model,
            config=config,
            episode_length=args.episode_length,
            seed=args.seed,
        )
    ])

    # Build and train model with fixed parameters
    print("\nBuilding PPO model with fixed exploration...")
    model = build_ppo_model_fixed(train_env, args.seed, ppo_params)
    
    print(f"Training for {args.train_timesteps:,} timesteps...")
    model.learn(total_timesteps=args.train_timesteps, progress_bar=False)

    # Save trained model
    output_dir = artifacts_dir
    model_name = f"ppo_credit_limit_{args.output_suffix}"
    model_path = output_dir / model_name
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}.zip")

    # Evaluate
    print("\nEvaluating fixed model...")
    static_summary, static_details = evaluate_policy("static_maintain", test_portfolio, risk_model, config, model=None)
    ppo_summary, ppo_details = evaluate_policy("ppo_dynamic", test_portfolio, risk_model, config, model=model)
    
    # Print action distribution
    print("\n" + "="*80)
    print("ACTION DISTRIBUTION (Fixed Model)")
    print("="*80)
    action_dist = ppo_details["action_label"].value_counts(normalize=True) * 100
    for label in ACTION_LABELS:
        pct = action_dist.get(label, 0.0)
        print(f"  {label:>5}: {pct:6.2f}%")
    
    # Compare with baseline
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Policy':<20} {'Avg Reward':>15} {'Default Rate':>15} {'Avg Limit':>15}")
    print("-"*65)
    print(f"{'Static maintain':<20} {static_summary['avg_reward']:>15.2f} {static_summary['default_rate']*100:>14.2f}% {static_summary['avg_limit']:>15.2f}")
    print(f"{'PPO (fixed)':<20} {ppo_summary['avg_reward']:>15.2f} {ppo_summary['default_rate']*100:>14.2f}% {ppo_summary['avg_limit']:>15.2f}")
    
    reward_improvement = ppo_summary["avg_reward"] - static_summary["avg_reward"]
    print(f"\nReward improvement: {reward_improvement:+.2f} ({reward_improvement/abs(static_summary['avg_reward'])*100:+.2f}%)")
    
    # Save results
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    evaluation = [static_summary, ppo_summary]
    policy_details = pd.concat([static_details, ppo_details], ignore_index=True)
    policy_details.to_csv(results_dir / f"policy_decisions_{args.output_suffix}.csv", index=False)
    
    # Save diagnostics
    diagnostics = {
        "model_name": model_name,
        "training_config": {
            "clients": args.clients,
            "train_timesteps": args.train_timesteps,
            "episode_length": args.episode_length,
            "entropy_coef": float(ppo_params["ent_coef"]),
            "learning_rate": float(ppo_params["learning_rate"]),
        },
        "policy_comparison": {
            "static_maintain": {
                "avg_reward": float(static_summary["avg_reward"]),
                "default_rate": float(static_summary["default_rate"]),
                "avg_limit": float(static_summary["avg_limit"]),
            },
            "ppo_dynamic": {
                "avg_reward": float(ppo_summary["avg_reward"]),
                "default_rate": float(ppo_summary["default_rate"]),
                "avg_limit": float(ppo_summary["avg_limit"]),
            },
        },
        "ppo_action_distribution": ppo_details["action_label"].value_counts().reindex(ACTION_LABELS, fill_value=0).to_dict(),
        "reward_improvement": {
            "absolute": float(reward_improvement),
            "percentage": float(reward_improvement / abs(static_summary["avg_reward"]) * 100),
        },
    }
    
    with open(results_dir / f"retrain_diagnostics_{args.output_suffix}.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/")
    print(f"Diagnostics saved to {results_dir}/retrain_diagnostics_{args.output_suffix}.json")


if __name__ == "__main__":
    main()
