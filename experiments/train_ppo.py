"""Small PPO integration run with unchanged economics and held-out trajectories."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.evaluation.metrics import summarize_trajectories
from credit_rl.policies.baselines import RiskThresholdPolicy, SB3Policy, StaticPolicy
from credit_rl.risk.longitudinal import LongitudinalPDModel
from credit_rl.simulation.simulator import simulate_customer
from credit_rl.simulation.synthetic_snapshot import generate_synthetic_portfolio
from credit_rl.utils.seeding import seed_everything
from .common import load_run_config, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--experiments", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--pd-model", type=Path, default=Path("outputs/models/pd/logistic_calibrated.joblib"))
    args = parser.parse_args()
    config, run = SimulationConfig.from_yaml(args.config), load_run_config(args.experiments)
    if args.timesteps is not None:
        if args.timesteps < 1:
            parser.error("--timesteps must be positive")
        run["ppo_timesteps"] = args.timesteps
    seed = run["seed"]
    seed_everything(seed, torch=True)
    train = generate_synthetic_portfolio(run["risk_training_clients"], seed=seed)
    test = generate_synthetic_portfolio(run["ppo_eval_customers"], seed=seed + 1)
    if not args.pd_model.exists():
        parser.error("Train the longitudinal risk model with python -m experiments.train_pd first")
    risk_model = LongitudinalPDModel.load(args.pd_model)
    env = CreditLimitEnv(train, risk_model, config)
    check_env(env, warn=True)
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **run["ppo"])
    model.learn(total_timesteps=run["ppo_timesteps"])
    models = args.output / "models" / "ppo_smoke_v2"
    results = args.output / "results" / "ppo_smoke_v2"
    models.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    model.save(models / "ppo_longitudinal")
    risk_model.save(models / "longitudinal_pd.joblib")
    # Reload verifies the artifact against the actual observation/action spaces.
    reloaded = PPO.load(models / "ppo_longitudinal", env=env, device="cpu")
    frames = []
    policies = {"static": StaticPolicy(config.environment), "ppo": SB3Policy(reloaded),
                "risk_threshold": RiskThresholdPolicy(config.environment)}
    for name, policy in policies.items():
        for customer in range(len(test)):
            frame = simulate_customer(CreditLimitEnv(test, risk_model, config), policy,
                seed=seed + 10000 + customer, options={"customer_index": customer}).to_dataframe()
            frame["policy"], frame["episode_id"] = name, customer
            frames.append(frame)
    history = pd.concat(frames, ignore_index=True)
    history.to_csv(results / "trajectories.csv", index=False)
    summary = summarize_trajectories(history)
    summary.to_csv(results / "summary.csv", index=False)
    write_manifest(results / "manifest.json", config, run,
        actual_timesteps=model.num_timesteps, pd_metadata=risk_model.metadata,
        purpose="Integration smoke test; no tuning, no claim of comparative policy performance")
    env.close()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
