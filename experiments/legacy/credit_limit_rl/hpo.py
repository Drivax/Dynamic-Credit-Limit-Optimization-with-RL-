from __future__ import annotations

import json
import logging
from pathlib import Path

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import PortfolioConfig
from .env import CreditLimitEnv
from .evaluation import evaluate_policy

logger = logging.getLogger(__name__)


def create_trial_config(trial: optuna.Trial) -> dict:
    """
    Define the hyperparameter search space for PPO and reward weights.
    
    Returns:
        Dictionary with sampled hyperparameters
    """
    config = {
        # Reward weights
        "lambda_default": trial.suggest_float("lambda_default", 3.0, 8.0, step=0.5),
        "lambda_rwa": trial.suggest_float("lambda_rwa", 1.0, 2.5, step=0.25),
        # PPO hyperparameters
        "gamma": trial.suggest_float("gamma", 0.95, 0.99, step=0.01),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.05, log=True),
        "n_steps": trial.suggest_int("n_steps", 512, 2048, step=256),
        "batch_size": trial.suggest_int("batch_size", 64, 512, step=64),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3, step=0.05),
        "n_epochs": trial.suggest_int("n_epochs", 5, 15, step=1),
        # Network architecture (hidden layer size)
        "net_arch_size": trial.suggest_int("net_arch_size", 128, 512, step=64),
    }
    return config


def objective(
    trial: optuna.Trial,
    train_portfolio: pd.DataFrame,
    test_portfolio: pd.DataFrame,
    risk_model,
    train_timesteps: int = 100_000,
    seed: int = 42,
) -> float:
    """
    Objective function for Optuna to maximize average test-set reward.
    
    Args:
        trial: Optuna trial object
        train_portfolio: Training dataset
        test_portfolio: Test dataset
        risk_model: Pre-trained risk model
        train_timesteps: Number of PPO training steps
        seed: Random seed
    
    Returns:
        Average reward on test set (higher is better)
    """
    params = create_trial_config(trial)
    
    # Create portfolio config with tuned reward weights
    config = PortfolioConfig(
        lambda_default=params["lambda_default"],
        lambda_rwa=params["lambda_rwa"],
    )
    
    try:
        # Create training environment
        train_env = DummyVecEnv([
            lambda: CreditLimitEnv(
                portfolio=train_portfolio,
                risk_model=risk_model,
                config=config,
                episode_length=512,
                seed=seed,
            )
        ])
        
        # Create and train PPO model
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            gamma=params["gamma"],
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            ent_coef=params["ent_coef"],
            clip_range=params["clip_range"],
            policy_kwargs={"net_arch": [params["net_arch_size"], params["net_arch_size"]]},
            seed=seed,
        )
        
        # Train with early stopping via pruning callback
        model.learn(total_timesteps=train_timesteps, progress_bar=False)
        
        # Evaluate on test set
        summary, _ = evaluate_policy(
            "ppo_tuned",
            test_portfolio,
            risk_model,
            config,
            model=model,
        )
        
        test_reward = float(summary["avg_reward"])
        trial.set_user_attr("test_default_rate", summary["default_rate"])
        trial.set_user_attr("test_avg_limit", summary["avg_limit"])
        
        logger.info(f"Trial {trial.number}: reward={test_reward:.2f}, default_rate={summary['default_rate']:.2%}")
        
        return test_reward
        
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialError(str(e))


def run_hpo_study(
    train_portfolio: pd.DataFrame,
    test_portfolio: pd.DataFrame,
    risk_model,
    n_trials: int = 50,
    train_timesteps: int = 100_000,
    seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[optuna.Study, dict]:
    """
    Run hyperparameter optimization study.
    
    Args:
        train_portfolio: Training data
        test_portfolio: Test data
        risk_model: Pre-trained risk model
        n_trials: Number of trials to run
        train_timesteps: PPO training budget per trial
        seed: Random seed
        output_dir: Directory to save study artifacts
    
    Returns:
        (study object, best parameters dictionary)
    """
    output_dir = output_dir or Path("hpo_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
    )
    
    logger.info(f"Starting HPO study with {n_trials} trials...")
    study.optimize(
        lambda trial: objective(
            trial,
            train_portfolio,
            test_portfolio,
            risk_model,
            train_timesteps=train_timesteps,
            seed=seed,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # Save results
    best_trial = study.best_trial
    best_params = {k: v for k, v in best_trial.params.items()}
    
    results_df = study.trials_dataframe()
    results_df.to_csv(output_dir / "hpo_trials.csv", index=False)
    
    with open(output_dir / "best_params.json", "w") as f:
        # Convert numpy types to native Python for JSON serialization
        json_params = {k: float(v) if isinstance(v, (int, float)) else int(v) 
                       for k, v in best_params.items()}
        json.dump(json_params, f, indent=2)
    
    logger.info(f"Best trial: {best_trial.number}")
    logger.info(f"Best reward: {best_trial.value:.2f}")
    logger.info(f"Best params: {best_params}")
    
    return study, best_params


def load_best_params(params_path: Path) -> dict:
    """Load best parameters from HPO results."""
    with open(params_path) as f:
        return json.load(f)
