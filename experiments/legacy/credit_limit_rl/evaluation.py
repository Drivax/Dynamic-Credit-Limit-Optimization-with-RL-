from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .config import PortfolioConfig
from .env import ACTION_ADJUSTMENTS, ACTION_LABELS, CreditLimitEnv, build_observation, simulate_credit_decision


def evaluate_policy(
    policy_name: str,
    portfolio: pd.DataFrame,
    risk_model,
    config: PortfolioConfig,
    model: PPO | None = None,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Evaluate a policy on a portfolio and return summary metrics + decision details."""
    rng = np.random.default_rng(123)
    rows: list[dict[str, float | int | str]] = []

    for _, client in portfolio.iterrows():
        if model is None:
            action_index = 2  # static maintain
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


def kfold_split(
    portfolio: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate k-fold train/test splits with stratification by risk level.
    
    Args:
        portfolio: Full dataset with 'internal_score' column
        n_splits: Number of folds
        seed: Random seed
    
    Returns:
        List of (train, test) DataFrame tuples
    """
    rng = np.random.default_rng(seed)
    portfolio = portfolio.reset_index(drop=True)
    
    # Stratify by risk quartile (use internal_score as proxy)
    risk_quartiles = pd.qcut(portfolio["internal_score"], q=4, labels=False, duplicates="drop")
    folds = []
    
    indices = np.arange(len(portfolio))
    rng.shuffle(indices)
    fold_size = len(portfolio) // n_splits
    
    for fold_idx in range(n_splits):
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < n_splits - 1 else len(portfolio)
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        train = portfolio.iloc[train_indices].reset_index(drop=True)
        test = portfolio.iloc[test_indices].reset_index(drop=True)
        folds.append((train, test))
    
    return folds


def risk_regime_split(
    portfolio: pd.DataFrame,
    score_column: str = "internal_score",
) -> dict[str, pd.DataFrame]:
    """
    Split portfolio into risk regimes based on internal score percentiles.
    
    Args:
        portfolio: Dataset with risk scoring column
        score_column: Name of the risk score column
    
    Returns:
        Dictionary with keys "high_risk", "medium_risk", "low_risk"
    """
    low_threshold = portfolio[score_column].quantile(0.33)
    high_threshold = portfolio[score_column].quantile(0.67)
    
    regimes = {
        "low_risk": portfolio[portfolio[score_column] < low_threshold].reset_index(drop=True),
        "medium_risk": portfolio[
            (portfolio[score_column] >= low_threshold) & (portfolio[score_column] < high_threshold)
        ].reset_index(drop=True),
        "high_risk": portfolio[portfolio[score_column] >= high_threshold].reset_index(drop=True),
    }
    return regimes


def timeseries_split(
    portfolio: pd.DataFrame,
    n_splits: int = 5,
    temporal_column: str | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate time-series aware train/test splits (walk-forward).
    If no temporal column is provided, uses row order as proxy.
    
    Args:
        portfolio: Dataset (ideally with temporal order already set)
        n_splits: Number of walk-forward folds
        temporal_column: Column name for sorting (optional)
    
    Returns:
        List of (train, test) DataFrame tuples in temporal order
    """
    portfolio = portfolio.reset_index(drop=True)
    
    if temporal_column and temporal_column in portfolio.columns:
        portfolio = portfolio.sort_values(temporal_column).reset_index(drop=True)
    
    folds = []
    total_size = len(portfolio)
    test_size = total_size // (n_splits + 1)
    
    for fold_idx in range(n_splits):
        train_end = (fold_idx + 1) * test_size
        test_end = train_end + test_size
        
        train = portfolio.iloc[:train_end].reset_index(drop=True)
        test = portfolio.iloc[train_end:test_end].reset_index(drop=True)
        folds.append((train, test))
    
    return folds


def evaluate_across_regimes(
    portfolio: pd.DataFrame,
    risk_model,
    config: PortfolioConfig,
    model: PPO | None = None,
) -> pd.DataFrame:
    """
    Evaluate policy across different risk regimes and return regime-level breakdown.
    
    Returns:
        DataFrame with rows for each regime showing avg_reward, default_rate, etc.
    """
    regimes = risk_regime_split(portfolio)
    results = []
    
    for regime_name, regime_portfolio in regimes.items():
        if len(regime_portfolio) == 0:
            continue
            
        summary, _ = evaluate_policy(
            f"ppo_dynamic_{regime_name}",
            regime_portfolio,
            risk_model,
            config,
            model=model,
        )
        summary["regime"] = regime_name
        summary["n_clients"] = len(regime_portfolio)
        results.append(summary)
    
    return pd.DataFrame(results)
