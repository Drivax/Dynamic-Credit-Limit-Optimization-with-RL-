"""Research-only diagnostics, explicitly separated from policy features."""

import numpy as np
import pandas as pd


def hazard_summary(history: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Closing hazard among customers alive at the start of each observed month."""
    return history.groupby(by, observed=True).agg(
        at_risk_months=("p_default_true", "size"),
        hazard_mean=("p_default_true", "mean"),
        hazard_std=("p_default_true", "std"),
        hazard_p01=("p_default_true", lambda s: s.quantile(0.01)),
        hazard_p50=("p_default_true", "median"),
        hazard_p99=("p_default_true", lambda s: s.quantile(0.99)),
        realized_default_rate=("realized_default", "mean"),
    ).reset_index()


def population_summary(history: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return history.groupby(by, observed=True).agg(
        at_risk_months=("customer_id", "size"), default_rate=("realized_default", "mean"),
        delinquency_rate=("delinquency_status", "mean"), utilization=("utilization", "mean"),
        balance=("balance", "mean"), spending=("spending", "mean"),
        payment_ratio=("payment_ratio", "mean"), payment=("payment", "mean"),
        income_mean=("income", "mean"), income_p10=("income", lambda s: s.quantile(0.1)),
        income_p50=("income", "median"), income_p90=("income", lambda s: s.quantile(0.9)),
        limit_mean=("credit_limit", "mean"), limit_p10=("credit_limit", lambda s: s.quantile(0.1)),
        limit_p90=("credit_limit", lambda s: s.quantile(0.9)),
    ).reset_index()


def paired_difference(reference: np.ndarray, comparison: np.ndarray) -> dict[str, float]:
    """Descriptive paired mean difference and normal-approximation standard error."""
    delta = np.asarray(comparison, dtype=float)-np.asarray(reference, dtype=float)
    se = float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta) > 1 else 0.0
    return {"difference": float(delta.mean()), "standard_error": se,
            "ci95_low": float(delta.mean()-1.96*se), "ci95_high": float(delta.mean()+1.96*se)}
