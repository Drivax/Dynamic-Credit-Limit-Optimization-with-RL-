"""Single observable schema and strictly backward-looking monthly features."""
from collections import deque

import numpy as np
import pandas as pd

ALLOWED_AT_DECISION_TIME = (
    "credit_limit", "balance", "utilization", "income", "payment_ratio",
    "months_delinquent", "behavioral_score", "tenure_months", "monthly_spend",
    "late_payments_6m", "income_log_change", "macro_income_growth",
    "macro_spending_growth", "macro_credit_stress",
)
HISTORICAL_FEATURES = (
    "history_months", "debt_to_income", "utilization_lag1", "utilization_mean3",
    "utilization_max6", "payment_mean3", "payment_min6", "income_volatility6",
    "balance_change1", "income_change3", "limit_change6",
)
FEATURE_NAMES = ALLOWED_AT_DECISION_TIME + HISTORICAL_FEATURES
SIMULATOR_ONLY = frozenset({
    "p_default_true", "true_pd", "true_default_probability", "creditworthiness",
    "spending_propensity", "payment_propensity", "income_stability",
    "adverse_income_event", "spending_elasticity", "initial_score", "initial_income",
})
TARGET_ONLY = frozenset({"defaulted", "target", "default_next_month", "realized_default",
                         "label_end", "scheduled_end", "followup_end"})


def validate_feature_names(names):
    names = tuple(names)
    if not names or len(set(names)) != len(names) or set(names) - set(FEATURE_NAMES):
        raise ValueError("Feature allowlist violation: unknown, duplicate or forbidden feature")
    return names


def classify_column(name):
    if name in SIMULATOR_ONLY or name.startswith(("latent_", "shock_", "future_", "next_")):
        return "SIMULATOR_ONLY"
    if name in TARGET_ONLY:
        return "TARGET_ONLY"
    if name in FEATURE_NAMES:
        return "ALLOWED_AT_DECISION_TIME"
    return "EXCLUDED_METADATA_OR_UNREVIEWED"


def observable_row(state):
    """Explicit projection: never asdict(state), traits, outcomes or diagnostics."""
    return dict(
        credit_limit=state.credit_limit, balance=state.balance, utilization=state.utilization,
        income=state.income, payment_ratio=state.payment_ratio,
        months_delinquent=state.months_delinquent, behavioral_score=state.behavioral_score,
        tenure_months=state.tenure_months, monthly_spend=state.monthly_spend,
        late_payments_6m=sum(state.late_history), income_log_change=state.income_log_change,
        macro_income_growth=state.macro_state.income_growth,
        macro_spending_growth=state.macro_state.spending_growth,
        macro_credit_stress=state.macro_state.credit_stress,
    )


def history_vector(history):
    """At most seven observable rows, ending at decision t; no pandas online.

    Historical summaries exclude t: e.g. mean3 uses t-3,...,t-1. Current
    quantities are separate features. Missing pre-enrolment history stays NaN.
    """
    if not history:
        raise ValueError("Empty observation history")
    rows = list(history)[-7:]
    now, past = rows[-1], rows[:-1]
    def values(name, window):
        return [row[name] for row in past[-window:]]
    def summary(name, window, operation):
        x = values(name, window)
        return float(operation(x)) if x else np.nan
    def change(name, lag):
        if len(past) < lag:
            return np.nan
        return (now[name] - past[-lag][name]) / max(abs(past[-lag][name]), 1.0)
    derived = [len(past), now["balance"] / now["income"],
        past[-1]["utilization"] if past else np.nan,
        summary("utilization", 3, np.mean), summary("utilization", 6, np.max),
        summary("payment_ratio", 3, np.mean), summary("payment_ratio", 6, np.min),
        summary("income_log_change", 6, np.std) if len(past) >= 2 else np.nan,
        change("balance", 1), change("income", 3), change("credit_limit", 6)]
    return np.array([now[name] for name in ALLOWED_AT_DECISION_TIME] + derived, dtype=float)


def build_features(trajectories):
    """Group by identity before constructing prefix-only features; preserve index."""
    if trajectories.duplicated(["customer_id", "month"]).any():
        raise ValueError("Duplicate customer/month")
    result = pd.DataFrame(index=trajectories.index, columns=FEATURE_NAMES, dtype=float)
    for _, group in trajectories.groupby("customer_id", sort=False):
        group = group.sort_values("month")
        months = group.month.to_numpy()
        if len(months) > 1 and not np.all(np.diff(months) == 1):
            raise ValueError("Monthly histories must be contiguous")
        history = deque(maxlen=7)
        vectors = []
        for row in group[list(ALLOWED_AT_DECISION_TIME)].to_dict("records"):
            history.append(row)
            vectors.append(history_vector(history))
        result.loc[group.index] = np.array(vectors)
    return result


def feature_matrix(frame, names=FEATURE_NAMES):
    """The only dataset-to-X selector; never infer features from numeric dtypes."""
    names = validate_feature_names(names)
    return frame.loc[:, list(names)].copy()
