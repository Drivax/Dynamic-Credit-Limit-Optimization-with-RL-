"""Agent information boundary; only current observed quantities enter this projection."""

from math import tanh

import numpy as np

from credit_rl.config import SimulationConfig
from credit_rl.simulation.customer import CustomerState

OBSERVATION_NAMES = (
    "month_fraction", "credit_limit_scaled", "balance_scaled", "utilization_bounded",
    "payment_ratio", "delinquency_status", "months_delinquent_bounded", "income_bounded",
    "behavioral_score_scaled", "macro_stress", "predicted_pd", "spend_bounded",
    "late_payments_fraction", "tenure_bounded", "defaulted", "income_log_change_scaled",
    "macro_income_growth_scaled", "macro_spending_growth_scaled",
    "macro_expansion", "macro_normal", "macro_stress_regime",
)


def build_observation(state: CustomerState, predicted_pd: float, elapsed: int,
                      config: SimulationConfig) -> np.ndarray:
    s, d, cap = state, config.dynamics, config.environment.max_limit
    def bounded(value: float) -> float:
        return value/(1+value)
    m = s.macro_state
    return np.array([
        elapsed/config.environment.horizon, s.credit_limit/cap, bounded(s.balance/cap),
        bounded(s.utilization), s.payment_ratio, s.delinquency_status, bounded(s.months_delinquent),
        bounded(s.income/cap), (s.behavioral_score-d.score_min)/(d.score_max-d.score_min),
        bounded(m.credit_stress), predicted_pd, bounded(s.monthly_spend/cap),
        sum(s.late_history)/6, bounded(s.tenure_months), int(s.defaulted),
        (1+tanh(s.income_log_change))/2, (1+m.income_growth/0.2)/2,
        (1+m.spending_growth/0.5)/2, *(int(m.regime == i) for i in range(3)),
    ], dtype=np.float32)
