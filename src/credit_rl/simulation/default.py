"""Simulator hazard. No estimator, predicted PD, label or future macro dependency."""

from credit_rl.config import DefaultConfig
from .customer import CustomerState, CustomerTraits, sigmoid


def true_default_probability(state: CustomerState, traits: CustomerTraits,
                             config: DefaultConfig) -> float:
    """Closing-month hazard conditional on hidden traits and realized behavior.

    This is simulator truth. It is not the beginning-month forecast shown to a policy.
    Credit limits affect risk through utilization/exposure, with no direct limit term.
    """
    if state.defaulted:
        return 1.0
    if state.balance == 0:
        return 0.0
    c = config
    return sigmoid(c.intercept + c.utilization*state.utilization
        + c.debt_to_income*state.balance/state.income + c.months_delinquent*state.months_delinquent
        + c.low_payment*(1-state.payment_ratio) - c.creditworthiness*traits.creditworthiness
        + c.stress*state.macro_state.credit_stress + c.adverse_income*max(0, -state.income_log_change))
