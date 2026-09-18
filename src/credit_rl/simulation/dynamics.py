"""Monthly principal dynamics and a simulator-only default mechanism."""

from dataclasses import dataclass, replace

import numpy as np

from credit_rl.config import SimulationConfig
from .customer import CustomerState, CustomerTraits, MacroState, sigmoid


@dataclass(frozen=True)
class TransitionOutcome:
    state: CustomerState
    payment: float
    spending: float
    exposure: float


class TransitionModel:
    """Payments precede purchases; default occurs on closing principal.

    Limit cuts never forgive principal. Above-limit debt is allowed after cuts,
    but purchases cannot increase it. Interest/fees are cash reward proxies,
    not capitalized into principal. No PD predictor is accepted by this class.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def true_default_probability(self, state: CustomerState, traits: CustomerTraits) -> float:
        """Conditional monthly probability under the assumed DGP, not empirical truth."""
        if state.defaulted:
            return 1.0
        if state.balance == 0:
            return 0.0
        c = self.config.default
        return sigmoid(c.intercept + c.utilization * state.utilization
                       + c.debt_to_income * state.balance / state.income
                       + c.months_delinquent * state.months_delinquent
                       + c.low_payment * (1 - state.payment_ratio)
                       - c.creditworthiness * traits.creditworthiness
                       + c.stress * int(state.macro_state))

    def step(self, state: CustomerState, multiplier: float, traits: CustomerTraits,
             rng: np.random.Generator, macro_rng: np.random.Generator) -> TransitionOutcome:
        if state.defaulted:
            raise RuntimeError("Default is absorbing; reset before further transitions")
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("multiplier must be finite and positive")
        d, e = self.config.dynamics, self.config.environment
        new_limit = float(np.clip(state.credit_limit * multiplier, e.min_limit, e.max_limit))
        # Draw the same number of shocks every active month, regardless of action.
        payment_noise, spend_noise, score_noise = rng.normal(size=3)
        missed_uniform, default_uniform = rng.random(2)
        macro_uniform = macro_rng.random()
        stress = int(state.macro_state)
        utilization = state.balance / new_limit
        burden = state.balance / state.income
        payment_logit = np.log(traits.payment_propensity / (1 - traits.payment_propensity))
        desired_ratio = sigmoid(payment_logit + d.payment_credit * traits.creditworthiness
                                - d.payment_utilization * utilization - d.payment_burden * burden
                                - d.payment_delinquency * state.months_delinquent
                                - d.payment_stress * stress + d.payment_shock_sigma * payment_noise)
        payment_ratio = (d.payment_persistence * state.payment_ratio
                         + (1 - d.payment_persistence) * desired_ratio)
        missed_probability = sigmoid(d.missed_intercept + d.missed_utilization * utilization
                                     + d.missed_burden * burden
                                     + d.missed_delinquency * state.months_delinquent
                                     - d.missed_credit * traits.creditworthiness + d.missed_stress * stress)
        if missed_uniform < missed_probability:
            payment_ratio = min(payment_ratio, d.missed_payment_fraction)
        payment = min(state.balance * payment_ratio, d.payment_income_cap * state.income)
        actual_ratio = payment / state.balance if state.balance > 0 else 1.0
        delinquent = state.balance > 0 and actual_ratio < d.minimum_payment_ratio
        months_late = state.months_delinquent + 1 if delinquent else 0
        remaining = max(0.0, state.balance - payment)
        demand = ((d.spend_persistence * state.monthly_spend
                   + (1 - d.spend_persistence) * d.spend_income_fraction * state.income
                   * traits.spending_propensity)
                  * (new_limit / state.credit_limit) ** d.spend_limit_elasticity
                  * (1 - d.spend_stress * stress)
                  * np.exp(d.spend_shock_sigma * spend_noise - d.spend_shock_sigma**2 / 2))
        spending = float(min(demand, max(0.0, new_limit - remaining)))
        balance = remaining + spending
        score = float(np.clip(
            state.behavioral_score + d.score_reversion * (state.initial_score - state.behavioral_score)
            - d.score_delinquency_drop * delinquent + d.score_payment_gain * actual_ratio
            + d.score_noise * score_noise, d.score_min, d.score_max))
        candidate = replace(state, month=state.month + 1, credit_limit=new_limit,
                            balance=balance, payment_ratio=actual_ratio, months_delinquent=months_late,
                            behavioral_score=score, tenure_months=state.tenure_months + 1,
                            monthly_spend=spending, late_history=state.late_history[1:] + (int(delinquent),))
        # Current M_t governs this month; M_(t+1) is revealed for the next decision.
        defaulted = bool(default_uniform < self.true_default_probability(candidate, traits))
        if state.macro_state == MacroState.NORMAL:
            macro_next = MacroState.STRESS if macro_uniform < d.normal_to_stress else MacroState.NORMAL
        else:
            macro_next = MacroState.NORMAL if macro_uniform < d.stress_to_normal else MacroState.STRESS
        return TransitionOutcome(replace(candidate, macro_state=macro_next, defaulted=defaulted),
                                 float(payment), spending, float(balance))
