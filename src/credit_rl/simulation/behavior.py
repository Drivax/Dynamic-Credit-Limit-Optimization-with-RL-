"""Income, repayment and purchase behavior; pure functions of state, traits and shocks."""

from dataclasses import dataclass, replace
from math import exp, log

from credit_rl.config import DynamicsConfig
from .customer import CustomerState, CustomerTraits, sigmoid
from .shocks import MonthlyShocks


def clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


@dataclass(frozen=True)
class BehaviorOutcome:
    state: CustomerState
    payment: float
    spending: float
    income_log_change: float
    adverse_income_event: bool
    spending_elasticity: float


def evolve_behavior(state: CustomerState, traits: CustomerTraits, new_limit: float,
                    shocks: MonthlyShocks, config: DynamicsConfig) -> BehaviorOutcome:
    """Realize income, pay opening principal, purchase within headroom, update score.

    Principal accounting: B'=B-P+C. Fees/interest remain separate economic cash
    proxies as in v1, not capitalized. Cuts do not forgive over-limit principal.
    """
    d, m = config, state.macro_state
    instability = 1-traits.income_stability
    sigma = d.income_sigma * (1+d.income_unstable_volatility*instability) * (1+d.income_stress_volatility*m.credit_stress)
    adverse_probability = clip(d.income_adverse_probability
        + d.income_stress_adverse_probability*m.credit_stress*instability, 0, 1)
    adverse = shocks.income_uniform < adverse_probability
    log_change = (d.income_reversion*(log(state.initial_income)-log(state.income))
                  + m.income_growth*(1+d.income_macro_vulnerability*instability)
                  + sigma*clip(shocks.income_normal, -d.shock_clip, d.shock_clip)-sigma*sigma/2
                  - d.income_adverse_log_drop*adverse)
    log_change = clip(log_change, -d.income_max_log_drop, d.income_max_log_gain)
    income = clip(state.income*exp(log_change), d.income_min, d.income_max)
    income_change = log(income/state.income)
    utilization, burden = state.balance/new_limit, state.balance/income
    payment_logit = log(traits.payment_propensity)-log(1-traits.payment_propensity)
    desired_ratio = sigmoid(payment_logit + d.payment_credit*traits.creditworthiness
        - d.payment_utilization*utilization - d.payment_burden*burden
        - d.payment_delinquency*state.months_delinquent - d.payment_stress*m.credit_stress
        + d.payment_shock_sigma*clip(shocks.payment_normal, -d.shock_clip, d.shock_clip))
    payment_ratio = d.payment_persistence*state.payment_ratio + (1-d.payment_persistence)*desired_ratio
    missed_probability = sigmoid(d.missed_intercept + d.missed_utilization*utilization
        + d.missed_burden*burden + d.missed_delinquency*state.months_delinquent
        - d.missed_credit*traits.creditworthiness + d.missed_stress*m.credit_stress)
    if shocks.missed_uniform < missed_probability:
        payment_ratio = min(payment_ratio, d.missed_payment_fraction)
    payment = min(state.balance*payment_ratio, d.payment_income_cap*income)
    actual_ratio = payment/state.balance if state.balance > 0 else 1.0
    delinquent = state.balance > 0 and actual_ratio < d.minimum_payment_ratio
    months_late = state.months_delinquent+1 if delinquent else 0
    remaining = max(0.0, state.balance-payment)
    if d.spend_limit_elasticity == 0 or d.spend_max_elasticity == 0:
        elasticity = 0.0
    else:
        elasticity = exp(min(log(d.spend_max_elasticity), log(d.spend_limit_elasticity)
                             + d.spend_elasticity_loading*log(traits.spending_propensity)))
    log_spend_multiplier = (elasticity*log(new_limit/state.credit_limit)
        + d.spend_shock_sigma*clip(shocks.spending_normal, -d.shock_clip, d.shock_clip)
        - d.spend_shock_sigma**2/2)
    demand = (d.spend_persistence*state.monthly_spend
        + (1-d.spend_persistence)*d.spend_income_fraction*income*traits.spending_propensity)
    # Headroom is applied before exponentiation to avoid overflow for large inputs.
    headroom = max(0.0, new_limit-remaining)
    if demand <= 0 or headroom == 0:
        spending = 0.0
    else:
        log_demand = log(demand) + log(1+m.spending_growth) + log_spend_multiplier
        spending = exp(min(log_demand, log(headroom)))
        spending = min(spending, headroom)
    balance = remaining+spending
    score = clip(state.behavioral_score + d.score_reversion*(state.initial_score-state.behavioral_score)
        - d.score_delinquency_drop*delinquent + d.score_payment_gain*actual_ratio
        + d.score_noise*clip(shocks.score_normal, -d.shock_clip, d.shock_clip), d.score_min, d.score_max)
    next_state = replace(state, month=state.month+1, credit_limit=new_limit, balance=balance,
        payment_ratio=actual_ratio, months_delinquent=months_late, income=income,
        income_log_change=income_change, behavioral_score=score, tenure_months=state.tenure_months+1,
        monthly_spend=spending, late_history=state.late_history[1:]+(int(delinquent),))
    return BehaviorOutcome(next_state, payment, spending, income_change, adverse, elasticity)
