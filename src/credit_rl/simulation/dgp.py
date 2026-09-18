"""Hidden DGP orchestration: effective action -> behavior -> hazard -> default."""

from dataclasses import dataclass, replace
from math import isfinite

from credit_rl.config import SimulationConfig
from .behavior import clip, evolve_behavior
from .customer import CustomerState, CustomerTraits
from .default import true_default_probability
from .macro import MacroState
from .shocks import MonthlyShocks


@dataclass(frozen=True)
class TransitionOutcome:
    state: CustomerState
    payment: float
    spending: float
    exposure: float
    p_default_true: float = 0.0
    requested_multiplier: float = 1.0
    effective_multiplier: float = 1.0
    adverse_income_event: bool = False
    spending_elasticity: float = 0.0


class CreditDGP:
    """Pure, RNG-free one-month DGP; caller supplies indexed shocks and next macro.

    No policy or PD estimator enters this object. Diagnostic outcomes are private
    to the simulator; the Gymnasium wrapper explicitly selects public info fields.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def true_default_probability(self, state: CustomerState, traits: CustomerTraits) -> float:
        return true_default_probability(state, traits, self.config.default)

    def step(self, state: CustomerState, multiplier: float, traits: CustomerTraits,
             shocks: MonthlyShocks, next_macro: MacroState) -> TransitionOutcome:
        if state.defaulted:
            raise RuntimeError("Default is absorbing; reset before further transitions")
        if not isfinite(multiplier) or multiplier <= 0:
            raise ValueError("multiplier must be finite and positive")
        e = self.config.environment
        capped_multiplier = clip(multiplier, 1-e.max_monthly_decrease, 1+e.max_monthly_increase)
        limit = clip(state.credit_limit*capped_multiplier, e.min_limit, e.max_limit)
        behavior = evolve_behavior(state, traits, limit, shocks, self.config.dynamics)
        hazard = self.true_default_probability(behavior.state, traits)
        defaulted = shocks.default_uniform < hazard
        # Hazard uses current M_t, never the future regime supplied for the next observation.
        end_state = replace(behavior.state, defaulted=defaulted, macro_state=next_macro)
        return TransitionOutcome(end_state, behavior.payment, behavior.spending, end_state.balance,
            hazard, multiplier, limit/state.credit_limit, behavior.adverse_income_event, behavior.spending_elasticity)
