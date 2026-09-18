"""Observable monthly snapshots and separately held persistent hidden traits."""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import numpy as np

from credit_rl.config import SimulationConfig
from .macro import MacroState, MacroRegime


@dataclass(frozen=True)
class CustomerTraits:
    creditworthiness: float
    spending_propensity: float
    payment_propensity: float
    income_stability: float = 0.7

    def __post_init__(self) -> None:
        if not all(isfinite(x) for x in (self.creditworthiness, self.spending_propensity,
                                        self.payment_propensity, self.income_stability)):
            raise ValueError("Traits must be finite")
        if self.spending_propensity <= 0 or not 0 < self.payment_propensity < 1:
            raise ValueError("Invalid propensity")
        if not 0 <= self.income_stability <= 1:
            raise ValueError("income_stability must lie in [0,1]")


@dataclass(frozen=True)
class CustomerState:
    customer_id: str
    month: int
    credit_limit: float
    balance: float
    payment_ratio: float
    months_delinquent: int
    income: float
    behavioral_score: float
    initial_score: float
    tenure_months: int
    monthly_spend: float
    late_history: tuple[int, ...]
    macro_state: MacroState = MacroState()
    defaulted: bool = False
    initial_income: float | None = None
    income_log_change: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_income is None:
            object.__setattr__(self, "initial_income", self.income)
        for value in (self.credit_limit, self.balance, self.payment_ratio, self.income,
                      self.behavioral_score, self.initial_score, self.monthly_spend,
                      self.initial_income, self.income_log_change):
            if not isfinite(value):
                raise ValueError("State values must be finite")
        if self.credit_limit <= 0 or self.income <= 0 or self.balance < 0 or self.monthly_spend < 0:
            raise ValueError("Invalid financial state")
        if not 0 <= self.payment_ratio <= 1:
            raise ValueError("payment_ratio must be in [0, 1]")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
               for v in (self.month, self.months_delinquent, self.tenure_months)):
            raise ValueError("Month counters must be nonnegative integers")
        if len(self.late_history) != 6 or any(v not in (0, 1) for v in self.late_history):
            raise ValueError("late_history must contain six binary observations")
        if self.initial_income <= 0:
            raise ValueError("Initial income must be positive")
        if not isinstance(self.macro_state, MacroState):
            raise ValueError("Unknown macro state")

    @property
    def utilization(self) -> float:
        return self.balance / self.credit_limit

    @property
    def delinquency_status(self) -> int:
        return int(self.months_delinquent > 0)

    @property
    def delinquency_bucket(self) -> int:
        """0/current, 1/one, 2/two, 3/three-plus consecutive shortfall months; not legal DPD."""
        return min(self.months_delinquent, 3)


def sigmoid(value: float) -> float:
    """Numerically stable logistic link."""
    return float(np.exp(-np.logaddexp(0.0, -value)))


def initialize_customer(record: Mapping, customer_id: str, config: SimulationConfig,
                        rng: np.random.Generator) -> tuple[CustomerState, CustomerTraits]:
    """Use an initial snapshot once; never consume its true_pd or future target.

    No ordered payment history exists in snapshots: late counts are placed in
    the most recent slots as an explicit initialization assumption.
    """
    d, e = config.dynamics, config.environment
    score = float(np.clip(record["internal_score"], d.score_min, d.score_max))
    late_count = int(np.clip(record.get("late_payments_6m", 0), 0, 6))
    months_late = int(max(0, record.get("delinquency_30d", 0)))
    state = CustomerState(
        customer_id=str(customer_id), month=0,
        credit_limit=float(np.clip(record["current_limit"], e.min_limit, e.max_limit)),
        balance=float(record["current_balance"]), payment_ratio=d.initial_payment_ratio,
        months_delinquent=months_late, income=float(record["income"]),
        behavioral_score=score, initial_score=score,
        tenure_months=int(record["tenure_months"]), monthly_spend=float(record["monthly_spend"]),
        late_history=tuple([0] * (6 - late_count) + [1] * late_count),
        macro_state=MacroState.from_config(MacroRegime.NORMAL, config.macro),
    )
    common, *residuals = np.clip(rng.normal(size=5), -d.shock_clip, d.shock_clip)
    factors = [loading*common + np.sqrt(1-loading**2)*residual
               for loading, residual in zip(d.latent_factor_loadings, residuals)]
    factors = np.clip(factors, -d.shock_clip, d.shock_clip)
    traits = CustomerTraits(
        creditworthiness=(score-d.score_reference)/d.score_scale + d.latent_credit_noise*factors[0],
        spending_propensity=float(np.exp(-d.spending_log_sigma**2/2 + d.spending_log_sigma*factors[1])),
        payment_propensity=float(np.clip(sigmoid(d.payment_logit_mean + d.payment_logit_sigma*factors[2]),
                                         np.finfo(float).eps, 1-np.finfo(float).eps)),
        income_stability=sigmoid(d.stability_logit_mean + d.stability_logit_sigma*factors[3]),
    )
    return state, traits
