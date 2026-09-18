"""Observable monthly snapshots and separately held persistent hidden traits."""

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from typing import Mapping

import numpy as np

from credit_rl.config import SimulationConfig


class MacroState(IntEnum):
    NORMAL = 0
    STRESS = 1


@dataclass(frozen=True)
class CustomerTraits:
    creditworthiness: float
    spending_propensity: float
    payment_propensity: float

    def __post_init__(self) -> None:
        if not all(isfinite(x) for x in (self.creditworthiness, self.spending_propensity,
                                        self.payment_propensity)):
            raise ValueError("Traits must be finite")
        if self.spending_propensity <= 0 or not 0 < self.payment_propensity < 1:
            raise ValueError("Invalid propensity")


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
    macro_state: MacroState = MacroState.NORMAL
    defaulted: bool = False

    def __post_init__(self) -> None:
        for value in (self.credit_limit, self.balance, self.payment_ratio, self.income,
                      self.behavioral_score, self.initial_score, self.monthly_spend):
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
        if self.macro_state not in (MacroState.NORMAL, MacroState.STRESS):
            raise ValueError("Unknown macro state")

    @property
    def utilization(self) -> float:
        return self.balance / self.credit_limit

    @property
    def delinquency_status(self) -> int:
        return int(self.months_delinquent > 0)


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
    )
    traits = CustomerTraits(
        creditworthiness=(score - d.score_reference) / d.score_scale + rng.normal(0, d.latent_credit_noise),
        spending_propensity=float(rng.lognormal(-d.spending_log_sigma**2 / 2, d.spending_log_sigma)),
        payment_propensity=sigmoid(float(rng.normal(d.payment_logit_mean, d.payment_logit_sigma))),
    )
    return state, traits
