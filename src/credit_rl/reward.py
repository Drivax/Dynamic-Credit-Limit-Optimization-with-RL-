"""Economic reward proxies, with realized credit loss counted exactly once."""

from dataclasses import asdict, dataclass

from .config import RewardConfig
from .simulation.customer import CustomerState
from .simulation.dgp import TransitionOutcome


@dataclass(frozen=True)
class RewardBreakdown:
    interest_income: float
    fee_income: float
    credit_loss: float
    funding_cost: float
    capital_cost: float
    constraint_penalty: float

    @property
    def total(self) -> float:
        return (self.interest_income + self.fee_income - self.credit_loss
                - self.funding_cost - self.capital_cost - self.constraint_penalty)

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "total": self.total}


def calculate_reward(previous: CustomerState, outcome: TransitionOutcome,
                     decision_pd: float, config: RewardConfig) -> RewardBreakdown:
    """Opening-balance interest proxy; closing exposure losses and costs.

    No interest is recognized in a default month. Interchange on purchases is
    retained. PD-driven capital/risk terms are soft proxies, not regulation.
    """
    defaulted = outcome.state.defaulted
    return RewardBreakdown(
        interest_income=0.0 if defaulted else previous.balance * config.annual_percentage_rate / 12,
        fee_income=outcome.spending * config.fee_rate,
        credit_loss=float(defaulted) * config.loss_given_default * outcome.exposure,
        funding_cost=config.annual_funding_rate / 12 * outcome.exposure,
        capital_cost=config.capital_weight * config.rwa_factor * decision_pd * outcome.exposure,
        constraint_penalty=config.constraint_weight * max(0.0, decision_pd - config.max_pd_threshold),
    )
