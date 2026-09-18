"""Monthly DGP assumptions. YAML overrides are strict; unknown keys fail fast."""

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EnvironmentConfig:
    horizon: int = 24
    action_multipliers: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    min_limit: float = 500.0
    max_limit: float = 15000.0


@dataclass(frozen=True)
class DynamicsConfig:
    # Once-per-customer draws, conditional on observed initial score.
    latent_credit_noise: float = 0.6
    score_reference: float = 650.0
    score_scale: float = 100.0
    spending_log_sigma: float = 0.25
    payment_logit_mean: float = -0.4
    payment_logit_sigma: float = 0.5
    initial_payment_ratio: float = 0.4
    # Monthly spending demand (EUR), bounded by remaining available credit.
    spend_income_fraction: float = 0.35
    spend_persistence: float = 0.35
    spend_limit_elasticity: float = 0.15
    spend_stress: float = 0.15
    spend_shock_sigma: float = 0.25
    # Payment fraction of opening principal, then an income affordability cap.
    payment_persistence: float = 0.5
    payment_credit: float = 0.35
    payment_utilization: float = 0.65
    payment_burden: float = 0.3
    payment_delinquency: float = 0.65
    payment_stress: float = 0.6
    payment_shock_sigma: float = 0.4
    payment_income_cap: float = 0.5
    minimum_payment_ratio: float = 0.05
    # Missed-payment event, with memory through consecutive delinquent months.
    missed_intercept: float = -3.0
    missed_utilization: float = 0.8
    missed_burden: float = 0.3
    missed_delinquency: float = 0.6
    missed_credit: float = 0.7
    missed_stress: float = 0.7
    missed_payment_fraction: float = 0.02
    score_reversion: float = 0.1
    score_delinquency_drop: float = 18.0
    score_payment_gain: float = 4.0
    score_noise: float = 3.0
    score_min: float = 300.0
    score_max: float = 950.0
    # Two-state exogenous Markov chain, probabilities per month.
    normal_to_stress: float = 0.04
    stress_to_normal: float = 0.2


@dataclass(frozen=True)
class DefaultConfig:
    intercept: float = -6.0
    utilization: float = 1.2
    debt_to_income: float = 0.45
    months_delinquent: float = 0.65
    low_payment: float = 0.6
    creditworthiness: float = 0.8
    stress: float = 0.9


@dataclass(frozen=True)
class PDConfig:
    """Observable-only, deliberately imperfect fallback risk proxy; not fitted."""
    intercept: float = -5.0
    utilization: float = 1.0
    debt_to_income: float = 0.35
    months_delinquent: float = 0.5
    score: float = -0.004
    score_reference: float = 650.0
    stress: float = 0.5
    # Compatibility mapping for the preserved snapshot-trained GB classifier.
    normal_unemployment: float = 0.07
    stress_unemployment: float = 0.10
    normal_inflation: float = 0.024
    stress_inflation: float = 0.04
    normal_policy_rate: float = 0.022
    stress_policy_rate: float = 0.04
    spend_burden_weight: float = 0.35


@dataclass(frozen=True)
class RewardConfig:
    annual_percentage_rate: float = 0.18
    fee_rate: float = 0.012
    loss_given_default: float = 0.55
    annual_funding_rate: float = 0.03
    rwa_factor: float = 0.08
    capital_weight: float = 1.5
    max_pd_threshold: float = 0.12
    constraint_weight: float = 25.0


@dataclass(frozen=True)
class SimulationConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    default: DefaultConfig = field(default_factory=DefaultConfig)
    pd: PDConfig = field(default_factory=PDConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        e, d, r = self.environment, self.dynamics, self.reward
        for section in asdict(self).values():
            for value in section.values():
                values = value if isinstance(value, (list, tuple)) else (value,)
                if not all(isfinite(v) for v in values):
                    raise ValueError("Configuration must be finite")
        if isinstance(e.horizon, bool) or not isinstance(e.horizon, int) or e.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not 0 < e.min_limit <= e.max_limit:
            raise ValueError("Invalid limit bounds")
        if not e.action_multipliers or any(v <= 0 for v in e.action_multipliers):
            raise ValueError("Action multipliers must be positive")
        if e.action_multipliers.count(1.0) != 1:
            raise ValueError("Exactly one maintain action is required")
        for name in ("initial_payment_ratio", "spend_persistence", "payment_persistence",
                     "minimum_payment_ratio", "missed_payment_fraction", "score_reversion",
                     "normal_to_stress", "stress_to_normal", "spend_stress"):
            if not 0 <= getattr(d, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name, value in asdict(d).items():
            if name not in ("payment_logit_mean", "missed_intercept") and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if d.score_scale <= 0 or d.score_min >= d.score_max:
            raise ValueError("Invalid score scaling/bounds")
        if d.missed_payment_fraction >= d.minimum_payment_ratio:
            raise ValueError("A missed payment must be below the minimum ratio")
        if not 0 <= r.loss_given_default <= 1 or not 0 <= r.max_pd_threshold <= 1:
            raise ValueError("Invalid reward probabilities")
        if any(v < 0 for v in asdict(r).values()):
            raise ValueError("Reward coefficients must be nonnegative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        constructors = {"environment": EnvironmentConfig, "dynamics": DynamicsConfig,
                        "default": DefaultConfig, "pd": PDConfig, "reward": RewardConfig}
        if not isinstance(data, dict) or set(data) - constructors.keys():
            raise ValueError("Unknown configuration section")
        if "action_multipliers" in data.get("environment", {}):
            data["environment"]["action_multipliers"] = tuple(data["environment"]["action_multipliers"])
        return cls(**{key: constructors[key](**value) for key, value in data.items()})
