"""Monthly DGP assumptions. YAML overrides are strict; unknown keys fail fast."""

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path

import yaml

DGP_VERSION = "2.0"


@dataclass(frozen=True)
class EnvironmentConfig:
    horizon: int = 24
    action_multipliers: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)
    min_limit: float = 500.0
    max_limit: float = 15000.0
    max_monthly_increase: float = 0.20
    max_monthly_decrease: float = 0.20


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
    # One shared normal factor plus independent residuals (not discrete types).
    latent_factor_loadings: tuple[float, ...] = (0.7, -0.25, 0.65, 0.65)
    stability_logit_mean: float = 0.8
    stability_logit_sigma: float = 0.6
    # Income: monthly log changes around the customer's initial income.
    income_reversion: float = 0.08
    income_sigma: float = 0.025
    income_unstable_volatility: float = 2.0
    income_stress_volatility: float = 0.5
    income_macro_vulnerability: float = 1.0
    income_adverse_probability: float = 0.01
    income_stress_adverse_probability: float = 0.06
    income_adverse_log_drop: float = 0.15
    income_max_log_drop: float = 0.35
    income_max_log_gain: float = 0.20
    income_min: float = 300.0
    income_max: float = 30000.0
    # Monthly spending demand (EUR), bounded by remaining available credit.
    spend_income_fraction: float = 0.35
    spend_persistence: float = 0.35
    spend_limit_elasticity: float = 0.15
    spend_elasticity_loading: float = 1.0
    spend_max_elasticity: float = 0.60
    spend_shock_sigma: float = 0.25
    shock_clip: float = 6.0
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


@dataclass(frozen=True)
class DefaultConfig:
    intercept: float = -6.0
    utilization: float = 1.2
    debt_to_income: float = 0.45
    months_delinquent: float = 0.65
    low_payment: float = 0.6
    creditworthiness: float = 0.8
    stress: float = 0.9
    adverse_income: float = 1.5


@dataclass(frozen=True)
class MacroConfig:
    # Row/column order: EXPANSION, NORMAL, STRESS.
    transition_matrix: tuple[tuple[float, ...], ...] = (
        (0.88, 0.11, 0.01), (0.035, 0.93, 0.035), (0.01, 0.14, 0.85))
    # Exactly three observable factors per regime.
    income_growth: tuple[float, ...] = (0.003, 0.001, -0.01)
    spending_growth: tuple[float, ...] = (0.02, 0.0, -0.12)
    credit_stress: tuple[float, ...] = (0.0, 0.0, 1.0)
    # (fraction of horizon, regime index, severity multiplier).
    scenarios: dict[str, tuple[tuple[float, int, float], ...]] = field(default_factory=lambda: {
        "baseline": ((1.0, 1, 1.0),),
        "mild_stress": ((0.25, 1, 1.0), (0.25, 2, 0.65), (0.5, 1, 1.0)),
        "severe_stress": ((1/6, 1, 1.0), (2/3, 2, 1.6), (1/6, 1, 1.0)),
        "recovery": ((0.25, 2, 1.3), (0.25, 2, 0.6), (0.25, 1, 1.0), (0.25, 0, 1.0)),
    })

    def __post_init__(self) -> None:
        if len(self.transition_matrix) != 3 or any(len(row) != 3 for row in self.transition_matrix):
            raise ValueError("Macro transition matrix must be 3 by 3")
        for row in self.transition_matrix:
            if any(not isfinite(p) or not 0 <= p <= 1 for p in row) or abs(sum(row)-1) > 1e-9:
                raise ValueError("Macro transition rows must be probabilities summing to one")
        for name, bound in (("income_growth", 0.2), ("spending_growth", 0.5), ("credit_stress", 3.0)):
            values = getattr(self, name)
            if len(values) != 3 or any(not isfinite(v) or abs(v) > bound for v in values):
                raise ValueError(f"Invalid macro factor {name}")
        if min(self.credit_stress) < 0:
            raise ValueError("Credit stress must be nonnegative")
        for name, phases in self.scenarios.items():
            if not phases or abs(sum(p[0] for p in phases)-1) > 1e-9:
                raise ValueError(f"Scenario {name} fractions must sum to one")
            for fraction, regime, severity in phases:
                if not all(isfinite(x) for x in (fraction, regime, severity)):
                    raise ValueError("Scenario must be finite")
                if fraction <= 0 or type(regime) is not int or regime not in (0, 1, 2) or severity < 0:
                    raise ValueError("Invalid scenario phase")
                if (abs(self.income_growth[regime]*severity) > 0.2
                    or abs(self.spending_growth[regime]*severity) > 0.5
                    or self.credit_stress[regime]*severity > 3):
                    raise ValueError("Scenario factors exceed supported bounds")


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
    macro: MacroConfig = field(default_factory=MacroConfig)

    def __post_init__(self) -> None:
        e, d, r = self.environment, self.dynamics, self.reward
        for section_name, section in asdict(self).items():
            if section_name == "macro":
                continue
            for value in section.values():
                values = value if isinstance(value, (list, tuple)) else (value,)
                if not all(isfinite(v) for v in values):
                    raise ValueError("Configuration must be finite")
        if isinstance(e.horizon, bool) or not isinstance(e.horizon, int) or e.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not 0 < e.min_limit < e.max_limit:
            raise ValueError("Invalid limit bounds")
        if not e.action_multipliers or any(v <= 0 for v in e.action_multipliers):
            raise ValueError("Action multipliers must be positive")
        if e.action_multipliers.count(1.0) != 1:
            raise ValueError("Exactly one maintain action is required")
        if not 0 <= e.max_monthly_increase <= 1 or not 0 <= e.max_monthly_decrease < 1:
            raise ValueError("Invalid monthly action bounds")
        for name in ("initial_payment_ratio", "spend_persistence", "payment_persistence",
                     "minimum_payment_ratio", "missed_payment_fraction", "score_reversion",
                     "income_reversion", "income_adverse_probability", "income_stress_adverse_probability"):
            if not 0 <= getattr(d, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name, value in asdict(d).items():
            if name not in ("payment_logit_mean", "missed_intercept", "stability_logit_mean", "latent_factor_loadings") and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if d.score_scale <= 0 or d.score_min >= d.score_max:
            raise ValueError("Invalid score scaling/bounds")
        if len(d.latent_factor_loadings) != 4 or any(abs(v) > 1 for v in d.latent_factor_loadings):
            raise ValueError("Four factor loadings in [-1, 1] required")
        if not 0 < d.income_min < d.income_max or not 0 < d.shock_clip <= 10:
            raise ValueError("Invalid income or shock bounds")
        if max(d.income_max_log_drop, d.income_max_log_gain) > 1:
            raise ValueError("Monthly income changes must be bounded in log units")
        if d.spend_shock_sigma * d.shock_clip > 20 or d.spending_log_sigma * d.shock_clip > 20:
            raise ValueError("Exponential coefficients too large")
        if d.missed_payment_fraction >= d.minimum_payment_ratio:
            raise ValueError("A missed payment must be below the minimum ratio")
        if not 0 <= r.loss_given_default <= 1 or not 0 <= r.max_pd_threshold <= 1:
            raise ValueError("Invalid reward probabilities")
        if any(v < 0 for v in asdict(r).values()):
            raise ValueError("Reward coefficients must be nonnegative")

    @classmethod
    def from_yaml(cls, path: str | Path, macro_path: str | Path | None = None) -> "SimulationConfig":
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if data is None:
            data = {}
        constructors = {"environment": EnvironmentConfig, "dynamics": DynamicsConfig,
                        "default": DefaultConfig, "pd": PDConfig, "reward": RewardConfig}
        if not isinstance(data, dict) or set(data) - constructors.keys():
            raise ValueError("Unknown configuration section")
        if "action_multipliers" in data.get("environment", {}):
            data["environment"]["action_multipliers"] = tuple(data["environment"]["action_multipliers"])
        if "latent_factor_loadings" in data.get("dynamics", {}):
            data["dynamics"]["latent_factor_loadings"] = tuple(data["dynamics"]["latent_factor_loadings"])
        result = {key: constructors[key](**value) for key, value in data.items()}
        macro_file = Path(macro_path) if macro_path is not None else Path(path).with_name("macro_scenarios.yaml")
        if macro_file.exists():
            macro_data = yaml.safe_load(macro_file.read_text(encoding="utf-8"))
            if not isinstance(macro_data, dict):
                raise ValueError("Macro configuration must be a mapping")
            for name in ("income_growth", "spending_growth", "credit_stress"):
                if name in macro_data:
                    macro_data[name] = tuple(macro_data[name])
            if "transition_matrix" in macro_data:
                macro_data["transition_matrix"] = tuple(tuple(row) for row in macro_data["transition_matrix"])
            if "scenarios" in macro_data:
                macro_data["scenarios"] = {key: tuple(tuple(phase) for phase in phases)
                                           for key, phases in macro_data["scenarios"].items()}
            result["macro"] = MacroConfig(**macro_data)
        elif macro_path is not None:
            raise FileNotFoundError(macro_file)
        return cls(**result)
