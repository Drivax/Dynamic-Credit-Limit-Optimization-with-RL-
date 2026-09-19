"""Independent risk-estimation interface; no access to hidden traits or DGP."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from credit_rl.config import PDConfig
from credit_rl.simulation.customer import CustomerState, sigmoid


@dataclass(frozen=True)
class ObservedRiskFeatures:
    income: float
    tenure_months: int
    behavioral_score: float
    utilization: float
    monthly_spend: float
    months_delinquent: int
    late_payments_6m: int
    debt_to_income: float
    macro_stress: float

    @classmethod
    def from_state(cls, state: CustomerState) -> "ObservedRiskFeatures":
        return cls(state.income, state.tenure_months, state.behavioral_score, state.utilization,
                   state.monthly_spend, state.months_delinquent, sum(state.late_history),
                   state.balance / state.income, state.macro_state.credit_stress)


class PDModel(Protocol):
    def predict(self, features: ObservedRiskFeatures) -> float: ...


class HistoryPDModel(Protocol):
    """Stateless forecast from a bounded sequence of explicitly observable rows."""
    def predict_history(self, observable_history: tuple[dict, ...]) -> float: ...


@dataclass(frozen=True)
class ObservedLogisticPD:
    """Transparent fallback proxy; assumed coefficients, no calibration claim."""
    config: PDConfig = PDConfig()

    def predict(self, features: ObservedRiskFeatures) -> float:
        c, f = self.config, features
        return sigmoid(c.intercept + c.utilization * f.utilization
                       + c.debt_to_income * f.debt_to_income
                       + c.months_delinquent * f.months_delinquent
                       + c.score * (f.behavioral_score - c.score_reference) + c.stress * f.macro_stress)


@dataclass
class SnapshotPDModel:
    """Adapt the preserved GB classifier to current observed monthly features.

    Original training labels are synthetic one-month snapshot defaults, not the
    new longitudinal DGP. This domain mismatch is intentional technical debt.
    """
    model: object
    config: PDConfig = PDConfig()
    _array_model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from .training import RISK_FEATURES
        self._array_model = self.model
        if hasattr(self.model, "feature_names_in_"):
            if list(self.model.feature_names_in_) != RISK_FEATURES:
                raise ValueError("Snapshot model feature schema mismatch")
            self._array_model = deepcopy(self.model)
            del self._array_model.feature_names_in_

    def predict(self, features: ObservedRiskFeatures) -> float:
        # Lazy import keeps sklearn optional for core simulation.
        from .training import RISK_FEATURES

        f, c = features, self.config
        stress = min(f.macro_stress, 1.0)
        row = {
            "income": f.income, "tenure_months": f.tenure_months,
            "internal_score": f.behavioral_score, "utilization": f.utilization,
            "monthly_spend": f.monthly_spend, "delinquency_30d": f.months_delinquent,
            "late_payments_6m": f.late_payments_6m,
            "macro_unemployment": c.normal_unemployment + stress*(c.stress_unemployment-c.normal_unemployment),
            "macro_inflation": c.normal_inflation + stress*(c.stress_inflation-c.normal_inflation),
            "macro_policy_rate": c.normal_policy_rate + stress*(c.stress_policy_rate-c.normal_policy_rate),
            "debt_to_income": f.debt_to_income + c.spend_burden_weight * f.monthly_spend / f.income,
        }
        values = np.array([[row[name] for name in RISK_FEATURES]], dtype=np.float64)
        return float(self._array_model.predict_proba(values)[0, 1])
