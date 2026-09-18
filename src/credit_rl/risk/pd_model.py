"""Independent risk-estimation interface; no access to hidden traits or DGP."""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from credit_rl.config import PDConfig
from credit_rl.simulation.customer import CustomerState, MacroState, sigmoid


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
    macro_stress: int

    @classmethod
    def from_state(cls, state: CustomerState) -> "ObservedRiskFeatures":
        return cls(state.income, state.tenure_months, state.behavioral_score, state.utilization,
                   state.monthly_spend, state.months_delinquent, sum(state.late_history),
                   state.balance / state.income, int(state.macro_state))


class PDModel(Protocol):
    def predict(self, features: ObservedRiskFeatures) -> float: ...


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

    def predict(self, features: ObservedRiskFeatures) -> float:
        # Lazy import keeps sklearn optional for core simulation.
        from .training import RISK_FEATURES

        f, c = features, self.config
        stress = f.macro_stress == MacroState.STRESS
        frame = pd.DataFrame([{
            "income": f.income, "tenure_months": f.tenure_months,
            "internal_score": f.behavioral_score, "utilization": f.utilization,
            "monthly_spend": f.monthly_spend, "delinquency_30d": f.months_delinquent,
            "late_payments_6m": f.late_payments_6m,
            "macro_unemployment": c.stress_unemployment if stress else c.normal_unemployment,
            "macro_inflation": c.stress_inflation if stress else c.normal_inflation,
            "macro_policy_rate": c.stress_policy_rate if stress else c.normal_policy_rate,
            "debt_to_income": f.debt_to_income + c.spend_burden_weight * f.monthly_spend / f.income,
        }], columns=RISK_FEATURES)
        return float(self.model.predict_proba(frame)[0, 1])
