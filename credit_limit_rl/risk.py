from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

RISK_FEATURES = [
    "income",
    "tenure_months",
    "internal_score",
    "utilization",
    "monthly_spend",
    "delinquency_30d",
    "late_payments_6m",
    "macro_unemployment",
    "macro_inflation",
    "macro_policy_rate",
    "debt_to_income",
]


def train_risk_model(portfolio: pd.DataFrame, random_state: int = 42) -> tuple[GradientBoostingClassifier, dict[str, Any]]:
    features = portfolio[RISK_FEATURES]
    target = portfolio["default_next_month"]
    # Stratification preserves class imbalance between train and validation sets.
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )

    model = GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=100,
        random_state=random_state,
    )
    # Risk model approximates next-month default probability (PD).
    model.fit(x_train, y_train)

    valid_scores = model.predict_proba(x_valid)[:, 1]
    fpr, tpr, _ = roc_curve(y_valid, valid_scores)
    importance_frame = pd.DataFrame(
        {
            "feature": RISK_FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    validation_frame = pd.DataFrame(
        {
            "default_next_month": y_valid.to_numpy(),
            "predicted_pd": valid_scores,
        }
    )
    metrics = {
        "roc_auc": float(roc_auc_score(y_valid, valid_scores)),
        "average_precision": float(average_precision_score(y_valid, valid_scores)),
        "brier_score": float(brier_score_loss(y_valid, valid_scores)),
        "log_loss": float(log_loss(y_valid, valid_scores)),
        "ks_statistic": float(np.max(tpr - fpr)),
        "default_rate": float(target.mean()),
        "validation_samples": int(len(validation_frame)),
    }
    # Diagnostics feed both reporting and dashboard visualizations.
    diagnostics = {
        "metrics": metrics,
        "validation_predictions": validation_frame,
        "feature_importance": importance_frame,
    }
    return model, diagnostics
