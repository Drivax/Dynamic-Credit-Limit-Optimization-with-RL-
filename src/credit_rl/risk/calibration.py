"""Sigmoid recalibration on a dedicated, later customer cohort."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def log_odds(probabilities):
    p = np.clip(probabilities, 1e-8, 1-1e-8)
    return (np.log(p) - np.log1p(-p)).reshape(-1, 1)


class SigmoidCalibration:
    def fit(self, probabilities, labels):
        if len(np.unique(labels)) != 2:
            raise ValueError("Calibration requires both outcome classes")
        self.estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.estimator.fit(log_odds(probabilities), labels)
        return self

    def predict(self, probabilities):
        return self.estimator.predict_proba(log_odds(probabilities))[:, 1]
