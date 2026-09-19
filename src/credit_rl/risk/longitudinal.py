"""Serializable probability estimator; stateless across environment instances."""
from dataclasses import dataclass
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import SigmoidCalibration
from .features import FEATURE_NAMES, feature_matrix, history_vector, validate_feature_names


@dataclass
class LongitudinalPDModel:
    estimator: object
    metadata: dict
    calibrator: object = None
    feature_names: tuple = FEATURE_NAMES

    def __post_init__(self):
        self.feature_names = validate_feature_names(self.feature_names)

    def predict_array(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("Wrong feature array shape")
        p = self.estimator.predict_proba(values)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid predicted probability")
        return p

    def predict_proba(self, features):
        # Reject full trajectories/extra columns even if they are numeric.
        if tuple(features.columns) != self.feature_names:
            raise ValueError("Exact ordered feature schema required")
        return self.predict_array(features.to_numpy(dtype=float))

    def predict_history(self, observable_history):
        vector = history_vector(observable_history)
        indices = [FEATURE_NAMES.index(name) for name in self.feature_names]
        return float(self.predict_array(vector[indices][None, :])[0])

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        # Pickle/joblib artifacts must come from a trusted local source.
        result = joblib.load(path)
        if not isinstance(result, cls):
            raise ValueError("Not a longitudinal PD artifact")
        result.__post_init__()
        return result


def train_models(train, calibration, settings, metadata):
    for frame, role in ((train, "train"), (calibration, "calibration")):
        if "partition" not in frame or set(frame.partition) != {role}:
            raise ValueError(f"Expected explicitly designated {role} partition")
    if set(train.customer_id) & set(calibration.customer_id):
        raise ValueError("Calibration customers overlap training")
    if train.label_end.max() >= calibration.observation_month.min():
        raise ValueError("Calibration must follow matured training labels")
    if train.target.nunique() != 2:
        raise ValueError("Training requires both outcome classes")
    names = validate_feature_names(settings["features"])
    x = feature_matrix(train, names).to_numpy()
    xc = feature_matrix(calibration, names).to_numpy()
    seed = settings["seeds"]["model"]
    candidates = {
        "logistic": make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(), LogisticRegression(max_iter=2000, C=settings["models"]["logistic_C"], random_state=seed)),
        "boosting": make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            HistGradientBoostingClassifier(max_iter=settings["models"]["boosting_iterations"],
                max_leaf_nodes=settings["models"]["boosting_leaves"], l2_regularization=1.,
                early_stopping=False, random_state=seed)),
    }
    models, timing = {}, {}
    for name, estimator in candidates.items():
        started = perf_counter()
        estimator.fit(x, train.target)
        info = {**metadata, "model_type": name, "calibration_method": "none",
                "training_prevalence": float(train.target.mean()), "features": list(names)}
        models[name] = LongitudinalPDModel(estimator, info, feature_names=names)
        timing[name] = perf_counter() - started
        calibrator = SigmoidCalibration().fit(estimator.predict_proba(xc)[:, 1], calibration.target)
        models[name + "_calibrated"] = LongitudinalPDModel(estimator,
            {**info, "calibration_method": "sigmoid_log_odds"}, calibrator, names)
    return models, timing
