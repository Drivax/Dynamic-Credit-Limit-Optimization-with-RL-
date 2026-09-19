"""Small, strict experiment configuration; no silent ignored keys."""
from pathlib import Path
import yaml
import numpy as np
from .features import validate_feature_names


def load_settings(path, simulation):
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sections = {"target": {"horizon_months"},
        "seeds": {"dataset", "dgp", "macro", "policy", "model", "bootstrap"},
        "split": {"train", "validation", "calibration", "test", "oot"},
        "behavior_policy": {"action_probabilities"}, "models": {"logistic_C", "boosting_iterations", "boosting_leaves"},
        "calibration": {"method"}, "evaluation": {"bootstrap_repetitions", "selected_model"}}
    if not isinstance(settings, dict) or set(settings) != set(sections) | {"features"}:
        raise ValueError("Unknown/missing PD configuration section")
    for section, keys in sections.items():
        if not isinstance(settings[section], dict) or set(settings[section]) != keys:
            raise ValueError(f"Invalid keys in {section}")
    validate_feature_names(settings["features"])
    h = settings["target"]["horizon_months"]
    if type(h) is not int or not 1 <= h <= simulation.environment.horizon:
        raise ValueError("PD horizon outside simulated horizon")
    for section in ("split", "seeds"):
        if any(type(v) is not int or v < (2 if section == "split" else 0) for v in settings[section].values()):
            raise ValueError(f"Invalid {section}")
    p = settings["behavior_policy"]["action_probabilities"]
    if len(p) != len(simulation.environment.action_multipliers) or not np.isfinite(p).all() or min(p) <= 0 or not np.isclose(sum(p), 1):
        raise ValueError("Behavior probabilities must cover every action and sum to 1")
    if settings["calibration"]["method"] != "sigmoid_log_odds":
        raise ValueError("Only sigmoid_log_odds calibration is supported")
    models = settings["models"]
    if not np.isfinite(models["logistic_C"]) or models["logistic_C"] <= 0:
        raise ValueError("logistic_C must be positive and finite")
    for key, minimum in (("boosting_iterations", 1), ("boosting_leaves", 2)):
        if type(models[key]) is not int or models[key] < minimum:
            raise ValueError(f"Invalid {key}")
    if settings["evaluation"]["selected_model"] not in ("logistic", "boosting", "logistic_calibrated", "boosting_calibrated"):
        raise ValueError("Unknown selected model")
    if type(settings["evaluation"]["bootstrap_repetitions"]) is not int or settings["evaluation"]["bootstrap_repetitions"] < 20:
        raise ValueError("At least 20 cluster bootstrap repetitions required")
    return settings
