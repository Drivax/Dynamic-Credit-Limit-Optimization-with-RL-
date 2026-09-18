from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.evaluation.metrics import summarize_trajectories
from credit_rl.reward import calculate_reward
from credit_rl.risk.pd_model import ObservedRiskFeatures, SnapshotPDModel
from credit_rl.simulation.dgp import TransitionOutcome


def test_reward_formula_by_hand(initial_state):
    cfg = SimulationConfig().reward
    result = TransitionOutcome(replace(initial_state, balance=3000), payment=500, spending=1000, exposure=3000)
    parts = calculate_reward(initial_state, result, 0.2, cfg)
    assert parts.interest_income == pytest.approx(2500 * 0.18 / 12)
    assert parts.fee_income == 12
    assert parts.funding_cost == pytest.approx(7.5)
    assert parts.capital_cost == pytest.approx(72)
    assert parts.constraint_penalty == pytest.approx(2)
    assert parts.credit_loss == 0
    assert parts.total == pytest.approx(37.5 + 12 - 7.5 - 72 - 2)
    default = replace(result, state=replace(result.state, defaulted=True))
    parts = calculate_reward(initial_state, default, 0.2, cfg)
    assert parts.credit_loss == pytest.approx(1650)
    assert parts.total == pytest.approx(12 - 1650 - 7.5 - 72 - 2)


def test_adapter_feature_allowlist_and_refresh(initial_state):
    pytest.importorskip("sklearn")
    from credit_rl.risk.training import RISK_FEATURES

    class SpyClassifier:
        def predict_proba(self, frame):
            assert isinstance(frame, np.ndarray)
            self.frame = pd.DataFrame(frame, columns=RISK_FEATURES)
            return np.array([[0.7, 0.3]])

    classifier = SpyClassifier()
    model = SnapshotPDModel(classifier)
    assert model.predict(ObservedRiskFeatures.from_state(initial_state)) == 0.3
    first = classifier.frame.copy()
    changed = replace(initial_state, balance=3500, monthly_spend=2200, behavioral_score=580,
                      months_delinquent=2, late_history=(0, 0, 0, 0, 1, 1))
    model.predict(ObservedRiskFeatures.from_state(changed))
    assert classifier.frame.utilization.iloc[0] == 3500 / 4000
    assert classifier.frame.late_payments_6m.iloc[0] == 2
    assert classifier.frame.debt_to_income.iloc[0] > first.debt_to_income.iloc[0]
    assert classifier.frame.internal_score.iloc[0] == 580
    assert not {"defaulted", "true_pd", "default_next_month", "creditworthiness"}.intersection(
        asdict(ObservedRiskFeatures.from_state(changed)))


@pytest.mark.parametrize("probability", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_pd_fails_early(probability, initial_state, traits):
    class BadPD:
        def predict(self, features):
            return probability

    env = CreditLimitEnv(pd_model=BadPD())
    with pytest.raises(ValueError, match="probability"):
        env.reset(seed=1, options={"initial_state": initial_state, "traits": traits})


def test_metrics_are_customer_level_and_ignore_initialization():
    frame = pd.DataFrame({
        "policy": ["static"] * 5, "episode_id": [0, 0, 1, 1, 1],
        "action": [np.nan, 2, np.nan, 2, 2], "month": [0, 1, 0, 1, 2],
        "reward": [0, -10, 0, 3, 5], "defaulted": [False, True, False, False, False],
        "credit_limit": [1000] * 5,
    })
    result = summarize_trajectories(frame).iloc[0]
    assert result.episodes == 2
    assert result.mean_return == -1
    assert result.default_incidence == 0.5  # not one default / three active months
    assert result.mean_months == 1.5


def test_config_yaml_matches_defaults_and_rejects_unknown(tmp_path):
    assert SimulationConfig.from_yaml("configs/simulation.yaml") == SimulationConfig()
    path = tmp_path / "bad.yaml"
    path.write_text("dynamics:\n  misspelled_coefficient: 1\n")
    with pytest.raises(TypeError):
        SimulationConfig.from_yaml(path)
    path.write_text("environment:\n  horizon: 0\n")
    with pytest.raises(ValueError):
        SimulationConfig.from_yaml(path)


@pytest.mark.parametrize("section,overrides", [
    ("environment", {"min_limit": 0}),
    ("environment", {"action_multipliers": (0.8, 1.2)}),
    ("dynamics", {"income_adverse_probability": 1.1}),
    ("dynamics", {"spend_shock_sigma": -0.1}),
    ("reward", {"loss_given_default": 1.1}),
    ("default", {"intercept": float("nan")}),
])
def test_invalid_configuration(section, overrides):
    config = SimulationConfig()
    with pytest.raises(ValueError):
        replace(config, **{section: replace(getattr(config, section), **overrides)})
