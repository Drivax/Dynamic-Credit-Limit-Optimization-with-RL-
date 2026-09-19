from collections import deque
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.risk.dataset import build_dataset, assert_split_integrity
from credit_rl.risk.features import (FEATURE_NAMES, SIMULATOR_ONLY, build_features, feature_matrix,
                                    history_vector, observable_row, validate_feature_names)
from credit_rl.risk.generation import generate_cohort
from credit_rl.risk.longitudinal import LongitudinalPDModel, train_models
from credit_rl.risk.settings import load_settings
from credit_rl.risk.evaluation import cluster_intervals, metrics, calibration_table
from credit_rl.simulation.macro import MacroProcess


def trajectory(state, length=8, event=None, identity="A", start=0, scheduled=8):
    rows = []
    for t in range(length+1):
        current = replace(state, month=t, balance=1000+100*t, income=3000+10*t,
                          payment_ratio=.2+.01*t, income_log_change=.001*t)
        rows.append(dict(customer_id=identity, month=t, defaulted=t == event,
            scheduled_end=scheduled, cohort_start=start, macro_state=1, **observable_row(current)))
        if t == event:
            break
    return pd.DataFrame(rows)


def test_exact_window_boundaries_and_absorbing_default(initial_state):
    data, audit = build_dataset(trajectory(initial_state, event=4), 3)
    assert data.month.tolist() == [0, 1, 2, 3]
    assert data.target.tolist() == [0, 1, 1, 1]  # t+H included; t+H+1 excluded
    assert audit.already_defaulted.iloc[0] == 1


def test_censoring_and_administrative_eligibility(initial_state):
    truncated = trajectory(initial_state, length=4)
    data, audit = build_dataset(truncated, 3)
    assert data.month.tolist() == [0, 1]
    assert data.target.sum() == 0
    assert audit.censored_unknown.iloc[0] == 3
    # No inclusion of near-end positives just because their early event is observed.
    data, _ = build_dataset(trajectory(initial_state, event=8), 3)
    assert data.month.max() == 5
    assert data.loc[data.month == 5, "target"].item() == 1


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_horizon(initial_state, horizon):
    with pytest.raises(ValueError):
        build_dataset(trajectory(initial_state), horizon)


def test_past_summaries_exclude_current_and_future(initial_state):
    raw = trajectory(initial_state)
    features = build_features(raw)
    assert features.loc[3, "utilization_mean3"] == pytest.approx(raw.loc[:2, "utilization"].mean())
    assert features.loc[3, "balance_change1"] == pytest.approx(100/1200)
    assert np.isnan(features.loc[0, "utilization_lag1"])
    assert features.loc[0, "history_months"] == 0
    poison = raw.copy()
    poison.loc[4:, list(FEATURE_NAMES[:14])] = 9999
    pd.testing.assert_frame_equal(build_features(poison).iloc[:4], features.iloc[:4])
    poison.loc[3, "utilization"] = 9999
    assert build_features(poison).loc[3, "utilization_mean3"] == features.loc[3, "utilization_mean3"]


def test_customer_history_isolation_and_online_parity(initial_state):
    a = trajectory(initial_state)
    b = trajectory(replace(initial_state, credit_limit=8000), identity="B")
    raw = pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=9)
    offline = build_features(raw)
    for _, group in raw.groupby("customer_id"):
        history = deque(maxlen=7)
        for index, row in group.sort_values("month").iterrows():
            history.append(row.to_dict())
            np.testing.assert_allclose(history_vector(history), offline.loc[index].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("column", sorted(SIMULATOR_ONLY) + ["latent_creditworthiness", "future_balance", "next_income", "target", "numeric_new_dgp_column"])
def test_aggressive_allowlist(initial_state, column):
    raw = trajectory(initial_state)
    raw[column] = 123.
    assert column not in build_features(raw).columns
    with pytest.raises(ValueError, match="allowlist"):
        validate_feature_names(FEATURE_NAMES + (column,))
    with pytest.raises(ValueError):
        feature_matrix(raw, tuple(raw.columns))


def test_duplicate_and_missing_month_fail(initial_state):
    raw = trajectory(initial_state)
    with pytest.raises(ValueError, match="contiguous"):
        build_features(raw.drop(index=2))
    with pytest.raises(ValueError, match="Duplicate"):
        build_features(pd.concat([raw, raw]))


def test_split_identity_and_label_maturity(initial_state):
    a, _ = build_dataset(trajectory(initial_state), 3)
    b, _ = build_dataset(trajectory(initial_state, identity="B", start=9), 3)
    assert_split_integrity({"train": a, "test": b})
    with pytest.raises(ValueError, match="Customer overlap"):
        assert_split_integrity({"train": a, "test": b.assign(customer_id="A")})
    with pytest.raises(ValueError, match="Temporal overlap"):
        assert_split_integrity({"train": a, "test": b.assign(observation_month=8)})


@pytest.fixture(scope="module")
def fitted():
    cfg = SimulationConfig()
    settings = load_settings("configs/pd_model.yaml", cfg)
    settings["models"]["boosting_iterations"] = 5
    samples = []
    raw_train = None
    for index in (0, 1):
        raw, _ = generate_cohort(cfg, customers=70, cohort_index=index, cohort_start=index*25,
            seeds=settings["seeds"], macro_path=MacroProcess(cfg.macro).scenario("baseline", 24))
        data, _ = build_dataset(raw, 12)
        data["partition"] = "train" if index == 0 else "calibration"
        samples.append(data)
        if index == 0:
            raw_train = raw
    with threadpool_limits(limits=1):
        models, _ = train_models(*samples, settings, {"horizon_months": 12})
    return models, samples, settings, raw_train


def test_train_only_preprocessing_and_probability_roundtrip(fitted, tmp_path):
    models, (train, cal), settings, _ = fitted
    expected = np.nanmedian(feature_matrix(train).to_numpy(), axis=0)
    for name, model in models.items():
        np.testing.assert_allclose(model.estimator[0].statistics_, expected)
        x = feature_matrix(cal)
        with threadpool_limits(limits=1):
            p = model.predict_proba(x)
            assert np.all((p >= 0) & (p <= 1))
            path = tmp_path/f"{name}.joblib"
            model.save(path)
            np.testing.assert_array_equal(p, LongitudinalPDModel.load(path).predict_proba(x))
            with pytest.raises(ValueError, match="schema"):
                model.predict_proba(x.assign(p_default_true=0.2))
    with pytest.raises(ValueError, match="overlap"):
        train_models(train, train.assign(partition="calibration"), settings, {})
    with pytest.raises(ValueError, match="partition"):
        train_models(train, cal.assign(partition="test"), settings, {})


def test_calibration_cannot_refit_preprocessor_or_base_model(fitted):
    models, (train, cal), settings, _ = fitted
    changed = cal.copy()
    changed["income"] *= 100
    with threadpool_limits(limits=1):
        again, _ = train_models(train, changed, settings, {})
        for name in ("logistic", "boosting"):
            x = feature_matrix(cal)
            np.testing.assert_array_equal(models[name].predict_proba(x), again[name].predict_proba(x))


def test_environment_prediction_before_future_and_reset(fitted, initial_state, traits, monkeypatch):
    model = fitted[0]["logistic_calibrated"]
    env = CreditLimitEnv(pd_model=model, record_history=False)
    obs, _ = env.reset(seed=19, options={"initial_state": initial_state, "traits": traits})
    assert obs[10] == pytest.approx(model.predict_history([observable_row(initial_state)]))
    expected = env._predicted_pd
    def forbidden(*args, **kwargs):
        raise AssertionError("Pandas used during online inference")
    monkeypatch.setattr(pd.DataFrame, "__init__", forbidden)
    _, _, _, _, info = env.step(2)
    assert info["decision_pd"] == expected
    assert info["predicted_pd"] == model.predict_history(tuple(env._risk_history))
    obs2, _ = env.reset(seed=19, options={"initial_state": initial_state, "traits": traits})
    np.testing.assert_array_equal(obs, obs2)


def test_generation_reproducible_and_paired_initial_states(fitted):
    settings = fitted[2]
    cfg = SimulationConfig()
    kwargs = dict(customers=8, cohort_index=3, cohort_start=75, seeds=settings["seeds"],
                  macro_path=MacroProcess(cfg.macro).scenario("baseline", 24))
    a, da = generate_cohort(cfg, **kwargs)
    b, db = generate_cohort(cfg, **kwargs)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(da, db)
    c, _ = generate_cohort(cfg, **kwargs, policy="always_increase")
    pd.testing.assert_frame_equal(a[a.month == 0].reset_index(drop=True), c[c.month == 0].reset_index(drop=True))


def test_cluster_bootstrap_constant_ties_and_single_class():
    frame = pd.DataFrame({"customer_id": ["a"]*3+["b"]*3, "target": [0]*3+[1]*3})
    p = np.full(6, .5)
    ci = cluster_intervals(frame, p, 100, 9)
    assert ci["brier_lower"] == ci["brier_upper"] == .25
    assert ci["roc_auc_bootstrap_valid"] < 100  # one-class cluster draws skipped
    assert ci == cluster_intervals(frame, p, 100, 9)
    assert len(calibration_table(frame.target, p, quantile=True)) == 1
    assert np.isnan(metrics([0, 0], [.1, .2])["roc_auc"])


def test_quantile_plot_handles_filtered_nonzero_index():
    y = pd.Series([0, 0, 1, 1], index=[100, 101, 102, 103])
    p = pd.Series([.1, .2, .8, .9], index=y.index)
    result = calibration_table(y, p, quantile=True)
    assert result.n_obs.sum() == 4
    assert len(result) == 4
