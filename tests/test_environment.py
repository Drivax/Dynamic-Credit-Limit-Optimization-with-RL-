from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.envs.credit_limit_env import OBSERVATION_NAMES
from credit_rl.simulation.synthetic_snapshot import generate_synthetic_portfolio


def test_gymnasium_api_and_space():
    env = CreditLimitEnv(generate_synthetic_portfolio(12, seed=2))
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=12)
    assert env.observation_space.contains(obs)
    assert env.action_space.contains(env.static_action_index)
    for _ in range(24):
        obs, reward, terminated, truncated, info = env.step(env.static_action_index)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if terminated or truncated:
            break


@pytest.mark.parametrize("action,multiplier", list(enumerate((0.8, 0.9, 1.0, 1.1, 1.2))))
def test_action_updates_same_customer_recursively(action, multiplier, initial_state, traits, no_default_config):
    env = CreditLimitEnv(config=no_default_config)
    env.reset(seed=4, options={"initial_state": initial_state, "traits": traits})
    for month in range(1, 5):
        before = env.state
        env.step(action)
        assert env.state.customer_id == initial_state.customer_id
        assert env.state.month == month
        assert env.state.credit_limit == pytest.approx(np.clip(before.credit_limit * multiplier, 500, 15000))
        assert env.state.tenure_months == before.tenure_months + 1
        assert before.month == month - 1  # snapshots are not retroactively mutated
    history = env.get_history()
    assert len(history) == 5
    history.loc[0, "balance"] = -1
    assert env.get_history().loc[0, "balance"] == initial_state.balance


def test_horizon_and_reset(initial_state, traits, no_default_config):
    cfg = replace(no_default_config, environment=replace(no_default_config.environment, horizon=3))
    env = CreditLimitEnv(config=cfg)
    with pytest.raises(RuntimeError):
        env.step(2)
    options = {"initial_state": initial_state, "traits": traits}
    env.reset(seed=8, options=options)
    for month in range(1, 4):
        _, _, terminated, truncated, _ = env.step(2)
        assert not terminated
        assert truncated == (month == 3)
    with pytest.raises(RuntimeError):
        env.step(2)
    env.reset(seed=8, options=options)
    assert len(env.get_history()) == 1
    assert env.state == initial_state


def test_default_is_terminal_and_loss_is_once(initial_state, traits):
    cfg = SimulationConfig()
    cfg = replace(cfg, default=replace(cfg.default, intercept=1000.0),
                  environment=replace(cfg.environment, horizon=1))
    env = CreditLimitEnv(config=cfg)
    env.reset(seed=1, options={"initial_state": initial_state, "traits": traits})
    obs, reward, terminated, truncated, info = env.step(2)
    assert terminated and not truncated and env.state.defaulted
    assert env.observation_space.contains(obs)
    components = info["reward_components"]
    assert components["credit_loss"] == pytest.approx(cfg.reward.loss_given_default * env.state.balance)
    assert components["interest_income"] == 0
    assert reward == components["total"]
    with pytest.raises(RuntimeError):
        env.step(2)
    assert len(env.get_history()) == 2


@pytest.mark.parametrize("action", [-1, 5, 1.5, "2"])
def test_invalid_action(action, initial_state, traits):
    env = CreditLimitEnv()
    env.reset(seed=0, options={"initial_state": initial_state, "traits": traits})
    with pytest.raises(ValueError):
        env.step(action)
    assert env.state == initial_state


def test_no_hidden_state_or_future_label_leakage():
    portfolio = generate_synthetic_portfolio(10, seed=4)
    poisoned = portfolio.copy()
    poisoned["true_pd"] = 1 - poisoned.true_pd
    poisoned["default_next_month"] = 1 - poisoned.default_next_month
    envs = [CreditLimitEnv(frame) for frame in (portfolio, poisoned)]
    for env in envs:
        env.reset(seed=1, options={"customer_index": 0})
        _, _, _, _, info = env.step(2)
        forbidden = {"traits", "creditworthiness", "spending_propensity", "payment_propensity",
                     "true_pd", "p_default_true", "default_next_month"}
        assert not forbidden.intersection(info)
        assert not forbidden.intersection(env.get_history().columns)
        assert not forbidden.intersection(OBSERVATION_NAMES)
    pd.testing.assert_frame_equal(envs[0].get_history(), envs[1].get_history())


def test_snapshot_only_read_on_reset(no_default_config):
    env = CreditLimitEnv(generate_synthetic_portfolio(10), config=no_default_config)
    env.reset(seed=0, options={"customer_index": 3})
    identifier = env.state.customer_id
    env._portfolio = None  # no further data access should be needed
    for _ in range(4):
        env.step(2)
        assert env.state.customer_id == identifier


def test_long_run_invariants():
    env = CreditLimitEnv(generate_synthetic_portfolio(30))
    for seed in range(30):
        obs, _ = env.reset(seed=seed)
        for _ in range(24):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            assert env.config.environment.min_limit <= env.state.credit_limit <= env.config.environment.max_limit
            assert env.state.balance >= 0 and env.state.utilization >= 0
            assert 0 <= info["predicted_pd"] <= 1
            assert np.isfinite(reward)
            if term or trunc:
                break


def test_limit_bounds_hold_over_full_trajectory(initial_state, traits, no_default_config):
    for action, target in ((0, 500.0), (4, 15000.0)):
        env = CreditLimitEnv(config=no_default_config)
        env.reset(seed=1, options={"initial_state": initial_state, "traits": traits})
        for _ in range(no_default_config.environment.horizon):
            env.step(action)
        assert env.state.credit_limit == target


def test_zero_balance_and_above_limit_observations(initial_state, traits, no_default_config):
    for balance in (0.0, initial_state.credit_limit * 2):
        env = CreditLimitEnv(config=no_default_config)
        obs, _ = env.reset(seed=8, options={"initial_state": replace(initial_state, balance=balance), "traits": traits})
        assert env.observation_space.contains(obs)
        obs, _, _, _, _ = env.step(0)
        assert env.observation_space.contains(obs)
