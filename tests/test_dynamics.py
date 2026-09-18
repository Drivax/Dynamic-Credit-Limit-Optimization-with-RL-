from dataclasses import replace

import numpy as np
import pytest

from credit_rl import SimulationConfig
from credit_rl.simulation.customer import MacroState
from credit_rl.simulation.dynamics import TransitionModel


def transition(state, traits, config, multiplier=1.0, seed=4):
    return TransitionModel(config).step(state, multiplier, traits,
                                        np.random.default_rng(seed), np.random.default_rng(seed + 1))


def test_balance_accounting_and_no_debt_forgiveness(initial_state, traits, no_default_config):
    state = replace(initial_state, balance=3900, payment_ratio=0)
    cfg = replace(no_default_config, dynamics=replace(no_default_config.dynamics, payment_income_cap=0))
    result = transition(state, traits, cfg, multiplier=0.8)
    assert result.state.credit_limit == 3200
    assert result.state.balance == 3900
    assert result.spending == 0 and result.payment == 0
    assert result.state.utilization > 1
    assert result.state.balance == state.balance - result.payment + result.spending


def test_previous_balance_changes_future(initial_state, traits, no_default_config):
    high = transition(initial_state, traits, no_default_config)
    low = transition(replace(initial_state, balance=200), traits, no_default_config)
    assert high.state.balance != low.state.balance
    assert high.payment != low.payment


def test_default_absorption_in_transition(initial_state, traits):
    with pytest.raises(RuntimeError):
        transition(replace(initial_state, defaulted=True), traits, SimulationConfig())


def test_zero_debt_default_probability(initial_state, traits):
    assert TransitionModel(SimulationConfig()).true_default_probability(
        replace(initial_state, balance=0), traits) == 0


def test_delinquency_memory_and_cure(initial_state, traits, no_default_config):
    cfg = replace(no_default_config, dynamics=replace(no_default_config.dynamics, missed_intercept=1000))
    state = initial_state
    for month in range(1, 4):
        result = transition(state, traits, cfg)
        assert result.state.months_delinquent == month
        assert result.state.late_history[-1] == 1
        state = result.state
    cure_cfg = replace(no_default_config, dynamics=replace(no_default_config.dynamics,
        missed_intercept=-1000, payment_persistence=1, payment_income_cap=10))
    cured = transition(replace(state, payment_ratio=1), traits, cure_cfg)
    assert cured.state.months_delinquent == 0
    assert sum(cured.state.late_history) == 3  # history survives cure


def test_latent_traits_and_delinquency_have_persistent_effects(initial_state, traits, no_default_config):
    current, delinquent, weak = [], [], []
    for seed in range(400):
        current.append(transition(initial_state, traits, no_default_config, seed=seed).state.delinquency_status)
        delinquent.append(transition(replace(initial_state, months_delinquent=3), traits,
                                     no_default_config, seed=seed).state.delinquency_status)
        weak.append(transition(initial_state, replace(traits, creditworthiness=-2),
                               no_default_config, seed=seed).state.delinquency_status)
    assert np.mean(delinquent) > np.mean(current) + 0.15
    assert np.mean(weak) > np.mean(current) + 0.1


def test_macro_stress_worsens_risk_statistically(initial_state, traits):
    cfg = SimulationConfig()
    normal, stress = [], []
    for seed in range(1000):
        normal.append(transition(initial_state, traits, cfg, seed=seed).state)
        stress.append(transition(replace(initial_state, macro_state=MacroState.STRESS), traits, cfg, seed=seed).state)
    assert np.mean([s.payment_ratio for s in stress]) < np.mean([s.payment_ratio for s in normal])
    assert np.mean([s.defaulted for s in stress]) > np.mean([s.defaulted for s in normal])
