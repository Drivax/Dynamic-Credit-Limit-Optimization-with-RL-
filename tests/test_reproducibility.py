from dataclasses import replace

import numpy as np
import pandas as pd

from credit_rl import CreditLimitEnv
from credit_rl.policies.baselines import ConstantPolicy
from credit_rl.simulation.simulator import simulate_customer


def run(initial_state, traits, config, action, seed=19, pd_model=None):
    return simulate_customer(CreditLimitEnv(config=config, pd_model=pd_model), ConstantPolicy(action),
        seed=seed, options={"initial_state": initial_state, "traits": traits}).to_dataframe()


def test_reproducibility_and_global_rng_isolation(initial_state, traits, no_default_config):
    first = run(initial_state, traits, no_default_config, 2)
    np.random.seed(999)
    np.random.random(1000)
    second = run(initial_state, traits, no_default_config, 2)
    pd.testing.assert_frame_equal(first, second)
    third = run(initial_state, traits, no_default_config, 2, seed=20)
    assert not first.balance.equals(third.balance)


def test_actions_change_future_but_not_exogenous_macro(initial_state, traits, no_default_config):
    increase = run(initial_state, traits, no_default_config, 3)
    decrease = run(initial_state, traits, no_default_config, 1)
    assert not increase.balance.equals(decrease.balance)
    assert not increase.utilization.equals(decrease.utilization)
    assert not increase.reward.equals(decrease.reward)
    pd.testing.assert_series_equal(increase.macro_state, decrease.macro_state)


def test_pd_model_does_not_generate_default(initial_state, traits):
    from credit_rl import SimulationConfig

    class ConstantPD:
        def __init__(self, value):
            self.value = value

        def predict(self, features):
            assert not hasattr(features, "creditworthiness")
            assert not hasattr(features, "defaulted")
            return self.value

    a = run(initial_state, traits, SimulationConfig(), 2, pd_model=ConstantPD(0.0))
    b = run(initial_state, traits, SimulationConfig(), 2, pd_model=ConstantPD(1.0))
    for column in ("balance", "credit_limit", "defaulted", "months_delinquent", "macro_state"):
        pd.testing.assert_series_equal(a[column], b[column])
    assert not a.predicted_pd.equals(b.predicted_pd)


def test_hidden_traits_not_directly_observable(initial_state, traits):
    envs = [CreditLimitEnv(), CreditLimitEnv()]
    a, _ = envs[0].reset(seed=5, options={"initial_state": initial_state, "traits": traits})
    b, _ = envs[1].reset(seed=5, options={"initial_state": initial_state,
        "traits": replace(traits, creditworthiness=-3, spending_propensity=2)})
    np.testing.assert_array_equal(a, b)
    envs[0].step(2)
    envs[1].step(2)
    assert envs[0].state != envs[1].state
