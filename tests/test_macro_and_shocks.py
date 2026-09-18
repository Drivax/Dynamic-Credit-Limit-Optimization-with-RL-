from dataclasses import replace

import numpy as np
import pytest

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.config import MacroConfig
from credit_rl.simulation.macro import MacroPath, MacroProcess, MacroRegime, MacroState
from credit_rl.simulation.shocks import MonthlyShocks, ShockPath


@pytest.mark.parametrize("name", ["baseline", "mild_stress", "severe_stress", "recovery"])
def test_macro_scenario_roundtrip_and_length(name, tmp_path):
    process = MacroProcess(MacroConfig())
    path = process.scenario(name, 24)
    assert len(path.states) == 25
    assert path == process.scenario(name, 24)
    path.save(tmp_path / "macro.json")
    assert path == MacroPath.load(tmp_path / "macro.json")


def test_markov_reproducibility_persistence_and_all_regimes():
    process = MacroProcess(MacroConfig())
    a = process.generate(10000, np.random.default_rng(55))
    b = process.generate(10000, np.random.default_rng(55))
    assert a == b
    regimes = np.array([s.regime for s in a.states])
    assert set(regimes) == {0, 1, 2}
    assert np.mean(regimes[1:] == regimes[:-1]) > 0.8


@pytest.mark.parametrize("matrix", [((1,0,0),), ((0.8,0.3,-0.1),)*3, ((0.5,0.5,0.5),)*3,
                                     ((float("nan"),0,1),)*3])
def test_invalid_macro_matrix(matrix):
    with pytest.raises(ValueError):
        MacroConfig(transition_matrix=matrix)


def test_invalid_scenario_and_length(initial_state, traits):
    with pytest.raises(ValueError):
        MacroConfig(scenarios={"bad": ((1, 1.0, 1),)})
    with pytest.raises(ValueError):
        MacroConfig(scenarios={"bad": ((0.3, 1, 1),)})
    with pytest.raises(ValueError):
        MacroConfig(scenarios={"bad": ((1, 2, 50),)})
    env = CreditLimitEnv()
    with pytest.raises(ValueError, match="length"):
        env.reset(seed=1, options={"initial_state": initial_state, "traits": traits,
            "macro_path": MacroPath.constant(MacroState(), 12)})


def test_shocks_prefix_identity_and_serialization(tmp_path):
    a, b = ShockPath.generate("A", 17, 24), ShockPath.generate("A", 17, 36)
    assert a.months == b.months[:24]
    assert a.months != ShockPath.generate("B", 17, 24).months
    assert a.months != ShockPath.generate("A", 18, 24).months
    a.save(tmp_path/"shocks.json")
    assert a == ShockPath.load(tmp_path/"shocks.json")
    with pytest.raises(ValueError):
        replace(a.months[0], default_uniform=1.0)


def test_future_macro_shocks_do_not_leak(initial_state, traits):
    cfg = SimulationConfig()
    baseline = MacroProcess(cfg.macro).scenario("baseline", 24)
    stress = MacroState.from_config(MacroRegime.STRESS, cfg.macro)
    future_changed = MacroPath((baseline.states[0],)+(stress,)*24)
    shocks = ShockPath.generate(initial_state.customer_id, 17, 24)
    changed_shocks = replace(shocks, months=(replace(shocks.months[0], default_uniform=0.0),)+shocks.months[1:])
    envs = [CreditLimitEnv(record_diagnostics=True), CreditLimitEnv(record_diagnostics=True)]
    observations = []
    for env, macro, path in zip(envs, (baseline, future_changed), (shocks, changed_shocks)):
        obs, info = env.reset(seed=1, options={"initial_state": initial_state, "traits": traits,
            "macro_path": macro, "shock_path": path})
        observations.append(obs)
        assert "p_default_true" not in info
    np.testing.assert_array_equal(*observations)


def test_common_shocks_survive_different_terminal_times(initial_state, traits, no_default_config):
    path = ShockPath.generate(initial_state.customer_id, 4, 24)
    macro = MacroProcess(no_default_config.macro).scenario("severe_stress", 24)
    from credit_rl.config import DefaultConfig
    forced = replace(no_default_config, default=replace(DefaultConfig(), intercept=1000))
    for cfg, action in ((forced, 0), (no_default_config, 4)):
        env = CreditLimitEnv(config=cfg, record_diagnostics=True)
        env.reset(seed=1, options={"initial_state": initial_state, "traits": traits,
            "macro_path": macro, "shock_path": path})
        while True:
            _, _, term, trunc, info = env.step(action)
            assert "p_default_true" not in info
            if term or trunc:
                break
        diagnostics = env.get_diagnostics()
        for t, row in enumerate(diagnostics.itertuples()):
            assert row.shock_default_uniform == path.months[t].default_uniform
            assert row.shock_income_normal == path.months[t].income_normal
        assert (len(diagnostics) == 1) if cfg == forced else (len(diagnostics) == 24)


def test_requested_and_effective_action(initial_state, traits):
    cfg = SimulationConfig()
    cfg = replace(cfg, environment=replace(cfg.environment, max_monthly_increase=0.05, max_monthly_decrease=0.03))
    for action, requested, effective in ((4, 0.2, 0.05), (0, -0.2, -0.03)):
        env = CreditLimitEnv(config=cfg)
        env.reset(seed=3, options={"initial_state": initial_state, "traits": traits})
        _, _, _, _, info = env.step(action)
        assert info["requested_action"] == pytest.approx(requested)
        assert info["effective_action"] == pytest.approx(effective)
