from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.simulation.customer import initialize_customer
from credit_rl.simulation.dgp import CreditDGP
from credit_rl.simulation.macro import MacroState, MacroRegime
from credit_rl.simulation.shocks import ShockPath


def test_custom_macro_initialization_and_extreme_payment_logits():
    cfg = SimulationConfig()
    cfg = replace(cfg, macro=replace(cfg.macro, income_growth=(.003, .02, -.01)),
                  dynamics=replace(cfg.dynamics, payment_logit_mean=1000))
    record = {"internal_score": 650, "income": 3000, "tenure_months": 36,
              "current_limit": 4000, "current_balance": 2000, "monthly_spend": 1200}
    state, latent = initialize_customer(record, "custom", cfg, np.random.default_rng(1))
    assert state.macro_state.income_growth == .02
    assert 0 < latent.payment_propensity < 1


def test_large_elasticity_loading_stays_bounded(initial_state, traits):
    cfg = SimulationConfig()
    cfg = replace(cfg, dynamics=replace(cfg.dynamics, spend_elasticity_loading=10000))
    outcome = CreditDGP(cfg).step(initial_state, 1.2, replace(traits, spending_propensity=2),
        ShockPath.generate("x", 0, 1).months[0], initial_state.macro_state)
    assert outcome.spending_elasticity == pytest.approx(cfg.dynamics.spend_max_elasticity)


def test_correlated_traits_and_persistence(initial_state, traits):
    record = {"internal_score": 650, "income": 3000, "tenure_months": 36,
              "current_limit": 4000, "current_balance": 2000, "monthly_spend": 1200}
    rng = np.random.default_rng(123)
    draws = [initialize_customer(record, str(i), SimulationConfig(), rng)[1] for i in range(3000)]
    table = pd.DataFrame([asdict(t) for t in draws])
    assert table.creditworthiness.corr(table.payment_propensity) > 0.3
    assert table.creditworthiness.corr(table.income_stability) > 0.3
    assert table.creditworthiness.corr(table.spending_propensity) < -0.05
    env = CreditLimitEnv(record_diagnostics=True)
    env.reset(seed=4, options={"initial_state": initial_state, "traits": traits})
    start = env.get_latent_diagnostics()
    for _ in range(24):
        _, _, term, trunc, _ = env.step(2)
        assert env.get_latent_diagnostics() == start
        if term or trunc:
            break
    assert "income_stability" not in env.get_history().columns
    assert "p_default_true" in env.get_diagnostics().columns


def test_income_volatility_and_stress(initial_state, traits):
    cfg = SimulationConfig()
    dgp = CreditDGP(cfg)
    results = {"stable": [], "unstable": [], "stress": []}
    stress = MacroState.from_config(MacroRegime.STRESS, cfg.macro)
    for seed in range(1200):
        shocks = ShockPath.generate("x", seed, 1).months[0]
        for name, latent, state in (
            ("stable", replace(traits, income_stability=1), initial_state),
            ("unstable", replace(traits, income_stability=0), initial_state),
            ("stress", replace(traits, income_stability=0), replace(initial_state, macro_state=stress))):
            end = dgp.step(state, 1, latent, shocks, state.macro_state).state
            assert cfg.dynamics.income_min <= end.income <= cfg.dynamics.income_max
            results[name].append(end.income)
    assert np.std(results["unstable"]) > np.std(results["stable"])*1.5
    assert np.mean(results["stress"]) < np.mean(results["unstable"])


def test_extreme_shocks_are_finite_and_accounting_holds(initial_state, traits):
    cfg = SimulationConfig()
    dgp = CreditDGP(cfg)
    base = ShockPath.generate("x", 0, 1).months[0]
    for sign in (-1, 1):
        shock = replace(base, income_normal=sign*1e12, spending_normal=sign*1e12,
                        payment_normal=sign*1e12, score_normal=sign*1e12)
        outcome = dgp.step(initial_state, 1.2, traits, shock, initial_state.macro_state)
        assert 0 <= outcome.payment <= initial_state.balance
        assert 0 <= outcome.spending <= max(0, outcome.state.credit_limit-initial_state.balance+outcome.payment)
        assert outcome.state.balance == pytest.approx(initial_state.balance-outcome.payment+outcome.spending)
        assert 0 <= outcome.p_default_true <= 1
        assert np.isfinite(outcome.state.income)


def test_exact_hazard_directions(initial_state, traits):
    cfg = SimulationConfig()
    dgp = CreditDGP(cfg)
    base = dgp.true_default_probability(initial_state, traits)
    assert dgp.true_default_probability(replace(initial_state, months_delinquent=2), traits) > base
    assert dgp.true_default_probability(replace(initial_state, balance=3500), traits) > base
    assert dgp.true_default_probability(replace(initial_state, income_log_change=-0.2), traits) > base
    assert dgp.true_default_probability(initial_state, replace(traits, creditworthiness=1)) < base
    stress = MacroState.from_config(MacroRegime.STRESS, cfg.macro)
    assert dgp.true_default_probability(replace(initial_state, macro_state=stress), traits) > base


def test_spending_elasticity_heterogeneity(initial_state, traits):
    cfg = SimulationConfig()
    state = replace(initial_state, credit_limit=10000, balance=500)
    shock = ShockPath.generate("x", 1, 1).months[0]
    dgp = CreditDGP(cfg)
    for propensity in (0.5, 1.5):
        latent = replace(traits, spending_propensity=propensity)
        low = dgp.step(state, 1.0, latent, shock, state.macro_state)
        high = dgp.step(state, 1.2, latent, shock, state.macro_state)
        assert high.spending > low.spending
        assert high.spending-low.spending < high.state.credit_limit-state.credit_limit
        assert high.spending_elasticity == pytest.approx(cfg.dynamics.spend_limit_elasticity*propensity)


def test_no_pandas_in_online_step(monkeypatch, initial_state, traits):
    env = CreditLimitEnv()
    env.reset(seed=4, options={"initial_state": initial_state, "traits": traits})
    def forbidden(*args, **kwargs):
        raise AssertionError("Pandas called in online step")
    monkeypatch.setattr(pd, "DataFrame", forbidden)
    monkeypatch.setattr(pd, "Series", forbidden)
    env.step(2)
