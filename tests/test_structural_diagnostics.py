from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.evaluation.structural import (
    DecisionSnapshot, METRICS, SIGNS, action_statistics, admissible,
    decompose, hypothetical_paths, rollout, value_tables,
)
from credit_rl.experiments.structural_diagnostics import dominance_tables
from credit_rl.policies.decision import ConstantAdjustment
from credit_rl.simulation.dgp import CreditDGP
from credit_rl.simulation.shocks import ShockPath


class FixedPD:
    def predict(self, features):
        return .25


def setup(initial_state, traits, config=None):
    cfg = config or SimulationConfig()
    env = CreditLimitEnv(config=cfg, pd_model=FixedPD(), severe_delinquency_months=3)
    env.reset(seed=17, options=dict(initial_state=initial_state, traits=traits))
    return env, DecisionSnapshot.capture(env)


def test_decomposition_and_batch_matches_canonical_env(initial_state, traits):
    env, snap = setup(initial_state, traits)
    cfg = env.config
    before = (asdict(env.state), tuple(env._risk_history), asdict(cfg))
    paths = hypothetical_paths(snap, 8, 24, 989)
    policy = ConstantAdjustment(cfg, .8)
    cube = rollout(snap, cfg, env.pd_model, policy, paths, 6)
    assert cube.shape == (8, 5, 6, len(METRICS))
    for draw, path in enumerate(paths):
        for action in range(5):
            branch = deepcopy(env)
            branch._shock_path = path
            for k in range(6):
                if branch._done:
                    assert (cube[draw, action, k:] == 0).all()
                    break
                requested = action if k == 0 else policy.act(branch._get_observation())
                _, reward, _, _, info = branch.step(requested)
                np.testing.assert_allclose(cube[draw, action, k, 6], reward, rtol=1e-12)
                parts = np.array([info['reward_components'][name] for name in METRICS[:6]])
                np.testing.assert_allclose(cube[draw, action, k, :6], parts)
                assert cube[draw, action, k, 7] == pytest.approx(parts[:4] @ SIGNS[:4])
    assert before == (asdict(env.state), tuple(env._risk_history), asdict(cfg))
    np.testing.assert_array_equal(cube, rollout(snap, cfg, env.pd_model, policy, paths, 6))


def test_crn_identical_channels_across_actions_and_branches(initial_state, traits, monkeypatch):
    env, snap = setup(initial_state, traits)
    paths = hypothetical_paths(snap, 8, 24, 324)
    records = []
    original = CreditDGP.step

    def capture(self, state, multiplier, hidden, shocks, macro):
        records.append((state.month, shocks))
        return original(self, state, multiplier, hidden, shocks, macro)

    monkeypatch.setattr(CreditDGP, 'step', capture)
    rollout(snap, env.config, env.pd_model, ConstantAdjustment(env.config), paths, 1)
    assert len(records) == 40
    for draw in range(8):
        assert all(records[draw*5+a][1] is paths[draw].months[0] for a in range(5))
    assert paths == hypothetical_paths(snap, 8, 24, 324)
    assert paths != hypothetical_paths(snap, 8, 24, 325)
    assert paths[0] != env._shock_path


def test_remaining_horizon_default_absorption_and_input_guards(initial_state, traits):
    env, snap = setup(initial_state, traits)
    paths = hypothetical_paths(snap, 8, 24, 41)
    late = replace(snap, elapsed=23, state=replace(snap.state, month=23))
    cube = rollout(late, env.config, env.pd_model, ConstantAdjustment(env.config), paths, 12)
    assert cube.shape[2] == 1
    config = replace(env.config, default=replace(env.config.default, intercept=1000))
    absorbed = rollout(snap, config, env.pd_model, ConstantAdjustment(config), paths, 12)
    assert (absorbed[:, :, 1:] == 0).all()
    assert (absorbed[:, :, 0, METRICS.index('default_incidence')] == 1).all()
    with pytest.raises(ValueError):
        rollout(replace(snap, elapsed=24), config, env.pd_model, None, paths, 1)
    with pytest.raises(ValueError):
        rollout(replace(snap, state=replace(snap.state, defaulted=True)), config, env.pd_model, None, paths, 1)
    with pytest.raises(ValueError):
        hypothetical_paths(snap, 1, 1, 1)


def test_constraints_floor_and_blocked_aliases(initial_state, traits):
    state = replace(initial_state, credit_limit=500, months_delinquent=3)
    env, snap = setup(state, traits)
    paths = hypothetical_paths(snap, 8, 24, 2)
    cube = rollout(snap, env.config, env.pd_model, ConstantAdjustment(env.config), paths, 1)
    assert admissible(snap, env.config) == [2]
    for a in range(5):
        np.testing.assert_array_equal(cube[:, a], cube[:, 2])
    values, gaps, _ = value_tables(snap, env.config, cube, 0, 'Static', (1,), (.98,))
    assert values.admissible.sum() == 1
    assert gaps.best_action.iloc[0] == 2 and gaps.gap.iloc[0] == 0


def test_actor_information_boundary_and_no_future_leak(initial_state, traits):
    env, snap = setup(initial_state, traits)

    class Spy:
        def __init__(self):
            self.observations = []

        def act(self, observation):
            assert isinstance(observation, np.ndarray) and observation.shape == (21,)
            self.observations.append(observation.copy())
            return 2

    paths = hypothetical_paths(snap, 8, 24, 77)
    policy = Spy()
    rollout(snap, env.config, env.pd_model, policy, paths, 2)
    first = np.array(policy.observations)
    changed = tuple(ShockPath(p.customer_id, p.seed,
        (p.months[0], *hypothetical_paths(snap, 8, 23, 994)[i].months)) for i, p in enumerate(paths))
    policy2 = Spy()
    rollout(snap, env.config, env.pd_model, policy2, changed, 2)
    np.testing.assert_array_equal(first, np.array(policy2.observations))
    # Actual hidden traits affect transitions, but cannot alter the current actor projection.
    from credit_rl.envs.observation import build_observation
    alternate = replace(snap, traits=replace(traits, creditworthiness=-9))
    np.testing.assert_array_equal(build_observation(snap.state, snap.predicted_pd, snap.elapsed, env.config),
        build_observation(alternate.state, alternate.predicted_pd, alternate.elapsed, env.config))


def test_gaps_planning_discounting_and_weighted_dominance(initial_state, traits):
    env, snap = setup(initial_state, traits)
    samples = np.tile([1., 3., 3., 0., -1.], (8, 1))
    stats = action_statistics(samples, list(range(5)), env.config.environment.action_multipliers)
    assert stats['best_action'] == 2 and stats['gap'] == 0 and stats['tie_count'] == 2
    cube = np.zeros((8, 5, 3, len(METRICS)))
    cube[:, :, 0, 6] = [4., 2., 0., -1., -2.]
    cube[:, 4, 1:, 6] = 10.
    _, gaps, planning = value_tables(snap, env.config, cube, 0, 'Static', (1, 3, 24), (.5, 1.))
    row = planning[(planning.horizon == 3) & (planning.gamma == 1)].iloc[0]
    assert row.one_step_action == 0 and row.multi_step_action == 4
    assert row.opportunity == row.heldout_opportunity == 14
    assert row.immediate_sacrifice == 6
    assert (gaps[gaps.horizon == 24].effective_horizon == 3).all()
    from credit_rl.experiments.structural_diagnostics import describe
    states = pd.DataFrame([{**describe(snap, 'baseline', 'Static'), 'state_id': 0, 'weight': 7}])
    dominance, summary = dominance_tables(gaps, states)
    assert not summary.empty
    assert np.allclose(dominance.groupby(['continuation', 'horizon', 'gamma', 'stratification', 'bucket']).share.sum(), 1)


def test_decomposition_default_month(initial_state, traits):
    env, snap = setup(initial_state, traits)
    cfg = replace(env.config, default=replace(env.config.default, intercept=1000))
    outcome = CreditDGP(cfg).step(snap.state, 1., traits,
        hypothetical_paths(snap, 8, 24, 22)[0].months[0], snap.state.macro_state)
    parts, reward, net = decompose(snap.state, outcome, snap.predicted_pd, cfg)
    assert parts[0] == 0 and parts[2] == cfg.reward.loss_given_default*outcome.exposure
    assert reward == pytest.approx(net-parts[4]-parts[5])


def test_batch_pd_with_visited_history_matches_environment(initial_state, traits, no_default_config):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from credit_rl.risk.features import FEATURE_NAMES
    from credit_rl.risk.longitudinal import LongitudinalPDModel
    rng = np.random.default_rng(10)
    x = rng.normal(size=(60, len(FEATURE_NAMES)))
    estimator = make_pipeline(SimpleImputer(keep_empty_features=True), LogisticRegression()).fit(x, x[:, 0] > 0)
    risk = LongitudinalPDModel(estimator, {'horizon_months': 12})
    env = CreditLimitEnv(config=no_default_config, pd_model=risk, severe_delinquency_months=3)
    env.reset(seed=12, options=dict(initial_state=initial_state, traits=traits))
    for _ in range(4):
        env.step(2)
    snap = DecisionSnapshot.capture(env)
    original_history = deepcopy(snap.history)
    paths = hypothetical_paths(snap, 8, 24, 98)
    policy = ConstantAdjustment(env.config, .8)
    cube = rollout(snap, env.config, risk, policy, paths, 4)
    for action in range(5):
        branch = deepcopy(env)
        branch._shock_path = ShockPath(snap.state.customer_id, 98,
            (*env._shock_path.months[:4], *paths[0].months[:20]))
        for k in range(4):
            requested = action if k == 0 else policy.act(branch._get_observation())
            _, reward, *_ = branch.step(requested)
            assert cube[0, action, k, 6] == pytest.approx(reward, abs=1e-8)
    assert snap.history == original_history


def test_policy_decomposition_preserves_unvisited_bucket_contributions():
    from credit_rl.experiments.structural_diagnostics import complete_buckets
    frame = pd.DataFrame([
        dict(scenario='baseline', policy='Static', policy_seed=-1, initial_customers=2,
             stratification='utilization', bucket='low', revenue=10.),
        dict(scenario='baseline', policy='AlwaysDecrease20', policy_seed=-1, initial_customers=2,
             stratification='utilization', bucket='high', revenue=3.),
    ])
    complete = complete_buckets(frame, ['revenue'])
    assert len(complete) == 4
    assert complete.groupby('policy').revenue.sum().to_dict() == {'AlwaysDecrease20': 3., 'Static': 10.}
    pivot = complete.pivot(index='bucket', columns='policy', values='revenue')
    assert (pivot.AlwaysDecrease20-pivot.Static).sum() == -7.
