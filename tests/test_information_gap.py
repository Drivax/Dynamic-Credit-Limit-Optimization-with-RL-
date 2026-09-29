"""Information boundaries and independent diagnostic replay."""
from argparse import Namespace
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.evaluation.structural import DecisionSnapshot, hypothetical_paths, rollout
from credit_rl.experiments.information_analysis import effective_action, weighted_summary
from credit_rl.experiments.information_gap import cohorts, fit_planners
from credit_rl.experiments.main_evaluation import settings_for
from credit_rl.policies.information import (
    public_features, privileged_features, ObservationPlanner, HistoryPlanner,
    FullStatePlanner, QEstimator, choose,
)


class FixedPD:
    def predict(self, features):
        return .25


class FixedQ:
    def predict(self, x):
        return np.tile([0, 1, 2, 3, 4], (len(x), 1))


def test_public_information_boundary_and_privileged_separation(initial_state, traits):
    env = CreditLimitEnv(pd_model=FixedPD())
    obs, _ = env.reset(seed=1, options=dict(initial_state=initial_state, traits=traits))
    snap = DecisionSnapshot.capture(env)
    altered = replace(snap, traits=replace(traits, creditworthiness=-4), macro_path=object())
    original, names = public_features([obs], [])
    np.testing.assert_array_equal(original, obs)
    assert len(names) == 21 and not any(n.startswith(('latent', 'future')) for n in names)
    assert privileged_features(obs, snap)[0][21] != privileged_features(obs, altered)[0][21]
    # The privileged feature builder must not inspect macro_path, even if it is unusable.
    assert len(privileged_features(obs, altered)[0]) == len(privileged_features(obs, snap)[0])
    assert not hasattr(ObservationPlanner(FixedQ(), env.config), 'env')
    assert not hasattr(HistoryPlanner(FixedQ(), env.config), 'env')
    assert FullStatePlanner.information_set == 'PRIVILEGED_FULL_STATE'
    assert isinstance(FullStatePlanner(FixedQ(), env.config, env).act(obs), int)
    with pytest.raises(ValueError):
        public_features([np.ones(22)], [])
    with pytest.raises(ValueError):
        public_features([obs], [], 'Full')
    with pytest.raises(ValueError):
        ObservationPlanner(FixedQ(), env.config, 'F5')
    with pytest.raises(ValueError):
        HistoryPlanner(FixedQ(), env.config, 'Full')


def test_history_prefix_truncation_and_feature_masks():
    obs = np.arange(210, dtype=float).reshape(10, 21)/210
    actions = [0, 1, 2, 3, 4, 0, 1, 2, 3]
    for kind, depth in [('F3', 3), ('F4', 6)]:
        a, names = public_features(obs, actions, kind)
        b, _ = public_features(obs[-depth-1:], actions[-depth:], kind)
        np.testing.assert_array_equal(a, b)
        assert len(a) == len(names)
    full, names = public_features(obs, actions, 'F5')
    assert full[names.index('initial_month_fraction')] == obs[0, 0]
    short, _ = public_features(obs[:1], [], 'F3')
    assert np.isnan(short).any()
    f1, names = public_features(obs[-1:], [], 'F1')
    assert len(f1) == 20 and 'predicted_pd' not in names
    _, names = public_features(obs[-1:], [], 'F2')
    assert not any(n in names for n in ('payment_ratio', 'behavioral_score_scaled', 'spend_bounded'))


def test_masks_requested_effective_and_history_isolation(initial_state, traits):
    config = SimulationConfig()
    env = CreditLimitEnv(config=config, pd_model=FixedPD(), severe_delinquency_months=3)
    state = replace(initial_state, credit_limit=config.environment.min_limit, months_delinquent=3)
    obs, _ = env.reset(seed=1, options=dict(initial_state=state, traits=traits))
    snap = DecisionSnapshot.capture(env)
    assert choose(np.arange(5), obs, config) == 2
    assert effective_action(snap, 0, config) == effective_action(snap, 2, config) == 0
    a, b = HistoryPlanner(FixedQ(), config), HistoryPlanner(FixedQ(), config)
    assert a.act(obs) == 2 and b.actions == []
    assert len(a.observations) == 1


def test_disjoint_customer_cohorts_and_deterministic_shocks():
    config, settings, _ = settings_for('smoke', 'configs')
    p = dict(population_seed=84932, train_customers=4, validation_customers=4, test_customers=4)
    a, b = cohorts(config, settings, p), cohorts(config, settings, p)
    train, validation, test = [set(s.customer_id for s in a['baseline', r]) for r in ('train', 'validation', 'test')]
    assert not train & validation and not test & (train | validation)
    assert a == b
    assert a['baseline', 'test'][0].shock_path == a['severe_stress', 'test'][0].shock_path


def test_target_selection_never_reads_test_values(tmp_path):
    rng = np.random.default_rng(1)
    features = rng.normal(size=(18, 3))
    data = dict(states=pd.DataFrame(dict(role=['train']*10+['validation']*4+['test']*4,
        weight=np.ones(18))), features={k: features for k in ('F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'Full')})
    q = rng.normal(size=(18, 4, 2, 5))
    p = dict(horizons=[6, 24], tree_leaves=[3], boosting_iterations=3, minimum_leaf=2, diagnostic_seed=7)
    with threadpool_limits(limits=1):
        first = fit_planners(data, q, p, tmp_path)
        q[-4:] = 1e20
        second = fit_planners(data, q, p, tmp_path)
        for k in first:
            assert first[k]['horizon'] == second[k]['horizon']
            np.testing.assert_array_equal(first[k]['model'].predict(features), second[k]['model'].predict(features))
        fitted = QEstimator(3, 3, 2, 1).fit(features, np.ones((18, 5)))
        assert np.allclose(fitted.predict(features), 1)


def test_indexed_continuation_keeps_crn_and_default_api(initial_state, traits):
    env = CreditLimitEnv(pd_model=FixedPD())
    env.reset(seed=1, options=dict(initial_state=initial_state, traits=traits))
    snap = DecisionSnapshot.capture(env)
    paths = hypothetical_paths(snap, 4, 24, 7)
    class Hold:
        def act(self, obs):
            return 2
    class IndexedHold:
        def act_indexed(self, obs, active, month, actions):
            assert len(obs) == len(active) and month > 0 and actions == 5
            return [2]*len(obs)
    a = rollout(snap, env.config, env.pd_model, Hold(), paths, 3)
    b = rollout(snap, env.config, env.pd_model, IndexedHold(), paths, 3)
    np.testing.assert_array_equal(a, b)


def test_regret_bootstrap_keeps_negative_values_and_customer_clusters():
    frame = pd.DataFrame(dict(customer_id=['a', 'a', 'b'], weight=[1, 1, 2], regret=[-2, -2, 4]))
    s = weighted_summary(frame, 'regret', dict(diagnostic_seed=1, bootstrap_repetitions=100))
    assert s['mean'] == 1 and s['lower'] < 0 and s['customers'] == 2


def test_information_pipeline_smoke_and_reproducibility(tmp_path):
    from credit_rl.experiments.main_evaluation import run as canonical_run
    from credit_rl.experiments.information_gap import run
    from credit_rl.experiments.information_ppo import run as audit
    canonical = tmp_path/'canonical'
    canonical_run(output=canonical)
    output = tmp_path/'phase_b'
    args = Namespace(profile='smoke', canonical_profile='smoke', canonical=canonical, output=output, stage='all')
    frozen = {str(p): p.read_bytes() for p in canonical.glob('models/**/*.zip')}
    run(args)
    audit(args)
    tables = {name: pd.read_csv(output/f'{name}.csv') for name in
        ('benchmark_values', 'information_gap', 'planner_regret', 'latent_predictability',
         'feature_importance', 'counterfactual_regret', 'critic_diagnostics', 'ppo_advantages')}
    assert tables['benchmark_values'].policy.nunique() == 10
    assert pd.read_csv(output/'replay_verification.csv').selected_weights_identical.all()
    assert pd.read_csv(output/'replay_verification.csv').final_weights_identical.all()
    draw = next((output/'draws').glob('*.npz'))
    before_draw = np.load(draw)['q'].copy()
    draw.unlink()
    run(args)
    np.testing.assert_array_equal(before_draw, np.load(draw)['q'])
    audit(args)
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, pd.read_csv(output/f'{name}.csv'), check_exact=True)
    assert all(p.read_bytes() == frozen[str(p)] for p in canonical.glob('models/**/*.zip'))
    assert json.loads((output/'protected_artifacts.json').read_text())['unchanged']
    mismatched = Namespace(**vars(args))
    mismatched.canonical_profile = 'standard'
    with pytest.raises(ValueError, match='identity'):
        audit(mismatched)
    from credit_rl.experiments.information_report import report
    report(output)
    assert (output/'figures/action_confusion.png').stat().st_size > 1000
    assert 'What is the dominant bottleneck?' in (output/'report.md').read_text(encoding='utf-8')
    args.output = canonical
    with pytest.raises(ValueError, match='protected'):
        run(args)


@pytest.mark.parametrize('supported,label', [
    ([], 'unresolved'), (['history'], 'memory'), (['ppo_gap'], 'optimization'),
    (['privileged'], 'information'), (['history', 'ppo_gap'], 'mixed'),
])
def test_decision_gate(supported, label):
    from credit_rl.experiments.information_report import decision_gate
    gaps = pd.DataFrame([dict(metric='discounted_reward', gap=g, lower=1 if g in supported else -1)
                         for g in ('history', 'ppo_gap', 'privileged')])
    assert label in decision_gate(gaps)[0]
