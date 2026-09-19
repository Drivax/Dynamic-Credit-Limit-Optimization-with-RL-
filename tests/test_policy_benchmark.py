from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.envs.observation import build_observation, OBSERVATION_NAMES
from credit_rl.evaluation.scenarios import make_scenarios, assert_disjoint
from credit_rl.evaluation.policy_engine import evaluate_policy, aggregate_episodes
from credit_rl.evaluation.policy_statistics import paired_comparisons, interval
from credit_rl.policies.decision import PDThreshold, MyopicEconomic, decode, transform_observation
from credit_rl.policies.registry import baseline_specs, ppo_spec
from credit_rl.policies.training import TrainingEnvironment, train_agent
from credit_rl.risk.longitudinal import LongitudinalPDModel
from credit_rl.risk.features import FEATURE_NAMES


@pytest.fixture
def settings():
    s = yaml.safe_load(Path('configs/policy_evaluation.yaml').read_text())
    s['population'].update(train=8, validation=4, test=4, oot=4)
    s['ppo'].update(n_steps=16, batch_size=16, n_epochs=1, validation_every=32, network=[8], total_timesteps=32)
    s['evaluation']['oracle_draws'] = 2
    s['evaluation']['bootstrap_repetitions'] = 20
    return s


@pytest.fixture
def risk():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, len(FEATURE_NAMES)))
    estimator = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression()).fit(x, np.arange(40)%2)
    return LongitudinalPDModel(estimator, {'horizon_months':12})


def test_projection_of_severe_delinquent_increases(initial_state, traits):
    state = replace(initial_state, months_delinquent=3)
    env = CreditLimitEnv(severe_delinquency_months=3)
    env.reset(seed=5, options=dict(initial_state=state, traits=traits))
    _, _, _, _, info = env.step(4)
    assert info['requested_action'] == pytest.approx(.2)
    assert info['effective_action'] == 0
    assert info['guardrail_blocked']
    assert env.state.credit_limit == state.credit_limit
    row = env.get_history().iloc[-1]
    assert row.action == 4 and row.effective_action == 0


def test_all_policy_actions_valid_static_and_thresholds(initial_state, traits, settings, risk):
    cfg = SimulationConfig()
    scenarios = make_scenarios(cfg, settings, 'test')
    env = CreditLimitEnv(pd_model=risk, severe_delinquency_months=3)
    obs, _ = env.reset(seed=1, options=scenarios[0].reset_options())
    for spec in baseline_specs(cfg, settings):
        policy = spec.factory(env, scenarios[0])
        for pd_value in [.01, .4, .95]:
            current = obs.copy(); current[10] = pd_value
            assert env.action_space.contains(policy.act(current))
            if spec.name == 'Static': assert policy.act(current) == 2
        if spec.information_set == 'SIMULATOR_ONLY_ORACLE':
            assert spec.name.startswith('SIMULATOR_ONLY_ORACLE')
            assert policy.information_set == spec.information_set
    threshold = PDThreshold(cfg, low=.2, high=.6)
    state_obs = build_observation(initial_state, .1, 0, cfg)
    assert threshold.act(state_obs) == 3
    state_obs[10] = .4; assert threshold.act(state_obs) == 2
    state_obs[10] = .7; assert threshold.act(state_obs) == 1


def test_myopic_public_only_and_argmax(initial_state):
    cfg = SimulationConfig()
    model = MyopicEconomic(cfg)
    obs = build_observation(initial_state, .3, 0, cfg)
    assert set(model.__dict__) == {'config','pd_horizon','severe_months','utilization_risk_slope','burden_risk_slope'}
    values = model.values(obs)
    assert values[model.act(obs)] == np.max(values)
    decoded = decode(obs, cfg)
    assert decoded['balance'] == pytest.approx(initial_state.balance, rel=1e-6)
    assert decoded['income'] == pytest.approx(initial_state.income, rel=1e-6)


def test_populations_prefix_and_train_guard(settings, risk):
    cfg = SimulationConfig()
    groups = [make_scenarios(cfg, settings, name) for name in ('train','validation','test','oot')]
    assert_disjoint(*groups)
    assert groups[0][:3] == make_scenarios(cfg, settings, 'train', customers=3)
    with pytest.raises(ValueError, match='overlap'):
        assert_disjoint(groups[0], groups[0])
    with pytest.raises(ValueError, match='RL_TRAIN'):
        TrainingEnvironment(groups[2], risk, cfg, settings)


def test_crn_reproducibility_and_denominators(settings, risk):
    cfg = SimulationConfig()
    scenarios = make_scenarios(cfg, settings, 'test')
    spec = baseline_specs(cfg, settings)[0]
    e, history, _ = evaluate_policy(spec, scenarios, cfg, risk, settings, 'baseline')
    again, replay, _ = evaluate_policy(spec, scenarios, cfg, risk, settings, 'baseline')
    pd.testing.assert_frame_equal(e, again)
    pd.testing.assert_frame_equal(history, replay)
    summary = aggregate_episodes(e)
    assert summary['episodes'] == len(scenarios)
    assert summary['default_rate'] == e.defaulted.sum()/len(scenarios)
    assert summary['credit_loss_rate'] == pytest.approx(e.credit_loss.sum()/e.initial_balance.sum())
    assert summary['no_change_fraction'] == 1
    np.testing.assert_allclose(e.cumulative_reward, e.net_economic_value-e.capital_charge-e.constraint_penalty)


def test_two_policies_use_identical_exogenous_shocks(settings, risk):
    cfg = SimulationConfig()
    scenario = make_scenarios(cfg, settings, 'test')[0]
    frames = []
    for action in (0,4):
        env = CreditLimitEnv(pd_model=risk, record_diagnostics=True)
        env.reset(seed=123, options=scenario.reset_options())
        for _ in range(3):
            _,_,term,trunc,_ = env.step(action)
            if term or trunc: break
        frames.append(env.get_diagnostics())
    n = min(map(len,frames))
    for column in [c for c in frames[0] if c.startswith('shock_')]:
        np.testing.assert_array_equal(frames[0][column].iloc[:n],frames[1][column].iloc[:n])


def test_actor_pd_ablation_does_not_change_world_or_reward(settings, risk):
    cfg = SimulationConfig()
    scenarios = make_scenarios(cfg, settings, 'test')
    static = baseline_specs(cfg, settings)[0]
    reference, history, _ = evaluate_policy(static, scenarios, cfg, risk, settings, 'baseline')
    for variant in (replace(static, without_pd=True), replace(static, pd_multiplier=.5)):
        episodes, altered, _ = evaluate_policy(variant, scenarios, cfg, risk, settings, 'baseline')
        pd.testing.assert_frame_equal(reference, episodes)
        pd.testing.assert_frame_equal(history.drop(columns='actor_pd'), altered.drop(columns='actor_pd'))
        assert not history.actor_pd.equals(altered.actor_pd)


def test_no_hidden_or_future_ppo_observation(settings, risk):
    cfg = SimulationConfig()
    s = make_scenarios(cfg, settings, 'train')[0]
    env = TrainingEnvironment([s], risk, cfg, settings)
    obs, _ = env.reset(seed=5)
    assert len(obs) == len(OBSERVATION_NAMES) == 21
    assert not {'p_default_true','creditworthiness','future_macro'}.intersection(OBSERVATION_NAMES)
    poisoned = replace(s, traits=replace(s.traits, creditworthiness=-100))
    env2 = TrainingEnvironment([poisoned], risk, cfg, settings)
    same, _ = env2.reset(seed=5)
    np.testing.assert_array_equal(obs,same)
    hidden = transform_observation(obs, without_pd=True)
    assert hidden[10] == 0
    np.testing.assert_array_equal(np.delete(obs,10),np.delete(hidden,10))


def test_cluster_paired_comparison_preserves_customer_pairing(settings):
    frame = pd.DataFrame(dict(policy=['Static']*3+['PPO']*6,policy_seed=[-1]*3+[1]*3+[2]*3,
        scenario=['baseline']*9,customer_id=['a','b','c']*3, information_set=['OBSERVABLE_ONLY']*9))
    for metric in ('cumulative_reward','net_economic_value','credit_loss','defaulted'):
        frame[metric]=[10,20,30,11,21,31,11,21,31]
    paired,_ = paired_comparisons(frame,settings)
    assert (paired.difference == 1).all()
    assert (paired.lower == 1).all() and (paired.upper == 1).all()


def test_short_end_to_end_saved_pd_and_ppo(tmp_path, settings, risk):
    cfg = SimulationConfig()
    risk.save(tmp_path/'pd.joblib')
    loaded = LongitudinalPDModel.load(tmp_path/'pd.joblib')
    train = make_scenarios(cfg, settings, 'train')
    val = make_scenarios(cfg, settings, 'validation',customers=2)
    with threadpool_limits(limits=1):
        meta = train_agent(cfg, loaded, settings, train, val, seed=3, destination=tmp_path/'ppo')
        policy = PPO.load(tmp_path/'ppo/selected.zip',device='cpu')
        scenarios = make_scenarios(cfg, settings,'test',customers=2)
        e, history, _ = evaluate_policy(ppo_spec(policy,3,'test_artifact'),scenarios,cfg,loaded,settings,'baseline')
    assert meta['actual_timesteps'] == 32
    assert len(e) == 2 and len(history) > 2
    assert not set(meta['training_customer_ids']) & set(e.customer_id)
    assert np.isfinite(e.cumulative_reward).all()
