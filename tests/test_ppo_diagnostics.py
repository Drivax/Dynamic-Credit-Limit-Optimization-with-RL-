"""Phase C controls: exact replay, single-factor isolation and diagnostic information boundaries."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
import torch

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.evaluation.structural import DecisionSnapshot
from credit_rl.experiments.main_evaluation import settings_for
from credit_rl.experiments.ppo_diagnostics import experiment_rows, population, run_job
from credit_rl.experiments.ppo_learning import LearningPPO, state_dependence
from credit_rl.experiments.ppo_measurements import collapse_times, gae_from_arrays, observable_regret, mc_probe


class FixedPD:
    def predict(self, features):
        return .25


def test_preregistered_one_factor_matrix_and_paired_customers():
    config, settings, _ = settings_for('standard', 'configs')
    before = deepcopy(settings)
    rows = experiment_rows('standard', settings)
    assert all(len(r['changes']) <= 1 for r in rows)
    assert {r['seed'] for r in rows if r['family'] == 'canonical'} == {101, 202, 303, 404, 505}
    assert {r['changes']['gae_lambda'] for r in rows if r['family'] == 'advantage'} == {0, .5, .8, 1}
    assert settings == before
    scenarios = population(config, settings, dict(population_seed=6543, test_customers=4))
    for a, b in zip(scenarios['baseline'], scenarios['severe_stress']):
        assert a.customer_id == b.customer_id and a.shock_path == b.shock_path


def test_collapse_first_passage_censoring_and_recrossings():
    frame = pd.DataFrame(dict(experiment_id='x', seed=1, scenario='baseline', timesteps=[0, 512, 1024, 1536],
        deterministic_contraction=[.5, .96, .8, .96], stochastic_contraction=[.2, .3, .4, .5]))
    result = collapse_times(frame)
    deterministic = result[(result.kind == 'deterministic_contraction') & (result.threshold == .95)].iloc[0]
    assert deterministic.first_crossing == 512 and deterministic.recrossings == 1
    assert result[result.kind == 'stochastic_contraction'].censored.all()


def test_state_dependence_separates_internal_entropy_from_action_dependence():
    obs = np.zeros((100, 21))
    obs[50:, 10] = .9
    independent = state_dependence(np.zeros(100, dtype=int), obs)
    dependent = state_dependence(np.r_[np.zeros(50), np.ones(50)].astype(int), obs)
    assert independent['diversity'] == 0 and independent['mi_pd'] == 0
    assert dependent['diversity'] == .5 and dependent['mi_pd'] == pytest.approx(np.log(2))


def test_gae_terminal_telescoping_and_td_zero_lambda():
    rewards = np.arange(30).reshape(2, 5, 3).astype(float)
    values = np.ones_like(rewards)*7
    mc = gae_from_arrays(rewards, values, .98, 1.)
    np.testing.assert_allclose(mc, (rewards*.98**np.arange(3)).sum(2)-7)
    td = gae_from_arrays(rewards, values, .98, 0.)
    np.testing.assert_allclose(td, rewards[:, :, 0]+.98*7-7)


def test_aliases_do_not_create_economic_regret(initial_state, traits):
    from dataclasses import replace
    env = CreditLimitEnv(pd_model=FixedPD(), severe_delinquency_months=3)
    obs, _ = env.reset(options=dict(initial_state=replace(initial_state, credit_limit=500, months_delinquent=3), traits=traits))
    q = np.array([[100, 200, 10, 300, 400]])
    assert observable_regret(q, [0], [obs], env.config)[0] == 0


def make_model(architecture=None, intervention=None, reference=None):
    env = CreditLimitEnv(pd_model=FixedPD())
    return LearningPPO('MlpPolicy', env, seed=12, n_steps=8, batch_size=8, n_epochs=1,
        policy_kwargs=dict(net_arch=architecture or dict(pi=[64, 64], vf=[64, 64])),
        intervention=intervention, reference_initial=reference)


def test_critic_only_updates_leave_actor_unchanged_and_capacity_preserves_actor(tmp_path):
    torch.set_num_threads(1)
    model = make_model(intervention={'critic_extra_epochs': 5})
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    model.rollout_buffer.observations = np.zeros((8, 1, 21), dtype=np.float32)
    model.rollout_buffer.returns = np.ones((8, 1), dtype=np.float32)
    model.extra_critic_updates()
    for k, value in before.items():
        if k.startswith(('mlp_extractor.policy_net', 'action_net')):
            assert torch.equal(value, model.policy.state_dict()[k])
    assert not torch.equal(before['value_net.bias'], model.policy.value_net.bias)
    reference = tmp_path/'initial.zip'
    canonical = make_model()
    canonical.save(reference)
    larger = make_model(dict(pi=[64, 64], vf=[128, 128]), {'critic_network': [128, 128]}, reference)
    for k, value in canonical.policy.state_dict().items():
        if k.startswith(('mlp_extractor.policy_net', 'action_net')):
            assert torch.equal(value, larger.policy.state_dict()[k])


def test_mc_probe_does_not_consume_training_rng_and_replays(initial_state, traits):
    env = CreditLimitEnv(pd_model=FixedPD())
    env.reset(seed=19, options=dict(initial_state=initial_state, traits=traits))
    snapshot = DecisionSnapshot.capture(env)
    model = make_model()
    numpy_state, torch_state, python_state = np.random.get_state(), torch.get_rng_state(), random.getstate()
    with threadpool_limits(limits=1):
        first = mc_probe(snapshot, model, env.config, env.pd_model, 4, 134, .98, .95, .001)
        second = mc_probe(snapshot, model, env.config, env.pd_model, 4, 134, .98, .95, .001)
    for k in first:
        np.testing.assert_array_equal(first[k], second[k])
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state) and random.getstate() == python_state


def test_imitation_rejects_nonpublic_inputs(tmp_path):
    from credit_rl.experiments.ppo_imitation import fit_actor
    with pytest.raises(ValueError, match='21 observable'):
        fit_actor(np.ones((4, 22)), np.ones(4), np.ones(4), np.ones((4, 21)), np.ones(4),
            SimulationConfig(), FixedPD(), [64, 64], 1, {}, {}, tmp_path)


def test_fine_canonical_smoke_exact_replay(tmp_path):
    # Reuse the canonical smoke integration artifact when present; otherwise build fresh.
    from credit_rl.experiments.main_evaluation import run as main_run
    from credit_rl.experiments.information_gap import run as information_run
    from credit_rl.experiments.ppo_diagnostics import prepare, registry
    from credit_rl.experiments.ppo_analysis import measurement_job, aggregate, evaluate_references
    canonical = tmp_path/'canonical'
    main_run(output=canonical)
    phase_b = tmp_path/'phase_b'
    information_run(Namespace(profile='smoke', canonical_profile='smoke', canonical=canonical, output=phase_b, stage='all'))
    args = Namespace(profile='smoke', canonical=canonical, phase_b=phase_b, output=tmp_path/'phase_c', family='audit', workers=1)
    config, settings, _, protocol, risk = prepare(args)
    hashes = {str(p): p.read_bytes() for p in canonical.glob('models/**/*.zip')}
    row = next(r for r in experiment_rows('smoke', settings) if r['family'] == 'canonical')
    first = run_job((args, row))
    registry(args, [row])
    assert first['selected_canonical_identical'] and first['final_canonical_identical']
    assert run_job((args, row)) == first
    measurement_job((args, row))
    aggregate(args, protocol)
    assert (args.output/'gae_alignment.csv').is_file()
    assert all(Path(p).read_bytes() == b for p, b in hashes.items())
    records = pd.read_csv(args.output/'runs/canonical/101/snapshots.csv')
    from credit_rl.experiments.main_evaluation import digest
    assert all(digest(p) == h for p, h in zip(records.checkpoint, records.sha256))
    snapshot = PPO.load(records.checkpoint.iloc[-1])
    assert snapshot.num_timesteps == int(records.timesteps.iloc[-1])
    changes = json.loads((args.output/'runs/canonical/101/intervention.json').read_text())['changes']
    assert changes == {}
    from credit_rl.experiments.ppo_imitation import run as imitate
    from credit_rl.experiments.ppo_counterfactual import run as counterfactual
    imitate(args)
    counterfactual(args)
    evaluate_references(args, config, settings, risk)
    from credit_rl.experiments.ppo_report import run as report
    args.report = args.output/'report.md'
    report(args)
    assert (args.output/'imitation.csv').is_file()
    assert (args.output/'mc_value_benchmark.csv').is_file()
    assert len(list((args.output/'figures').glob('*.png'))) == 13
    assert '## 15. Decision gate' in args.report.read_text(encoding='utf-8')


def test_confirmation_gate_uses_mechanisms_not_test_scores(tmp_path, monkeypatch):
    from credit_rl.experiments import ppo_confirmation as confirmation
    def mechanism(folder):
        name = folder.parent.name
        return dict(collapse=4096 if name == 'lambda_3' else 512,
                    regret=0., rank=.5)
    monkeypatch.setattr(confirmation, 'mechanism', mechanism)
    rows = [dict(experiment_id=name, family='advantage', seed=seed, changes={'gae_lambda': value})
            for name, value in [('lambda_0', 0.), ('lambda_3', 1.)] for seed in (101, 202, 303)]
    result = confirmation.select_factors(rows, tmp_path, dict(seeds=[101, 202, 303]), ['advantage'])
    assert [r['experiment_id'] for r in result] == ['lambda_3']
    # No episode/test values exist in this fixture; selection cannot read any.
    assert not (tmp_path/'episode_metrics.csv').exists()


def test_alignment_keeps_preupdate_selection_separate_from_postupdate_snapshot():
    from credit_rl.experiments.ppo_measurements import summarize_alignment
    rows = []
    for kind, prediction in [('selected', 1.), ('time_8192', 3.)]:
        for action, advantage in [(0, 1.), (1, -1.)]:
            rows.append(dict(experiment_id='canonical', seed=101, timesteps=8192, scenario='baseline',
                checkpoint_kind=kind, state_id=0, action=action, gae_mean=advantage,
                mc_advantage=advantage, value_prediction=prediction, value_mc=0.))
    result = summarize_alignment(pd.DataFrame(rows)).set_index('checkpoint_kind')
    assert len(result) == 2
    assert result.loc['selected', 'critic_rmse'] == 1.
    assert result.loc['time_8192', 'critic_rmse'] == 3.
