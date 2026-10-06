"""Phase D information boundaries, pairing, transfer and preservation estimands."""
import inspect
import ast
import textwrap
import json
import joblib
import shutil
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
import torch
import yaml

from credit_rl import CreditLimitEnv
from credit_rl.experiments.main_evaluation import settings_for
from credit_rl.experiments.policy_initialization import (
    populations, transfer_actor, paired_visit_interval, qualification, imitate,
)
from credit_rl.experiments.initialization_learning import InitializedPPO, EvaluationActor
from credit_rl.experiments.initialization_learning import run_pair
from credit_rl.experiments.ppo_imitation import fit_actor
from credit_rl.experiments.initialization_metrics import drift, survival, preservation
from credit_rl.experiments.information_ppo import distribution


def test_preregistered_pairing_splits_and_macro_crn():
    protocol = yaml.safe_load(Path('configs/policy_initialization.yaml').read_text())
    assert protocol['standard']['seeds'] == [101, 202, 303, 404, 505]
    assert protocol['standard']['budgets'] == [32768, 262144]
    config, settings, _ = settings_for('smoke', 'configs')
    groups = populations(config, settings, protocol['smoke'])
    ids = [{s.customer_id for s in groups['baseline', role]} for role in ('train', 'validation', 'test')]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for role in ('train', 'validation', 'test'):
        for a, b in zip(groups['baseline', role], groups['severe_stress', role]):
            assert a.customer_id == b.customer_id and a.shock_path == b.shock_path


def test_actor_transfer_exact_outputs_and_critic_separation(tmp_path):
    torch.set_num_threads(1)
    env = CreditLimitEnv()
    source = PPO('MlpPolicy', env, seed=101, n_steps=64, batch_size=64)
    target = PPO('MlpPolicy', env, seed=202, n_steps=64, batch_size=64)
    obs = np.random.default_rng(8).random((16, 21)).astype(np.float32)
    check = transfer_actor(source, target, obs)
    assert check['max_probability_error'] == check['max_logit_error'] == 0
    assert check['critic_unchanged']
    path = tmp_path/'actor.zip'
    source.save(path)
    reference = InitializedPPO('MlpPolicy', env, seed=303, n_steps=64, batch_size=64)
    expected_rng = (np.random.random(), random.random(), torch.rand(1))
    initialized = InitializedPPO('MlpPolicy', env, seed=303, n_steps=64, batch_size=64,
        actor_checkpoint=path, transfer_observations=obs)
    actual_rng = (np.random.random(), random.random(), torch.rand(1))
    assert expected_rng[:2] == actual_rng[:2]
    assert torch.equal(expected_rng[2], actual_rng[2])
    for key, value in reference.policy.state_dict().items():
        if key.startswith(('mlp_extractor.value_net.', 'value_net.')):
            assert torch.equal(value, initialized.policy.state_dict()[key])
    np.testing.assert_array_equal(distribution(source, obs)[0], distribution(initialized, obs)[0])
    env.close()


def test_teacher_has_no_training_objective_and_test_not_loaded():
    source = inspect.getsource(InitializedPPO.train)
    tree = ast.parse(textwrap.dedent(source))
    identifiers = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert 'super().train()' in source and not identifiers & {'teacher', 'loss', 'labels', 'q'}
    fit = inspect.getsource(imitate)
    assert 'dataset_test' not in fit
    assert "train['observations'], train['labels']" in fit


def test_drift_survival_and_preservation_definitions():
    initial = np.array([[.8, .2], [.3, .7]])
    current = np.array([[.4, .6], [.3, .7]])
    result = drift(current, initial, initial)
    np.testing.assert_allclose(result['kl_initial'], [.4*np.log(.5)+.6*np.log(3), 0])
    np.testing.assert_array_equal(result['action_flip'], [1, 0])
    np.testing.assert_allclose(drift(initial, initial, initial)['kl_previous'], 0)
    actual = survival(np.array([1, 1, 2]), np.array([0, 1, 1]), np.array([0, 1, 2]))
    np.testing.assert_allclose(actual[:2], [0, 1])
    assert np.isnan(actual[2])
    np.testing.assert_array_equal(preservation([4, 1], [2, 2]), [2, -1])


def test_bootstrap_is_paired_reproducible_and_qualification_requires_both_scenarios():
    frame = pd.DataFrame([dict(seed=s, customer_id=c, scenario=m,
        agreement_difference=.2, regret_difference=-2.)
        for s in range(5) for c in range(6) for m in ('baseline', 'severe_stress')])
    assert paired_visit_interval(frame, 'regret_difference', 30) == dict(mean=-2., low=-2., high=-2.)
    protocol = yaml.safe_load(Path('configs/policy_initialization.yaml').read_text())
    summary = pd.DataFrame([dict(seed=s, scenario=m, diversity=.2, effective_actions=3)
        for s in range(5) for m in ('baseline', 'severe_stress')])
    p = dict(seeds=list(range(5)), bootstrap_repetitions=30)
    assert qualification(frame, summary, protocol['qualification'], p)['qualified']
    frame.loc[frame.scenario == 'severe_stress', 'regret_difference'] = 1
    assert not qualification(frame, summary, protocol['qualification'], p)['qualified']


def test_stochastic_evaluation_uses_local_customer_month_crn():
    torch.set_num_threads(1)
    env = CreditLimitEnv()
    model = PPO('MlpPolicy', env, seed=101, n_steps=64, batch_size=64)
    one, two = EvaluationActor(model, 7788, 101), EvaluationActor(model, 7788, 101)
    np.testing.assert_array_equal(one.uniforms, two.uniforms)
    np.random.seed(45)
    expected = np.random.random()
    np.random.seed(45)
    obs = np.zeros(21, dtype=np.float32)
    assert [one.act(obs) for _ in range(24)] == [two.act(obs) for _ in range(24)]
    assert np.random.random() == pytest.approx(expected)
    env.close()


def test_smoke_imitation_paired_training_snapshot_hashes_and_reproduction(tmp_path, monkeypatch):
    from credit_rl.experiments.main_evaluation import digest
    from credit_rl.experiments.information_ppo import same_parameters
    config, settings, _ = settings_for('smoke', 'configs')
    settings['population'].update(train=4, validation=2)
    from credit_rl.experiments import initialization_learning
    monkeypatch.setattr(initialization_learning, 'ROOT', tmp_path/'inputs')
    reference = tmp_path/'inputs/ppo_diagnostics/runs/canonical/101/checkpoint_0.zip'
    reference.parent.mkdir(parents=True)
    env = CreditLimitEnv(config=config)
    PPO('MlpPolicy', env, seed=101, n_steps=64, batch_size=64).save(reference)
    env.close()
    protocol = yaml.safe_load(Path('configs/policy_initialization.yaml').read_text())
    protocol['smoke']['snapshots'] = [0, 64, 128]
    obs = np.random.default_rng(2).random((16, 21)).astype(np.float32)
    labels = np.arange(16) % 5
    for name in ('first', 'replay'):
        output = tmp_path/name
        model_dir = output/'imitation_models/Imitation_64x64/101'
        model_dir.mkdir(parents=True)
        fit_actor(obs, labels, np.ones(16), obs, labels, config, None, [64, 64], 101,
            protocol['imitation'], dict(imitation_epochs=2, imitation_check_every=1), model_dir)
        joblib.dump(dict(observations=obs), output/'dataset_validation.joblib')
        (output/'initialization_selection.json').write_text(json.dumps(dict(selected_architecture=[64, 64])))
        run_pair(config, settings, protocol, None, output, 'smoke', 128, 101, [64, 64])
        for arm in ('RandomInit', 'ImitationInit'):
            folder = output/'runs/128'/arm/'101'
            snapshots = pd.read_csv(folder/'snapshots.csv')
            assert snapshots.timesteps.tolist() == [0, 64, 128]
            assert snapshots.checkpoint_kind.tolist() == ['initial', 'post_update', 'post_update']
            assert all(digest(Path(row.checkpoint)) == row.sha256 for row in snapshots.itertuples())
            completed = json.loads((folder/'completed.json').read_text())
            assert completed['teacher_in_objective'] is False
            assert not pd.read_csv(folder/'rollouts.csv.gz').empty
    for arm in ('RandomInit', 'ImitationInit'):
        assert same_parameters(PPO.load(tmp_path/'first/runs/128'/arm/'101/final.zip'),
                               PPO.load(tmp_path/'replay/runs/128'/arm/'101/final.zip'))


def test_saved_smoke_analysis_seed_schema_csv_figures_and_protection(tmp_path, monkeypatch):
    """Build isolated software inputs; never depend on ignored research artifacts."""
    from credit_rl.experiments import initialization_analysis, initialization_learning
    from credit_rl.experiments import policy_initialization as experiment
    from credit_rl.experiments.final_smoke_inputs import build_inputs
    from credit_rl.experiments.initialization_report import registry, figures, scientific_report
    from credit_rl.experiments.main_evaluation import digest
    from credit_rl.risk.longitudinal import LongitudinalPDModel
    from threadpoolctl import threadpool_limits
    root = tmp_path/'inputs'
    for module in (experiment, initialization_analysis, initialization_learning):
        monkeypatch.setattr(module, 'ROOT', root)
    torch.set_num_threads(1)
    config, settings, _ = settings_for('smoke', 'configs')
    settings['population'].update(train=4, validation=2)
    protocol = yaml.safe_load(Path('configs/policy_initialization.yaml').read_text())
    protocol['smoke'].update(budgets=[128], snapshots=[0, 64, 128],
        train_customers=3, validation_customers=3, test_customers=3,
        imitation_epochs=2, imitation_check_every=1, mc_draws=2,
        mc_states_per_scenario=1, bootstrap_repetitions=20)
    protocol['imitation']['architectures'] = [[64, 64]]
    output = tmp_path/'diagnostic'
    output.mkdir()
    with threadpool_limits(limits=1):
        build_inputs(root)
        assert json.loads((root/'fixture_complete.json').read_text())['scientific_evidence'] is False
        risk = LongitudinalPDModel.load(root/'standard/models/pd/logistic_calibrated.joblib')
        shutil.copyfile(root/'standard/models/ppo_101/selected.zip',
                        root/'ppo_diagnostics/runs/canonical/101/selected.zip')
        experiment.build_dataset(config, settings, protocol['smoke'], risk, output)
        experiment.imitate(config, settings, protocol, risk, output, 'smoke')
        experiment.write_json(output/'preregistration.json', dict(identity=dict(profile='smoke',
            protocol=protocol, teacher_sha256=digest(root/'information_gap/planners.joblib'),
            ppo_config=settings['ppo'])))
        protected = experiment.protected_files()
        experiment.write_json(output/'protected_artifacts.json', protected)
        run_pair(config, settings, protocol, risk, output, 'smoke', 128, 101, [64, 64],
                 smoke_unqualified=True)
        initialization_analysis.analyze_budget(config, settings, protocol, risk, output, 'smoke', 128)
    results = pd.read_csv(output/'statistical_comparisons.csv')
    assert not results.empty and set(results.scenario) == {'baseline', 'severe_stress'}
    assert pd.read_csv(output/'preservation_pooled.csv')['mean'].notna().all()
    registry(output)
    figures(output)
    hashes = {p.name: digest(p) for p in (output/'figures').glob('*.png')}
    figures(output)
    assert hashes == {p.name: digest(p) for p in (output/'figures').glob('*.png')}
    scientific_report(output)
    assert '## 17. Decision gate' in (output/'report.md').read_text(encoding='utf-8')
    assert experiment.verify_protection(output) == len(protected) > 0
    sentinel = root/'ppo_diagnostics/runs/canonical/101/checkpoint_0.zip'
    sentinel.write_bytes(b'corrupted test fixture')
    with pytest.raises(AssertionError, match='Protected artifacts changed'):
        experiment.verify_protection(output)
