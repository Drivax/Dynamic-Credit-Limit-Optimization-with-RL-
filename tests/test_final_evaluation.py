"""Final treatment's numerical and information boundaries."""
import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces
import json
import inspect
from pathlib import Path

import pandas as pd
import yaml

from credit_rl.experiments.final_learning import BCRegularizedPPO
from credit_rl.experiments.initialization_learning import InitializedPPO


def test_only_auxiliary_loss_changes_the_pinned_ppo_update():
    from stable_baselines3 import PPO
    original = inspect.getsource(PPO.train)
    actual = inspect.getsource(BCRegularizedPPO._regularized_update)
    expected = original.replace('def train(self) -> None:', 'def _regularized_update(self) -> None:')
    start = actual.index('                # Only treatment change:')
    end = actual.index('\n\n', start)
    actual = actual[:start]+actual[end+1:]
    assert actual == expected


class PublicToy(gym.Env):
    observation_space = spaces.Box(0, 1, (21,), dtype=np.float32)
    action_space = spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.full(21, .3, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return np.full(21, .3, dtype=np.float32), float(action == 2), self.t == 4, False, {}


def model(cls=BCRegularizedPPO, **kwargs):
    return cls('MlpPolicy', PublicToy(), seed=101, n_steps=8, batch_size=4,
               n_epochs=2, device='cpu', **kwargs)


def test_bc_zero_is_exact_canonical_update():
    torch.set_num_threads(1)
    a = model(InitializedPPO)
    a.learn(16)
    b = model(beta=0.)
    b.learn(16)
    for key, value in a.policy.state_dict().items():
        assert torch.equal(value, b.policy.state_dict()[key]), key


def test_bc_loss_gradient_and_local_rng():
    x = np.full((8, 21), .3, dtype=np.float32)
    learner = model(beta=.05, bc_observations=x, bc_labels=np.full(8, 4))
    state = np.random.get_state()
    expected = -learner.policy.get_distribution(torch.tensor(x)).log_prob(torch.full((8,), 4)).mean()
    actual = learner.auxiliary_loss()
    torch.testing.assert_close(actual, expected)
    assert np.array_equal(state[1], np.random.get_state()[1])
    actual.backward()
    assert learner.policy.action_net.weight.grad.abs().sum() > 0
    assert learner.policy.value_net.weight.grad is None
    learner.learn(16)
    assert np.isfinite(learner.logger.name_to_value['train/bc_loss'])


def test_bc_schedule_and_no_extra_features():
    x = np.ones((4, 21), dtype=np.float32)
    learner = model(beta=.05, bc_schedule='decay', bc_observations=x, bc_labels=np.zeros(4))
    learner._total_timesteps = 100
    for step, expected in ((0, .05), (50, .025), (100, 0), (120, 0)):
        learner.num_timesteps = step
        assert learner.coefficient() == expected
    with pytest.raises(ValueError, match='21 public'):
        model(beta=.05, bc_observations=np.ones((4, 22)), bc_labels=np.zeros(4))
    with pytest.raises(ValueError, match='Invalid BC'):
        model(beta=.05, bc_observations=x, bc_labels=np.full(4, 5))
    with pytest.raises(ValueError, match='configuration'):
        model(beta=-1)


def test_world_pairing_and_moderate_shift_reproducibility():
    from credit_rl.experiments.final_panels import worlds
    from credit_rl.experiments.main_evaluation import settings_for
    config, settings, _ = settings_for('smoke', 'configs')
    protocol = yaml.safe_load(Path('configs/final_evaluation.yaml').read_text())
    a = worlds(config, settings, protocol, protocol['smoke'])
    b = worlds(config, settings, protocol, protocol['smoke'])
    assert list(a) == ['nominal', 'severe_stress', 'population_shift', 'behavioral_shift', 'risk_shift']
    for name in a:
        assert a[name] == b[name]
        for nominal, shifted in zip(a['nominal'][1], a[name][1]):
            assert nominal.customer_id == shifted.customer_id
            assert nominal.shock_path == shifted.shock_path
    assert a['behavioral_shift'][0].reward == config.reward
    assert a['risk_shift'][0].default.utilization == 1.45
    assert config.default.utilization == 1.2


def test_overlap_probabilities_and_worst_world_estimand():
    from credit_rl.experiments.final_ope import OverlapBehavior
    from credit_rl.experiments.final_analysis import worst_interval, behavior_interval
    target = type('Target', (), {'act': lambda self, obs: 3})()
    for overlap in (.95, .70, .30):
        behavior = OverlapBehavior(target, overlap, 101)
        probabilities = behavior.probabilities(np.zeros(21))
        assert probabilities[3] == overlap
        assert (probabilities > 0).all()
        assert np.isclose(probabilities.sum(), 1)
    with pytest.raises(ValueError):
        OverlapBehavior(target, 0, 1)
    left = np.ones((2, 3, 5))*4
    right = np.ones((2, 3, 5))*2
    result = worst_interval(left, right, 20)
    assert result['mean'] == result['lower'] == result['upper'] == 2
    with pytest.raises(ValueError):
        worst_interval(left, right[0], 20)
    counts = np.zeros((2, 3, 7))
    counts[:, :, 0] = 2
    counts[:, :, 6] = 2
    assert behavior_interval(counts, counts, 'diversity', 10)['mean'] == 0
    assert behavior_interval(counts, counts, 'teacher_regret', 10)['mean'] == 0


def test_portable_final_pipeline_and_frozen_regeneration(tmp_path, monkeypatch):
    """No scientific model archives are required by the CI smoke path."""
    from threadpoolctl import threadpool_limits
    from credit_rl.experiments import final_evaluation as experiment
    from credit_rl.experiments import final_report
    from credit_rl.experiments.main_evaluation import digest
    # Public, compact plot inputs exercise presentation without model archives.
    for relative in ('structural_diagnostics/action_maps.csv', 'policy_initialization/training_trajectory.csv'):
        source = Path('outputs/main')/relative
        target = tmp_path/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setattr(experiment, 'DROOT', tmp_path/'policy_initialization')
    monkeypatch.setattr(final_report, 'ROOT', tmp_path)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        output = experiment.run('smoke')
        assert json.loads((output/'inputs/fixture_complete.json').read_text())['scientific_evidence'] is False
        assert set(pd.read_csv(output/'main_table.csv').policy) == set(final_report.POLICIES)
        assert len(pd.read_csv(output/'ope_summary.csv')) == 12
        segments = pd.read_csv(output/'customer_heterogeneity.csv')
        assert set(segments.variable) == {'pd_bucket', 'utilization_bucket', 'income_bucket'}
        counts = segments.groupby(['budget', 'policy', 'policy_seed', 'world', 'variable']).customers.sum()
        assert (counts == 4).all()
        final_report.report('smoke')
        before = {str(f): digest(f) for f in (output/'figures').glob('*.png')}
        final_report.report('smoke')
        assert before == {str(f): digest(f) for f in (output/'figures').glob('*.png')}
        experiment.run('smoke', 'prepare')
        # Existing cached evaluations must verify successfully on resume.
        from credit_rl.experiments.final_panels import evaluate
        evaluate('smoke', diagnostics=False)
        from credit_rl.experiments.final_verify import manifest
        frozen = manifest('smoke')
        assert frozen['scientific_evidence'] is False
        assert frozen['headline_csv_sha256'] and frozen['figure_sha256']
        experiment.verify_frozen(output)
        table_path = output/'main_table.csv'
        original_table = table_path.read_bytes()
        table_path.write_text('corrupted')
        with pytest.raises(ValueError, match='Frozen final products'):
            experiment.verify_frozen(output)
        table_path.write_bytes(original_table)
    text = final_report.START+'\nold\n'+final_report.END
    assert 'replacement' in final_report.replace_block(text, 'replacement')
    with pytest.raises(ValueError):
        final_report.replace_block('missing markers', 'replacement')
    protected = json.loads((output/'protected_artifacts.json').read_text())
    sentinel = tmp_path/'protected.txt'
    sentinel.write_text('initial')
    protected[str(sentinel)] = digest(sentinel)
    (output/'protected_artifacts.json').write_text(json.dumps(protected))
    sentinel.write_text('changed')
    with pytest.raises(ValueError, match='Historical artifacts'):
        experiment.verify_protected(output)


def test_frozen_document_publishing_without_research_artifacts(tmp_path, monkeypatch):
    from credit_rl.experiments import final_report as report
    output = Path('outputs/main/final')
    block = (output/'measured_results.md').read_text(encoding='utf-8').strip()
    classification = (output/'final_classification.json').read_text(encoding='utf-8')
    # Only compact, publishable text is present; no simulator or model inputs.
    monkeypatch.chdir(tmp_path)
    output.mkdir(parents=True)
    (output/'measured_results.md').write_text(block, encoding='utf-8')
    (output/'final_classification.json').write_text(classification, encoding='utf-8')
    Path('docs').mkdir()
    template = report.START+'\nold\n'+report.END+'\n'+report.CONCLUSION_START+'\nold\n'+report.CONCLUSION_END
    for path in (Path('README.md'), Path('docs/technical_paper.md')):
        path.write_text(template, encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['final_report', '--documents-only'])
    report.main()
    readme = Path('README.md').read_text(encoding='utf-8')
    assert '| BCRegularizedPPO |' in readme and '| Static |' not in readme
    assert '| 262,144 |' not in readme and '| 32,768 |' in readme
    assert '| nominal | 0.2 [-33.4, 42.5]' in readme
    assert block in Path('docs/technical_paper.md').read_text(encoding='utf-8')
    monkeypatch.setattr('sys.argv', ['final_report', '--documents-only', '--profile', 'smoke'])
    with pytest.raises(SystemExit):
        report.main()
    Path('README.md').write_text(report.START+report.END, encoding='utf-8')
    with pytest.raises(ValueError, match='conclusion'):
        report.publish_documents(output, block)
