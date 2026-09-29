"""Preregistered Phase C experiments; local, resumable and outcome-independent."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import stable_baselines3
from threadpoolctl import threadpool_limits
import torch
import yaml

from credit_rl.experiments.main_evaluation import settings_for, digest
from credit_rl.experiments.information_gap import protected_hashes, cohorts
from credit_rl.experiments.information_ppo import same_parameters, record_gradients
from credit_rl.evaluation.scenarios import make_scenarios
from credit_rl.policies.training import train_agent
from credit_rl.risk.longitudinal import LongitudinalPDModel


def protection(canonical, phase_b):
    hashes = protected_hashes(canonical)
    paths = list(phase_b.rglob('*')) + [Path('docs/information_planning_gap.md'), Path('docs/information_gap_protocol.md')]
    paths += list(Path('src/credit_rl/simulation').glob('*.py'))
    paths += [Path('src/credit_rl/reward.py'), Path('configs/simulation.yaml')]
    hashes.update({str(p): digest(p) for p in paths if p.is_file()})
    return hashes


def load_protocol(profile):
    matrix = yaml.safe_load(Path('configs/ppo_diagnostics.yaml').read_text())
    return matrix, matrix[profile]


def diagnostic_provenance(args, destination, specification, checkpoint=None):
    """Trace frozen diagnostics independently from the training-run provenance."""
    record = dict(specification=specification, profile=args.profile,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        pd_sha256=digest(args.canonical/'models/pd/logistic_calibrated.joblib'),
        reward_sha256=digest(Path('src/credit_rl/reward.py')),
        protocol_sha256=digest(args.output/'preregistration.json'),
        checkpoint_sha256=digest(checkpoint) if checkpoint else None,
        source_hashes={str(p): digest(p) for p in Path('src/credit_rl/experiments').glob('ppo_*.py')},
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      torch=torch.__version__, sb3=stable_baselines3.__version__))
    destination.write_text(json.dumps(record, indent=2))


def experiment_rows(profile, canonical_settings):
    matrix, p = load_protocol(profile)
    rows = []
    for experiment in matrix['experiments']:
        for index, value in enumerate(experiment['values']):
            name = experiment['id'] if len(experiment['values']) == 1 else f'{experiment["id"]}_{index}'
            changes = {} if experiment['parameter'] == 'none' else {experiment['parameter']: value}
            if profile == 'smoke':
                if experiment['parameter'] == 'n_steps':
                    changes['n_steps'] = 64 if index == 0 else 128 if index == 1 else 256
                if experiment['parameter'] == 'total_timesteps':
                    changes['total_timesteps'] = [256, 512, 1024][index]
            budget = changes.get('total_timesteps', canonical_settings['ppo']['total_timesteps'])
            budget = max(budget, changes.get('n_steps', canonical_settings['ppo']['n_steps']))
            seeds = p['important_seeds'] if experiment.get('important') else p['seeds']
            for seed in seeds:
                rows.append(dict(experiment_id=name, family=experiment['family'], hypothesis=experiment['hypothesis'],
                    parameter=experiment['parameter'], canonical_value=json.dumps(canonical_settings['ppo'].get(experiment['parameter'],
                        [64, 64] if 'network' in experiment['parameter'] else 0)),
                    intervention_value=json.dumps(value), changes=changes, seed=seed, budget=budget, status='planned'))
    return rows


def population(config, settings, protocol):
    p = dict(population_seed=protocol['population_seed'], train_customers=4,
             validation_customers=4, test_customers=protocol['test_customers'])
    return {macro: group for (macro, role), group in cohorts(config, settings, p).items() if role == 'test'}


def prepare(args):
    canonical, phase_b, output = args.canonical, args.phase_b, args.output
    for protected in (canonical, phase_b, Path('outputs/main/structural_diagnostics')):
        if output.resolve() == protected.resolve() or protected.resolve() in output.resolve().parents:
            raise ValueError('Phase C output overlaps protected artifacts')
    output.mkdir(parents=True, exist_ok=True)
    config, settings, _ = settings_for(args.profile, 'configs')
    matrix, p = load_protocol(args.profile)
    identity = dict(profile=args.profile, protocol=matrix, canonical=str(canonical.resolve()), phase_b=str(phase_b.resolve()),
                    pd_sha256=digest(canonical/'models/pd/logistic_calibrated.joblib'))
    manifest = output/'preregistration.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Preregistered experiment changed; do not reuse this output')
    manifest.write_text(json.dumps(identity, indent=2))
    protected = output/'protected_artifacts.json'
    if not protected.exists():
        protected.write_text(json.dumps(protection(canonical, phase_b), indent=2))
    risk = LongitudinalPDModel.load(canonical/'models/pd/logistic_calibrated.joblib')
    initial = PPO.load(canonical/'models/ppo_101/checkpoint_0.zip', device='cpu')
    audit = dict(settings['ppo'], activation=type(initial.policy.activation_fn()).__name__,
        optimizer=type(initial.policy.optimizer).__name__, optimizer_defaults=str(initial.policy.optimizer.defaults),
        initialization='orthogonal hidden gain sqrt(2), policy head .01, value head 1; zero bias',
        architecture=str(initial.policy.net_arch), learnable_shared_features=False,
        action_mask='none in PPO; environment applies floor/cap/delinquency guard',
        evaluation='deterministic argmax; unchanged 21D observation',
        finite_horizon='TrainingEnvironment converts term or trunc to terminal; no TimeLimit bootstrap',
        bootstrap='V(next observation) at rollout edge unless episode terminal',
        advantage_normalization='per optimization minibatch, sample std plus 1e-8',
        selection='maximum full baseline validation discounted reward; earliest tie; rollout-boundary selection before pending update plus after final update',
        buffer='single environment, fixed n_steps, GAE computed backward before optimization')
    pd.DataFrame([dict(parameter=k, value=json.dumps(v, default=str)) for k, v in audit.items()]).to_csv(output/'canonical_ppo_configuration.csv', index=False)
    cache = output/'panels.joblib'
    if not cache.exists():
        b = joblib.load(phase_b/'dataset.joblib')
        validation_ids = np.flatnonzero(b['states'].role.eq('validation'))
        joblib.dump(dict(test=population(config, settings, p),
            validation_observations=b['features']['F0'][validation_ids].astype(np.float32),
            validation_states=b['states'].iloc[validation_ids].reset_index(drop=True),
            validation_snapshots=[b['snapshots'][i] for i in validation_ids],
            validation_source_ids=validation_ids), cache)
    return config, settings, matrix, p, risk


def run_job(job):
    args, row = job
    from credit_rl.experiments.ppo_learning import LearningPPO, LearningRecorder
    config, settings, _ = settings_for(args.profile, 'configs')
    _, protocol = load_protocol(args.profile)
    output = args.output/'runs'/row['experiment_id']/str(row['seed'])
    output.mkdir(parents=True, exist_ok=True)
    completed = output/'completed.json'
    if completed.exists():
        return json.loads(completed.read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    panels = joblib.load(args.output/'panels.joblib')
    risk = LongitudinalPDModel.load(args.canonical/'models/pd/logistic_calibrated.joblib')
    with threadpool_limits(limits=1):
        train = make_scenarios(config, settings, 'train', 'markov')
        validation = make_scenarios(config, settings, 'validation')
        settings = deepcopy(settings)
        changes = row['changes'].copy()
        settings['ppo'].update({k: v for k, v in changes.items() if k not in ('critic_extra_epochs', 'entropy_schedule')})
        recorder = LearningRecorder(row, protocol, config, settings, risk, panels, output)
        reference = args.output/'runs'/row.get('reference_experiment', 'canonical')/str(row['seed'])/'checkpoint_0.zip'
        factory = partial(LearningPPO, intervention=changes, reference_initial=reference if reference.exists() else None)
        saved_intervention = output/'intervention.json'
        if saved_intervention.exists() and json.loads(saved_intervention.read_text()) == row:
            metadata = json.loads((output/'metadata.json').read_text())
        else:
            with record_gradients(recorder):
                metadata = train_agent(config, risk, settings, train, validation, seed=row['seed'], destination=output,
                    total_timesteps=row['budget'], diagnostic_callback=recorder, model_class=factory)
            recorder.flush()
    checks = {}
    if row['experiment_id'] == 'canonical' and row['seed'] in (101, 202, 303):
        for name in ('selected', 'final'):
            original = args.canonical/f'models/ppo_{row["seed"]}/{name}.zip'
            if original.exists():
                checks[name+'_canonical_identical'] = same_parameters(PPO.load(original), PPO.load(output/name))
                if not checks[name+'_canonical_identical']:
                    raise AssertionError(f'Canonical replay differs: {row["seed"]} {name}')
    result = dict(row, status='complete', **checks,
        selected_timesteps=metadata['selected_timesteps'], validation_score=metadata['validation_score'],
        checkpoint_sha256=digest(output/'selected.zip'))
    provenance = dict(result, expanded_settings=settings, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      torch=torch.__version__, sb3=stable_baselines3.__version__),
        timestamp=datetime.now(timezone.utc).isoformat(), scenario='training Markov / selection baseline',
        source_hashes={str(p): digest(p) for p in Path('src/credit_rl').rglob('*.py')},
        pd_sha256=digest(args.canonical/'models/pd/logistic_calibrated.joblib'),
        reward_sha256=digest(Path('src/credit_rl/reward.py')))
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    completed.write_text(json.dumps(result, indent=2))
    print(f'COMPLETE {row["experiment_id"]} seed={row["seed"]}', flush=True)
    return result


def registry(args, rows):
    records = []
    for row in rows:
        path = args.output/'runs'/row['experiment_id']/str(row['seed'])/'completed.json'
        records.append(json.loads(path.read_text()) if path.exists() else row)
    frame = pd.DataFrame(records)
    frame['changes'] = frame.changes.map(lambda x: json.dumps(x, sort_keys=True))
    frame.to_csv(args.output/'experiment_registry.csv', index=False)


def run(args):
    _, settings, _, protocol, _ = prepare(args)
    rows = experiment_rows(args.profile, settings)
    registry(args, rows)
    if args.family == 'audit':
        return
    selected = [r for r in rows if args.family == 'all' or r['family'] == args.family]
    if not selected:
        raise ValueError('Unknown experiment family')
    families = list(dict.fromkeys(r['family'] for r in selected))
    for family in families:
        with ProcessPoolExecutor(max_workers=args.workers or protocol['workers']) as pool:
            futures = [pool.submit(run_job, (args, row)) for row in selected if row['family'] == family]
            for future in as_completed(futures):
                future.result()
                registry(args, rows)
    before = json.loads((args.output/'protected_artifacts.json').read_text())
    if before != protection(args.canonical, args.phase_b):
        raise AssertionError('Protected Phase A/B or simulator artifact changed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    p.add_argument('--canonical', type=Path, default=Path('outputs/main/standard'))
    p.add_argument('--phase-b', type=Path, default=Path('outputs/main/information_gap'))
    p.add_argument('--output', type=Path, default=Path('outputs/main/ppo_diagnostics'))
    p.add_argument('--family', default='audit')
    p.add_argument('--workers', type=int)
    run(p.parse_args())


if __name__ == '__main__':
    main()
