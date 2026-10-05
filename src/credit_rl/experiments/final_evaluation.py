"""Final frozen benchmark, reusing historical models without rewriting them."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import shutil
import subprocess

import joblib
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import yaml

from credit_rl.evaluation.scenarios import make_scenarios
from credit_rl.experiments.information_ppo import record_gradients
from credit_rl.experiments.initialization_learning import InitializationRecorder
from credit_rl.experiments.main_evaluation import settings_for, digest
from credit_rl.experiments.policy_initialization import ROOT, write_json
from credit_rl.policies.training import train_agent
from credit_rl.risk.longitudinal import LongitudinalPDModel
from .final_learning import BCRegularizedPPO

DROOT = ROOT/'policy_initialization'
CONFIG = Path('configs/final_evaluation.yaml')


def output_for(profile):
    return ROOT/('final' if profile == 'standard' else 'final_smoke')


def input_paths(profile):
    root = ROOT
    if profile == 'smoke' and not (DROOT/'verification.json').exists():
        from .final_smoke_inputs import build_inputs
        root = output_for(profile)/'inputs'
        build_inputs(root)
    return root, root/'policy_initialization'


def context(profile):
    root, _ = input_paths(profile)
    protocol = yaml.safe_load(CONFIG.read_text())
    config, settings, _ = settings_for(profile, 'configs')
    p = protocol[profile]
    p = {**p, 'seeds': protocol['seeds'] if profile == 'standard' else [101]}
    risk = LongitudinalPDModel.load(root/'standard/models/pd/logistic_calibrated.joblib')
    return config, settings, protocol, p, risk


def verify_protected(output):
    hashes = json.loads((output/'protected_artifacts.json').read_text())
    changed = [name for name, sha in hashes.items() if not Path(name).is_file() or digest(name) != sha]
    if changed:
        raise ValueError(f'Historical artifacts changed: {changed[:5]}')
    return len(hashes)


def verify_frozen(output):
    path = output/'frozen_products.json'
    if path.exists():
        hashes = json.loads(path.read_text())
        changed = [name for name, sha in hashes.items() if not Path(name).is_file() or digest(name) != sha]
        if changed:
            raise ValueError(f'Frozen final products changed: {changed[:5]}')


def prepare(profile):
    config, settings, protocol, p, risk = context(profile)
    output = output_for(profile)
    output.mkdir(parents=True, exist_ok=True)
    root, droot = input_paths(profile)
    identity = dict(profile=profile, protocol=protocol, config=asdict(config), ppo=settings['ppo'],
                    teacher=digest(root/'information_gap/planners.joblib'),
                    pd=digest(root/'standard/models/pd/logistic_calibrated.joblib'))
    # JSON canonicalization makes tuple/list representations consistent on reload.
    identity = json.loads(json.dumps(identity))
    frozen = output/'preregistration.json'
    if frozen.exists():
        if json.loads(frozen.read_text())['identity'] != identity:
            raise ValueError('Final protocol changed after registration')
        verify_protected(output)
        verify_frozen(output)
    else:
        audit = json.loads((droot/'verification.json').read_text()) if root == ROOT else {'status': 'software_fixture_only'}
        if root == ROOT and audit['status'] != 'passed':
            raise ValueError('Initialization verification must pass')
        folders = ['standard', 'structural_diagnostics', 'information_gap', 'ppo_diagnostics',
                   'policy_initialization', 'smoke', 'structural_smoke', 'information_gap_smoke',
                   'ppo_diagnostics_smoke', 'policy_initialization_smoke', 'information_canonical_smoke']
        files = [f for folder in folders for f in (ROOT/folder).rglob('*') if f.is_file()]
        files += [f for directory in ('src/credit_rl/envs', 'src/credit_rl/simulation', 'src/credit_rl/risk',
                                      'src/credit_rl/policies') for f in Path(directory).glob('*.py')]
        files += [f for f in Path('configs').glob('*.yaml') if f != CONFIG]
        write_json(output/'protected_artifacts.json', {str(f): digest(f) for f in sorted(set(files))})
        archive = output/'historical_documents'
        archive.mkdir(exist_ok=True)
        for source in (Path('README.md'), Path('docs/technical_paper.md')):
            shutil.copyfile(source, archive/source.name)
        write_json(output/'prior_audit.json', dict(verification=audit,
            validation=json.loads((droot/'validation.json').read_text()) if root == ROOT else {'scientific_evidence': False}))
        write_json(frozen, dict(identity=identity, frozen_at=datetime.now(timezone.utc).isoformat(),
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            protocol_sha256=digest(CONFIG), protocol_document_sha256=digest('docs/final_protocol.md')))
    return config, settings, protocol, p, risk, output


def train_one(profile, budget, seed, schedule):
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    config, settings, protocol, p, risk = context(profile)
    output = output_for(profile)
    root, droot = input_paths(profile)
    folder = output/'runs'/str(budget)/schedule/str(seed)
    if (folder/'completed.json').exists():
        completion = json.loads((folder/'completed.json').read_text())
        if completion['implementation_sha256'] != digest(Path(__file__).with_name('final_learning.py')):
            raise ValueError('BC update implementation changed after training')
        for name in ('selected', 'final'):
            if digest(folder/f'{name}.zip') != completion[name+'_sha256']:
                raise ValueError('Completed checkpoint changed')
        return completion
    folder.mkdir(parents=True, exist_ok=True)
    # The learner receives only these two public arrays, never the snapshot object.
    data = joblib.load(droot/'dataset_train.joblib')
    observations, labels = data['observations'].copy(), data['labels'].copy()
    del data
    validation_data = joblib.load(droot/'dataset_validation.joblib')
    panel = validation_data['observations']
    del validation_data
    actor = droot/f'imitation_models/Imitation_64x64/{seed}/selected.zip'
    critic = root/f'ppo_diagnostics/runs/canonical/{seed}/checkpoint_0.zip'
    factory = partial(BCRegularizedPPO, actor_checkpoint=actor, critic_checkpoint=critic,
        transfer_observations=panel, bc_observations=observations, bc_labels=labels,
        beta=0. if schedule == 'BC0' else protocol['beta_candidate'],
        bc_schedule='constant' if schedule == 'BC0' else schedule, bc_seed=seed+810001)
    snapshots = sorted(set([0, budget, *[x for x in (512, 2048, 8192, 32768) if x <= budget]]))
    recorder = InitializationRecorder(seed, folder, panel, snapshots)
    with record_gradients(recorder):
        metadata = train_agent(config, risk, settings, make_scenarios(config, settings, 'train', 'markov'),
            make_scenarios(config, settings, 'validation'), seed=seed, destination=folder,
            total_timesteps=budget, model_class=factory, diagnostic_callback=recorder)
    recorder.flush()
    result = dict(metadata=metadata, transfer=recorder.model.transfer_check,
        selected_sha256=digest(folder/'selected.zip'), final_sha256=digest(folder/'final.zip'),
        actor_sha256=digest(actor), critic_sha256=digest(critic),
        bc_train_sha256=digest(droot/'dataset_train.joblib'),
        implementation_sha256=digest(Path(__file__).with_name('final_learning.py')))
    write_json(folder/'completed.json', result)
    return result


def validation(profile):
    _, _, protocol, p, _, output = prepare(profile)
    records = []
    for schedule in ('constant', 'decay'):
        for seed in p['seeds']:
            result = train_one(profile, p['budgets'][0], seed, schedule)
            records.append(dict(schedule=schedule, seed=seed, beta=protocol['beta_candidate'],
                                score=result['metadata']['validation_score']))
    table = pd.DataFrame(records)
    table.to_csv(output/'schedule_validation.csv', index=False)
    selected = table.groupby('schedule').score.mean().sort_index().idxmax()
    controls = []
    _, droot = input_paths(profile)
    for seed in p['seeds']:
        control = droot/f'runs/32768/ImitationInit/{seed}/metadata.json'
        controls.append(json.loads(control.read_text())['validation_score'])
    result = dict(schedule=selected, beta=protocol['beta_candidate'], selection_split='validation',
        validation_csv_sha256=digest(output/'schedule_validation.csv'),
        selected_validation_score=float(table[table.schedule == selected].score.mean()),
        BC0_standard_validation_mean=sum(controls)/len(controls),
        smoke_control_budget_mismatch=profile == 'smoke')
    path = output/'selection.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('Frozen schedule selection changed')
    write_json(path, result)
    print('FROZEN SELECTION', result, flush=True)


def run(profile='smoke', stage='all'):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        _, _, _, p, _, output = prepare(profile)
        if stage == 'prepare':
            return output
        if stage in ('all', 'validation'):
            validation(profile)
        if stage == 'validation':
            return output
        selection = json.loads((output/'selection.json').read_text())
        if stage in ('all', 'train'):
            if profile == 'smoke':
                train_one(profile, p['budgets'][1], p['seeds'][0], selection['schedule'])
            else:
                with ProcessPoolExecutor(max_workers=3) as pool:
                    futures = [pool.submit(train_one, profile, p['budgets'][1], seed, selection['schedule'])
                               for seed in p['seeds']]
                    for future in futures:
                        future.result()
        if stage == 'train':
            return output
        from .final_panels import evaluate
        if stage in ('all', 'evaluate'):
            evaluate(profile)
        if stage in ('all', 'analysis'):
            from .final_analysis import analyze
            analyze(profile)
        verify_protected(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'standard'], default='smoke')
    parser.add_argument('--stage', choices=['all', 'prepare', 'validation', 'train', 'evaluate', 'analysis'], default='all')
    args = parser.parse_args()
    run(args.profile, args.stage)


if __name__ == '__main__':
    main()
