"""Frozen source/artifact provenance and exact BC0 replay verification."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from stable_baselines3 import PPO
import torch

from .final_evaluation import context, output_for, train_one, verify_protected, DROOT, input_paths
from .main_evaluation import digest
from .policy_initialization import write_json


def audit_bc0(profile='standard'):
    _, _, _, p, _ = context(profile)
    output = output_for(profile)
    rows = []
    for seed in p['seeds']:
        budget = p['budgets'][0]
        train_one(profile, budget, seed, 'BC0')
        if profile != 'standard':
            continue
        for kind in ('selected', 'final'):
            left = PPO.load(output/f'runs/{budget}/BC0/{seed}/{kind}.zip', device='cpu')
            right = PPO.load(DROOT/f'runs/{budget}/ImitationInit/{seed}/{kind}.zip', device='cpu')
            for key, value in left.policy.state_dict().items():
                if not torch.equal(value, right.policy.state_dict()[key]):
                    raise AssertionError(f'BC0 differs from frozen BCInit: {seed} {kind} {key}')
            rows.append(dict(seed=seed, checkpoint=kind, exact_policy_tensors=True))
    write_json(output/'bc0_verification.json', dict(profile=profile, comparisons=rows,
                                                  status='passed' if rows else 'smoke_only'))


def manifest(profile='standard'):
    config, settings, protocol, p, _ = context(profile)
    output = output_for(profile)
    root, droot = input_paths(profile)
    count = verify_protected(output)
    tables = {str(f): digest(f) for f in output.glob('*.csv')}
    figures = {str(f): digest(f) for f in (output/'figures').glob('*.png')}
    sources = {str(f): digest(f) for f in Path('src/credit_rl').rglob('*.py')}
    models = {str(f): digest(f) for f in (output/'runs').rglob('*.zip')}
    models.update({str(f): digest(f) for f in (droot/'imitation_models').rglob('selected.zip')})
    models.update({str(f): digest(f) for f in (droot/'runs').rglob('selected.zip')})
    selection = json.loads((output/'selection.json').read_text())
    result = dict(profile=profile, scientific_evidence=profile == 'standard',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        working_tree_source_sha256=sources, python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in
                  ('numpy', 'pandas', 'torch', 'stable-baselines3', 'scikit-learn', 'gymnasium')},
        dgp_sha256=digest('configs/simulation.yaml'),
        pd_sha256=digest(root/'standard/models/pd/logistic_calibrated.joblib'),
        teacher_sha256=digest(root/'information_gap/planners.joblib'),
        ppo_config=settings['ppo'], evaluation_config=protocol, seeds=p['seeds'],
        worlds=json.loads((output/'worlds.json').read_text()), selection=selection,
        model_sha256=models, headline_csv_sha256=tables, figure_sha256=figures,
        preregistration_sha256=digest(output/'preregistration.json'),
        protected_historical_files=count,
        documents={str(f): digest(f) for f in (Path('README.md'), Path('docs/technical_paper.md'))})
    target = Path('outputs/main/final_manifest.json') if profile == 'standard' else output/'final_manifest.json'
    write_json(target, result)
    write_json(output/'frozen_products.json', {**tables, **figures,
        str(output/'selection.json'): digest(output/'selection.json')})
    return result
