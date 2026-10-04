"""Audit Phase D checkpoint identity, paired controls and CSV-only regeneration."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import torch

from credit_rl.experiments.information_ppo import same_parameters
from credit_rl.experiments.initialization_report import figures, registry
from credit_rl.experiments.main_evaluation import digest
from credit_rl.experiments.policy_initialization import ROOT, verify_protection, write_json
from credit_rl.experiments.initialization_metrics import bucket_quality
from credit_rl.envs.observation import OBSERVATION_NAMES


def verify(output):
    protected = verify_protection(output)
    protocol = json.loads((output/'preregistration.json').read_text())['identity']
    selection = json.loads((output/'initialization_selection.json').read_text())
    p = protocol['protocol'][protocol['profile']]
    identities = []
    for role in ('train', 'validation', 'test'):
        data = joblib.load(output/f'dataset_{role}.joblib')
        assert data['observations'].shape[1] == 21
        assert tuple(data['feature_names']) == OBSERVATION_NAMES
        assert data['states'].role.eq(role).all()
        assert np.isfinite(data['observations']).all()
        identities.append(set(data['states'].customer_id))
        expected = json.loads((output/'dataset_hashes.json').read_text())[role]
        assert digest(output/f'dataset_{role}.joblib') == expected
    assert not (identities[0] & identities[1] or identities[0] & identities[2] or identities[1] & identities[2])
    validation = joblib.load(output/'dataset_validation.joblib')
    pd.DataFrame(validation['observations'], columns=OBSERVATION_NAMES).to_csv(
        output/'public_validation_observations.csv', index=False)
    buckets = []
    for folder in (output/'imitation_models').glob('*/*'):
        if not (folder/'validation_states.csv').exists():
            continue
        frame = pd.read_csv(folder/'validation_states.csv')
        buckets.append(bucket_quality(frame, validation['observations']).assign(
            experiment_id=folder.parent.name, seed=int(folder.name), split='validation'))
    pd.concat(buckets, ignore_index=True).to_csv(output/'imitation_validation_buckets.csv', index=False)
    checked = []
    for budget in p['budgets']:
        for seed in p['seeds']:
            models = {}
            for arm in ('RandomInit', 'ImitationInit'):
                folder = output/'runs'/str(budget)/arm/str(seed)
                completion = json.loads((folder/'completed.json').read_text())
                assert completion['teacher_in_objective'] is False
                for name in ('selected', 'final'):
                    assert digest(folder/f'{name}.zip') == completion[f'{name}_sha256']
                snapshots = pd.read_csv(folder/'snapshots.csv')
                assert snapshots.timesteps.tolist() == [s for s in p['snapshots'] if s <= budget]
                for row in snapshots.itertuples():
                    assert digest(Path(row.checkpoint)) == row.sha256
                    assert row.checkpoint_kind == ('initial' if row.timesteps == 0 else 'post_update')
                models[arm] = PPO.load(folder/'temporal_0.zip', device='cpu')
                checked.append(dict(arm=arm, seed=seed, budget=budget, snapshots=len(snapshots)))
            a, b = (models[arm].policy.state_dict() for arm in ('RandomInit', 'ImitationInit'))
            assert all(torch.equal(a[k], b[k]) for k in a if k.startswith(('value_net.', 'mlp_extractor.value_net.')))
            if protocol['profile'] == 'standard':
                reference = PPO.load(ROOT/f'ppo_diagnostics/runs/canonical/{seed}/checkpoint_0.zip', device='cpu')
                assert same_parameters(models['RandomInit'], reference)
                if budget == min(p['budgets']):
                    for name in ('selected', 'final'):
                        reference = PPO.load(ROOT/f'ppo_diagnostics/runs/canonical/{seed}/{name}.zip', device='cpu')
                        replay = PPO.load(output/'runs'/str(budget)/'RandomInit'/str(seed)/f'{name}.zip', device='cpu')
                        assert same_parameters(reference, replay), f'Canonical replay mismatch: {seed}, {name}'
                else:
                    for arm in ('RandomInit', 'ImitationInit'):
                        short = min(p['budgets'])
                        first = PPO.load(output/'runs'/str(short)/arm/str(seed)/f'temporal_{short}.zip', device='cpu')
                        prefix = PPO.load(output/'runs'/str(budget)/arm/str(seed)/f'temporal_{short}.zip', device='cpu')
                        assert same_parameters(first, prefix), f'Short/long prefix mismatch: {seed}, {arm}'
            source = output/'imitation_models/Imitation_64x64'/str(seed)/'selected.zip'
            if selection['selected_architecture'] == [64, 64]:
                assert digest(source) == selection['decisions'][0]['checkpoint_hashes'][str(seed)]
    historical = json.loads((output/'historical_prefixes.json').read_text())
    for name, info in historical.items():
        prefix = Path(name).read_bytes()[:info['size']]
        assert hashlib.sha256(prefix).hexdigest() == info['sha256']
    registry(output)
    figures(output)
    before = {p.name: digest(p) for p in (output/'figures').glob('*.png')}
    figures(output)
    after = {p.name: digest(p) for p in (output/'figures').glob('*.png')}
    assert before == after
    write_json(output/'figure_hashes.json', after)
    result = dict(status='passed', protected_files=protected, paired_runs=len(checked),
        checkpoints=sum(r['snapshots'] for r in checked), csv_figure_regeneration_identical=True,
        historical_document_prefixes_identical=True, runs=checked,
        source_hashes={str(p): digest(p) for p in Path('src/credit_rl/experiments').glob('*initialization*.py')})
    write_json(output/'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('runs', 'source_hashes')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify(ROOT/('policy_initialization' if args.profile == 'standard' else 'policy_initialization_smoke'))


if __name__ == '__main__':
    main()
