"""Verify Phase C completion, pairing, immutable inputs and CSV-only regeneration."""
import argparse
import json
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
import torch

from credit_rl.experiments.main_evaluation import digest
from credit_rl.experiments.information_ppo import same_parameters
from credit_rl.experiments.ppo_diagnostics import protection
from credit_rl.experiments.ppo_report import run as regenerate


def verify(args):
    root = args.output
    protected = json.loads((root/'protected_artifacts.json').read_text())
    assert protection(args.canonical, args.phase_b) == protected, 'Protected Phase A/B, reward or DGP changed'
    registry = pd.read_csv(root/'experiment_registry.csv')
    assert registry.status.eq('complete').all(), 'Incomplete preregistered runs'
    checked = 0
    for path in root.glob('runs/*/*/snapshots.csv'):
        records = pd.read_csv(path)
        for row in records.itertuples(index=False):
            assert digest(Path(row.checkpoint)) == row.sha256, f'Changed snapshot: {row.checkpoint}'
            checked += 1
    torch.set_num_threads(1)
    for seed in (101, 202, 303):
        for kind in ('selected', 'final'):
            replay = root/f'runs/canonical/{seed}/{kind}.zip'
            if replay.exists():
                assert same_parameters(PPO.load(replay), PPO.load(args.canonical/f'models/ppo_{seed}/{kind}.zip'))
    episodes = pd.read_csv(root/'episode_metrics.csv')
    population_sets = episodes.groupby(['policy', 'policy_seed', 'scenario']).customer_id.agg(lambda x: tuple(sorted(x)))
    assert len(set(population_sets)) == 1, 'Evaluation customers differ across policies or scenarios'
    required = ['canonical_ppo_configuration', 'canonical_training', 'experiment_registry', 'collapse_timing',
                'action_distribution', 'entropy', 'gae_alignment', 'critic_diagnostics', 'exploration', 'representation',
                'imitation', 'policy_regret', 'scenario_comparison', 'intervention_results', 'confirmatory_results']
    assert all((root/f'{name}.csv').exists() for name in required), 'Missing required output'
    assert len(list((root/'figures').glob('*.png'))) == 13, 'Missing figure'
    confirmation = pd.read_csv(root/'confirmatory_registry.csv')
    assert confirmation.status.isin(['complete', 'not_triggered']).all(), 'Incomplete factorial confirmation'
    # Re-rendering derives tables and every figure solely from the frozen CSV inputs.
    regenerate(args)
    products = list(root.glob('*.csv'))+list((root/'figures').glob('*.png'))
    before = {str(p): digest(p) for p in products}
    regenerate(args)
    assert before == {str(p): digest(p) for p in products}, 'CSV/figure regeneration is not byte reproducible'
    result = dict(status='passed', protected_files=len(protected), runs=len(registry), snapshots=checked,
                  confirmation_runs=int(confirmation.status.eq('complete').sum()),
                  identical_canonical_weights=True, paired_customers=True, regenerated_products=len(products),
                  exact_csv_figure_regeneration=True)
    (root/'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    p.add_argument('--canonical', type=Path, default=Path('outputs/main/standard'))
    p.add_argument('--phase-b', type=Path, default=Path('outputs/main/information_gap'))
    p.add_argument('--output', type=Path, default=Path('outputs/main/ppo_diagnostics'))
    p.add_argument('--report', type=Path)
    verify(p.parse_args())


if __name__ == '__main__':
    main()
