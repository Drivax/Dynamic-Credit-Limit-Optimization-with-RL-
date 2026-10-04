"""Finish Phase D tables, report and verification after all registered runs."""
import argparse
import json

import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
import torch

from credit_rl.experiments.initialization_analysis import aggregate
from credit_rl.experiments.initialization_report import scientific_report
from credit_rl.experiments.initialization_verify import verify
from credit_rl.experiments.policy_initialization import ROOT


def advantage_tables(output):
    rows = []
    for path in (output/'analysis').glob('*/*/actual_advantage_pressure.csv.gz'):
        frame = pd.read_csv(path)
        for (seed, arm, budget, steps, teacher, action), g in frame.groupby(
                ['seed', 'experiment_id', 'budget', 'timesteps', 'teacher_action', 'action']):
            rows.append(dict(seed=seed, experiment_id=arm, budget=budget, timesteps=steps,
                teacher_action=teacher, sampled_action=action, count=len(g), raw_gae_mean=g.advantage.mean(),
                negative_fraction=(g.advantage < 0).mean(),
                critic_gae_rmse=((g.value-g.returns)**2).mean()**.5,
                units='scaled actual on-policy GAE; pre-update critic versus GAE returns'))
    pd.DataFrame(rows).to_csv(output/'actual_advantage_pressure.csv', index=False)
    normalized, updates = [], []
    for path in (output/'runs').glob('*/*/*/completed.json'):
        folder = path.parent
        seed, arm, budget = int(folder.name), folder.parent.name, int(folder.parent.parent.name)
        mini = pd.read_csv(folder/'minibatch_advantages.csv')
        mini = mini[mini.timesteps <= 8192]
        group = mini.groupby(['timesteps', 'action'])[['count', 'normalized_sum', 'normalized_negative']].sum().reset_index()
        group['normalized_advantage_mean'] = group.normalized_sum/group['count']
        group['negative_fraction'] = group.normalized_negative/group['count']
        normalized.append(group.assign(seed=seed, experiment_id=arm, budget=budget,
            scope='sampled action; not conditioned on teacher agreement'))
        updates.append(pd.read_csv(folder/'updates.csv').assign(experiment_id=arm, budget=budget))
    pd.concat(normalized, ignore_index=True).to_csv(output/'normalized_advantage_pressure.csv', index=False)
    pd.concat(updates, ignore_index=True).to_csv(output/'update_diagnostics.csv', index=False)
    efficiency = []
    folders = list((output/'runs').glob('*/*/*/completed.json'))
    short = min(int(p.parent.parent.parent.name) for p in folders)
    for path in folders:
        folder = path.parent
        seed, arm, budget = int(folder.name), folder.parent.name, int(folder.parent.parent.name)
        control = pd.read_csv(output/'runs'/str(short)/'RandomInit'/str(seed)/'validation.csv')
        threshold = control.validation_discounted_reward.max()
        curve = pd.read_csv(folder/'validation.csv')
        passed = curve.validation_discounted_reward >= threshold
        efficiency.append(dict(seed=seed, experiment_id=arm, budget=budget,
            reference='paired short-budget random best validation reward', threshold=threshold,
            first_crossing=curve.loc[passed, 'timesteps'].min() if passed.any() else float('nan'),
            censored=not passed.any(), last_observed=budget,
            scope='descriptive threshold comparison; no model selection change'))
    pd.DataFrame(efficiency).to_csv(output/'sample_efficiency.csv', index=False)
    initial_actions = []
    teacher_added = False
    for path in (output/'imitation_models').glob('*/*/validation_states.csv'):
        frame = pd.read_csv(path)
        for scenario, group in frame.groupby('scenario'):
            for action in range(5):
                initial_actions.append(dict(policy=path.parent.parent.name, seed=int(path.parent.name),
                    scenario=scenario, action=action, share=group.action.eq(action).mean()))
                if not teacher_added:
                    initial_actions.append(dict(policy='Frozen F0 teacher', seed=-1,
                        scenario=scenario, action=action, share=group.teacher_action.eq(action).mean()))
        teacher_added = True
    pd.DataFrame(initial_actions).to_csv(output/'initial_action_distribution.csv', index=False)
    actual_path = output/'actual_teacher_minibatch_pressure.csv'
    if actual_path.exists():
        actual = pd.read_csv(actual_path)
        actual = actual[actual.teacher_action.eq(actual.sampled_action)].copy()
        actual['normalized_sum'] = actual.normalized_gae_mean*actual['count']
        actual['negative_count'] = actual.normalized_negative_fraction*actual['count']
        records = []
        for (arm, action), group in actual.groupby(['experiment_id', 'teacher_action']):
            seeds = group.groupby('seed')[['count', 'normalized_sum', 'negative_count']].sum()
            draws = np.random.default_rng(784).integers(len(seeds), size=(2000, len(seeds)))
            estimates = seeds.normalized_sum.to_numpy()[draws].sum(1)/seeds['count'].to_numpy()[draws].sum(1)
            low, high = np.quantile(estimates, [.025, .975])
            records.append(dict(experiment_id=arm, teacher_action=action, seed_count=len(seeds),
                count=int(seeds['count'].sum()), normalized_mean=seeds.normalized_sum.sum()/seeds['count'].sum(),
                low=low, high=high, negative_fraction=seeds.negative_count.sum()/seeds['count'].sum(),
                interval='seed-cluster bootstrap; conditional on sampled action matching teacher; Markov occupancy'))
        pd.DataFrame(records).to_csv(output/'actual_teacher_pressure_summary.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    output = ROOT/('policy_initialization' if args.profile == 'standard' else 'policy_initialization_smoke')
    identity = json.loads((output/'preregistration.json').read_text())['identity']
    p = identity['protocol'][args.profile]
    for budget in p['budgets']:
        for seed in p['seeds']:
            for arm in ('RandomInit', 'ImitationInit'):
                if not (output/'analysis'/str(budget)/f'{arm}_{seed}'/'completed.json').exists():
                    raise ValueError(f'Incomplete analysis: {budget}, {arm}, {seed}')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        aggregate(output, p)
        advantage_tables(output)
        scientific_report(output, publish=args.publish and args.profile == 'standard')
        verify(output)


if __name__ == '__main__':
    main()
