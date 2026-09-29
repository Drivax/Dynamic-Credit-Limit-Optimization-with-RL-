"""Validation/mechanism-only factorial confirmation with five fresh seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
from pathlib import Path

import pandas as pd
import numpy as np

from credit_rl.evaluation.policy_statistics import matrix, interval
from credit_rl.experiments.ppo_analysis import measurement_job
from credit_rl.experiments.ppo_diagnostics import prepare, experiment_rows, run_job, protection
from credit_rl.experiments.ppo_measurements import collapse_times, summarize_alignment


def mechanism(folder):
    panel = pd.read_csv(folder/'panel.csv')
    timing = collapse_times(panel)
    timing = timing[(timing.kind == 'deterministic_contraction') & (timing.threshold == .95)]
    crossing = timing.first_crossing.fillna(timing.last_observed).mean()
    regret = pd.read_csv(folder/'evaluation/validation_regret.csv').regret.mean()
    alignment = summarize_alignment(pd.read_csv(folder/'evaluation/mc_selected.csv')).ranking_accuracy.mean()
    return dict(collapse=crossing, regret=regret, rank=alignment)


def select_factors(rows, root, protocol, candidates):
    records = []
    canonical = {seed: mechanism(root/'runs/canonical'/str(seed)) for seed in protocol['seeds']}
    for row in rows:
        if row['family'] not in candidates or row['seed'] not in protocol['seeds']:
            continue
        current = mechanism(root/'runs'/row['experiment_id']/str(row['seed']))
        base = canonical[row['seed']]
        delay = current['collapse']-base['collapse']
        gain = base['regret']-current['regret']
        rank = current['rank']-base['rank']
        records.append(dict(experiment_id=row['experiment_id'], family=row['family'], seed=row['seed'],
            collapse_delay=delay, validation_regret_reduction=gain, ranking_gain=rank,
            supports=delay >= 2048 or gain >= 10 or rank >= .10))
    frame = pd.DataFrame(records)
    frame.to_csv(root/'confirmation_eligibility.csv', index=False)
    summary = frame.groupby(['experiment_id', 'family']).agg(support=('supports', 'sum'), delay=('collapse_delay', 'median')).reset_index()
    required = min(2, len(protocol['seeds']))
    summary = summary[summary.support >= required].sort_values(['support', 'delay', 'experiment_id'], ascending=[False, False, True])
    summary = summary.drop_duplicates('family').head(2)
    return [next(r for r in rows if r['experiment_id'] == name) for name in summary.experiment_id]


def plan(args, settings, matrix_config, protocol):
    path = args.output/'confirmatory_plan.json'
    if path.exists():
        return json.loads(path.read_text())
    rows = experiment_rows(args.profile, settings)
    factors = select_factors(rows, args.output, protocol, matrix_config['confirmation']['candidate_families'])
    jobs = []
    if factors:
        for bits in product((0, 1), repeat=len(factors)):
            active = [f for f, bit in zip(factors, bits) if bit]
            name = 'confirm_'+'__'.join(f['experiment_id'] for f in active) if active else 'confirm_control'
            changes = {k: v for f in active for k, v in f['changes'].items()}
            for seed in protocol['confirmation_seeds']:
                jobs.append(dict(experiment_id=name, family='confirmation', hypothesis='factorial mechanism confirmation',
                    parameter='factorial', canonical_value='canonical', intervention_value=json.dumps(changes),
                    changes=changes, seed=seed, budget=settings['ppo']['total_timesteps'], status='planned',
                    reference_experiment='confirm_control', factor_bits=list(bits)))
    result = dict(factors=[f['experiment_id'] for f in factors], jobs=jobs,
        selection_data='validation and collapse mechanisms only; test files never loaded',
        status='planned' if jobs else 'not_triggered',
        reason='Predeclared eligibility rule applied; no test-based rescue selection')
    path.write_text(json.dumps(result, indent=2))
    pd.DataFrame(jobs if jobs else [dict(status='not_triggered', reason=result['reason'])]).to_csv(args.output/'confirmatory_registry.csv', index=False)
    return result


def confirm_results(args, plan_result, protocol):
    if not plan_result['jobs']:
        pd.DataFrame([dict(status='not_triggered', reason=plan_result['reason'])]).to_csv(args.output/'confirmatory_results.csv', index=False)
        return
    frames = [pd.read_csv(args.output/'runs'/r['experiment_id']/str(r['seed'])/'evaluation/episodes.csv') for r in plan_result['jobs']]
    episodes = pd.concat(frames, ignore_index=True)
    rows = []
    for scenario, group in episodes.groupby('scenario'):
        for metric in ('discounted_reward', 'net_economic_value'):
            control = matrix(group[group.policy == 'confirm_control'], metric)
            values = {}
            for name, policy in group.groupby('policy'):
                if name == 'confirm_control':
                    continue
                current = matrix(policy, metric).reindex(index=control.index, columns=control.columns)
                delta = current.to_numpy()-control.to_numpy()
                values[name] = delta
                lo, hi = interval(delta, protocol['bootstrap_repetitions'], protocol['population_seed'])
                rows.append(dict(experiment_id=name, scenario=scenario, metric=metric, difference=delta.mean(),
                    lower=lo, upper=hi, seeds=len(current), customers=len(current.columns), analysis='fresh-seed factorial confirmation'))
            if len(plan_result['factors']) == 2:
                a, b = plan_result['factors']
                interaction = values[f'confirm_{a}__{b}']-values[f'confirm_{a}']-values[f'confirm_{b}']
                lo, hi = interval(interaction, protocol['bootstrap_repetitions'], protocol['population_seed'])
                rows.append(dict(experiment_id='factorial_interaction', scenario=scenario, metric=metric,
                    difference=interaction.mean(), lower=lo, upper=hi, seeds=interaction.shape[0],
                    customers=interaction.shape[1], analysis='fresh-seed factorial interaction'))
    pd.DataFrame(rows).to_csv(args.output/'confirmatory_results.csv', index=False)
    measurements = []
    for row in plan_result['jobs']:
        values = mechanism(args.output/'runs'/row['experiment_id']/str(row['seed']))
        measurements.append(dict(experiment_id=row['experiment_id'], seed=row['seed'], **values))
    frame = pd.DataFrame(measurements)
    frame.to_csv(args.output/'confirmatory_mechanism_seeds.csv', index=False)
    controls = frame[frame.experiment_id == 'confirm_control'].set_index('seed')
    estimates = []
    for name, group in frame.groupby('experiment_id'):
        if name == 'confirm_control':
            continue
        for metric in ('collapse', 'regret', 'rank'):
            current = group.set_index('seed')[metric]
            delta = (current-controls[metric].reindex(current.index)).to_numpy()
            rng = np.random.default_rng(protocol['population_seed'])
            draws = delta[rng.integers(len(delta), size=(protocol['bootstrap_repetitions'], len(delta)))].mean(1)
            lo, hi = np.quantile(draws, [.025, .975])
            estimates.append(dict(experiment_id=name, metric=metric, difference=delta.mean(), lower=lo, upper=hi,
                seeds=len(delta), estimand='paired fresh seeds; scenario-mean validation mechanism; collapse right-censored at budget'))
    pd.DataFrame(estimates).to_csv(args.output/'confirmatory_mechanisms.csv', index=False)


def run(args):
    _, settings, matrix_config, protocol, _ = prepare(args)
    planned = plan(args, settings, matrix_config, protocol)
    if args.stage == 'plan':
        return
    # Control first so changed architectures can retain the unchanged component initialization.
    for controls in (True, False):
        jobs = [r for r in planned['jobs'] if (r['experiment_id'] == 'confirm_control') == controls]
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            for future in as_completed([pool.submit(run_job, (args, r)) for r in jobs]):
                future.result()
    with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
        for future in as_completed([pool.submit(measurement_job, (args, r)) for r in planned['jobs']]):
            future.result()
    confirm_results(args, planned, protocol)
    completed = [json.loads((args.output/'runs'/r['experiment_id']/str(r['seed'])/'completed.json').read_text())
                 for r in planned['jobs']]
    if completed:
        pd.DataFrame(completed).to_csv(args.output/'confirmatory_registry.csv', index=False)
    if json.loads((args.output/'protected_artifacts.json').read_text()) != protection(args.canonical, args.phase_b):
        raise AssertionError('Protected artifacts changed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    p.add_argument('--canonical', type=Path, default=Path('outputs/main/standard'))
    p.add_argument('--phase-b', type=Path, default=Path('outputs/main/information_gap'))
    p.add_argument('--output', type=Path, default=Path('outputs/main/ppo_diagnostics'))
    p.add_argument('--stage', choices=['plan', 'all'], default='all')
    run(p.parse_args())


if __name__ == '__main__':
    main()
