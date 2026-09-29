"""Frozen-policy evaluation, aggregation and mechanistic confirmation selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch
import joblib

from credit_rl.evaluation.policy_engine import aggregate_episodes
from credit_rl.evaluation.policy_statistics import matrix, interval
from credit_rl.experiments.ppo_diagnostics import prepare, experiment_rows, protection, diagnostic_provenance
from credit_rl.experiments.ppo_measurements import measure_run, collapse_times, summarize_alignment, diagnostic_panel
from credit_rl.evaluation.policy_engine import PolicySpec, evaluate_policy
from credit_rl.policies.decision import ConstantAdjustment, MyopicEconomic
from credit_rl.policies.information import ObservationPlanner


def evaluate_references(args, config, settings, risk):
    """Frozen public references on exactly the same new paired test customers."""
    path = args.output/'reference_episodes.csv'
    if path.exists():
        return
    populations = joblib.load(args.output/'panels.joblib')['test']
    teacher = joblib.load(args.phase_b/'planners.joblib')['F0']['model']
    actors = {'AlwaysDecrease20': ConstantAdjustment(config, .8),
              'ObservationPlanner': ObservationPlanner(teacher, config), 'MyopicEconomic': MyopicEconomic(config)}
    frames = []
    for name, actor in actors.items():
        spec = PolicySpec(name, lambda e, s, actor=actor: actor)
        for macro, group in populations.items():
            frame, _, _ = evaluate_policy(spec, group, config, risk, settings, macro, keep_history=False)
            frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def measurement_job(job):
    args, row = job
    from credit_rl.experiments.main_evaluation import settings_for
    from credit_rl.experiments.ppo_diagnostics import load_protocol
    from credit_rl.risk.longitudinal import LongitudinalPDModel
    config, settings, _ = settings_for(args.profile, 'configs')
    _, protocol = load_protocol(args.profile)
    risk = LongitudinalPDModel.load(args.canonical/'models/pd/logistic_calibrated.joblib')
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        measure_run(args, row, config, settings, protocol, risk, temporal=row['family'] == 'canonical')
    folder = args.output/'runs'/row['experiment_id']/str(row['seed'])
    diagnostic_provenance(args, folder/'evaluation/provenance.json',
        dict(row=row, measurements=protocol, scenarios=['baseline', 'severe_stress']), folder/'selected.zip')
    print(f'MEASURED {row["experiment_id"]} seed={row["seed"]}', flush=True)


def aggregate(args, protocol):
    root = args.output
    records = [json.loads(p.read_text()) for p in root.glob('runs/*/*/completed.json')]
    panels, updates, episodes, policies, mc, exploration = [], [], [], [], [], []
    validation, fine_validation, actual_advantages, final_episodes = [], [], [], []
    summaries = []
    for row in records:
        folder = root/'runs'/row['experiment_id']/str(row['seed'])
        tag = dict(experiment_id=row['experiment_id'], seed=row['seed'], family=row['family'])
        validation.append(pd.read_csv(folder/'validation.csv').assign(**tag))
        fine_validation.append(pd.read_csv(folder/'fine_validation.csv').assign(**tag))
        panels.append(pd.read_csv(folder/'panel.csv').assign(family=row['family']))
        updates.append(pd.read_csv(folder/'updates.csv').assign(family=row['family']))
        destination = folder/'evaluation'
        if (destination/'episodes.csv').exists():
            if (destination/'final_episodes.csv').exists():
                final_episodes.append(pd.read_csv(destination/'final_episodes.csv'))
            if (destination/'actual_training_gae.csv').exists():
                actual_advantages.append(pd.read_csv(destination/'actual_training_gae.csv'))
            e = pd.read_csv(destination/'episodes.csv')
            episodes.append(e)
            metrics = pd.read_csv(destination/'policy_summary.csv')
            policies.append(pd.read_csv(destination/'policy_states.csv'))
            for scenario, group in e.groupby('scenario'):
                m = metrics[metrics.scenario == scenario].iloc[0].to_dict()
                summaries.append(dict(**aggregate_episodes(group), purchases=group.purchases.mean(),
                    training_discounted_reward=group.training_discounted_reward.mean(), **m, family=row['family'],
                    budget=row['budget'], selected_timesteps=row['selected_timesteps']))
            for p in destination.glob('mc_*.csv'):
                mc.append(pd.read_csv(p).assign(checkpoint_kind=p.stem.removeprefix('mc_')))
        buffer = pd.read_csv(folder/'rollouts.csv.gz', usecols=['timesteps', 'action', 'obs_0', 'obs_3', 'obs_10'])
        buffer['pd_bucket'] = np.searchsorted([.2, .6], buffer.obs_10)
        buffer['utilization_bucket'] = np.searchsorted([1/3, .5], buffer.obs_3)
        buffer['horizon_bucket'] = np.searchsorted([1/3, 2/3], buffer.obs_0)
        counts = buffer.groupby(['timesteps', 'pd_bucket', 'utilization_bucket', 'horizon_bucket', 'action']).size().rename('count').reset_index()
        exploration.append(counts.assign(**tag))
    panel = pd.concat(panels, ignore_index=True)
    update = pd.concat(updates, ignore_index=True)
    panel.to_csv(root/'action_distribution.csv', index=False)
    panel[['experiment_id', 'seed', 'scenario', 'timesteps', 'entropy']].to_csv(root/'entropy.csv', index=False)
    collapse = collapse_times(panel)
    collapse.to_csv(root/'collapse_timing.csv', index=False)
    update.to_csv(root/'training_updates.csv', index=False)
    pd.concat(exploration, ignore_index=True).to_csv(root/'exploration.csv', index=False)
    pd.concat(validation, ignore_index=True).to_csv(root/'checkpoint_selection.csv', index=False)
    pd.concat(fine_validation, ignore_index=True).to_csv(root/'validation_trajectories.csv', index=False)
    if actual_advantages:
        pd.concat(actual_advantages, ignore_index=True).to_csv(root/'actual_training_gae.csv', index=False)
    if final_episodes:
        pd.concat(final_episodes, ignore_index=True).to_csv(root/'final_episode_metrics.csv', index=False)
    canonical = panel[panel.experiment_id == 'canonical'].merge(update[update.experiment_id == 'canonical'],
        on=['experiment_id', 'seed', 'timesteps'], suffixes=('_panel', '_rollout'), how='left')
    fine = []
    for r in records:
        if r['experiment_id'] == 'canonical':
            fine.append(pd.read_csv(root/f'runs/canonical/{r["seed"]}/fine_validation.csv').assign(seed=r['seed']))
    canonical = canonical.merge(pd.concat(fine), on=['seed', 'timesteps', 'scenario'], how='left')
    canonical.to_csv(root/'canonical_training.csv', index=False)
    if not episodes:
        return
    e = pd.concat(episodes, ignore_index=True)
    e.to_csv(root/'episode_metrics.csv', index=False)
    pd.DataFrame(summaries).to_csv(root/'intervention_results.csv', index=False)
    states = pd.concat(policies, ignore_index=True)
    states.to_csv(root/'policy_regret.csv', index=False)
    pd.DataFrame(summaries)[['experiment_id', 'seed', 'scenario', 'diversity', 'mi_pd', 'mi_utilization', 'mi_horizon', 'mi_macro']].to_csv(root/'representation.csv', index=False)
    raw = pd.concat(mc, ignore_index=True)
    raw.to_csv(root/'gae_alignment_states.csv', index=False)
    alignment = summarize_alignment(raw.drop_duplicates(['experiment_id', 'seed', 'timesteps', 'scenario', 'checkpoint_kind', 'state_id', 'action']))
    alignment.to_csv(root/'gae_alignment.csv', index=False)
    alignment[['experiment_id', 'seed', 'timesteps', 'scenario', 'checkpoint_kind', 'states', 'critic_bias', 'critic_rmse', 'critic_r2']].to_csv(root/'critic_diagnostics.csv', index=False)
    comparisons = []
    for experiment, group in e.groupby('policy'):
        if experiment == 'canonical':
            continue
        for metric in ('discounted_reward', 'net_economic_value', 'defaulted', 'credit_loss', 'capital_charge'):
            deltas = {}
            for scenario, g in group.groupby('scenario'):
                a = matrix(g, metric)
                b = matrix(e[(e.policy == 'canonical') & (e.scenario == scenario)], metric).reindex(index=a.index, columns=a.columns)
                if b.isna().any().any():
                    continue
                delta = a.to_numpy()-b.to_numpy()
                deltas[scenario] = delta
                lo, hi = interval(delta, protocol['bootstrap_repetitions'], protocol['population_seed'])
                comparisons.append(dict(experiment_id=experiment, scenario=scenario, metric=metric,
                    difference=delta.mean(), lower=lo, upper=hi, seeds=len(a), customers=len(a.columns),
                    customer_sd=delta.mean(0).std(ddof=1), analysis='exploratory paired seeds/customers'))
            if len(deltas) == 2:
                delta = deltas['severe_stress']-deltas['baseline']
                lo, hi = interval(delta, protocol['bootstrap_repetitions'], protocol['population_seed'])
                comparisons.append(dict(experiment_id=experiment, scenario='stress_minus_baseline_interaction', metric=metric,
                    difference=delta.mean(), lower=lo, upper=hi, seeds=delta.shape[0], customers=delta.shape[1],
                    customer_sd=delta.mean(0).std(ddof=1), analysis='exploratory treatment x scenario'))
    pd.DataFrame(comparisons).to_csv(root/'scenario_comparison.csv', index=False)


def run(args):
    config, settings, _, protocol, risk = prepare(args)
    if args.stage == 'measure':
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            diagnostic_panel(args, config, settings, protocol, risk)
            evaluate_references(args, config, settings, risk)
        rows = experiment_rows(args.profile, settings)
        selected = [r for r in rows if (args.family == 'all' or r['family'] == args.family)
                    and (args.output/'runs'/r['experiment_id']/str(r['seed'])/'completed.json').exists()]
        with ProcessPoolExecutor(max_workers=args.workers or protocol['workers']) as pool:
            futures = [pool.submit(measurement_job, (args, row)) for row in selected]
            for f in as_completed(futures):
                f.result()
    aggregate(args, protocol)
    if json.loads((args.output/'protected_artifacts.json').read_text()) != protection(args.canonical, args.phase_b):
        raise AssertionError('Protected artifacts changed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    p.add_argument('--canonical', type=Path, default=Path('outputs/main/standard'))
    p.add_argument('--phase-b', type=Path, default=Path('outputs/main/information_gap'))
    p.add_argument('--output', type=Path, default=Path('outputs/main/ppo_diagnostics'))
    p.add_argument('--family', default='all')
    p.add_argument('--stage', choices=['measure', 'aggregate'], default='measure')
    p.add_argument('--workers', type=int)
    run(p.parse_args())


if __name__ == '__main__':
    main()
