"""Reproducible local structural diagnosis; frozen canonical artifacts required."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
import torch

from credit_rl import CreditLimitEnv
from credit_rl.evaluation.scenarios import make_scenarios
from credit_rl.evaluation.structural import (
    DecisionSnapshot, COMPONENTS, SIGNS, hypothetical_paths,
    rollout, value_tables, action_statistics, admissible,
)
from credit_rl.experiments.main_evaluation import settings_for, digest
from credit_rl.policies.baselines import SB3Policy
from credit_rl.policies.decision import ConstantAdjustment, MyopicEconomic
from credit_rl.risk.longitudinal import LongitudinalPDModel

GAMMAS = (.90, .95, .98, .995, 1.)
HORIZONS = (1, 3, 6, 12, 24)  # 24 means full remaining horizon, explicitly truncated


def describe(snapshot, scenario, source):
    s = snapshot.state
    return dict(customer_id=s.customer_id, scenario=scenario, source_policy=source,
        month=snapshot.elapsed, remaining_horizon=24-snapshot.elapsed,
        phase=('early' if snapshot.elapsed < 8 else 'middle' if snapshot.elapsed < 16 else 'late'),
        pd=snapshot.predicted_pd, utilization=s.utilization, balance=s.balance,
        current_limit=s.credit_limit, income=s.income, payment_ratio=s.payment_ratio,
        delinquency=s.months_delinquent, behavioral_score=s.behavioral_score,
        macro_regime=int(s.macro_state.regime), macro_stress=s.macro_state.credit_stress,
        pd_bucket=int(np.searchsorted([.2, .6], snapshot.predicted_pd)),
        utilization_bucket=int(np.searchsorted([.5, 1.], s.utilization)),
        income_bucket='low' if s.income < 3000 else 'high',
        **{'latent_'+k: v for k, v in asdict(snapshot.traits).items()})


def sample_states(config, settings, risk, policies, per_stratum, seed, output):
    """Stratified uniform sample of actual active visits; inverse inclusion weights."""
    snapshots, rows = [], []
    for scenario in ('baseline', 'severe_stress'):
        for sc in make_scenarios(config, settings, 'test', scenario):
            for source in ('Static', 'MyopicEconomic', 'AlwaysDecrease20'):
                env = CreditLimitEnv(config=config, pd_model=risk, record_history=False,
                    severe_delinquency_months=settings['guardrails']['severe_delinquency_months'])
                obs, _ = env.reset(seed=sc.customer_seed, options=sc.reset_options())
                while not env._done:
                    snap = DecisionSnapshot.capture(env)
                    snapshots.append(snap)
                    rows.append(describe(snap, scenario, source))
                    obs, *_ = env.step(policies[source].act(obs))
        print(f'Visited states collected: {scenario}, {len(rows)}', flush=True)
    pool = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    chosen = []
    for _, group in pool.groupby(['scenario', 'source_policy', 'phase'], sort=True):
        n = min(per_stratum, len(group))
        indices = rng.choice(group.index.to_numpy(), size=n, replace=False)
        for index in indices:
            chosen.append((int(index), len(group)/n))
    chosen.sort()
    frame = pool.loc[[i for i, _ in chosen]].reset_index(drop=True)
    frame['weight'] = [w for _, w in chosen]
    frame['state_id'] = np.arange(len(frame))
    frame.to_csv(output/'states.csv', index=False)
    support = []
    for column in ('pd_bucket', 'utilization_bucket', 'income_bucket', 'phase', 'macro_regime',
                   'delinquency', 'remaining_horizon'):
        counts = pool.groupby(column).size()
        selected = frame.groupby(column).size()
        for bucket, count in counts.items():
            support.append(dict(variable=column, bucket=bucket, visited=count,
                                sampled=int(selected.get(bucket, 0))))
    pd.DataFrame(support).to_csv(output/'state_support.csv', index=False)
    return [snapshots[i] for i, _ in chosen], frame


def save_tables(output, frames):
    for name, frame in frames.items():
        frame.to_csv(output/f'{name}.csv', index=False)


def dominance_tables(gaps, states):
    merged = gaps.merge(states, on='state_id', validate='many_to_one')
    records, distributions = [], []
    groups = ['continuation', 'horizon', 'gamma']
    for variable in ('all', 'pd_bucket', 'utilization_bucket', 'macro_regime',
                     'remaining_horizon', 'income_bucket', 'phase', 'source_policy'):
        frame = merged.assign(bucket='all') if variable == 'all' else merged.assign(bucket=merged[variable].astype(str))
        for key, group in frame.groupby(groups+['bucket']):
            tags = dict(zip(groups+['bucket'], key)) | {'stratification': variable, 'states': len(group)}
            weight = group.weight.to_numpy()
            shares = [float(np.average(group.best_action == a, weights=weight)) for a in range(5)]
            for a, share in enumerate(shares):
                records.append({**tags, 'action': a, 'share': share, 'degeneracy': max(shares),
                    'resolved_share': float(np.average((group.best_action == a) & (group.gap > 1.96*group.gap_se), weights=weight)),
                    'tie_share': float(np.average(group.tie_count > 1, weights=weight))})
            order = np.argsort(group.gap.to_numpy())
            quantiles = np.interp([.1, .25, .5, .75, .9, .95],
                (np.cumsum(weight[order])-.5*weight[order])/weight.sum(), group.gap.to_numpy()[order])
            distributions.append({**tags, 'mean': float(np.average(group.gap, weights=weight)),
                **dict(zip(('p10', 'p25', 'median', 'p75', 'p90', 'p95'), quantiles)),
                'unresolved_share': float(np.average(group.gap <= 1.96*group.gap_se, weights=weight))})
    return pd.DataFrame(records), pd.DataFrame(distributions)


def complete_buckets(totals, components):
    """Zero occupancy contributes zero, so subgroup attribution sums to total.

    Different policies need not visit the same utilization/PD/default buckets.
    An inner join would silently drop the reference contribution in absent bins.
    """
    identities = ['scenario', 'policy', 'policy_seed']
    policies = totals[identities+['initial_customers']].drop_duplicates()
    buckets = totals[['scenario', 'stratification', 'bucket']].drop_duplicates()
    grid = policies.merge(buckets, on='scenario')
    result = grid.merge(totals.drop(columns='initial_customers'),
                        on=identities+['stratification', 'bucket'], how='left', validate='one_to_one')
    result[components] = result[components].fillna(0.)
    return result


def panel_diagnostics(canonical, output, config):
    """Reuse saved canonical panel; no retraining or parallel evaluation protocol."""
    h = pd.read_csv(canonical/'results/trajectories.csv.gz')
    e = pd.read_csv(canonical/'results/episode_metrics.csv')
    keys = ['scenario', 'policy', 'policy_seed', 'customer_id']
    h = h.sort_values(keys+['month'])
    for name in ('utilization', 'balance', 'credit_limit', 'income'):
        h['opening_'+name] = h.groupby(keys)[name].shift(1)
    h['cumulative_reward'] = h.groupby(keys).reward.cumsum()
    t = h[h.month > 0].copy()
    parts = t[['reward_'+x for x in COMPONENTS]].to_numpy()
    t['net_economic_value'] = parts[:, :4] @ SIGNS[:4]
    t['reward_error'] = t.reward - parts @ SIGNS
    t['net_identity_error'] = t.net_economic_value - (t.reward+t.reward_capital_cost+t.reward_constraint_penalty)
    t['pd_bucket'] = np.searchsorted([.2, .6], t.decision_pd)
    t['utilization_bucket'] = np.searchsorted([.5, 1.], t.opening_utilization)
    t['episode_defaulted'] = t.groupby(keys).defaulted.transform('max')
    t.to_csv(output/'reward_decomposition.csv', index=False)
    comparisons = []
    components = ['reward_'+x for x in COMPONENTS]+['reward', 'net_economic_value']
    # Conditional buckets are descriptive policy-specific occupancy, not matched causal strata.
    for variable in ('all', 'pd_bucket', 'utilization_bucket', 'month', 'episode_defaulted'):
        frame = t.assign(bucket='all') if variable == 'all' else t.assign(bucket=t[variable].astype(str))
        sums = frame.groupby(['scenario', 'policy', 'policy_seed', 'bucket'])[components].sum().reset_index()
        denominators = e.groupby(['scenario', 'policy', 'policy_seed']).size()
        for _, row in sums.iterrows():
            n = denominators.loc[(row.scenario, row.policy, row.policy_seed)]
            comparisons.append(dict(scenario=row.scenario, policy=row.policy, policy_seed=row.policy_seed,
                stratification=variable, bucket=row.bucket, initial_customers=n,
                **{c: row[c]/n for c in components}))
    totals = complete_buckets(pd.DataFrame(comparisons), components)
    ref = totals[totals.policy == 'AlwaysDecrease20'].drop(columns=['policy', 'policy_seed', 'initial_customers'])
    deltas = totals.merge(ref, on=['scenario', 'stratification', 'bucket'], suffixes=('', '_reference'))
    # Positive signed contributions favour AlwaysDecrease20 over the comparison policy.
    for c, sign in zip(components, [*SIGNS, 1, 1]):
        deltas['decrease_minus_policy_'+c] = sign*(deltas[c+'_reference']-deltas[c])
    deltas.to_csv(output/'policy_value_decomposition.csv', index=False)
    collapse, differences = [], []
    reference = t[t.policy == 'AlwaysDecrease20']
    for (scenario, seed), ppo in t[t.policy == 'PPO'].groupby(['scenario', 'policy_seed']):
        paired = ppo.merge(reference[reference.scenario == scenario], on=['customer_id', 'month'],
                           suffixes=('_ppo', '_decrease'), how='outer', indicator=True)
        both = paired[paired._merge == 'both'].copy()
        row = dict(scenario=scenario, seed=seed, matched=len(both), unmatched=int((paired._merge != 'both').sum()))
        for c in ('requested_action', 'effective_action', 'credit_limit', 'balance', 'utilization',
                  'defaulted', 'cumulative_reward', 'net_economic_value'):
            delta = both[c+'_ppo'].astype(float)-both[c+'_decrease'].astype(float)
            row[c+'_agreement'] = float(np.isclose(delta, 0, atol=1e-8, rtol=0).mean())
            row[c+'_max_difference'] = float(abs(delta).max())
        row['policy_collapse'] = row['effective_action_agreement'] if not row['unmatched'] else np.nan
        collapse.append(row)
        changed = both[~np.isclose(both.requested_action_ppo, both.requested_action_decrease, atol=1e-8)]
        differences.append(changed.assign(scenario=scenario, seed=seed))
    pd.DataFrame(collapse).to_csv(output/'policy_collapse.csv', index=False)
    pd.concat(differences).to_csv(output/'ppo_disagreement_states.csv', index=False)
    floor_rows = []
    floor = config.environment.min_limit
    for (scenario, policy, seed), group in t.groupby(['scenario', 'policy', 'policy_seed']):
        at_floor = np.isclose(group.credit_limit, floor, atol=.01)
        initial = h[(h.scenario == scenario) & (h.policy == policy) & (h.policy_seed == seed) & (h.month == 0)]
        hits = h[(h.scenario == scenario) & (h.policy == policy) & (h.policy_seed == seed) & np.isclose(h.credit_limit, floor, atol=.01)]
        first = hits.groupby('customer_id').month.min()
        requested = group.requested_action.to_numpy()
        effective = group.effective_action.to_numpy()
        floor_rows.append(dict(scenario=scenario, policy=policy, seed=seed,
            customers=len(initial), reached_fraction=len(first)/len(initial),
            mean_first_floor_month_among_reachers=first.mean(),
            observed_transition_floor_fraction=at_floor.mean(),
            scheduled_episode_floor_fraction=at_floor.sum()/(len(initial)*config.environment.horizon),
            already_at_floor_fraction=np.isclose(initial.credit_limit, floor).mean(),
            partial_minus20=int(((requested < -.199) & (effective > -.199)).sum()),
            no_effect_minus20=int(((requested < -.199) & (abs(effective) < 1e-8)).sum()),
            no_effect_nonhold=int(((abs(requested) > 1e-8) & (abs(effective) < 1e-8)).sum()),
            censored_fraction=float((abs(requested-effective) > 1e-8).mean()),
            blocked_increases=int(group.guardrail_blocked.sum())))
    pd.DataFrame(floor_rows).to_csv(output/'limit_floor.csv', index=False)
    t['effective_bucket'] = np.where(abs(t.effective_action) < 1e-8, 'hold',
        np.where(np.isclose(t.requested_action, t.effective_action, atol=1e-8), 'as_requested', 'partial'))
    t['floor_modified'] = (abs(t.requested_action-t.effective_action)>1e-8) & np.isclose(t.credit_limit, floor)
    t['cap_modified'] = (abs(t.requested_action-t.effective_action)>1e-8) & np.isclose(t.credit_limit, config.environment.max_limit)
    matrix = t.groupby(['scenario', 'policy', 'policy_seed', 'requested_action', 'effective_bucket']).agg(
        count=('month', 'size'), mean_effective_action=('effective_action', 'mean'),
        min_effective_action=('effective_action', 'min'), max_effective_action=('effective_action', 'max'),
        blocked=('guardrail_blocked', 'sum'), floor_modified=('floor_modified', 'sum'), cap_modified=('cap_modified', 'sum')).reset_index()
    matrix.to_csv(output/'action_constraints.csv', index=False)
    return dict(max_reward_identity_error=float(abs(t.reward_error).max()),
                max_net_identity_error=float(abs(t.net_identity_error).max()))


def run(args):
    output, canonical = args.output, args.canonical
    output.mkdir(parents=True, exist_ok=True)
    (output/'draws').mkdir(exist_ok=True)
    config, settings, _ = settings_for(args.profile, 'configs')
    risk_path = canonical/'models/pd/logistic_calibrated.joblib'
    risk = LongitudinalPDModel.load(risk_path)
    severe = settings['guardrails']['severe_delinquency_months']
    policies = {'Static': ConstantAdjustment(config, 1., severe),
                'MyopicEconomic': MyopicEconomic(config, severe_months=severe),
                'AlwaysDecrease20': ConstantAdjustment(config, .8, severe)}
    hashes = {'pd': digest(risk_path)}
    for seed in settings['ppo']['seeds']:
        path = canonical/f'models/ppo_{seed}/selected.zip'
        hashes[f'ppo_{seed}'] = digest(path)
        policies[f'PPO_{seed}'] = SB3Policy(PPO.load(path, device='cpu'))
    expected = json.loads((canonical/'results/model_hashes.json').read_text())
    if hashes != expected:
        raise ValueError('Frozen canonical model hashes do not match')
    source_hashes = {str(p): digest(p) for p in sorted(Path('src/credit_rl').rglob('*.py'))}
    identity = dict(config=asdict(config), settings=settings, draws=args.draws,
                    per_stratum=args.per_stratum, seed=args.seed, hashes=hashes)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Diagnostic settings changed: use a fresh output')
    if args.resume and manifest.exists():
        old_sources = json.loads(manifest.read_text())['sources']
        kernel = lambda data: {k: v for k, v in data.items() if 'experiments' not in k}
        if kernel(old_sources) != kernel(source_hashes):
            raise ValueError('Scientific source changed: cached draws cannot be reused')
    manifest.write_text(json.dumps(dict(signature=signature, **identity, sources=source_hashes,
        estimand='full-state fixed-scenario Q_H under named continuation; not Q*',
        weighting='pooled active visits of Static/MyopicEconomic/AlwaysDecrease20; inverse stratum sampling',
        horizons=HORIZONS, gammas=GAMMAS), indent=2))
    started = perf_counter()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        snapshots, states = sample_states(config, settings, risk, policies, args.per_stratum, args.seed, output)
        # Accounting -> one-step -> multistep; H1 is a prefix of identical CRN rolls.
        all_values, all_gaps, all_planning = [], [], []
        for sid, snap in enumerate(snapshots):
            paths = hypothetical_paths(snap, args.draws, config.environment.horizon, args.seed+100000+sid*args.draws)
            for name, policy in policies.items():
                cache = output/f'draws/state_{sid}_{name}.npz'
                if args.resume and cache.exists():
                    cube = np.load(cache)['values']
                else:
                    cube = rollout(snap, config, risk, policy, paths, config.environment.horizon)
                    np.savez_compressed(cache, values=cube)
                v, g, p = value_tables(snap, config, cube, sid, name, HORIZONS, GAMMAS)
                all_values.append(v)
                all_gaps.append(g)
                all_planning.append(p)
            if sid % 4 == 0:
                print(f'Action values {sid+1}/{len(snapshots)}; {perf_counter()-started:.0f}s', flush=True)
        values, gaps, planning = map(lambda xs: pd.concat(xs, ignore_index=True), (all_values, all_gaps, all_planning))
        dominance, gap_summary = dominance_tables(gaps, states)
        save_tables(output, dict(action_values=values, action_gaps=gaps, action_dominance=dominance,
                                gap_summary=gap_summary, planning_opportunity=planning))
        verification = panel_diagnostics(canonical, output, config)
        sensitivity(args, snapshots, states, config, risk, policies)
    verification['seconds'] = perf_counter()-started
    verification['states'] = len(states)
    verification['model_hashes_unchanged'] = hashes == {'pd': digest(risk_path), **{
        f'ppo_{seed}': digest(canonical/f'models/ppo_{seed}/selected.zip') for seed in settings['ppo']['seeds']}}
    (output/'verification.json').write_text(json.dumps(verification, indent=2))
    from credit_rl.experiments.structural_report import report
    report(output, update_docs=getattr(args, 'update_docs', False))
    print(json.dumps(verification, indent=2), flush=True)


def sensitivity(args, snapshots, states, config, risk, policies):
    """OAT intervention on frozen copies; no fitting, no calibration search.

    Reward coefficients can be rescored exactly on frozen Static continuations.
    Dynamic coefficients require resimulation of the same states and same shocks.
    """
    output = args.output
    records = []
    reward_parameters = dict(annual_percentage_rate=0, fee_rate=1, loss_given_default=2,
                             annual_funding_rate=3, capital_weight=4, constraint_weight=5)
    for parameter, component in reward_parameters.items():
        for factor in (.5, .75, 1., 1.25, 1.5):
            for sid, snap in enumerate(snapshots):
                cube = np.load(output/f'draws/state_{sid}_Static.npz')['values']
                for h in (1, 6, 24):
                    length = min(h, cube.shape[2])
                    samples = ((cube[:, :, :length, 6]+SIGNS[component]*(factor-1)*cube[:, :, :length, component])
                               *(.98**np.arange(length))[None, None, :]).sum(axis=2)
                    stats = action_statistics(samples, admissible(snap, config), config.environment.action_multipliers)
                    records.append(dict(parameter=parameter, factor=factor, state_id=sid, horizon=h,
                        continuation='Static', gamma=.98, weight=states.iloc[sid].weight, draws=len(cube), **stats))
    # A separately declared stratified sub-sample, one state per visitation stratum.
    subset = states.groupby(['scenario', 'source_policy', 'phase'], sort=True).sample(n=1, random_state=args.seed+1)
    subset.to_csv(output/'sensitivity_states.csv', index=False)
    parameters = [('dynamics', 'spend_limit_elasticity'), ('dynamics', 'payment_utilization'),
                  ('dynamics', 'missed_utilization'), ('default', 'utilization'), ('default', 'debt_to_income')]
    for section, parameter in parameters:
        print(f'Sensitivity {section}.{parameter}', flush=True)
        for factor in (.5, .75, 1., 1.25, 1.5):
            original = getattr(config, section)
            changed = replace(config, **{section: replace(original, **{parameter: getattr(original, parameter)*factor})})
            for _, row in subset.iterrows():
                sid = int(row.state_id)
                snap = snapshots[sid]
                paths = hypothetical_paths(snap, args.draws, 24, args.seed+100000+sid*args.draws)
                cache = output/f'draws/sensitivity_{section}_{parameter}_{factor}_{sid}.npz'
                if args.resume and cache.exists():
                    cube = np.load(cache)['values']
                else:
                    cube = rollout(snap, changed, risk, policies['Static'], paths, 6)
                    np.savez_compressed(cache, values=cube)
                for h in (1, 6):
                    length = min(h, cube.shape[2])
                    samples = (cube[:, :, :length, 6]*(.98**np.arange(length))[None, None, :]).sum(axis=2)
                    stats = action_statistics(samples, admissible(snap, config), config.environment.action_multipliers)
                    # Full stratum mass, not the main sample's individual weight.
                    weight = states[(states.scenario == row.scenario) & (states.source_policy == row.source_policy)
                                    & (states.phase == row.phase)].weight.sum()
                    records.append(dict(parameter=parameter, factor=factor, state_id=sid, horizon=h,
                        continuation='Static', gamma=.98, weight=weight, draws=len(cube), **stats))
    pd.DataFrame(records).to_csv(output/'sensitivity.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', type=Path, default=Path('outputs/main/standard'))
    parser.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    parser.add_argument('--output', type=Path, default=Path('outputs/main/structural_diagnostics'))
    parser.add_argument('--draws', type=int, default=128)
    parser.add_argument('--per-stratum', type=int, default=4)
    parser.add_argument('--seed', type=int, default=92801)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--update-docs', action='store_true')
    args = parser.parse_args()
    if args.draws < 8 or args.per_stratum < 1:
        parser.error('draws >= 8 and per-stratum >= 1 required')
    if args.report_only:
        from credit_rl.experiments.structural_report import report
        report(args.output, update_docs=args.update_docs)
    else:
        run(args)


if __name__ == '__main__':
    main()
