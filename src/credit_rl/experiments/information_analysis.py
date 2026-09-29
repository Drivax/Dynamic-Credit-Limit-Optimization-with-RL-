"""Paired Phase B policy values and held-out information diagnostics."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from credit_rl.envs.constraints import effective_limit
from credit_rl.evaluation.policy_engine import PolicySpec, evaluate_policy, aggregate_episodes
from credit_rl.evaluation.policy_statistics import matrix, interval
from credit_rl.policies.baselines import SB3Policy
from credit_rl.policies.decision import MyopicEconomic, ConstantAdjustment
from credit_rl.policies.information import ObservationPlanner, HistoryPlanner, FullStatePlanner, choose
from credit_rl.experiments.information_gap import relative_error

NAMES = dict(F0='ObservationPlanner', F1='WithoutPD', F2='WithoutBehavior',
             F3='History3', F4='History6', F5='HistoryPlanner', Full='FullStatePlanner')


def save(output, name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(output/f'{name}.csv', index=False)
    return frame


def policy_specs(models, config, ppos):
    specs = [PolicySpec('MyopicEconomic', lambda e, s: MyopicEconomic(config)),
             PolicySpec('AlwaysDecrease20', lambda e, s: ConstantAdjustment(config, .8))]
    for kind, item in models.items():
        model = item['model']
        if kind == 'Full':
            factory = lambda e, s, m=model: FullStatePlanner(m, config, e)
            information = 'PRIVILEGED_FULL_STATE'
        elif kind in ('F3', 'F4', 'F5'):
            factory = lambda e, s, m=model, k=kind: HistoryPlanner(m, config, k)
            information = 'OBSERVABLE_HISTORY'
        else:
            factory = lambda e, s, m=model, k=kind: ObservationPlanner(m, config, k)
            information = 'OBSERVABLE_ONLY'
        specs.append(PolicySpec(NAMES[kind], factory, information_set=information))
    specs.extend(PolicySpec('PPO', lambda e, s, m=m: SB3Policy(m), seed=seed) for seed, m in ppos.items())
    return specs


def evaluate(models, groups, config, settings, protocol, risk, ppos, output):
    episodes, actions = [], []
    for macro in ('baseline', 'severe_stress'):
        for spec in policy_specs(models, config, ppos):
            e, h, _ = evaluate_policy(spec, groups[macro, 'test'], config, risk, settings, macro)
            purchases = h[h.month > 0].groupby('customer_id').purchases.sum() if 'purchases' in h else None
            if purchases is None:
                # Simulator history uses spending for realized purchases.
                purchases = h[h.month > 0].groupby('customer_id').spending.sum()
            e['purchases'] = e.customer_id.map(purchases)
            episodes.append(e)
            for action_type in ('requested_action', 'effective_action'):
                counts = h.loc[h.month > 0, action_type].round(8).value_counts()
                actions.extend(dict(scenario=macro, policy=spec.name, seed=spec.seed,
                    action_type=action_type, action=a, count=n, fraction=n/counts.sum()) for a, n in counts.items())
            print(f'Evaluated {macro} {spec.name} {spec.seed}', flush=True)
    episodes = save(output, 'episode_metrics', pd.concat(episodes, ignore_index=True))
    save(output, 'action_distribution', actions)
    rows = []
    for (macro, policy), group in episodes.groupby(['scenario', 'policy']):
        row = dict(scenario=macro, policy=policy, **aggregate_episodes(group), purchases=group.purchases.mean())
        for metric in ('discounted_reward', 'net_economic_value'):
            lo, hi = interval(matrix(group, metric), protocol['bootstrap_repetitions'], protocol['diagnostic_seed'])
            row[metric+'_lower'], row[metric+'_upper'] = lo, hi
        rows.append(row)
    benchmark = save(output, 'benchmark_values', rows)
    gaps = []
    comparisons = [('ObservationPlanner', 'MyopicEconomic', 'observable_planning'),
        ('HistoryPlanner', 'ObservationPlanner', 'history'),
        ('FullStatePlanner', 'HistoryPlanner', 'privileged'),
        ('ObservationPlanner', 'PPO', 'ppo_gap')]
    for macro, group in episodes.groupby('scenario'):
        for policy, reference, label in comparisons:
            for metric in ('discounted_reward', 'net_economic_value'):
                a = matrix(group[group.policy == policy], metric)
                b = matrix(group[group.policy == reference], metric)
                if not a.columns.equals(b.columns):
                    raise ValueError('Unpaired customers')
                delta = a.to_numpy()-b.to_numpy()
                lo, hi = interval(delta, protocol['bootstrap_repetitions'], protocol['diagnostic_seed'])
                gaps.append(dict(scenario=macro, gap=label, policy=policy, reference=reference,
                    metric=metric, difference=delta.mean(), lower=lo, upper=hi,
                    uncertainty='paired customers and PPO seeds; conditional fitted models'))
    save(output, 'information_gap', gaps)
    save(output, 'history_depth', benchmark[benchmark.policy.isin(NAMES.values())])
    return episodes


def weighted_summary(frame, value, protocol):
    v, w = frame[value].to_numpy(), frame.weight.to_numpy()
    order = np.argsort(v)
    quantiles = np.interp([.5, .9, .95], (np.cumsum(w[order])-.5*w[order])/w.sum(), v[order])
    # Cluster bootstrap entire customer contributions; repeated states never independent.
    sums = frame.assign(numerator=v*w).groupby('customer_id')[['numerator', 'weight']].sum().to_numpy()
    rng = np.random.default_rng(protocol['diagnostic_seed'])
    indices = rng.integers(len(sums), size=(protocol['bootstrap_repetitions'], len(sums)))
    draws = sums[indices].sum(1)
    lo, hi = np.quantile(draws[:, 0]/draws[:, 1], [.025, .975])
    return dict(mean=np.average(v, weights=w), median=quantiles[0], p90=quantiles[1], p95=quantiles[2],
                lower=lo, upper=hi, weighted_total=np.dot(v, w), weight=w.sum(), states=len(v),
                customers=len(sums))


def effective_action(snapshot, action, config):
    limit, _ = effective_limit(snapshot.state.credit_limit, snapshot.state.months_delinquent,
        config.environment.action_multipliers[action], config.environment, snapshot.severe_months)
    return round(limit/snapshot.state.credit_limit-1, 8)


def state_diagnostics(data, q, models, config, protocol, ppos, output):
    ids = np.flatnonzero(data['states'].role.eq('test'))
    observations = np.array([data['prefixes'][i][0][-1] for i in ids])
    actions = {NAMES[k]: np.array([choose(v, o, config) for v, o in zip(
        item['model'].predict(data['features'][k][ids]), observations)]) for k, item in models.items()}
    actions['MyopicEconomic'] = np.array([MyopicEconomic(config).act(o) for o in observations])
    actions['AlwaysDecrease20'] = np.array([ConstantAdjustment(config, .8).act(o) for o in observations])
    for seed, model in ppos.items():
        actions[f'PPO_{seed}'] = model.predict(observations, deterministic=True)[0]
    rows, confusions, errors = [], [], []
    half = q.shape[1]//2
    for kind, item in models.items():
        h = protocol['horizons'].index(item['horizon'])
        pred = item['model'].predict(data['features'][kind][ids])
        truth = q[ids, half:, h].mean(1)
        errors.append(dict(feature_set=kind, horizon=item['horizon'],
            test_relative_mse=relative_error(pred, truth, data['states'].iloc[ids].weight),
            test_relative_rmse=np.sqrt(relative_error(pred, truth, data['states'].iloc[ids].weight)),
            target_relative_rms=np.sqrt(np.average(np.mean((truth-truth[:, 2, None])**2, axis=1),
                weights=data['states'].iloc[ids].weight))))
    save(output, 'planner_fit_diagnostics', errors)
    for j, sid in enumerate(ids):
        state = data['states'].iloc[sid].to_dict()
        snap = data['snapshots'][sid]
        full = int(actions['FullStatePlanner'][j])
        for h, horizon in enumerate(protocol['horizons']):
            teacher = choose(q[sid, :half, h].mean(0), observations[j], config)
            heldout = q[sid, half:, h].mean(0)
            for name, aa in actions.items():
                a = int(aa[j])
                rows.append(dict(**state, policy=name, horizon=horizon, action=a, full_action=full,
                    teacher_action=teacher, effective_action=effective_action(snap, a, config),
                    requested_agreement=int(a == full),
                    effective_agreement=int(effective_action(snap, a, config) == effective_action(snap, full, config)),
                    regret_full=heldout[full]-heldout[a], regret_teacher=heldout[teacher]-heldout[a],
                    teacher_value=heldout[teacher], policy_value=heldout[a]))
        for name in ('ObservationPlanner', 'HistoryPlanner', 'FullStatePlanner'):
            for seed in ppos:
                for mode in ('requested', 'effective'):
                    a, b = int(actions[name][j]), int(actions[f'PPO_{seed}'][j])
                    if mode == 'effective':
                        a, b = effective_action(snap, a, config), effective_action(snap, b, config)
                    confusions.append(dict(policy=name, ppo_seed=seed, scenario=state['scenario'],
                        mode=mode, benchmark_action=a, ppo_action=b, weight=state['weight']))
    frame = save(output, 'state_regret', rows)
    keys = ['policy', 'ppo_seed', 'scenario', 'mode', 'benchmark_action', 'ppo_action']
    save(output, 'action_confusion', pd.DataFrame(confusions).groupby(keys, as_index=False).weight.sum())
    summaries, agreements = [], []
    for variable in ('all', 'pd_bucket', 'utilization_bucket', 'macro_regime', 'remaining_horizon'):
        f = frame.assign(bucket='all' if variable == 'all' else frame[variable].astype(str))
        for (scenario, policy, horizon, bucket), group in f.groupby(['scenario', 'policy', 'horizon', 'bucket']):
            tags = dict(scenario=scenario, policy=policy, horizon=horizon, stratification=variable, bucket=bucket)
            for reference in ('full', 'teacher'):
                summaries.append(dict(**tags, reference=reference,
                    **weighted_summary(group, 'regret_'+reference, protocol)))
            agreements.append(dict(**tags, requested=np.average(group.requested_agreement, weights=group.weight),
                effective=np.average(group.effective_agreement, weights=group.weight)))
    save(output, 'planner_regret', summaries)
    save(output, 'policy_agreement', agreements)
    # Only source PPO_101 actually generated these visits. Other seeds are same-state diagnostics.
    actual = frame[(frame.source_policy == 'PPO') & (frame.policy == 'PPO_101') & (frame.horizon == 24)]
    counters = []
    for (scenario, pd_bucket, utilization_bucket), group in actual.groupby(['scenario', 'pd_bucket', 'utilization_bucket']):
        counters.append(dict(scenario=scenario, pd_bucket=pd_bucket, utilization_bucket=utilization_bucket,
            **weighted_summary(group, 'regret_teacher', protocol)))
    for scenario, group in actual.groupby('scenario'):
        counters.append(dict(scenario=scenario, pd_bucket='all', utilization_bucket='all',
            **weighted_summary(group, 'regret_teacher', protocol)))
    save(output, 'counterfactual_regret', counters)
    return frame


def feature_group(name):
    if name.startswith(('lag', 'history_', 'initial_')):
        return 'historical trends'
    if name == 'predicted_pd':
        return 'PD'
    if 'utilization' in name or 'balance' in name:
        return 'utilization/balance'
    if name.startswith('macro'):
        return 'macro'
    if 'income' in name:
        return 'income'
    if 'delinquen' in name or 'late_payment' in name:
        return 'delinquency'
    if 'payment' in name:
        return 'repayment'
    if 'score' in name:
        return 'behavioral score'
    if 'limit' in name:
        return 'current limit'
    return 'other current'


def information_diagnostics(data, q, models, protocol, output):
    states = data['states']
    train, test = (states.role.eq(r).to_numpy() for r in ('train', 'test'))
    latent_rows, importances = [], []
    for kind in ('F0', 'F5'):
        x = data['features'][kind]
        for target in [c for c in states if c.startswith('latent_')]:
            model = HistGradientBoostingRegressor(max_leaf_nodes=7, max_iter=protocol['boosting_iterations'],
                min_samples_leaf=protocol['minimum_leaf'], early_stopping=False,
                random_state=protocol['diagnostic_seed'])
            model.fit(x[train], states.loc[train, target], sample_weight=states.weight[train])
            pred = model.predict(x[test])
            truth = states.loc[test, target]
            latent_rows.append(dict(feature_set=kind, latent=target,
                r2=r2_score(truth, pred, sample_weight=states.weight[test]),
                rmse=np.sqrt(np.average((truth-pred)**2, weights=states.weight[test]))))
        item = models[kind]
        h = protocol['horizons'].index(item['horizon'])
        truth = q[test, q.shape[1]//2:, h].mean(1)
        model = item['model']
        baseline = relative_error(model.predict(x[test]), truth, states.weight[test])
        rng = np.random.default_rng(protocol['diagnostic_seed'])
        groups = {}
        for i, name in enumerate(data['names'][kind]):
            groups.setdefault(feature_group(name), []).append(i)
        for group, indices in groups.items():
            for repetition in range(protocol['permutation_repetitions']):
                permuted = x[test].copy()
                order = rng.permutation(len(permuted))
                permuted[:, indices] = permuted[order][:, indices]
                importances.append(dict(feature_set=kind, group=group, repetition=repetition,
                    relative_mse_increase=relative_error(model.predict(permuted), truth, states.weight[test])-baseline))
    latent = pd.DataFrame(latent_rows)
    base = latent[latent.feature_set == 'F0'].set_index('latent')
    latent['r2_gain_over_current'] = latent.r2-latent.latent.map(base.r2)
    latent['rmse_reduction_over_current'] = latent.latent.map(base.rmse)-latent.rmse
    save(output, 'latent_predictability', latent)
    save(output, 'feature_importance', importances)


def analyze(data, q, models, groups, config, settings, protocol, risk, ppos, output):
    evaluate(models, groups, config, settings, protocol, risk, ppos, output)
    state_diagnostics(data, q, models, config, protocol, ppos, output)
    information_diagnostics(data, q, models, protocol, output)
