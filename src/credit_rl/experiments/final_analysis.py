"""Paired final contrasts, world risk and descriptive public-state heterogeneity."""
import numpy as np
import pandas as pd

from credit_rl.evaluation.policy_statistics import matrix, interval
from credit_rl.experiments.ppo_learning import state_dependence
from .final_evaluation import context, output_for
from .final_panels import POLICIES
from .policy_initialization import paired_visit_interval

METRICS = ('net_economic_value', 'discounted_reward', 'credit_loss', 'defaulted')
CONTRASTS = (('C1', 'BCInitPPO', 'CanonicalPPO'), ('C2', 'BCRegularizedPPO', 'BCInitPPO'),
             ('C3', 'BCRegularizedPPO', 'AlwaysDecrease20'), ('C4', 'BCRegularizedPPO', 'MyopicEconomic'))


def comparison(values, repetitions):
    lower, upper = interval(values, repetitions, 1064921)
    return dict(mean=float(np.mean(values)), lower=float(lower), upper=float(upper),
                seeds=values.shape[0], customers=values.shape[1])


def worst_interval(left, right, repetitions):
    """Bootstrap minimum of world means, retaining paired seed/customer/world draws."""
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError('Paired seed/customer/world tensors required')
    rng = np.random.default_rng(1064921)
    values = []
    for _ in range(repetitions):
        seeds = rng.integers(left.shape[0], size=left.shape[0])
        customers = rng.integers(left.shape[1], size=left.shape[1])
        ix = np.ix_(seeds, customers, np.arange(left.shape[2]))
        values.append(left[ix].mean((0, 1)).min()-right[ix].mean((0, 1)).min())
    lo, hi = np.quantile(values, [.025, .975])
    return dict(mean=float(left.mean((0, 1)).min()-right.mean((0, 1)).min()),
                lower=float(lo), upper=float(hi), seeds=left.shape[0], customers=left.shape[1])


def behavior_interval(left, right, metric, repetitions):
    """Resample paired customers and seeds, then recompute nonlinear visit statistics."""
    def value(x):
        totals = x.sum(axis=1)
        if metric == 'diversity':
            return np.mean(1-totals[:, :5].max(axis=1)/totals[:, 6])
        return np.mean(totals[:, 5]/totals[:, 6])
    rng = np.random.default_rng(1064921)
    draws = []
    for _ in range(repetitions):
        ix = np.ix_(rng.integers(left.shape[0], size=left.shape[0]),
                    rng.integers(left.shape[1], size=left.shape[1]), np.arange(7))
        draws.append(value(left[ix])-value(right[ix]))
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(mean=value(left)-value(right), lower=lo, upper=hi,
                seeds=left.shape[0], customers=left.shape[1])


def conditional_interval(frame, column, repetitions):
    """Eligible visits need not occur for every seed/customer; empty cells weigh zero."""
    sums = frame.pivot_table(index='seed', columns='customer_id', values=column, aggfunc='sum').fillna(0).to_numpy()
    counts = frame.pivot_table(index='seed', columns='customer_id', values=column, aggfunc='count').fillna(0).to_numpy()
    rng = np.random.default_rng(1064921)
    draws = []
    for _ in range(repetitions):
        ix = np.ix_(rng.integers(len(sums), size=len(sums)),
                    rng.integers(sums.shape[1], size=sums.shape[1]))
        denominator = counts[ix].sum()
        if denominator:
            draws.append(sums[ix].sum()/denominator)
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(mean=sums.sum()/counts.sum(), low=lo, high=hi)


def analyze(profile):
    _, _, _, p, _ = context(profile)
    output = output_for(profile)
    episode_frames, behavior, actions, heterogeneity, customer_behavior = [], [], [], [], []
    customer_heterogeneity = []
    for file in sorted((output/'evaluation').rglob('episodes.csv')):
        seed, policy, world, budget = file.parts[-2], file.parts[-3], file.parts[-4], int(file.parts[-5])
        tags = dict(budget=budget, policy=policy, policy_seed=int(seed), world=world)
        e = pd.read_csv(file).assign(budget=budget, world=world)
        episode_frames.append(e)
        v = pd.read_csv(file.with_name('visits.csv.gz'))
        obs = v[[f'o{i}' for i in range(21)]].to_numpy()
        requested = v.action.to_numpy(int)
        for customer, group in v.groupby('customer_id'):
            counts = np.bincount(group.action.astype(int), minlength=5)
            customer_behavior.append(dict(**tags, customer_id=customer,
                **{f'a{i}': int(n) for i, n in enumerate(counts)},
                regret_sum=group.teacher_regret.sum(), visits=len(group)))
        behavior.append(dict(**tags, **state_dependence(requested, obs),
            teacher_regret=v.teacher_regret.mean(), agreement=v.agreement.mean(),
            contraction_share=(requested == 0).mean(), hold_share=(requested == 2).mean(),
            increase_share=(requested > 2).mean(),
            effective_increase=e.increases.sum()/e.steps.sum(), effective_hold=e.no_change.sum()/e.steps.sum()))
        v['pd_bucket'] = np.searchsorted([.2, .6], v.o10)
        v['utilization_bucket'] = np.searchsorted([1/3, .5], v.o3)
        v['income_bucket'] = np.searchsorted([2000/17000, 4000/19000], v.o7)
        v['horizon_bucket'] = np.searchsorted([1/3, 2/3], v.o0)
        # Initial observable buckets define customer cohorts; unlike visit buckets,
        # they are fixed before the policy changes occupancy or survival.
        bucket_names = ['pd_bucket', 'utilization_bucket', 'income_bucket']
        initial = v.sort_values('month').drop_duplicates('customer_id')[['customer_id', *bucket_names]]
        customer_panel = e.merge(initial, on='customer_id', validate='one_to_one')
        for bucket in bucket_names:
            for level, group in customer_panel.groupby(bucket):
                customer_heterogeneity.append(dict(**tags, variable=bucket, bucket=level,
                    customers=len(group), net_economic_value=group.net_economic_value.mean(),
                    discounted_reward=group.discounted_reward.mean(), credit_loss=group.credit_loss.mean(),
                    default_rate=group.defaulted.mean(), mean_exposure=group.mean_balance.mean()))
        for region, group in v.groupby(['pd_bucket', 'utilization_bucket']):
            counts = np.bincount(group.action.astype(int), minlength=5)
            actions.extend(dict(**tags, pd_bucket=region[0], utilization_bucket=region[1],
                action=a, count=int(count), visits=len(group), share=count/len(group)) for a, count in enumerate(counts))
        for bucket in ('pd_bucket', 'utilization_bucket', 'income_bucket', 'horizon_bucket'):
            for level, group in v.groupby(bucket):
                heterogeneity.append(dict(**tags, variable=bucket, bucket=level, visits=len(group),
                    customers=group.customer_id.nunique(), economic_value_per_visit=group.economic_value.mean(),
                    reward_per_visit=group.reward.mean(), teacher_regret=group.teacher_regret.mean(),
                    increase_share=(group.action > 2).mean()))
    episodes = pd.concat(episode_frames, ignore_index=True)
    episodes.to_csv(output/'episodes.csv', index=False)
    b = pd.DataFrame(behavior)
    cb = pd.DataFrame(customer_behavior)
    cb.to_csv(output/'behavior_by_customer.csv', index=False)
    b.to_csv(output/'behavior_by_seed.csv', index=False)
    pd.DataFrame(actions).to_csv(output/'conditional_actions.csv', index=False)
    pd.DataFrame(heterogeneity).to_csv(output/'heterogeneity.csv', index=False)
    pd.DataFrame(customer_heterogeneity).to_csv(output/'customer_heterogeneity.csv', index=False)
    rows, comparisons, robustness = [], [], []
    reps = p['bootstrap_repetitions']
    for budget, panel in episodes.groupby('budget'):
        for (policy, world), group in panel.groupby(['policy', 'world']):
            selected = b[(b.budget == budget) & (b.policy == policy) & (b.world == world)]
            row = dict(budget=budget, policy=policy, world=world, customers=group.customer_id.nunique(),
                seeds=group.policy_seed.nunique(), mean_exposure=group.mean_balance.mean(),
                mean_limit=group.mean_limit.mean(),
                loss_cvar95=group.credit_loss.nlargest(max(1, int(np.ceil(.05*len(group))))).mean())
            for metric in METRICS:
                values = matrix(group, metric).to_numpy()
                result = comparison(values, reps)
                row.update({metric: result['mean'], metric+'_lower': result['lower'], metric+'_upper': result['upper']})
            for metric in ('diversity', 'teacher_regret', 'agreement', 'contraction_share', 'hold_share', 'increase_share'):
                row[metric] = selected[metric].mean()
            rows.append(row)
        for world, frame in panel.groupby('world'):
            for contrast, left, right in CONTRASTS:
                for metric in METRICS:
                    difference = matrix(frame[frame.policy == left], metric)-matrix(frame[frame.policy == right], metric)
                    comparisons.append(dict(budget=budget, contrast=contrast, left=left, right=right, world=world,
                        metric=metric, **comparison(difference.to_numpy(), reps)))
                def behavior_tensor(policy):
                    group = cb[(cb.budget == budget) & (cb.world == world) & (cb.policy == policy)]
                    return np.stack([matrix(group, column).to_numpy()
                        for column in [*[f'a{i}' for i in range(5)], 'regret_sum', 'visits']], axis=2)
                for metric in ('diversity', 'teacher_regret'):
                    comparisons.append(dict(budget=budget, contrast=contrast, left=left, right=right, world=world,
                        metric=metric, **behavior_interval(behavior_tensor(left), behavior_tensor(right), metric, reps)))
        for contrast, left, right in CONTRASTS:
            for metric in METRICS:
                effects = {}
                for world in ('nominal', 'severe_stress'):
                    frame = panel[panel.world == world]
                    effects[world] = matrix(frame[frame.policy == left], metric)-matrix(frame[frame.policy == right], metric)
                comparisons.append(dict(budget=budget, contrast=contrast, left=left, right=right,
                    world='stress_minus_baseline', metric=metric,
                    **comparison((effects['severe_stress']-effects['nominal']).to_numpy(), reps)))
        for policy, group in panel.groupby('policy'):
            for metric in METRICS[:2]:
                means = group.groupby('world')[metric].mean()
                robustness.append(dict(budget=budget, policy=policy, metric=metric,
                    mean_world=means.mean(), worst_world=means.min(), worst_world_name=means.idxmin(),
                    world_sd=means.std(ddof=0), worst_degradation=means.min()-means['nominal']))
            for world in sorted(set(group.world)-{'nominal'}):
                for metric in METRICS:
                    difference = matrix(group[group.world == world], metric)-matrix(group[group.world == 'nominal'], metric)
                    comparisons.append(dict(budget=budget, contrast='C5' if policy == 'BCRegularizedPPO' else 'exploratory_shift',
                        left=policy, right=policy, world=world+'_minus_nominal', metric=metric,
                        **comparison(difference.to_numpy(), reps)))
        def tensor(policy, metric):
            group = panel[panel.policy == policy]
            return np.stack([matrix(group[group.world == world], metric).to_numpy()
                             for world in sorted(group.world.unique())], axis=2)
        for policy in POLICIES:
            if policy == 'AlwaysDecrease20':
                continue
            for metric in METRICS[:2]:
                comparisons.append(dict(budget=budget, contrast='C3' if policy == 'BCRegularizedPPO' else 'exploratory_complexity',
                    left=policy, right='AlwaysDecrease20', world='worst_world', metric=metric,
                    **worst_interval(tensor(policy, metric), tensor('AlwaysDecrease20', metric), reps)))
    pd.DataFrame(rows).to_csv(output/'main_table.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output/'paired_comparisons.csv', index=False)
    pd.DataFrame(robustness).to_csv(output/'robustness.csv', index=False)
    preserve = pd.read_csv(output/'preservation_visits.csv.gz')
    summary = []
    for key, group in preserve.groupby(['experiment_id', 'seed', 'scenario', 'timesteps']):
        initial = preserve[(preserve.experiment_id == key[0]) & (preserve.seed == key[1]) &
                           (preserve.scenario == key[2]) & (preserve.timesteps == 0)].set_index('state_id')
        group = group.set_index('state_id')
        initial = initial.loc[group.index]
        eligible = (initial.effective_agreement & (initial.teacher_action >= 3))
        summary.append(dict(policy=key[0], policy_seed=key[1], scenario=key[2], timesteps=key[3],
            diversity=1-group.action.value_counts(normalize=True).max(), teacher_regret=group.teacher_regret.mean(),
            agreement=group.effective_agreement.mean(), preservation_loss=(group.teacher_regret-initial.teacher_regret).mean(),
            increase_survival=group.loc[eligible, 'effective_agreement'].mean() if eligible.any() else np.nan,
            eligible_increase_states=int(eligible.sum())))
    pd.DataFrame(summary).to_csv(output/'preservation.csv', index=False)
    preservation_comparisons = []
    for budget in p['budgets']:
        for scenario in ('baseline', 'severe_stress'):
            panel = preserve[(preserve.timesteps == budget) & (preserve.scenario == scenario)]
            left = panel[panel.experiment_id == 'BCRegularizedPPO']
            right = panel[panel.experiment_id == 'BCInitPPO']
            joined = left.merge(right, on=['seed', 'state_id', 'customer_id'], suffixes=('_left', '_right'), validate='one_to_one')
            joined['regret_difference'] = joined.teacher_regret_left-joined.teacher_regret_right
            result = paired_visit_interval(joined, 'regret_difference', reps)
            preservation_comparisons.append(dict(budget=budget, scenario=scenario, contrast='C2',
                metric='fixed_panel_teacher_regret_difference', **result,
                seeds=joined.seed.nunique(), customers=joined.customer_id.nunique()))
            initial = preserve[(preserve.timesteps == 0) & (preserve.scenario == scenario) &
                               (preserve.experiment_id == 'BCInitPPO')]
            eligible = initial[initial.effective_agreement & (initial.teacher_action >= 3)][['seed', 'state_id']]
            joined = joined.merge(eligible, on=['seed', 'state_id'], validate='one_to_one')
            if len(joined):
                joined['survival_difference'] = joined.effective_agreement_left.astype(float)-joined.effective_agreement_right.astype(float)
                result = conditional_interval(joined, 'survival_difference', reps)
                preservation_comparisons.append(dict(budget=budget, scenario=scenario, contrast='C2',
                    metric='initial_increase_survival_difference', **result,
                    seeds=joined.seed.nunique(), customers=joined.customer_id.nunique()))
    pd.DataFrame(preservation_comparisons).to_csv(output/'preservation_comparisons.csv', index=False)
    cf = pd.read_csv(output/'counterfactual.csv')
    cf_rows = []
    for keys, group in cf.groupby(['budget', 'policy', 'world']):
        for region in ('all', 'pd_bucket', 'utilization_bucket', 'horizon_bucket'):
            groups = [('all', group)] if region == 'all' else group.groupby(region)
            for bucket, g in groups:
                cf_rows.append(dict(budget=keys[0], policy=keys[1], world=keys[2], region=region, bucket=bucket,
                    mean=g.regret.mean(), median=g.regret.median(), p90=g.regret.quantile(.9), p95=g.regret.quantile(.95),
                    beneficial_share=g.beneficial.mean(), harmful_share=g.harmful.mean(), states=g.state_id.nunique(),
                    deviation_gain=g.teacher_deviation_gain.mean()))
    pd.DataFrame(cf_rows).to_csv(output/'counterfactual_summary.csv', index=False)
    ope = pd.read_csv(output/'ope_estimates.csv')
    ope.groupby(['policy', 'overlap', 'estimator']).agg(bias=('bias', 'mean'),
        rmse=('squared_error', lambda x: np.sqrt(x.mean())), coverage=('covered', 'mean'),
        valid_fraction=('valid', 'mean'), mean_ess=('ESS', 'mean'),
        mean_estimate=('estimate', 'mean'), truth=('truth', 'mean'), truth_se=('truth_se', 'mean'),
        zero_weight_fraction=('zero_weight_fraction', 'mean')).reset_index().to_csv(output/'ope_summary.csv', index=False)
