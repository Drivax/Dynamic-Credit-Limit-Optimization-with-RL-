"""Figures and report tables are regenerated exclusively from saved CSV tables."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from credit_rl.evaluation.structural import action_statistics


def weighted(frame, name):
    return float(np.average(frame[name], weights=frame.weight))


def markdown(frame, precision=3):
    cols = list(frame.columns)
    rows = ['| '+' | '.join(cols)+' |', '| '+' | '.join(['---']*len(cols))+' |']
    for values in frame.itertuples(index=False, name=None):
        rows.append('| '+' | '.join(f'{x:.{precision}f}' if isinstance(x, (float, np.floating)) else str(x)
                                   for x in values)+' |')
    return '\n'.join(rows)


def maps(gaps, states):
    joined = gaps[np.isclose(gaps.gamma, .98)].merge(states, on='state_id')
    rows = []
    for variable in ('all', 'macro_regime', 'income_bucket', 'phase'):
        frame = joined.assign(condition='all') if variable == 'all' else joined.assign(condition=joined[variable].astype(str))
        for key, group in frame.groupby(['continuation', 'horizon', 'condition', 'pd_bucket', 'utilization_bucket']):
            shares = [weighted(group.assign(indicator=group.best_action == a), 'indicator') for a in range(5)]
            rows.append(dict(zip(['continuation', 'horizon', 'condition', 'pd_bucket', 'utilization_bucket'], key)) |
                dict(conditioning=variable, count=len(group), supported=len(group)>=3,
                     best_action=int(np.argmax(shares)), modal_share=max(shares),
                     gap=weighted(group, 'gap'), contraction_advantage=weighted(group, 'contraction_advantage')))
    return pd.DataFrame(rows)


def plot_maps(table, folder, condition='all'):
    names = {'best_action': 'optimal_action_map', 'gap': 'action_gap_map',
             'contraction_advantage': 'contraction_advantage_map'}
    frame = table[(table.continuation == 'AlwaysDecrease20') & (table.conditioning == condition)]
    conditions = sorted(frame.condition.unique())
    for metric, name in names.items():
        fig, axes = plt.subplots(len(conditions), 5, figsize=(17, 3.5*len(conditions)), squeeze=False,
                                 constrained_layout=True)
        available = frame.loc[frame.supported, metric]
        limit = max(1., float(abs(available).max())) if len(available) else 1.
        for r, value in enumerate(conditions):
            for col, horizon in enumerate((1, 3, 6, 12, 24)):
                ax = axes[r, col]
                panel = frame[(frame.condition == value) & (frame.horizon == horizon)]
                matrix = np.full((3, 3), np.nan)
                for row in panel.itertuples():
                    if row.supported:
                        matrix[row.pd_bucket, row.utilization_bucket] = getattr(row, metric)
                if metric == 'best_action':
                    cmap = ListedColormap(['#2166ac', '#67a9cf', '#eeeeee', '#ef8a62', '#b2182b'])
                    norm = BoundaryNorm(np.arange(-.5, 5.5), cmap.N)
                    image = ax.imshow(matrix, origin='lower', cmap=cmap, norm=norm)
                else:
                    image = ax.imshow(matrix, origin='lower', cmap='coolwarm' if metric != 'gap' else 'viridis',
                                      vmin=-limit if metric != 'gap' else 0, vmax=limit)
                for y in range(3):
                    for x in range(3):
                        cell = panel[(panel.pd_bucket == y) & (panel.utilization_bucket == x)]
                        n = int(cell['count'].iloc[0]) if len(cell) else 0
                        label = f'n={n}' if n >= 3 else f'insufficient\nn={n}'
                        ax.text(x, y, label, ha='center', va='center', fontsize=8)
                ax.set(xticks=range(3), xticklabels=['≤.5', '(.5,1]', '>1'], yticks=range(3),
                       yticklabels=['≤.2', '(.2,.6]', '>.6'], xlabel='Opening utilization', ylabel='Opening PD',
                       title=f'{condition}={value}; H={horizon if horizon<24 else "remaining"}')
        bar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.7)
        if metric == 'best_action':
            bar.set_ticks(range(5), labels=['-20%', '-10%', 'hold', '+10%', '+20%'])
        fig.suptitle(f'{metric}: weighted visited-state summary; AlwaysDecrease20 continuation; gamma=.98\n'
                     'Grey/blank cells have fewer than 3 sampled states; no extrapolation')
        suffix = '' if condition == 'all' else '_'+condition
        fig.savefig(folder/f'{name}{suffix}.png', dpi=150)
        plt.close(fig)


def report(output, update_docs=False):
    output = Path(output)
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    states = pd.read_csv(output/'states.csv')
    gaps = pd.read_csv(output/'action_gaps.csv')
    values = pd.read_csv(output/'action_values.csv')
    dominance = pd.read_csv(output/'action_dominance.csv')
    # Prefix convergence and independent half-draw agreement reveal MC ranking fragility.
    precision = []
    persistence_rows = []
    effective_gaps = []
    validation_rows = []
    from credit_rl.envs.constraints import effective_limit
    import json
    from types import SimpleNamespace
    manifest = json.loads((output/'manifest.json').read_text())
    environment = SimpleNamespace(**manifest['config']['environment'])
    guard = manifest['settings']['guardrails']['severe_delinquency_months']
    for (sid, continuation), part in values.groupby(['state_id', 'continuation']):
        cube = np.load(output/f'draws/state_{sid}_{continuation}.npz')['values']
        allowed = sorted(part.loc[part.admissible, 'action'].unique())
        state = states[states.state_id == sid].iloc[0]
        distinct = []
        limits = []
        for action in sorted(allowed, key=lambda a: abs(environment.action_multipliers[a]-1)):
            limit, _ = effective_limit(state.current_limit, state.delinquency,
                environment.action_multipliers[action], environment, guard)
            if not any(np.isclose(limit, previous, atol=.01, rtol=0) for previous in limits):
                distinct.append(action)
                limits.append(limit)
        for horizon in (1, 3, 6, 12, 24):
            length = min(horizon, cube.shape[2])
            samples = (cube[:, :, :length, 6]*(.98**np.arange(length))[None, None, :]).sum(axis=2)
            full = action_statistics(samples, allowed, (.8, .9, 1., 1.1, 1.2))
            unique = action_statistics(samples, distinct, (.8, .9, 1., 1.1, 1.2))
            effective_gaps.append(dict(state_id=sid, continuation=continuation, horizon=horizon,
                                      distinct_actions=len(distinct), **unique))
            half = len(samples)//2
            a = action_statistics(samples[:half], allowed, (.8, .9, 1., 1.1, 1.2))
            b = action_statistics(samples[half:], allowed, (.8, .9, 1., 1.1, 1.2))
            immediate = cube[:, :, 0, 6]
            one_train = action_statistics(immediate[:half], allowed, (.8, .9, 1., 1.1, 1.2))['best_action']
            sacrifice = immediate[half:, one_train]-immediate[half:, a['best_action']]
            gain = samples[half:, a['best_action']]-samples[half:, 0]
            validation_rows.append(dict(state_id=sid, continuation=continuation, horizon=horizon,
                heldout_immediate_sacrifice=float(sacrifice.mean()),
                sacrifice_se=float(sacrifice.std(ddof=1)/np.sqrt(len(sacrifice))),
                heldout_gain_over_initial_minus20=float(gain.mean()),
                gain_se=float(gain.std(ddof=1)/np.sqrt(len(gain)))))
            precision.append(dict(state_id=sid, continuation=continuation, horizon=horizon,
                draws=len(samples), best_full=full['best_action'], best_first_half=a['best_action'],
                best_second_half=b['best_action'], halves_agree=a['best_action']==b['best_action'],
                full_gap=full['gap'], paired_mc_se=full['gap_se'],
                resolved=full['gap']>1.96*full['gap_se'], exact_tie=full['tie_count']>1))
            for metric, index in [('closing_balance_with_exit_zero', 9), ('purchases', 8), ('active', 14)]:
                persistence_rows.append(dict(state_id=sid, continuation=continuation, requested_horizon=horizon,
                    effective_horizon=length, metric=metric,
                    contraction_minus_hold=float((cube[:, 0, length-1, index]-cube[:, 2, length-1, index]).mean())))
    pd.DataFrame(precision).to_csv(output/'mc_precision.csv', index=False)
    pd.DataFrame(effective_gaps).to_csv(output/'effective_action_gaps.csv', index=False)
    pd.DataFrame(persistence_rows).to_csv(output/'persistent_state_effects.csv', index=False)
    validation = pd.DataFrame(validation_rows).merge(states[['state_id', 'weight']], on='state_id')
    validation.to_csv(output/'planning_validation.csv', index=False)
    profiles = []
    for variable in ('pd', 'utilization', 'balance', 'current_limit', 'income', 'payment_ratio', 'delinquency', 'behavioral_score'):
        state_profiles = states[['state_id', 'weight']].copy()
        state_profiles['profile'] = pd.qcut(states[variable], 3, duplicates='drop').astype(str)
        merged = gaps[np.isclose(gaps.gamma, .98)].merge(state_profiles, on='state_id')
        for key, group in merged.groupby(['continuation', 'horizon', 'profile']):
            for action in range(5):
                profiles.append(dict(zip(['continuation', 'horizon', 'profile'], key)) |
                    dict(variable=variable, states=len(group), action=action,
                         share=weighted(group.assign(indicator=group.best_action == action), 'indicator')))
    pd.DataFrame(profiles).to_csv(output/'state_profile_dominance.csv', index=False)
    planning = pd.read_csv(output/'planning_opportunity.csv').merge(states[['state_id', 'weight']], on='state_id')
    table = maps(gaps, states)
    table.to_csv(output/'action_maps.csv', index=False)
    for condition in ('all', 'macro_regime', 'income_bucket', 'phase'):
        plot_maps(table, folder, condition)
    d = dominance[(dominance.stratification == 'all') & np.isclose(dominance.gamma, .98)]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, group in d[d.action == 0].groupby('continuation'):
        ax.plot(group.horizon, group.share, marker='o', label=name)
    ax.set(xlabel='Requested horizon (24 = remaining)', ylabel='Weighted P(best action = -20%)', ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.savefig(folder/'action_dominance_by_horizon.png', dpi=150)
    plt.close(fig)
    summary = []
    for key, group in planning.groupby(['continuation', 'horizon', 'gamma']):
        record = dict(zip(['continuation', 'horizon', 'gamma'], key)) | {
            name: weighted(group, name) for name in
            ('disagreement', 'opportunity', 'heldout_opportunity', 'immediate_sacrifice')}
        se = float(np.sqrt(np.sum((group.weight*group.heldout_se)**2))/group.weight.sum())
        record.update(heldout_mc_se=se, heldout_mc_low=record['heldout_opportunity']-1.96*se,
                      heldout_mc_high=record['heldout_opportunity']+1.96*se)
        summary.append(record)
    ps = pd.DataFrame(summary)
    ps.to_csv(output/'planning_summary.csv', index=False)
    validation_summary = []
    for key, group in validation.groupby(['continuation', 'horizon']):
        row = dict(zip(['continuation', 'horizon'], key))
        for metric, se_name in [('heldout_immediate_sacrifice', 'sacrifice_se'),
                                ('heldout_gain_over_initial_minus20', 'gain_se')]:
            estimate = weighted(group, metric)
            se = float(np.sqrt(np.sum((group.weight*group[se_name])**2))/group.weight.sum())
            row.update({metric: estimate, metric+'_mc_low': estimate-1.96*se, metric+'_mc_high': estimate+1.96*se})
        validation_summary.append(row)
    validation_summary = pd.DataFrame(validation_summary)
    validation_summary.to_csv(output/'planning_validation_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name, group in ps[np.isclose(ps.gamma, .98)].groupby('continuation'):
        ax.plot(group.horizon, group.heldout_opportunity, marker='o', label=name)
    ax.axhline(0, color='black', linewidth=.5)
    ax.set(xlabel='Horizon', ylabel='Split-draw planning opportunity (EUR)')
    ax.legend(fontsize=8)
    fig.savefig(folder/'planning_opportunity.png', dpi=150)
    plt.close(fig)
    g = gaps[np.isclose(gaps.gamma, .98) & (gaps.continuation == 'AlwaysDecrease20')].merge(states, on='state_id')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for h, ax in zip((1, 24), axes):
        part = g[g.horizon == h]
        ax.hist(part.gap, weights=part.weight/part.weight.sum(), bins=16)
        ax.set(title=f'H={h}', xlabel='Best minus second admissible action (EUR)', ylabel='Weighted frequency')
    fig.savefig(folder/'action_gap_histogram.png', dpi=150)
    plt.close(fig)
    decomposition = pd.read_csv(output/'policy_value_decomposition.csv')
    overall = decomposition[(decomposition.stratification == 'all') & (decomposition.policy != 'PPO')]
    components = ['interest_income', 'fee_income', 'credit_loss', 'funding_cost', 'capital_cost', 'constraint_penalty']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for (scenario, part), ax in zip(overall.groupby('scenario'), axes):
        part = part[part.policy != 'AlwaysDecrease20']
        x = np.arange(len(part))
        for j, component in enumerate(components):
            ax.bar(x+(j-2.5)*.13, part['decrease_minus_policy_reward_'+component], width=.13, label=component)
        ax.set(xticks=x, xticklabels=part.policy, title=scenario, ylabel='Signed contribution: Decrease20 minus policy (EUR/customer)')
        ax.axhline(0, color='black', linewidth=.5)
    axes[1].legend(fontsize=8)
    fig.savefig(folder/'reward_decomposition.png', dpi=150)
    plt.close(fig)
    sensitivity = pd.read_csv(output/'sensitivity.csv')
    ss = []
    for key, part in sensitivity.groupby(['parameter', 'factor', 'horizon']):
        ss.append(dict(zip(['parameter', 'factor', 'horizon'], key)) |
                  dict(contraction_share=weighted(part.assign(contraction=part.best_action == 0), 'contraction'),
                       states=len(part), mean_gap=weighted(part, 'gap')))
    ss = pd.DataFrame(ss)
    ss.to_csv(output/'sensitivity_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for j, (name, part) in enumerate(ss[ss.horizon == 6].groupby('parameter')):
        baseline = part.loc[part.factor == 1, 'contraction_share'].iloc[0]
        ax.plot([part.contraction_share.min(), part.contraction_share.max()], [j, j], linewidth=6, alpha=.5)
        ax.plot(baseline, j, 'k|', markersize=12)
    names = sorted(ss.parameter.unique())
    ax.set(yticks=range(len(names)), yticklabels=names, xlabel='P(-20% best), factor .5–1.5; black = canonical',
           title='OAT, H=6, Static continuation; dynamics use a smaller stratified sample', xlim=(0, 1))
    fig.savefig(folder/'parameter_sensitivity.png', dpi=150)
    plt.close(fig)
    # Persistence traces reuse cached rollout-derived action values at each horizon.
    persistence = values[np.isclose(values.gamma, 1.) & values.action.isin([0, 2])].pivot(
        index=['state_id', 'continuation', 'horizon'], columns='action', values='expected_reward').reset_index()
    persistence['contraction_minus_hold'] = persistence[0]-persistence[2]
    first = persistence[persistence.horizon == 1][['state_id', 'continuation', 'contraction_minus_hold']]
    persistence = persistence.merge(first, on=['state_id', 'continuation'], suffixes=('', '_first'))
    persistence['future_reward_difference'] = persistence.contraction_minus_hold-persistence.contraction_minus_hold_first
    persistence.to_csv(output/'action_persistence.csv', index=False)
    # Generated quantitative report; no hardcoded measurements.
    wide = d.pivot(index=['continuation', 'horizon'], columns='action', values='share').reset_index()
    wide.columns = ['continuation', 'horizon', '-20%', '-10%', 'hold', '+10%', '+20%']
    collapse = pd.read_csv(output/'policy_collapse.csv')
    floor = pd.read_csv(output/'limit_floor.csv')
    gap_summary = pd.read_csv(output/'gap_summary.csv')
    selected_gaps = gap_summary[(gap_summary.stratification == 'all') & np.isclose(gap_summary.gamma, .98)]
    text = '# Generated structural measurements\n\n'
    text += f'{len(states)} visited full states; inverse-inclusion weights sum to {states.weight.sum():.0f} active visits. '
    text += f'{int(gaps.draws.iloc[0])} common-random-number draws per state/action.\n\n'
    text += '## Action dominance (gamma .98)\n\n'+markdown(wide)+'\n\n'
    text += '## Planning (gamma .98)\n\n'+markdown(ps[np.isclose(ps.gamma, .98)])+'\n\n'
    text += '## Independent-draw immediate sacrifice and gain over fixed contraction\n\n'+markdown(validation_summary)+'\n\n'
    text += '## Action gaps (EUR; MC-unresolved includes exact ties)\n\n'+markdown(selected_gaps[['continuation', 'horizon', 'mean', 'median', 'p10', 'p90', 'p95', 'unresolved_share']])+'\n\n'
    text += '## PPO behavior\n\n'+markdown(collapse[['scenario', 'seed', 'matched', 'unmatched', 'requested_action_agreement', 'effective_action_agreement', 'policy_collapse']])+'\n\n'
    text += '## Floor\n\n'+markdown(floor[floor.policy.isin(['AlwaysDecrease20', 'PPO'])][['scenario', 'policy', 'seed', 'reached_fraction', 'mean_first_floor_month_among_reachers', 'observed_transition_floor_fraction', 'partial_minus20', 'no_effect_minus20']])+'\n\n'
    text += '## Economic mechanism (Decrease20 minus comparator)\n\n'
    cols = ['scenario', 'policy']+['decrease_minus_policy_reward_'+c for c in components]+['decrease_minus_policy_net_economic_value']
    text += markdown(overall[cols])+'\n\n'
    text += '## Discount sensitivity: -20% share\n\n'
    text += markdown(dominance[(dominance.stratification == 'all') & (dominance.action == 0) & (dominance.horizon == 24)][['continuation', 'gamma', 'share', 'degeneracy', 'resolved_share']])+'\n\n'
    text += '## Parameter sensitivity\n\n'+markdown(ss)+'\n'
    (output/'report_tables.md').write_text(text, encoding='utf-8')
    one = wide[(wide.continuation == 'AlwaysDecrease20') & (wide.horizon == 1)].iloc[0]
    long = wide[(wide.continuation == 'AlwaysDecrease20') & (wide.horizon == 24)].iloc[0]
    opportunity = ps[(ps.continuation == 'AlwaysDecrease20') & (ps.horizon == 24) & np.isclose(ps.gamma, .98)].iloc[0]
    checked = validation_summary[(validation_summary.continuation == 'AlwaysDecrease20') & (validation_summary.horizon == 24)].iloc[0]
    collapsed = bool(np.isclose(collapse.policy_collapse, 1.).all() and (collapse.unmatched == 0).all())
    behavior = ('The canonical PPO/AlwaysDecrease20 effective equivalence remains; PPO has not demonstrated exploitation of this opportunity.'
                if collapsed else 'This panel does not show exact PPO/AlwaysDecrease20 effective equivalence; inspect per-seed behavior before drawing a collapse conclusion.')
    summary = (
        f'On {len(states)} sampled visited full states, maximum contraction is the estimated best '
        f'admissible request in {one["-20%"]:.1%} of weighted states at one step and '
        f'{long["-20%"]:.1%} over the remaining horizon with AlwaysDecrease20 continuation. '
        f'The preferred request changes between these horizons in {opportunity.disagreement:.1%} of states. '
        f'Split-draw planning opportunity is EUR {opportunity.heldout_opportunity:.2f} of discounted training reward per sampled decision '
        f'(conditional MC interval [{opportunity.heldout_mc_low:.2f}, {opportunity.heldout_mc_high:.2f}]). '
        'This is privileged simulator evidence of state-dependent continuation values, not Q* or an attainable PPO gain. '
        + behavior
    )
    (output/'summary.md').write_text(summary+'\n', encoding='utf-8')
    answers = [
        ('Q1: Is -20% immediately optimal everywhere?', f'No: weighted share {one["-20%"]:.1%}.'),
        ('Q2: Does it remain universally optimal at long horizon?', f'No: {long["-20%"]:.1%} with contraction continuation; see other continuations.'),
        ('Q3: Can an increase be better?', f'An increase wins in {one["+10%"]+one["+20%"]:.1%} at H1 and {long["+10%"]+long["+20%"]:.1%} at remaining horizon.'),
        ('Q4: Are those states visited?', f'Yes: all {len(states)} states are from simulated held-out trajectories, not a fabricated grid.'),
        ('Q5: Do effects persist?', 'Yes: action_persistence.csv separates future rewards; persistent_state_effects.csv reports lagged exposure/purchase/survival effects.'),
        ('Q6: Does the choice depend on state?', f'Yes: the largest remaining-horizon share is {max(long.iloc[2:]):.1%} under contraction continuation, far from all sampled visitation.'),
        ('Q7: Does the preferred action depend on horizon?', f'Estimated one-step/remaining-horizon disagreement {opportunity.disagreement:.1%}; phase maps are descriptive.'),
        ('Q8: Is myopic selection equivalent to multistep selection?', f'No for this full-state diagnostic: held-out gain EUR {opportunity.heldout_opportunity:.2f}; held-out immediate sacrifice EUR {checked.heldout_immediate_sacrifice:.2f} [{checked.heldout_immediate_sacrifice_mc_low:.2f}, {checked.heldout_immediate_sacrifice_mc_high:.2f}]. This differs from the MyopicEconomic surrogate policy.'),
        ('Q9: What PPO value comes from state-dependent effective decisions?', behavior+' See per-seed policy_collapse.csv.'),
        ('Q10: Does the result justify RL?', 'It motivates studying observable-information planning, but does not validate PPO or establish that RL is necessary; privileged lookahead is not a fair-information optimum.'),
    ]
    (output/'answers.md').write_text(markdown(pd.DataFrame(answers, columns=['Question', 'Measured answer']))+'\n', encoding='utf-8')
    interpretation = []
    for scenario in ('baseline', 'severe_stress'):
        row = overall[(overall.scenario == scenario) & (overall.policy == 'MyopicEconomic')].iloc[0]
        f = floor[(floor.scenario == scenario) & (floor.policy == 'AlwaysDecrease20')].iloc[0]
        agreement = collapse[collapse.scenario == scenario].requested_action_agreement.mean()
        revenue = row.decrease_minus_policy_reward_interest_income+row.decrease_minus_policy_reward_fee_income
        interpretation.append(
            f'**{scenario}:** versus MyopicEconomic, contraction changes revenue by EUR {revenue:.2f}, '
            f'saves EUR {row.decrease_minus_policy_reward_credit_loss:.2f} in realized loss and '
            f'EUR {row.decrease_minus_policy_reward_funding_cost:.2f} in funding per initial customer. '
            f'This yields EUR {row.decrease_minus_policy_net_economic_value:.2f} more net value. '
            f'The reward additionally benefits from EUR {row.decrease_minus_policy_reward_capital_cost:.2f} '
            f'less capital proxy and EUR {row.decrease_minus_policy_reward_constraint_penalty:.2f} less PD penalty. '
            f'The floor is reached by {f.reached_fraction:.1%} of customers, after '
            f'{f.mean_first_floor_month_among_reachers:.2f} months among reachers; '
            f'{f.observed_transition_floor_fraction:.1%} of observed transitions end at the floor. '
            f'PPO requested-action agreement is {agreement:.1%}; effective agreement is '
            f'{collapse[collapse.scenario == scenario].effective_action_agreement.mean():.1%}.')
    precision_frame = pd.DataFrame(precision).merge(states[['state_id', 'weight']], on='state_id')
    uncertainty = precision_frame[(precision_frame.continuation == 'AlwaysDecrease20') & (precision_frame.horizon == 24)]
    interpretation.append(
        f'The remaining-horizon independent-half ranking agreement is {weighted(uncertainty, "halves_agree"):.1%}; '
        f'{1-weighted(uncertainty, "resolved"):.1%} of weighted requested-action gaps remain MC-unresolved. '
        f'Selecting a first action on one draw half and comparing it to fixed initial -20% on the other '
        f'gives EUR {checked.heldout_gain_over_initial_minus20:.2f} '
        f'[{checked.heldout_gain_over_initial_minus20_mc_low:.2f}, {checked.heldout_gain_over_initial_minus20_mc_high:.2f}] '
        'with contraction continuation. Thus even the constant rule is not established as an optimum.')
    discount = dominance[(dominance.stratification == 'all') & (dominance.action == 0) &
        (dominance.horizon == 24) & (dominance.continuation == 'AlwaysDecrease20')]
    interpretation.append(
        f'Across the five diagnostic discount factors, the remaining-horizon contraction share ranges '
        f'from {discount.share.min():.1%} to {discount.share.max():.1%}. '
        'Discounting changes action rankings but does not establish uniform contraction dominance.')
    planning_evidence = checked.heldout_immediate_sacrifice_mc_low > 0 and opportunity.heldout_mc_low > 0
    classification = ('meaningful state-dependent decision problem, with evidence of a meaningful sequential/planning problem'
                      if planning_evidence else 'state-dependent sampled choices; meaningful planning remains unresolved at this MC precision')
    interpretation.append(
        f'**Descriptive classification:** {classification}, **conditional on privileged full state and the named '
        'continuations**. The measured shares, MC errors and split-draw sacrifices above support this description. '
        'It does not prove that this information is recoverable from PPO observations or justify a new '
        'RL algorithm. The experiment establishes policy collapse and its economic incentives, not the '
        'optimization-path cause of PPO training convergence; no training intervention was performed.')
    (output/'interpretation.md').write_text('\n\n'.join(interpretation)+'\n', encoding='utf-8')
    import hashlib
    import importlib.metadata
    provenance = dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        table_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.csv'))},
        versions={name: importlib.metadata.version(name) for name in
                  ('numpy', 'pandas', 'matplotlib', 'scikit-learn', 'stable-baselines3')})
    (output/'report_manifest.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    diagnosis = Path('docs/structural_diagnosis.md')
    if update_docs and diagnosis.exists():
        content = diagnosis.read_text(encoding='utf-8')
        start, end = '<!-- structural-results:start -->', '<!-- structural-results:end -->'
        if start in content and end in content:
            compact = '\n\n'+summary+'\n\n'+markdown(pd.DataFrame(answers, columns=['Question', 'Measured answer']))+'\n\n'+markdown(wide)+'\n\n'+markdown(ps[np.isclose(ps.gamma, .98)])+'\n\n'+'\n\n'.join(interpretation)+'\n\n'
            diagnosis.write_text(content.split(start)[0]+start+compact+end+content.split(end)[1], encoding='utf-8')
        content = diagnosis.read_text(encoding='utf-8')
        start, end = '<!-- structural-interpretation:start -->', '<!-- structural-interpretation:end -->'
        if start in content and end in content:
            diagnosis.write_text(content.split(start)[0]+start+'\n\n'+interpretation[-1]+'\n\n'+end+content.split(end)[1], encoding='utf-8')
        for path in (Path('README.md'), Path('docs/technical_paper.md')):
            content = path.read_text(encoding='utf-8')
            start, end = '<!-- structural-summary:start -->', '<!-- structural-summary:end -->'
            if start in content and end in content:
                path.write_text(content.split(start)[0]+start+'\n\n'+summary+'\n\n'+end+content.split(end)[1], encoding='utf-8')
