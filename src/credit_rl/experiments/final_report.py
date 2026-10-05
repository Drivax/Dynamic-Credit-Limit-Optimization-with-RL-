"""Regenerate final figures, measured documentation and their consistency audit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from .final_evaluation import output_for, context, ROOT
from .final_panels import POLICIES
from .main_evaluation import digest
from .policy_initialization import write_json

LABELS = {policy: policy for policy in POLICIES}
COLORS = dict(zip(POLICIES, plt.get_cmap('tab10').colors[:8]))
START, END = '<!-- final-measured:start -->', '<!-- final-measured:end -->'
CONCLUSION_START, CONCLUSION_END = '<!-- final-conclusion:start -->', '<!-- final-conclusion:end -->'


def replace_block(text, block):
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('Exactly one final measured block required')
    left, rest = text.split(START)
    _, right = rest.split(END)
    return left+START+'\n'+block+'\n'+END+right


def overview_block(block):
    """Select the primary-budget overview from the generated full result block."""
    lines = block.splitlines()
    primary = lines[0].split('**')[1].split()[0]
    excluded = ('| Static |', '| PDThreshold |', '| MyopicEconomic |')
    selected = []
    for line in lines:
        if line.startswith(excluded) or line.startswith('Validation selected '):
            continue
        if line.startswith('| ') and line.split('|')[1].strip().replace(',', '').isdigit():
            if line.split('|')[1].strip() != primary:
                continue
        if line.startswith('Final classification:') or line.startswith('Learned sequential control '):
            continue
        selected.append(line)
    return '\n'.join(selected).strip().replace('\n\n\n', '\n\n')


def publish_documents(output, block):
    """Publish frozen measured text without models, fitting or statistical analysis."""
    answer = json.loads((output/'final_classification.json').read_text())['answer']
    for path in (Path('README.md'), Path('docs/technical_paper.md')):
        document_block = overview_block(block) if path.name == 'README.md' else block
        text = replace_block(path.read_text(encoding='utf-8'), document_block)
        if text.count(CONCLUSION_START) != 1 or text.count(CONCLUSION_END) != 1:
            raise ValueError('Exactly one generated conclusion required')
        left, remainder = text.split(CONCLUSION_START)
        _, right = remainder.split(CONCLUSION_END)
        text = left+CONCLUSION_START+'\n'+answer+'\n'+CONCLUSION_END+right
        path.write_text(text, encoding='utf-8')
        assert text.split(START)[1].split(END)[0].strip() == document_block
        assert text.split(CONCLUSION_START)[1].split(CONCLUSION_END)[0].strip() == answer


def measured(output):
    table = pd.read_csv(output/'main_table.csv')
    comparisons = pd.read_csv(output/'paired_comparisons.csv')
    robust = pd.read_csv(output/'robustness.csv')
    budget = int(table.budget.min())
    panel = table[table.budget == budget].set_index(['policy', 'world'])
    lines = ['Primary budget: **'+f'{budget:,}'+' steps**. Values are synthetic EUR per customer; '
        'diversity is one minus the largest requested-action share. Default and diversity below refer to nominal.', '',
        '| Policy | Baseline NEV | Stress NEV | Baseline reward | Stress reward | Default | Diversity | Worst-world NEV |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for policy in POLICIES:
        base, stress = panel.loc[policy, 'nominal'], panel.loc[policy, 'severe_stress']
        worst = robust[(robust.budget == budget) & (robust.policy == policy) & (robust.metric == 'net_economic_value')].worst_world.iloc[0]
        lines.append(f'| {policy} | {base.net_economic_value:.1f} | {stress.net_economic_value:.1f} | '
            f'{base.discounted_reward:.1f} | {stress.discounted_reward:.1f} | {base.defaulted:.1%} | {base.diversity:.3f} | {worst:.1f} |')
    lines += ['', 'Incremental NEV of BCRegularizedPPO over AlwaysDecrease20 (paired 95% intervals):', '',
              '| Budget | World | Difference | 95% interval | Seeds | Customers |', '|---:|---|---:|---:|---:|---:|']
    chosen = comparisons[(comparisons.contrast == 'C3') & (comparisons.metric == 'net_economic_value') &
                         comparisons.world.isin(['nominal', 'severe_stress', 'worst_world'])]
    for row in chosen.itertuples():
        lines.append(f'| {row.budget:,} | {row.world} | {row.mean:.1f} | [{row.lower:.1f}, {row.upper:.1f}] | {row.seeds} | {row.customers} |')
    selection = json.loads((output/'selection.json').read_text())
    lines += ['', f"Validation selected **{selection['schedule']}**, β₀={selection['beta']:.2f}. "
        f"Mean validation reward: {selection['selected_validation_score']:.2f}; BC0: {selection['BC0_standard_validation_mean']:.2f}. "
        'The latter comparison is descriptive and did not trigger additional tuning.', '',
        'Intervals are marginal, conditional on the synthetic simulator, and do not establish real-bank population effects. '
        'Full intervals, stress interactions, both budgets and all five worlds are retained in the supporting CSVs.']
    primary = chosen[chosen.budget == budget]
    worlds = ['nominal', 'severe_stress', 'population_shift', 'behavioral_shift', 'risk_shift']
    economic = comparisons[(comparisons.budget == budget) & (comparisons.contrast == 'C3') &
                           (comparisons.metric == 'net_economic_value') & comparisons.world.isin(worlds)]
    objective = comparisons[(comparisons.budget == budget) & (comparisons.contrast == 'C3') &
                            (comparisons.metric == 'discounted_reward') & comparisons.world.isin(worlds)]
    if (economic.lower > 0).all() and (objective.lower > 0).all() and (primary.lower > 0).all():
        complexity, robustness = 'learned policy clearly justified', 'robust'
    elif (primary.upper <= 0).all():
        complexity, robustness = 'simple policy preferred', 'fragile'
    else:
        complexity = 'conditionally justified' if (economic['mean'] > 0).all() else 'simple policy competitive'
        robustness = 'scenario-dependent'
    conclusion = dict(SequentialValue='conditional', Learnability='initialization-sensitive',
                      Robustness=robustness, ComplexityValue=complexity,
                      classification_basis='Descriptive primary-budget interpretation, not an additional confirmatory test: NEV and reward across all declared worlds; historical structural evidence')
    lines += ['', f"Final classification: **SequentialValue: {conclusion['SequentialValue']}; "
        f"Learnability: {conclusion['Learnability']}; Robustness: {robustness}; ComplexityValue: {complexity}.**"]
    if complexity == 'learned policy clearly justified':
        answer = 'In this predeclared synthetic suite, learned sequential control earns its complexity through positive incremental value over contraction in nominal, stress and worst-world comparisons. This conclusion is restricted to these modeled worlds.'
    elif complexity == 'conditionally justified':
        answer = 'Learned sequential control adds net economic value over contraction across the declared worlds, but does not establish a reward improvement in every world. Its complexity is justified conditionally on the economic objective and synthetic setting, not by universal policy superiority.'
    else:
        answer = 'Useful state-dependent decisions can be learned, but the final evidence does not establish enough robust incremental value to displace the simple contraction rule. Sequential learning earns its complexity only where an independently evaluated economic gain survives adverse worlds and uncertainty.'
    lines += ['', 'Regularization versus initialization alone (primary budget; paired 95% intervals):', '',
              '| Scenario | NEV difference | Reward difference |', '|---|---:|---:|']
    for world in ('nominal', 'severe_stress'):
        g = comparisons[(comparisons.budget == budget) & (comparisons.contrast == 'C2') &
                        (comparisons.world == world)].set_index('metric')
        cells = []
        for metric in ('net_economic_value', 'discounted_reward'):
            r = g.loc[metric]
            cells.append(f'{r["mean"]:.1f} [{r.lower:.1f}, {r.upper:.1f}]')
        lines.append(f'| {world} | {cells[0]} | {cells[1]} |')
    c2 = comparisons[(comparisons.budget == budget) & (comparisons.contrast == 'C2') &
                     (comparisons.metric == 'net_economic_value') & comparisons.world.isin(['nominal', 'severe_stress'])]
    if not (c2.lower > 0).all():
        answer += ' The additional BC regularizer does not establish an incremental NEV gain over initialization alone in both macro scenarios.'
    lines += ['', answer]
    write_json(output/'final_classification.json', {**conclusion, 'answer': answer})
    return '\n'.join(lines)


def figures(output):
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120, 'savefig.dpi': 150})
    sources = {}
    def save(fig, name, csvs):
        fig.tight_layout()
        fig.savefig(folder/(name+'.png'), metadata={'Software': 'credit-rl final report'})
        plt.close(fig)
        sources[name] = [str(p) for p in csvs]
    table = pd.read_csv(output/'main_table.csv')
    budget = table.budget.min()
    table = table[table.budget == budget]
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set(xlim=(0, 10), ylim=(0, 2.2)); ax.axis('off')
    boxes = [(1, 1.7, 'Public state\n21 observations'), (3.6, 1.7, 'Requested limit\n−20%, −10%, 0, +10%, +20%'),
             (6.4, 1.7, 'Admission + dynamics\nspending / payment / default'), (9, 1.7, 'Reward + next state\n24-month horizon')]
    for x, y, text in boxes:
        ax.text(x, y, text, ha='center', va='center', bbox=dict(boxstyle='round,pad=.6', fc='#edf3fa', ec='#37516d'))
    for a, b in ((1.9, 2.5), (4.8, 5.2), (7.7, 8.1)):
        ax.annotate('', (b, 1.7), (a, 1.7), arrowprops=dict(arrowstyle='->'))
    ax.text(5, .5, 'Hidden persistent traits + common exogenous shocks → transitions only\nFrozen PD estimator and public teacher; no latent or future inputs to the actor', ha='center')
    save(fig, '01_decision_process', ['configs/simulation.yaml'])
    action_maps = ROOT/'structural_diagnostics/action_maps.csv'
    if action_maps.exists():
        frame = pd.read_csv(action_maps)
        frame = frame[(frame.continuation == 'AlwaysDecrease20') & (frame.conditioning == 'all')]
        fig, axes = plt.subplots(2, 3, figsize=(9, 6.5))
        cmap = ListedColormap(['#2166ac', '#67a9cf', '#eeeeee', '#ef8a62', '#b2182b'])
        for ax, horizon in zip(axes.flat, (1, 3, 6, 12, 24)):
            panel = frame[frame.horizon == horizon]
            matrix = np.full((3, 3), np.nan)
            for row in panel.itertuples():
                if row.supported:
                    matrix[row.pd_bucket, row.utilization_bucket] = row.best_action
            im = ax.imshow(matrix, origin='lower', cmap=cmap,
                           norm=BoundaryNorm(np.arange(-.5, 5.5), cmap.N))
            for y in range(3):
                for x in range(3):
                    cell = panel[(panel.pd_bucket == y) & (panel.utilization_bucket == x)]
                    n = int(cell['count'].iloc[0]) if len(cell) else 0
                    ax.text(x, y, f'n={n}' if n >= 3 else f'n={n}\n<3 states',
                            ha='center', va='center', fontsize=8,
                            color='white' if matrix[y, x] in (0, 4) else 'black')
            ax.set(xticks=range(3), xticklabels=['≤0.5', '(0.5, 1]', '>1'],
                   yticks=range(3), yticklabels=['≤0.2', '(0.2, 0.6]', '>0.6'],
                   xlabel='Opening utilization', ylabel='Opening PD',
                   title=f'Horizon: {horizon} month'+('s' if horizon > 1 else '') if horizon < 24 else 'Horizon: remaining months')
        legend = axes.flat[-1]
        legend.axis('off')
        bar = fig.colorbar(im, ax=legend, location='left', fraction=.2, shrink=.85)
        bar.set_ticks(range(5), labels=['−20%', '−10%', 'hold', '+10%', '+20%'])
        legend.text(.1, .5, 'Weighted modal\naction across\nvisited states\n\nBlank: <3 states\nNo extrapolation',
                    ha='left', va='center', transform=legend.transAxes, fontsize=8)
        fig.suptitle('State-dependent action preferences\nAlwaysDecrease20 continuation; discount 0.98')
        save(fig, '02_structural_action_map', [action_maps])
    trajectory = ROOT/'policy_initialization/training_trajectory.csv'
    if trajectory.exists():
        t = pd.read_csv(trajectory)
        t = t[t.budget == 32768]
        for number, metric, title in [('03', 'deterministic_contraction', 'Canonical greedy contraction'),
                                      ('04', 'diversity', 'Initialization changes the learning trajectory')]:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
            for ax, scenario in zip(axes, ['baseline', 'severe_stress']):
                for arm, group in t[t.scenario == scenario].groupby('experiment_id'):
                    if number == '03' and arm != 'RandomInit':
                        continue
                    g = group.groupby('timesteps')[metric].agg(['mean', 'min', 'max']).reset_index()
                    label = {'RandomInit': 'CanonicalPPO', 'ImitationInit': 'BCInitPPO'}.get(arm, arm)
                    ax.plot(g.timesteps, g['mean'], label=label)
                    ax.fill_between(g.timesteps, g['min'], g['max'], alpha=.15)
                ax.set(title={'baseline': 'Nominal', 'severe_stress': 'Severe stress'}[scenario],
                       xlabel='PPO steps', ylabel=metric.replace('_', ' '), ylim=(-.02, 1.02))
                ax.legend(fontsize=8)
            fig.suptitle(title+' (bands: seed ranges)')
            save(fig, number+('_canonical_collapse' if number == '03' else '_initialization'), [trajectory])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    b = pd.read_csv(output/'behavior_by_seed.csv')
    for ax, world in zip(axes, ['nominal', 'severe_stress']):
        g = b[(b.budget == budget) & (b.world == world)].groupby('policy').mean(numeric_only=True).reindex(POLICIES)
        bottom = np.zeros(len(g))
        for column, label, color in [('contraction_share', '−20%', '#355070'), ('hold_share', 'hold', '#e9c46a'), ('increase_share', 'increase', '#2a9d8f')]:
            ax.barh(range(len(g)), g[column], left=bottom, label=label, color=color)
            bottom += g[column].to_numpy()
        ax.barh(range(len(g)), 1-bottom, left=bottom, label='−10%', color='#b7c9df')
        ax.set(yticks=range(len(g)), yticklabels=[LABELS[x] for x in POLICIES], xlabel='Requested action share', title=world, xlim=(0, 1))
    axes[-1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.17), ncol=4)
    save(fig, '05_behavior', [output/'behavior_by_seed.csv'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, world in zip(axes, ['nominal', 'severe_stress']):
        for row in table[table.world == world].itertuples():
            ax.scatter(row.credit_loss, row.net_economic_value, color=COLORS[row.policy], label=LABELS[row.policy])
        ax.set(xlabel='Realized credit loss (EUR)', ylabel='Net economic value (EUR)', title=world)
    axes[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    save(fig, '06_economic_risk', [output/'main_table.csv'])
    comp = pd.read_csv(output/'paired_comparisons.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric in zip(axes, ['net_economic_value', 'discounted_reward']):
        g = comp[(comp.budget == budget) & (comp.contrast == 'C3') & (comp.metric == metric) &
                 comp.world.isin(['nominal', 'severe_stress', 'worst_world'])].set_index('world').reindex(['nominal', 'severe_stress', 'worst_world'])
        ax.vlines(range(3), g.lower, g.upper, color='#355070')
        ax.scatter(range(3), g['mean'], color='#355070')
        ax.axhline(0, color='gray', lw=1)
        ax.set(xticks=range(3), xticklabels=['Nominal', 'Stress', 'Worst world'],
               ylabel='Incremental synthetic EUR / customer', title=metric.replace('_', ' '))
    fig.suptitle('BCRegularizedPPO − AlwaysDecrease20; paired 95% intervals')
    save(fig, '07_incremental_value', [output/'paired_comparisons.csv'])
    values = table.pivot(index='policy', columns='world', values='net_economic_value').reindex(POLICIES)
    delta = values-values.loc['AlwaysDecrease20']
    fig, ax = plt.subplots(figsize=(9, 4))
    bound = max(1, abs(delta.to_numpy()).max())
    im = ax.imshow(delta, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
    for i in range(len(delta)):
        for j in range(len(delta.columns)):
            ax.text(j, i, f'{delta.iloc[i, j]:.0f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(delta.iloc[i, j]) > .55*bound else 'black')
    ax.set(yticks=range(len(delta)), yticklabels=[LABELS[x] for x in delta.index],
           xticks=range(len(delta.columns)), xticklabels=[x.replace('_', '\n') for x in delta.columns],
           title='World-specific incremental NEV versus constant contraction')
    fig.colorbar(im, ax=ax, label='EUR per customer')
    save(fig, '08_world_robustness', [output/'main_table.csv'])
    preservation = pd.read_csv(output/'preservation.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, scenario in zip(axes, ['baseline', 'severe_stress']):
        for policy, group in preservation[preservation.scenario == scenario].groupby('policy'):
            g = group.groupby('timesteps').teacher_regret.mean()
            ax.plot(range(len(g)), g, 'o-', label=LABELS[policy], color=COLORS[policy])
            ax.set(xticks=range(len(g)), xticklabels=[f'{x:,}' for x in g.index], xlabel='Post-update steps')
        ax.set_yscale('symlog', linthresh=1)
        ax.set_yticks([0, 1, 5, 10, 50, 200], labels=['0', '1', '5', '10', '50', '200'])
        ax.set(title=scenario, ylabel='Teacher regret (EUR; symlog scale)')
    axes[-1].legend(fontsize=7)
    save(fig, '09_preservation', [output/'preservation.csv'])
    ope = pd.read_csv(output/'ope_summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for row_index, policy in enumerate(['AlwaysDecrease20', 'BCRegularizedPPO']):
        for col, estimator in enumerate(['IS', 'WIS']):
            ax = axes[row_index, col]
            g = ope[(ope.policy == policy) & (ope.estimator == estimator)].set_index('overlap').reindex(['high', 'medium', 'low'])
            ax.plot(range(3), g.mean_estimate, 'o-', label=estimator)
            ax.axhline(g.truth.iloc[0], color='black', ls='--', label='Independent simulator mean')
            for i, row in enumerate(g.itertuples()):
                ax.text(i, .96, f'ESS {row.mean_ess:.1f}', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8)
            ax.set(xticks=range(3), xticklabels=['High', 'Medium', 'Low'], title=LABELS[policy]+' / '+estimator,
                   ylabel='Estimated NEV (EUR)', xlabel='Behavior overlap')
            ax.legend(fontsize=7, loc='lower left')
    save(fig, '10_ope', [output/'ope_summary.csv'])
    write_json(output/'figure_sources.json', sources)
    write_json(output/'figure_hashes.json', {str(f): digest(f) for f in sorted(folder.glob('*.png'))})


def report(profile='standard', publish=True):
    output = output_for(profile)
    context(profile)  # Refuse reporting without valid frozen inputs.
    # Rebuild tables from frozen raw evaluations when available, never refit.
    if (output/'evaluation').exists():
        from .final_analysis import analyze
        analyze(profile)
    secondary = ROOT/'../results/portfolio/standard/summary.csv'
    if secondary.exists():
        source = pd.read_csv(secondary)
        source = source[source['case'].isin(['normal_medium', 'stress_medium']) &
                        source.policy.isin(['Static', 'RiskBased', 'Decrease20', 'PPO_hard', 'PPO_penalty'])]
        source[['case', 'policy', 'value', 'default_rate', 'el_violation_rate', 'any_violation_rate']].to_csv(
            output/'constrained_secondary.csv', index=False)
    block = measured(output)
    if profile == 'smoke':
        block = '**SMOKE: software integration only; not scientific evidence.**\n\n'+block
    figures(output)
    (output/'measured_results.md').write_text(block+'\n', encoding='utf-8')
    if profile == 'standard' and publish:
        publish_documents(output, block)
    write_json(output/'documentation_audit.json', dict(profile=profile,
        published=profile == 'standard' and publish, csv_block_identical=True,
        measured_sha256=digest(output/'measured_results.md'),
        table_sha256=digest(output/'main_table.csv'), comparisons_sha256=digest(output/'paired_comparisons.csv')))
    if profile == 'standard' and publish:
        from .final_verify import manifest
        manifest(profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    parser.add_argument('--no-publish', action='store_true')
    parser.add_argument('--documents-only', action='store_true',
                        help='Publish existing measured text only; no models or computations required')
    args = parser.parse_args()
    if args.documents_only:
        if args.profile != 'standard' or args.no_publish:
            parser.error('--documents-only requires standard publishing')
        output = output_for('standard')
        publish_documents(output, (output/'measured_results.md').read_text(encoding='utf-8').strip())
        return
    report(args.profile, not args.no_publish)


if __name__ == '__main__':
    main()
