"""CSV-only Phase D figures and reproducible scientific report."""
import argparse
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from credit_rl.experiments.main_evaluation import digest
from credit_rl.experiments.policy_initialization import ROOT, write_json


def registry(output):
    record = json.loads((output/'preregistration.json').read_text())['identity']
    selection = json.loads((output/'initialization_selection.json').read_text())
    p = record['protocol'][record['profile']]
    architecture = selection['selected_architecture']
    rows = []
    for budget in p['budgets']:
        for arm in ('RandomInit', 'ImitationInit'):
            for seed in p['seeds']:
                for scenario in ('baseline', 'severe_stress'):
                    marker = output/'runs'/str(budget)/arm/str(seed)/'completed.json'
                    checkpoint = output/'imitation_models'/('Imitation_'+'x'.join(map(str, architecture or [64, 64])))/str(seed)/'selected.zip'
                    rows.append(dict(experiment_id=f'{arm}_{budget}', initialization=arm,
                        architecture='x'.join(map(str, architecture)) if architecture else 'unqualified',
                        seed=seed, budget=budget, scenario=scenario, teacher_version='frozen Phase B F0',
                        teacher_sha256=record['teacher_sha256'], imitation_checkpoint_hash=digest(checkpoint),
                        ppo_config_hash=hashlib.sha256(json.dumps(dict(record['ppo_config'],
                            actor_network=architecture or [64, 64]), sort_keys=True).encode()).hexdigest(),
                        status='complete' if marker.exists() else 'planned' if architecture else 'not_run_unqualified'))
    pd.DataFrame(rows).to_csv(output/'policy_initialization_registry.csv', index=False)


def figures(output):
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    quality = pd.read_csv(output/'imitation_quality.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ('effective_accuracy', 'teacher_regret')):
        quality.groupby(['scenario', 'experiment_id'])[metric].mean().unstack().plot.bar(ax=axis, rot=0)
        axis.set_title(metric.replace('_', ' '))
    fig.tight_layout()
    fig.savefig(folder/'imitation_quality.png', dpi=150)
    plt.close(fig)
    if (output/'initial_action_distribution.csv').exists():
        actions = pd.read_csv(output/'initial_action_distribution.csv')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, scenario in zip(axes, ('baseline', 'severe_stress')):
            actions[actions.scenario.eq(scenario)].groupby(['action', 'policy']).share.mean().unstack().plot.bar(ax=ax, rot=0)
            ax.set(title=scenario, ylabel='Fraction of natural validation visits', xlabel='Requested action: 0=-20%, 2=hold, 4=+20%')
        fig.tight_layout()
        fig.savefig(folder/'teacher_vs_initial.png', dpi=150)
        plt.close(fig)
    path = output/'training_trajectory.csv'
    if not path.exists():
        return
    trajectory = pd.read_csv(path)
    temporal = trajectory[trajectory.checkpoint_kind.isin(['initial', 'post_update'])]
    # Extended runs carry their own early prefix; avoid counting the short prefix twice.
    temporal = temporal[temporal.budget.eq(temporal.budget.max())]
    plots = {'preservation_curve': 'preservation_loss', 'teacher_regret_over_time': 'teacher_regret',
             'contraction_share_over_time': 'deterministic_contraction', 'policy_drift': 'kl_initial',
             'random_vs_imitation': 'diversity', 'stochastic_contraction': 'stochastic_contraction'}
    for name, metric in plots.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, scenario in zip(axes, ('baseline', 'severe_stress')):
            part = temporal[temporal.scenario.eq(scenario)]
            for arm, group in part.groupby('experiment_id'):
                values = group.groupby('timesteps')[metric].agg(['mean', 'min', 'max'])
                ax.plot(values.index, values['mean'], label=arm)
                ax.fill_between(values.index, values['min'], values['max'], alpha=.15)
            ax.set(xscale='symlog', xlabel='PPO steps (post-update snapshots)', title=scenario, ylabel=metric.replace('_', ' '))
            if metric in ('teacher_regret', 'preservation_loss'):
                ax.set_yscale('symlog', linthresh=1)
                ax.set_ylabel(metric.replace('_', ' ')+' (EUR; symlog)')
            ax.legend()
        fig.suptitle('Lines: seed means; bands: seed range, not confidence intervals')
        fig.tight_layout()
        fig.savefig(folder/f'{name}.png', dpi=150)
        plt.close(fig)
    economic = pd.read_csv(output/'economic_results.csv')
    economic = economic[(economic.split == 'validation') & economic.checkpoint_kind.isin(['initial', 'post_update']) &
                        (economic['mode'] == 'deterministic') & economic.budget.eq(economic.budget.max())]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scenario in zip(axes, ('baseline', 'severe_stress')):
        for arm, group in economic[economic.scenario.eq(scenario)].groupby('experiment_id'):
            values = group.groupby('timesteps').discounted_reward.mean()
            ax.plot(values.index.to_numpy(), values.to_numpy(), label=arm)
        ax.set(xscale='symlog', title=scenario, ylabel='Mean validation discounted reward (EUR)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'validation_economics.png', dpi=150)
    plt.close(fig)
    observation_path = output/'public_validation_observations.csv'
    if observation_path.exists():
        observations = pd.read_csv(observation_path)
        largest = int(temporal.budget.max())
        panels = [pd.read_csv(p) for p in (output/'analysis'/str(largest)).glob('*/panel_states.csv.gz')]
        states = pd.concat(panels, ignore_index=True)
        states = states[states.checkpoint_kind.isin(['initial', 'post_update'])]
        times = [t for t in (0, 8192, 32768, 262144) if t in states.timesteps.unique()]
        fig, axes = plt.subplots(4, len(times), figsize=(4*len(times), 11), squeeze=False)
        for row, (scenario, arm) in enumerate((s, a) for s in ('baseline', 'severe_stress') for a in ('RandomInit', 'ImitationInit')):
            for col, steps in enumerate(times):
                group = states[(states.scenario == scenario) & (states.experiment_id == arm) & (states.timesteps == steps)]
                actions = group.groupby('state_id').action.agg(lambda x: x.value_counts().index[0])
                obs = observations.iloc[actions.index]
                scatter = axes[row, col].scatter(obs.predicted_pd, obs.utilization_bounded, c=actions, vmin=0, vmax=4,
                                      cmap='viridis', s=4, alpha=.5)
                axes[row, col].set(title=f'{scenario} / {arm} / {steps}', xlabel='PD', ylabel='Bounded utilization')
        fig.suptitle('Observed-state projections; modal greedy action across seeds (0 contraction, 4 increase)')
        fig.tight_layout(rect=(0, 0, .9, .96))
        cax = fig.add_axes((.93, .15, .015, .7))
        colorbar = fig.colorbar(scatter, cax=cax, ticks=[0, 1, 2, 3, 4])
        colorbar.ax.set_yticklabels(['-20%', '-10%', 'hold', '+10%', '+20%'])
        fig.savefig(folder/'decision_maps_over_time.png', dpi=150)
        plt.close(fig)
    boundaries = pd.read_csv(output/'boundary_survival.csv')
    boundaries = boundaries[boundaries.budget.eq(boundaries.budget.max()) & boundaries.checkpoint_kind.isin(['initial', 'post_update'])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scenario in zip(axes, ('baseline', 'severe_stress')):
        for region, group in boundaries[boundaries.scenario.eq(scenario) & boundaries.experiment_id.eq('ImitationInit')].groupby('teacher_region'):
            group = group.assign(survivors=group.boundary_survival.fillna(0)*group.eligible_states)
            values = group.groupby('timesteps')[['survivors', 'eligible_states']].sum()
            ax.plot(values.index.to_numpy(), (values.survivors/values.eligible_states.replace(0, float('nan'))).to_numpy(), label=region)
        ax.set(xscale='symlog', title=scenario, ylim=(-.02, 1.02), ylabel='Conditional agreement with initial teacher boundary')
        ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'boundary_survival.png', dpi=150)
    plt.close(fig)
    for filename, source, grouping, metric in [
        ('baseline_vs_stress', 'statistical_comparisons.csv', ['scenario', 'budget'], 'mean'),
        ('short_vs_long_budget', 'statistical_comparisons.csv', ['budget', 'scenario'], 'mean'),
        ('fixed_state_gae_pressure' if (output/'actual_teacher_pressure_summary.csv').exists() else 'advantage_pressure',
         'advantage_pressure.csv', ['experiment_id', 'teacher_action'], 'preceding_teacher_gae')]:
        frame = pd.read_csv(output/source)
        if source == 'statistical_comparisons.csv':
            frame = frame[(frame.metric == 'discounted_reward') & (frame.split == 'test') &
                          (frame.checkpoint_kind == 'validation_selected') & (frame['mode'] == 'deterministic')]
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = frame.groupby(grouping)[metric].mean().unstack()
        bars.plot.bar(ax=ax, rot=0)
        if source == 'statistical_comparisons.csv':
            bounds = frame.set_index(grouping)
            for column, container in enumerate(list(ax.containers)):
                for index, bar in enumerate(container):
                    row = bounds.loc[(bars.index[index], bars.columns[column])]
                    ax.errorbar([bar.get_x()+bar.get_width()/2], [row['mean']],
                        yerr=[[row['mean']-row.low], [row.high-row['mean']]], fmt='none', color='black', capsize=4)
            ax.set_ylabel('Reward effect (EUR/customer)\nImitation − Random; 95% paired CI')
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_title(filename.replace('_', ' '))
        fig.tight_layout()
        fig.savefig(folder/f'{filename}.png', dpi=150)
        plt.close(fig)
    if (output/'actual_teacher_pressure_summary.csv').exists():
        actual = pd.read_csv(output/'actual_teacher_pressure_summary.csv')
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, arm in zip(axes, ('RandomInit', 'ImitationInit')):
            group = actual[actual.experiment_id.eq(arm)].sort_values('teacher_action')
            ax.bar(group.teacher_action, group.normalized_mean, color='#397a9e')
            ax.errorbar(group.teacher_action, group.normalized_mean,
                yerr=[group.normalized_mean-group.low, group.high-group.normalized_mean], fmt='none', color='black', capsize=4)
            ax.axhline(0, color='black', linewidth=.7)
            ax.set(xticks=[0, 1, 2, 3, 4], xticklabels=['-20%', '-10%', 'hold', '+10%', '+20%'],
                   title=arm, ylabel='Actual normalized PPO advantage')
        fig.suptitle('Teacher-matching sampled actions, steps 0–8192; 95% seed-cluster intervals')
        fig.tight_layout()
        fig.savefig(folder/'advantage_pressure.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    args = parser.parse_args()
    output = ROOT/('policy_initialization' if args.profile == 'standard' else 'policy_initialization_smoke')
    registry(output)
    figures(output)
    write_json(output/'figure_hashes.json', {p.name: digest(p) for p in (output/'figures').glob('*.png')})


def markdown_table(frame):
    frame = frame.copy()
    for col in frame.select_dtypes(include='number'):
        frame[col] = frame[col].map(lambda v: f'{v:.4f}')
    header = '| '+' | '.join(frame.columns)+' |\n'
    return header+'| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join(
        '| '+' | '.join(map(str, row))+' |\n' for row in frame.itertuples(index=False, name=None))


def scientific_report(output, publish=False):
    selection = json.loads((output/'initialization_selection.json').read_text())
    if selection['status'] != 'qualified':
        text = ('# Phase D smoke engineering report\n\nThe smoke imitation did not qualify. '
            'Forced smoke training checks software execution only and provides no scientific preservation evidence.\n\n'
            '## 17. Decision gate for next phase\n\nUnresolved. Consult the qualified standard experiment for scientific conclusions.\n')
        (output/'report.md').write_text(text, encoding='utf-8')
        write_json(output/'decision_gate.json', dict(classification=['Unresolved'],
            scope='Unqualified smoke engineering exercise; no scientific inference.', initialization_qualified=False))
        return
    trajectory = pd.read_csv(output/'training_trajectory.csv')
    comparisons = pd.read_csv(output/'statistical_comparisons.csv')
    interaction = pd.read_csv(output/'scenario_interaction.csv')
    economic = pd.read_csv(output/'economic_results.csv')
    boundaries = pd.read_csv(output/'boundary_survival.csv')
    preservation = pd.read_csv(output/'preservation_pooled.csv')
    pressure = pd.read_csv(output/'advantage_pressure.csv')
    headline = comparisons[(comparisons.split == 'test') & (comparisons.checkpoint_kind == 'validation_selected')]
    final = trajectory[trajectory.checkpoint_kind == 'final'].groupby(
        ['budget', 'experiment_id', 'scenario'])[['diversity', 'teacher_regret', 'agreement',
            'deterministic_contraction', 'stochastic_contraction']].mean().reset_index()
    initial = trajectory[(trajectory.checkpoint_kind == 'initial') & trajectory.budget.eq(trajectory.budget.min())]
    initial = initial.groupby(['experiment_id', 'scenario'])[['diversity', 'teacher_regret', 'agreement', 'entropy']].mean().reset_index()
    initial_economic = economic[economic.checkpoint_kind.eq('initial') & economic.budget.eq(economic.budget.min())].groupby(
        ['experiment_id', 'scenario', 'mode'])[['discounted_reward', 'net_economic_value']].mean().reset_index()
    short = int(trajectory.budget.min())
    long = int(trajectory.budget.max())
    short_baseline = headline[(headline.budget == short) & (headline.scenario == 'baseline') &
        (headline.metric == 'discounted_reward') & (headline['mode'] == 'deterministic')]
    categories = ['InitializationSensitive'] if (short_baseline.low > 0).all() else ['Unresolved']
    effects = interaction[(interaction.split == 'test') & (interaction.checkpoint_kind == 'validation_selected') &
                          (interaction.metric == 'discounted_reward') & (interaction['mode'] == 'deterministic')]
    if ((effects.low > 0) | (effects.high < 0)).any():
        categories.append('ScenarioDependent')
    baseline_final = final[(final.budget == short) & (final.scenario == 'baseline')].set_index('experiment_id')
    if (baseline_final.loc['ImitationInit', 'diversity'] > .05 and
        baseline_final.loc['RandomInit', 'diversity'] < .05 and (short_baseline.low > 0).all()):
        categories.append('DiscoveryLimited')
    if 'Unresolved' not in categories:
        categories.append('Unresolved')
    decision = dict(classification=categories,
        scope='DiscoveryLimited, when present, is restricted to short-budget baseline behavior; not a uniquely identified universal cause.',
        unresolved='Teacher fidelity can erode without full collapse; stress reward and net economic value are different objectives. No equivalence margin was preregistered for long-run convergence.',
        long_budget_observed=long, initialization_qualified=selection['status'] == 'qualified')
    write_json(output/'decision_gate.json', decision)
    intro = ('The 64x64 public-state imitation passed the frozen validation gate before PPO. '
        'Only actor weights differ within paired runs; the random critic and canonical PPO objective are shared. '
        'The experiment distinguishes retention of nonconstant decisions, preservation of the fitted teacher boundary, '
        'and realized economic performance. These are not interchangeable outcomes.')
    sections = []
    def section(number, title, body):
        sections.append(f'## {number}. {title}\n\n{body}\n')
    section(1, 'Motivation from Phase C',
        'Phase C found early greedy contraction, later stochastic concentration and partial late recovery. '
        'Critic, entropy and actor-capacity interventions did not identify a unique mechanism. Its imitation '
        'was only partial and did not reduce teacher regret; Phase D therefore first qualifies a new initialization.')
    section(2, 'Experimental design', intro+'\n\n'
        'Five paired seeds (101, 202, 303, 404, 505), independent 32,768 and 262,144-step runs, '
        'unchanged canonical training population and baseline-only checkpoint selection. Natural imitation visits '
        'come from equally allocated PPO, F0 and myopic customer trajectories, with 600/150/150 new train/validation/test '
        'customers per macro scenario. Both macro versions retain the same customer partition. Only the 21 public '
        'features enter the actor. All visits are retained without class balancing. Six hundred epochs, Adam .001, '
        'batch 256 and validation cross entropy select the imitation checkpoint. The teacher is frozen Phase B F0. '
        'The protocol and its SHA256 were saved before collection. No optional anchored loss or critic warm start was run.\n\n'
        'Selected checkpoints retain canonical pre-pending-update timing at validation boundaries, plus a final '
        'post-update evaluation. Temporal snapshots are explicitly post-update. Stochastic evaluation uses local '
        'customer/month uniforms shared across arms. Endogenous trajectories and termination times can diverge. '
        'The primary estimand is the imitation-minus-random difference for validation-selected policies on held-out '
        'customers. Seed/customer bootstrap intervals are descriptive, unadjusted for correlated endpoints.')
    section(3, 'Imitation qualification',
        'The gate uses validation only: significant effective-agreement gain, significant fitted-teacher-regret '
        'reduction, diversity >=.05 and at least two effective actions with >=1% share in at least four seeds, '
        'separately in both scenarios. The 64x64 architecture passed; the 128x128 fallback was not used.\n\n'+
        markdown_table(pd.DataFrame([dict(scenario=r['scenario'], agreement_gain=r['agreement_difference']['mean'],
            agreement_low=r['agreement_difference']['low'], agreement_high=r['agreement_difference']['high'],
            regret_change=r['regret_difference']['mean'], regret_low=r['regret_difference']['low'],
            regret_high=r['regret_difference']['high'], diverse_seeds=r['diverse_seeds'])
            for r in selection['decisions'][0]['scenarios']]))+
        '\nTest fidelity was measured only after architecture selection and short PPO training. See '
        '`imitation_test_quality.csv`; it did not change the decision. Higher agreement does not certify the teacher as optimal.')
    if (output/'imitation_test_quality.csv').exists():
        heldout = pd.read_csv(output/'imitation_test_quality.csv').groupby(['experiment_id', 'scenario'])[
            ['requested_accuracy', 'effective_accuracy', 'teacher_regret', 'diversity']].mean().reset_index()
        sections[-1] += '\nIndependent test fidelity (seed means):\n\n'+markdown_table(heldout)+'\n'
    section(4, 'Initial decision boundary', markdown_table(initial)+
        '\nThese are fixed natural-validation-state means, not whole-policy values. Both deterministic and '
        'stochastic t=0 economic evaluations were saved before the first PPO update in each run. '
        'Actor probabilities/logits and greedy actions match the supervised actor numerically; critic tensors remain identical within pairs.\n\n'+
        markdown_table(initial_economic))
    section(5, 'Random-init PPO trajectory', markdown_table(final[final.experiment_id == 'RandomInit'])+
        '\nThe random actor is evaluated before any update as well. Early greedy and stochastic contraction must '
        'be distinguished: the former describes argmax decisions, the latter the mean contraction probability. '
        '`collapse_timing.csv` reports first crossings and censoring, not an absorbing-state assumption.')
    section(6, 'Imitation-init PPO trajectory', markdown_table(final[final.experiment_id == 'ImitationInit'])+
        '\nCompare these final policies with initialization and the separately selected checkpoints. '
        'An actor can remain state-dependent while losing some teacher agreement. Random initialization is not '
        'the only trajectory compatible with the unchanged PPO objective.')
    selected_preservation = preservation[(preservation.experiment_id == 'ImitationInit') &
        (preservation.checkpoint_kind == 'final')]
    section(7, 'Does PPO preserve a useful policy?',
        'Preservation loss is fitted teacher regret at t minus regret at t=0 on the same states. Positive values '
        'indicate teacher-boundary erosion; they do not alone demonstrate worse simulator performance.\n\n'+
        markdown_table(selected_preservation[['budget', 'scenario', 'metric', 'mean', 'low', 'high']])+
        '\nThe four interpretive quadrants are retained: teacher preserved/economics preserved; teacher lost/economics '
        'improved; teacher preserved/economics worsened; teacher lost/economics worsened. Near-zero confidence '
        'intervals are inconclusive, not proof of equivalence. Reward and NEV can place a trajectory in different quadrants.')
    economic_changes = pd.read_csv(output/'economic_preservation.csv')
    economic_changes = economic_changes[(economic_changes.experiment_id == 'ImitationInit') &
        (economic_changes.checkpoint_kind == 'final') & (economic_changes['mode'] == 'deterministic')]
    sections[-1] += ('\nFinal imitation-initialized policy minus its own initialization on fixed validation customers '
        '(economic changes, EUR/customer; paired seed/customer intervals):\n\n'+markdown_table(economic_changes[
            ['budget', 'scenario', 'metric', 'mean', 'low', 'high']])+'\n')
    early = trajectory[(trajectory.timesteps <= 8192) & (trajectory.experiment_id == 'ImitationInit') &
        trajectory.checkpoint_kind.isin(['initial', 'post_update']) & trajectory.budget.eq(short)]
    section(8, 'When does policy drift begin?', markdown_table(early.groupby(['scenario', 'timesteps'])[
        ['kl_initial', 'action_flip', 'preservation_loss', 'deterministic_contraction']].mean().reset_index())+
        '\nKL is KL(pi_t || pi_0); the previous-snapshot KL and flip fraction are in `policy_drift.csv`. '
        'Intervals between snapshots localize changes, not the exact optimization step.')
    boundary = boundaries[(boundaries.experiment_id == 'ImitationInit') & (boundaries.checkpoint_kind == 'final')].copy()
    boundary['survivors'] = boundary.boundary_survival.fillna(0)*boundary.eligible_states
    boundary = boundary.groupby(['budget', 'scenario', 'teacher_region'])[['survivors', 'eligible_states']].sum()
    boundary['boundary_survival'] = boundary.survivors/boundary.eligible_states.replace(0, float('nan'))
    section(9, 'Which action regions are lost first?', markdown_table(boundary[
        ['eligible_states', 'boundary_survival']].reset_index())+
        '\nSurvival conditions on initial requested-action agreement. Ineligible states are missing, not failures. '
        'Low/medium/high confidence uses validation effective-action Q-gap terciles. The CSV retains counts and '
        'confidence strata; sparse strata should not be treated as equally precise evidence.')
    section(10, 'Advantage pressure on teacher-preferred actions',
        'Actual rollout files retain raw GAE, returns, pre-update critic values and observations; minibatch files '
        'retain actual normalized advantage sums by sampled action. Diagnostic joins label teacher-preferred '
        'actions after training. No labels or Q values enter PPO. Fixed-panel probes below use the previous '
        'snapshot and independent simulated continuation to estimate preceding teacher-action GAE; they are '
        'not the actual minibatch signal for those fixed states.\n\n'+markdown_table(pressure.groupby(
            ['experiment_id', 'scenario', 'teacher_action'])[['preceding_teacher_gae', 'preceding_teacher_mc_advantage']].mean().reset_index())+
        '\nDo not infer causal update pressure from the sign of raw GAE alone: PPO centers and scales '
        'advantages per minibatch. Probe and actual-rollout distributions differ.')
    if (output/'supplemental_boundary_pressure.csv').exists():
        supplement = pd.read_csv(output/'supplemental_boundary_pressure.csv')
        coverage = pd.read_csv(output/'supplemental_boundary_coverage.csv')
        sections[-1] += ('\nThe original random MC panel lacked stress hold/increase coverage. An explicitly '
            'post-outcome, validation-only diagnostic supplement samples up to four states per available teacher '
            'action and scenario. It changes neither the original bank nor any headline estimand. This is '
            'exploratory coverage repair, not preregistered confirmation. Missing action support remains missing.\n\n'+
            markdown_table(coverage)+'\n'+markdown_table(supplement[supplement.experiment_id.eq('ImitationInit')].groupby(
                ['scenario', 'teacher_action'])[['preceding_teacher_gae', 'preceding_teacher_mc_advantage', 'critic_error']].mean().reset_index())+'\n')
    if (output/'actual_teacher_pressure_summary.csv').exists():
        actual_summary = pd.read_csv(output/'actual_teacher_pressure_summary.csv')
        sections[-1] += ('\nActual normalized PPO advantages on sampled teacher-matching actions, '
            'steps 0–8,192; 95% seed-cluster bootstrap intervals. This is Markov training occupancy, '
            'not separate baseline/stress evaluation populations. Repeated samples across five epochs '
            'are not treated as independent. Labels are joined after the diagnostic replay; all selected '
            'and final tensors must equal the original runs.\n\n'+markdown_table(actual_summary[
                ['experiment_id', 'teacher_action', 'seed_count', 'normalized_mean', 'low', 'high', 'negative_fraction']])+'\n')
        preferred = actual_summary[(actual_summary.experiment_id == 'ImitationInit') &
                                   actual_summary.teacher_action.isin([2, 3, 4])]
        if len(preferred) == 3 and preferred.low.gt(0).all():
            sections[-1] += ('\nAll three non-contraction teacher-preferred actions have positive mean normalized '
                'advantages with seed-cluster intervals above zero. These observations do not support uniformly '
                'negative average PPO pressure on hold/+10%/+20%. Individual negative advantages remain common; '
                'conditional means over visited states neither determine every held-out decision nor isolate '
                'the effect of shared actor parameters, clipping and changes in occupancy.\n')
    elif (output/'actual_teacher_minibatch_pressure.csv').exists():
        actual = pd.read_csv(output/'actual_teacher_minibatch_pressure.csv')
        actual = actual[actual.teacher_action.eq(actual.sampled_action)].copy()
        actual['normalized_sum'] = actual.normalized_gae_mean*actual['count']
        actual['negative_count'] = actual.normalized_negative_fraction*actual['count']
        actual = actual.groupby(['experiment_id', 'teacher_action'])[['normalized_sum', 'negative_count', 'count']].sum()
        actual['normalized_mean'] = actual.normalized_sum/actual['count']
        actual['negative_fraction'] = actual.negative_count/actual['count']
        sections[-1] += ('\nActual PPO minibatch signal, restricted to sampled actions matching the teacher, '
            'over steps 0–8,192. This is common Markov training occupancy, not separately sampled baseline/stress '
            'evaluation populations. Samples repeat across optimization epochs and are correlated. The diagnostic '
            'replay uses the same 32,768-step budget; selected and final tensors are checked identical for all '
            'five pairs. Labels are joined only after training.\n\n'+markdown_table(actual.reset_index()[
                ['experiment_id', 'teacher_action', 'count', 'normalized_mean', 'negative_fraction']])+ '\n')
    for number, title, budget in ((11, 'Short-budget results', short), (12, 'Long-budget results', long)):
        section(number, title, 'Imitation minus random, held-out customers, validation-selected checkpoint; 95% '
            'paired seed/customer intervals. Units are EUR per customer.\n\n'+markdown_table(
                headline[headline.budget.eq(budget)][['mode', 'metric', 'scenario', 'mean', 'low', 'high']])+
            '\nThe final-policy estimates are retained separately in `statistical_comparisons.csv`. '
            'Temporal snapshots were never ranked using test outcomes.')
    section(13, 'Baseline vs severe-stress interaction',
        'Stress-minus-baseline treatment effect, matched seed/customer resampling:\n\n'+
        markdown_table(effects[['budget', 'metric', 'mean', 'low', 'high']])+
        '\nA positive NEV difference alongside a negative reward difference is possible because reward includes '
        'capital and constraint penalties beyond NEV. Neither endpoint replaces the other.')
    section(14, 'Discovery vs preservation',
        'The short-budget baseline comparison supports an initialization-sensitive discovery limitation: a useful '
        'observable boundary can remain nonconstant under the same PPO objective that yields contraction from '
        'random initialization. This does not establish exact teacher-boundary preservation or universal economic '
        'improvement under stress. Erosion and retention can coexist. Long-run similarity is descriptive: '
        'no equivalence margin was preregistered, so absence of a significant difference cannot establish LongRunConvergence.')
    section(15, 'Mechanistic interpretation',
        'The randomized paired comparison changes actor initialization only. It identifies sensitivity to that '
        'initial condition, not a unique decomposition into critic error, optimization curvature or advantage noise. '
        'Critic errors, actual action-conditioned GAE, clipping, entropy, KL and preclip gradient norms are retained '
        'in each run. Independent counterfactual comparisons use H=12, the same frozen canonical continuation, '
        'common action draws and separate selection/evaluation banks. They measure first-action opportunity, '
        'not lifetime value or an oracle policy. No post-outcome PPO tuning or auxiliary objective was introduced.')
    section(16, 'Limitations',
        'Synthetic credit environment; fitted public teacher; five training seeds; one new customer cohort; '
        'natural mixture rather than target-policy occupancy; finite and small MC panel; unadjusted correlated '
        'intervals; selected-policy and post-update timing differences; baseline-only canonical selection; '
        'deterministic/stochastic estimand differences. Confidence-gap estimates inherit teacher uncertainty. '
        'The fixed-panel GAE probes are model-based diagnostics, not measured gradients of those exact states. '
        'Imitation uses 13,537 extra labeled training visits and 600 supervised epochs outside the PPO step '
        'budget; improved PPO-step sample efficiency is not an end-to-end cost-matched learning comparison. '
        'Epoch selection and qualification share validation data, so qualification intervals can be optimistic; '
        'independent test fidelity is reported without reopening model selection. '
        'Pooled boundary survival weights initially agreeing states; sparse strata require their denominators. '
        'Test data were not used to qualify the initialization or choose checkpoints. The A/B/C and canonical '
        'artifact protection manifest is checked separately. These conclusions apply to this protocol, not deployed lending decisions.')
    section(17, 'Decision gate for next phase', '**'+', '.join(categories)+'**.\n\n'+decision['scope']+' '+decision['unresolved']+
        '\n\nPPO can retain useful observable state dependence after imitation initialization in this experiment. '
        'That does not imply zero erosion of the initial boundary, nor uniform reward gains under macro stress. '
        'Phase D ends here; no further phase or economic-model change is implemented.')
    text = '# Policy initialization and preservation — Phase D\n\n'+intro+'\n\n'+'\n'.join(sections)
    text += ('\n## Main figures\n\nSeed ranges on trajectory plots are descriptive, not confidence intervals. '
        'Decision maps are projections of observed states, not controlled two-dimensional decision slices.\n\n'+
        '\n\n'.join(f'![{name.replace("_", " ")}](../outputs/main/policy_initialization/figures/{name}.png)'
            for name in ('teacher_vs_initial', 'teacher_regret_over_time', 'contraction_share_over_time',
                         'random_vs_imitation', 'boundary_survival', 'validation_economics',
                         'baseline_vs_stress', 'decision_maps_over_time'))+'\n')
    (output/'report.md').write_text(text, encoding='utf-8')
    if publish:
        Path('docs/policy_initialization_preservation.md').write_text(text, encoding='utf-8')
        appendix = ('\n\n<!-- policy-initialization:start -->\n## Phase D — Actor initialization and preservation\n\n'
            'Imitation initialization retains state-dependent decisions under PPO and improves baseline reward '
            'relative to random initialization at both budgets. Teacher regret nevertheless increases during '
            'training, and stress reward gains are not established. The results support a short-budget discovery '
            'limitation in baseline, without establishing exact teacher-boundary preservation or a universal mechanism.'+
            '\n\nDecision gate: **'+', '.join(categories)+'**. '+decision['scope']+
            '\n\n'+markdown_table(headline[headline['mode'].eq('deterministic')][
                ['budget', 'metric', 'scenario', 'mean', 'low', 'high']])+
            '\nFull protocol, limitations and temporal diagnostics: [Phase D report](policy_initialization_preservation.md).'
            '\n<!-- policy-initialization:end -->\n')
        for path in (Path('docs/technical_paper.md'), Path('README.md')):
            addition = appendix if path.name != 'README.md' else appendix.replace(
                '(policy_initialization_preservation.md)', '(docs/policy_initialization_preservation.md)')
            content = path.read_bytes()
            marker = b'\n\n<!-- policy-initialization:start -->'
            if marker in content:
                content = content.split(marker)[0]
            path.write_bytes(content+addition.encode('utf-8'))


if __name__ == '__main__':
    main()
