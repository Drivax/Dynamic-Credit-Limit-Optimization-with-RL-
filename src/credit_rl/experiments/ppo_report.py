"""Regenerate Phase C tables, thirteen figures and report from saved CSVs only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_rl.evaluation.policy_statistics import interval, matrix
from credit_rl.experiments.ppo_measurements import summarize_alignment


def read(root, name):
    path = root/f'{name}.csv'
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def table(frame, columns=None):
    """Small dependency-free Markdown table; never silently truncate results."""
    if frame.empty:
        return '*No measurements available.*'
    frame = frame[columns] if columns else frame
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f'{x:.3f}' if np.isfinite(x) else 'NA'
        return str(x).replace('|', '/')
    rows = ['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    rows += ['| '+' | '.join(map(fmt, row))+' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def paired_state_endpoints(root, protocol):
    states = read(root, 'policy_regret')
    if states.empty:
        return
    states['contraction_share'] = (states.action == 0).astype(float)
    records = []
    for (name, scenario), group in states.groupby(['experiment_id', 'scenario']):
        control = 'confirm_control' if name.startswith('confirm_') else 'canonical'
        base = states[(states.experiment_id == control) & (states.scenario == scenario)]
        for metric in ('entropy', 'contraction_share', 'teacher_regret'):
            a = group.pivot(index='seed', columns='state_id', values=metric)
            b = base.pivot(index='seed', columns='state_id', values=metric).reindex(index=a.index, columns=a.columns)
            if b.isna().any().any():
                continue
            delta = a.to_numpy()-b.to_numpy()
            weights = group.drop_duplicates('state_id').set_index('state_id').weight.reindex(a.columns).to_numpy()
            if metric != 'teacher_regret':
                weights = np.ones(len(weights))
            rng = np.random.default_rng(protocol['population_seed'])
            draws = []
            for _ in range(protocol['bootstrap_repetitions']):
                seed_indices = rng.integers(len(a), size=len(a))
                customer_indices = rng.integers(len(weights), size=len(weights))
                draws.append(np.average(delta[seed_indices][:, customer_indices], axis=1, weights=weights[customer_indices]).mean())
            lo, hi = np.quantile(draws, [.025, .975])
            records.append(dict(experiment_id=name, scenario=scenario, metric=metric, control=control,
                difference=np.average(delta, axis=1, weights=weights).mean(), lower=lo, upper=hi,
                customer_sd=delta.mean(0).std(ddof=1), seeds=len(a), customers=len(weights)))
    pd.DataFrame(records).to_csv(root/'state_endpoint_comparisons.csv', index=False)
    timing = read(root, 'collapse_timing')
    timing = timing[(timing.threshold == .95) & (timing.kind == 'deterministic_contraction')].copy()
    # Restricted first-passage time: censored runs contribute their observed budget.
    timing['restricted_time'] = timing.first_crossing.fillna(timing.last_observed)
    rows = []
    for (name, scenario), group in timing.groupby(['experiment_id', 'scenario']):
        control = 'confirm_control' if name.startswith('confirm_') else 'canonical'
        base = timing[(timing.experiment_id == control) & (timing.scenario == scenario)].set_index('seed')
        current = group.set_index('seed')
        b = base.reindex(current.index)
        if b.restricted_time.isna().any():
            continue
        # Cap both members of a pair at the shorter observation budget.
        cap = np.minimum(current.last_observed.to_numpy(), b.last_observed.to_numpy())
        delta = np.minimum(current.restricted_time.to_numpy(), cap)-np.minimum(b.restricted_time.to_numpy(), cap)
        rng = np.random.default_rng(protocol['population_seed'])
        draws = delta[rng.integers(len(delta), size=(protocol['bootstrap_repetitions'], len(delta)))].mean(1)
        lo, hi = np.quantile(draws, [.025, .975])
        rows.append(dict(experiment_id=name, scenario=scenario, difference=delta.mean(), lower=lo, upper=hi,
                         seeds=len(delta), treatment_censored=int(group.censored.sum()), estimand='paired restricted first-passage delay; seed bootstrap'))
    pd.DataFrame(rows).to_csv(root/'collapse_comparisons.csv', index=False)


def derived_tables(root, protocol):
    paired_state_endpoints(root, protocol)
    raw = read(root, 'gae_alignment_states')
    if not raw.empty:
        keys = ['experiment_id', 'seed', 'timesteps', 'scenario', 'checkpoint_kind']
        raw = raw.drop_duplicates(keys+['state_id', 'action'])
        frames = []
        for column, bins in [('pd', [.2, .6]), ('utilization', [.5, 1.]), ('remaining_horizon', [6, 12])]:
            for bucket in range(3):
                group = raw[np.searchsorted(bins, raw[column]) == bucket]
                if not group.empty:
                    frames.append(summarize_alignment(group).assign(region_variable=column, region_bucket=bucket))
        pd.concat(frames, ignore_index=True).to_csv(root/'alignment_by_region.csv', index=False)
        regions = []
        for key, group in raw.groupby(keys):
            best = group.loc[group.groupby('state_id').q_mean.idxmax()]
            for label, selected in [('all', best), ('mc_best_not_contraction', best[best.action != 0])]:
                error = selected.value_prediction-selected.value_mc
                regions.append(dict(zip(keys, key)) |
                    dict(region=label, states=len(selected), bias=error.mean(), rmse=np.sqrt((error**2).mean())))
        pd.DataFrame(regions).to_csv(root/'critic_by_region.csv', index=False)
        opportunity = []
        for key, group in raw.groupby(keys):
            states = []
            for _, state in group.groupby('state_id'):
                best = state.loc[state.q_mean.idxmax()]
                center = float(np.dot(state.probability, state.gae_mean))
                states.append(dict(best_is_contraction=int(best.action == 0),
                    best_gain_vs_contraction=best.q_mean-float(state[state.action == 0].q_mean.iloc[0]),
                    centered_gae_rmse=np.sqrt(np.mean((state.gae_mean-center-state.mc_advantage)**2)),
                    noncontraction_positive_mc=bool(((state.action != 0) & (state.mc_advantage > 0)).any()),
                    noncontraction_above_two_se=bool(((state.action != 0) & (state.mc_advantage > 2*state.mc_se)).any())))
            opportunity.append(dict(zip(keys, key)) |
                               pd.DataFrame(states).mean().to_dict())
        pd.DataFrame(opportunity).to_csv(root/'mc_action_opportunity.csv', index=False)
    updates = read(root, 'training_updates')
    if not updates.empty:
        panels = read(root, 'action_distribution')
        final_panel = panels.sort_values('timesteps').groupby(['experiment_id', 'seed', 'scenario']).tail(1)
        final_update = updates.sort_values('timesteps').groupby(['experiment_id', 'seed']).tail(1)
        final_panel.merge(final_update, on=['experiment_id', 'seed', 'timesteps'], suffixes=('_panel', '_rollout')).to_csv(
            root/'final_training_diagnostics.csv', index=False)
        early = updates[(updates.experiment_id == 'canonical') & (updates.timesteps <= 2048)].copy()
        early['contraction_gae_rank'] = early[[f'normalized_gae_{a}' for a in range(5)]].rank(axis=1, ascending=False)["normalized_gae_0"]
        early[['seed', 'timesteps', 'contraction_gae_rank']+[f'normalized_gae_{a}' for a in range(5)]].to_csv(root/'early_advantage_ordering.csv', index=False)
    exploration = read(root, 'exploration')
    if not exploration.empty:
        keys = ['experiment_id', 'seed', 'timesteps']
        buckets = ['pd_bucket', 'utilization_bucket', 'horizon_bucket']
        coverage = exploration.drop_duplicates(keys+buckets).groupby(keys).size().rename('occupied_state_buckets')
        coverage = coverage.to_frame().join(exploration.groupby(keys).size().rename('occupied_state_action_buckets'))
        coverage.reset_index().to_csv(root/'state_coverage.csv', index=False)
    episodes = read(root, 'episode_metrics')
    if episodes.empty:
        return
    reference = read(root, 'reference_episodes')
    imitation = read(root, 'imitation_episodes')
    all_episodes = pd.concat([episodes, reference, imitation], ignore_index=True)
    means = all_episodes.groupby(['policy', 'policy_seed', 'scenario'])[
        ['discounted_reward', 'net_economic_value', 'defaulted', 'credit_loss', 'capital_charge']].mean().reset_index()
    means.to_csv(root/'all_policy_results.csv', index=False)
    comparisons = []
    if not reference.empty:
        for (name, macro), group in all_episodes.groupby(['policy', 'scenario']):
            a = matrix(group, 'discounted_reward')
            ref = reference[(reference.policy == 'AlwaysDecrease20') & (reference.scenario == macro)]
            base = ref.set_index('customer_id').discounted_reward.reindex(a.columns).to_numpy()
            delta = a.to_numpy()-base
            lo, hi = interval(delta, protocol['bootstrap_repetitions'], protocol['population_seed'])
            comparisons.append(dict(policy=name, scenario=macro, reward_difference=delta.mean(), lower=lo, upper=hi,
                                    seeds=len(a), customers=len(a.columns)))
        pd.DataFrame(comparisons).to_csv(root/'paired_vs_constant.csv', index=False)
    finals = read(root, 'final_episode_metrics')
    if not finals.empty:
        finals.groupby(['policy', 'policy_seed', 'scenario'])[['discounted_reward', 'net_economic_value']].mean().reset_index().to_csv(root/'final_policy_results.csv', index=False)
    # Paired fixed-state regret differences, with customers and seeds resampled jointly.
    counterfactual = read(root, 'counterfactual_states')
    if not counterfactual.empty:
        records = []
        for (name, scenario), group in counterfactual.groupby(['experiment_id', 'scenario']):
            a = group.pivot(index='seed', columns='state_id', values='regret_mc').sort_index(axis=1)
            reference = counterfactual[(counterfactual.experiment_id == 'AlwaysDecrease20') &
                                      (counterfactual.scenario == scenario)].set_index('state_id')
            delta = a.to_numpy()-reference.regret_mc.reindex(a.columns).to_numpy()
            weights = reference.weight.reindex(a.columns).to_numpy()
            rng = np.random.default_rng(protocol['population_seed'])
            draws = []
            for _ in range(protocol['bootstrap_repetitions']):
                seeds = rng.integers(len(a), size=len(a))
                customers = rng.integers(len(weights), size=len(weights))
                draws.append(np.average(delta[seeds][:, customers], axis=1, weights=weights[customers]).mean())
            lo, hi = np.quantile(draws, [.025, .975])
            records.append(dict(experiment_id=name, scenario=scenario,
                regret_difference=np.average(delta, axis=1, weights=weights).mean(), lower=lo, upper=hi,
                seeds=len(a), customers=len(weights), comparison='regret minus AlwaysDecrease20; negative is better'))
        pd.DataFrame(records).to_csv(root/'paired_counterfactual_regret.csv', index=False)


def save(fig, root, name):
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(root/'figures'/f'{name}.png', dpi=150)
    plt.close(fig)


def lines(ax, frame, x, y, group='seed', title=None):
    if frame.empty or y not in frame:
        ax.text(.5, .5, 'No observations', ha='center', transform=ax.transAxes)
    else:
        for name, g in frame.groupby(group):
            g = g.groupby(x)[y].mean().sort_index()
            ax.plot(g.index, g, label=str(name), linewidth=1.1)
        ax.legend(fontsize=7, ncol=2)
    ax.set(xlabel=x, ylabel=y, title=title)
    ax.grid(alpha=.2)


def figures(root):
    (root/'figures').mkdir(exist_ok=True)
    c = read(root, 'canonical_training')
    if c.empty:
        raise ValueError('Canonical training CSV is required')
    fig, axes = plt.subplots(10, 1, figsize=(11, 24), sharex=True)
    for ax, prefix in zip(axes[:2], ['raw_gae', 'normalized_gae']):
        means = c[c.scenario == 'baseline'].groupby('timesteps').mean(numeric_only=True)
        for action in range(5):
            ax.plot(means.index, means[f'{prefix}_{action}'], label=f'action {action}')
        ax.set_ylabel(prefix+'; seed mean')
        ax.legend(ncol=5, fontsize=7)
    measures = ['deterministic_contraction', 'stochastic_contraction', 'entropy_panel', 'action_fraction_0',
                'critic_gae_rmse', 'train/policy_gradient_loss', 'discounted_reward']
    for ax, y in zip(axes[2:9], measures):
        lines(ax, c[c.scenario == 'baseline'], 'timesteps', y)
        ax.set_xlim(0, c.timesteps.max())
    coverage = read(root, 'state_coverage')
    lines(axes[9], coverage[coverage.experiment_id == 'canonical'], 'timesteps', 'occupied_state_action_buckets')
    independent = read(root, 'critic_diagnostics')
    independent = independent[(independent.experiment_id == 'canonical') & (independent.scenario == 'baseline')]
    if 'checkpoint_kind' in independent:
        independent = independent[independent.checkpoint_kind.str.startswith('time_')]
    if not independent.empty:
        points = independent.groupby('timesteps').critic_rmse.mean()
        axes[6].scatter(points.index, points, marker='x', color='black', label='Independent MC mean; sparse')
        axes[6].legend(fontsize=7, ncol=2)
    fig.suptitle('Canonical temporal order; diagnostic association, not causal identification')
    save(fig, root, 'collapse_timeline')
    panel = read(root, 'action_distribution')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scenario in zip(axes, ['baseline', 'severe_stress']):
        g = panel[(panel.experiment_id == 'canonical') & (panel.scenario == scenario)].groupby('timesteps').mean(numeric_only=True)
        for a in range(5):
            ax.plot(g.index, g[f'probability_{a}'], label=f'action {a}')
        ax.set(title=scenario, xlabel='Training steps', ylabel='Mean stochastic probability')
        ax.legend()
    save(fig, root, 'action_distribution_training')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scenario in zip(axes, ['baseline', 'severe_stress']):
        lines(ax, c[c.scenario == scenario], 'timesteps', 'entropy_panel', title=scenario)
    save(fig, root, 'entropy_training')
    raw = read(root, 'gae_alignment_states')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scenario in zip(axes, ['baseline', 'severe_stress']):
        g = raw[(raw.experiment_id == 'canonical') & (raw.scenario == scenario)]
        ax.scatter(g.mc_advantage, g.gae_mean, s=8, alpha=.3)
        ax.axhline(0, color='grey', linewidth=.5)
        ax.axvline(0, color='grey', linewidth=.5)
        ax.set(title=scenario, xlabel='Independent MC advantage (EUR)', ylabel='Frozen critic GAE (EUR)')
    save(fig, root, 'gae_vs_mc')
    critic = read(root, 'critic_diagnostics')
    if 'checkpoint_kind' in critic:
        critic = critic[critic.checkpoint_kind.str.startswith('time_')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scenario in zip(axes, ['baseline', 'severe_stress']):
        lines(ax, critic[(critic.experiment_id == 'canonical') & (critic.scenario == scenario)], 'timesteps', 'critic_rmse', title=scenario)
    save(fig, root, 'critic_error')
    timing = read(root, 'collapse_timing')
    for family, name in [('advantage', 'collapse_by_lambda'), ('exploration', 'collapse_by_entropy')]:
        ids = panel[panel.family.isin([family, 'canonical'])].experiment_id.unique()
        g = timing[timing.experiment_id.isin(ids) & (timing.threshold == .95) & (timing.kind == 'deterministic_contraction')].copy()
        g['observed_or_censored'] = g.first_crossing.fillna(g.last_observed)
        fig, ax = plt.subplots(figsize=(10, 4))
        for index, (label, group) in enumerate(g.groupby('experiment_id')):
            ax.scatter(np.full(len(group), index), group.observed_or_censored, marker='o', label=label)
            censored = group[group.censored]
            ax.scatter(np.full(len(censored), index), censored.last_observed, marker='x', color='black')
        labels = sorted(g.experiment_id.unique())
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel='First >95% contraction (steps)', title='Black x: right-censored; seed × scenario observations')
        save(fig, root, name)
    results = read(root, 'intervention_results')
    final = read(root, 'final_policy_results')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scenario in zip(axes, ['baseline', 'severe_stress']):
        g = results[results.family.isin(['canonical', 'budget']) & (results.scenario == scenario)]
        lines(ax, g, 'budget', 'discounted_reward', title=scenario+'; selected actor')
        if not final.empty:
            lookup = g.drop_duplicates('experiment_id').set_index('experiment_id').budget
            f = final[final.scenario == scenario].copy()
            f['budget'] = f.policy.str.removesuffix('_final').map(lookup)
            means = f.groupby('budget').discounted_reward.mean()
            ax.plot(means.index, means, 'k--', label='Final actor mean')
            ax.legend(fontsize=7)
    save(fig, root, 'budget_effect')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scenario in zip(axes, ['baseline', 'severe_stress']):
        g = results[results.family.isin(['canonical', 'representation']) & (results.scenario == scenario)]
        g.groupby('experiment_id').diversity.mean().plot.bar(ax=ax)
        ax.set(title=scenario, ylabel='1 − largest deterministic action share')
    save(fig, root, 'architecture_effect')
    imitation = read(root, 'imitation')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scenario in zip(axes, ['baseline', 'severe_stress']):
        if not imitation.empty:
            imitation[imitation.scenario == scenario].groupby('experiment_id')[['accuracy', 'effective_accuracy', 'diversity']].mean().plot.bar(ax=ax)
        ax.set(title=scenario, ylim=(0, 1))
    save(fig, root, 'planner_imitation')
    states = read(root, 'policy_regret')
    imitation_states = read(root, 'imitation_states')
    important = list(results[results.family.isin(['canonical', 'critic', 'representation', 'budget', 'learning_rate', 'confirmation'])].experiment_id.unique())
    important = ['canonical']+sorted(set(important)-{'canonical'})
    for name, color in [('decision_maps', 'action'), ('regret_maps', 'teacher_regret')]:
        fig, axes = plt.subplots(len(important)+2, 2, figsize=(12, 3.2*(len(important)+2)))
        for column, scenario in enumerate(['baseline', 'severe_stress']):
            canonical = states[(states.experiment_id == 'canonical') & (states.seed == 101) & (states.scenario == scenario)]
            teacher = canonical.copy()
            teacher['action'], teacher['teacher_regret'] = teacher.teacher_action, 0.
            imitated = imitation_states[(imitation_states.experiment_id == 'Imitation_64x64') &
                                        (imitation_states.seed == 101) & (imitation_states.scenario == scenario)] if not imitation_states.empty else canonical.iloc[:0]
            maps = []
            for policy in important:
                group = states[(states.experiment_id == policy) & (states.scenario == scenario)]
                first_seed = group.seed.min()
                maps.append((f'{policy}, seed{first_seed}', group[group.seed == first_seed]))
            maps += [('ObservationPlanner F0', teacher), ('Imitation 64×64, seed101', imitated)]
            for row, (label, g) in enumerate(maps):
                ax = axes[row, column]
                points = ax.scatter(g.pd, g.utilization, c=g[color], s=15, cmap='viridis',
                                    vmin=0, vmax=4 if color == 'action' else max(states.teacher_regret.max(), 1))
                fig.colorbar(points, ax=ax, label=color)
                ax.set(title=scenario+'; '+label, xlabel='Observable PD', ylabel='Utilization')
        save(fig, root, name)
    comparisons = read(root, 'paired_vs_constant')
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    if not comparisons.empty:
        names = sorted(comparisons.policy.unique())
        for ax in axes:
            for offset, scenario in [(-.15, 'baseline'), (.15, 'severe_stress')]:
                g = comparisons[comparisons.scenario == scenario].set_index('policy').reindex(names)
                ax.errorbar(np.arange(len(names))+offset, g.reward_difference,
                    yerr=[g.reward_difference-g.lower, g.upper-g.reward_difference], fmt='o', label=scenario, markersize=3)
            ax.legend(fontsize=8)
        axes[1].set_xticks(range(len(names)), names, rotation=90)
        detail = comparisons[comparisons.policy != 'MyopicEconomic']
        lower, upper = min(float(detail.lower.min()), -10.), max(float(detail.upper.max()), 10.)
        margin = .1*(upper-lower)
        axes[1].set_ylim(lower-margin, upper+margin)
    for ax in axes:
        ax.axhline(0, color='black', linewidth=.6)
        ax.set_ylabel('Reward difference vs contraction (EUR)')
    axes[0].set_title('Full scale; paired 95% intervals; exploratory, not multiplicity-adjusted')
    axes[1].set_title('Detail: MyopicEconomic falls outside this view; full interval shown above')
    save(fig, root, 'baseline_vs_stress')


HEADINGS = [
    'Canonical PPO training dynamics', 'When does collapse occur?',
    'Does the advantage estimator favor contraction?', 'Does GAE agree with Monte-Carlo action values?',
    'Is critic error responsible?', 'Does exploration disappear too quickly?',
    'Does additional training recover state dependence?', 'Is actor representation sufficient?',
    'Can the same network imitate ObservationPlanner?', 'Which interventions alter collapse?',
    'Do those interventions improve counterfactual regret?', 'Are effects robust across macro scenarios?',
    'Mechanistic interpretation', 'Limitations', 'Decision gate for the next phase']


def report(root, profile, destination):
    registry = read(root, 'experiment_registry')
    timing = read(root, 'collapse_timing')
    canonical_timing = timing[(timing.experiment_id == 'canonical') & (timing.threshold == .95)]
    results = read(root, 'intervention_results')
    alignment = read(root, 'gae_alignment')
    imitation = read(root, 'imitation')
    eligibility = read(root, 'confirmation_eligibility')
    confirmation = read(root, 'confirmatory_results')
    pairing = read(root, 'paired_vs_constant')
    early = read(root, 'early_advantage_ordering')
    canonical_alignment = alignment[alignment.experiment_id == 'canonical']
    if 'checkpoint_kind' in canonical_alignment:
        canonical_alignment = canonical_alignment[canonical_alignment.checkpoint_kind.str.startswith('time_')]
    chosen_alignment = canonical_alignment.groupby(['scenario', 'timesteps'])[
        ['sign_agreement', 'ranking_accuracy', 'critic_rmse']].mean().reset_index()
    family_means = results.groupby(['family', 'experiment_id', 'scenario'])[
        ['discounted_reward', 'diversity', 'teacher_regret']].mean().reset_index()
    texts = [
        'The original canonical result is retained. Fine replay uses the unchanged 21 public observations, separate Tanh actor/critic MLPs '
        '(64×64), one Markov training environment, 32,768 steps, γ=.98, λ=.95, five epochs, batch 128, rollout 512, Adam .0003, '
        'entropy .01, clipping .2 and reward scale .001. Canonical seeds 101/202/303 are checked against original selected and final weights; '
        '404/505 extend replication. `canonical_ppo_configuration.csv` records the full implementation audit. '
        'Fine eight-customer validation is diagnostic only; the unchanged full 100-customer baseline validation selects the checkpoint. '
        'That canonical selection occurs at rollout boundaries before the pending update, plus after the final update; temporal snapshots '
        'are after updates. Selected and temporal MC probes remain separate even when their step labels coincide. '
        'All reported primary policies use that selected checkpoint.\n\n'+table(registry.groupby(['family', 'status']).size().rename('runs').reset_index()),
        'Thresholds are strict first passages, measured after updates on the same observation panel. Deterministic argmax collapse and '
        'loss of stochastic exploration are different events. NA denotes right censoring, not absence of all contraction. Recrossings are retained.\n\n'+
        table(canonical_timing, ['seed', 'scenario', 'kind', 'first_crossing', 'censored', 'recrossings']),
        'Actual rollout advantages and the actual minibatch-normalized values are recorded before optimization; gradient norms are captured '
        'at the real clipping call. Across-action conditional averages also reflect which states sampled each action. '
        'They must not be read as an unbiased action comparison. Early contraction ranks (1 = largest normalized GAE) are:\n\n'+
        table(early.groupby('seed')[['contraction_gae_rank', 'normalized_gae_0']].mean().reset_index())+
        '\n\n`actual_training_gae.csv` compares actual first-transition GAE with independent MC using its behavior checkpoint; '
        '`training_updates.csv` retains all five action means, KL, clipping, losses, explained variance and gradients.',
        'Frozen-state probes use separate random draw banks for GAE and independent MC Q; each action receives common random numbers. '
        'The continuation is the checkpoint’s stochastic policy, and MC advantage is Q−ΣπQ. These full-episode frozen-critic probes '
        'are distinct from finite-rollout, sampled training GAE. Validation panels contain eight states per scenario; '
        'a separate six-state training panel is also saved. Correlations, sign, ranking and calibration errors are in the CSVs.\n\n'+table(chosen_alignment),
        'The baseline networks already have separate learned actor/critic parameters; a second “separate critic” treatment is therefore N/A. '
        'C1 enlarges only the critic while preserving initial actor weights. C3 adds five critic-only optimization passes. '
        'C4 is a supervised public-state value benchmark, never substituted into PPO. Its deterministic continuation differs from the '
        'stochastic policy that trained the PPO critic, so its error comparison is diagnostic rather than a controlled causal test.\n\n'+
        table(read(root, 'mc_value_benchmark'))+'\n\n'+table(family_means[family_means.family.isin(['canonical', 'critic'])]),
        'Constant entropy interventions precede the fixed .02→.01 linear schedule. The entropy plot and stochastic contraction thresholds '
        'measure exploration separately from greedy action collapse. Bucket visitation/action counts are retained in `exploration.csv`; '
        'increased entropy alone is not evidence that economically relevant alternatives were learned.\n\n'+
        table(family_means[family_means.family.isin(['exploration', 'entropy_schedule'])]),
        'Budgets are fixed before observing test outcomes. Selected checkpoints and final actors are both evaluated; '
        'a larger training budget can leave the validation-selected actor unchanged.\n\n'+
        table(family_means[family_means.family == 'budget'])+'\n\nFinal actor economic outcomes:\n\n'+table(read(root, 'final_policy_results')),
        'Actor-only architecture treatments preserve initial critic parameters. Diversity and action–public-bucket mutual information '
        'describe state dependence; neither establishes optimality. Per-scenario macro MI can be zero when macro is constant.\n\n'+
        table(family_means[family_means.family.isin(['canonical', 'representation'])]),
        'The exact SB3 Tanh actor architecture is trained by supervised cross-entropy on frozen ObservationPlanner F0 labels. '
        'Only the original 21 public inputs are supplied. Epoch selection uses validation cross-entropy; all five seeds and both fixed '
        'architectures are retained. Requested agreement, effective-action agreement and economic teacher regret are separate endpoints. '
        'Partial imitation demonstrates representational signal but does not prove that the network can represent the full planner.\n\n'+
        table(imitation.groupby(['experiment_id', 'scenario'])[['accuracy', 'effective_accuracy', 'diversity', 'teacher_regret']].mean().reset_index()) if not imitation.empty else 'Imitation measurements unavailable.',
        'Every preregistered OFAT result, including failures to improve, appears below. Discount treatments use a common .98 evaluation '
        'discount and additionally retain their own training objective. Exploratory comparisons are not adjusted for multiple testing.\n\n'+
        table(family_means)+'\n\nEligibility uses validation regret, MC ranking and collapse delay only; test outcomes cannot select factors.\n\n'+
        table(eligibility)+'\n\nFresh-seed factorial confirmation (or explicit no-trigger result):\n\n'+table(confirmation),
        'Counterfactual regret uses a common frozen deterministic PPO101 continuation. The first independent half-bank chooses the reference '
        'action and the second evaluates it; negative estimates are possible from noise and selection error. This is a one-step deviation '
        'estimand, not optimal-policy lifetime regret. Public teacher regret collapses economically equivalent floor/cap/guard actions; '
        'requested-action regret remains available separately.\n\n'+
        table(read(root, 'paired_counterfactual_regret'))+
        '\n\nIntervals resample seeds and paired customers jointly with visitation weights; negative differences mean lower regret than constant contraction.',
        'The new test cohort has 100 customers under paired baseline/stress paths. Intervals resample paired customers and seeds, '
        'and preserve each customer across treatments. Scenario interaction estimates are in `scenario_comparison.csv`. '
        'No baseline/stress pooling is used to declare success.\n\n'+table(pairing),
        'The aligned timeline can support a sequence of advantage preference, policy concentration, entropy change and reduced sampling. '
        'Temporal precedence alone does not identify causation. Controlled OFAT changes and fresh-seed confirmation must agree with '
        'independent advantage/regret evidence before assigning a specific mechanism. Supervised imitation tests the architecture under '
        'a different optimization objective; it is not evidence that PPO can learn the same mapping with its reward signal.',
        'Three seeds cover most OFATs and five cover the canonical, critic and actor-capacity treatments. Monte Carlo state samples are '
        'small (eight validation states/scenario, six fixed training states), with 32 draws per independent bank. The regret benchmark '
        'uses 32 total draws split into halves and has wide intervals. Test-state visitation is a fixed mixture of three policies, not '
        'the state distribution of every treatment. The teacher is fitted and imperfect; latent simulator state is used only by the '
        'diagnostic oracle. None of these probes is deployable future information. MC alignment rankings ignore economic ties and can '
        'be noisy. No multiple-testing correction or best-test configuration selection is claimed. Smoke results only validate software.',
        '**Unresolved** is the conservative gate until the complete intervention and confirmation evidence is reviewed. '
        'The decomposition into advantage estimation, critic error, exploration, representation and optimization dynamics is a research '
        'framework, not an additive mathematical identity. No new algorithm, DGP modification or next-phase implementation is introduced.'
    ]
    gate_path = root/'decision_gate.json'
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding='utf-8'))
        texts[12] = gate['mechanistic_interpretation']
        texts[14] = gate['decision_text']
    if not confirmation.empty:
        texts[9] += '\n\nConfirmation of the mechanism itself, on fresh paired seeds:\n\n'+table(read(root, 'confirmatory_mechanisms'))
    title = '# PPO optimization diagnosis — Phase C\n\n'
    title += f'Profile: **{profile}**. Generated exclusively from saved CSVs. '
    title += 'See [the frozen protocol](ppo_diagnostics_protocol.md) for hypotheses, selection rules and estimands.\n'
    validation_path = root/'validation.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        title += f'\nSoftware validation: **{validation["tests_passed"]} tests passed**, coverage **{validation["coverage_percent"]:.2f}%**, Ruff passed.\n'
    verification_path = root/'verification.json'
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        title += f'\nReproducibility: {verification["protected_files"]} protected files unchanged, {verification["snapshots"]} snapshot hashes checked, canonical weights identical, and CSV/figure regeneration byte-identical.\n'
    if profile == 'smoke':
        title += '\n**Software validation only. Narrative describes the standard design; smoke uses the reduced counts in its manifest and is not scientific evidence.**\n'
    if not registry.status.eq('complete').all():
        title += '\n**INCOMPLETE RUN: some preregistered experiments are still pending. This report is a draft, not a final mechanism classification.**\n'
    confirm_registry = read(root, 'confirmatory_registry')
    if not confirm_registry.empty and not confirm_registry.status.isin(['complete', 'not_triggered']).all():
        title += '\n**Factorial confirmation is still pending.**\n'
    for i, (heading, text) in enumerate(zip(HEADINGS, texts), 1):
        title += f'\n## {i}. {heading}\n\n{text}\n'
    title += '\n## Reproduction\n\nRun commands from the repository root using the project Python environment:\n\n```powershell\n'
    for module in ['ppo_diagnostics --family all', 'ppo_analysis --workers 3', 'ppo_imitation', 'ppo_counterfactual',
                   'ppo_confirmation', 'ppo_analysis --stage aggregate', 'ppo_counterfactual', 'ppo_report', 'ppo_verify']:
        title += f'python -m credit_rl.experiments.{module}\n'
    title += 'python -m ruff check .\npython -m pytest --cov=credit_rl --cov-report=term-missing --cov-fail-under=70\n```\n\n'
    title += 'For smoke, use `--profile smoke --canonical outputs/main/information_canonical_smoke '
    title += '--phase-b outputs/main/information_gap_smoke --output outputs/main/ppo_diagnostics_smoke` on experiment commands. '
    title += 'The report command needs only `--profile smoke --output outputs/main/ppo_diagnostics_smoke`. '
    title += 'Raw checkpoints and Monte Carlo caches remain local. The report and all thirteen figures regenerate without training or model loading.\n'
    title += '\n## Figures\n\n'
    for path in sorted((root/'figures').glob('*.png')):
        import os
        relative = Path(os.path.relpath(path, destination.parent)).as_posix()
        title += f'![{path.stem.replace("_", " ")}]({relative})\n\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(title, encoding='utf-8')


def run(args):
    manifest = json.loads((args.output/'preregistration.json').read_text())
    if args.profile != manifest['profile']:
        raise ValueError('Report profile differs from the frozen run')
    protocol = manifest['protocol'][args.profile]
    diagnostics = [dict(experiment_id='C2_separate_features', status='not_applicable',
        reason='Canonical actor and critic already have separate learned feature networks')]
    for path in sorted((args.output/'imitation_models').glob('*/*/selection.json')):
        selected = json.loads(path.read_text())
        diagnostics.append(dict(experiment_id=path.parent.parent.name, seed=selected['seed'], status='complete',
            budget=protocol['imitation_epochs'], selected_epoch=selected['selected_epoch'],
            parameters=json.dumps(manifest['protocol']['imitation'], sort_keys=True), reason='public-state supervised diagnostic'))
    value_provenance = args.output/'counterfactual_provenance.json'
    if value_provenance.exists():
        specification = json.loads(value_provenance.read_text())['specification']
        diagnostics.append(dict(experiment_id='C4_supervised_mc_value', seed=specification['seed'], status='complete',
            budget=specification['value_predictor']['max_iter'], parameters=json.dumps(specification['value_predictor'], sort_keys=True),
            reason='fixed-implementation benchmark; never substituted into PPO'))
    pd.DataFrame(diagnostics).to_csv(args.output/'diagnostic_registry.csv', index=False)
    derived_tables(args.output, protocol)
    figures(args.output)
    destination = args.report or (Path('docs/ppo_optimization_diagnosis.md') if args.profile == 'standard' else args.output/'report.md')
    report(args.output, args.profile, destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    p.add_argument('--output', type=Path, default=Path('outputs/main/ppo_diagnostics'))
    p.add_argument('--report', type=Path)
    run(p.parse_args())


if __name__ == '__main__':
    main()
