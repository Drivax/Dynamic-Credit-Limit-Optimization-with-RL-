"""Regenerate Phase B figures and narrative strictly from saved diagnostic tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORE = ['MyopicEconomic', 'PPO', 'AlwaysDecrease20', 'ObservationPlanner', 'HistoryPlanner', 'FullStatePlanner']


def table(frame, digits=2):
    frame = frame.copy()
    for column in frame.select_dtypes('number'):
        frame[column] = frame[column].map(lambda x: f'{x:.{digits}f}')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(map(str, row))+' |' for row in frame.itertuples(index=False, name=None)])


def plots(output):
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    def read(name):
        return pd.read_csv(output/f'{name}.csv')
    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(folder/f'{name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    gaps = read('information_gap')
    gaps = gaps[gaps.metric == 'discounted_reward']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for column, (scenario, whole) in enumerate(gaps.groupby('scenario')):
        for row in range(2):
            ax = axes[row, column]
            group = whole if row == 0 else whole[whole.gap != 'observable_planning']
            ax.errorbar(group.difference, np.arange(len(group)),
                xerr=np.array([np.maximum(0, group.difference-group.lower), np.maximum(0, group.upper-group.difference)]), fmt='o', capsize=3)
            ax.set_yticks(np.arange(len(group)), group.gap)
            ax.axvline(0, color='gray', lw=1)
            ax.set(title=scenario+(' (detail)' if row else ''), xlabel='Discounted reward difference (EUR/customer)')
    finish(fig, 'information_gap')
    values = read('benchmark_values')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ('discounted_reward', 'net_economic_value')):
        values[values.policy.isin(CORE)].pivot(index='policy', columns='scenario', values=metric).plot.barh(ax=ax)
        ax.set(xlabel='EUR per initial customer', title=metric.replace('_', ' '), ylabel='')
    finish(fig, 'benchmark_values')
    regret = read('planner_regret')
    selected = regret[(regret.horizon == 24) & (regret.reference == 'full') & (regret.stratification == 'all')]
    fig, ax = plt.subplots(figsize=(9, 5))
    selected.pivot(index='policy', columns='scenario', values='mean').plot.barh(ax=ax)
    ax.set(xlabel='Held-out first-action regret vs fitted FullStatePlanner (EUR)', ylabel='')
    finish(fig, 'planner_regret')
    fig, ax = plt.subplots(figsize=(10, 4))
    order = ['ObservationPlanner', 'WithoutPD', 'WithoutBehavior', 'History3', 'History6', 'HistoryPlanner', 'FullStatePlanner']
    values.pivot(index='policy', columns='scenario', values='discounted_reward').reindex(order).plot.bar(ax=ax)
    ax.set(ylabel='Discounted reward (EUR/customer)', xlabel='Information set')
    ax.tick_params(axis='x', rotation=25)
    finish(fig, 'history_value')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    latents = read('latent_predictability')
    for ax, metric in zip(axes, ('r2', 'rmse')):
        latents.pivot(index='latent', columns='feature_set', values=metric).plot.barh(ax=ax)
        ax.set(title=metric, ylabel='')
    finish(fig, 'latent_predictability')
    fig, ax = plt.subplots(figsize=(10, 5))
    read('feature_importance').groupby(['group', 'feature_set']).relative_mse_increase.mean().unstack().plot.barh(ax=ax)
    ax.set(xlabel='Held-out relative-Q MSE increase after grouped permutation', ylabel='')
    finish(fig, 'feature_importance')
    training = read('ppo_training_diagnostics')
    fig, axes = plt.subplots(1, training.seed.nunique(), figsize=(14, 4), squeeze=False)
    for ax, (seed, group) in zip(axes[0], training.groupby('seed')):
        for a in range(5):
            ax.plot(group.timesteps, group[f'action_fraction_{a}'], label=f'{(a-2)*10:+d}%')
        ax.set(title=f'Seed {seed}', xlabel='Training steps', ylabel='Collected action fraction', ylim=(0, 1))
    axes[0, -1].legend()
    finish(fig, 'ppo_action_training')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for seed, group in training.groupby('seed'):
        axes[0].plot(group.timesteps, group.entropy, label=str(seed))
    for seed, group in read('checkpoint_validation').groupby('seed'):
        axes[1].plot(group.timesteps, group.validation_discounted_reward, 'o-', label=str(seed))
    axes[0].set(xlabel='Training steps', ylabel='Entropy (nats)')
    axes[1].set(xlabel='Training steps', ylabel='Validation discounted reward (EUR)')
    axes[0].legend(title='Seed')
    finish(fig, 'ppo_entropy_training')
    critic = read('critic_states')
    fig, ax = plt.subplots(figsize=(6, 5))
    for seed, group in critic.groupby('seed'):
        ax.scatter(group.target, group.prediction, s=15, alpha=.5, label=str(seed))
    lo, hi = min(critic.target.min(), critic.prediction.min()), max(critic.target.max(), critic.prediction.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set(xlabel='Independent MC stochastic-actor value (EUR)', ylabel='Critic prediction (EUR)')
    ax.legend(title='Seed')
    finish(fig, 'critic_calibration')
    counter = read('counterfactual_regret')
    counter = counter[counter.pd_bucket.astype(str) != 'all']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (scenario, group) in zip(axes, counter.groupby('scenario')):
        grid = group.pivot(index='pd_bucket', columns='utilization_bucket', values='mean')
        im = ax.imshow(grid.to_numpy(), aspect='auto', cmap='coolwarm')
        ax.set(xticks=range(len(grid.columns)), xticklabels=grid.columns,
            yticks=range(len(grid)), yticklabels=grid.index, xlabel='Utilization bucket', ylabel='PD bucket', title=scenario)
        fig.colorbar(im, ax=ax, label='Split-draw regret (EUR/decision)')
    finish(fig, 'ppo_regret_map')
    confusion = read('action_confusion')
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, mode in enumerate(('requested', 'effective')):
        for col, policy in enumerate(('ObservationPlanner', 'HistoryPlanner', 'FullStatePlanner')):
            group = confusion[(confusion['mode'] == mode) & (confusion.policy == policy)].copy()
            if mode == 'effective':
                # Keep exact admitted adjustments in CSV; display directional classes.
                for field in ('benchmark_action', 'ppo_action'):
                    group[field] = np.sign(group[field]).astype(int)
            grid = group.groupby(['benchmark_action', 'ppo_action']).weight.sum().unstack(fill_value=0)
            classes = list(range(5)) if mode == 'requested' else [-1, 0, 1]
            labels = ['-20%', '-10%', 'hold', '+10%', '+20%'] if mode == 'requested' else ['decrease', 'hold', 'increase']
            grid = grid.reindex(index=classes, columns=classes, fill_value=0)
            im = axes[row, col].imshow(grid/grid.to_numpy().sum(), aspect='auto', vmin=0, vmax=1, cmap='Blues')
            axes[row, col].set(title=f'{policy}: {mode}', xlabel='PPO action', ylabel='Benchmark action',
                xticks=range(len(grid.columns)), xticklabels=labels,
                yticks=range(len(grid)), yticklabels=labels)
            fig.colorbar(im, ax=axes[row, col], label='Weighted joint fraction')
    finish(fig, 'action_confusion')


def decision_gate(gaps):
    considered = gaps[(gaps.metric == 'discounted_reward') & gaps.gap.isin(['history', 'privileged', 'ppo_gap'])]
    supported = set(considered.loc[considered.lower > 0, 'gap'])
    if len(supported) > 1:
        return 'mixed limitation', 'A factorial experiment separating the supported information, memory and optimization factors.'
    if supported == {'history'}:
        return 'primarily memory-limited', 'A recurrent / memory-based policy experiment.'
    if supported == {'ppo_gap'}:
        return 'primarily optimization-limited', 'A PPO optimization and representation experiment.'
    if supported == {'privileged'}:
        return 'primarily information-limited', 'A DGP observability / information-design analysis, after checking approximation error.'
    return 'unresolved with current evidence', 'Increase diagnostic precision and validate the supervised Q approximation before choosing a policy or DGP intervention.'


def report(output, destination=None, update_docs=False):
    def read(name):
        return pd.read_csv(output/f'{name}.csv')
    plots(output)
    values, gaps = read('benchmark_values'), read('information_gap')
    gate_classification, recommendation = decision_gate(gaps)
    classification = gate_classification
    causal_gaps = gaps[(gaps.metric == 'discounted_reward') & gaps.gap.isin(['history', 'privileged', 'ppo_gap'])]
    reversals = [name for name, group in causal_gaps.groupby('gap') if (group.lower > 0).any() and (group.upper < 0).any()]
    if reversals:
        classification = 'unresolved with current evidence'
        recommendation += ' Stratify by macro scenario: the supported gap reverses sign across scenarios.'
    qualification = ('The directional gate alone suggests '+gate_classification+'. However, statistically signed '
        'scenario reversals in '+', '.join(reversals)+' prevent a scenario-general dominant-bottleneck claim; '
        'the overall classification is therefore unresolved.' if reversals else
        'No supported gap reverses sign with an interval wholly on the opposite side in another scenario.')
    main = values[values.policy.isin(CORE)]
    core_table = table(main[['scenario', 'policy', 'discounted_reward', 'net_economic_value', 'default_rate']])
    gap_table = table(gaps[gaps.metric == 'discounted_reward'][['scenario', 'gap', 'difference', 'lower', 'upper']])
    train = read('ppo_training_diagnostics')
    first_last = pd.concat([g.head(1).assign(stage='first') for _, g in train.groupby('seed')]+
                          [g.tail(1).assign(stage='last') for _, g in train.groupby('seed')])
    advantages = read('ppo_advantages')
    advantage_means = advantages.assign(numerator=advantages['mean']*advantages['count']).groupby('action')[['numerator', 'count']].sum()
    advantage_means['mean_raw_gae'] = advantage_means.numerator/advantage_means['count']
    critics = read('critic_diagnostics')
    counters = read('counterfactual_regret')
    top_regions = counters[counters.pd_bucket.astype(str) != 'all'].sort_values(
        ['scenario', 'weighted_total'], ascending=[True, False]).groupby('scenario').head(3)
    population = read('states').groupby('role').agg(customers=('customer_id', 'nunique'), states=('state_id', 'size'))
    selected = read('model_selection').sort_values(['feature_set', 'validation_relative_mse', 'horizon', 'leaves']).groupby('feature_set').head(1)
    alignment = read('advantage_alignment')
    matched = alignment[alignment.action == alignment.sampled_action]
    resolved = matched[abs(matched.mc_advantage) > 1.96*matched.mc_se]
    same_sign = float((np.sign(resolved.gae_raw_eur) == np.sign(resolved.mc_advantage)).mean()) if len(resolved) else np.nan
    replay = read('replay_verification')
    checkpoint = read('checkpoint_diagnostics')
    onset = checkpoint.groupby(['seed', 'timesteps']).action.apply(lambda x: (x == 0).mean()).rename('contraction_share').reset_index()
    onset = onset[onset.contraction_share >= .95].groupby('seed').head(1)
    normalized = read('ppo_normalized_advantages').groupby('action')[['normalized_sum', 'count']].sum()
    normalized['mean_normalized_advantage'] = normalized.normalized_sum/normalized['count']
    importance = read('feature_importance').groupby(['feature_set', 'group']).relative_mse_increase.mean().reset_index()
    importance = importance.sort_values(['feature_set', 'relative_mse_increase'], ascending=[True, False]).groupby('feature_set').head(3)
    def gap_sentence(label):
        rows = gaps[(gaps.gap == label) & (gaps.metric == 'discounted_reward')]
        return ' '.join(f'{r.scenario}: {r.difference:+.2f} EUR, 95% CI [{r.lower:.2f}, {r.upper:.2f}]'
                        + (' (positive interval).' if r.lower > 0 else ' (interval includes zero).' if r.upper >= 0 else ' (negative interval).')
                        for r in rows.itertuples())
    text = f'''# Information and planning gap

This is Phase B of the unchanged canonical simulator. Phase A established a
state-dependent, multi-step opportunity under privileged full-state conditioning;
it did not demonstrate attainable improvement by the observable PPO actor.
The present measurements are generated from `{output.as_posix()}`. Smoke runs
are software checks, not scientific evidence. No DGP, reward, economic coefficient,
PPO hyperparameter or canonical checkpoint has been changed.

## 1. What information does PPO actually observe?

O_t is exactly the 21 float32 entries of `build_observation`: time, limit, balance,
utilization, payment, delinquency flag/count, income, score, macro stress, PD,
spend, late-payment fraction, tenure, default, income change, macro growth and
regime indicators. X_t additionally contains four traits, income/score reversion
anchors, ordered late history and the PD model's seven-row history. H_t retains
only public observation/action prefixes. Anchors were observed at enrollment;
they should not be confused with genuinely hidden persistent traits. PD itself
already compresses observable history. No fitted planner receives future shocks,
future macro paths, outcomes, customer identities or scenario labels.

See [the predeclared protocol](information_gap_protocol.md), `feature_names.json`
and the explicit public/privileged classes in `policies/information.py`.
Train, validation and test customers are disjoint, with new populations unrelated
to canonical fitting/selection. Phase A targets are not reused to fit these models.
Training targets expand the same simulator/CRN protocol. Test targets use separate
draws and split-draw reference selection. Public planners receive only their arrays;
FullStatePlanner alone receives a privileged snapshot, without future information.

{table(population.reset_index(), 0)}

The two scenarios reuse identities and exogenous shocks for pairing. The standard
protocol uses 64 first-action MC draws per state and 32 independent stochastic
critic draws. State samples restore source-trajectory visitation through inverse
inclusion weights; whole-episode J gives each initial customer equal weight.

## 2. How much planning value exists with current observation only?

ObservationPlanner minus MyopicEconomic: {gap_sentence('observable_planning')}

J is discounted raw reward in EUR per initial customer. Net economic value is
undiscounted and excludes the capital proxy and PD penalty. All policies below
use the same test customers and shock paths, including defaults and early exit.

{core_table}

The conditional paired customer/PPO-seed bootstrap intervals are:

{gap_table}

These are empirical benchmark gaps, not an additive causal identity. Confidence
intervals condition on fitted PD/planners, sampled training states, MC targets and
fixed scenarios; they do not include refitting or simulator misspecification.
All canonical components, purchases and action distributions are saved separately
in `benchmark_values.csv`, `episode_metrics.csv` and `action_distribution.csv`.

In particular, net-value improvements need not improve the training objective:
reward additionally charges the unchanged capital proxy and PD penalty, and
discounting also differs. The net-value paired intervals are preserved in
`information_gap.csv`; objective disagreement must not be labelled PPO failure.

## 3. How much additional value does observable history provide?

HistoryPlanner minus ObservationPlanner: {gap_sentence('history')}

F3 and F4 add three/six prior observations/actions plus window mean, volatility and
trend; F5 adds whole-prefix summaries and enrollment observations. Each episode
gets fresh planner memory. F1 removes PD; F2 removes payment/delinquency/score/spend
variables, while retaining the public feasibility mask. Thus F2 is an information
ablation with a declared guardrail side channel. No latent estimates enter PPO.

{table(values[values.policy.isin(['ObservationPlanner', 'WithoutPD', 'WithoutBehavior', 'History3', 'History6', 'HistoryPlanner'])][['scenario', 'policy', 'discounted_reward']])}

Latent predictability is assessed by train-only regressors and held-out weighted
R2/RMSE. Negative R2 is retained; prediction accuracy is not decision value.

{table(read('latent_predictability')[['feature_set', 'latent', 'r2', 'rmse', 'r2_gain_over_current']])}

History can improve latent prediction without improving the policy objective.
The two claims are tested separately; no equivalence between prediction and
economic usefulness is assumed.

Grouped permutation importance is in `feature_importance.csv` and its figure.
It measures changes in held-out action-difference MSE, not causality; correlated
features and off-manifold permutations limit interpretation.

The three largest mean grouped importance values for each information set are:

{table(importance)}

The 'other current' group contains time-to-horizon, spending, tenure and the
default indicator. Their grouped importance does not identify an individual
feature effect or show that PPO ignores that feature.

## 4. How much value requires privileged latent information?

FullStatePlanner minus HistoryPlanner: {gap_sentence('privileged')}

FullStatePlanner adds the four traits, anchors, ordered late indicators and exact
available PD-history rows. It is a supervised approximation to continuation Q,
not Q*, not deployable, and not a guaranteed upper bound. Full-minus-history also
contains representation and estimation differences; it cannot prove latents are
unrecoverable. In particular, a nonsignificant gap does not establish equivalence.

Complexity and target horizon were selected only by validation relative-Q MSE:

{table(selected[['feature_set', 'horizon', 'leaves', 'validation_relative_mse']])}

Held-out approximation error is material to interpretation:

{table(read('planner_fit_diagnostics'))}

Different selected horizons and finite regressors confound pure information
comparisons. Repeated greedy execution also differs from the fixed PPO101
continuation used for targets. MC teachers condition on fixed macro scenarios,
which are not supplied as future features. These are explicit limitations of
the benchmark hierarchy, not evidence for an inaccessible-information bound.

## 5. Where does PPO sit relative to these benchmarks?

ObservationPlanner minus PPO: {gap_sentence('ppo_gap')}

Observation-minus-PPO above is the direct same-current-information comparison.
The canonical contraction result remains intact. `policy_agreement.csv` reports
requested and effective agreement with FullStatePlanner. `planner_regret.csv`
reports weighted mean/median/P90/P95 and customer-cluster intervals against both
the fitted full-state planner and split-draw MC teacher, by PD, utilization,
macro and remaining horizon. Negative held-out regret is intentionally retained.
Confusion CSVs preserve exact admitted-limit changes, including floor/cap aliases;
the effective-action figure groups these only by direction for readability.

## 6. Does PPO receive the correct long-horizon learning signal?

Original GAE buffers were not saved. An isolated replay uses canonical Markov
training populations, baseline validation, identical budget, seeds and optimizer.
It reads actual completed rollout buffers and actual normalized minibatches,
without consuming RNG draws. Replay verification:

{table(replay)}

Raw advantages averaged over collected transitions (scaled reward units):

{table(advantage_means.reset_index()[['action', 'count', 'mean_raw_gae']], 5)}

After actual minibatch normalization (all optimization epochs):

{table(normalized.reset_index()[['action', 'count', 'mean_normalized_advantage']], 5)}

`ppo_normalized_advantages.csv` additionally records the actual per-minibatch
normalization used by PPO, including repeated training epochs. Action-conditional
raw GAE means reflect state occupancy and baseline error, not causal action effects.
At predetermined rollout boundaries, `advantage_alignment.csv` compares the
sampled GAE with independent full-horizon stochastic-policy MC advantages at the
same training state and pre-update actor. There are {len(matched)} matched probes;
{len(resolved)} have |MC mean| > 1.96 MC SE, with sign agreement {same_sign:.1%} among
those probes. This small audit conditions on the realized training macro path.
Single-trajectory GAE, lambda=.95, finite rollouts, critic bootstrapping and MC
conditional expectations are different estimands; disagreement does not uniquely
identify optimization failure or a reward-scaling error.

## 7. Is the critic accurate enough?

Critic outputs are divided by the unchanged reward scale .001. Independent MC
uses the selected stochastic actor, including a probability-weighted first action,
and remaining finite horizon. Shared private action uniforms preserve CRN across
counterfactual branches. This avoids confusing a stochastic critic with the
deterministic policy used for headline evaluation. The separately labelled
deterministic comparison uses PPO101 continuation.

{table(critics[critics.stratification == 'all'][['seed', 'scenario', 'states', 'bias', 'rmse', 'r2', 'mean_mc_se']])}

`critic_states.csv` and `critic_diagnostics.csv` retain PD, utilization, horizon,
MC-preferred action and latent-profile diagnostics. Latents here are diagnostic
labels only. MC uncertainty contributes to measured RMSE and depresses R2;
the selected-action subgroup is noisy and is not a causal attribution.

## 8. Does exploration collapse?

The first saved checkpoint with at least 95% deterministic contraction on the
fixed validation observation panel is:

{table(onset)}

This bounds the time of behavioral collapse at checkpoint resolution; it is not
an exact transition time. The training-action figure gives finer rollout resolution.
Compare the validation trajectory before and after that point: a flat trajectory
with persistent contraction indicates that checkpoint selection retains an already
collapsed actor, rather than creating that behavior by selecting among different
final policies.

Actual rollout behavior at the first and final updates:

{table(first_last[['seed', 'stage', 'timesteps', 'entropy', 'action_fraction_0', 'action_fraction_3', 'action_fraction_4']], 4)}

`ppo_action_training.png` shows collected requests; `ppo_entropy_training.png`
shows entropy and the original validation trajectory. Checkpoint distributions
are evaluated on the same held-out validation observation panel. State/action
coverage is conditional on PD, utilization and elapsed-horizon buckets in
`exploration.csv`; absent rows mean no visits. Gradient norms are recorded before
the unchanged clipping operation. Original SB3 KL, clipping and explained-variance
logs are retained in the training table; SB3 logs training statistics from the
preceding update at the following rollout row. Deterministic action collapse
does not imply zero stochastic exploration. Selection uses maximum validation
discounted reward, earliest on ties, including final post-update evaluation.

## 9. Where is PPO regret concentrated?

These are genuinely PPO101-visited held-out states, never a synthetic grid:

{table(counters[counters.pd_bucket.astype(str) == 'all'][['scenario', 'mean', 'median', 'p90', 'p95', 'lower', 'upper', 'weighted_total']])}

The largest weighted regional contributions are shown below. This exploratory
ranking is computed after evaluation and does not select policies or hyperparameters.

{table(top_regions[['scenario', 'pd_bucket', 'utilization_bucket', 'states', 'mean', 'lower', 'upper', 'weighted_total']])}

These magnitudes need not reproduce Phase A's pooled-policy planning opportunity:
they condition on PPO visitation, use a first-action PPO comparison rather than
the myopic/full-horizon disagreement estimand, and have a different independent
sample and MC budget. A small PPO-occupancy regret does not erase Phase A's result.

The PD x utilization map and full CSV retain regional estimates. PD buckets are
<.2, [.2,.6), >=.6; utilization buckets are <.5, [.5,1), >=1. The teacher selects
on one half of CRN draws and evaluates on the other. Inverse inclusion weights
restore active visits in sampled source trajectories. Weighted total regret is
a sum of first-action counterfactual opportunities, **not** attainable cumulative
episode improvement. PPO202/303 same-state comparisons are not claimed to be
their own visitation distributions.

## 10. What is the dominant bottleneck?

**{classification}.** {qualification}

The mechanical gate tests whether the paired 95% interval
for History-Observation, Observation-PPO or Full-History is wholly positive in
either named scenario. It retains scenario dependence and does not correct for
multiple comparisons. Supported gaps motivate experiments, not exclusive causal
diagnoses; nonsignificance does not establish equivalence or inaccessible value.

Recommendation: **{recommendation}** No next-phase intervention is implemented.
Finite test cohorts, fixed continuation, horizon selection, supervised approximation
and limited MC precision remain potential explanations. The saved evidence must
be read jointly with these limitations rather than interpreting an approximate
planner as an information-theoretic bound.

## Reproduction and integrity

```shell
python -m credit_rl.experiments.information_gap
python -m credit_rl.experiments.information_ppo
python -m credit_rl.experiments.information_report --update-docs
```

The canonical standard artifacts must already exist. Use `--profile smoke`,
`--canonical-profile smoke`, a separate `--canonical` smoke folder and a separate
`--output` for software checks. All figures regenerate from CSV with the report
command alone. `protected_artifacts.json` verifies Phase A files and canonical
models remain byte-identical. Cached runs require matching protocol, canonical
profile, model and scientific-source identity. `information_gap --stage evaluate`
reuses fitted planners to reproduce the scientific tables without refitting.
Raw feature prefixes, MC draws and replay checkpoints stay local; compact tables,
figures and manifests are the reviewable scientific outputs.
'''
    destination = destination or output/'report.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding='utf-8')
    if update_docs:
        Path('docs/information_planning_gap.md').write_text(text, encoding='utf-8')
        concise = f'''## Information and planning gap

Phase A found state-dependent planning opportunity with privileged information.
Phase B compares planners using current observations, public history and full
state on disjoint held-out customers, alongside frozen PPO. The empirical
classification is **{classification}**; approximate planners are not optimal-policy
bounds. Current-observation planner minus PPO in discounted reward:
{gap_sentence('ppo_gap')}
See [the quantitative report](docs/information_planning_gap.md) for paired
gaps, critic/exploration diagnostics and limitations. No simulator or PPO tuning
was performed.
'''
        paper = f'''## Information and planning gap

PPO's canonical effective contraction behavior is preserved. Phase A established
that maximum contraction is not universally optimal under privileged conditional
planning. Phase B tests how much of this opportunity is recovered from the 21D
observation, observable history and a separately typed privileged state estimator.
Customer-disjoint fitting/validation/test and split-draw evaluation prevent target
reuse. Every planner is evaluated by the canonical simulator and accounting engine.

{core_table}

{gap_table}

The descriptive classification is **{classification}**. Recommendation:
{recommendation} These gaps are not a causal decomposition: supervised approximation,
validation-selected target horizons and fixed-continuation/repeated-greedy mismatch
remain confounders. FullStatePlanner is not Q* or a guaranteed bound. Intervals
condition on fitted models and MC targets. The [full report](information_planning_gap.md)
documents latent predictability, ablations, actual GAE replay, stochastic critic
calibration, conditional exploration and PPO counterfactual regret. Phase C is not
implemented and Phase A artifacts remain unchanged.
'''
        for path, block in ((Path('README.md'), concise), (Path('docs/technical_paper.md'), paper)):
            start, end = '<!-- information-gap:start -->', '<!-- information-gap:end -->'
            original = path.read_text(encoding='utf-8')
            section = start+'\n'+block+'\n'+end
            if start in original:
                original = original[:original.index(start)]+section+original[original.index(end)+len(end):]
            else:
                original = original.rstrip()+'\n\n'+section+'\n'
            path.write_text(original, encoding='utf-8')
    return classification


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/main/information_gap'))
    p.add_argument('--update-docs', action='store_true')
    args = p.parse_args()
    print(report(args.output, update_docs=args.update_docs))


if __name__ == '__main__':
    main()
