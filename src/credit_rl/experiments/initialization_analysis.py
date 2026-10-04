"""Phase D diagnostics from frozen checkpoints; no training or selection feedback."""
import json

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from credit_rl.evaluation.scenarios import make_scenarios
from credit_rl.evaluation.structural import hypothetical_paths, rollout
from credit_rl.experiments.information_ppo import distribution
from credit_rl.experiments.initialization_learning import economics
from credit_rl.experiments.initialization_metrics import panel_measurements, bucket_quality
from credit_rl.experiments.policy_initialization import (
    ROOT, populations, quality, paired_visit_interval, write_json,
)
from credit_rl.experiments.ppo_measurements import collapse_times, mc_probe
from credit_rl.experiments.main_evaluation import digest
from credit_rl.policies.baselines import SB3Policy
from credit_rl.policies.information import choose
from credit_rl.policies.decision import legal_actions
from credit_rl.experiments.information_analysis import effective_action


def concat_files(paths, destination):
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(destination, index=False)


def test_imitation(config, protocol, output, profile):
    """Called only after the validation-only initialization decision is frozen."""
    decision = json.loads((output/'initialization_selection.json').read_text())
    assert decision['test_used'] is False
    data = joblib.load(output/'dataset_test.joblib')
    teacher = joblib.load(ROOT/'information_gap/planners.joblib')['F0']['model']
    frames, metrics, buckets = [], [], []
    for item in decision['decisions']:
        name = 'Imitation_'+'x'.join(map(str, item['architecture']))
        for seed in protocol[profile]['seeds']:
            path = output/'imitation_models'/name/str(seed)/'selected.zip'
            if digest(path) != item['checkpoint_hashes'][str(seed)]:
                raise AssertionError('Qualified imitation checkpoint changed')
            model = PPO.load(path, device='cpu')
            frame, summary = quality(model, teacher, data, config, name, seed)
            frames.append(frame)
            metrics.append(summary)
            buckets.append(bucket_quality(frame, data['observations']).assign(experiment_id=name, seed=seed, split='test'))
    pd.concat(frames, ignore_index=True).to_csv(output/'imitation_test_states.csv', index=False)
    pd.concat(metrics, ignore_index=True).assign(split='test').to_csv(output/'imitation_test_quality.csv', index=False)
    pd.concat(buckets, ignore_index=True).to_csv(output/'imitation_test_buckets.csv', index=False)


def fixed_counterfactual(config, risk, data, p, output):
    path = output/'counterfactual_bank.joblib'
    if path.exists():
        return joblib.load(path)
    rng = np.random.default_rng(p['population_seed']+333)
    ids = []
    for macro in ('baseline', 'severe_stress'):
        candidates = np.flatnonzero(data['states'].scenario.eq(macro))
        ids.extend(rng.choice(candidates, min(p['mc_states_per_scenario'], len(candidates)), replace=False))
    actor = SB3Policy(PPO.load(ROOT/'standard/models/ppo_101/selected.zip', device='cpu'))
    banks = []
    for index in ids:
        snapshot = data['snapshots'][index]
        paths = hypothetical_paths(snapshot, p['mc_draws']*2, 12, p['population_seed']+int(index)*1000)
        reward = rollout(snapshot, config, risk, actor, paths, 12)[:, :, :, 6]
        banks.append((reward*.98**np.arange(reward.shape[2])).sum(2))
    result = dict(ids=np.array(ids), q=np.array(banks), draws=p['mc_draws'],
                  continuation='frozen canonical seed 101 deterministic', horizon=12)
    joblib.dump(result, path)
    return result


def analyze_budget(config, settings, protocol, risk, output, profile, budget):
    p = protocol[profile]
    data = joblib.load(output/'dataset_validation.joblib')
    teacher = joblib.load(ROOT/'information_gap/planners.joblib')['F0']['model']
    cf = fixed_counterfactual(config, risk, data, p, output)
    validation = {macro: make_scenarios(config, settings, 'validation', macro)
                  for macro in ('baseline', 'severe_stress')}
    test = {macro: group for (macro, role), group in populations(config, settings, p).items() if role == 'test'}
    destination = output/'analysis'/str(budget)
    destination.mkdir(parents=True, exist_ok=True)
    for seed in p['seeds']:
        for arm in ('RandomInit', 'ImitationInit'):
            folder = output/'runs'/str(budget)/arm/str(seed)
            target = destination/f'{arm}_{seed}'
            target.mkdir(parents=True, exist_ok=True)
            if (target/'completed.json').exists():
                continue
            snapshots = pd.read_csv(folder/'snapshots.csv')
            initial = np.load(folder/'temporal_0.npz')['probabilities']
            previous = initial
            frames, summary, economic, counterfactual, pressure = [], [], [], [], []
            prior_model = None
            metadata = json.loads((folder/'metadata.json').read_text())
            checkpoints = [(int(r.timesteps), r.checkpoint_kind, r.checkpoint)
                           for r in snapshots.itertuples()]
            checkpoints += [(metadata['selected_timesteps'], 'validation_selected', str(folder/'selected.zip')),
                            (budget, 'final', str(folder/'final.zip'))]
            for steps, kind, path in checkpoints:
                model = PPO.load(path, device='cpu')
                probabilities = distribution(model, data['observations'])[0]
                frame = panel_measurements(data, probabilities, initial, previous, config)
                frame = frame.assign(seed=seed, experiment_id=arm, budget=budget, timesteps=steps,
                    checkpoint_kind=kind, checkpoint_sha256=digest(path))
                # Fitted-teacher confidence uses distinct effective-action groups.
                gap = effective_gaps(data, config)
                frame['teacher_gap'] = gap
                frame['confidence_bucket'] = np.searchsorted(np.quantile(gap, [1/3, 2/3]), gap)
                frames.append(frame)
                for scenario, g in frame.groupby('scenario'):
                    row = dict(seed=seed, experiment_id=arm, budget=budget, timesteps=steps,
                        checkpoint_kind=kind, scenario=scenario,
                        diversity=1-g.action.value_counts(normalize=True).max(),
                        deterministic_contraction=g.contraction.mean())
                    for col in ('teacher_regret', 'stochastic_regret', 'preservation_loss', 'stochastic_preservation_loss',
                                'agreement', 'stochastic_agreement', 'entropy', 'stochastic_contraction',
                                'kl_initial', 'kl_previous', 'action_flip'):
                        row[col] = g[col].mean()
                    summary.append(row)
                economic.append(economics(model, validation, config, risk, settings, seed, arm, steps, kind).assign(
                    split='validation', budget=budget))
                for i, index in enumerate(cf['ids']):
                    q = cf['q'][i]
                    selected_action = int(q[:cf['draws']].mean(0).argmax())
                    evaluation_q = q[cf['draws']:].mean(0)
                    action, label = int(probabilities[index].argmax()), int(data['labels'][index])
                    counterfactual.append(dict(seed=seed, experiment_id=arm, budget=budget, timesteps=steps,
                        checkpoint_kind=kind, scenario=data['states'].iloc[index].scenario, state_id=int(index),
                        customer_id=data['states'].iloc[index].customer_id,
                        policy_minus_teacher=evaluation_q[action]-evaluation_q[label],
                        stochastic_minus_teacher=probabilities[index] @ evaluation_q-evaluation_q[label],
                        independent_selection_regret=evaluation_q[selected_action]-evaluation_q[action],
                        horizon=12, continuation=cf['continuation']))
                    if kind == 'post_update' and steps <= 8192 and prior_model is not None:
                        probe = mc_probe(data['snapshots'][index], prior_model, config, risk, p['mc_draws'],
                            p['population_seed']+int(index)*1000+seed, .98, .95, settings['ppo']['reward_scale'])
                        pressure.append(dict(seed=seed, experiment_id=arm, budget=budget, timesteps=steps,
                            state_id=int(index), scenario=data['states'].iloc[index].scenario, teacher_action=label,
                            initially_agreed=int(initial[index].argmax()) == label,
                            survived=action == label, preceding_teacher_gae=probe['gae'][:, label].mean(),
                            preceding_teacher_mc_advantage=probe['advantage'][:, label].mean(),
                            teacher_logit_change=frame.iloc[index].teacher_logit_change,
                            contraction_logit_change=frame.iloc[index].contraction_logit_change,
                            signal='independent fixed-state probe under previous snapshot, not actual minibatch GAE'))
                if kind in ('initial', 'post_update'):
                    previous, prior_model = probabilities, model
                if kind in ('validation_selected', 'final'):
                    economic.append(economics(model, test, config, risk, settings, seed, arm, steps, kind).assign(
                        split='test', budget=budget))
            pd.concat(frames, ignore_index=True).to_csv(target/'panel_states.csv.gz', index=False)
            pd.DataFrame(summary).to_csv(target/'trajectory.csv', index=False)
            pd.concat(economic, ignore_index=True).to_csv(target/'economic_episodes.csv', index=False)
            pd.DataFrame(counterfactual).to_csv(target/'counterfactual.csv', index=False)
            pd.DataFrame(pressure).to_csv(target/'boundary_pressure.csv', index=False)
            # Actual rollout and minibatch advantages are diagnostic-only teacher joins.
            raw = pd.read_csv(folder/'rollouts.csv.gz')
            obs = raw[[f'obs_{i}' for i in range(21)]].to_numpy(dtype=np.float32)
            labels = np.array([choose(q, o, config) for q, o in zip(teacher.predict(obs), obs)])
            raw['teacher_action'] = labels
            raw['teacher_preferred_action_sampled'] = raw.action.eq(raw.teacher_action)
            raw['experiment_id'], raw['budget'] = arm, budget
            raw.to_csv(target/'actual_advantage_pressure.csv.gz', index=False)
            write_json(target/'completed.json', dict(seed=seed, arm=arm, budget=budget))
            print(f'ANALYSIS {budget} {arm} {seed}', flush=True)
    test_imitation(config, protocol, output, profile)
    aggregate(output, p)


def effective_gaps(data, config):
    gaps = []
    for snapshot, obs, values in zip(data['snapshots'], data['observations'], data['q']):
        groups = {}
        for action in legal_actions(obs, config):
            effect = effective_action(snapshot, int(action), config)
            groups[effect] = max(groups.get(effect, -np.inf), values[action])
        distinct = sorted(groups.values(), reverse=True)
        gaps.append(distinct[0]-distinct[1] if len(distinct) > 1 else 0.)
    return np.asarray(gaps)


def aggregate(output, p):
    folders = sorted((output/'analysis').glob('*/*'))
    for source, dest in [('trajectory.csv', 'training_trajectory.csv'),
                         ('economic_episodes.csv', 'economic_results.csv'),
                         ('counterfactual.csv', 'counterfactual_results.csv'),
                         ('boundary_pressure.csv', 'advantage_pressure.csv')]:
        concat_files([f/source for f in folders], output/dest)
    trajectory = pd.read_csv(output/'training_trajectory.csv')
    trajectory[trajectory.checkpoint_kind.eq('initial')].to_csv(output/'initial_policy.csv', index=False)
    trajectory[trajectory.checkpoint_kind.isin(['validation_selected', 'final'])].to_csv(output/'checkpoint_results.csv', index=False)
    keys = ['experiment_id', 'seed', 'budget', 'scenario', 'timesteps', 'checkpoint_kind']
    trajectory[keys+['teacher_regret', 'stochastic_regret']].to_csv(output/'teacher_regret.csv', index=False)
    trajectory[keys+['kl_initial', 'kl_previous', 'action_flip']].to_csv(output/'policy_drift.csv', index=False)
    collapse, intervals, boundaries, pooled = [], [], [], []
    for budget, group in trajectory[trajectory.checkpoint_kind.isin(['initial', 'post_update'])].groupby('budget'):
        collapse.append(collapse_times(group).assign(budget=budget))
    pd.concat(collapse, ignore_index=True).to_csv(output/'collapse_timing.csv', index=False)
    for folder in folders:
        path = folder/'panel_states.csv.gz'
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        pooled.append(frame[keys+['customer_id', 'state_id', 'preservation_loss', 'stochastic_preservation_loss',
                                  'teacher_regret', 'agreement', 'stochastic_regret', 'stochastic_agreement']])
        for identity, group in frame.groupby(keys):
            row = dict(zip(keys, identity))
            for col in ('preservation_loss', 'stochastic_preservation_loss'):
                ci = paired_visit_interval(group, col, p['bootstrap_repetitions'])
                intervals.append(dict(**row, metric=col, **ci))
            for (region, confidence), g in group.groupby(['teacher_region', 'confidence_bucket']):
                eligible = g.boundary_survival.notna()
                boundaries.append(dict(**row, teacher_region=region, confidence_bucket=confidence,
                    eligible_states=int(eligible.sum()), boundary_survival=g.boundary_survival.mean()))
    pd.DataFrame(intervals).to_csv(output/'preservation.csv', index=False)
    pd.DataFrame(boundaries).to_csv(output/'boundary_survival.csv', index=False)
    combined = pd.concat(pooled, ignore_index=True)
    pooled_intervals = []
    for identity, group in combined.groupby([k for k in keys if k != 'seed']):
        for metric in ('preservation_loss', 'stochastic_preservation_loss'):
            pooled_intervals.append(dict(zip([k for k in keys if k != 'seed'], identity), metric=metric,
                **paired_visit_interval(group, metric, p['bootstrap_repetitions'])))
    pd.DataFrame(pooled_intervals).to_csv(output/'preservation_pooled.csv', index=False)
    panel_comparisons = []
    comparison_states = combined.copy()
    comparison_states.loc[comparison_states.checkpoint_kind.eq('validation_selected'), 'timesteps'] = -1
    for identity, group in comparison_states.groupby(['budget', 'scenario', 'timesteps', 'checkpoint_kind']):
        for metric in ('teacher_regret', 'agreement', 'stochastic_regret', 'stochastic_agreement'):
            pivot = group.pivot(index=['seed', 'customer_id', 'state_id'], columns='experiment_id', values=metric)
            if not {'RandomInit', 'ImitationInit'} <= set(pivot.columns):
                continue
            delta = (pivot.ImitationInit-pivot.RandomInit).dropna().rename('difference').reset_index()
            if delta.empty:
                continue
            panel_comparisons.append(dict(zip(['budget', 'scenario', 'timesteps', 'checkpoint_kind'], identity),
                metric=metric, **paired_visit_interval(delta, 'difference', p['bootstrap_repetitions'])))
    pd.DataFrame(panel_comparisons).to_csv(output/'panel_statistical_comparisons.csv', index=False)
    seed_comparisons = []
    for identity, group in trajectory.groupby(['budget', 'scenario', 'checkpoint_kind']):
        steps = [-1] if identity[-1] == 'validation_selected' else sorted(group.timesteps.unique())
        for step in steps:
            g = group if step == -1 else group[group.timesteps == step]
            for metric in ('diversity', 'entropy', 'deterministic_contraction', 'stochastic_contraction'):
                pivot = g.pivot(index='seed', columns='experiment_id', values=metric)
                delta = (pivot.ImitationInit-pivot.RandomInit).rename('difference').reset_index()
                delta['customer_id'] = 'seed_summary_only'
                seed_comparisons.append(dict(zip(['budget', 'scenario', 'checkpoint_kind'], identity),
                    timesteps=step, metric=metric, **paired_visit_interval(delta, 'difference', p['bootstrap_repetitions'])))
    pd.DataFrame(seed_comparisons).to_csv(output/'seed_statistical_comparisons.csv', index=False)
    economic = pd.read_csv(output/'economic_results.csv')
    if 'seed' not in economic:
        economic['seed'] = economic['policy_seed']
    economic.groupby(['budget', 'experiment_id', 'scenario', 'split', 'checkpoint_kind', 'mode', 'timesteps']).mean(
        numeric_only=True).to_csv(output/'economic_summary.csv')
    changes = []
    validation_economics = economic[economic.split.eq('validation')]
    for identity, group in validation_economics.groupby(['budget', 'experiment_id', 'scenario', 'mode']):
        initial = group[group.checkpoint_kind.eq('initial')].set_index(['seed', 'customer_id'])
        for kind in ('final', 'validation_selected'):
            later = group[group.checkpoint_kind.eq(kind)].set_index(['seed', 'customer_id'])
            for metric in ('discounted_reward', 'net_economic_value'):
                difference = (later[metric]-initial[metric]).rename('difference').reset_index()
                changes.append(dict(zip(['budget', 'experiment_id', 'scenario', 'mode'], identity),
                    checkpoint_kind=kind, metric=metric, **paired_visit_interval(difference, 'difference', p['bootstrap_repetitions'])))
    changes = pd.DataFrame(changes)
    changes.to_csv(output/'economic_preservation.csv', index=False)
    teacher_change = pd.DataFrame(pooled_intervals)
    quadrants = []
    for row in changes.to_dict('records'):
        metric = 'preservation_loss' if row['mode'] == 'deterministic' else 'stochastic_preservation_loss'
        matching = teacher_change[(teacher_change.experiment_id == row['experiment_id']) &
            (teacher_change.budget == row['budget']) & (teacher_change.scenario == row['scenario']) &
            (teacher_change.checkpoint_kind == row['checkpoint_kind']) & (teacher_change.metric == metric)]
        if len(matching) != 1:
            continue
        teacher_row = matching.iloc[0]
        teacher_status = 'lost' if teacher_row.low > 0 else 'improved' if teacher_row.high < 0 else 'uncertain'
        economic_status = 'improved' if row['low'] > 0 else 'worsened' if row['high'] < 0 else 'uncertain'
        quadrants.append(dict(**row, teacher_status=teacher_status, economic_status=economic_status,
            teacher_change=teacher_row['mean'], teacher_low=teacher_row.low, teacher_high=teacher_row.high,
            interpretation='No equivalence margin: CI spanning zero is uncertain, not preserved'))
    pd.DataFrame(quadrants).to_csv(output/'preservation_quadrants.csv', index=False)
    comparisons, interactions = [], []
    for identity, group in economic[economic.checkpoint_kind.isin(['validation_selected', 'final'])].groupby(
            ['budget', 'split', 'checkpoint_kind', 'mode']):
        for metric in ('discounted_reward', 'net_economic_value'):
            pivot = group.pivot(index=['seed', 'customer_id', 'scenario'], columns='experiment_id', values=metric)
            difference = (pivot.ImitationInit-pivot.RandomInit).rename('difference').reset_index()
            for scenario, g in difference.groupby('scenario'):
                comparisons.append(dict(zip(['budget', 'split', 'checkpoint_kind', 'mode'], identity),
                    metric=metric, scenario=scenario, **paired_visit_interval(g, 'difference', p['bootstrap_repetitions'])))
            interaction = difference.pivot(index=['seed', 'customer_id'], columns='scenario', values='difference')
            interaction['difference'] = interaction.severe_stress-interaction.baseline
            interactions.append(dict(zip(['budget', 'split', 'checkpoint_kind', 'mode'], identity),
                metric=metric, **paired_visit_interval(interaction.reset_index(), 'difference', p['bootstrap_repetitions'])))
    pd.DataFrame(comparisons).to_csv(output/'statistical_comparisons.csv', index=False)
    pd.DataFrame(interactions).to_csv(output/'scenario_interaction.csv', index=False)
