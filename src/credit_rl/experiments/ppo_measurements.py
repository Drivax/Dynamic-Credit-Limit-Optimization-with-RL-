"""Shared-state, scenario-stratified measurements for preregistered PPO interventions."""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from stable_baselines3 import PPO

from credit_rl import CreditLimitEnv
from credit_rl.envs.observation import build_observation
from credit_rl.evaluation.policy_engine import PolicySpec, evaluate_policy
from credit_rl.evaluation.structural import DecisionSnapshot, hypothetical_paths, rollout
from credit_rl.experiments.information_analysis import effective_action
from credit_rl.experiments.information_ppo import distribution, StochasticContinuation
from credit_rl.experiments.structural_diagnostics import describe
from credit_rl.experiments.ppo_learning import state_dependence
from credit_rl.policies.baselines import SB3Policy
from credit_rl.policies.decision import MyopicEconomic, legal_actions, decode
from credit_rl.envs.constraints import effective_limit
from credit_rl.policies.information import ObservationPlanner, choose


def collapse_times(panel, thresholds=(.5, .75, .9, .95)):
    rows = []
    for (experiment, seed, scenario), group in panel.groupby(['experiment_id', 'seed', 'scenario']):
        group = group.sort_values('timesteps')
        for kind in ('deterministic_contraction', 'stochastic_contraction'):
            for threshold in thresholds:
                passed = group[kind] > threshold
                first = group.loc[passed, 'timesteps'].min() if passed.any() else np.nan
                recrossings = int(((~passed) & passed.shift(1, fill_value=False)).sum())
                rows.append(dict(experiment_id=experiment, seed=seed, scenario=scenario, kind=kind,
                    threshold=threshold, first_crossing=first, censored=not passed.any(),
                    last_observed=group.timesteps.max(), recrossings=recrossings))
    return pd.DataFrame(rows)


def gae_from_arrays(rewards, values, gamma, lam):
    """Finite-episode, frozen-critic GAE; inactive/default/horizon tails are zero."""
    next_values = np.concatenate([values[:, :, 1:], np.zeros_like(values[:, :, :1])], axis=2)
    deltas = rewards+gamma*next_values-values
    return np.sum(deltas*(gamma*lam)**np.arange(deltas.shape[2]), axis=2)


class ValuedContinuation(StochasticContinuation):
    def __init__(self, model, draws, horizon, seed, initial_value):
        super().__init__(model, draws, horizon, seed)
        self.values = np.zeros((draws, 5, horizon))
        self.values[:, :, 0] = initial_value

    def act_indexed(self, observations, active, month, action_count):
        probs, _, _, values = distribution(self.actor, observations)
        active = np.asarray(active)
        self.values[active//action_count, active % action_count, month] = values
        u = self.uniforms[active//action_count, month]
        return (u[:, None] > probs.cumsum(1)).sum(1).clip(0, action_count-1)


def mc_probe(snapshot, model, config, risk, draws, seed, gamma, lam, scale):
    obs = build_observation(snapshot.state, snapshot.predicted_pd, snapshot.elapsed, config)
    probs, _, _, predicted = distribution(model, obs[None])
    remaining = config.environment.horizon-snapshot.elapsed
    actor = ValuedContinuation(model, draws*2, remaining, seed+10000000, predicted[0])
    paths = hypothetical_paths(snapshot, draws*2, remaining, seed)
    cube = rollout(snapshot, config, risk, actor, paths, remaining)
    reward = cube[:, :, :, 6]
    q = (reward*gamma**np.arange(remaining)).sum(2)
    gae = gae_from_arrays(reward[:draws], actor.values[:draws]/scale, gamma, lam)
    truth = q[draws:]
    v = truth @ probs[0]
    advantage = truth-v[:, None]
    return dict(gae=gae, advantage=advantage, q=truth, value=v, prediction=float(predicted[0]/scale),
                probabilities=probs[0], obs=obs)


def diagnostic_panel(args, config, settings, protocol, risk):
    path = args.output/'test_states.joblib'
    if path.exists():
        return joblib.load(path)
    panels = joblib.load(args.output/'panels.joblib')
    teacher = joblib.load(args.phase_b/'planners.joblib')['F0']['model']
    canonical = PPO.load(args.canonical/'models/ppo_101/selected.zip', device='cpu')
    rng = np.random.default_rng(protocol['population_seed']+1)
    snapshots, observations, rows = [], [], []
    for scenario, group in panels['test'].items():
        for i, customer in enumerate(group):
            env = CreditLimitEnv(config=config, pd_model=risk,
                severe_delinquency_months=settings['guardrails']['severe_delinquency_months'])
            actor = [SB3Policy(canonical), ObservationPlanner(teacher, config), MyopicEconomic(config)][i % 3]
            obs, _ = env.reset(seed=customer.customer_seed, options=customer.reset_options())
            visits = []
            while not env._done:
                visits.append((DecisionSnapshot.capture(env), obs.copy()))
                obs, *_ = env.step(actor.act(obs))
            index = int(rng.integers(len(visits)))
            snapshot, observation = visits[index]
            snapshots.append(snapshot)
            observations.append(observation)
            rows.append(dict(**describe(snapshot, scenario, ['PPO', 'ObservationPlanner', 'MyopicEconomic'][i % 3]),
                             state_id=len(rows), weight=len(visits)))
            env.close()
    result = dict(snapshots=snapshots, observations=np.array(observations), states=pd.DataFrame(rows))
    joblib.dump(result, path)
    result['states'].to_csv(args.output/'test_states.csv', index=False)
    return result


def evaluate_actor(model, config, settings, risk, scenarios, seed, experiment):
    spec = PolicySpec(experiment, lambda e, s: SB3Policy(model), seed=seed)
    episodes, histories = [], []
    # Primary common-discount endpoint, regardless of treatment training gamma.
    from copy import deepcopy
    evaluation = deepcopy(settings)
    evaluation['ppo']['gamma'] = .98
    for macro, group in scenarios.items():
        e, h, _ = evaluate_policy(spec, group, config, risk, evaluation, macro)
        purchases = h[h.month > 0].groupby('customer_id').spending.sum()
        e['purchases'] = e.customer_id.map(purchases)
        e['training_discounted_reward'] = e.customer_id.map(h[h.month > 0].groupby('customer_id').apply(
            lambda g: float(np.dot(g.reward, settings['ppo']['gamma']**np.arange(len(g)))), include_groups=False))
        episodes.append(e)
        histories.append(h)
    return pd.concat(episodes, ignore_index=True), pd.concat(histories, ignore_index=True)


def policy_measurements(model, teacher, data, config, experiment, seed):
    obs = data['observations']
    p, logits, entropy, v = distribution(model, obs)
    actions = p.argmax(1)
    q = teacher.predict(obs)
    targets = np.array([choose(x, o, config) for x, o in zip(q, obs)])
    frame = data['states'].copy()
    frame['experiment_id'], frame['seed'] = experiment, seed
    frame['action'], frame['teacher_action'], frame['entropy'] = actions, targets, entropy
    frame['requested_teacher_regret'] = q[np.arange(len(q)), targets]-q[np.arange(len(q)), actions]
    frame['teacher_regret'] = observable_regret(q, actions, obs, config)
    frame['agreement'] = actions == targets
    frame['effective_action'] = [effective_action(s, int(a), config) for s, a in zip(data['snapshots'], actions)]
    frame['effective_agreement'] = [effective_action(s, int(a), config) == effective_action(s, int(b), config)
                                   for s, a, b in zip(data['snapshots'], actions, targets)]
    for a in range(5):
        frame[f'probability_{a}'], frame[f'logit_{a}'] = p[:, a], logits[:, a]
    metrics = []
    for scenario, group in frame.groupby('scenario'):
        mask = frame.scenario.eq(scenario).to_numpy()
        metrics.append(dict(experiment_id=experiment, seed=seed, scenario=scenario,
            entropy=entropy[mask].mean(), contraction_share=(actions[mask] == 0).mean(),
            stochastic_contraction=p[mask, 0].mean(), teacher_regret=np.average(group.teacher_regret, weights=group.weight),
            **state_dependence(actions[mask], obs[mask])))
    return frame, pd.DataFrame(metrics)


def observable_regret(q, actions, observations, config):
    """Economic Q regret collapses floor/cap/guard aliases using public information."""
    result = []
    for values, action, observation in zip(q, actions, observations):
        state = decode(observation, config)
        limits = [effective_limit(state['limit'], state['delinquency'], m, config.environment, 3)[0]
                  for m in config.environment.action_multipliers]
        allowed = legal_actions(observation, config)
        equivalents = [a for a in allowed if abs(limits[a]-limits[int(action)]) <= .01]
        result.append(max(values[a] for a in allowed)-max(values[a] for a in equivalents))
    return np.asarray(result)


def probe_rows(probe, snapshot, scenario, experiment, seed, steps, state_id):
    result = []
    for a in range(5):
        gae, advantage = probe['gae'][:, a], probe['advantage'][:, a]
        result.append(dict(experiment_id=experiment, seed=seed, timesteps=steps, state_id=state_id,
            scenario=scenario, action=a, gae_mean=gae.mean(), gae_se=gae.std(ddof=1)/np.sqrt(len(gae)),
            mc_advantage=advantage.mean(), mc_se=advantage.std(ddof=1)/np.sqrt(len(advantage)),
            q_mean=probe['q'][:, a].mean(), value_mc=probe['value'].mean(),
            value_prediction=probe['prediction'], probability=probe['probabilities'][a],
            pd=snapshot.predicted_pd, utilization=snapshot.state.utilization,
            remaining_horizon=24-snapshot.elapsed))
    return result


def summarize_alignment(raw):
    records = []
    keys = ['experiment_id', 'seed', 'timesteps', 'scenario']
    if 'checkpoint_kind' in raw:
        keys.append('checkpoint_kind')
    for key, group in raw.groupby(keys):
        tags = dict(zip(keys, key))
        x, y = group.gae_mean.to_numpy(), group.mc_advantage.to_numpy()
        ranks = []
        for _, state in group.groupby('state_id'):
            ranks.append(int(state.loc[state.gae_mean.idxmax(), 'action'] == state.loc[state.mc_advantage.idxmax(), 'action']))
        unique = group.drop_duplicates('state_id')
        error = unique.value_prediction-unique.value_mc
        variance = np.sum((unique.value_mc-unique.value_mc.mean())**2)
        records.append(dict(**tags, states=len(unique), sign_agreement=np.mean(np.sign(x) == np.sign(y)),
            pearson=np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan,
            spearman=spearmanr(x, y).statistic if np.std(x) > 0 and np.std(y) > 0 else np.nan,
            advantage_rmse=np.sqrt(np.mean((x-y)**2)), ranking_accuracy=np.mean(ranks),
            critic_bias=error.mean(), critic_rmse=np.sqrt(np.mean(error**2)),
            critic_r2=1-np.sum(error**2)/variance if variance > 0 else np.nan))
    return pd.DataFrame(records)


def measure_run(args, row, config, settings, protocol, risk, temporal=False):
    folder = args.output/'runs'/row['experiment_id']/str(row['seed'])
    metadata = json.loads((folder/'metadata.json').read_text())
    settings = {**settings, 'ppo': metadata['hyperparameters']}
    teacher = joblib.load(args.phase_b/'planners.joblib')['F0']['model']
    panels = joblib.load(args.output/'panels.joblib')
    data = diagnostic_panel(args, config, settings, protocol, risk)
    model = PPO.load(folder/'selected.zip', device='cpu')
    destination = folder/'evaluation'
    destination.mkdir(exist_ok=True)
    if not (destination/'episodes.csv').exists():
        episodes, histories = evaluate_actor(model, config, settings, risk, panels['test'], row['seed'], row['experiment_id'])
        episodes.to_csv(destination/'episodes.csv', index=False)
        histories.to_csv(destination/'histories.csv.gz', index=False)
    if row['family'] in ('canonical', 'budget') and not (destination/'final_episodes.csv').exists():
        final_model = PPO.load(folder/'final.zip', device='cpu')
        final_episodes, _ = evaluate_actor(final_model, config, settings, risk, panels['test'], row['seed'], row['experiment_id']+'_final')
        final_episodes.to_csv(destination/'final_episodes.csv', index=False)
    if (not (destination/'policy_states.csv').exists() or
            'requested_teacher_regret' not in pd.read_csv(destination/'policy_states.csv', nrows=0).columns):
        frame, summary = policy_measurements(model, teacher, data, config, row['experiment_id'], row['seed'])
        frame.to_csv(destination/'policy_states.csv', index=False)
        summary.to_csv(destination/'policy_summary.csv', index=False)
    # Fixed validation states only, for mechanism selection; test evaluation cannot feed this gate.
    samples = []
    for scenario, group in panels['validation_states'].groupby('scenario'):
        n = min(protocol['diagnostic_states_per_scenario'], len(group))
        samples.extend((scenario, int(i)) for i in group.iloc[np.linspace(0, len(group)-1, n, dtype=int)].index)
    checkpoints = [('selected', model, int(metadata['selected_timesteps']))]
    if row['family'] in ('canonical', 'budget'):
        checkpoints.append(('final', None, row['budget']))
    if temporal:
        snapshots = pd.read_csv(folder/'snapshots.csv')
        requested = [0, 512, 1024, 1536, 2048, 4096, 8192, row['budget']]
        for step in requested:
            matches = snapshots[snapshots.timesteps == step]
            if not matches.empty:
                checkpoints.append((f'time_{step}', None, step))
    for label, supplied, steps in checkpoints:
        path = destination/f'mc_{label}.csv'
        if path.exists():
            continue
        actor = supplied or PPO.load(folder/f'snapshots/update_{steps}.zip', device='cpu')
        records = []
        for scenario, sid in samples:
            snapshot = panels['validation_snapshots'][sid]
            probe = mc_probe(snapshot, actor, config, risk, protocol['mc_draws'],
                protocol['population_seed']+7000000+sid*1000, settings['ppo']['gamma'],
                settings['ppo']['gae_lambda'], settings['ppo']['reward_scale'])
            records.extend(probe_rows(probe, snapshot, scenario, row['experiment_id'], row['seed'], steps, sid))
        pd.DataFrame(records).to_csv(path, index=False)
    # Equal validation observation panel for the confirmatory eligibility rule.
    obs = panels['validation_observations']
    q = teacher.predict(obs)
    aa = model.predict(obs, deterministic=True)[0]
    value = observable_regret(q, aa, obs, config)
    pd.DataFrame(dict(scenario=panels['validation_states'].scenario, regret=value)).groupby('scenario').regret.mean().to_csv(destination/'validation_regret.csv')
    training_path = args.output/'runs/canonical/101/training_states.joblib'
    if training_path.exists() and not (destination/'mc_training_fixed.csv').exists():
        training = joblib.load(training_path)
        indices = np.unique(np.linspace(0, len(training)-1, min(6, len(training)), dtype=int))
        records = []
        for sid in indices:
            snapshot = training[int(sid)]['snapshot']
            probe = mc_probe(snapshot, model, config, risk, protocol['mc_draws'],
                int(protocol['population_seed']+11000000+sid*1000), settings['ppo']['gamma'],
                settings['ppo']['gae_lambda'], settings['ppo']['reward_scale'])
            records.extend(probe_rows(probe, snapshot, 'training_markov_fixed', row['experiment_id'],
                row['seed'], int(metadata['selected_timesteps']), int(sid)))
        pd.DataFrame(records).to_csv(destination/'mc_training_fixed.csv', index=False)
    if temporal and not (destination/'actual_training_gae.csv').exists():
        training = joblib.load(folder/'training_states.joblib')
        requested = [512, 2048, 8192, row['budget']] if row['budget'] > 128 else [64, 128]
        records = []
        for state in training:
            if state['timesteps'] not in requested:
                continue
            before = state['timesteps']-settings['ppo']['n_steps']
            path = folder/f'snapshots/update_{before}.zip'
            if not path.exists():
                continue
            actor = PPO.load(path, device='cpu')
            probe = mc_probe(state['snapshot'], actor, config, risk, protocol['mc_draws'],
                protocol['population_seed']+13000000+state['timesteps'], settings['ppo']['gamma'],
                settings['ppo']['gae_lambda'], settings['ppo']['reward_scale'])
            a = state['action']
            records.append(dict(experiment_id=row['experiment_id'], seed=row['seed'], timesteps=state['timesteps'],
                behavior_checkpoint=before, sampled_action=a, actual_gae=state['raw_gae']/settings['ppo']['reward_scale'],
                independent_mc_advantage=probe['advantage'][:, a].mean(),
                independent_mc_se=probe['advantage'][:, a].std(ddof=1)/np.sqrt(protocol['mc_draws']),
                counterfactual_gae_mean=probe['gae'][:, a].mean(), scenario='actual training Markov path'))
        pd.DataFrame(records).to_csv(destination/'actual_training_gae.csv', index=False)
    return destination
