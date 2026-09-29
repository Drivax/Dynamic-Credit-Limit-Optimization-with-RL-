"""Independent common-continuation action values and supervised value diagnostic."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
import torch

from credit_rl.evaluation.structural import hypothetical_paths, rollout
from credit_rl.experiments.information_analysis import weighted_summary
from credit_rl.experiments.information_ppo import distribution
from credit_rl.experiments.ppo_diagnostics import prepare, protection, diagnostic_provenance
from credit_rl.experiments.ppo_measurements import diagnostic_panel
from credit_rl.policies.baselines import SB3Policy
from credit_rl.policies.decision import ConstantAdjustment
from credit_rl.policies.information import choose


def run(args):
    config, settings, _, protocol, risk = prepare(args)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        data = diagnostic_panel(args, config, settings, protocol, risk)
        model = PPO.load(args.canonical/'models/ppo_101/selected.zip', device='cpu')
        teacher = joblib.load(args.phase_b/'planners.joblib')['F0']['model']
        cache = args.output/'counterfactual_q.npz'
        if cache.exists():
            q = np.load(cache)['q']
        else:
            values = []
            for sid, snapshot in enumerate(data['snapshots']):
                paths = hypothetical_paths(snapshot, protocol['mc_draws'], 24, protocol['population_seed']+9000000+sid*1000)
                cube = rollout(snapshot, config, risk, SB3Policy(model), paths, 24)
                values.append((cube[:, :, :, 6]*.98**np.arange(cube.shape[2])).sum(2))
            q = np.array(values)
            np.savez_compressed(cache, q=q)
        obs = data['observations']
        half = q.shape[1]//2
        best = np.array([choose(v, o, config) for v, o in zip(q[:, :half].mean(1), obs)])
        selected = q[:, half:].mean(1)
        actor_actions = model.predict(obs, deterministic=True)[0]
        actions = dict(ObservationPlanner=np.array([choose(v, o, config) for v, o in zip(teacher.predict(obs), obs)]),
            AlwaysDecrease20=np.array([ConstantAdjustment(config, .8).act(o) for o in obs]))
        records = []
        for name, aa in actions.items():
            frame = data['states'].copy()
            frame['experiment_id'], frame['seed'], frame['action'] = name, -1, aa
            records.append(frame)
        policies = args.output/'policy_regret.csv'
        if policies.exists():
            records.append(pd.read_csv(policies))
        imitation = args.output/'imitation_states.csv'
        if imitation.exists():
            records.append(pd.read_csv(imitation))
        all_states = pd.concat(records, ignore_index=True)
        all_states['regret_mc'] = [selected[int(sid), best[int(sid)]]-selected[int(sid), int(a)]
                                   for sid, a in zip(all_states.state_id, all_states.action)]
        all_states['regret_vs_observation_mc'] = [selected[int(sid), actions['ObservationPlanner'][int(sid)]]-selected[int(sid), int(a)]
                                   for sid, a in zip(all_states.state_id, all_states.action)]
        all_states.to_csv(args.output/'counterfactual_states.csv', index=False)
        summaries = []
        bootstrap = dict(bootstrap_repetitions=protocol['bootstrap_repetitions'], diagnostic_seed=protocol['population_seed'])
        for key, group in all_states.groupby(['experiment_id', 'seed', 'scenario']):
            summaries.append(dict(zip(['experiment_id', 'seed', 'scenario'], key)) | weighted_summary(group, 'regret_mc', bootstrap))
        pd.DataFrame(summaries).to_csv(args.output/'counterfactual_regret.csv', index=False)
        # C4: supervised deterministic-continuation V benchmark, not a replacement PPO critic.
        b = joblib.load(args.phase_b/'dataset.joblib')
        train = np.flatnonzero(b['states'].role.eq('train'))
        train_obs = b['features']['F0'][train]
        train_actions = model.predict(train_obs, deterministic=True)[0]
        y = np.array([np.load(args.phase_b/f'draws/{sid:05d}.npz')['q'][:, -1, int(a)].mean()
                      for sid, a in zip(train, train_actions)])
        predictor = HistGradientBoostingRegressor(max_leaf_nodes=7, max_iter=100 if args.profile == 'standard' else 3,
            min_samples_leaf=12 if args.profile == 'standard' else 2, early_stopping=False, random_state=protocol['population_seed'])
        predictor.fit(train_obs, y, sample_weight=b['states'].iloc[train].weight)
        joblib.dump(predictor, args.output/'mc_value_benchmark.joblib')
        predicted = predictor.predict(obs)
        critic = distribution(model, obs)[3]/settings['ppo']['reward_scale']
        target = selected[np.arange(len(selected)), actor_actions]
        rows = []
        for name, estimate in [('SupervisedMCValue', predicted), ('CanonicalCritic', critic)]:
            for scenario in ('baseline', 'severe_stress'):
                for region in ('all', 'mc_best_not_contraction'):
                    mask = data['states'].scenario.eq(scenario).to_numpy(copy=True)
                    if region != 'all':
                        mask &= best != 0
                    error = estimate[mask]-target[mask]
                    rows.append(dict(benchmark=name, scenario=scenario, region=region, states=int(mask.sum()),
                        bias=np.mean(error), rmse=np.sqrt(np.mean(error**2)),
                        r2=r2_score(target[mask], estimate[mask]) if mask.sum() > 1 else np.nan,
                        estimand='deterministic PPO101 continuation; critic trained under stochastic actor'))
        pd.DataFrame(rows).to_csv(args.output/'mc_value_benchmark.csv', index=False)
        diagnostic_provenance(args, args.output/'counterfactual_provenance.json',
            dict(seed=protocol['population_seed'], draws=protocol['mc_draws'], scenarios=['baseline', 'severe_stress'],
                 continuation='frozen deterministic canonical PPO101', value_predictor=predictor.get_params(),
                 training_states=len(train), test_states=len(obs)), args.canonical/'models/ppo_101/selected.zip')
    if json.loads((args.output/'protected_artifacts.json').read_text()) != protection(args.canonical, args.phase_b):
        raise AssertionError('Protected artifacts changed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    p.add_argument('--canonical', type=Path, default=Path('outputs/main/standard'))
    p.add_argument('--phase-b', type=Path, default=Path('outputs/main/information_gap'))
    p.add_argument('--output', type=Path, default=Path('outputs/main/ppo_diagnostics'))
    run(p.parse_args())


if __name__ == '__main__':
    main()
