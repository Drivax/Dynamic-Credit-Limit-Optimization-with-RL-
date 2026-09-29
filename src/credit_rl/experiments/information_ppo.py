"""Read-only canonical checkpoint audit and instrumented, isolated PPO replay."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from threadpoolctl import threadpool_limits
import torch
import yaml

from credit_rl.evaluation.scenarios import make_scenarios
from credit_rl.evaluation.structural import DecisionSnapshot, hypothetical_paths, rollout
from credit_rl.experiments.main_evaluation import settings_for
from credit_rl.experiments.information_gap import protected_hashes, validate_destination
from credit_rl.policies.training import train_agent
from credit_rl.risk.longitudinal import LongitudinalPDModel


def distribution(model, observations):
    with torch.no_grad():
        x, _ = model.policy.obs_to_tensor(np.asarray(observations, dtype=np.float32))
        d = model.policy.get_distribution(x).distribution
        return d.probs.cpu().numpy(), d.logits.cpu().numpy(), d.entropy().cpu().numpy(), model.policy.predict_values(x).cpu().numpy().ravel()


class RolloutRecorder(BaseCallback):
    """Read completed GAE buffers without sampling or touching any RNG state."""
    def __init__(self, seed, destination):
        super().__init__()
        self.seed, self.destination = seed, Path(destination)
        self.rows, self.gradients = [], []
        self.minibatches, self.signal_states = [], []
        self.start_snapshot = None

    def _on_training_start(self):
        self.original_get = self.model.rollout_buffer.get
        def observe_batches(*args, **kwargs):
            for batch in self.original_get(*args, **kwargs):
                advantages = batch.advantages
                normalized = ((advantages-advantages.mean())/(advantages.std()+1e-8)
                              if len(advantages) > 1 else advantages)
                for action in range(5):
                    mask = batch.actions.flatten() == action
                    if mask.any():
                        self.minibatches.append(dict(seed=self.seed, timesteps=self.num_timesteps,
                            action=action, count=int(mask.sum()),
                            normalized_sum=float(normalized[mask].sum()),
                            normalized_negative=int((normalized[mask] < 0).sum())))
                yield batch
        self.model.rollout_buffer.get = observe_batches

    def _on_rollout_start(self):
        # Predetermined four standard checkpoints (one in smoke), never selected on outcomes.
        period = self.model.n_steps if self.model.n_steps < 512 else 8192
        if (self.num_timesteps+self.model.n_steps) % period == 0:
            self.start_snapshot = DecisionSnapshot.capture(self.training_env.envs[0].unwrapped)
        else:
            self.start_snapshot = None

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        b = self.model.rollout_buffer
        observations = b.observations.reshape(-1, 21)
        probs, logits, entropy, _ = distribution(self.model, observations)
        frame = pd.DataFrame(dict(seed=self.seed, timesteps=self.num_timesteps,
            action=b.actions.ravel().astype(int), advantage=b.advantages.ravel(),
            returns=b.returns.ravel(), value=b.values.ravel(), reward=b.rewards.ravel(), entropy=entropy,
            episode_start=b.episode_starts.ravel()))
        for i in range(21):
            frame[f'obs_{i}'] = observations[:, i]
        for a in range(5):
            frame[f'prob_{a}'], frame[f'logit_{a}'] = probs[:, a], logits[:, a]
        self.rows.append(frame)
        if self.start_snapshot is not None:
            self.signal_states.append(dict(snapshot=self.start_snapshot,
                observation=observations[0].copy(), action=int(b.actions.ravel()[0]),
                advantage=float(b.advantages.ravel()[0]), seed=self.seed, timesteps=self.num_timesteps))
            # Capture the actual behavior policy before its next parameter update.
            self.model.save(self.destination/f'signal_{self.num_timesteps}')

    def _on_training_end(self):
        self.model.rollout_buffer.get = self.original_get

    def flush(self):
        frame = pd.concat(self.rows, ignore_index=True)
        frame.to_csv(self.destination/'rollouts.csv.gz', index=False)
        pd.DataFrame(self.gradients).to_csv(self.destination/'gradients.csv', index=False)
        pd.DataFrame(self.minibatches).to_csv(self.destination/'minibatch_advantages.csv', index=False)
        joblib.dump(self.signal_states, self.destination/'signal_states.joblib')
        # Aggregate the persisted representation on both fresh and cached runs.
        return pd.read_csv(self.destination/'rollouts.csv.gz')


@contextmanager
def record_gradients(recorder):
    """Observe the pre-clip norm returned by the unchanged PyTorch operation."""
    original = torch.nn.utils.clip_grad_norm_
    def recorded(parameters, *args, **kwargs):
        value = original(parameters, *args, **kwargs)
        recorder.gradients.append(dict(seed=recorder.seed, timesteps=recorder.num_timesteps,
                                        preclip_norm=float(value)))
        return value
    torch.nn.utils.clip_grad_norm_ = recorded
    try:
        yield
    finally:
        torch.nn.utils.clip_grad_norm_ = original


def same_parameters(a, b):
    x, y = a.policy.state_dict(), b.policy.state_dict()
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def replay(config, settings, protocol, risk, canonical, output):
    train = make_scenarios(config, settings, 'train', 'markov')
    validation = make_scenarios(config, settings, 'validation')
    all_buffers, verification = [], []
    for seed in protocol['replay_ppo_seeds']:
        folder = output/f'replay/ppo_{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder/'replay_verification.json'
        if marker.exists() and json.loads(marker.read_text()).get('protocol_version') == 2:
            verification.append(json.loads(marker.read_text()))
            all_buffers.append(pd.read_csv(folder/'rollouts.csv.gz'))
            continue
        recorder = RolloutRecorder(seed, folder)
        with record_gradients(recorder):
            train_agent(config, risk, settings, train, validation, seed=seed, destination=folder,
                        diagnostic_callback=recorder)
        all_buffers.append(recorder.flush())
        row = dict(seed=seed, protocol_version=2, training_macro='markov')
        for name in ('selected', 'final'):
            row[name+'_weights_identical'] = same_parameters(PPO.load(folder/name, device='cpu'),
                PPO.load(canonical/f'models/ppo_{seed}/{name}', device='cpu'))
        marker.write_text(json.dumps(row, indent=2))
        verification.append(row)
    pd.DataFrame(verification).to_csv(output/'replay_verification.csv', index=False)
    return pd.concat(all_buffers, ignore_index=True)


def training_tables(buffer, canonical, output, scale):
    rows, advantages = [], []
    for (seed, steps), group in buffer.groupby(['seed', 'timesteps']):
        gradients = pd.read_csv(output/f'replay/ppo_{seed}/gradients.csv')
        grad = gradients[gradients.timesteps == steps].preclip_norm
        row = dict(seed=seed, timesteps=steps, entropy=group.entropy.mean(),
            reward_scale=scale, scaled_reward_mean=group.reward.mean(),
            raw_reward_mean=group.reward.mean()/scale, value_mean=group.value.mean(),
            return_mean=group.returns.mean(), gradient_norm_mean=grad.mean(),
            gradient_norm_max=grad.max())
        for a in range(5):
            row[f'action_fraction_{a}'] = (group.action == a).mean()
            row[f'prob_{a}'] = group[f'prob_{a}'].mean()
            row[f'logit_{a}'] = group[f'logit_{a}'].mean()
            selected = group[group.action == a]
            quantiles = selected.advantage.quantile([.05, .5, .95])
            advantages.append(dict(seed=seed, timesteps=steps, action=a, count=len(selected),
                mean=selected.advantage.mean(), std=selected.advantage.std(),
                p05=quantiles.loc[.05], median=quantiles.loc[.5], p95=quantiles.loc[.95],
                negative_fraction=(selected.advantage < 0).mean(),
                positive_fraction=(selected.advantage > 0).mean(),
                units='scaled raw GAE before minibatch normalization'))
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    logs = []
    for seed in buffer.seed.unique():
        progress = pd.read_csv(canonical/f'models/ppo_{seed}/progress.csv')
        progress['seed'] = seed
        progress = progress.rename(columns={'time/total_timesteps': 'timesteps'})
        logs.append(progress)
    diagnostics = diagnostics.merge(pd.concat(logs), on=['seed', 'timesteps'], how='left')
    diagnostics.to_csv(output/'ppo_training_diagnostics.csv', index=False)
    pd.DataFrame(advantages).to_csv(output/'ppo_advantages.csv', index=False)
    normalized = pd.concat([pd.read_csv(output/f'replay/ppo_{seed}/minibatch_advantages.csv')
                            for seed in buffer.seed.unique()])
    normalized = normalized.groupby(['seed', 'timesteps', 'action'], as_index=False).sum()
    normalized['normalized_mean'] = normalized.normalized_sum/normalized['count']
    normalized['normalized_negative_fraction'] = normalized.normalized_negative/normalized['count']
    normalized.to_csv(output/'ppo_normalized_advantages.csv', index=False)
    buffer['pd_bucket'] = np.searchsorted([.2, .6], buffer.obs_10)
    utilization = buffer.obs_3/np.maximum(1-buffer.obs_3, 1e-8)
    buffer['utilization_bucket'] = np.searchsorted([.5, 1.], utilization)
    buffer['horizon_bucket'] = np.searchsorted([1/3, 2/3], buffer.obs_0)
    keys = ['seed', 'timesteps', 'pd_bucket', 'utilization_bucket', 'horizon_bucket']
    coverage = buffer.groupby(keys+['action']).size().rename('count').reset_index()
    coverage['state_visits'] = coverage.groupby(keys)['count'].transform('sum')
    coverage['conditional_fraction'] = coverage['count']/coverage.state_visits
    coverage.to_csv(output/'exploration.csv', index=False)


def checkpoints(data, settings, canonical, output):
    ids = np.flatnonzero(data['states'].role.eq('validation'))
    obs = np.array([data['prefixes'][i][0][-1] for i in ids])
    rows, validation = [], []
    for seed in settings['ppo']['seeds']:
        folder = canonical/f'models/ppo_{seed}'
        v = pd.read_csv(folder/'validation.csv').assign(seed=seed)
        validation.append(v)
        metadata = json.loads((folder/'metadata.json').read_text())
        for path in sorted(folder.glob('checkpoint_*.zip')):
            steps = int(path.stem.split('_')[-1])
            model = PPO.load(path, device='cpu')
            p, logits, entropy, values = distribution(model, obs)
            for j, sid in enumerate(ids):
                rows.append(dict(seed=seed, timesteps=steps, state_id=sid,
                    selected=steps == metadata['selected_timesteps'], action=int(p[j].argmax()),
                    entropy=entropy[j], value_raw_eur=values[j]/settings['ppo']['reward_scale'],
                    **{f'prob_{a}': p[j, a] for a in range(5)},
                    **{f'logit_{a}': logits[j, a] for a in range(5)}))
    pd.DataFrame(rows).to_csv(output/'checkpoint_diagnostics.csv', index=False)
    pd.concat(validation).to_csv(output/'checkpoint_validation.csv', index=False)


class StochasticContinuation:
    """Indexed private uniforms share action randomness across counterfactual branches."""
    def __init__(self, model, draws, horizon, seed):
        self.actor = model
        self.uniforms = np.random.default_rng(seed).random((draws, horizon))

    def act_indexed(self, observations, active, month, action_count):
        probs, _, _, _ = distribution(self.actor, observations)
        u = self.uniforms[np.asarray(active)//action_count, month]
        return (u[:, None] > probs.cumsum(1)).sum(1).clip(0, action_count-1)


def advantage_alignment(config, settings, protocol, risk, output):
    rows = []
    for seed in protocol['replay_ppo_seeds']:
        folder = output/f'replay/ppo_{seed}'
        for i, state in enumerate(joblib.load(folder/'signal_states.joblib')):
            model = PPO.load(folder/f'signal_{state["timesteps"]}', device='cpu')
            key = protocol['diagnostic_seed']+5000000+seed*100+i*1000
            paths = hypothetical_paths(state['snapshot'], protocol['draws'], config.environment.horizon, key)
            actor = StochasticContinuation(model, len(paths), config.environment.horizon, key+10000000)
            cube = rollout(state['snapshot'], config, risk, actor, paths, config.environment.horizon)
            q = (cube[:, :, :, 6]*settings['ppo']['gamma']**np.arange(cube.shape[2])).sum(2)
            probs = distribution(model, state['observation'][None])[0][0]
            v = q @ probs
            for a in range(5):
                advantage = q[:, a]-v
                rows.append(dict(seed=seed, timesteps=state['timesteps'], action=a,
                    sampled_action=state['action'], gae_raw_eur=state['advantage']/settings['ppo']['reward_scale'],
                    mc_advantage=advantage.mean(), mc_se=advantage.std(ddof=1)/np.sqrt(len(advantage)),
                    probability=probs[a],
                    conditioning='actual training state; pre-update stochastic actor; fixed realized macro path'))
    pd.DataFrame(rows).to_csv(output/'advantage_alignment.csv', index=False)


def critics(data, q, config, settings, protocol, risk, canonical, output):
    # All sampled held-out PPO visits, avoiding outcome-dependent state selection.
    ids = np.flatnonzero(data['states'].role.eq('test') & data['states'].source_policy.eq('PPO'))
    rows = []
    folder = output/'critic_draws'
    folder.mkdir(exist_ok=True)
    for seed in settings['ppo']['seeds']:
        model = PPO.load(canonical/f'models/ppo_{seed}/selected.zip', device='cpu')
        obs = np.array([data['prefixes'][i][0][-1] for i in ids])
        probs, _, _, values = distribution(model, obs)
        for j, sid in enumerate(ids):
            snap = data['snapshots'][sid]
            cache = folder/f'{seed}_{sid}.npz'
            if cache.exists():
                stochastic = np.load(cache)['q']
            else:
                key = int(protocol['diagnostic_seed']+2000000+sid*1000)
                paths = hypothetical_paths(snap, protocol['critic_draws'], config.environment.horizon, key)
                actor = StochasticContinuation(model, len(paths), config.environment.horizon, key+10000000)
                cube = rollout(snap, config, risk, actor, paths, config.environment.horizon)
                stochastic = (cube[:, :, :, 6]*settings['ppo']['gamma']**np.arange(cube.shape[2])).sum(2)
                np.savez_compressed(cache, q=stochastic)
            stochastic_v = stochastic @ probs[j]
            # Deterministic comparison has seed101 continuation; label rather than conflate.
            det = q[sid, q.shape[1]//2:, -1, int(probs[j].argmax())]
            row = data['states'].iloc[sid].to_dict()
            row.update(seed=seed, prediction=values[j]/settings['ppo']['reward_scale'],
                target=stochastic_v.mean(), mc_se=stochastic_v.std(ddof=1)/np.sqrt(len(stochastic_v)),
                deterministic_101_target=det.mean(),
                best_mc_action=int(stochastic.mean(0).argmax()))
            rows.append(row)
        print(f'Critic audited seed={seed}, states={len(ids)}', flush=True)
    raw = pd.DataFrame(rows)
    raw['error'] = raw.prediction-raw.target
    raw.to_csv(output/'critic_states.csv', index=False)
    summaries = []
    for variable in ('all', 'pd_bucket', 'utilization_bucket', 'remaining_horizon', 'best_mc_action',
                     'latent_creditworthiness', 'latent_payment_propensity', 'latent_spending_propensity', 'latent_income_stability'):
        bucket = 'all' if variable == 'all' else raw[variable].astype(str)
        if variable.startswith('latent_'):
            bucket = pd.qcut(raw[variable], 3, duplicates='drop').astype(str)
        f = raw.assign(bucket=bucket)
        for (seed, scenario, b), group in f.groupby(['seed', 'scenario', 'bucket']):
            w = group.weight
            summaries.append(dict(seed=seed, scenario=scenario, stratification=variable, bucket=b,
                states=len(group), bias=np.average(group.error, weights=w),
                rmse=np.sqrt(np.average(group.error**2, weights=w)),
                r2=r2_score(group.target, group.prediction, sample_weight=w) if len(group) > 1 else np.nan,
                mean_mc_se=np.average(group.mc_se, weights=w)))
    pd.DataFrame(summaries).to_csv(output/'critic_diagnostics.csv', index=False)


def run(args):
    validate_destination(args.output, args.canonical)
    config, settings, _ = settings_for(args.canonical_profile, 'configs')
    protocol = yaml.safe_load(Path('configs/information_gap.yaml').read_text())[args.profile]
    manifest = args.output/'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if (saved['protocol'] != protocol or saved['canonical'] != str(args.canonical.resolve())
                or saved['canonical_profile'] != args.canonical_profile):
            raise ValueError('PPO audit must match the Phase B dataset identity')
    risk = LongitudinalPDModel.load(args.canonical/'models/pd/logistic_calibrated.joblib')
    before = protected_hashes(args.canonical)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if args.stage in ('replay', 'all'):
            buffer = replay(config, settings, protocol, risk, args.canonical, args.output)
            training_tables(buffer, args.canonical, args.output, settings['ppo']['reward_scale'])
            advantage_alignment(config, settings, protocol, risk, args.output)
        if args.stage in ('critic', 'all'):
            data = joblib.load(args.output/'dataset.joblib')
            q = np.array([np.load(args.output/f'draws/{i:05d}.npz')['q'] for i in range(len(data['states']))])
            checkpoints(data, settings, args.canonical, args.output)
            critics(data, q, config, settings, protocol, risk, args.canonical, args.output)
    if protected_hashes(args.canonical) != before:
        raise RuntimeError('Canonical or Phase A artifacts were modified')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['smoke', 'standard'], default='standard')
    p.add_argument('--canonical-profile', choices=['smoke', 'standard'], default='standard')
    p.add_argument('--canonical', type=Path, default=Path('outputs/main/standard'))
    p.add_argument('--output', type=Path, default=Path('outputs/main/information_gap'))
    p.add_argument('--stage', choices=['all', 'replay', 'critic'], default='all')
    run(p.parse_args())


if __name__ == '__main__':
    main()
