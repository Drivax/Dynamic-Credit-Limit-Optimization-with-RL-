"""Exact short-run diagnostic replay capturing actual normalized sample advantages."""
from concurrent.futures import ProcessPoolExecutor
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
import torch

from credit_rl.experiments import initialization_learning as learning
from credit_rl.experiments.information_ppo import same_parameters
from credit_rl.experiments.main_evaluation import settings_for
from credit_rl.experiments.policy_initialization import ROOT, write_json
from credit_rl.policies.information import choose
from credit_rl.risk.longitudinal import LongitudinalPDModel


class SampleRecorder(learning.InitializationRecorder):
    def _on_training_start(self):
        super()._on_training_start()
        observed_get = self.model.rollout_buffer.get
        self.sample_batches = []
        def sample_batches(*args, **kwargs):
            for batch in observed_get(*args, **kwargs):
                if self.num_timesteps <= 8192:
                    advantages = batch.advantages
                    normalized = (advantages-advantages.mean())/(advantages.std()+1e-8)
                    self.sample_batches.append(dict(timesteps=self.num_timesteps,
                        observations=batch.observations.detach().cpu().numpy().copy(),
                        actions=batch.actions.flatten().detach().cpu().numpy().copy(),
                        advantages=advantages.detach().cpu().numpy().copy(),
                        normalized=normalized.detach().cpu().numpy().copy()))
                yield batch
        self.model.rollout_buffer.get = sample_batches

    def flush(self):
        result = super().flush()
        joblib.dump(self.sample_batches, self.destination/'actual_minibatches.joblib', compress=3)
        return result


def replay(seed):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        output = ROOT/'policy_initialization'
        replay_root = output/'diagnostic_replay'
        identity = json.loads((output/'preregistration.json').read_text())['identity']
        protocol = identity['protocol']
        config, settings, _ = settings_for('standard', 'configs')
        risk = LongitudinalPDModel.load(ROOT/'standard/models/pd/logistic_calibrated.joblib')
        learning.InitializationRecorder = SampleRecorder
        learning.run_pair(config, settings, protocol, risk, replay_root, 'standard', 32768, seed, [64, 64])
        teacher = joblib.load(ROOT/'information_gap/planners.joblib')['F0']['model']
        rows, checks = [], []
        for arm in ('RandomInit', 'ImitationInit'):
            original = output/'runs/32768'/arm/str(seed)
            folder = replay_root/'runs/32768'/arm/str(seed)
            for name in ('selected', 'final'):
                equal = same_parameters(PPO.load(original/f'{name}.zip'), PPO.load(folder/f'{name}.zip'))
                if not equal:
                    raise AssertionError(f'Instrumentation changed weights: {seed}, {arm}, {name}')
                checks.append(dict(seed=seed, arm=arm, checkpoint=name, identical_weights=equal))
            frames = []
            for index, batch in enumerate(joblib.load(folder/'actual_minibatches.joblib')):
                x = batch['observations']
                labels = np.array([choose(q, o, config) for q, o in zip(teacher.predict(x), x)])
                frames.append(pd.DataFrame(dict(seed=seed, experiment_id=arm, timesteps=batch['timesteps'],
                    batch_id=index, teacher_action=labels, sampled_action=batch['actions'].astype(int),
                    raw_gae=batch['advantages'], normalized_gae=batch['normalized'])))
            frame = pd.concat(frames, ignore_index=True)
            frame.to_csv(folder/'teacher_labeled_minibatches.csv.gz', index=False)
            for identity, group in frame.groupby(['seed', 'experiment_id', 'timesteps', 'teacher_action', 'sampled_action']):
                rows.append(dict(zip(['seed', 'experiment_id', 'timesteps', 'teacher_action', 'sampled_action'], identity),
                    count=len(group), raw_gae_mean=group.raw_gae.mean(), normalized_gae_mean=group.normalized_gae.mean(),
                    normalized_negative_fraction=group.normalized_gae.lt(0).mean()))
        pd.DataFrame(rows).to_csv(replay_root/f'pressure_{seed}.csv', index=False)
        write_json(replay_root/f'verification_{seed}.json', checks)
        print(f'EXACT MINIBATCH REPLAY {seed}', flush=True)


def main():
    output = ROOT/'policy_initialization'
    root = output/'diagnostic_replay'
    root.mkdir(exist_ok=True)
    for name in ('initialization_selection.json', 'dataset_validation.joblib'):
        if not (root/name).exists():
            shutil.copy2(output/name, root/name)
    for source in (output/'imitation_models').glob('Imitation_64x64/*/selected.zip'):
        destination = root/source.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
    write_json(root/'purpose.json', dict(
        purpose='Read-only sample-level instrumentation replay of existing short-budget experiment.',
        budget=32768, new_scientific_treatment=False, first_steps_instrumented=8192,
        teacher_use='Only after training, to label captured public observations.',
        acceptance='Selected and final tensors must be identical to the original run for both arms.'))
    seeds = [101, 202, 303, 404, 505]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(replay, seed) for seed in seeds]
        for future in futures:
            future.result()
    pd.concat([pd.read_csv(root/f'pressure_{seed}.csv') for seed in seeds], ignore_index=True).to_csv(
        output/'actual_teacher_minibatch_pressure.csv', index=False)
    write_json(root/'verification.json', dict(status='passed', exact_selected_and_final_weights=True,
        seeds=seeds, number_of_paired_replays=5))


if __name__ == '__main__':
    main()
