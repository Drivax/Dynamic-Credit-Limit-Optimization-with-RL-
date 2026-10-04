"""Pure PPO actor warm start; read-only diagnostics after initialization."""
from functools import partial
import json
import random

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import torch
from threadpoolctl import threadpool_limits

from credit_rl.evaluation.policy_engine import PolicySpec, evaluate_policy
from credit_rl.evaluation.scenarios import make_scenarios
from credit_rl.experiments.information_ppo import RolloutRecorder, distribution, record_gradients
from credit_rl.experiments.main_evaluation import digest
from credit_rl.experiments.policy_initialization import ROOT, transfer_actor, write_json
from credit_rl.policies.baselines import SB3Policy
from credit_rl.policies.training import train_agent


class InitializedPPO(PPO):
    def __init__(self, *args, actor_checkpoint=None, critic_checkpoint=None,
                 transfer_observations=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = None
        self.transfer_check = None
        numpy_state, python_state = np.random.get_state(), random.getstate()
        with torch.random.fork_rng():
            if critic_checkpoint is not None:
                reference = PPO.load(critic_checkpoint, device='cpu').policy.state_dict()
                weights = self.policy.state_dict()
                for key in weights:
                    if key.startswith(('mlp_extractor.value_net.', 'value_net.')):
                        weights[key] = reference[key].clone()
                self.policy.load_state_dict(weights)
            if actor_checkpoint is not None:
                source = PPO.load(actor_checkpoint, device='cpu')
                self.transfer_check = transfer_actor(source, self, transfer_observations)
        np.random.set_state(numpy_state)
        random.setstate(python_state)

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['recorder']

    def train(self):
        # No teacher objects, targets or auxiliary loss are available here.
        super().train()
        if self.recorder is not None:
            self.recorder.after_update()


class EvaluationActor:
    def __init__(self, model, customer_seed, seed):
        self.model = model
        self.uniforms = np.random.default_rng(np.random.SeedSequence(
            [customer_seed, seed, 18401])).random(24)
        self.month = 0

    def act(self, observation):
        probabilities = distribution(self.model, observation[None])[0][0]
        action = min(int(np.searchsorted(probabilities.cumsum(), self.uniforms[self.month])), 4)
        self.month += 1
        return action


def economics(model, groups, config, risk, settings, seed, label, steps, kind):
    rows = []
    for scenario, customers in groups.items():
        for mode in ('deterministic', 'stochastic'):
            def factory(env, customer, mode=mode):
                return SB3Policy(model) if mode == 'deterministic' else EvaluationActor(model, customer.customer_seed, seed)
            spec = PolicySpec(label, factory, seed=seed)
            frame, _, _ = evaluate_policy(spec, customers, config, risk, settings, scenario, keep_history=False)
            rows.append(frame.assign(experiment_id=label, timesteps=steps, checkpoint_kind=kind, mode=mode))
    return pd.concat(rows, ignore_index=True)


class InitializationRecorder(RolloutRecorder):
    def __init__(self, seed, destination, observations, snapshots, initial_evaluation=None):
        super().__init__(seed, destination)
        self.observations, self.snapshot_steps = observations, snapshots
        self.updates, self.saved = [], []
        self.initial_evaluation = initial_evaluation

    def _on_training_start(self):
        super()._on_training_start()
        self.model.recorder = self
        self.save_snapshot(0)
        if self.initial_evaluation is not None:
            self.initial_evaluation(self.model)

    def _on_rollout_start(self):
        self.start_snapshot = None

    def after_update(self):
        row = dict(timesteps=self.num_timesteps, seed=self.seed,
                   buffer_timing='behavior before update; losses during update')
        row.update({key: float(value) for key, value in self.model.logger.name_to_value.items()
                    if key.startswith('train/') and np.isscalar(value)})
        norms = [g['preclip_norm'] for g in self.gradients if g['timesteps'] == self.num_timesteps]
        row['gradient_norm_mean'] = float(np.mean(norms))
        self.updates.append(row)
        if self.num_timesteps in self.snapshot_steps:
            self.save_snapshot(self.num_timesteps)
        pd.DataFrame(self.updates).to_csv(self.destination/'updates.csv', index=False)
        # Full raw diagnostics are needed only through 8192; avoid huge long-run buffers.
        if self.num_timesteps > 8192:
            self.rows.pop()

    def save_snapshot(self, steps):
        path = self.destination/f'temporal_{steps}.zip'
        self.model.save(path)
        probabilities, logits, entropy, values = distribution(self.model, self.observations)
        np.savez_compressed(self.destination/f'temporal_{steps}.npz',
            probabilities=probabilities, logits=logits, entropy=entropy, values=values)
        self.saved.append(dict(timesteps=steps, checkpoint_kind='initial' if steps == 0 else 'post_update',
                               checkpoint=str(path), sha256=digest(path)))
        pd.DataFrame(self.saved).to_csv(self.destination/'snapshots.csv', index=False)


def run_pair(config, settings, protocol, risk, output, profile, budget, seed, architecture,
             smoke_unqualified=False):
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    selection = json.loads((output/'initialization_selection.json').read_text())
    if selection['selected_architecture'] is None and not (profile == 'smoke' and smoke_unqualified):
        raise ValueError('Scientific PPO comparison requires qualified imitation')
    data = joblib.load(output/'dataset_validation.joblib')
    obs = data['observations']
    name = 'Imitation_'+'x'.join(map(str, architecture))
    actor = output/'imitation_models'/name/str(seed)/'selected.zip'
    reference = ROOT/f'ppo_diagnostics/runs/canonical/{seed}/checkpoint_0.zip'
    train = make_scenarios(config, settings, 'train', 'markov')
    validation = make_scenarios(config, settings, 'validation')
    for arm in ('RandomInit', 'ImitationInit'):
        folder = output/'runs'/str(budget)/arm/str(seed)
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'completed.json').exists():
            continue
        groups = {macro: make_scenarios(config, settings, 'validation', macro)
                  for macro in ('baseline', 'severe_stress')}
        def initial_evaluation(model):
            economics(model, groups, config, risk, settings, seed, arm, 0, 'initial').to_csv(
                folder/'initial_economics.csv', index=False)
        recorder = InitializationRecorder(seed, folder, obs, protocol[profile]['snapshots'], initial_evaluation)
        factory = partial(InitializedPPO, actor_checkpoint=actor if arm == 'ImitationInit' else None,
                          critic_checkpoint=reference, transfer_observations=obs)
        with record_gradients(recorder):
            metadata = train_agent(config, risk, settings, train, validation, seed=seed,
                destination=folder, total_timesteps=budget, overrides=dict(actor_network=architecture),
                diagnostic_callback=recorder, model_class=factory)
        recorder.flush()
        write_json(folder/'completed.json', dict(metadata=metadata,
            transfer=recorder.model.transfer_check, initialization=arm,
            teacher_in_objective=False, smoke_unqualified=smoke_unqualified,
            selected_sha256=digest(folder/'selected.zip'), final_sha256=digest(folder/'final.zip')))
    first = PPO.load(output/'runs'/str(budget)/'RandomInit'/str(seed)/'temporal_0.zip').policy.state_dict()
    second = PPO.load(output/'runs'/str(budget)/'ImitationInit'/str(seed)/'temporal_0.zip').policy.state_dict()
    if any(not torch.equal(v, second[k]) for k, v in first.items()
           if k.startswith(('mlp_extractor.value_net.', 'value_net.'))):
        raise AssertionError('Paired critics differ')
