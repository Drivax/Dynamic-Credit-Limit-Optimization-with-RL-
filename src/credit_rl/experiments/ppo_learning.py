"""Read-only fine logging around SB3 updates and explicitly declared interventions."""
import json
import random

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import torch

from credit_rl.evaluation.policy_engine import evaluate_policy
from credit_rl.evaluation.scenarios import make_scenarios
from credit_rl.evaluation.structural import DecisionSnapshot
from credit_rl.experiments.information_ppo import RolloutRecorder, distribution
from credit_rl.experiments.main_evaluation import digest
from credit_rl.policies.registry import ppo_spec


def state_dependence(actions, observations):
    actions = np.asarray(actions, dtype=int)
    probabilities = np.bincount(actions, minlength=5)/len(actions)
    bins = [np.searchsorted([.2, .6], observations[:, 10]),
            np.searchsorted([1/3, .5], observations[:, 3]),
            np.searchsorted([1/3, 2/3], observations[:, 0]), observations[:, 18:].argmax(1)]
    result = dict(diversity=1-probabilities.max())
    for label, bucket in zip(('pd', 'utilization', 'horizon', 'macro'), bins):
        joint = np.zeros((5, 3))
        np.add.at(joint, (actions, bucket), 1)
        joint /= joint.sum()
        expected = joint.sum(1)[:, None]*joint.sum(0)[None, :]
        positive = joint > 0
        result['mi_'+label] = float(np.sum(joint[positive]*np.log(joint[positive]/expected[positive])))
    return result


class LearningPPO(PPO):
    def __init__(self, *args, intervention=None, reference_initial=None, **kwargs):
        self.intervention = intervention or {}
        super().__init__(*args, **kwargs)
        if reference_initial is not None and any(k in self.intervention for k in ('actor_network', 'critic_network')):
            numpy_state, python_state = np.random.get_state(), random.getstate()
            with torch.random.fork_rng():
                reference = PPO.load(reference_initial, device='cpu').policy.state_dict()
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            target = self.policy.state_dict()
            preserve_actor = 'critic_network' in self.intervention
            for key in target:
                actor = key.startswith(('mlp_extractor.policy_net', 'action_net'))
                if actor == preserve_actor:
                    target[key] = reference[key].clone()
            self.policy.load_state_dict(target)
        self.recorder = None
        self.value_optimizer = None

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['recorder', 'value_optimizer']

    def extra_critic_updates(self):
        epochs = self.intervention.get('critic_extra_epochs', 0)
        if not epochs:
            return
        parameters = list(self.policy.mlp_extractor.value_net.parameters())+list(self.policy.value_net.parameters())
        if self.value_optimizer is None:
            self.value_optimizer = torch.optim.Adam(parameters, lr=float(self.lr_schedule(1)), eps=1e-5)
        observations = self.rollout_buffer.observations.reshape(-1, 21)
        returns = self.rollout_buffer.returns.ravel()
        for _ in range(epochs):
            # Fixed ordered batches add no training RNG draws or policy updates.
            for start in range(0, len(returns), self.batch_size):
                x = torch.as_tensor(observations[start:start+self.batch_size], device=self.device)
                y = torch.as_tensor(returns[start:start+self.batch_size], device=self.device)
                loss = torch.nn.functional.mse_loss(self.policy.predict_values(x).flatten(), y)
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
                self.value_optimizer.step()

    def train(self):
        if self.intervention.get('entropy_schedule') == 'linear_002_to_001':
            self.ent_coef = .01+.01*max(0., 1-self.num_timesteps/self._total_timesteps)
        super().train()
        self.extra_critic_updates()
        if self.recorder is not None:
            self.recorder.after_update()


class LearningRecorder(RolloutRecorder):
    def __init__(self, row, protocol, config, settings, risk, panels, destination):
        super().__init__(row['seed'], destination)
        self.row, self.protocol, self.config, self.settings = row, protocol, config, settings
        self.risk, self.panels = risk, panels
        self.updates, self.panel_rows, self.snapshots, self.fine_validation = [], [], [], []
        self.training_states = []
        self.diagnostic_scenarios = {macro: make_scenarios(config, settings, 'validation', macro,
            customers=min(protocol['diagnostic_validation_customers'], settings['population']['validation']))
            for macro in ('baseline', 'severe_stress')}

    def _on_training_start(self):
        super()._on_training_start()
        self.model.recorder = self
        self.observe_policy(0, save=True)

    def _on_rollout_start(self):
        self.training_snapshot = DecisionSnapshot.capture(self.training_env.envs[0].unwrapped)
        self.start_snapshot = None

    def _on_rollout_end(self):
        super()._on_rollout_end()
        self.training_states.append(dict(timesteps=self.num_timesteps, snapshot=self.training_snapshot,
            action=int(self.model.rollout_buffer.actions.ravel()[0]),
            raw_gae=float(self.model.rollout_buffer.advantages.ravel()[0])))

    def after_update(self):
        group = self.rows[-1]
        normalized = [x for x in self.minibatches if x['timesteps'] == self.num_timesteps]
        gradients = [x['preclip_norm'] for x in self.gradients if x['timesteps'] == self.num_timesteps]
        row = dict(timesteps=self.num_timesteps, seed=self.seed, experiment_id=self.row['experiment_id'],
            entropy=group.entropy.mean(), gradient_mean=np.mean(gradients), gradient_max=max(gradients),
            critic_gae_rmse=float(np.sqrt(np.mean((group.value-group.returns)**2)))/self.settings['ppo']['reward_scale'],
            entropy_coefficient=self.model.ent_coef)
        for key, value in self.model.logger.name_to_value.items():
            if key.startswith('train/') and np.isscalar(value):
                row[key] = float(value)
        for a in range(5):
            g = group[group.action == a]
            mini = [x for x in normalized if x['action'] == a]
            count = sum(x['count'] for x in mini)
            row[f'raw_gae_{a}'] = g.advantage.mean()
            row[f'normalized_gae_{a}'] = sum(x['normalized_sum'] for x in mini)/count if count else np.nan
            row[f'action_fraction_{a}'] = len(g)/len(group)
            row[f'probability_{a}'] = group[f'prob_{a}'].mean()
        obs = group[[f'obs_{i}' for i in range(21)]].to_numpy()
        row.update(state_dependence(group.action.to_numpy(), obs))
        self.updates.append(row)
        period = self.protocol['dense_every'] if self.row['family'] == 'canonical' else self.protocol['snapshot_every']
        self.observe_policy(self.num_timesteps, save=self.num_timesteps % period == 0 or self.num_timesteps == self.row['budget'])

    def observe_policy(self, steps, save):
        obs = self.panels['validation_observations']
        p, logits, entropy, values = distribution(self.model, obs)
        states = self.panels['validation_states']
        for scenario in ('baseline', 'severe_stress'):
            mask = states.scenario.eq(scenario).to_numpy()
            deterministic = p[mask].argmax(1)
            row = dict(experiment_id=self.row['experiment_id'], seed=self.seed, timesteps=steps, scenario=scenario,
                entropy=entropy[mask].mean(), deterministic_contraction=(deterministic == 0).mean(),
                stochastic_contraction=p[mask, 0].mean(), value_mean=values[mask].mean()/self.settings['ppo']['reward_scale'],
                **state_dependence(deterministic, obs[mask]))
            for a in range(5):
                row[f'probability_{a}'] = p[mask, a].mean()
                row[f'logit_{a}'] = logits[mask, a].mean()
                row[f'deterministic_fraction_{a}'] = (deterministic == a).mean()
            self.panel_rows.append(row)
        if save:
            folder = self.destination/'snapshots'
            folder.mkdir(exist_ok=True)
            path = folder/f'update_{steps}.zip'
            self.model.save(path)
            self.snapshots.append(dict(timesteps=steps, checkpoint=str(path), sha256=digest(path),
                timing='after PPO update' if steps else 'initial before any update'))
            # Fine diagnostic validation is never connected to checkpoint selection.
            for scenario, group in self.diagnostic_scenarios.items():
                e, _, _ = evaluate_policy(ppo_spec(self.model, self.seed, str(path)), group, self.config,
                    self.risk, self.settings, scenario, keep_history=False)
                self.fine_validation.append(dict(timesteps=steps, scenario=scenario,
                    discounted_reward=e.discounted_reward.mean(), net_economic_value=e.net_economic_value.mean()))
        # Persist incrementally, so interrupted runs expose real completed updates only.
        pd.DataFrame(self.updates).to_csv(self.destination/'updates.csv', index=False)
        pd.DataFrame(self.panel_rows).to_csv(self.destination/'panel.csv', index=False)

    def flush(self):
        frame = super().flush()
        pd.DataFrame(self.snapshots).to_csv(self.destination/'snapshots.csv', index=False)
        pd.DataFrame(self.fine_validation).to_csv(self.destination/'fine_validation.csv', index=False)
        joblib.dump(self.training_states, self.destination/'training_states.joblib')
        (self.destination/'intervention.json').write_text(json.dumps(self.row, indent=2))
        return frame
