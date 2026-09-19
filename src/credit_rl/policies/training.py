"""PPO on a fixed training population; validation-only checkpoint selection."""
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from credit_rl import CreditLimitEnv
from credit_rl.evaluation.policy_engine import evaluate_policy
from .decision import transform_observation
from .registry import ppo_spec


class TrainingEnvironment(CreditLimitEnv):
    """Only supplied RL_TRAIN scenarios may be sampled. No holdout access."""
    def __init__(self, scenarios, pd_model, config, settings, without_pd=False):
        if not scenarios or any(s.partition != "train" for s in scenarios):
            raise ValueError("Training environment requires RL_TRAIN customers only")
        super().__init__(pd_model=pd_model, config=config, record_history=False,
                         severe_delinquency_months=settings["guardrails"]["severe_delinquency_months"])
        self.scenarios = tuple(scenarios)
        self.selection_rng = np.random.default_rng(0)
        self.without_pd = without_pd
        self.reward_scale = settings["ppo"]["reward_scale"]

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("External scenario overrides forbidden in PPO training")
        if seed is not None:
            self.selection_rng = np.random.default_rng(seed)
        scenario = self.scenarios[int(self.selection_rng.integers(len(self.scenarios)))]
        obs, info = super().reset(seed=seed, options=scenario.reset_options())
        return transform_observation(obs, self.without_pd), info

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        info["economic_reward_eur"] = reward
        return transform_observation(obs, self.without_pd), reward*self.reward_scale, term, trunc, info


class ValidationCheckpoint(BaseCallback):
    def __init__(self, spec_seed, validation, config, pd_model, settings, destination, without_pd):
        super().__init__()
        if any(s.partition != "validation" for s in validation):
            raise ValueError("Checkpoint selection requires validation population")
        self.seed, self.validation, self.config, self.pd_model = spec_seed, validation, config, pd_model
        self.settings, self.destination, self.without_pd = settings, Path(destination), without_pd
        self.best_score, self.best_steps = -np.inf, 0
        self.records = []
        self.evaluation_seconds = 0.

    def _on_training_start(self):
        self.evaluate()

    def _on_step(self):
        obs = self.locals.get("new_obs")
        rewards = self.locals.get("rewards")
        if obs is not None and not np.isfinite(obs).all():
            raise FloatingPointError("Nonfinite PPO observation")
        if rewards is not None and not np.isfinite(rewards).all():
            raise FloatingPointError("Nonfinite PPO reward")
        if self.num_timesteps % self.settings["ppo"]["validation_every"] == 0:
            self.evaluate()
        return True

    def evaluate(self):
        tick = perf_counter()
        spec = ppo_spec(self.model, self.seed, "in_memory", self.without_pd)
        episodes, _, _ = evaluate_policy(spec, self.validation, self.config, self.pd_model,
                                         self.settings, "validation_baseline", keep_history=False)
        score = float(episodes.discounted_reward.mean())
        self.records.append(dict(timesteps=self.num_timesteps, validation_discounted_reward=score,
            validation_net_value=float(episodes.net_economic_value.mean()),
            validation_default_rate=float(episodes.defaulted.mean())))
        self.model.save(self.destination/f"checkpoint_{self.num_timesteps}")
        if score > self.best_score:
            self.best_score, self.best_steps = score, self.num_timesteps
            self.model.save(self.destination/"selected")
        pd.DataFrame(self.records).to_csv(self.destination/"validation.csv", index=False)
        self.evaluation_seconds += perf_counter()-tick
        print(f"seed={self.seed} without_pd={self.without_pd} steps={self.num_timesteps} validation={score:.2f}", flush=True)

    def _on_training_end(self):
        # At a rollout boundary, the last callback precedes the final parameter update.
        # Evaluate that final update explicitly, even when the timestep is the same.
        self.evaluate()


def train_agent(config, pd_model, settings, train, validation, *, seed, destination, without_pd=False,
                overrides=None, total_timesteps=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    p = {**settings["ppo"], **(overrides or {})}
    env = Monitor(TrainingEnvironment(train, pd_model, config, settings, without_pd), str(destination/"episodes"))
    model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
        policy_kwargs=dict(net_arch=dict(pi=p["network"], vf=p["network"])),
        **{key: p[key] for key in ("learning_rate", "gamma", "gae_lambda", "clip_range", "n_steps", "batch_size",
                                  "n_epochs", "ent_coef", "vf_coef", "max_grad_norm")})
    model.set_logger(configure(str(destination), ["csv"]))
    callback = ValidationCheckpoint(seed, validation, config, pd_model, settings, destination, without_pd)
    tick = perf_counter()
    model.learn(total_timesteps=total_timesteps or p["total_timesteps"], callback=callback, log_interval=1)
    seconds = perf_counter()-tick
    model.save(destination/"final")
    selected = PPO.load(destination/"selected", device="cpu")
    obs, _ = env.reset()
    action = selected.predict(obs, deterministic=True)[0]
    reload = PPO.load(destination/"selected", device="cpu")
    np.testing.assert_array_equal(action, reload.predict(obs, deterministic=True)[0])
    metadata = dict(seed=seed, without_pd=without_pd, actual_timesteps=model.num_timesteps,
        training_seconds=seconds, validation_seconds=callback.evaluation_seconds,
        training_steps_per_second=model.num_timesteps/max(seconds-callback.evaluation_seconds, 1e-9),
        selected_timesteps=callback.best_steps, validation_score=callback.best_score,
        criterion="maximum validation mean discounted raw reward, earliest on tie",
        hyperparameters=p, reward_scale=settings["ppo"]["reward_scale"],
        observation_normalization="fixed public transforms, no fitted statistics",
        training_customer_ids=[s.customer_id for s in train], validation_customer_ids=[s.customer_id for s in validation])
    (destination/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    env.close()
    return metadata
