"""Deterministic baselines using the same observation contract as PPO."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from credit_rl.config import EnvironmentConfig
from credit_rl.envs.credit_limit_env import OBSERVATION_NAMES


class Policy(Protocol):
    def act(self, observation: np.ndarray) -> int: ...


@dataclass(frozen=True)
class StaticPolicy:
    config: EnvironmentConfig = EnvironmentConfig()

    def act(self, observation: np.ndarray) -> int:
        return self.config.action_multipliers.index(1.0)


@dataclass(frozen=True)
class ConstantPolicy:
    action: int

    def act(self, observation: np.ndarray) -> int:
        return self.action


@dataclass(frozen=True)
class RiskThresholdPolicy:
    """Preserves the prototype's low-PD expansion/high-PD contraction heuristic."""
    config: EnvironmentConfig = EnvironmentConfig()
    low_pd: float = 0.03
    high_pd: float = 0.12

    def act(self, observation: np.ndarray) -> int:
        pd = observation[OBSERVATION_NAMES.index("predicted_pd")]
        multipliers = self.config.action_multipliers
        if pd > self.high_pd:
            return int(np.argmin(multipliers))
        if pd < self.low_pd:
            return int(np.argmax(multipliers))
        return multipliers.index(1.0)


@dataclass
class SB3Policy:
    model: object

    def act(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
