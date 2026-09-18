"""Collect whole trajectories through the same interface used for learning."""

from dataclasses import dataclass

import pandas as pd

from credit_rl.envs.credit_limit_env import CreditLimitEnv
from credit_rl.policies.baselines import Policy


@dataclass(frozen=True)
class Trajectory:
    _frame: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)


def simulate_customer(env: CreditLimitEnv, policy: Policy, *, seed: int,
                      options: dict | None = None) -> Trajectory:
    """Run one customer to default/horizon; include the month-zero snapshot."""
    observation, _ = env.reset(seed=seed, options=options)
    while True:
        observation, _, terminated, truncated, _ = env.step(policy.act(observation))
        if terminated or truncated:
            return Trajectory(env.get_history())
