"""SIMULATOR-ONLY ORACLE. Privileged, non-deployable one-step diagnostic.

Uses hidden traits but never the actual future shock/macro path. This is not
a bound on the optimal multi-period value, nor a same-information comparison.
"""
from dataclasses import dataclass
import numpy as np
from credit_rl.simulation.dgp import CreditDGP
from credit_rl.simulation.shocks import ShockPath
from credit_rl.policies.decision import legal_actions, PD_INDEX


@dataclass
class OracleMyopic:
    env: object
    draws: int = 12
    information_set: str = "SIMULATOR_ONLY_ORACLE"

    def __post_init__(self):
        self.dgp = CreditDGP(self.env.config)
        # Independent hypothetical draws, shared across candidate actions.
        self.draws_by_month = [ShockPath.generate("oracle_hypothetical", 991000+t, self.draws).months
                               for t in range(self.env.config.environment.horizon)]

    def act(self, observation):
        env, state, r = self.env, self.env.state, self.env.config.reward
        pd = float(observation[PD_INDEX])
        values = np.full(env.action_space.n, -np.inf)
        for action in legal_actions(observation, env.config, env.severe_delinquency_months):
            reward = []
            for shocks in self.draws_by_month[state.month]:
                outcome = self.dgp.step(state, env.config.environment.action_multipliers[action], env._traits,
                                        shocks, state.macro_state)
                h, b = outcome.p_default_true, outcome.exposure
                reward.append((1-h)*state.balance*r.annual_percentage_rate/12 + outcome.spending*r.fee_rate
                    -h*r.loss_given_default*b-r.annual_funding_rate/12*b-r.capital_weight*r.rwa_factor*pd*b
                    -r.constraint_weight*max(0., pd-r.max_pd_threshold))
            values[action] = np.mean(reward)
        return int(np.argmax(values))
