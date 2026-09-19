"""Deployable policies consume only the public 21-dimensional observation."""
from dataclasses import dataclass
import numpy as np

from credit_rl.envs.observation import OBSERVATION_NAMES
from credit_rl.envs.constraints import effective_limit

PD_INDEX = OBSERVATION_NAMES.index("predicted_pd")


def transform_observation(observation, without_pd=False, pd_multiplier=1.):
    out = np.asarray(observation, dtype=np.float32).copy()
    out[PD_INDEX] = 0. if without_pd else np.clip(out[PD_INDEX]*pd_multiplier, 0, 1)
    return out


def decode(observation, config):
    x = dict(zip(OBSERVATION_NAMES, map(float, observation)))
    def inverse(name):
        v = x[name]
        return v/max(1-v, 1e-8)
    cap = config.environment.max_limit
    return dict(limit=x["credit_limit_scaled"]*cap, balance=inverse("balance_scaled")*cap,
        income=inverse("income_bounded")*cap, utilization=inverse("utilization_bounded"),
        payment_ratio=x["payment_ratio"], spending=inverse("spend_bounded")*cap,
        delinquency=round(inverse("months_delinquent_bounded")), pd=x["predicted_pd"],
        macro_spending_growth=(2*x["macro_spending_growth_scaled"]-1)*.5,
        macro_stress=inverse("macro_stress"))


def legal_actions(observation, config, severe_months=3):
    state = decode(observation, config)
    allowed = []
    for action, multiplier in enumerate(config.environment.action_multipliers):
        limit, blocked = effective_limit(state["limit"], state["delinquency"], multiplier,
                                         config.environment, severe_months)
        if multiplier == 1 or (not blocked and abs(limit-state["limit"]) > .01):
            allowed.append(action)
    return allowed


@dataclass
class ConstantAdjustment:
    config: object
    multiplier: float = 1.
    severe_months: int = 3

    def act(self, observation):
        action = self.config.environment.action_multipliers.index(self.multiplier)
        return action if action in legal_actions(observation, self.config, self.severe_months) else self.config.environment.action_multipliers.index(1.)


@dataclass
class RandomPolicy:
    config: object
    seed: int
    severe_months: int = 3

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def act(self, observation):
        return int(self.rng.choice(legal_actions(observation, self.config, self.severe_months)))


@dataclass
class PDThreshold:
    config: object
    low: float = .2
    high: float = .6
    severe_months: int = 3

    def act(self, observation):
        p = float(observation[PD_INDEX])
        multiplier = .9 if p >= self.high else 1.1 if p < self.low else 1.
        return ConstantAdjustment(self.config, multiplier, self.severe_months).act(observation)


@dataclass
class UtilizationPD(PDThreshold):
    utilization_threshold: float = .75

    def act(self, observation):
        s = decode(observation, self.config)
        if s["pd"] >= self.high or s["delinquency"] >= self.severe_months:
            multiplier = .9
        elif s["pd"] < self.low and s["utilization"] >= self.utilization_threshold and s["delinquency"] == 0:
            multiplier = 1.1
        else:
            multiplier = 1.
        return ConstantAdjustment(self.config, multiplier, self.severe_months).act(observation)


@dataclass
class MyopicEconomic:
    """One-step observable surrogate; flat-hazard conversion is an approximation.

    No DGP call, latent traits or future values. Risk sensitivities are explicit
    assumed surrogate coefficients, not estimates of causal action effects.
    """
    config: object
    pd_horizon: int = 12
    severe_months: int = 3
    utilization_risk_slope: float = 1.
    burden_risk_slope: float = .3

    def values(self, observation):
        s, c = decode(observation, self.config), self.config
        values = np.full(len(c.environment.action_multipliers), -np.inf)
        monthly = np.clip(-np.expm1(np.log1p(-min(s["pd"], 1-1e-8))/self.pd_horizon), 1e-8, 1-1e-8)
        for action in legal_actions(observation, c, self.severe_months):
            limit, _ = effective_limit(s["limit"], s["delinquency"], c.environment.action_multipliers[action], c.environment, self.severe_months)
            payment = min(s["balance"]*s["payment_ratio"], c.dynamics.payment_income_cap*s["income"])
            remaining = max(0., s["balance"]-payment)
            spend = min(max(0., limit-remaining), s["spending"]*(limit/s["limit"])**c.dynamics.spend_limit_elasticity*(1+s["macro_spending_growth"]))
            balance = remaining+spend
            logit = np.log(monthly/(1-monthly)) + self.utilization_risk_slope*(balance/limit-s["utilization"]) + self.burden_risk_slope*(balance-s["balance"])/s["income"]
            hazard = float(np.exp(-np.logaddexp(0., -logit))) if balance else 0.
            r = c.reward
            values[action] = ((1-hazard)*s["balance"]*r.annual_percentage_rate/12 + spend*r.fee_rate
                -hazard*r.loss_given_default*balance-r.annual_funding_rate/12*balance
                -r.capital_weight*r.rwa_factor*s["pd"]*balance
                -r.constraint_weight*max(0., s["pd"]-r.max_pd_threshold))
        return values

    def act(self, observation):
        values = self.values(observation)
        # Prefer no change among numerically equal choices, then smallest adjustment.
        choices = np.flatnonzero(np.isclose(values, np.max(values), rtol=0, atol=1e-8))
        return int(min(choices, key=lambda i: abs(self.config.environment.action_multipliers[i]-1)))
