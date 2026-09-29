"""Privileged full-state diagnostics using canonical transitions, never training.

Q estimates condition on a visited full state and a fixed macro scenario. They
are continuation-policy values, not Q* or observable-history posterior values.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_rl.envs.constraints import effective_limit
from credit_rl.envs.observation import build_observation
from credit_rl.policies.decision import legal_actions
from credit_rl.reward import calculate_reward
from credit_rl.risk.features import FEATURE_NAMES, history_vector, observable_row
from credit_rl.risk.pd_model import ObservedRiskFeatures
from credit_rl.simulation.dgp import CreditDGP
from credit_rl.simulation.shocks import ShockPath

COMPONENTS = ('interest_income', 'fee_income', 'credit_loss', 'funding_cost',
              'capital_cost', 'constraint_penalty')
SIGNS = np.array([1., 1., -1., -1., -1., -1.])
METRICS = (*COMPONENTS, 'reward', 'net_economic_value', 'purchases',
           'next_balance', 'utilization', 'default_probability', 'default_incidence',
           'effective_action', 'active')


@dataclass(frozen=True)
class DecisionSnapshot:
    state: object
    traits: object
    history: tuple
    predicted_pd: float
    elapsed: int
    macro_path: object
    severe_months: int | None

    @classmethod
    def capture(cls, env):
        if env._done:
            raise ValueError('Cannot diagnose an inactive episode')
        return cls(env.state, env._traits, tuple(dict(x) for x in env._risk_history),
                   env._predicted_pd, env._elapsed, env._macro_path,
                   env.severe_delinquency_months)


def decompose(previous, outcome, decision_pd, config):
    """Signed identities follow the existing RewardBreakdown/evaluator exactly."""
    breakdown = calculate_reward(previous, outcome, decision_pd, config.reward)
    parts = np.array([getattr(breakdown, key) for key in COMPONENTS])
    reward = float(parts @ SIGNS)
    net = float(parts[:4] @ SIGNS[:4])
    if not np.isclose(reward, breakdown.total, rtol=1e-12, atol=1e-9):
        raise AssertionError('Reward decomposition mismatch')
    return parts, reward, net


def hypothetical_paths(snapshot, draws, horizon, seed):
    """Fresh draws independent of the visitation path; shared by every action."""
    if draws < 2 or horizon < 1:
        raise ValueError('Need at least two draws and a positive horizon')
    return tuple(ShockPath.generate(snapshot.state.customer_id, seed+i, horizon)
                 for i in range(draws))


def predict_batch(model, histories, states):
    if hasattr(model, 'predict_array'):
        indices = [FEATURE_NAMES.index(x) for x in model.feature_names]
        return model.predict_array(np.array([history_vector(h)[indices] for h in histories]))
    if hasattr(model, 'predict_history'):
        return np.array([model.predict_history(h) for h in histories])
    return np.array([model.predict(ObservedRiskFeatures.from_state(s)) for s in states])


def rollout(snapshot, config, pd_model, policy, paths, horizon):
    """Return [draw, requested initial action, month, metric]; zero after exit.

    Batches only PD/PPO inference; every transition and component calls canonical
    code. Histories end at the current decision. The policy sees observations only.
    No original environment, config, model or visited shock path is mutated.
    """
    remaining = config.environment.horizon - snapshot.elapsed
    if horizon < 1 or remaining < 1 or snapshot.state.defaulted:
        raise ValueError('Positive active remaining horizon required')
    length = min(horizon, remaining)
    if any(len(p.months) < length for p in paths):
        raise ValueError('Insufficient shock path')
    actions = len(config.environment.action_multipliers)
    count = len(paths)*actions
    data = np.zeros((len(paths), actions, length, len(METRICS)))
    states = [snapshot.state]*count
    histories = [deque((dict(x) for x in snapshot.history), maxlen=7) for _ in range(count)]
    pds = np.full(count, snapshot.predicted_pd)
    active = list(range(count))
    dgp = CreditDGP(config)
    for k in range(length):
        if not active:
            break
        observations = np.array([build_observation(states[j], pds[j], snapshot.elapsed+k, config)
                                 for j in active])
        if k == 0:
            chosen = [j % actions for j in active]
        elif hasattr(policy, 'act_indexed'):
            chosen = policy.act_indexed(observations, active, k, actions)
        elif hasattr(policy, 'model'):
            chosen = policy.model.predict(observations, deterministic=True)[0]
        else:
            chosen = [policy.act(o) for o in observations]
        survivors = []
        for j, action in zip(active, chosen):
            previous = states[j]
            multiplier = config.environment.action_multipliers[int(action)]
            limit, _ = effective_limit(previous.credit_limit, previous.months_delinquent,
                                       multiplier, config.environment, snapshot.severe_months)
            outcome = dgp.step(previous, limit/previous.credit_limit, snapshot.traits,
                               paths[j//actions].months[k],
                               snapshot.macro_path.states[snapshot.elapsed+k+1])
            parts, reward, net = decompose(previous, outcome, pds[j], config)
            data[j//actions, j % actions, k] = [*parts, reward, net, outcome.spending,
                outcome.exposure, outcome.state.utilization, outcome.p_default_true,
                float(outcome.state.defaulted), outcome.effective_multiplier-1, 1.]
            states[j] = outcome.state
            histories[j].append(observable_row(outcome.state))
            if not outcome.state.defaulted:
                survivors.append(j)
        active = survivors
        if active and k+1 < length:
            pds[active] = predict_batch(pd_model, [histories[j] for j in active],
                                       [states[j] for j in active])
    return data


def admissible(snapshot, config):
    obs = build_observation(snapshot.state, snapshot.predicted_pd, snapshot.elapsed, config)
    return legal_actions(obs, config, snapshot.severe_months)


def action_statistics(samples, allowed, multipliers):
    """Paired CRN gaps; ties prefer hold then smallest adjustment.

    Selection-adjusted planning uses disjoint draw halves elsewhere. These SEs
    are descriptive MC errors conditional on the selected pair, not certificates.
    """
    means = samples.mean(axis=0)
    order = sorted(allowed, key=lambda a: (-round(float(means[a]), 8), abs(multipliers[a]-1)))
    best = order[0]
    second = order[1] if len(order) > 1 else best
    delta = samples[:, best]-samples[:, second]
    return dict(best_action=best, second_action=second, gap=float(delta.mean()),
                gap_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),
                tie_count=int(sum(np.isclose(means[a], means[best], atol=1e-8, rtol=0)
                                  for a in allowed)))


def value_tables(snapshot, config, cube, state_id, continuation, horizons, gammas):
    values, gaps, planning = [], [], []
    allowed = admissible(snapshot, config)
    mult = config.environment.action_multipliers
    reward_index = METRICS.index('reward')
    immediate = cube[:, :, 0, reward_index]
    one = action_statistics(immediate, allowed, mult)['best_action']
    split = len(cube)//2
    one_train = action_statistics(immediate[:split], allowed, mult)['best_action'] if split > 1 else one
    for h in horizons:
        length = min(h, cube.shape[2])
        for gamma in gammas:
            summed = (cube[:, :, :length] * (gamma**np.arange(length))[None, None, :, None]).sum(axis=2)
            samples = summed[:, :, reward_index]
            stats = action_statistics(samples, allowed, mult)
            tags = dict(state_id=state_id, continuation=continuation, horizon=h,
                        effective_horizon=length, gamma=gamma, draws=len(cube))
            gaps.append({**tags, **stats, 'contraction_advantage': float((samples[:, 0]-samples[:, 2]).mean()),
                         'contraction_advantage_se': float((samples[:, 0]-samples[:, 2]).std(ddof=1)/np.sqrt(len(cube)))})
            best = stats['best_action']
            train_best = action_statistics(samples[:split], allowed, mult)['best_action'] if split > 1 else best
            heldout = samples[split:, train_best]-samples[split:, one_train]
            planning.append({**tags, 'one_step_action': one, 'multi_step_action': best,
                'disagreement': int(one != best), 'opportunity': float((samples[:, best]-samples[:, one]).mean()),
                'heldout_opportunity': float(heldout.mean()),
                'heldout_se': float(heldout.std(ddof=1)/np.sqrt(len(heldout))) if len(heldout)>1 else np.nan,
                'immediate_sacrifice': float((immediate[:, one]-immediate[:, best]).mean())})
            for a in range(len(mult)):
                values.append({**tags, 'action': a, 'requested_action': mult[a]-1,
                    'admissible': a in allowed, 'expected_reward': float(samples[:, a].mean()),
                    'reward_se': float(samples[:, a].std(ddof=1)/np.sqrt(len(cube))),
                    **{name: float(summed[:, a, i].mean()) for i, name in enumerate(METRICS[:8])},
                    **{'first_'+name: float(cube[:, a, 0, METRICS.index(name)].mean())
                       for name in METRICS[8:]},
                    'revenue': float(summed[:, a, :2].sum(axis=1).mean()),
                    'undiscounted_net_economic_value': float(cube[:, a, :length, 7].sum(axis=1).mean())})
    return pd.DataFrame(values), pd.DataFrame(gaps), pd.DataFrame(planning)
