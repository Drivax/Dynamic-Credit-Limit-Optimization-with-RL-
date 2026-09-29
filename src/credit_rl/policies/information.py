"""Information-restricted, supervised diagnostic planners (never dynamic optima)."""
from dataclasses import asdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from credit_rl.envs.observation import OBSERVATION_NAMES
from credit_rl.policies.decision import legal_actions
from credit_rl.risk.features import ALLOWED_AT_DECISION_TIME

FEATURE_SETS = ('F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'Full')
BEHAVIORAL = {4, 5, 6, 8, 11, 12}


def public_features(observations, actions, kind='F0'):
    """Only a supplied public prefix; no environment, identity, targets or traits."""
    x = np.asarray(observations, dtype=np.float32)
    a = np.asarray(actions, dtype=float)
    if x.ndim != 2 or x.shape[1] != 21 or len(a) != len(x)-1:
        raise ValueError('Expected a 21D observation prefix and preceding actions')
    if kind not in FEATURE_SETS[:-1]:
        raise ValueError('Unknown public information set')
    indices = [i for i in range(21) if not (kind == 'F1' and i == 10)
               and not (kind == 'F2' and i in BEHAVIORAL)]
    values = list(x[-1, indices])
    names = [OBSERVATION_NAMES[i] for i in indices]
    if kind in ('F3', 'F4', 'F5'):
        depth = 3 if kind == 'F3' else 6
        for lag in range(1, depth+1):
            values.extend(x[-1-lag] if len(x) > lag else [np.nan]*21)
            names.extend(f'lag{lag}_{n}' for n in OBSERVATION_NAMES)
            values.append(a[-lag] if len(a) >= lag else np.nan)
            names.append(f'lag{lag}_action')
        past = x[:-1] if kind == 'F5' else x[max(0, len(x)-depth-1):-1]
        for label, vector in (
            ('mean', past.mean(0) if len(past) else np.full(21, np.nan)),
            ('std', past.std(0) if len(past) else np.full(21, np.nan)),
            ('trend', (x[-1]-past[0])/len(past) if len(past) else np.full(21, np.nan)),
        ):
            values.extend(vector)
            names.extend(f'history_{label}_{n}' for n in OBSERVATION_NAMES)
        aa = a if kind == 'F5' else a[-depth:]
        values.extend([float(np.mean(aa == i)) if len(aa) else 0. for i in range(5)])
        names.extend(f'history_action_fraction_{i}' for i in range(5))
        if kind == 'F5':
            values.extend(x[0])
            names.extend(f'initial_{n}' for n in OBSERVATION_NAMES)
    return np.asarray(values, dtype=float), names


def privileged_features(observation, snapshot):
    """Explicitly privileged projection; excludes future shocks and macro paths."""
    values, names = public_features([observation], [], 'F0')
    traits = asdict(snapshot.traits)
    extra = list(traits.values()) + [snapshot.state.initial_income, snapshot.state.initial_score]
    labels = ['latent_'+n for n in traits] + ['initial_income', 'initial_score']
    extra.extend(snapshot.state.late_history)
    labels.extend(f'ordered_late_{i}' for i in range(6))
    history = list(snapshot.history)
    for lag in range(7):
        row = history[-1-lag] if len(history) > lag else {}
        extra.extend(row.get(n, np.nan) for n in ALLOWED_AT_DECISION_TIME)
        labels.extend(f'pd_history_{lag}_{n}' for n in ALLOWED_AT_DECISION_TIME)
    return np.concatenate([values, extra]), names+labels


class QEstimator:
    """Common hold level plus action differences; deterministic small tree models."""
    def __init__(self, leaves=7, iterations=100, minimum_leaf=12, seed=0):
        self.parameters = dict(max_leaf_nodes=leaves, max_iter=iterations,
            min_samples_leaf=minimum_leaf, early_stopping=False, random_state=seed)

    def fit(self, x, q, weight=None):
        self.hold = HistGradientBoostingRegressor(**self.parameters).fit(x, q[:, 2], sample_weight=weight)
        self.differences = {}
        for a in (0, 1, 3, 4):
            self.differences[a] = HistGradientBoostingRegressor(**self.parameters).fit(
                x, q[:, a]-q[:, 2], sample_weight=weight)
        return self

    def predict(self, x):
        result = np.repeat(self.hold.predict(x)[:, None], 5, axis=1)
        for a, model in self.differences.items():
            result[:, a] += model.predict(x)
        return result


def choose(values, observation, config):
    allowed = legal_actions(observation, config)
    best = max(values[a] for a in allowed)
    ties = [a for a in allowed if np.isclose(values[a], best, atol=1e-8, rtol=0)]
    return min(ties, key=lambda a: (abs(a-2), a))


class ObservationPlanner:
    information_set = 'OBSERVABLE_ONLY'

    def __init__(self, estimator, config, kind='F0'):
        if kind not in ('F0', 'F1', 'F2'):
            raise ValueError('Current observation planner cannot consume history')
        self.estimator, self.config, self.kind = estimator, config, kind

    def act(self, observation):
        features, _ = public_features([observation], [], self.kind)
        return choose(self.estimator.predict(features[None])[0], observation, self.config)


class HistoryPlanner:
    information_set = 'OBSERVABLE_HISTORY'

    def __init__(self, estimator, config, kind='F5'):
        if kind not in ('F3', 'F4', 'F5'):
            raise ValueError('Unknown history depth')
        self.estimator, self.config, self.kind = estimator, config, kind
        self.observations, self.actions = [], []

    def act(self, observation):
        self.observations.append(np.asarray(observation).copy())
        features, _ = public_features(self.observations, self.actions, self.kind)
        action = choose(self.estimator.predict(features[None])[0], observation, self.config)
        self.actions.append(action)
        return action


class FullStatePlanner:
    """Privileged diagnostic, not deployable and not a guaranteed upper bound."""
    information_set = 'PRIVILEGED_FULL_STATE'

    def __init__(self, estimator, config, env):
        self.estimator, self.config, self.env = estimator, config, env

    def act(self, observation):
        from credit_rl.evaluation.structural import DecisionSnapshot
        features, _ = privileged_features(observation, DecisionSnapshot.capture(self.env))
        return choose(self.estimator.predict(features[None])[0], observation, self.config)
