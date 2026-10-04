"""Fixed-panel preservation estimands, independent of PPO optimization."""
import numpy as np
import pandas as pd

from credit_rl.experiments.ppo_measurements import observable_regret


def drift(probabilities, initial, previous):
    p = np.clip(probabilities, 1e-30, 1)
    q = np.clip(initial, 1e-30, 1)
    r = np.clip(previous, 1e-30, 1)
    return dict(kl_initial=np.sum(p*np.log(p/q), axis=1),
                kl_previous=np.sum(p*np.log(p/r), axis=1),
                action_flip=(p.argmax(1) != q.argmax(1)).astype(float))


def survival(actions, initial_actions, teacher_actions):
    eligible = initial_actions == teacher_actions
    # NaN means outside the conditional population, never a failure.
    return np.where(eligible, (actions == teacher_actions).astype(float), np.nan)


def preservation(regret, initial_regret):
    return np.asarray(regret)-np.asarray(initial_regret)


def panel_measurements(data, probabilities, initial, previous, config):
    observations, labels, q = data['observations'], data['labels'], data['q']
    actions, initial_actions = probabilities.argmax(1), initial.argmax(1)
    frame = data['states'].copy()
    regret = np.stack([observable_regret(q, np.full(len(q), a), observations, config)
                       for a in range(5)], axis=1)
    frame['action'], frame['teacher_action'] = actions, labels
    frame['teacher_region'] = np.select([labels < 2, labels == 2], ['contraction', 'hold'], default='increase')
    frame['agreement'] = (actions == labels).astype(float)
    frame['stochastic_agreement'] = probabilities[np.arange(len(q)), labels]
    frame['teacher_regret'] = regret[np.arange(len(q)), actions]
    frame['stochastic_regret'] = (probabilities*regret).sum(1)
    frame['preservation_loss'] = preservation(frame.teacher_regret, regret[np.arange(len(q)), initial_actions])
    frame['stochastic_preservation_loss'] = (probabilities*regret).sum(1)-(initial*regret).sum(1)
    frame['boundary_survival'] = survival(actions, initial_actions, labels)
    frame['entropy'] = -(probabilities*np.log(np.clip(probabilities, 1e-30, 1))).sum(1)
    frame['contraction'] = (actions == 0).astype(float)
    frame['stochastic_contraction'] = probabilities[:, 0]
    for key, value in drift(probabilities, initial, previous).items():
        frame[key] = value
    for a in range(5):
        frame[f'probability_{a}'] = probabilities[:, a]
    frame['contraction_logit_change'] = np.log(np.clip(probabilities[:, 0], 1e-30, 1))-np.log(np.clip(previous[:, 0], 1e-30, 1))
    frame['teacher_logit_change'] = np.log(np.clip(probabilities[np.arange(len(q)), labels], 1e-30, 1))-np.log(np.clip(previous[np.arange(len(q)), labels], 1e-30, 1))
    return frame


def bucket_quality(frame, observations):
    rows = []
    buckets = dict(pd=np.searchsorted([.2, .6], observations[:, 10]),
                   utilization=np.searchsorted([1/3, .5], observations[:, 3]),
                   elapsed=np.searchsorted([1/3, 2/3], observations[:, 0]),
                   macro=observations[:, 18:].argmax(1))
    for name, values in buckets.items():
        for (scenario, bucket), group in frame.assign(bucket=values).groupby(['scenario', 'bucket']):
            rows.append(dict(scenario=scenario, feature=name, bucket=int(bucket), count=len(group),
                requested_accuracy=group.agreement.mean(), effective_accuracy=group.effective_agreement.mean(),
                teacher_regret=group.teacher_regret.mean()))
    return pd.DataFrame(rows)
