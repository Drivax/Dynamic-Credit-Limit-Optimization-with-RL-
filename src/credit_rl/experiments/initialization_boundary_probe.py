"""Supplemental validation-only action-stratified probes; never headline values."""
import json

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
import torch

from credit_rl.experiments.information_ppo import distribution
from credit_rl.experiments.policy_initialization import ROOT, write_json
from credit_rl.experiments.main_evaluation import settings_for
from credit_rl.experiments.ppo_measurements import mc_probe
from credit_rl.risk.longitudinal import LongitudinalPDModel


def run():
    output = ROOT/'policy_initialization'
    identity = json.loads((output/'preregistration.json').read_text())['identity']
    p = identity['protocol']['standard']
    config, settings, _ = settings_for('standard', 'configs')
    risk = LongitudinalPDModel.load(ROOT/'standard/models/pd/logistic_calibrated.joblib')
    data = joblib.load(output/'dataset_validation.joblib')
    labels = data['labels']
    rng = np.random.default_rng(p['population_seed']+77001)
    ids, coverage = [], []
    for scenario in ('baseline', 'severe_stress'):
        for action in range(5):
            available = np.flatnonzero(data['states'].scenario.eq(scenario).to_numpy() & (labels == action))
            chosen = sorted(rng.choice(available, min(4, len(available)), replace=False))
            ids.extend(chosen)
            coverage.append(dict(scenario=scenario, teacher_action=action, available_states=len(available),
                                 selected_states=len(chosen)))
    pd.DataFrame(coverage).to_csv(output/'supplemental_boundary_coverage.csv', index=False)
    data['states'].iloc[ids].assign(teacher_action=labels[ids]).to_csv(output/'supplemental_boundary_panel.csv', index=False)
    write_json(output/'supplemental_boundary_protocol.json', dict(
        reason='Original random MC panel contained only contraction teacher actions under stress; add rare-action coverage.',
        declared_after='Short-budget outcomes and initial panel coverage inspection; exploratory diagnostic supplement.',
        sampling='Up to four states per available teacher action per scenario, local fixed seed, validation only.',
        not_changed='Original MC bank, natural population estimands, initialization gate, PPO settings and checkpoint selection.',
        absent_actions='Report zero support; do not synthesize states or infer pressure for unavailable teacher actions.',
        repeated_budgets='Use short-run prefix only; separately verify identical long-run prefix weights.',
        draws=p['mc_draws'], states=[int(i) for i in ids]))
    pieces = []
    folder = output/'supplemental_boundary'
    folder.mkdir(exist_ok=True)
    for seed in p['seeds']:
        for arm in ('RandomInit', 'ImitationInit'):
            path = folder/f'{arm}_{seed}.csv'
            if path.exists():
                pieces.append(pd.read_csv(path))
                continue
            base = output/'runs'/'32768'/arm/str(seed)
            initial = PPO.load(base/'temporal_0.zip', device='cpu')
            initial_probs = distribution(initial, data['observations'][ids])[0]
            rows = []
            previous = initial
            previous_steps = 0
            for steps in (512, 1024, 2048, 4096, 8192):
                current = PPO.load(base/f'temporal_{steps}.zip', device='cpu')
                probabilities, logits, _, _ = distribution(current, data['observations'][ids])
                prior_logits = distribution(previous, data['observations'][ids])[1]
                for j, index in enumerate(ids):
                    teacher = int(labels[index])
                    probe = mc_probe(data['snapshots'][index], previous, config, risk, p['mc_draws'],
                        p['population_seed']+int(index)*1000+seed, .98, .95, settings['ppo']['reward_scale'])
                    rows.append(dict(seed=seed, experiment_id=arm, timesteps=steps, preceding_snapshot=previous_steps,
                        state_id=int(index), customer_id=data['states'].iloc[index].customer_id,
                        scenario=data['states'].iloc[index].scenario, teacher_action=teacher,
                        initially_agreed=int(initial_probs[j].argmax()) == teacher,
                        survived=int(probabilities[j].argmax()) == teacher,
                        preceding_teacher_gae=float(probe['gae'][:, teacher].mean()),
                        preceding_teacher_mc_advantage=float(probe['advantage'][:, teacher].mean()),
                        critic_prediction=probe['prediction'], mc_value=float(probe['value'].mean()),
                        critic_error=probe['prediction']-float(probe['value'].mean()),
                        contraction_logit_change=float(logits[j, 0]-prior_logits[j, 0]),
                        teacher_logit_change=float(logits[j, teacher]-prior_logits[j, teacher]),
                        scope='Supplemental action-balanced validation probes; not actual minibatch gradients'))
                previous, previous_steps = current, steps
            frame = pd.DataFrame(rows)
            frame.to_csv(path, index=False)
            pieces.append(frame)
            print(f'SUPPLEMENT {arm} {seed}', flush=True)
    pd.concat(pieces, ignore_index=True).to_csv(output/'supplemental_boundary_pressure.csv', index=False)


if __name__ == '__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run()
