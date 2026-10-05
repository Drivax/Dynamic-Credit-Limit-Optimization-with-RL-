"""Small, isolated software fixtures when frozen scientific models are unavailable.

These inputs exercise the same pipeline, but never stand in for standard results.
No historical directory is written and no fixture is called a verified experiment.
"""
from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import torch

from credit_rl import CreditLimitEnv
from credit_rl.evaluation.scenarios import make_scenarios
from credit_rl.evaluation.structural import DecisionSnapshot, hypothetical_paths, rollout
from credit_rl.experiments.main_evaluation import run, settings_for
from credit_rl.experiments.ppo_imitation import fit_actor
from credit_rl.policies.baselines import SB3Policy
from credit_rl.policies.information import QEstimator, choose
from credit_rl.risk.longitudinal import LongitudinalPDModel


def build_inputs(root):
    root = Path(root)
    if (root/'fixture_complete.json').exists():
        return
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=True)
    canonical = root/'standard'
    run('smoke', canonical, stage='train')
    config, settings, _ = settings_for('smoke', 'configs')
    risk = LongitudinalPDModel.load(canonical/'models/pd/logistic_calibrated.joblib')
    base = PPO.load(canonical/'models/ppo_101/selected.zip', device='cpu')
    datasets = {}
    for role in ('train', 'validation'):
        observations, snapshots, states = [], [], []
        for macro in ('baseline', 'severe_stress'):
            for customer in make_scenarios(config, settings, role, macro, customers=4):
                env = CreditLimitEnv(config=config, pd_model=risk, severe_delinquency_months=3)
                obs, _ = env.reset(seed=customer.customer_seed, options=customer.reset_options())
                while not env._done:
                    observations.append(obs.copy())
                    snapshots.append(DecisionSnapshot.capture(env))
                    states.append(dict(state_id=len(states), customer_id=customer.customer_id,
                        scenario=macro, role=role, weight=1.))
                    obs, *_ = env.step(2)
                env.close()
        datasets[role] = dict(observations=np.array(observations), snapshots=snapshots, states=pd.DataFrame(states))
    train = datasets['train']
    indices = np.linspace(0, len(train['snapshots'])-1, 16, dtype=int)
    q = []
    for i in indices:
        snapshot = train['snapshots'][i]
        length = min(2, 24-snapshot.elapsed)
        paths = hypothetical_paths(snapshot, 2, length, int(901+i*10))
        cube = rollout(snapshot, config, risk, SB3Policy(base), paths, length)
        q.append((cube[:, :, :, 6]*.98**np.arange(length)).sum(2).mean(0))
    teacher = QEstimator(leaves=3, iterations=3, minimum_leaf=2, seed=101).fit(
        train['observations'][indices], np.array(q))
    (root/'information_gap').mkdir(exist_ok=True)
    joblib.dump({'F0': dict(model=teacher, horizon=2, leaves=3)}, root/'information_gap/planners.joblib')
    droot = root/'policy_initialization'
    droot.mkdir(exist_ok=True)
    for role, data in datasets.items():
        data['q'] = teacher.predict(data['observations'])
        data['labels'] = np.array([choose(v, o, config) for v, o in zip(data['q'], data['observations'])])
        joblib.dump(data, droot/f'dataset_{role}.joblib')
    train, validation = datasets['train'], datasets['validation']
    actor_folder = droot/'imitation_models/Imitation_64x64/101'
    actor_folder.mkdir(parents=True, exist_ok=True)
    fit_actor(train['observations'], train['labels'], np.ones(len(train['labels'])),
        validation['observations'], validation['labels'], config, risk, [64, 64], 101,
        dict(learning_rate=.001, batch_size=32), dict(imitation_epochs=2, imitation_check_every=1), actor_folder)
    reference = root/'ppo_diagnostics/runs/canonical/101'
    reference.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(canonical/'models/ppo_101/checkpoint_0.zip', reference/'checkpoint_0.zip')
    for arm in ('RandomInit', 'ImitationInit'):
        folder = droot/f'runs/32768/{arm}/101'
        folder.mkdir(parents=True, exist_ok=True)
        # Names preserve loader compatibility only; metadata discloses the actual tiny budget.
        source = canonical/'models/ppo_101/selected.zip' if arm == 'RandomInit' else actor_folder/'selected.zip'
        for name in ('selected', 'final'):
            shutil.copyfile(source, folder/f'{name}.zip')
        (folder/'metadata.json').write_text(json.dumps(dict(validation_score=0., software_fixture=True,
            actual_timesteps=128 if arm == 'RandomInit' else 0)))
    (root/'fixture_complete.json').write_text(json.dumps(dict(scientific_evidence=False,
        purpose='Portable smoke integration; tiny newly fitted public teacher, not the frozen scientific teacher')))
