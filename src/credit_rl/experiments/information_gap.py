"""Phase B diagnostic: independent targets, validation-only selection, paired evaluation."""
import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits
import torch
import yaml

from credit_rl import CreditLimitEnv
from credit_rl.evaluation.scenarios import make_scenarios, assert_disjoint
from credit_rl.evaluation.structural import DecisionSnapshot, hypothetical_paths, rollout
from credit_rl.experiments.main_evaluation import settings_for, digest
from credit_rl.experiments.structural_diagnostics import describe
from credit_rl.policies.baselines import SB3Policy
from credit_rl.policies.decision import ConstantAdjustment, MyopicEconomic, RandomPolicy
from credit_rl.policies.information import (
    FEATURE_SETS, public_features, privileged_features, QEstimator,
)
from credit_rl.risk.longitudinal import LongitudinalPDModel
from credit_rl.simulation.shocks import ShockPath


def cohorts(config, settings, protocol):
    settings = deepcopy(settings)
    settings['population']['seed'] = protocol['population_seed']
    for role in ('train', 'validation', 'test'):
        settings['population'][role] = protocol[role+'_customers']
    result = {}
    for macro in ('baseline', 'severe_stress'):
        for role in ('train', 'validation', 'test'):
            group = []
            for s in make_scenarios(config, settings, role, macro):
                identity = 'B_'+s.customer_id
                group.append(replace(s, customer_id=identity,
                    initial_state=replace(s.initial_state, customer_id=identity),
                    shock_path=ShockPath.generate(identity, protocol['population_seed'], config.environment.horizon)))
            result[macro, role] = group
        assert_disjoint(*(result[macro, r] for r in ('train', 'validation', 'test')))
    return result


def protected_hashes(canonical):
    paths = list(canonical.glob('models/**/*'))
    paths += list(Path('outputs/main/structural_diagnostics').rglob('*'))
    paths += [Path('docs/structural_diagnosis.md'), Path('docs/structural_audit.md')]
    return {str(p): digest(p) for p in paths if p.is_file()}


def validate_destination(output, canonical):
    for protected in (canonical, Path('outputs/main/structural_diagnostics')):
        if output.resolve() == protected.resolve() or protected.resolve() in output.resolve().parents:
            raise ValueError('Output must be separate from protected artifacts')


def scientific_sources():
    paths = [Path('src/credit_rl/policies/information.py'), Path('src/credit_rl/evaluation/structural.py'),
             Path('src/credit_rl/reward.py'), Path('src/credit_rl/config.py'),
             Path('src/credit_rl/experiments/information_gap.py')]
    for directory in ('simulation', 'risk', 'envs'):
        paths.extend(Path('src/credit_rl', directory).glob('*.py'))
    paths.extend(Path('configs').glob('*.yaml'))
    return {str(p): digest(p) for p in sorted(paths)}


def dataset(config, settings, protocol, risk, ppo, groups, output):
    cache = output/'dataset.joblib'
    if cache.exists():
        return joblib.load(cache)
    snapshots, prefixes, rows = [], [], []
    rng = np.random.default_rng(protocol['diagnostic_seed'])
    for (macro, role), group in groups.items():
        for i, scenario in enumerate(group):
            policies = [ConstantAdjustment(config), MyopicEconomic(config),
                        RandomPolicy(config, scenario.customer_seed), SB3Policy(ppo)]
            source = ['Static', 'MyopicEconomic', 'Random', 'PPO'][i % 4]
            actor = policies[i % 4]
            env = CreditLimitEnv(config=config, pd_model=risk, record_history=False,
                severe_delinquency_months=settings['guardrails']['severe_delinquency_months'])
            obs, _ = env.reset(seed=scenario.customer_seed, options=scenario.reset_options())
            visits, observations, actions = [], [], []
            while not env._done:
                observations.append(obs.copy())
                visits.append((DecisionSnapshot.capture(env), np.array(observations), list(actions)))
                action = actor.act(obs)
                actions.append(action)
                obs, *_ = env.step(action)
            n = min(len(visits), protocol['states_per_trajectory'])
            for index in sorted(rng.choice(len(visits), n, replace=False)):
                snap, observations, actions = visits[index]
                snapshots.append(snap)
                prefixes.append((observations, actions))
                rows.append(dict(**describe(snap, macro, source), role=role,
                    weight=len(visits)/n, state_id=len(rows)))
            env.close()
        print(f'Collected {role} {macro}: {len(rows)} states', flush=True)
    states = pd.DataFrame(rows)
    features, names = {}, {}
    for kind in FEATURE_SETS:
        values = [privileged_features(o[-1], s) if kind == 'Full' else public_features(o, a, kind)
                  for s, (o, a) in zip(snapshots, prefixes)]
        features[kind], names[kind] = np.array([v for v, _ in values]), values[0][1]
    result = dict(snapshots=snapshots, prefixes=prefixes, states=states, features=features, names=names)
    joblib.dump(result, cache)
    states.to_csv(output/'states.csv', index=False)
    (output/'feature_names.json').write_text(json.dumps(names, indent=2))
    return result


def targets(data, config, protocol, risk, ppo, output):
    folder = output/'draws'
    folder.mkdir(exist_ok=True)
    result = []
    for sid, snapshot in enumerate(data['snapshots']):
        path = folder/f'{sid:05d}.npz'
        if path.exists():
            q = np.load(path)['q']
        else:
            paths = hypothetical_paths(snapshot, protocol['draws'], config.environment.horizon,
                protocol['diagnostic_seed']+10000+sid*1000)
            cube = rollout(snapshot, config, risk, SB3Policy(ppo), paths, config.environment.horizon)
            reward = cube[:, :, :, 6]
            q = np.stack([(reward[:, :, :h]*.98**np.arange(min(h, reward.shape[2]))).sum(2)
                          for h in protocol['horizons']], axis=1)
            np.savez_compressed(path, q=q)
        result.append(q)
        if sid % 25 == 0:
            print(f'MC {sid+1}/{len(data["snapshots"])}', flush=True)
    return np.array(result)


def relative_error(prediction, truth, weights):
    error = (prediction-prediction[:, 2, None])-(truth-truth[:, 2, None])
    return float(np.average(np.mean(error[:, [0, 1, 3, 4]]**2, axis=1), weights=weights))


def fit_planners(data, q, protocol, output):
    states = data['states']
    train, validation = (states.role.eq(r).to_numpy() for r in ('train', 'validation'))
    models, records = {}, []
    means = q.mean(1)
    for kind in FEATURE_SETS:
        x = data['features'][kind]
        candidates = []
        for h, horizon in enumerate(protocol['horizons']):
            for leaves in protocol['tree_leaves']:
                model = QEstimator(leaves, protocol['boosting_iterations'], protocol['minimum_leaf'],
                    protocol['diagnostic_seed']).fit(x[train], means[train, h], states.weight[train])
                error = relative_error(model.predict(x[validation]), means[validation, h], states.weight[validation])
                records.append(dict(feature_set=kind, horizon=horizon, leaves=leaves,
                    validation_relative_mse=error))
                candidates.append((error, horizon, leaves, model))
        _, horizon, leaves, model = min(candidates, key=lambda c: c[:3])
        models[kind] = dict(model=model, horizon=horizon, leaves=leaves)
        print(f'Selected {kind}: H={horizon}, leaves={leaves}', flush=True)
    pd.DataFrame(records).to_csv(output/'model_selection.csv', index=False)
    joblib.dump(models, output/'planners.joblib')
    return models


def run(args):
    output = args.output
    canonical = args.canonical
    # Refuse any destination capable of overwriting a protected experiment.
    validate_destination(output, canonical)
    output.mkdir(parents=True, exist_ok=True)
    config, settings, _ = settings_for(args.canonical_profile, Path('configs'))
    protocol = yaml.safe_load(Path('configs/information_gap.yaml').read_text())[args.profile]
    risk_path = canonical/'models/pd/logistic_calibrated.joblib'
    risk = LongitudinalPDModel.load(risk_path)
    ppos = {seed: PPO.load(canonical/f'models/ppo_{seed}/selected.zip', device='cpu')
            for seed in settings['ppo']['seeds']}
    identity = dict(protocol=protocol, canonical=str(canonical.resolve()), canonical_profile=args.canonical_profile,
        models={str(p): digest(p) for p in canonical.glob('models/**/selected.zip')}, pd=digest(risk_path),
        sources=scientific_sources())
    manifest = output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Cached experiment identity mismatch; choose a fresh output directory')
    manifest.write_text(json.dumps(identity, indent=2))
    before = protected_hashes(canonical)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        groups = cohorts(config, settings, protocol)
        data = dataset(config, settings, protocol, risk, ppos[101], groups, output)
        q = targets(data, config, protocol, risk, ppos[101], output)
        if args.stage != 'data':
            models = joblib.load(output/'planners.joblib') if args.stage == 'evaluate' else fit_planners(data, q, protocol, output)
            from credit_rl.experiments.information_analysis import analyze
            analyze(data, q, models, groups, config, settings, protocol, risk, ppos, output)
    after = protected_hashes(canonical)
    if before != after:
        raise RuntimeError('Protected artifact changed during Phase B')
    (output/'protected_artifacts.json').write_text(json.dumps(dict(unchanged=True, sha256=after), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    parser.add_argument('--canonical-profile', choices=['standard', 'smoke'], default='standard')
    parser.add_argument('--canonical', type=Path, default=Path('outputs/main/standard'))
    parser.add_argument('--output', type=Path, default=Path('outputs/main/information_gap'))
    parser.add_argument('--stage', choices=['data', 'all', 'evaluate'], default='all')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
