"""Phase D: frozen, validation-gated actor initialization experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
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
from credit_rl.envs.observation import OBSERVATION_NAMES
from credit_rl.evaluation.scenarios import make_scenarios, assert_disjoint
from credit_rl.evaluation.structural import DecisionSnapshot
from credit_rl.experiments.main_evaluation import settings_for, digest
from credit_rl.experiments.ppo_diagnostics import protection
from credit_rl.experiments.ppo_imitation import fit_actor
from credit_rl.experiments.ppo_measurements import policy_measurements
from credit_rl.experiments.information_ppo import distribution
from credit_rl.policies.baselines import SB3Policy
from credit_rl.policies.decision import MyopicEconomic
from credit_rl.policies.information import ObservationPlanner, choose
from credit_rl.risk.longitudinal import LongitudinalPDModel
from credit_rl.simulation.shocks import ShockPath


ROOT = Path('outputs/main')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def protected_files():
    hashes = protection(ROOT/'standard', ROOT/'information_gap')
    paths = list((ROOT/'standard').rglob('*')) + list((ROOT/'ppo_diagnostics').rglob('*'))
    paths += list(Path('docs').glob('ppo_*.md'))
    paths += [Path('configs/ppo_diagnostics.yaml'), Path('configs/information_gap.yaml')]
    for folder in ('risk', 'envs', 'policies'):
        paths += list(Path('src/credit_rl', folder).glob('*.py'))
    hashes.update({str(p): digest(p) for p in paths if p.is_file()})
    return hashes


def verify_protection(output):
    expected = json.loads((output/'protected_artifacts.json').read_text())
    actual = protected_files()
    if actual != expected:
        changed = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise AssertionError(f'Protected artifacts changed: {changed[:10]}')
    supplement = output/'protected_smoke_artifacts.json'
    if supplement.exists():
        extra = json.loads(supplement.read_text())['sha256']
        for name, sha in extra.items():
            if not Path(name).exists() or digest(Path(name)) != sha:
                raise AssertionError(f'Protected smoke artifact changed: {name}')
        return len(set(expected) | set(extra))
    return len(expected)


def prepare(profile, output):
    output = Path(output)
    allowed = ROOT/('policy_initialization' if profile == 'standard' else 'policy_initialization_smoke')
    if output.resolve() != allowed.resolve():
        raise ValueError('Use the dedicated Phase D output directory')
    output.mkdir(parents=True, exist_ok=True)
    protocol = yaml.safe_load(Path('configs/policy_initialization.yaml').read_text())
    config, settings, _ = settings_for(profile, 'configs')
    identity = dict(profile=profile, protocol=protocol,
        protocol_sha256=digest(Path('configs/policy_initialization.yaml')),
        teacher_sha256=digest(ROOT/'information_gap/planners.joblib'),
        ppo_config=settings['ppo'])
    path = output/'preregistration.json'
    if path.exists():
        if json.loads(path.read_text())['identity'] != identity:
            raise ValueError('Frozen Phase D protocol changed')
    else:
        audit = json.loads((ROOT/'ppo_diagnostics/verification.json').read_text())
        if audit['status'] != 'passed':
            raise ValueError('Phase C verification did not pass')
        write_json(output/'phase_c_audit.json', dict(verification=audit,
            decision_gate=json.loads((ROOT/'ppo_diagnostics/decision_gate.json').read_text()),
            audited_at=datetime.now(timezone.utc).isoformat()))
        write_json(output/'protected_artifacts.json', protected_files())
        write_json(output/'historical_prefixes.json', {
            str(p): dict(size=p.stat().st_size, sha256=digest(p))
            for p in (Path('README.md'), Path('docs/technical_paper.md'))})
        write_json(path, dict(identity=identity, frozen_at=datetime.now(timezone.utc).isoformat()))
    risk = LongitudinalPDModel.load(ROOT/'standard/models/pd/logistic_calibrated.joblib')
    return config, settings, protocol, risk


def populations(config, settings, protocol):
    settings = deepcopy(settings)
    settings['population']['seed'] = protocol['population_seed']
    result = {}
    for role in ('train', 'validation', 'test'):
        settings['population'][role] = protocol[role+'_customers']
    for macro in ('baseline', 'severe_stress'):
        for role in ('train', 'validation', 'test'):
            group = []
            for s in make_scenarios(config, settings, role, macro):
                identity = f'D_{protocol["population_seed"]}_{s.customer_id}'
                group.append(replace(s, customer_id=identity,
                    initial_state=replace(s.initial_state, customer_id=identity),
                    shock_path=ShockPath.generate(identity, protocol['population_seed'], config.environment.horizon)))
            result[macro, role] = group
        assert_disjoint(*(result[macro, role] for role in ('train', 'validation', 'test')))
    return result


def build_dataset(config, settings, p, risk, output):
    """Store splits separately: fitting never opens test observations or labels."""
    groups = populations(config, settings, p)
    teacher = joblib.load(ROOT/'information_gap/planners.joblib')['F0']['model']
    canonical = PPO.load(ROOT/'standard/models/ppo_101/selected.zip', device='cpu')
    for role in ('train', 'validation', 'test'):
        path = output/f'dataset_{role}.joblib'
        if path.exists():
            continue
        observations, snapshots, rows = [], [], []
        for macro in ('baseline', 'severe_stress'):
            for i, customer in enumerate(groups[macro, role]):
                actor = [SB3Policy(canonical), ObservationPlanner(teacher, config), MyopicEconomic(config)][i % 3]
                env = CreditLimitEnv(config=config, pd_model=risk, record_history=False,
                    severe_delinquency_months=settings['guardrails']['severe_delinquency_months'])
                obs, _ = env.reset(seed=customer.customer_seed, options=customer.reset_options())
                while not env._done:
                    observations.append(obs.copy())
                    snapshots.append(DecisionSnapshot.capture(env))
                    rows.append(dict(state_id=len(rows), customer_id=customer.customer_id,
                        scenario=macro, role=role, source_policy=['PPO', 'F0', 'MyopicEconomic'][i % 3],
                        elapsed=env._elapsed, weight=1.0))
                    obs, *_ = env.step(actor.act(obs))
                env.close()
            print(f'DATA {role} {macro}: {len(rows)} visits', flush=True)
        x = np.asarray(observations, dtype=np.float32)
        q = teacher.predict(x)
        labels = np.array([choose(v, o, config) for v, o in zip(q, x)])
        data = dict(observations=x, states=pd.DataFrame(rows), snapshots=snapshots,
                    labels=labels, q=q, feature_names=OBSERVATION_NAMES)
        joblib.dump(data, path)
        data['states'].to_csv(output/f'dataset_{role}_manifest.csv', index=False)
        pd.DataFrame(x, columns=OBSERVATION_NAMES).describe().to_csv(output/f'coverage_{role}.csv')
    write_json(output/'dataset_hashes.json', {role: digest(output/f'dataset_{role}.joblib')
        for role in ('train', 'validation', 'test')})


def paired_visit_interval(frame, column, repetitions, seed=901):
    """Seed and customer cluster resampling; retain natural visit weighting."""
    sums = frame.pivot_table(index='seed', columns='customer_id', values=column, aggfunc='sum')
    counts = frame.pivot_table(index='seed', columns='customer_id', values=column, aggfunc='count')
    if sums.isna().any().any():
        raise ValueError('Unpaired seed/customer panel')
    values, weights = sums.to_numpy(), counts.to_numpy()
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        a = rng.integers(values.shape[0], size=values.shape[0])
        b = rng.integers(values.shape[1], size=values.shape[1])
        samples.append(values[np.ix_(a, b)].sum()/weights[np.ix_(a, b)].sum())
    low, high = np.quantile(samples, [.025, .975])
    return dict(mean=float(values.sum()/weights.sum()), low=float(low), high=float(high))


def quality(model, teacher, data, config, name, seed):
    frame, _ = policy_measurements(model, teacher, data, config, name, seed)
    rows = []
    for scenario, group in frame.groupby('scenario'):
        counts = group.action.value_counts(normalize=True)
        effective = group.effective_action.value_counts(normalize=True)
        rows.append(dict(experiment_id=name, seed=seed, scenario=scenario,
            requested_accuracy=group.agreement.mean(), effective_accuracy=group.effective_agreement.mean(),
            teacher_regret=group.teacher_regret.mean(), diversity=1-counts.max(),
            effective_actions=int((effective >= .01).sum()), entropy=group.entropy.mean(),
            **{f'action_share_{a}': counts.get(a, 0.) for a in range(5)}))
    return frame, pd.DataFrame(rows)


def qualification(frame, summary, protocol, p):
    rows, passing = [], True
    for scenario, group in frame.groupby('scenario'):
        agreement = paired_visit_interval(group, 'agreement_difference', p['bootstrap_repetitions'])
        regret = paired_visit_interval(group, 'regret_difference', p['bootstrap_repetitions'])
        metrics = summary[summary.scenario.eq(scenario)]
        diverse = int(((metrics.diversity >= protocol['minimum_diversity']) &
                       (metrics.effective_actions >= protocol['minimum_effective_actions'])).sum())
        passed = bool(agreement['low'] > protocol['effective_agreement_difference_lower_ci'] and
                      regret['high'] < protocol['regret_difference_upper_ci'] and
                      diverse >= min(protocol['required_seeds'], len(p['seeds'])))
        rows.append(dict(scenario=scenario, agreement_difference=agreement,
                         regret_difference=regret, diverse_seeds=diverse, passed=passed))
        passing &= passed
    return dict(qualified=passing, scenarios=rows, split='validation')


def imitate(config, settings, protocol, risk, output, profile):
    p = protocol[profile]
    train = joblib.load(output/'dataset_train.joblib')
    val = joblib.load(output/'dataset_validation.joblib')
    teacher = joblib.load(ROOT/'information_gap/planners.joblib')['F0']['model']
    controls = {}
    for seed in p['seeds']:
        model = PPO.load(ROOT/f'ppo_diagnostics/runs/canonical/{seed}/selected.zip', device='cpu')
        controls[seed] = quality(model, teacher, val, config, 'CanonicalSelected', seed)[0]
    decisions, all_metrics = [], []
    for architecture in protocol['imitation']['architectures']:
        name = 'Imitation_'+'x'.join(map(str, architecture))
        frames, metrics = [], []
        for seed in p['seeds']:
            folder = output/'imitation_models'/name/str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            if (folder/'selection.json').exists():
                model = PPO.load(folder/'selected.zip', device='cpu')
            else:
                model = fit_actor(train['observations'], train['labels'], np.ones(len(train['labels'])),
                    val['observations'], val['labels'], config, risk, architecture, seed,
                    protocol['imitation'], p, folder)
            frame, metric = quality(model, teacher, val, config, name, seed)
            frame['agreement_difference'] = frame.effective_agreement.astype(float)-controls[seed].effective_agreement.astype(float)
            frame['regret_difference'] = frame.teacher_regret-controls[seed].teacher_regret
            frame.to_csv(folder/'validation_states.csv', index=False)
            frames.append(frame)
            metrics.append(metric)
            print(f'IMITATION {name} {seed}: validation measured', flush=True)
        summary = pd.concat(metrics, ignore_index=True)
        all_metrics.append(summary)
        decision = qualification(pd.concat(frames, ignore_index=True), summary, protocol['qualification'], p)
        decision.update(architecture=architecture, checkpoint_hashes={str(seed): digest(
            output/'imitation_models'/name/str(seed)/'selected.zip') for seed in p['seeds']})
        decisions.append(decision)
        write_json(output/f'qualification_{name}.json', decision)
        pd.concat(all_metrics, ignore_index=True).assign(split='validation').to_csv(output/'imitation_quality.csv', index=False)
        if decision['qualified']:
            break
    selected = next((d['architecture'] for d in decisions if d['qualified']), None)
    result = dict(selected_architecture=selected, decisions=decisions,
                  status='qualified' if selected else 'qualification_failed',
                  test_used=False, frozen_at=datetime.now(timezone.utc).isoformat())
    write_json(output/'initialization_selection.json', result)
    return result


def transfer_actor(source, target, observations):
    """Actor-only tensor transfer, with exact public-output and critic invariants."""
    critic = {k: v.clone() for k, v in target.policy.state_dict().items()
              if k.startswith(('mlp_extractor.value_net.', 'value_net.'))}
    for name in ('policy_net',):
        getattr(target.policy.mlp_extractor, name).load_state_dict(
            getattr(source.policy.mlp_extractor, name).state_dict())
    target.policy.action_net.load_state_dict(source.policy.action_net.state_dict())
    a, b = distribution(source, observations), distribution(target, observations)
    for index in (0, 1):
        np.testing.assert_allclose(a[index], b[index], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(a[0].argmax(1), b[0].argmax(1))
    if any(not torch.equal(v, target.policy.state_dict()[k]) for k, v in critic.items()):
        raise AssertionError('Actor transfer modified critic')
    return dict(max_probability_error=float(np.max(abs(a[0]-b[0]))),
                max_logit_error=float(np.max(abs(a[1]-b[1]))), critic_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    parser.add_argument('--stage', choices=['audit', 'data', 'imitate', 'train'], default='audit')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    output = ROOT/('policy_initialization' if args.profile == 'standard' else 'policy_initialization_smoke')
    config, settings, protocol, risk = prepare(args.profile, output)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        if args.stage in ('data', 'imitate'):
            build_dataset(config, settings, protocol[args.profile], risk, output)
        if args.stage == 'imitate':
            imitate(config, settings, protocol, risk, output, args.profile)
        if args.stage == 'train':
            from credit_rl.experiments.initialization_learning import run_pair
            selection = json.loads((output/'initialization_selection.json').read_text())
            architecture = selection['selected_architecture']
            if architecture is None and args.profile == 'standard':
                print('Qualification failed: scientific PPO comparison not run', flush=True)
            else:
                architecture = architecture or [64, 64]
                for budget in protocol[args.profile]['budgets']:
                    with ProcessPoolExecutor(max_workers=args.workers) as pool:
                        futures = [pool.submit(run_pair, config, settings, protocol, risk, output,
                            args.profile, budget, seed, architecture,
                            smoke_unqualified=selection['selected_architecture'] is None)
                            for seed in protocol[args.profile]['seeds']]
                        for future in futures:
                            future.result()
                    from credit_rl.experiments.initialization_analysis import analyze_budget
                    analyze_budget(config, settings, protocol, risk, output, args.profile, budget)
    print(f'Protected files unchanged: {verify_protection(output)}', flush=True)


if __name__ == '__main__':
    main()
