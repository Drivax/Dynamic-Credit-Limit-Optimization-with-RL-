"""Paired held-out worlds, public behavior and independent counterfactual banks."""
from copy import deepcopy
from dataclasses import replace
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from credit_rl import CreditLimitEnv
from credit_rl.evaluation.policy_engine import PolicySpec, evaluate_policy
from credit_rl.evaluation.scenarios import make_scenarios
from credit_rl.evaluation.structural import DecisionSnapshot, hypothetical_paths, rollout
from credit_rl.evaluation.worlds import perturb
from credit_rl.experiments.information_analysis import effective_action
from credit_rl.experiments.ppo_measurements import observable_regret, policy_measurements
from credit_rl.policies.baselines import SB3Policy
from credit_rl.policies.information import ObservationPlanner, choose
from credit_rl.policies.registry import baseline_specs
from credit_rl.simulation.shocks import ShockPath
from .final_evaluation import context, output_for, input_paths
from .policy_initialization import write_json
from .main_evaluation import digest

POLICIES = ('Static', 'PDThreshold', 'MyopicEconomic', 'AlwaysDecrease20',
            'CanonicalPPO', 'ObservationPlanner', 'BCInitPPO', 'BCRegularizedPPO')


def worlds(config, settings, protocol, p):
    local = deepcopy(settings)
    local['population'].update(seed=p['population_seed'], test=p['customers'])
    groups = {}
    for macro in ('baseline', 'severe_stress'):
        group = []
        for customer in make_scenarios(config, local, 'test', macro):
            identity = f'E_{p["population_seed"]}_{customer.customer_id}'
            group.append(replace(customer, customer_id=identity,
                initial_state=replace(customer.initial_state, customer_id=identity),
                shock_path=ShockPath.generate(identity, p['population_seed'], config.environment.horizon)))
        groups[macro] = group
    cohort = groups['baseline']
    rng = np.random.default_rng(p['population_seed']+490)
    donor_indices = np.argsort([s.traits.creditworthiness for s in cohort])[:max(1, len(cohort)//4)]
    recipients = set(rng.choice(len(cohort), max(1, len(cohort)//4), replace=False))
    shifted = [replace(s, traits=cohort[int(rng.choice(donor_indices))].traits) if i in recipients else s
               for i, s in enumerate(cohort)]
    return dict(nominal=(config, cohort), severe_stress=(config, groups['severe_stress']),
        population_shift=(config, shifted),
        behavioral_shift=(perturb(config, protocol['worlds']['behavioral_shift']), cohort),
        risk_shift=(perturb(config, protocol['worlds']['risk_shift']), cohort))


def specs_for(config, settings, output, budget, seed, profile):
    root, droot = input_paths(profile)
    simple = {s.name: replace(s, seed=seed) for s in baseline_specs(config, settings, include_oracle=False)
              if s.name in POLICIES}
    teacher = joblib.load(root/'information_gap/planners.joblib')['F0']['model']
    simple['ObservationPlanner'] = PolicySpec('ObservationPlanner',
        lambda env, scenario: ObservationPlanner(teacher, config), seed=seed,
        model_identifier=str(root/'information_gap/planners.joblib'))
    schedule = json.loads((output/'selection.json').read_text())['schedule']
    old_budget = budget if profile == 'standard' else 32768
    paths = dict(CanonicalPPO=droot/f'runs/{old_budget}/RandomInit/{seed}/selected.zip',
        BCInitPPO=droot/f'runs/{old_budget}/ImitationInit/{seed}/selected.zip',
        BCRegularizedPPO=output/f'runs/{budget}/{schedule}/{seed}/selected.zip')
    for name, path in paths.items():
        model = PPO.load(path, device='cpu')
        simple[name] = PolicySpec(name, lambda env, s, m=model: SB3Policy(m), seed=seed,
                                 model_identifier=str(path))
    return [simple[name] for name in POLICIES]


def visit_frame(events, teacher, config):
    observations = np.asarray([e.pop('observation') for e in events])
    for e in events:
        e.pop('next_observation')
    frame = pd.DataFrame(events)
    for i in range(21):
        frame[f'o{i}'] = observations[:, i]
    q = teacher.predict(observations)
    targets = np.array([choose(v, o, config) for v, o in zip(q, observations)])
    frame['teacher_action'] = targets
    frame['teacher_regret'] = observable_regret(q, frame.action.to_numpy(), observations, config)
    frame['agreement'] = targets == frame.action.to_numpy()
    return frame


def evaluate(profile, budgets=None, diagnostics=True):
    config, settings, protocol, p, risk = context(profile)
    output = output_for(profile)
    root, droot = input_paths(profile)
    selected_hash = digest(output/'selection.json')
    all_worlds = worlds(config, settings, protocol, p)
    write_json(output/'worlds.json', {name: dict(parameters=protocol['worlds'][name],
        customers=[s.customer_id for s in cohort], population_seed=p['population_seed'])
        for name, (_, cohort) in all_worlds.items()})
    teacher = joblib.load(root/'information_gap/planners.joblib')['F0']['model']
    for budget in p['budgets'] if budgets is None else budgets:
        for world, (shifted_config, cohort) in all_worlds.items():
            for seed in p['seeds']:
                # Policies retain nominal configuration under model shift.
                for spec in specs_for(config, settings, output, budget, seed, profile):
                    folder = output/f'evaluation/{budget}/{world}/{spec.name}/{seed}'
                    folder.mkdir(parents=True, exist_ok=True)
                    if (folder/'complete.json').exists():
                        stored = json.loads((folder/'complete.json').read_text())
                        if stored['selection_sha256'] != selected_hash:
                            raise ValueError('Evaluation predates frozen selection')
                        if digest(folder/'episodes.csv') != stored['episodes_sha256'] or digest(folder/'visits.csv.gz') != stored['visits_sha256']:
                            raise ValueError('Cached evaluation changed')
                        continue
                    reference = output/f'evaluation/{p["budgets"][0]}/{world}/{spec.name}/{p["seeds"][0]}'
                    if spec.name in ('Static', 'PDThreshold', 'MyopicEconomic', 'AlwaysDecrease20', 'ObservationPlanner') and reference != folder and (reference/'complete.json').exists():
                        pd.read_csv(reference/'episodes.csv').assign(policy_seed=seed).to_csv(folder/'episodes.csv', index=False)
                        shutil.copyfile(reference/'visits.csv.gz', folder/'visits.csv.gz')
                        write_json(folder/'complete.json', dict(selection_sha256=selected_hash,
                            episodes_sha256=digest(folder/'episodes.csv'), visits_sha256=digest(folder/'visits.csv.gz'),
                            deterministic_replication_of=str(reference)))
                        continue
                    events = []
                    episodes, _, _ = evaluate_policy(spec, cohort, shifted_config, risk, settings,
                        world, keep_history=False, transition_observer=events.append)
                    visits = visit_frame(events, teacher, config)
                    episodes.to_csv(folder/'episodes.csv', index=False)
                    visits.to_csv(folder/'visits.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
                    write_json(folder/'complete.json', dict(selection_sha256=selected_hash,
                        episodes_sha256=digest(folder/'episodes.csv'), visits_sha256=digest(folder/'visits.csv.gz')))
                    print(f'FINAL {budget} {world} {spec.name} {seed} NEV={episodes.net_economic_value.mean():.2f}', flush=True)
    if diagnostics:
        preservation(profile)
        counterfactual(profile)
        from .final_ope import run_ope
        run_ope(profile)


def preservation(profile):
    config, _, _, p, _ = context(profile)
    output = output_for(profile)
    root, droot = input_paths(profile)
    if (output/'preservation_visits.csv.gz').exists():
        return
    data = joblib.load(droot/'dataset_validation.joblib')
    teacher = joblib.load(root/'information_gap/planners.joblib')['F0']['model']
    schedule = json.loads((output/'selection.json').read_text())['schedule']
    frames = []
    for seed in p['seeds']:
        for policy in ('CanonicalPPO', 'BCInitPPO', 'BCRegularizedPPO'):
            for steps in [0, *p['budgets']]:
                if steps == 0:
                    path = root/f'ppo_diagnostics/runs/canonical/{seed}/checkpoint_0.zip' if policy == 'CanonicalPPO' else droot/f'imitation_models/Imitation_64x64/{seed}/selected.zip'
                elif policy == 'BCRegularizedPPO':
                    path = output/f'runs/{steps}/{schedule}/{seed}/final.zip'
                else:
                    arm = 'RandomInit' if policy == 'CanonicalPPO' else 'ImitationInit'
                    old_steps = steps if profile == 'standard' else 32768
                    path = droot/f'runs/{old_steps}/{arm}/{seed}/final.zip'
                model = PPO.load(path, device='cpu')
                frame, _ = policy_measurements(model, teacher, data, config, policy, seed)
                frame['timesteps'] = steps
                frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(output/'preservation_visits.csv.gz', index=False,
                                              compression={'method': 'gzip', 'mtime': 0})


def counterfactual(profile):
    config, settings, protocol, p, risk = context(profile)
    output = output_for(profile)
    root, droot = input_paths(profile)
    path = output/'counterfactual_bank.joblib'
    teacher = joblib.load(root/'information_gap/planners.joblib')['F0']['model']
    continuation = SB3Policy(PPO.load(root/'standard/models/ppo_101/selected.zip', device='cpu'))
    if path.exists():
        bank = joblib.load(path)
    else:
        all_worlds = worlds(config, settings, protocol, p)
        rng = np.random.default_rng(p['population_seed']+601)
        bank = []
        for world in ('nominal', 'severe_stress'):
            _, cohort = all_worlds[world]
            specs = specs_for(config, settings, output, p['budgets'][0], p['seeds'][0], profile)
            sources = [s for s in specs if s.name in ('CanonicalPPO', 'ObservationPlanner', 'MyopicEconomic')]
            visits = []
            for i, customer in enumerate(cohort):
                env = CreditLimitEnv(config=config, pd_model=risk,
                    severe_delinquency_months=settings['guardrails']['severe_delinquency_months'])
                actor = sources[i % 3].factory(env, customer)
                obs, _ = env.reset(seed=customer.customer_seed, options=customer.reset_options())
                while not env._done:
                    visits.append((DecisionSnapshot.capture(env), obs.copy()))
                    obs, *_ = env.step(actor.act(obs))
                env.close()
            for index in rng.choice(len(visits), min(len(visits), p['mc_states_per_scenario']), replace=False):
                snapshot, obs = visits[index]
                state_id = len(bank)
                length = min(12, config.environment.horizon-snapshot.elapsed)
                paths = hypothetical_paths(snapshot, 2*p['mc_draws'], length, p['population_seed']+10000+state_id*100)
                cube = rollout(snapshot, config, risk, continuation, paths, length)
                q = (cube[:, :, :, 6]*settings['ppo']['gamma']**np.arange(length)).sum(2)
                bank.append(dict(world=world, state_id=state_id, snapshot=snapshot, observation=obs,
                    q_selection=q[:p['mc_draws']].mean(0), q_evaluation=q[p['mc_draws']:].mean(0),
                    q_draws=q, teacher_action=choose(teacher.predict(obs[None])[0], obs, config)))
                print(f'COUNTERFACTUAL {world} state={state_id}', flush=True)
        joblib.dump(bank, path)
    rows = []
    for budget in p['budgets']:
        for seed in p['seeds']:
            specs = specs_for(config, settings, output, budget, seed, profile)
            for entry in bank:
                q, obs, snapshot = entry['q_evaluation'], entry['observation'], entry['snapshot']
                best = choose(entry['q_selection'], obs, config)
                target = entry['teacher_action']
                for spec in specs:
                    action = spec.factory(type('RiskHolder', (), {'pd_model': risk})(), None).act(obs)
                    deviation = effective_action(snapshot, action, config) != effective_action(snapshot, target, config)
                    gain = float(q[action]-q[target])
                    rows.append(dict(policy=spec.name, policy_seed=seed, budget=budget, world=entry['world'],
                        state_id=entry['state_id'], customer_id=snapshot.state.customer_id,
                        action=action, teacher_action=target, selected_best_action=best,
                        regret=float(q[best]-q[action]), teacher_deviation_gain=gain,
                        beneficial=bool(deviation and gain > 0), harmful=bool(deviation and gain < 0),
                        deviation=deviation, pd_bucket=int(np.searchsorted([.2, .6], obs[10])),
                        utilization_bucket=int(np.searchsorted([1/3, .5], obs[3])),
                        horizon_bucket=int(np.searchsorted([1/3, 2/3], obs[0]))))
    pd.DataFrame(rows).to_csv(output/'counterfactual.csv', index=False)
