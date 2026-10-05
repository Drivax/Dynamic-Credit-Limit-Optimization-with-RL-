"""Compact known-propensity overlap experiment using existing IS/WIS estimators."""
import numpy as np
import pandas as pd

from credit_rl.evaluation.ope import validate_probabilities, trajectory_weights, estimates, bootstrap_estimates
from credit_rl.evaluation.policy_engine import PolicySpec, evaluate_policy
from credit_rl.evaluation.worlds import make_cohort
from .final_evaluation import context, output_for
from .final_panels import specs_for


class OverlapBehavior:
    def __init__(self, target, probability, seed):
        if not 0 < probability < 1:
            raise ValueError('Overlap must have full support')
        self.target, self.probability = target, probability
        self.rng = np.random.default_rng(seed)

    def probabilities(self, observation):
        result = np.full(5, (1-self.probability)/4)
        result[self.target.act(observation)] = self.probability
        return validate_probabilities(result)

    def act(self, observation):
        return int(self.rng.choice(5, p=self.probabilities(observation)))


def run_ope(profile):
    config, settings, protocol, p, risk = context(profile)
    output = output_for(profile)
    folder = output/'ope'
    folder.mkdir(exist_ok=True)
    rows = []
    budget = p['budgets'][0]
    for seed in p['seeds']:
        targets = [s for s in specs_for(config, settings, output, budget, seed, profile)
                   if s.name in protocol['ope']['targets']]
        for target in targets:
            truth = pd.read_csv(output/f'evaluation/{budget}/nominal/{target.name}/{seed}/episodes.csv')
            value = truth.net_economic_value.mean()
            for overlap, probability in protocol['ope']['target_action_probability'].items():
                for replicate in range(p['ope_replicates']):
                    # Deterministic contraction has no seed dependence; do not duplicate simulation.
                    log_seed = seed if target.name == 'BCRegularizedPPO' else p['seeds'][0]
                    prefix = folder/f'{target.name}_{log_seed}_{overlap}_{replicate}'
                    file = prefix.with_suffix('.csv.gz')
                    cohort = make_cohort(config, count=p['ope_customers'],
                        seed=p['population_seed']+200000+replicate,
                        namespace=f'E_OPE_{profile}_{replicate}')
                    target_actor = target.factory(type('RiskHolder', (), {'pd_model': risk})(), cohort[0])
                    if not file.exists():
                        events = []
                        def factory(env, scenario):
                            return OverlapBehavior(target.factory(env, scenario), probability,
                                                   scenario.customer_seed+7801)
                        behavior = OverlapBehavior(target_actor, probability, 0)
                        def record(event):
                            obs = event.pop('observation')
                            event.pop('next_observation')
                            action = event['action']
                            probs = behavior.probabilities(obs)
                            events.append(dict(**event, behavior_probability=float(probs[action]),
                                target_probability=float(action == target_actor.act(obs)),
                                **{f'o{i}': float(v) for i, v in enumerate(obs)},
                                **{f'mu{i}': float(v) for i, v in enumerate(probs)}))
                        evaluate_policy(PolicySpec('Behavior', factory), cohort, config, risk, settings,
                                        'nominal_logging', keep_history=False, transition_observer=record)
                        pd.DataFrame(events).to_csv(file, index=False, compression={'method': 'gzip', 'mtime': 0})
                    logged = pd.read_csv(file)
                    identities, weights = trajectory_weights(logged.behavior_probability,
                        logged.target_probability, logged.customer_id)
                    returns = logged.groupby('customer_id', sort=True).economic_value.sum()
                    if not np.array_equal(identities, returns.index):
                        raise ValueError('OPE customer ordering mismatch')
                    point = estimates(returns.to_numpy(), weights)
                    ci = bootstrap_estimates(returns.to_numpy(), weights, p['bootstrap_repetitions'], 1064919)
                    for estimator in ('IS', 'WIS'):
                        estimate = point[estimator]
                        rows.append(dict(policy=target.name, policy_seed=seed, overlap=overlap,
                            target_action_probability=probability, replicate=replicate, estimator=estimator,
                            estimate=estimate, truth=value,
                            truth_se=truth.net_economic_value.std(ddof=1)/np.sqrt(len(truth)),
                            bias=estimate-value, squared_error=(estimate-value)**2,
                            lower=ci[estimator+'_lower'], upper=ci[estimator+'_upper'],
                            covered=bool(ci[estimator+'_lower'] <= value <= ci[estimator+'_upper']),
                            valid=np.isfinite(estimate), ESS=point['ESS'],
                            zero_weight_fraction=point['zero_weight_fraction'],
                            valid_bootstraps=ci[estimator+'_valid_bootstraps'],
                            logged_customers=len(cohort), mc_customers=len(truth)))
                    print(f'OPE {target.name} {seed} {overlap} replicate={replicate} ESS={point["ESS"]:.1f}', flush=True)
    pd.DataFrame(rows).to_csv(output/'ope_estimates.csv', index=False)
