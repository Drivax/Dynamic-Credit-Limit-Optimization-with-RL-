"""Explicit factory registry; no policy-name branches in the evaluator."""
from credit_rl.evaluation.policy_engine import PolicySpec
from .decision import ConstantAdjustment, RandomPolicy, PDThreshold, UtilizationPD, MyopicEconomic
from .baselines import SB3Policy
from .oracle import OracleMyopic


def baseline_specs(config, settings, thresholds=None, include_oracle=True):
    severe = settings["guardrails"]["severe_delinquency_months"]
    low, high = thresholds or (settings["heuristics"]["low_pd"], settings["heuristics"]["high_pd"])
    utilization = settings["heuristics"]["utilization_threshold"]
    specs = [PolicySpec(name, lambda env, s, m=mult: ConstantAdjustment(config, m, severe))
             for name, mult in (("Static", 1.), ("AlwaysDecrease", .9), ("AlwaysDecrease20", .8), ("AlwaysIncrease", 1.1))]
    specs += [PolicySpec("Random", lambda env, s: RandomPolicy(config, s.customer_seed+991, severe), seed=991),
        PolicySpec("PDThreshold", lambda env, s: PDThreshold(config, low, high, severe)),
        PolicySpec("UtilizationPD", lambda env, s: UtilizationPD(config, low, high, severe, utilization)),
        PolicySpec("MyopicEconomic", lambda env, s: MyopicEconomic(config,
            int(env.pd_model.metadata["horizon_months"]), severe))]
    if include_oracle:
        specs.append(PolicySpec("SIMULATOR_ONLY_ORACLE_Myopic", lambda env, s: OracleMyopic(env, settings["evaluation"]["oracle_draws"]),
                                information_set="SIMULATOR_ONLY_ORACLE"))
    return specs


def ppo_spec(model, seed, path, without_pd=False):
    return PolicySpec("PPO_without_PD" if without_pd else "PPO", lambda env, s: SB3Policy(model),
                      seed=seed, without_pd=without_pd, model_identifier=str(path))
