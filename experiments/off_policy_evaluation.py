"""Known-propensity logged episodes versus independent simulator Monte Carlo."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits

from credit_rl import CreditLimitEnv
from credit_rl.evaluation.policy_engine import PolicySpec, evaluate_policy
from credit_rl.evaluation.worlds import make_cohort
from credit_rl.evaluation.ope import MixturePolicy, SoftTarget, trajectory_weights, estimates, bootstrap_estimates
from .study_common import load_frozen, register, write_json


def build_targets(base,benchmark,risk,specs,settings):
    selected={s.name:s for s in specs if s.name in ("Static","PDThreshold","MyopicEconomic")}
    def behavior(env,scenario):
        components={name:s.factory(env,scenario) for name,s in selected.items()}
        return MixturePolicy(components,settings["behavior_weights"],scenario.customer_seed+9917)
    targets=[s for s in specs if s.name in ("Static","AlwaysDecrease20","PDThreshold","UtilizationPD","MyopicEconomic","PPO","PPO_without_PD")]
    targets += [PolicySpec("Behavior",behavior),PolicySpec("SoftPDThreshold",
        lambda env,scenario: SoftTarget(behavior(env,scenario),selected["PDThreshold"].factory(env,scenario),
            settings["soft_target_behavior_weight"],scenario.customer_seed+817))]
    return behavior,targets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=["smoke","standard","full"],default="standard")
    parser.add_argument("--stage",choices=["evaluate","report","all"],default="all")
    args=parser.parse_args()
    settings=yaml.safe_load(Path("configs/ope.yaml").read_text()); profile=settings["profiles"][args.profile]
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        base,benchmark,risk,specs=load_frozen(profile["ppo_seeds"])
        behavior_factory,targets=build_targets(base,benchmark,risk,specs,settings)
        identity,registry=register("ope",args.profile,settings,base,benchmark,targets,dict(targets=[(s.name,s.seed) for s in targets]))
        output=Path("outputs/results/ope")/args.profile; output.mkdir(parents=True,exist_ok=True)
        if args.stage in ("evaluate","all"):
            started=perf_counter()
            mc=make_cohort(base,count=profile["mc_customers"],seed=settings["seed"]+100000,namespace="OPE_MC")
            mc_rows=[]; timing=[]
            for spec in targets:
                file=output/f"mc_{spec.name}_{spec.seed}.csv"
                if file.exists(): episodes=pd.read_csv(file)
                else:
                    episodes,_,t=evaluate_policy(spec,mc,base,risk,benchmark,"nominal_mc",keep_history=False)
                    episodes.to_csv(file,index=False); timing.append(dict(stage="mc",policy=spec.name,seed=spec.seed,**t))
                for metric in ("net_economic_value","cumulative_reward"):
                    values=episodes[metric].to_numpy(); rng=np.random.default_rng(settings["bootstrap_seed"])
                    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(settings["bootstrap_repetitions"])])
                    mc_rows.append(dict(policy=spec.name,policy_seed=spec.seed,metric=metric,true_MC_value=float(values.mean()),
                        MC_standard_error=float(values.std(ddof=1)/np.sqrt(len(values))),MC_lower=float(np.quantile(means,.025)),MC_upper=float(np.quantile(means,.975)),customers=len(values)))
                print(f"MC {spec.name} {spec.seed}: {episodes.net_economic_value.mean():.1f}",flush=True)
            pd.DataFrame(mc_rows).to_csv(output/"mc_summary.csv",index=False)
            estimates_rows=[]; support_rows=[]; weight_rows=[]
            for replicate in range(profile["replicates"]):
                cohort=make_cohort(base,count=profile["logged_customers"],seed=settings["seed"]+replicate,
                    namespace=f"OPE_LOG_{replicate}")
                dummy=CreditLimitEnv(config=base,pd_model=risk)
                behavior=behavior_factory(dummy,cohort[0])
                file=output/f"logged_{replicate}.csv.gz"
                if file.exists(): logged=pd.read_csv(file)
                else:
                    rows=[]
                    def record(event):
                        obs=event.pop("observation"); nxt=event.pop("next_observation"); probabilities=behavior.probabilities(obs)
                        rows.append(dict(**event,behavior_action_probability=float(probabilities[event["action"]]),
                            **{f"observation_{i}":float(v) for i,v in enumerate(obs)},
                            **{f"next_observation_{i}":float(v) for i,v in enumerate(nxt)},
                            **{f"behavior_probability_{i}":float(v) for i,v in enumerate(probabilities)}))
                    _,_,t=evaluate_policy(PolicySpec("Behavior",behavior_factory),cohort,base,risk,benchmark,
                        f"logged_{replicate}",keep_history=False,transition_observer=record)
                    logged=pd.DataFrame(rows); logged.to_csv(file,index=False)
                    timing.append(dict(stage="logging",policy="Behavior",seed=replicate,**t))
                observations=logged[[f"observation_{i}" for i in range(21)]].to_numpy(dtype=np.float32)
                mu=logged[[f"behavior_probability_{i}" for i in range(5)]].to_numpy()
                if not np.allclose(mu.sum(axis=1),1) or not (mu>0).all(): raise ValueError("Invalid behavior support")
                actions=logged.action.to_numpy(int); index=np.arange(len(logged))
                np.testing.assert_allclose(mu[index,actions],logged.behavior_action_probability)
                returns=logged.groupby("customer_id",sort=True)[["reward","economic_value"]].sum()
                # gamma=1 makes OPE estimands agree exactly with these cumulative economic KPIs.
                if settings["discount"]!=1: raise ValueError("This experiment reports undiscounted cumulative value")
                for spec in targets:
                    policy=spec.factory(dummy,cohort[0]); supplied=observations.copy()
                    if spec.without_pd: supplied[:,10]=0
                    if spec.name=="Behavior": evaluation=mu.copy()
                    elif spec.name=="SoftPDThreshold":
                        evaluation=np.stack([policy.probabilities(obs) for obs in supplied])
                    else:
                        choices=policy.model.predict(supplied,deterministic=True)[0] if hasattr(policy,"model") else np.array([policy.act(obs) for obs in supplied])
                        evaluation=np.eye(5)[choices.astype(int)]
                    identities,weights=trajectory_weights(mu[index,actions],evaluation[index,actions],logged.customer_id.to_numpy())
                    if not np.array_equal(identities,returns.index): raise ValueError("Return/weight ordering mismatch")
                    expected_support=(evaluation*mu).sum(axis=1)
                    support_rows.append(dict(policy=spec.name,policy_seed=spec.seed,replicate=replicate,
                        support_min=float(expected_support.min()),support_p10=float(np.quantile(expected_support,.1)),
                        support_median=float(np.median(expected_support)),
                        target_mass_below_threshold=float((evaluation*(mu<settings["support_threshold"])).sum(axis=1).mean()),
                        unsupported_target_mass=float((evaluation*(mu==0)).sum(axis=1).mean()),decisions=len(logged)))
                    weight_rows.extend(dict(policy=spec.name,policy_seed=spec.seed,replicate=replicate,customer_id=c,weight=float(w)) for c,w in zip(identities,weights))
                    for metric,column in (("net_economic_value","economic_value"),("cumulative_reward","reward")):
                        values=returns[column].to_numpy()
                        for clip in (None,settings["weight_clip"]):
                            stats=estimates(values,weights,clip)
                            intervals=bootstrap_estimates(values,weights,settings["bootstrap_repetitions"],settings["bootstrap_seed"]+replicate,clip)
                            estimates_rows.append(dict(policy=spec.name,policy_seed=spec.seed,replicate=replicate,metric=metric,
                                clipping="none" if clip is None else str(clip),**stats,**intervals,
                                unreliable=stats["ESS"]<settings["minimum_reliable_ess"],episodes=len(values)))
                dummy.close()
                print(f"Logged replicate {replicate}: {len(cohort)} episodes, {len(logged)} decisions",flush=True)
            table=pd.DataFrame(estimates_rows).merge(pd.DataFrame(mc_rows),on=["policy","policy_seed","metric"],validate="many_to_one")
            for estimator in ("IS","WIS"):
                table[estimator+"_error"]=table[estimator]-table.true_MC_value
                table[estimator+"_covers_MC_point"]=(table[estimator+"_lower"]<=table.true_MC_value)&(table[estimator+"_upper"]>=table.true_MC_value)
            table.to_csv(output/"estimates.csv",index=False)
            pd.DataFrame(support_rows).to_csv(output/"support.csv",index=False)
            pd.DataFrame(weight_rows).to_csv(output/"weights.csv.gz",index=False)
            if timing: pd.DataFrame(timing).to_csv(output/"timing.csv",index=False)
            write_json(output/"runtime.json",dict(wall_seconds=perf_counter()-started,experiment_id=identity))
        if args.stage in ("report","all"):
            from credit_rl.evaluation.ope_reporting import report
            report(output,settings)
        print(f"Experiment {identity}; registry {registry}",flush=True)


if __name__=="__main__": main()
