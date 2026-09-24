"""Frozen nominal policies under declared DGP, macro and actor-information shifts."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import os

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import pandas as pd
import torch
import yaml
from threadpoolctl import threadpool_limits
from credit_rl.evaluation.worlds import build_worlds, make_cohort, in_world
from credit_rl.evaluation.sensors import ObservationSensor
from credit_rl.evaluation.policy_engine import evaluate_policy
from .study_common import load_frozen, register, write_json

_CONTEXT=None


def initialize(settings, profile):
    global _CONTEXT
    torch.set_num_threads(1)
    limits=threadpool_limits(limits=1)
    base,benchmark,risk,specs=load_frozen(profile["ppo_seeds"])
    cohort=make_cohort(base,count=profile["customers"],seed=settings["seed"],namespace="ROBUST_TEST")
    _CONTEXT=(base,benchmark,risk,specs,cohort,settings,limits)


def evaluate_job(job, output, identity):
    world,signal_name=job
    base,benchmark,risk,specs,cohort,settings,_=_CONTEXT
    key=f"{world.world_id}__{signal_name}"; folder=Path(output)/"jobs"/key
    folder.mkdir(parents=True,exist_ok=True)
    done=folder/"done.json"
    if done.exists():
        if json.loads(done.read_text())["experiment_id"]!=identity: raise ValueError("Cached job mismatch")
        return key,"cached"
    started=perf_counter(); episodes=[]; timings=[]; monthly=[]; actions=[]
    scenarios=in_world(cohort,world)
    for spec in specs:
        sensor_settings=settings["signals"][signal_name]
        sensor=lambda scenario: ObservationSensor(scenario.customer_id,settings["sensor_seed"],**sensor_settings)
        current=replace(spec,observation_transform_factory=sensor)
        e,h,t=evaluate_policy(current,scenarios,world.config,risk,benchmark,key)
        tags=dict(world_id=world.world_id,world_kind=world.kind,signal=signal_name)
        e=e.assign(**tags); episodes.append(e)
        timings.append(dict(policy=spec.name,policy_seed=spec.seed,**tags,**t))
        transitions=h[h.month>0]
        m=transitions.groupby("month").agg(active_customers=("customer_id","nunique"),
            value=("reward_interest_income","sum"),fees=("reward_fee_income","sum"),
            losses=("reward_credit_loss","sum"),funding=("reward_funding_cost","sum"),
            mean_limit=("credit_limit","mean"),mean_balance=("balance","mean"),
            mean_pd=("decision_pd","mean"),mean_actor_pd=("actor_pd","mean"),
            effective_change=("effective_action","mean"),requested_change=("requested_action","mean"))
        monthly.append(m.reset_index().assign(policy=spec.name,policy_seed=spec.seed,**tags))
        for label, values in (("pd",pd.cut(transitions.decision_pd,[0,.2,.5,.8,1],include_lowest=True)),
                              ("macro",transitions.opening_macro),
                              ("delinquency",transitions.opening_delinquency.clip(upper=3))):
            for value,group in transitions.groupby(values,observed=True):
                actions.append(dict(policy=spec.name,policy_seed=spec.seed,**tags,segment=label,bucket=str(value),
                    n=len(group),increase=float((group.effective_action>1e-8).mean()),
                    decrease=float((group.effective_action< -1e-8).mean()),mean_requested=float(group.requested_action.mean())))
        # Full paired trajectories are retained for the nominal and surprise-shock diagnostics.
        if world.world_id in ("nominal","macro_unexpected") and signal_name=="identity":
            h.to_csv(folder/f"{spec.name}__{spec.seed}.csv.gz",index=False)
    pd.concat(episodes,ignore_index=True).to_csv(folder/"episodes.csv.gz",index=False)
    pd.concat(monthly,ignore_index=True).to_csv(folder/"monthly.csv.gz",index=False)
    pd.DataFrame(actions).to_csv(folder/"actions.csv.gz",index=False)
    pd.DataFrame(timings).to_csv(folder/"timing.csv",index=False)
    elapsed=perf_counter()-started
    write_json(done,dict(experiment_id=identity,seconds=elapsed,world=world.metadata(),signal=signal_name))
    return key,round(elapsed,1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=["smoke","standard","full"],default="standard")
    parser.add_argument("--config",type=Path,default=Path("configs/robustness.yaml"))
    parser.add_argument("--stage",choices=["evaluate","report","all"],default="all")
    args=parser.parse_args()
    settings=yaml.safe_load(args.config.read_text()); profile=settings["profiles"][args.profile]
    base,benchmark,_,specs=load_frozen(profile["ppo_seeds"])
    worlds=build_worlds(base,settings,profile)
    jobs=[(world,"identity") for world in worlds]
    anchors=[w for w in worlds if w.world_id in ("nominal","random_000","macro_unexpected")]
    if profile["sensitivity"]:
        jobs += [(w,name) for w in anchors for name in settings["signals"] if name!="identity"]
    identity,registry=register("robustness",args.profile,settings,base,benchmark,specs,
        dict(worlds=[w.metadata() for w in worlds],jobs=[(w.world_id,s) for w,s in jobs]))
    output=Path("outputs/results/robustness")/args.profile; output.mkdir(parents=True,exist_ok=True)
    write_json(output/"worlds.json",[w.metadata() for w in worlds])
    if args.stage in ("evaluate","all"):
        started=perf_counter()
        if profile["workers"]==1:
            initialize(settings,profile)
            for job in jobs: print(evaluate_job(job,output,identity),flush=True)
        else:
            with ProcessPoolExecutor(max_workers=profile["workers"],initializer=initialize,initargs=(settings,profile)) as pool:
                futures=[pool.submit(evaluate_job,job,output,identity) for job in jobs]
                for future in as_completed(futures): print(future.result(),flush=True)
        write_json(output/"runtime.json",dict(wall_seconds=perf_counter()-started,jobs=len(jobs),workers=profile["workers"]))
    if args.stage in ("report","all"):
        from credit_rl.evaluation.robustness_reporting import report
        report(output,settings,benchmark,worlds)
    print(f"Experiment {identity}; registry {registry}",flush=True)


if __name__=="__main__": main()
