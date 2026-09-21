"""Train, select on validation, and evaluate synchronized held-out portfolios."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import hashlib
import json
from datetime import datetime, timezone
import subprocess
import numpy as np
import pandas as pd
import yaml
import torch
from threadpoolctl import threadpool_limits
from stable_baselines3 import PPO
from credit_rl.config import SimulationConfig
from credit_rl.risk.longitudinal import LongitudinalPDModel
from credit_rl.portfolio.accounting import PortfolioConfig
from credit_rl.portfolio.environment import PortfolioEnv
from credit_rl.portfolio.policies import plan_month
from credit_rl.evaluation.worlds import sample_evaluation_world
from experiments.common import write_manifest
from experiments.study_common import write_json


def inputs(profile):
    base=SimulationConfig.from_yaml('configs/simulation.yaml')
    cfg=PortfolioConfig(**yaml.safe_load(Path('configs/portfolio.yaml').read_text()))
    settings=yaml.safe_load(Path('configs/constrained_policy.yaml').read_text())
    cfg=replace(cfg,size=settings['profiles'][profile]['size'])
    model=LongitudinalPDModel.load('outputs/models/pd/logistic_calibrated.joblib')
    return base,cfg,settings,model


def identity(profile,base,cfg,settings):
    sources=list(Path('src/credit_rl').rglob('*.py'))+[Path('experiments/portfolio_constraints.py')]
    sources=[p for p in sources if 'reporting' not in p.name and 'statistics' not in p.name]
    payload=dict(profile=profile,base=asdict(base),portfolio=asdict(cfg),settings=settings,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        pd_sha256=hashlib.sha256(Path('outputs/models/pd/logistic_calibrated.joblib').read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload


def mode_config(cfg,mode,penalty=0):
    return replace(cfg,mode='hard' if mode=='PPO_hard' else 'soft' if mode=='PPO_penalty' else 'unconstrained',penalty=penalty)


def rollout(base,risk,cfg,seed,policy,agent=None,world=None,macro=None,sensor=None,namespace='PORTFOLIO_TEST',diagnostics=True):
    env=PortfolioEnv(base,risk,cfg,world_config=world,macro_spec=macro,sensor=sensor,
        namespace=namespace,fixed_seed=seed,record=True,diagnostics=diagnostics)
    obs,_=env.reset();last_month=-1;plan={};solver=[];steps=0;started=perf_counter()
    while True:
        if agent is not None: action=int(agent.predict(obs,deterministic=True)[0])
        else:
            if env.month!=last_month:
                plan,stats=plan_month(policy,env.public_month(),base,cfg);solver.append(stats);last_month=env.month
            action=int(plan[int(env.order[env.cursor])])
        obs,_,terminated,truncated,_=env.step(action);steps+=1
        if terminated or truncated: break
    elapsed=perf_counter()-started
    monthly=pd.DataFrame(env.months);actions=pd.DataFrame(env.decisions)
    if diagnostics:
        monthly=monthly.merge(pd.DataFrame(env.diagnostic_export()),on='month')
        monthly['true_violation']=monthly.true_expected_loss>monthly.risk_budget
        monthly['predicted_safe_true_violated']=(~monthly.el_violation)&monthly.true_violation
        monthly['predicted_violated_true_safe']=monthly.el_violation&(~monthly.true_violation)
        monthly['both_violated']=monthly.el_violation&monthly.true_violation
        monthly['both_safe']=(~monthly.el_violation)&(~monthly.true_violation)
    # All metrics use initial portfolio size or explicit active decision counts.
    metrics=dict(value=monthly.economic_value.sum()/cfg.size,revenue=monthly.revenue.sum()/cfg.size,
        loss=monthly.credit_loss.sum()/cfg.size,portfolio_loss=monthly.credit_loss.sum(),
        reward=monthly.objective.sum()/cfg.size,default_rate=monthly.defaults.sum()/cfg.size,
        delinquent_rate=monthly.delinquent.sum()/monthly.active_customers.sum(),
        mean_ead=monthly.total_ead.mean()/cfg.size,mean_el=monthly.expected_loss.mean()/cfg.size,
        budget_utilization=monthly.risk_budget_utilization.mean(),el_violation_rate=monthly.el_violation.mean(),
        any_violation_rate=monthly.any_violation.mean(),violation_magnitude=monthly.risk_shortfall.mean()/cfg.size,
        worst_violation=monthly.risk_shortfall.max()/cfg.size,
        inherited_violation_rate=(monthly.initial_shortfall>1e-7).mean(),
        top_decile_el_share=monthly.top_decile_el_share.mean(),high_risk_share=monthly.high_risk_share.mean(),
        increase_rate=(actions.effective_change>1e-8).mean(),decrease_rate=(actions.effective_change< -1e-8).mean(),
        rejection_rate=(actions.constraint_reason=='portfolio_budget').mean(),
        hard_nonworsening=bool(actions.hard_nonworsening.all()),
        value_per_ead=monthly.economic_value.sum()/max(monthly.total_ead.mean(),1e-8),
        decisions=steps,portfolio_months=len(monthly),seconds=elapsed,
        solver_exact_fraction=float(np.mean([s['exact'] for s in solver])) if solver else np.nan,
        distinct_increase_share=actions[actions.effective_change>1e-8].customer_index.nunique()/cfg.size)
    reversals=0
    for _,g in actions.groupby('customer_index'):
        signs=np.sign(g.effective_change.to_numpy()); signs=signs[abs(signs)>0]
        reversals+=int((signs[1:]!=signs[:-1]).sum())
    metrics['reversals_per_customer']=reversals/cfg.size
    metrics['limit_growth']=(monthly.iloc[-1].total_credit_limit/monthly.iloc[0].total_credit_limit-1)
    # Outstanding live-book limits shrink on defaults: this growth includes attrition.
    stress=monthly.macro_stress>0
    metrics['stress_loss']=monthly.loc[stress,'credit_loss'].sum()/cfg.size
    first_stress=np.flatnonzero(stress)
    metrics['ead_before_stress']=monthly.iloc[max(0,first_stress[0]-1)].total_ead/cfg.size if len(first_stress) else np.nan
    if diagnostics:
        for name in ('true_violation','predicted_safe_true_violated','predicted_violated_true_safe','both_violated','both_safe'):
            metrics[name+'_rate']=monthly[name].mean()
        metrics['true_el_mc_se']=monthly.true_el_mc_se.mean()/cfg.size
    env.close();return metrics,monthly,actions


def train(profile,base,cfg,settings,risk,folder,run_id):
    p=settings['profiles'][profile];models=Path('outputs/models/portfolio')/profile;models.mkdir(parents=True,exist_ok=True)
    times=[]
    def fit(name,seed,config,steps):
        path=models/f'{name}_{seed}.zip'
        if path.exists(): return PPO.load(path,device='cpu')
        env=PortfolioEnv(base,risk,config,namespace='PORTFOLIO_TRAIN',record=False)
        hp=settings['ppo'].copy();network=hp.pop('network')
        started=perf_counter();agent=PPO('MlpPolicy',env,seed=seed,verbose=0,device='cpu',
            policy_kwargs={'net_arch':dict(pi=network,vf=network)},**hp)
        agent.learn(steps);agent.save(path);env.close()
        times.append(dict(name=name,seed=seed,steps=steps,seconds=perf_counter()-started))
        print(f'Trained {name} {seed}: {steps} decisions',flush=True)
        return agent
    selection_file=folder/'selection.json'
    if not selection_file.exists():
        validation=[]
        for penalty in settings['penalties']:
            candidate=mode_config(cfg,'PPO_penalty',penalty)
            agent=fit(f'pilot_{penalty}',p['seeds'][0],candidate,p['pilot_steps'])
            for j in range(p['validation']):
                row,_,_=rollout(base,risk,candidate,settings['seed']+10000+j,'PPO_penalty',agent,
                    namespace='PORTFOLIO_VALIDATION',diagnostics=False)
                validation.append(dict(penalty=penalty,portfolio=j,**row))
        v=pd.DataFrame(validation);v.to_csv(folder/'penalty_validation.csv',index=False)
        ranks=v.groupby('penalty')[['el_violation_rate','value']].mean().reset_index().sort_values(['el_violation_rate','value'],ascending=[True,False])
        buffers=[]
        for factor in settings['safety_factors']:
            for j in range(p['validation']):
                row,_,_=rollout(base,risk,replace(cfg,safety_factor=factor),settings['seed']+10000+j,'RiskBased',
                    namespace='PORTFOLIO_VALIDATION',diagnostics=True)
                buffers.append(dict(safety_factor=factor,portfolio=j,**row))
        b=pd.DataFrame(buffers);b.to_csv(folder/'buffer_validation.csv',index=False)
        rank=b.groupby('safety_factor')[['true_violation_rate','value']].mean().reset_index().sort_values(['true_violation_rate','value'],ascending=[True,False])
        write_json(selection_file,dict(experiment_id=run_id,penalty=float(ranks.iloc[0].penalty),
            safety_factor=float(rank.iloc[0].safety_factor),criterion='minimum mean validation violation, then maximum economic value'))
    selection=json.loads(selection_file.read_text())
    for mode in ('PPO_unconstrained','PPO_penalty','PPO_hard'):
        for seed in p['seeds']:
            fit(mode,seed,mode_config(cfg,mode,selection['penalty']),p['train_steps'])
    if times: pd.DataFrame(times).to_csv(folder/'training_timing.csv',index=False)
    write_json(folder/'model_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in models.glob('*.zip')})
    return selection


def cases(profile,base,cfg,settings,selection):
    p=settings['profiles'][profile];normal={};stress=dict(onset=8,duration=10,recovery=4,intensity=1.5)
    result=[]
    def add(name,config=cfg,macro=normal,world=base,sensor=None,count=None,kind='diagnostic'):
        result.append(dict(name=name,config=config,macro=macro,world=world,sensor=sensor or {},count=count or p['test'],kind=kind))
    for budget,value in settings['budgets'].items():
        for macro_name,macro in [('normal',normal),('stress',stress)]:
            add(f'{macro_name}_{budget}',replace(cfg,el_budget_per_customer=value),macro,kind='main')
    if profile=='smoke': return result
    small=min(8,p['test'])
    for name,sensor in [('pd_under',{'intercept':-.7}),('pd_over',{'intercept':.7}),('pd_slope',{'slope':.65}),('pd_noise',{'noise':.8})]:
        add(name,sensor=sensor,count=small,kind='sensor')
    for factor in settings['safety_factors']:
        add(f'buffer_{factor}',replace(cfg,safety_factor=factor),stress,count=small,kind='buffer')
    add('constant_budget_stress',replace(cfg,dynamic_budget=False),stress,kind='budget_rule')
    for size in (8,24): add(f'size_{size}',replace(cfg,size=size),count=p['test'],kind='size')
    for order in ('fixed','reverse'):add(f'order_{order}',replace(cfg,order=order),count=small,kind='order')
    rs=yaml.safe_load(Path('configs/robustness.yaml').read_text())
    for j in range(p['worlds']):
        world=sample_evaluation_world(base,rs,settings['seed']+50000+j,f'portfolio_world_{j}')
        add(world.world_id,world=world.config,macro=world.macro_spec,count=small,kind='world')
    return result


def policy_specs(profile,cfg,settings,selection):
    specs=[(name,-1,replace(cfg,mode='hard',safety_factor=selection['safety_factor'] if name=='BufferedRiskBased' else cfg.safety_factor),None)
           for name in ('Static','Decrease20','RiskBased','Greedy','Myopic','BufferedRiskBased')]
    for mode in ('PPO_unconstrained','PPO_penalty','PPO_hard'):
        for seed in settings['profiles'][profile]['seeds']:
            specs.append((mode,seed,mode_config(cfg,mode,selection['penalty']),Path(f'outputs/models/portfolio/{profile}/{mode}_{seed}.zip')))
    return specs


def _worker_init():
    global _worker_risk
    torch.set_num_threads(1)
    _worker_risk=LongitudinalPDModel.load('outputs/models/pd/logistic_calibrated.joblib')


def _evaluate_job(task):
    base,case,name,seed,pc,path,folder,settings,run_id=task
    key=f"{case['name']}__{name}__{seed}";destination=folder/'jobs'/key
    if (destination/'done.json').exists():
        if json.loads((destination/'done.json').read_text())['experiment_id']!=run_id:
            raise ValueError('Mismatched completed job')
        return key+' reused'
    destination.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        agent=PPO.load(path,device='cpu') if path else None
        rows=[];monthly=[];actions=[]
        for j in range(case['count']):
            row,m,a=rollout(base,_worker_risk,pc,settings['seed']+20000+j,
                'RiskBased' if name=='BufferedRiskBased' else name,agent,case['world'],case['macro'],case['sensor'])
            meta=dict(case=case['name'],kind=case['kind'],policy=name,policy_seed=seed,portfolio=j,size=pc.size,
                budget=pc.el_budget_per_customer,safety_factor=pc.safety_factor,mode=pc.mode)
            rows.append(dict(**meta,**row));monthly.append(m.assign(**meta));actions.append(a.assign(**meta))
        pd.DataFrame(rows).to_csv(destination/'portfolios.csv',index=False)
        pd.concat(monthly).to_csv(destination/'monthly.csv.gz',index=False)
        pd.concat(actions).to_csv(destination/'actions.csv.gz',index=False)
        write_json(destination/'done.json',dict(experiment_id=run_id))
    return f"{key}: {np.mean([r['value'] for r in rows]):.1f} EUR/client"


def evaluate(profile,base,cfg,settings,risk,folder,selection,run_id):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_cases=cases(profile,base,cfg,settings,selection)
    write_json(folder/'cases.json',[dict(**{k:v for k,v in c.items() if k not in ('config','world')},config=asdict(c['config']),world=asdict(c['world'])) for c in all_cases])
    tasks=[(base,c,name,seed,pc,path,folder,settings,run_id) for c in all_cases
           for name,seed,pc,path in policy_specs(profile,c['config'],settings,selection)]
    started=perf_counter()
    if profile=='smoke':
        _worker_init()
        for task in tasks:print(_evaluate_job(task),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=3,initializer=_worker_init) as pool:
            for future in as_completed([pool.submit(_evaluate_job,task) for task in tasks]):
                print(future.result(),flush=True)
    write_json(folder/'evaluation_runtime.json',dict(seconds=perf_counter()-started,jobs=len(tasks),workers=1 if profile=='smoke' else 3))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['smoke','standard','full'],default='standard')
    parser.add_argument('--stage',choices=['train','evaluate','report','tail','ope','all'],default='all')
    args=parser.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        base,cfg,settings,risk=inputs(args.profile);run_id,payload=identity(args.profile,base,cfg,settings)
        folder=Path('outputs/results/portfolio')/args.profile;folder.mkdir(parents=True,exist_ok=True)
        registry=Path('outputs/experiments/portfolio')/args.profile;registry.mkdir(parents=True,exist_ok=True)
        file=registry/'manifest.json'
        if file.exists():
            if json.loads(file.read_text())['experiment_id']!=run_id:raise ValueError('Inputs changed; preserve completed run before rerunning')
        else:write_manifest(file,base,settings,experiment_id=run_id,timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),**payload)
        for config in ('portfolio','constrained_policy','simulation','robustness'):
            (registry/f'{config}.yaml').write_bytes(Path(f'configs/{config}.yaml').read_bytes())
        if args.stage in ('train','all'):selection=train(args.profile,base,cfg,settings,risk,folder,run_id)
        else:
            selection=json.loads((folder/'selection.json').read_text())
            for model,digest in json.loads((folder/'model_hashes.json').read_text()).items():
                if hashlib.sha256(Path(model).read_bytes()).hexdigest()!=digest:raise ValueError('Trained model changed')
        if args.stage in ('evaluate','all'):evaluate(args.profile,base,cfg,settings,risk,folder,selection,run_id)
        if args.stage in ('tail','ope','all'):
            from experiments.portfolio_diagnostics import run_tail, run_ope
            if args.stage in ('tail','all'):run_tail(args.profile,base,cfg,settings,risk,folder,selection)
            if args.stage in ('ope','all'):run_ope(args.profile,base,cfg,settings,risk,folder,selection)
        if args.stage in ('report','all'):
            from credit_rl.portfolio.reporting import report
            report(folder,settings,args.profile)


if __name__=='__main__':main()
