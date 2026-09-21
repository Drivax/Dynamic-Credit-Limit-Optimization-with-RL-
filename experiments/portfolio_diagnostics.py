"""Independent loss replications and whole-portfolio OPE support diagnostics."""
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from credit_rl.portfolio.environment import PortfolioEnv
from credit_rl.policies.decision import PDThreshold
from credit_rl.evaluation.ope import trajectory_weights, estimates, bootstrap_estimates
from experiments.portfolio_constraints import rollout
from experiments.study_common import write_json


def run_tail(profile,base,cfg,settings,risk,folder,selection):
    count=settings['profiles'][profile]['tail']
    if not count:return
    started=perf_counter();rows=[]
    specs=[('Static',-1),('RiskBased',-1),('Myopic',-1)]+[('PPO_hard',s) for s in settings['profiles'][profile]['seeds']]
    for policy,seed in specs:
        file=folder/f'tail_{policy}_{seed}.csv'
        if file.exists():rows.extend(pd.read_csv(file).to_dict('records'));continue
        agent=PPO.load(f'outputs/models/portfolio/{profile}/{policy}_{seed}.zip',device='cpu') if seed!=-1 else None
        results=[]
        for j in range(count):
            row,_,_=rollout(base,risk,cfg,settings['seed']+30000+j,policy,agent,namespace='PORTFOLIO_TAIL',diagnostics=False)
            results.append(dict(policy=policy,policy_seed=seed,portfolio=j,**row))
        pd.DataFrame(results).to_csv(file,index=False);rows.extend(results)
        print(f'Independent loss distribution: {policy} {seed}, {count} portfolios',flush=True)
    pd.DataFrame(rows).to_csv(folder/'tail_portfolios.csv',index=False)
    write_json(folder/'tail_runtime.json',dict(seconds=perf_counter()-started,portfolios_per_instance=count))


def run_ope(profile,base,cfg,settings,risk,folder,selection):
    started=perf_counter();count=8 if profile=='smoke' else 100 if profile=='standard' else 500
    rule=PDThreshold(base,low=.1,high=.5)
    agents={f'PPO_hard_{s}':PPO.load(f'outputs/models/portfolio/{profile}/PPO_hard_{s}.zip',device='cpu')
            for s in settings['profiles'][profile]['seeds']}
    targets=['Behavior','RiskBased','Static',*agents]
    rows=[];rng=np.random.default_rng(settings['seed']+441)
    for j in range(count):
        env=PortfolioEnv(base,risk,cfg,namespace='PORTFOLIO_OPE',fixed_seed=settings['seed']+40000+j)
        obs,_=env.reset();step=0
        while True:
            p=np.full(5,.04);p[rule.act(obs[:21])]+=.6;p[2]+=.2
            action=int(rng.choice(5,p=p));state=env.state
            target_actions={'RiskBased':rule.act(obs[:21]),'Static':2,
                **{name:int(agent.predict(obs,deterministic=True)[0]) for name,agent in agents.items()}}
            nxt,reward,terminated,truncated,info=env.step(action)
            rows.append(dict(portfolio=j,step=step,month=state.month,action=action,
                economic_value=reward*1000,behavior_probability=p[action],
                expected_loss=state.expected_loss,remaining_budget=state.remaining_risk_budget,
                risk_shortfall=state.risk_shortfall,risk_budget=state.risk_budget,
                effective_action=info['decision']['effective_action'],constraint_reason=info['decision']['constraint_reason'],
                terminated=terminated,truncated=truncated,
                **{f'observation_{i}':float(x) for i,x in enumerate(obs)},
                **{f'next_observation_{i}':float(x) for i,x in enumerate(nxt)},
                **{f'mu_{a}':p[a] for a in range(5)},
                **{f'pi_{name}':p[action] if name=='Behavior' else float(target_actions[name]==action) for name in targets},
                **{f'support_{name}':float(np.dot(p,p)) if name=='Behavior' else p[target_actions[name]] for name in targets}))
            obs=nxt;step+=1
            if terminated or truncated:break
        env.close()
    logs=pd.DataFrame(rows);logs.to_csv(folder/'portfolio_ope_logs.csv.gz',index=False)
    returns=logs.groupby('portfolio',sort=True).economic_value.sum().to_numpy();summary=[];weights=[]
    for target in targets:
        ids,w=trajectory_weights(logs.behavior_probability,logs[f'pi_{target}'],logs.portfolio)
        stats=estimates(returns,w);ci=bootstrap_estimates(returns,w,settings['bootstrap_repetitions'],settings['bootstrap_seed'])
        summary.append(dict(target=target,portfolios=count,decisions=len(logs),**stats,**ci,
            support_min=logs[f'support_{target}'].min(),support_p10=logs[f'support_{target}'].quantile(.1),
            low_support_fraction=(logs[f'support_{target}']<.1).mean(),unreliable=stats['ESS']<30))
        weights.extend(dict(target=target,portfolio=int(i),weight=float(v)) for i,v in zip(ids,w))
    pd.DataFrame(summary).to_csv(folder/'portfolio_ope_summary.csv',index=False)
    pd.DataFrame(weights).to_csv(folder/'portfolio_ope_weights.csv',index=False)
    write_json(folder/'ope_runtime.json',dict(seconds=perf_counter()-started,portfolios=count,decisions=len(logs),
        evaluation_unit='whole coupled portfolio',same_hard_transition_kernel=True))
    print(f'Portfolio OPE: {count} complete portfolios, {len(logs)} decisions',flush=True)
