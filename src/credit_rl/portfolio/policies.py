"""Public-information portfolio baselines; frozen nominal economic surrogate."""
from dataclasses import replace
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from credit_rl.policies.decision import MyopicEconomic, PDThreshold, decode
from credit_rl.envs.constraints import effective_limit
from .accounting import ead, monthly_pd


def opportunity_tables(public, base, config):
    """One candidate per customer/command; no hidden state in inputs or outputs."""
    economic=replace(base,reward=replace(base.reward,capital_weight=0.,constraint_weight=0.))
    model=MyopicEconomic(economic)
    n=len(public['ids']); values=np.zeros((n,5)); limits=np.zeros((n,5))
    for i,obs in enumerate(public['observations']):
        values[i]=model.values(obs); s=decode(obs,base)
        for a,m in enumerate(base.environment.action_multipliers):
            limits[i,a]=effective_limit(public['limits'][i],s['delinquency'],m,base.environment,3)[0]
    exposure=ead(public['balance'][:,None],limits,config.ccf)
    probability=monthly_pd(np.minimum(1,public['pd']*config.safety_factor),config.pd_horizon)
    expected=probability[:,None]*config.lgd*exposure
    high=(public['pd']>config.high_risk_pd)[:,None]*exposure-config.high_risk_share*exposure
    return values,limits,np.stack([expected,exposure,high])


def solve_choices(values, resources, ceilings, *, time_limit=1.):
    """Multiple-choice linear integer program; return solver status and gap honestly."""
    n,k=values.shape
    if not n: return np.array([],int),dict(exact=True,status=0,gap=0.)
    valid=np.isfinite(values)
    objective=-np.where(valid,values,0).ravel()
    assignment=np.kron(np.eye(n),np.ones((1,k)))
    constraint=np.vstack([assignment,resources.reshape(len(ceilings),-1)])
    result=milp(objective,integrality=np.ones(n*k),bounds=Bounds(np.zeros(n*k),valid.ravel().astype(float)),
        constraints=LinearConstraint(constraint,np.r_[np.ones(n),np.full(len(ceilings),-np.inf)],np.r_[np.ones(n),ceilings]),
        options={'time_limit':time_limit,'mip_rel_gap':0.})
    if result.x is None: return None,dict(exact=False,status=int(result.status),gap=None)
    choice=result.x.reshape(n,k).argmax(axis=1)
    if not valid[np.arange(n),choice].all() or (resources[:,np.arange(n),choice].sum(axis=1)>ceilings+1e-5).any():
        return None,dict(exact=False,status=int(result.status),gap=None)
    return choice,dict(exact=result.status==0,status=int(result.status),gap=float(result.mip_gap))


def plan_month(name, public, base, config):
    ids=public['ids']; n=len(ids); actions=np.full(n,2,dtype=int)
    if name=='Static': return dict(zip(ids,actions)),dict(exact=True)
    if name=='Decrease20': return dict(zip(ids,np.zeros(n,dtype=int))),dict(exact=True)
    if name=='RiskBased':
        rule=PDThreshold(base,low=.1,high=.5)
        return dict(zip(ids,[rule.act(x) for x in public['observations']])),dict(exact=True)
    values,limits,resources=opportunity_tables(public,base,config)
    state=public['state']; ceilings=np.maximum([state.risk_budget,state.exposure_budget,0.],resources[:,:,2].sum(axis=1))
    # Joint proposals use relaxed ceilings only for inherited breaches. The shared
    # sequential admission layer can reject proposals before capacity is released.
    if name=='Myopic':
        # Replan each month; exact proposal does not imply exact executed allocation.
        chosen,stats=solve_choices(values,resources,ceilings)
        if chosen is not None: return dict(zip(ids,chosen)),stats
    # Interpretable greedy: best positive gain / incremental EL, feasible one at a time.
    stats=dict(exact=False,status='greedy' if name=='Greedy' else 'solver_fallback',gap=None)
    candidates=[]
    for i in range(n):
        for a in range(5):
            gain=values[i,a]-values[i,2]; risk=resources[0,i,a]-resources[0,i,2]
            if np.isfinite(gain) and gain>1e-8:
                candidates.append((gain/max(risk,1e-6),gain,i,a))
    used=set(); total=resources[:,:,2].sum(axis=1)
    for _,gain,i,a in sorted(candidates,reverse=True):
        if i in used: continue
        candidate=total+resources[:,i,a]-resources[:,i,2]
        if (candidate<=ceilings+1e-7).all():
            actions[i]=a; total=candidate; used.add(i)
    return dict(zip(ids,actions)),stats
