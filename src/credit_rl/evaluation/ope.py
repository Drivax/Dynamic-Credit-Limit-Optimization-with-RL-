"""Episodic IS/WIS on public logged trajectories with known action propensities."""
import numpy as np


def validate_probabilities(probabilities):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-10):
        raise ValueError("Invalid action probabilities")
    return p


class MixturePolicy:
    """Randomizes requested commands, including commands projected by the environment."""
    def __init__(self, policies, weights, seed, action_count=5):
        self.policies,self.weights,self.action_count=policies,weights,action_count
        validate_probabilities(list(weights.values()))
        if set(weights)-set(policies)-{"uniform"}: raise ValueError("Unknown mixture component")
        self.rng=np.random.default_rng(seed)

    def probabilities(self, observation):
        p=np.full(self.action_count,self.weights.get("uniform",0.)/self.action_count)
        for name,policy in self.policies.items():
            if name in self.weights: p[policy.act(observation)]+=self.weights[name]
        return validate_probabilities(p)

    def act(self, observation):
        return int(self.rng.choice(self.action_count,p=self.probabilities(observation)))


class SoftTarget:
    def __init__(self, behavior, target, behavior_weight, seed):
        if not 0<=behavior_weight<=1: raise ValueError("Invalid mixture weight")
        self.behavior,self.target,self.weight=behavior,target,behavior_weight
        self.rng=np.random.default_rng(seed)

    def probabilities(self, observation):
        p=self.weight*self.behavior.probabilities(observation)
        p[self.target.act(observation)]+=1-self.weight
        return validate_probabilities(p)

    def act(self, observation):
        return int(self.rng.choice(len(self.probabilities(observation)),p=self.probabilities(observation)))


def trajectory_weights(behavior, target, episode_ids):
    b,t=np.asarray(behavior,float),np.asarray(target,float)
    if b.shape!=t.shape or b.shape!=np.asarray(episode_ids).shape or not np.isfinite(b).all() or not np.isfinite(t).all():
        raise ValueError("Invalid propensity arrays")
    if ((b<=0)|(b>1)).any() or ((t<0)|(t>1)).any(): raise ValueError("Invalid logged propensity")
    identities, inverse=np.unique(episode_ids,return_inverse=True)
    log_ratio=np.log(np.maximum(t,np.finfo(float).tiny))-np.log(b)
    totals=np.bincount(inverse,weights=log_ratio,minlength=len(identities))
    zeros=np.bincount(inverse,weights=(t==0),minlength=len(identities))>0
    if (totals[~zeros]>700).any(): raise FloatingPointError("Importance weight overflow; report log weights")
    weights=np.exp(np.minimum(totals,700)); weights[zeros]=0.
    return identities,weights


def estimates(returns, weights, clip=None):
    y,w=np.asarray(returns,float),np.asarray(weights,float)
    if y.shape!=w.shape or len(y)==0 or not np.isfinite(y).all() or not np.isfinite(w).all() or (w<0).any():
        raise ValueError("Invalid trajectory returns/weights")
    if clip is not None:
        if clip<=0: raise ValueError("Weight clip must be positive")
        w=np.minimum(w,clip)
    total=float(w.sum()); squares=float(np.square(w).sum())
    return dict(IS=float(np.mean(w*y)),WIS=float(np.dot(w,y)/total) if total else np.nan,
        ESS=total**2/squares if squares else 0.,mean_weight=float(w.mean()),
        max_weight=float(w.max()),weight_p99=float(np.quantile(w,.99)),
        nonzero_weights=int((w>0).sum()),zero_weight_fraction=float((w==0).mean()))


def bootstrap_estimates(returns, weights, repetitions, seed, clip=None):
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(repetitions):
        indices=rng.integers(len(returns),size=len(returns))
        samples.append(estimates(np.asarray(returns)[indices],np.asarray(weights)[indices],clip))
    result={}
    for name in ("IS","WIS"):
        vals=np.array([s[name] for s in samples]); vals=vals[np.isfinite(vals)]
        lo,hi=np.quantile(vals,[.025,.975]) if len(vals) else (np.nan,np.nan)
        result.update({name+"_lower":float(lo),name+"_upper":float(hi),name+"_valid_bootstraps":len(vals)})
    return result
