"""Crossed world/customer/seed uncertainty; never resample customer-month rows."""
import numpy as np
import pandas as pd
from .policy_engine import aggregate_episodes

RISK_FIELDS={"default_rate":"max_default_rate","credit_loss_rate":"max_credit_loss_rate",
             "high_risk_exposure_share":"max_high_risk_exposure_share"}


def constraint_violations(frame, constraints):
    rows=[]
    for policy,group in frame.groupby("policy"):
        for metric,threshold_key in RISK_FIELDS.items():
            threshold=constraints[threshold_key]; delta=np.maximum(0,group[metric].to_numpy()-threshold)
            rows.append(dict(policy=policy,metric=metric,threshold=threshold,worlds=len(group),
                violation_frequency=float((delta>0).mean()),mean_excess=float(delta.mean()),
                mean_excess_when_violated=float(delta[delta>0].mean()) if (delta>0).any() else 0.,worst_excess=float(delta.max())))
    return pd.DataFrame(rows)


def panel(frame,metric):
    worlds=sorted(frame.world_id.unique()); seeds=sorted(frame.policy_seed.unique()); customers=sorted(frame.customer_id.unique())
    index=pd.MultiIndex.from_product([worlds,seeds,customers],names=['world_id','policy_seed','customer_id'])
    values=frame.set_index(list(index.names))[metric].reindex(index)
    if values.isna().any(): raise ValueError("Incomplete world/seed/customer panel")
    return values.to_numpy().reshape(len(worlds),len(seeds),len(customers)),(worlds,seeds,customers)


def crossed_interval(values,repetitions,seed,include_worlds=True):
    rng=np.random.default_rng(seed); values=np.asarray(values); w,s,c=values.shape
    draws=[]
    for _ in range(repetitions):
        wi=rng.integers(w,size=w) if include_worlds else np.arange(w)
        si=rng.integers(s,size=s); ci=rng.integers(c,size=c)
        draws.append(values[np.ix_(wi,si,ci)].mean())
    return np.quantile(draws,[.025,.975])


def summarize_worlds(episodes,settings,benchmark):
    keys=['world_id','world_kind','signal','policy','policy_seed','information_set']
    per_seed=pd.DataFrame([{**dict(zip(keys,key)),**aggregate_episodes(g)} for key,g in episodes.groupby(keys)])
    grouping=[k for k in keys if k!='policy_seed']
    per_world=per_seed.groupby(grouping,as_index=False).mean(numeric_only=True).drop(columns='policy_seed')
    random=per_world[(per_world.world_kind=='randomized')&(per_world.signal=='identity')]
    distributions=[]
    fields=['net_economic_value','default_rate','credit_loss','credit_loss_rate','high_risk_exposure_share','cumulative_reward']
    for policy,g in random.groupby('policy'):
        for metric in fields:
            v=g[metric]; worst_index=v.idxmin() if metric in ('net_economic_value','cumulative_reward') else v.idxmax()
            distributions.append(dict(policy=policy,metric=metric,worlds=len(v),mean=v.mean(),median=v.median(),
                std=v.std(ddof=1),p10=v.quantile(.1),p25=v.quantile(.25),p75=v.quantile(.75),p90=v.quantile(.9),
                worst=v.loc[worst_index],worst_world=g.loc[worst_index,'world_id'],information_set=g.information_set.iloc[0]))
    return per_seed,per_world,pd.DataFrame(distributions),constraint_violations(random,benchmark['risk_constraints'])


def comparisons(episodes,settings):
    selected=episodes[(episodes.world_kind=='randomized')&(episodes.signal=='identity')&(episodes.information_set=='OBSERVABLE_ONLY')]
    policies=sorted(selected.policy.unique()); arrays={}; uncertainty=[]
    for policy,g in selected.groupby('policy'):
        for metric in ('net_economic_value','credit_loss','defaulted'):
            values,labels=panel(g,metric); arrays[policy,metric]=(values,labels)
            lo,hi=crossed_interval(values,settings['bootstrap_repetitions'],settings['bootstrap_seed'])
            clo,chi=crossed_interval(values,settings['bootstrap_repetitions'],settings['bootstrap_seed'],False)
            uncertainty.append(dict(policy=policy,metric=metric,mean=values.mean(),lower=lo,upper=hi,
                conditional_lower=clo,conditional_upper=chi,between_world_sd=values.mean(axis=(1,2)).std(ddof=1),
                mean_within_world_customer_sd=values.mean(axis=1).std(axis=1,ddof=1).mean(),
                training_seed_sd=values.mean(axis=(0,2)).std(ddof=1) if values.shape[1]>1 else 0.))
    pairs=[]; wins=[]
    for a in policies:
        for b in policies:
            if a==b: continue
            for metric in ('net_economic_value','credit_loss','defaulted'):
                av,al=arrays[a,metric]; bv,bl=arrays[b,metric]
                if al[0]!=bl[0] or al[2]!=bl[2]: raise ValueError('World/customer pairing mismatch')
                if av.shape[1]!=bv.shape[1] and min(av.shape[1],bv.shape[1])!=1: raise ValueError('Seed panels differ')
                d=av-bv
                lo,hi=crossed_interval(d,settings['bootstrap_repetitions'],settings['bootstrap_seed'])
                pairs.append(dict(policy=a,reference=b,metric=metric,difference=d.mean(),lower=lo,upper=hi))
                if metric=='net_economic_value':
                    dw=d.mean(axis=(1,2)); wins.append(dict(policy=a,reference=b,win_frequency=float((dw>1e-8).mean()),
                        tie_frequency=float((abs(dw)<=1e-8).mean()),worlds=len(dw)))
    paired=pd.DataFrame(pairs); dominance=[]
    for (a,b),g in paired.groupby(['policy','reference']):
        v=g[g.metric=='net_economic_value'].iloc[0]; risk=g[g.metric=='defaulted'].iloc[0]
        practical=v.difference>settings['dominance']['value_margin_eur'] or risk.difference< -settings['dominance']['default_margin']
        point=v.difference>=-1e-8 and risk.difference<=1e-8 and practical
        supported=v.lower>=0 and risk.upper<=0 and practical
        dominance.append(dict(policy=a,reference=b,point_dominance=point,interval_supported_dominance=supported))
    return paired,pd.DataFrame(wins),pd.DataFrame(uncertainty),pd.DataFrame(dominance)
