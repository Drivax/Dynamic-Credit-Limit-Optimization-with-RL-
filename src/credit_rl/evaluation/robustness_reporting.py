"""Measured world distributions, mechanism diagnostics and reproducible figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .robustness_statistics import summarize_worlds, comparisons
from .policy_reporting import savefig
import matplotlib.pyplot as plt

SHOW=['Static','AlwaysDecrease','AlwaysDecrease20','PDThreshold','UtilizationPD','MyopicEconomic','PPO','PPO_without_PD']


def smoothness(output, settings):
    from experiments.study_common import load_frozen
    from .worlds import make_cohort
    from credit_rl import CreditLimitEnv
    from credit_rl.policies.decision import transform_observation
    torch.set_num_threads(1)
    base,benchmark,risk,specs=load_frozen()
    cohort=make_cohort(base,count=settings['profiles'][output.name]['customers'],seed=settings['seed'],namespace='ROBUST_TEST')
    rows=[]
    env=CreditLimitEnv(config=base,pd_model=risk,severe_delinquency_months=3)
    for scenario in cohort:
        observation,_=env.reset(seed=scenario.customer_seed,options=scenario.reset_options())
        for spec in specs:
            if spec.name not in SHOW: continue
            policy=spec.factory(env,scenario)
            action=policy.act(transform_observation(observation,spec.without_pd))
            for feature in ('pd','income','balance_utilization'):
                for direction in (-1,1):
                    changed=observation.copy()
                    if feature=='pd':
                        p=np.clip(changed[10],1e-7,1-1e-7)
                        changed[10]=np.exp(-np.logaddexp(0.,-(np.log(p/(1-p))+.1*direction)))
                    else:
                        for index in ((7,) if feature=='income' else (2,3)):
                            raw=float(changed[index])/max(1-float(changed[index]),1e-8)
                            raw*=1+.02*direction; changed[index]=raw/(1+raw)
                    new=policy.act(transform_observation(changed,spec.without_pd))
                    delta=base.environment.action_multipliers[new]-base.environment.action_multipliers[action]
                    rows.append(dict(policy=spec.name,policy_seed=spec.seed,customer_id=scenario.customer_id,
                        feature=feature,direction=direction,action_changed=int(action!=new),absolute_action_jump=abs(delta)))
    env.close()
    pd.DataFrame(rows).to_csv(output/'local_smoothness.csv',index=False)


def trajectory_diagnostics(output,figures):
    for world in ('nominal','macro_unexpected'):
        folder=output/'jobs'/f'{world}__identity'
        if not folder.exists(): continue
        frames=[]
        for file in folder.glob('*.csv.gz'):
            if file.name.split('__')[0] in ('Static','MyopicEconomic','PPO'):
                f=pd.read_csv(file); frames.append(f)
        if not frames: continue
        all_traces=pd.concat(frames,ignore_index=True)
        initial=all_traces[(all_traces.policy=='Static')&(all_traces.month==0)]
        chosen=[initial.sort_values('utilization').iloc[0].customer_id,
                initial.sort_values('utilization').iloc[-1].customer_id,
                initial.sort_values('behavioral_score').iloc[0].customer_id]
        for identity in dict.fromkeys(chosen):
            fig,axes=plt.subplots(4,2,figsize=(11,10),layout='constrained')
            columns=['credit_limit','spending','balance','utilization','predicted_pd','months_delinquent','cumulative_value']
            for (policy,seed),g in all_traces[all_traces.customer_id==identity].groupby(['policy','policy_seed']):
                g=g.sort_values('month').copy()
                g['cumulative_value']=(g.reward_interest_income.fillna(0)+g.reward_fee_income.fillna(0)-g.reward_credit_loss.fillna(0)-g.reward_funding_cost.fillna(0)).cumsum()
                for ax,col in zip(axes.flat,columns):
                    ax.plot(g.month,g[col],label=f'{policy} {seed}' if policy=='PPO' else policy,alpha=.65 if policy=='PPO' else 1)
                    ax.set(xlabel='Month',ylabel=col)
            axes.flat[0].legend(fontsize=6); axes.flat[-1].axis('off')
            fig.suptitle(f'{world}: {identity}; shared exogenous shocks; terminal PD=1 is a sentinel')
            savefig(fig,figures/f'trajectory_{world}_{identity}.png')
        all_traces[all_traces.customer_id.isin(chosen)].to_csv(output/f'representative_{world}.csv.gz',index=False)
        immediate=all_traces[all_traces.month==1][['customer_id','policy','policy_seed','reward']]
        immediate.to_csv(output/f'immediate_rewards_{world}.csv',index=False)


def report(output, settings, benchmark, worlds):
    required=[f'{w.world_id}__identity' for w in worlds]
    if settings['profiles'][output.name]['sensitivity']:
        required += [f'{w.world_id}__{signal}' for w in worlds
            if w.world_id in ('nominal','random_000','macro_unexpected')
            for signal in settings['signals'] if signal!='identity']
    missing=[key for key in required if not (output/'jobs'/key/'done.json').exists()]
    if missing: raise ValueError(f'Incomplete experiment: {len(missing)} jobs have not completed')
    figures=Path('outputs/figures/robustness')/output.name; figures.mkdir(parents=True,exist_ok=True)
    files=[output/'jobs'/key/'episodes.csv.gz' for key in sorted(required)]
    episodes=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    episodes.to_csv(output/'episode_metrics.csv.gz',index=False)
    seeds,world_metrics,distributions,violations=summarize_worlds(episodes,settings,benchmark)
    for name,frame in [('seed_metrics',seeds),('world_metrics',world_metrics),('distributions',distributions),('constraint_violations',violations)]:
        frame.to_csv(output/f'{name}.csv',index=False)
    paired,wins,uncertainty,dominance=comparisons(episodes,settings)
    for name,frame in [('paired_comparisons',paired),('win_frequencies',wins),('uncertainty',uncertainty),('dominance',dominance)]: frame.to_csv(output/f'{name}.csv',index=False)
    random=world_metrics[(world_metrics.world_kind=='randomized')&(world_metrics.signal=='identity')]
    meta={w.world_id:w for w in worlds}
    worst=[]
    for policy,g in random.groupby('policy'):
        row=g.loc[g.net_economic_value.idxmin()].to_dict(); w=meta[row['world_id']]
        worst.append(dict(**row,**w.parameters,**{'macro_'+k:v for k,v in w.macro_spec.items()}))
    pd.DataFrame(worst).to_csv(output/'worst_worlds.csv',index=False)
    oracle=random[random.information_set=='SIMULATOR_ONLY_ORACLE'][['world_id','net_economic_value']].rename(columns={'net_economic_value':'oracle_value'})
    gaps=random.merge(oracle,on='world_id'); gaps['simulator_only_oracle_gap']=gaps.oracle_value-gaps.net_economic_value
    gaps.to_csv(output/'simulator_only_oracle_gaps.csv',index=False)
    gaps.groupby('policy').simulator_only_oracle_gap.agg(['mean','median','max','min']).reset_index().rename(
        columns={'max':'largest_oracle_gap','min':'smallest_oracle_gap'}).to_csv(output/'simulator_only_oracle_gap_summary.csv',index=False)
    diagnostic=[]
    for world_id,g in world_metrics[world_metrics.signal=='identity'].groupby('world_id'):
        if meta[world_id].kind!='oat': continue
        static=g[g.policy=='Static'].net_economic_value.iloc[0]
        myopic=g[g.policy=='MyopicEconomic'].net_economic_value.iloc[0]
        for r in g.itertuples():
            diagnostic.append(dict(world_id=world_id,policy=r.policy,parameter=meta[world_id].varied_parameter,
                endpoint='low' if world_id.endswith('_low') else 'high' if world_id.endswith('_high') else 'reference',
                parameter_value=next(iter(meta[world_id].parameters.values()),np.nan),
                value=r.net_economic_value,delta_static=r.net_economic_value-static,delta_myopic=r.net_economic_value-myopic))
    oat=pd.DataFrame(diagnostic); oat.to_csv(output/'oat_sensitivity.csv',index=False)
    params=pd.DataFrame([dict(world_id=w.world_id,**w.parameters,**{'macro_'+k:v for k,v in w.macro_spec.items()}) for w in worlds if w.kind=='randomized'])
    correlations=[]
    joined=random.merge(params,on='world_id')
    for policy,g in joined.groupby('policy'):
        for parameter in params.columns.drop('world_id'):
            correlations.append(dict(policy=policy,parameter=parameter,spearman=float(g[parameter].corr(g.net_economic_value,method='spearman'))))
    pd.DataFrame(correlations).to_csv(output/'parameter_associations.csv',index=False)
    # Align diagnostic worlds with nominal values; avoid pooling these with random-world priors.
    nominal=world_metrics[(world_metrics.world_id=='nominal')&(world_metrics.signal=='identity')]
    mechanisms=world_metrics.merge(nominal,on=['policy','information_set'],suffixes=('','_nominal'))
    for field in ('net_economic_value','revenue','credit_loss','mean_balance','mean_utilization','default_rate',
                  'increase_fraction','decrease_fraction','limit_changes','reversals'):
        mechanisms[field+'_change_from_nominal']=mechanisms[field]-mechanisms[field+'_nominal']
    mechanisms.to_csv(output/'mechanism_differences.csv',index=False)
    timings=pd.concat([pd.read_csv(p) for p in (output/'jobs').glob('*/timing.csv')],ignore_index=True)
    timings.to_csv(output/'timing.csv',index=False)
    for name in ('monthly','actions'):
        pd.concat([pd.read_csv(p) for p in (output/'jobs').glob(f'*/{name}.csv.gz')],ignore_index=True).to_csv(output/f'{name}.csv.gz',index=False)
    # Per-episode comparison to the constant cut, including information shifts.
    rule=episodes[episodes.policy=='AlwaysDecrease20']
    comparisons_constant=[]
    fields=['net_economic_value','cumulative_reward','credit_loss','defaulted','final_balance','final_limit','increases','decreases','reversals']
    for (policy,seed,world,signal),g in episodes[episodes.policy.isin(['PPO','PPO_without_PD'])].groupby(['policy','policy_seed','world_id','signal']):
        ref=rule[(rule.world_id==world)&(rule.signal==signal)]
        combined=g.merge(ref,on='customer_id',suffixes=('_actor','_rule'),validate='one_to_one')
        comparisons_constant.append(dict(policy=policy,policy_seed=seed,world_id=world,signal=signal,
            **{field:float(abs(combined[field+'_actor']-combined[field+'_rule']).max()) for field in fields}))
    pd.DataFrame(comparisons_constant).to_csv(output/'constant_equivalence.csv',index=False)
    smoothness(output,settings)
    trajectory_diagnostics(output,figures)
    sequential=[]
    for world in ('nominal','macro_unexpected'):
        file=output/f'immediate_rewards_{world}.csv'
        if not file.exists(): continue
        immediate=pd.read_csv(file)
        terminal=episodes[(episodes.world_id==world)&(episodes.signal=='identity')]
        reference=terminal[terminal.policy=='MyopicEconomic'][['customer_id','net_economic_value']].merge(
            immediate[immediate.policy=='MyopicEconomic'][['customer_id','reward']],on='customer_id')
        for seed,g in terminal[terminal.policy=='PPO'].groupby('policy_seed'):
            paired_initial=g[['customer_id','net_economic_value']].merge(immediate[(immediate.policy=='PPO')&(immediate.policy_seed==seed)][['customer_id','reward']],on='customer_id')
            paired_initial=paired_initial.merge(reference,on='customer_id',suffixes=('_ppo','_myopic'))
            paired_initial['immediate_reward_gap']=paired_initial.reward_ppo-paired_initial.reward_myopic
            paired_initial['total_value_gap']=paired_initial.net_economic_value_ppo-paired_initial.net_economic_value_myopic
            paired_initial['lower_immediate_higher_total']=(paired_initial.immediate_reward_gap<0)&(paired_initial.total_value_gap>0)
            sequential.append(paired_initial.assign(world_id=world,policy_seed=seed))
    if sequential: pd.concat(sequential,ignore_index=True).to_csv(output/'sequential_cases.csv',index=False)
    figures_summary(world_metrics,distributions,violations,oat,wins,output,figures)
    print(f'Robustness report: {len(episodes)} episodes, {len(random.world_id.unique())} randomized worlds',flush=True)


def figures_summary(worlds,distributions,violations,oat,wins,output,figures):
    random=worlds[(worlds.world_kind=='randomized')&(worlds.signal=='identity')]
    names=[n for n in SHOW if n in set(random.policy)]
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    ax.boxplot([random[random.policy==n].net_economic_value for n in names],orientation='horizontal',tick_labels=names)
    ax.set(xlabel='Mean value (EUR/customer) in each sampled world',title='Declared synthetic world distribution; no optimality ranking')
    savefig(fig,figures/'world_distribution.png')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    means=[]
    for index,n in enumerate(names):
        g=random[random.policy==n]; means.append((n,g.default_rate.mean(),g.net_economic_value.mean()))
        color=plt.get_cmap('tab10')(index)
        ax.scatter(g.default_rate,g.net_economic_value,alpha=.16,s=12,color=color)
        ax.scatter(g.default_rate.mean(),g.net_economic_value.mean(),s=50,label=n,color=color)
    means=[(n,g.default_rate.mean(),g.net_economic_value.mean()) for n,g in
           random[random.information_set!='SIMULATOR_ONLY_ORACLE'].groupby('policy')]
    frontier=[(r,v) for n,r,v in means if not any(rr<=r and vv>=v and (rr<r-1e-8 or vv>v+1e-8) for _,rr,vv in means)]
    if frontier:
        xx,yy=zip(*sorted(set(frontier))); ax.plot(xx,yy,'k--',alpha=.5,label='Empirical mean frontier')
    ax.set(xlabel='Default incidence',ylabel='Value EUR/customer',title='Economic value and risk across worlds')
    ax.legend(fontsize=7,loc='center left',bbox_to_anchor=(1,0.5)); savefig(fig,figures/'risk_value.png')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    d=distributions[distributions.metric=='net_economic_value'].set_index('policy').loc[names]
    ax.scatter(d['mean'],names,label='Mean'); ax.scatter(d.worst,names,marker='x',label='Worst observed world')
    ax.set(xlabel='Value EUR/customer'); ax.legend(); savefig(fig,figures/'worst_world.png')
    if not oat.empty:
        endpoint=oat[oat.endpoint!='reference'].pivot(index=['parameter','policy'],columns='endpoint',values='delta_static')
        grid=(endpoint.high-endpoint.low).unstack('policy').reindex(columns=names)
        fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
        limit=max(1,abs(grid.to_numpy()).max()); m=ax.imshow(grid,vmin=-limit,vmax=limit,cmap='RdBu',aspect='auto')
        ax.set_xticks(range(len(names)),names,rotation=30,ha='right'); ax.set_yticks(range(len(grid)),grid.index)
        ax.set_title('OAT high minus low: change in value advantage over Static')
        fig.colorbar(m,ax=ax,label='EUR/customer'); savefig(fig,figures/'parameter_sensitivity.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    sensors=worlds[worlds.world_id=='nominal']
    for name in ('PDThreshold','MyopicEconomic','PPO'):
        g=sensors[sensors.policy==name].sort_values('signal')
        axes[0].plot(g.signal,g.net_economic_value,'o-',label=name); axes[1].plot(g.signal,g.default_rate,'o-',label=name)
    for ax in axes: ax.tick_params(axis='x',rotation=60); ax.legend(fontsize=7)
    axes[0].set(ylabel='Value EUR/customer'); axes[1].set(ylabel='Default incidence')
    fig.suptitle('Actor-only information stress in nominal world'); savefig(fig,figures/'pd_distortion.png')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    grid=violations.pivot(index='policy',columns='metric',values='violation_frequency').reindex(names)
    grid.plot.bar(ax=ax); ax.tick_params(axis='x',rotation=30); ax.set(ylabel='Fraction of randomized worlds violating threshold',ylim=(0,1.05)); ax.legend(fontsize=7)
    savefig(fig,figures/'constraint_violations.png')
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    grid=wins.pivot(index='policy',columns='reference',values='win_frequency').reindex(index=names,columns=names)
    mesh=ax.imshow(grid,vmin=0,vmax=1,cmap='Blues'); ax.set_xticks(range(len(names)),names,rotation=50,ha='right'); ax.set_yticks(range(len(names)),names)
    ax.set_title('Empirical strict-win frequency across sampled worlds; ties separate')
    fig.colorbar(mesh,ax=ax); savefig(fig,figures/'rank_stability.png')
    smooth=pd.read_csv(output/'local_smoothness.csv').groupby(['policy','feature']).action_changed.mean().unstack()
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained'); smooth.plot.bar(ax=ax)
    ax.set(ylabel='Fraction changing action'); ax.tick_params(axis='x',rotation=30)
    savefig(fig,figures/'local_smoothness.png')
