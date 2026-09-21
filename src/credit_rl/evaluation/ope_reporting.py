"""OPE diagnostics retain support failures and compare against uncertain MC values."""
from pathlib import Path
import numpy as np
import pandas as pd
from .policy_reporting import savefig
import matplotlib.pyplot as plt


def report(output,settings):
    data=pd.read_csv(output/'estimates.csv'); support=pd.read_csv(output/'support.csv')
    weights=pd.read_csv(output/'weights.csv.gz')
    expected=settings['profiles'][output.name]['replicates']
    if not (data.groupby(['policy','policy_seed','metric','clipping']).size()==expected).all():
        raise ValueError('Incomplete OPE replicate panel')
    rows=[]
    for keys,g in data.groupby(['policy','policy_seed','metric','clipping']):
        row=dict(zip(['policy','policy_seed','metric','clipping'],keys))
        row.update(true_MC_value=g.true_MC_value.iloc[0],MC_standard_error=g.MC_standard_error.iloc[0],
            mean_ESS=g.ESS.mean(),min_ESS=g.ESS.min(),max_weight=g.max_weight.max(),
            mean_zero_weight_fraction=g.zero_weight_fraction.mean(),unreliable_fraction=g.unreliable.mean(),replicates=len(g))
        for name in ('IS','WIS'):
            row.update({name+'_mean':g[name].mean(),name+'_sd':g[name].std(ddof=1),
                name+'_mean_error':g[name+'_error'].mean(),name+'_RMSE':np.sqrt(np.square(g[name+'_error']).mean()),
                name+'_coverage_MC_point':g[name+'_covers_MC_point'].mean(),name+'_available_replicates':g[name].notna().sum()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'summary.csv',index=False)
    support.groupby(['policy','policy_seed']).mean(numeric_only=True).reset_index().to_csv(output/'support_summary.csv',index=False)
    figures=Path('outputs/figures/ope')/output.name; figures.mkdir(parents=True,exist_ok=True)
    selected=data[(data.metric=='net_economic_value')&(data.clipping=='none')]
    # Every PPO seed contributes; displays average seeds within each logged replicate.
    plot=selected.groupby(['policy','replicate'],as_index=False).mean(numeric_only=True)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,estimator in zip(axes,('IS','WIS')):
        for name,g in plot.groupby('policy'):
            ax.scatter(g.true_MC_value,g[estimator],label=name,s=22)
        low=plot.true_MC_value.min(); high=plot.true_MC_value.max()
        padding=max(1,(high-low)*.1); low-=padding; high+=padding
        ax.plot([low,high],[low,high],'k--',alpha=.4)
        ax.set_xlim(low,high)
        ax.set(xlabel='Independent MC mean (EUR/customer)',ylabel=estimator+' estimate (EUR/customer)')
    axes[-1].legend(fontsize=6,loc='center left',bbox_to_anchor=(1,.5))
    fig.suptitle('Unclipped OPE; each point is a logged replicate, PPO seeds averaged')
    savefig(fig,figures/'ope_vs_mc.png')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    shown=weights[weights.policy.isin(['Behavior','SoftPDThreshold','Static','PDThreshold','PPO'])].weight
    positive_all=shown[shown>0]
    edges=np.linspace(np.log10(positive_all.min())-.1,np.log10(positive_all.max())+.1,40)
    for name in ('Behavior','SoftPDThreshold','Static','PDThreshold','PPO'):
        w=weights[weights.policy==name].weight.to_numpy(); positive=w[w>0]
        if len(positive): ax.hist(np.log10(positive),bins=edges,histtype='step',density=True,label=f'{name}: {(w==0).mean():.1%} zero')
    ax.set(xlabel='log10 trajectory importance weight (positive weights only)',ylabel='Density conditional on positive weights')
    ax.legend(fontsize=7); savefig(fig,figures/'importance_weights.png')
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    means=plot.groupby('policy').ESS.agg(['mean','min','max'])
    ax.errorbar(means.index,means['mean'],yerr=[means['mean']-means['min'],means['max']-means['mean']],fmt='o',capsize=3)
    ax.axhline(settings['minimum_reliable_ess'],color='red',ls='--',label='Diagnostic ESS threshold; not a reliability guarantee')
    ax.set_yscale('symlog',linthresh=1)
    ax.set_ylim(0,max(1,means['max'].max())*1.4)
    ax.set(ylabel='Trajectory ESS (mean and replicate range)'); ax.tick_params(axis='x',rotation=30); ax.legend(fontsize=7)
    savefig(fig,figures/'ess.png')
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    merged=plot.merge(support.groupby(['policy','replicate'],as_index=False).mean(numeric_only=True)[['policy','replicate','target_mass_below_threshold']],on=['policy','replicate'])
    for name,g in merged.groupby('policy'):
        ax.scatter(g.target_mass_below_threshold,abs(g.WIS-g.true_MC_value),label=name,s=24)
    ax.set(xlabel='Target action mass with behavior propensity <0.1',ylabel='Absolute WIS error vs MC mean (EUR)')
    ax.legend(fontsize=6,loc='center left',bbox_to_anchor=(1,.5)); savefig(fig,figures/'error_vs_support.png')
    print(f'OPE report: {len(data)} estimates; support failures retained',flush=True)
