"""Portfolio-level uncertainty, uncensored violations and measured report figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from credit_rl.evaluation.policy_reporting import savefig

KEYS=['case','kind','policy','policy_seed','portfolio','size','budget','safety_factor','mode']
SHOW=['Static','Decrease20','RiskBased','Greedy','Myopic','PPO_unconstrained','PPO_penalty','PPO_hard']


def bootstrap_panel(values,repetitions,seed):
    """Crossed training seeds and whole portfolios, shared portfolio draw per seed."""
    x=np.asarray(values);rng=np.random.default_rng(seed);s,n=x.shape
    samples=[x[np.ix_(rng.integers(s,size=s),rng.integers(n,size=n))].mean() for _ in range(repetitions)]
    return np.quantile(samples,[.025,.975])


def report(folder,settings,profile):
    cases=json.loads((folder/'cases.json').read_text());expected=sum(c['count'] for c in cases)*(6+3*len(settings['profiles'][profile]['seeds']))
    paths=sorted((folder/'jobs').glob('*/done.json'))
    data=pd.concat([pd.read_csv(p.parent/'portfolios.csv') for p in paths],ignore_index=True)
    if len(data)!=expected:raise ValueError(f'Incomplete portfolio experiment: {len(data)}/{expected}')
    monthly=pd.concat([pd.read_csv(p.parent/'monthly.csv.gz') for p in paths],ignore_index=True)
    actions=pd.concat([pd.read_csv(p.parent/'actions.csv.gz') for p in paths],ignore_index=True)
    data.to_csv(folder/'portfolios.csv',index=False)
    monthly.to_csv(folder/'monthly.csv.gz',index=False);actions.to_csv(folder/'actions.csv.gz',index=False)
    metrics=[c for c in data.select_dtypes(include=['number','bool']).columns if c not in KEYS]
    seeds=data.groupby(['case','kind','policy','policy_seed'],as_index=False)[metrics].mean()
    seeds.to_csv(folder/'seed_metrics.csv',index=False)
    summary=seeds.groupby(['case','kind','policy'],as_index=False)[metrics].mean()
    uncertainty=[];pairs=[]
    for (case,policy),g in data.groupby(['case','policy']):
        for metric in ['value','loss','default_rate','el_violation_rate','true_violation_rate']:
            panel=g.pivot(index='policy_seed',columns='portfolio',values=metric).to_numpy()
            lo,hi=bootstrap_panel(panel,settings['bootstrap_repetitions'],settings['bootstrap_seed'])
            uncertainty.append(dict(case=case,policy=policy,metric=metric,mean=panel.mean(),lower=lo,upper=hi,
                portfolio_sd=panel.mean(axis=0).std(ddof=1),training_seed_sd=panel.mean(axis=1).std(ddof=1) if len(panel)>1 else 0))
            for reference in ('Static','Greedy','Myopic','Decrease20'):
                if policy==reference:continue
                ref=data[(data.case==case)&(data.policy==reference)].pivot(index='policy_seed',columns='portfolio',values=metric).to_numpy()
                delta=panel-ref;plo,phi=bootstrap_panel(delta,settings['bootstrap_repetitions'],settings['bootstrap_seed'])
                pairs.append(dict(case=case,policy=policy,reference=reference,metric=metric,difference=delta.mean(),lower=plo,upper=phi))
    summary.to_csv(folder/'summary.csv',index=False);pd.DataFrame(uncertainty).to_csv(folder/'uncertainty.csv',index=False)
    pd.DataFrame(pairs).to_csv(folder/'paired_comparisons.csv',index=False)
    sensitivity=[]
    for case in cases:
        if case['kind'] not in ('sensor','buffer','budget_rule','order','world'):continue
        anchor='stress_medium' if case['kind'] in ('buffer','budget_rule') else 'normal_medium'
        changed=data[data.case==case['name']]
        reference=data[data.case==anchor]
        paired=changed.merge(reference,on=['policy','policy_seed','portfolio'],suffixes=('','_reference'),validate='one_to_one')
        for metric in ('value','el_violation_rate','true_violation_rate','predicted_safe_true_violated_rate'):
            paired['delta']=paired[metric]-paired[metric+'_reference']
            for policy,g in paired.groupby('policy'):
                panel=g.pivot(index='policy_seed',columns='portfolio',values='delta').to_numpy()
                lo,hi=bootstrap_panel(panel,settings['bootstrap_repetitions'],settings['bootstrap_seed'])
                sensitivity.append(dict(case=case['name'],reference=anchor,policy=policy,metric=metric,
                    portfolios=panel.shape[1],difference=panel.mean(),lower=lo,upper=hi))
    pd.DataFrame(sensitivity).to_csv(folder/'sensitivity_deltas.csv',index=False)
    # Repeated portfolios and seeds remain grouped, rather than treated as extra worlds.
    world=summary[summary.kind=='world'].copy()
    if not world.empty:
        world['predicted_feasible']=world.el_violation_rate<=1e-12;world['true_feasible']=world.true_violation_rate<=1e-12
        world.groupby('policy')[['predicted_feasible','true_feasible','value']].mean().to_csv(folder/'world_feasibility.csv')
    decision_rows=[]
    selected=actions[actions.case=='normal_medium'].copy()
    for field,bins in [('predicted_pd',[0,.2,.4,.6,.8,1.]),('utilization',[0,.25,.5,.75,1.,float('inf')]),
                       ('income',[0,1500,3000,5000,100000]),('score',[0,500,650,800,1000]),
                       ('remaining_fraction',[0,.01,.25,.5,.75,1.01])]:
        if field=='remaining_fraction':selected[field]=selected.remaining_budget_before/selected.risk_budget
        selected['bucket']=pd.cut(selected[field],bins,include_lowest=True).astype(str)
        for (policy,bucket),g in selected.groupby(['policy','bucket']):
            decision_rows.append(dict(policy=policy,feature=field,bucket=bucket,decisions=len(g),
                requested_change=g.requested_change.mean(),effective_change=g.effective_change.mean(),
                increase_rate=(g.effective_change>1e-8).mean(),rejection_rate=(g.constraint_reason=='portfolio_budget').mean()))
    pd.DataFrame(decision_rows).to_csv(folder/'action_segments.csv',index=False)
    # Episode-level diagnostics retain every seed and paired portfolio.
    fixed=data[data.policy=='Decrease20'][['case','portfolio','value','loss','default_rate']]
    gaps=data[data.policy.str.startswith('PPO')].merge(fixed,on=['case','portfolio'],suffixes=('','_fixed'),validate='many_to_one')
    for metric in ('value','loss','default_rate'):gaps[metric+'_gap']=gaps[metric]-gaps[metric+'_fixed']
    gaps[['case','policy','policy_seed','portfolio','value_gap','loss_gap','default_rate_gap']].to_csv(folder/'constant_rule_gaps.csv',index=False)
    first=monthly[monthly.month==0][['case','policy','policy_seed','portfolio','economic_value','expected_loss']]
    seq=data[data.policy=='PPO_hard'][['case','policy_seed','portfolio','value']].merge(
        first[first.policy=='PPO_hard'].drop(columns='policy'),on=['case','policy_seed','portfolio'])
    ref=data[data.policy=='Myopic'][['case','portfolio','value']].merge(
        first[first.policy=='Myopic'][['case','portfolio','economic_value','expected_loss']],on=['case','portfolio'])
    seq=seq.merge(ref,on=['case','portfolio'],suffixes=('_ppo','_myopic'))
    seq['lower_initial_value_higher_total']=(seq.economic_value_ppo<seq.economic_value_myopic)&(seq.value_ppo>seq.value_myopic)
    seq.to_csv(folder/'sequential_cases.csv',index=False)
    monthly['true_classification_uncertain']=(monthly.true_expected_loss-monthly.risk_budget).abs()<=1.96*monthly.true_el_mc_se
    monthly.groupby(['case','policy','policy_seed','portfolio']).true_classification_uncertain.mean().groupby(['case','policy']).mean().to_csv(folder/'diagnostic_mc_ambiguity.csv')
    monthly['ead_violation']=monthly.total_ead>monthly.exposure_budget+1e-7
    caps={c['name']:c['config']['high_risk_share'] for c in cases}
    monthly['high_risk_violation']=monthly.high_risk_share>monthly.case.map(caps)+1e-7
    monthly.groupby(['case','policy','policy_seed','portfolio'])[['ead_violation','high_risk_violation']].mean().groupby(['case','policy']).mean().to_csv(folder/'additional_constraints.csv')
    shadow=[]
    for macro in ('normal','stress'):
        for policy in summary.policy.unique():
            g=summary[(summary.policy==policy)&summary.case.isin([f'{macro}_{b}' for b in settings['budgets']])].copy()
            g['allowed_budget']=g.case.map({f'{macro}_{k}':v for k,v in settings['budgets'].items()});g=g.sort_values('allowed_budget')
            for i in range(1,len(g)):
                before,after=g.iloc[i-1],g.iloc[i]
                shadow.append(dict(macro=macro,policy=policy,from_budget=before.allowed_budget,to_budget=after.allowed_budget,
                    value_change=after.value-before.value,empirical_value_per_budget=(after.value-before.value)/(after.allowed_budget-before.allowed_budget)))
    pd.DataFrame(shadow).to_csv(folder/'budget_marginal_value.csv',index=False)
    tailfile=folder/'tail_portfolios.csv'
    if tailfile.exists():
        tail=pd.read_csv(tailfile);rows=[]
        seed_rows=[]
        for (policy,seed),g in tail.groupby(['policy','policy_seed']):
            loss=g.portfolio_loss;n=len(loss)
            seed_rows.append(dict(policy=policy,policy_seed=seed,portfolios=n,mean=loss.mean(),
                median=loss.median(),std=loss.std(ddof=1),p90=loss.quantile(.9) if n>=200 else np.nan,
                p95=loss.quantile(.95) if n>=400 else np.nan,p99=loss.quantile(.99) if n>=2000 else np.nan))
        pd.DataFrame(seed_rows).to_csv(folder/'loss_distribution_by_seed.csv',index=False)
        # Average PPO seeds for the portfolio distribution, retain raw seed distributions separately.
        for policy,g in tail.groupby('policy'):
            loss=g.groupby('portfolio').portfolio_loss.mean();n=len(loss)
            rows.append(dict(policy=policy,portfolios=n,mean=loss.mean(),median=loss.median(),std=loss.std(ddof=1),
                p90=loss.quantile(.9) if n>=200 else np.nan,p95=loss.quantile(.95) if n>=400 else np.nan,
                p99=loss.quantile(.99) if n>=2000 else np.nan,tail_note='Quantile reporting requires at least 20 expected tail observations'))
        pd.DataFrame(rows).to_csv(folder/'loss_distribution.csv',index=False)
    # Selected initial portfolio identifiers, not winners selected from outcomes.
    monthly[(monthly.portfolio.isin([0,1,2]))&monthly.case.isin(['normal_medium','stress_medium'])].to_csv(folder/'representative_portfolios.csv',index=False)
    figures=Path('outputs/figures/portfolio')/profile;figures.mkdir(parents=True,exist_ok=True)
    make_figures(summary,monthly,actions,folder,figures,settings)
    print(f'Portfolio report: {len(data)} portfolio episodes; {int(data.decisions.sum())} decisions',flush=True)


def make_figures(summary,monthly,actions,folder,figures,settings):
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for macro,ax in zip(('normal','stress'),axes):
        for policy in SHOW:
            g=summary[(summary.policy==policy)&summary.case.isin([f'{macro}_{b}' for b in settings['budgets']])].copy()
            g['budget']=g.case.map({f'{macro}_{k}':v for k,v in settings['budgets'].items()});g=g.sort_values('budget')
            ax.plot(g.budget,g.value,'o-',label=policy)
        ax.set(xlabel='Allowed monthly EL EUR / initial customer (normal regime)',ylabel='Economic value EUR / initial customer',title=macro)
    axes[-1].legend(fontsize=7,loc='center left',bbox_to_anchor=(1,.5));savefig(fig,figures/'budget_frontier.png')
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    markers=['o','s','^']
    for macro,ax in zip(('normal','stress'),axes):
        for index,policy in enumerate(SHOW):
            color=f'C{index}'
            for marker,budget in zip(markers,settings['budgets']):
                g=summary[(summary.policy==policy)&(summary.case==f'{macro}_{budget}')]
                ax.scatter(g.loss,g.value,marker=marker,color=color,s=55,label=policy if marker==markers[0] else None)
        ax.set(xlabel='Realized credit loss EUR / initial customer',ylabel='Economic value EUR / initial customer',title=macro)
    from matplotlib.lines import Line2D
    handles,labels=axes[-1].get_legend_handles_labels()
    handles.extend(Line2D([],[],marker=m,color='gray',linestyle='',label=f'b = {b:g} EUR') for m,b in zip(markers,settings['budgets'].values()))
    axes[-1].legend(handles=handles,fontsize=7,loc='center left',bbox_to_anchor=(1,.5))
    fig.suptitle('Measured policy/budget trade-offs; not an optimal frontier')
    savefig(fig,figures/'risk_value.png')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for policy in ['RiskBased','Greedy','Myopic','PPO_hard']:
        g=monthly[(monthly.case=='stress_medium')&(monthly.policy==policy)].groupby('month').mean(numeric_only=True)
        for ax,metric,label in zip(axes.flat,['risk_budget_utilization','total_ead','credit_loss','economic_value'],
            ['Predicted EL / budget','Live-book EAD EUR','Monthly realized loss EUR','Monthly economic value EUR']):
            ax.plot(g.index,g[metric],label=policy);ax.set(xlabel='Month',ylabel=label);ax.axvspan(8,18,alpha=.06,color='red')
            ax.axvspan(18,22,alpha=.035,color='orange')
    axes.flat[0].axhline(1,color='black',ls='--',lw=1);axes.flat[0].legend(fontsize=7)
    fig.suptitle('Stress months 8–17, recovery 18–21; live-portfolio means, no future signal')
    savefig(fig,figures/'budget_dynamics.png')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    g=summary[(summary.case=='normal_medium')&summary.policy.isin(SHOW)].set_index('policy')
    g[['both_safe_rate','predicted_safe_true_violated_rate','predicted_violated_true_safe_rate','both_violated_rate']].plot.bar(stacked=True,ax=ax)
    ax.set(ylabel='Mean within-portfolio month fraction',ylim=(0,1.05));ax.tick_params(axis='x',rotation=30)
    ax.legend(['Both safe','Predicted safe / simulator violated','Predicted violated / simulator safe','Both violated'],fontsize=7,loc='center left',bbox_to_anchor=(1,.5))
    savefig(fig,figures/'model_risk.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for policy in ['RiskBased','Myopic','PPO_hard']:
        g=summary[(summary.kind=='buffer')&(summary.policy==policy)].copy();g['buffer']=g.case.str.replace('buffer_','',regex=False).astype(float);g=g.sort_values('buffer')
        axes[0].plot(g.buffer,g.value,'o-',label=policy);axes[1].plot(g.buffer,g.true_violation_rate,'o-',label=policy)
    axes[0].set(xlabel='Prespecified safety factor',ylabel='Value EUR/customer');axes[1].set(xlabel='Prespecified safety factor',ylabel='Simulator conditional-risk violation rate');axes[1].legend(fontsize=7)
    savefig(fig,figures/'safety_buffer.png')
    if (folder/'tail_portfolios.csv').exists():
        tail=pd.read_csv(folder/'tail_portfolios.csv');fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        for policy,g in tail.groupby('policy'):
            x=np.sort(g.groupby('portfolio').portfolio_loss.mean());ax.plot(x,np.arange(1,len(x)+1)/len(x),label=policy)
        ax.set(xlabel='Total realized portfolio loss EUR',ylabel='Empirical CDF',title='Independent nominal portfolios; PPO seeds averaged within portfolio')
        ax.legend();savefig(fig,figures/'loss_distribution.png')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for policy,ax in zip(['Greedy','Myopic','PPO_hard'],axes):
        a=actions[(actions.case=='normal_medium')&(actions.policy==policy)].copy()
        a['pd_bin']=pd.cut(a.predicted_pd,[0,.2,.4,.6,.8,1],include_lowest=True)
        a['budget_bin']=pd.cut(a.remaining_budget_before/a.risk_budget,[0,.01,.25,.5,.75,1.01],include_lowest=True)
        grid=a.pivot_table(index='budget_bin',columns='pd_bin',values='requested_change',aggfunc='mean',observed=False)
        cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#cccccc')
        mesh=ax.imshow(grid.to_numpy()*100,origin='lower',vmin=-20,vmax=20,cmap=cmap)
        ax.set_xticks(range(5),['0-.2','.2-.4','.4-.6','.6-.8','.8-1'],rotation=35)
        ax.set_yticks(range(5),['0-.01','.01-.25','.25-.5','.5-.75','.75-1'])
        ax.set(xlabel='Observed PD bin',ylabel='Remaining EL budget fraction',title=policy)
    fig.colorbar(mesh,ax=axes,label='Mean requested limit change (%)')
    fig.suptitle('Observed allocation decisions; gray = unobserved, not a counterfactual response')
    savefig(fig,figures/'budget_action_heatmap.png')
    # Case studies preserve all seed paths; same portfolio and shocks per plot.
    for j in (0,1,2):
        if not ((monthly.case=='stress_medium')&(monthly.portfolio==j)).any():continue
        fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
        for (policy,seed),g in monthly[(monthly.case=='stress_medium')&(monthly.portfolio==j)&monthly.policy.isin(['Greedy','Myopic','PPO_hard'])].groupby(['policy','policy_seed']):
            for ax,metric in zip(axes.flat,['risk_budget','expected_loss','total_ead','economic_value','credit_loss']):
                ax.plot(g.month,g[metric],label=f'{policy} {seed}' if seed!=-1 else policy);ax.set(xlabel='Month',ylabel=metric)
            a=actions[(actions.case=='stress_medium')&(actions.portfolio==j)&(actions.policy==policy)&(actions.policy_seed==seed)]
            changes=a.groupby('month').effective_change.mean()*100
            axes.flat[5].plot(changes.index,changes.values)
            axes.flat[5].set(xlabel='Month',ylabel='Mean effective limit change (%)')
        axes.flat[0].legend(fontsize=6);fig.suptitle(f'Prespecified portfolio {j}; common exogenous shocks')
        savefig(fig,figures/f'portfolio_{j}.png')
