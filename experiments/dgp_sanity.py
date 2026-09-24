"""Large paired DGP diagnostics, controlled interventions, learnability and throughput."""

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_rl import CreditLimitEnv, SimulationConfig
from credit_rl.envs.observation import OBSERVATION_NAMES
from credit_rl.evaluation.dgp_diagnostics import hazard_summary, paired_difference, population_summary
from credit_rl.simulation.customer import initialize_customer
from credit_rl.simulation.dgp import CreditDGP
from credit_rl.simulation.macro import MacroProcess
from credit_rl.simulation.shocks import ShockPath
from credit_rl.simulation.synthetic_snapshot import generate_synthetic_portfolio
from .common import write_manifest


def learnability(frame: pd.DataFrame, seed: int) -> dict:
    """One untuned logistic regression; customer-disjoint train/test, features at t.

    Frame is collected directly from observations BEFORE calling step. Scientific
    hazard/latent logs are never joined to it. Labels are defaults in that next month.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ids = np.array(sorted(frame.customer_id.unique()))
    np.random.default_rng(seed).shuffle(ids)
    cutoff = int(len(ids)*0.7)
    training = frame.customer_id.isin(ids[:cutoff])
    test = ~training
    columns = list(OBSERVATION_NAMES)
    # 'defaulted' is constant zero in active pre-decision observations; exclude it
    # to keep the feature/target distinction explicit even in exported CSVs.
    columns.remove("defaulted")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(frame.loc[training, columns], frame.loc[training, "label_next_month_default"])
    y = frame.loc[test, "label_next_month_default"].to_numpy()
    estimated = model.predict_proba(frame.loc[test, columns])[:, 1]
    base_rate = frame.loc[training, "label_next_month_default"].mean()
    constant = np.full(len(y), base_rate)
    def metrics(p):
        return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
                "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p))}
    return {"model": metrics(estimated), "constant": metrics(constant), "features": columns,
        "train_customers": cutoff, "test_customers": len(ids)-cutoff,
        "train_rows": int(training.sum()), "test_rows": int(test.sum()),
        "test_default_rate": float(y.mean()), "split_seed": seed,
        "cutoff": "Observation at t -> realized default at t+1; same customer never in both partitions",
        "learnability_check_passed": bool(roc_auc_score(y, estimated) > 0.55 and brier_score_loss(y, estimated) < brier_score_loss(y, constant)),
        "near_perfect_flag": bool(roc_auc_score(y, estimated) > 0.98)}


def simulate_cohort(population, paths, config, macro_paths, action, name, results, *, capture_learning=False):
    """Use identical customer/latent/shock inputs; only policy or macro varies."""
    rows, episodes, learning_rows = [], [], []
    env = CreditLimitEnv(config=config, record_diagnostics=True, record_history=False)
    for index, ((state, traits), shocks, macro) in enumerate(zip(population, paths, macro_paths)):
        obs, _ = env.reset(seed=0, options={"initial_state": state, "traits": traits,
            "macro_path": macro, "shock_path": shocks})
        while True:
            before = obs.copy()
            month = env.state.month
            obs, _, term, trunc, info = env.step(action)
            if capture_learning:
                learning_rows.append({"customer_id": state.customer_id, "decision_month": month,
                    **dict(zip(OBSERVATION_NAMES, map(float, before))),
                    "label_next_month_default": int(term)})
            if term or trunc:
                break
        trace = env.get_diagnostics()
        # Latent quantiles are merged ONLY to research diagnostics, never ML features.
        rows.extend(trace.to_dict("records"))
        steps = len(trace)
        episodes.append({"customer_id": state.customer_id, "experiment": name, "months": steps,
            "defaulted": bool(trace.realized_default.iloc[-1]), "return": trace.reward.sum(),
            "credit_loss": trace.reward_credit_loss.sum(),
            "revenue": trace.reward_interest_income.sum()+trace.reward_fee_income.sum(),
            "spending_total": trace.spending.sum(), "payment_total": trace.payment.sum(),
            "mean_limit": trace.credit_limit.mean(), "mean_balance": trace.balance.mean(),
            "mean_spending": trace.spending.mean(), "mean_utilization": trace.utilization.mean(),
            "mean_income": trace.income.mean(), "mean_payment_ratio": trace.payment_ratio.mean(),
            "ever_delinquent": bool(trace.delinquency_status.any()),
            "delinquent_month_fraction": trace.delinquency_status.mean()})
    history, episode_frame = pd.DataFrame(rows), pd.DataFrame(episodes)
    episode_frame.to_csv(results/f"episodes_{name}.csv", index=False)
    learning = pd.DataFrame(learning_rows) if capture_learning else None
    env.close()
    return history, episode_frame, learning


def directional_checks(population, paths, config, trials: int) -> dict:
    """Paired single-period interventions to avoid post-default selection bias."""
    dgp = CreditDGP(config)
    rows = []
    normal = MacroProcess(config.macro).scenario("baseline", config.environment.horizon).states[0]
    for (state, traits), path in zip(population[:trials], paths[:trials]):
        state = replace(state, macro_state=normal)
        shock = path.months[0]
        base = dgp.step(state, 1, traits, shock, normal)
        weaker = dgp.step(state, 1, replace(traits, creditworthiness=traits.creditworthiness-1), shock, normal)
        late = dgp.step(replace(state, months_delinquent=state.months_delinquent+2), 1, traits, shock, normal)
        # Payment/spend channels tested on the same pre-state, without conditioning on survival.
        payer = dgp.step(state, 1, replace(traits, payment_propensity=(traits.payment_propensity+1)/2), shock, normal)
        spender = dgp.step(state, 1, replace(traits, spending_propensity=traits.spending_propensity*1.5), shock, normal)
        rows.append((base.state.defaulted, weaker.state.defaulted, late.state.defaulted,
            base.payment, payer.payment, base.spending, spender.spending,
            base.p_default_true, weaker.p_default_true, late.p_default_true))
    values = np.array(rows, dtype=float)
    checks = {}
    for name, before, after in (("worse_credit_default",0,1), ("higher_delinquency_default",0,2),
        ("better_payment_amount",3,4), ("higher_propensity_spending",5,6),
        ("worse_credit_hazard",7,8), ("higher_delinquency_hazard",7,9)):
        checks[name] = {**paired_difference(values[:,before],values[:,after]),
            "baseline_mean": float(values[:,before].mean()), "intervention_mean": float(values[:,after].mean()),
            "passed": bool(values[:,after].mean() > values[:,before].mean())}
    return {"trials": len(rows), "timing": "one month, identical pre-state/shocks; no survivor selection", "checks": checks}


def plot_diagnostics(main, all_months, sampled, figures):
    regimes = main.groupby("macro_state_used").realized_default.mean().reindex([0,1,2])
    fig, ax = plt.subplots(figsize=(7,4), layout="constrained")
    ax.bar(["Expansion","Normal","Stress"], regimes)
    ax.set(ylabel="Defaults / at-risk customer-months", title="Markov population: conditional default rate")
    fig.savefig(figures/"default_by_regime.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,4), layout="constrained")
    for name, frame in all_months.items():
        ax.plot(frame.month, frame.utilization, label=name)
    ax.set(xlabel="Month", ylabel="Mean utilization among at-risk observations",
           title="Static policy across macro scenarios (survivor composition changes)")
    ax.legend(fontsize=8); fig.savefig(figures/"utilization_over_time.png", dpi=150); plt.close(fig)

    distribution = pd.crosstab(main.month, main.delinquency_bucket, normalize="index").reindex(columns=[0,1,2,3],fill_value=0)
    fig, ax = plt.subplots(figsize=(8,4), layout="constrained")
    ax.stackplot(distribution.index, distribution.to_numpy().T, labels=["Current","1 month","2 months","3+ months"])
    ax.set(xlabel="Month", ylabel="Fraction of active month-end observations", title="Persistent payment shortfalls; default-month rows included")
    ax.legend(loc="upper left"); fig.savefig(figures/"delinquency_over_time.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4), layout="constrained")
    bins = pd.cut(main.credit_limit, bins=10)
    for group in sorted(main.spending_group.unique()):
        sub = main.loc[main.spending_group == group]
        means = sub.groupby(bins.loc[sub.index], observed=True)[["credit_limit","spending"]].mean()
        ax.plot(means.credit_limit, means.spending, marker="o", label=f"Propensity quartile {group+1}")
    ax.set(xlabel="Credit limit (EUR)", ylabel="Mean monthly spending (EUR)", title="Observational association; not an elasticity estimate")
    ax.legend(fontsize=8); fig.savefig(figures/"spending_vs_limit.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,5), layout="constrained")
    plot = ax.hexbin(main.decision_pd, main.p_default_true, gridsize=35, bins="log", mincnt=1)
    fig.colorbar(plot, ax=ax, label="Customer-month count (log scale)")
    ax.plot([0,1],[0,1],"--", color="grey")
    ax.set(xlabel="Opening observable PD proxy", ylabel="Closing conditional simulator hazard",
           title="Different information sets: not a calibration curve")
    fig.savefig(figures/"hazard_vs_predicted_pd.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(5,2,figsize=(12,14),layout="constrained")
    for ax, (identity, frame) in zip(axes.flat, sampled.groupby("customer_id",sort=False)):
        for field, label in (("credit_limit","Limit"),("balance","Balance"),("income","Income")):
            ax.plot(frame.month, frame[field], label=label, linewidth=1.2)
        end = frame.iloc[-1]
        if end.defaulted:
            ax.scatter([end.month],[end.balance],marker="x",color="black")
        ax.set(title=f"Customer {identity}; risk quintile {int(end.risk_group)+1}",xlabel="Month",ylabel="EUR")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Ten customers selected by latent-risk quantile; cross = default")
    fig.savefig(figures/"individual_trajectories.png",dpi=140); plt.close(fig)


def benchmark(population, paths, config, macro_path, count=200) -> dict:
    env = CreditLimitEnv(config=config, record_history=False)
    steps, timed_steps = 0, 0.0
    begin = time.perf_counter()
    for (state, traits), path in zip(population[:count],paths[:count]):
        env.reset(seed=0, options={"initial_state":state,"traits":traits,"macro_path":macro_path,"shock_path":path})
        while True:
            tick=time.perf_counter()
            _, _, term, trunc, _=env.step(env.static_action_index)
            timed_steps += time.perf_counter()-tick
            steps += 1
            if term or trunc: break
    total=time.perf_counter()-begin
    return {"customers":min(count,len(population)),"steps":steps,"seconds_in_step":timed_steps,
            "steps_per_second":steps/timed_steps,"seconds_including_reset":total,
            "mode":"observable logistic PD; history/diagnostics disabled; CPU single process; reset excluded from step rate"}


def run(config, customers, seed, output):
    results, figures = output/"results"/"dgp", output/"figures"/"dgp"
    results.mkdir(parents=True,exist_ok=True); figures.mkdir(parents=True,exist_ok=True)
    horizon=config.environment.horizon
    snapshots=generate_synthetic_portfolio(customers,seed=seed)
    population=[initialize_customer(record, str(i), config, np.random.default_rng(np.random.SeedSequence([seed,i,1])))
                for i,record in enumerate(snapshots.to_dict("records"))]
    paths=[ShockPath.generate(state.customer_id, seed, horizon) for state,_ in population]
    latent=pd.DataFrame([{"customer_id":s.customer_id,**asdict(t)} for s,t in population])
    latent["risk_group"]=pd.qcut(-latent.creditworthiness,5,labels=False)
    latent["spending_group"]=pd.qcut(latent.spending_propensity,4,labels=False)
    latent.to_csv(results/"latent_customers_DIAGNOSTIC_ONLY.csv",index=False)
    # Full initial conditions plus seeds suffice to reconstruct every path; save one path example too.
    (results/"initial_customers.json").write_text(json.dumps([{"state":asdict(s),"traits":asdict(t)} for s,t in population]),encoding="utf-8")
    paths[0].save(results/"example_shock_path.json")
    process=MacroProcess(config.macro)
    scenarios={name:process.scenario(name,horizon) for name in config.macro.scenarios}
    for name,path in scenarios.items(): path.save(results/f"macro_{name}.json")
    markov=[process.generate(horizon,np.random.default_rng(np.random.SeedSequence([seed,i,2]))) for i in range(customers)]
    # Persist every Markov path as a compact reusable table, not a hidden global RNG state.
    pd.DataFrame([{"customer_id":str(i),"month":t,**asdict(m)} for i,p in enumerate(markov) for t,m in enumerate(p.states)]).to_csv(results/"macro_markov_paths.csv.gz",index=False)
    started=time.perf_counter()
    main, main_episodes, learning=simulate_cohort(population,paths,config,markov,
        config.environment.action_multipliers.index(1.0),"markov_static",results,capture_learning=True)
    main=main.merge(latent[["customer_id","risk_group","spending_group"]],on="customer_id",validate="many_to_one")
    main.to_csv(results/"internal_history_DIAGNOSTIC_ONLY.csv.gz",index=False)
    learning.to_csv(results/"observed_learning_check.csv.gz",index=False)
    learn=learnability(learning,seed+1)
    (results/"learnability.json").write_text(json.dumps(learn,indent=2),encoding="utf-8")
    del learning
    for name, groups in {"month":["month"],"macro":["macro_state_used"],"delinquency":["delinquency_bucket"],
                         "latent_risk":["risk_group"]}.items():
        hazard_summary(main,groups).to_csv(results/f"hazard_by_{name}.csv",index=False)
        population_summary(main,groups).to_csv(results/f"population_by_{name}.csv",index=False)
    selected=latent.groupby("risk_group",observed=True).head(2).customer_id
    sampled=main[main.customer_id.isin(selected)].copy()
    sampled.to_csv(results/"representative_trajectories.csv",index=False)
    hazard_checks={"mean":float(main.p_default_true.mean()),"std":float(main.p_default_true.std()),
        "fraction_below_1e_6":float((main.p_default_true<1e-6).mean()),
        "fraction_above_0999":float((main.p_default_true>0.999).mean()),
        "percentiles":{str(q):float(main.p_default_true.quantile(q)) for q in (0.01,0.1,0.5,0.9,0.99)}}
    print('Markov cohort complete:',len(main),'customer-months; learnability AUC',round(learn['model']['roc_auc'],3),flush=True)
    summaries, cohorts, months = [], {}, {}
    for name, policy, macro in [
        ("baseline_decrease",0,"baseline"),("baseline_static",2,"baseline"),("baseline_increase",4,"baseline"),
        ("mild_stress_static",2,"mild_stress"),("severe_stress_static",2,"severe_stress"),("recovery_static",2,"recovery")]:
        # Map named policies to extrema/maintain for configurable action menus.
        action=int(np.argmin(config.environment.action_multipliers)) if policy==0 else (
            int(np.argmax(config.environment.action_multipliers)) if policy==4 else config.environment.action_multipliers.index(1.0))
        history, episodes, _=simulate_cohort(population,paths,config,[scenarios[macro]]*customers,action,name,results)
        cohorts[name]=episodes.set_index("customer_id")
        summary={"experiment":name,"customers":customers,"default_incidence":float(episodes.defaulted.mean())}
        for field in ("return","credit_loss","revenue","spending_total","payment_total","mean_limit","mean_balance",
                      "mean_spending","mean_utilization","mean_income","mean_payment_ratio","ever_delinquent","delinquent_month_fraction"):
            summary[field]=float(episodes[field].mean())
        summaries.append(summary)
        monthly=population_summary(history,["month"])
        monthly.to_csv(results/f"monthly_{name}.csv",index=False)
        if policy==2: months[macro]=monthly
        print(name, 'default incidence', round(summary['default_incidence'],4), 'mean return',round(summary['return'],2),flush=True)
        del history
    table=pd.DataFrame(summaries); table.to_csv(results/"policy_and_macro_summary.csv",index=False)
    paired={name:{field:paired_difference(cohorts['baseline_static'][field].to_numpy(),group[field].to_numpy())
                  for field in ('defaulted','ever_delinquent','credit_loss','return','mean_income','mean_payment_ratio')}
            for name,group in cohorts.items() if name!='baseline_static'}
    direction=directional_checks(population,paths,config,customers)
    (results/"directional_checks.json").write_text(json.dumps(direction,indent=2),encoding="utf-8")
    (results/"paired_differences.json").write_text(json.dumps(paired,indent=2),encoding="utf-8")
    (results/"hazard_distribution.json").write_text(json.dumps(hazard_checks,indent=2),encoding="utf-8")
    plot_diagnostics(main,months,sampled,figures)
    performance=benchmark(population,paths,config,scenarios['baseline'])
    (results/"benchmark.json").write_text(json.dumps(performance,indent=2),encoding="utf-8")
    severe=cohorts['severe_stress_static']; baseline=cohorts['baseline_static']
    checks={"all_directional_checks":all(c['passed'] for c in direction['checks'].values()),
        "stress_higher_default":bool(severe.defaulted.mean()>baseline.defaulted.mean()),
        "stress_higher_ever_delinquency":bool(severe.ever_delinquent.mean()>baseline.ever_delinquent.mean()),
        "learnable":learn['learnability_check_passed'],"not_near_perfect":not learn['near_perfect_flag'],
        "hazard_not_constant":hazard_checks['std']>0.001,
        "hazard_not_binary":hazard_checks['fraction_below_1e_6']+hazard_checks['fraction_above_0999']<0.5}
    (results/"sanity_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    write_manifest(results/"manifest.json",config,{"customers":customers,"seed":seed,"horizon":horizon},
        scenarios=list(scenarios),shock_index="customer_id, month, named channel, seed",
        runtime_seconds=time.perf_counter()-started,checks=checks,
        notes="No fitting of DGP coefficients or PPO. Closing true hazard and opening PD have different information sets.")
    print(json.dumps({"checks":checks,"benchmark":performance},indent=2),flush=True)
    if not all(checks.values()):
        raise RuntimeError("A scientific sanity check failed; inspect saved results")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/simulation.yaml"))
    parser.add_argument("--macro-config",type=Path,default=Path("configs/macro_scenarios.yaml"))
    parser.add_argument("--customers",type=int,default=5000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",type=Path,default=Path("outputs"))
    args=parser.parse_args()
    if args.customers<100: parser.error("At least 100 customers needed for diagnostics (5000+ recommended)")
    run(SimulationConfig.from_yaml(args.config,args.macro_config),args.customers,args.seed,args.output)


if __name__ == "__main__":
    main()
