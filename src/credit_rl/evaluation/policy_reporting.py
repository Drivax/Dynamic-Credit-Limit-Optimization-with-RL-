"""Economic, action, convergence and PD-feedback diagnostics from frozen policies."""
from dataclasses import replace
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_rl.envs.observation import build_observation
from credit_rl.evaluation.scenarios import make_scenarios
from credit_rl.risk.dataset import build_dataset
from credit_rl.risk.features import FEATURE_NAMES
from credit_rl.risk.evaluation import metrics, distribution_summary
from credit_rl.simulation.macro import MacroState, MacroRegime


def savefig(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)


def diagnostics(settings, results):
    monthly, actions, pd_scores, distributions = [], [], [], []
    for path in sorted((results/"trajectories").glob("*.csv.gz")):
        history = pd.read_csv(path)
        tags = history.iloc[0][["policy", "policy_seed", "scenario", "information_set"]].to_dict()
        steps = history[history.month > 0].copy()
        grouped = steps.groupby("month").agg(active_customers=("customer_id", "nunique"),
            mean_limit=("credit_limit", "mean"), mean_balance=("balance", "mean"),
            mean_utilization=("utilization", "mean"), mean_pd=("decision_pd", "mean"),
            reward_sum=("reward", "sum"), credit_loss_sum=("reward_credit_loss", "sum"),
            defaults=("defaulted", "sum"), mean_effective_action=("effective_action", "mean"))
        monthly.append(grouped.reset_index().assign(**tags))
        steps["PD_bucket"] = pd.cut(steps.decision_pd, [0, .2, .4, .6, .8, 1], include_lowest=True)
        steps["utilization_bucket"] = pd.cut(steps.opening_utilization, [0, .5, .75, 1, np.inf], include_lowest=True)
        steps["delinquency_bucket"] = steps.opening_delinquency.clip(upper=3)
        steps["macro_regime"] = steps.opening_macro
        for segment in ("PD_bucket", "utilization_bucket", "delinquency_bucket", "macro_regime"):
            for value, group in steps.groupby(segment, observed=True):
                actions.append(dict(**tags, segment=segment, value=str(value), n=len(group),
                    increase_fraction=float((group.effective_action > 1e-9).mean()),
                    decrease_fraction=float((group.effective_action < -1e-9).mean()),
                    no_change_fraction=float((abs(group.effective_action) <= 1e-9).mean()),
                    requested_change=float(group.requested_action.mean()), effective_change=float(group.effective_action.mean())))
        if tags["policy"] in ("Static", "MyopicEconomic", "PPO", "PPO_without_PD") and tags["scenario"] in ("baseline", "mild_stress", "severe_stress", "recovery"):
            data, _ = build_dataset(history, settings["pd_horizon_months"])
            matched = data.merge(history[["customer_id", "month", "predicted_pd"]], on=["customer_id", "month"], validate="one_to_one")
            pd_scores.append(dict(**tags, **metrics(matched.target, matched.predicted_pd)))
            distributions.append(distribution_summary(data, FEATURE_NAMES).assign(**tags))
    pd.DataFrame(actions).to_csv(results/"action_metrics.csv", index=False)
    pd.concat(monthly, ignore_index=True).to_csv(results/"monthly_metrics.csv", index=False)
    pd.DataFrame(pd_scores).to_csv(results/"pd_calibration_under_policies.csv", index=False)
    pd.concat(distributions, ignore_index=True).to_csv(results/"pd_feature_distributions.csv", index=False)
    training_path = Path(settings["pd_model"]).parents[2]/"results"/"pd"/"train_feature_distributions.csv"
    if training_path.exists():
        reference = pd.read_csv(training_path).rename(columns={"mean": "pd_training_mean", "q10": "pd_training_q10", "q90": "pd_training_q90"})
        shifts = pd.concat(distributions, ignore_index=True).merge(reference[["feature", "pd_training_mean", "pd_training_q10", "pd_training_q90"]], on="feature")
        shifts["mean_shift"] = shifts["mean"]-shifts.pd_training_mean
        shifts.to_csv(results/"pd_feature_shift_from_training.csv", index=False)


def convergence(models, figures):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    fields = ["rollout/ep_rew_mean", "train/value_loss", "train/policy_gradient_loss",
              "train/entropy_loss", "train/approx_kl", "train/explained_variance"]
    for folder in sorted(models.glob("with_pd_*")):
        frame = pd.read_csv(folder/"progress.csv")
        for ax, field in zip(axes.flat, fields):
            if field in frame:
                ax.plot(frame["time/total_timesteps"], frame[field], label=folder.name)
            ax.set(xlabel="Training transitions", title=field)
    axes.flat[0].set_ylabel("kEUR reward, episode mean")
    axes.flat[0].legend(fontsize=6)
    savefig(fig, figures/"ppo_training_diagnostics.png")
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for folder in sorted(models.glob("*_pd_*")):
        if not (folder/"validation.csv").exists(): continue
        f = pd.read_csv(folder/"validation.csv")
        ax.plot(f.timesteps, f.validation_discounted_reward, "o", label=folder.name,
                linestyle="--" if folder.name.startswith("without") else "-")
    ax.set(xlabel="Training transitions", ylabel="Validation discounted reward (EUR)", title="All seeds and PD ablation; fixed validation cohort")
    ax.legend(fontsize=6, ncol=2)
    savefig(fig, figures/"validation_convergence.png")


def policy_slices(config, settings, pd_model, specs, figures, results):
    scenario = make_scenarios(config, settings, "validation", customers=1)[0]
    policies = []
    from credit_rl import CreditLimitEnv
    env = CreditLimitEnv(config=config, pd_model=pd_model, severe_delinquency_months=settings["guardrails"]["severe_delinquency_months"])
    for spec in specs:
        if spec.name in ("PPO", "PDThreshold", "MyopicEconomic"):
            policies.append((spec, spec.factory(env, scenario)))
    rows = []
    for delinquency in (0, 3):
        for regime in (MacroRegime.NORMAL, MacroRegime.STRESS):
            for u in np.linspace(.1, 1.3, 13):
                state = replace(scenario.initial_state, credit_limit=5000, balance=5000*u,
                    months_delinquent=delinquency, macro_state=MacroState.from_config(regime, config.macro))
                for p in np.linspace(.05, .95, 19):
                    obs = build_observation(state, p, 0, config)
                    for spec, policy in policies:
                        action = policy.act(obs)
                        rows.append(dict(policy=spec.name, policy_seed=spec.seed, pd=p, utilization=u,
                            delinquency=delinquency, macro=int(regime), requested_multiplier=config.environment.action_multipliers[action]))
    frame = pd.DataFrame(rows)
    frame.to_csv(results/"policy_slices.csv", index=False)
    subset = frame[(frame.delinquency == 0)&(frame.macro == 1)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, policy in zip(axes, ("PDThreshold", "MyopicEconomic", "PPO")):
        # PPO displays mean requested action across all seeds, not a selected seed.
        grid = subset[subset.policy == policy].pivot_table(index="utilization", columns="pd", values="requested_multiplier", aggfunc="mean")
        mesh = ax.imshow((grid.to_numpy()-1)*100, origin="lower", aspect="auto", extent=[.05,.95,.1,1.3], vmin=-20,vmax=20,cmap="RdBu")
        ax.set(title=policy, xlabel="Supplied predicted PD", ylabel="Utilization")
    fig.colorbar(mesh, ax=axes, label="Requested change (%), PPO averaged over all seeds")
    fig.suptitle("Controlled observation slice; normal macro, no delinquency; not a causal intervention")
    savefig(fig, figures/"policy_heatmaps.png")
    checks = []
    for key, group in frame.groupby(["policy", "policy_seed", "delinquency", "macro", "utilization"]):
        differences = np.diff(group.sort_values("pd").requested_multiplier)
        checks.append(dict(zip(["policy", "policy_seed", "delinquency", "macro", "utilization"], key),
            increases_when_pd_increases=int((differences > 1e-8).sum()), adjacent_pd_pairs=len(differences)))
    pd.DataFrame(checks).to_csv(results/"policy_slice_monotonicity.csv", index=False)


def representative_trajectories(results, figures, config, settings):
    initial = make_scenarios(config, settings, "test")
    # Select by observable initial characteristics, before examining policy outcomes.
    groups = [min(initial, key=lambda s: s.initial_state.utilization),
              max(initial, key=lambda s: s.initial_state.utilization),
              min(initial, key=lambda s: s.initial_state.behavioral_score)]
    ids = list(dict.fromkeys(s.customer_id for s in groups))
    frames = []
    for path in (results/"trajectories").glob("baseline__*.csv.gz"):
        f = pd.read_csv(path)
        if f.policy.iloc[0] in ("Static", "MyopicEconomic", "PPO"):
            frames.append(f[f.customer_id.isin(ids)])
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(results/"representative_trajectories.csv", index=False)
    fields = ["credit_limit", "balance", "utilization", "predicted_pd", "months_delinquent", "reward", "cumulative_value"]
    for identity in ids:
        fig, axes = plt.subplots(4, 2, figsize=(11, 11), layout="constrained")
        selected = data[data.customer_id == identity]
        for (policy, seed), group in selected.groupby(["policy", "policy_seed"]):
            group = group.sort_values("month").copy()
            group["cumulative_value"] = (group.reward_interest_income.fillna(0)+group.reward_fee_income.fillna(0)-group.reward_credit_loss.fillna(0)-group.reward_funding_cost.fillna(0)).cumsum()
            for ax, field in zip(axes.flat, fields):
                ax.plot(group.month, group[field], label=f"{policy} {seed}" if policy == "PPO" else policy,
                        alpha=.6 if policy == "PPO" else 1.)
                ax.set(xlabel="Month", ylabel=field)
        axes.flat[-1].axis("off")
        axes.flat[0].legend(fontsize=6)
        fig.suptitle(f"{identity}: matched exogenous inputs; all PPO seeds")
        savefig(fig, figures/f"trajectory_{identity}.png")


def summary_figures(results, figures):
    summary = pd.read_csv(results/"summary.csv")
    baseline = summary[(summary.scenario == "baseline") & (summary.information_set == "OBSERVABLE_ONLY")].sort_values("policy")
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.errorbar(baseline.policy, baseline.net_economic_value,
        yerr=[np.maximum(0, baseline.net_economic_value-baseline.net_economic_value_lower), np.maximum(0,baseline.net_economic_value_upper-baseline.net_economic_value)],fmt="o",capsize=4)
    ax.tick_params(axis="x", rotation=30)
    ax.set(ylabel="Mean net economic value (EUR/customer)", title="Held-out baseline; 95% customer/seed bootstrap intervals")
    savefig(fig, figures/"economic_value.png")
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for r in baseline.itertuples():
        ax.scatter(r.default_rate, r.net_economic_value, s=50)
        ax.annotate(r.policy, (r.default_rate,r.net_economic_value), fontsize=8)
    ax.set(xlabel="Defaults / initial customers", ylabel="Net economic value (EUR/customer)", title="Economics and risk; no composite ranking")
    savefig(fig, figures/"risk_return.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for policy in ("Static", "UtilizationPD", "MyopicEconomic", "PPO", "PPO_without_PD"):
        g=summary[(summary.policy==policy)&summary.scenario.isin(["baseline","mild_stress","severe_stress","recovery"])].set_index("scenario").reindex(["baseline","mild_stress","severe_stress","recovery"])
        axes[0].plot(g.index,g.net_economic_value,"o-",label=policy)
        axes[1].plot(g.index,g.default_rate,"o-",label=policy)
    for ax in axes:
        ax.tick_params(axis="x",rotation=25)
        ax.legend(fontsize=7)
    axes[0].set(ylabel="Net economic value (EUR/customer)")
    axes[1].set(ylabel="Default incidence per customer")
    savefig(fig, figures/"stress_robustness.png")
    fig, ax=plt.subplots(figsize=(10,4),layout="constrained")
    bottom=np.zeros(len(baseline))
    for label,column in [("Decrease","decrease_fraction"),("Maintain","no_change_fraction"),("Increase","increase_fraction")]:
        ax.bar(baseline.policy,baseline[column],bottom=bottom,label=label)
        bottom+=baseline[column].to_numpy()
    ax.tick_params(axis="x",rotation=30); ax.legend()
    ax.set(ylabel="Share of effective actions",title="Action behavior on active customer-months")
    savefig(fig,figures/"action_distribution.png")
    differences=pd.read_csv(results/"paired_distributions.csv.gz")
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    for policy in ("PPO","UtilizationPD","MyopicEconomic"):
        vals=differences[(differences.policy==policy)&(differences.reference=="Static")&(differences.scenario=="baseline")&(differences.metric=="net_economic_value")].seed_mean_difference
        ax.hist(vals,bins=35,alpha=.4,label=policy)
    ax.axvline(0,color="black",linestyle="--"); ax.legend()
    ax.set(xlabel="Paired value difference vs Static (EUR/customer)",ylabel="Customers")
    savefig(fig,figures/"paired_value_distribution.png")
    scores=pd.read_csv(results/"pd_calibration_under_policies.csv")
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for name,g in scores.groupby("policy"):
        mean=g.groupby("scenario")[["default_rate","mean_predicted_pd","brier"]].mean().reindex(["baseline","mild_stress","severe_stress","recovery"])
        axes[0].plot(mean.index,mean.mean_predicted_pd-mean.default_rate,"o-",label=name)
        axes[1].plot(mean.index,mean.brier,"o-",label=name)
    axes[0].axhline(0,color="black",alpha=.3)
    for ax in axes: ax.tick_params(axis="x",rotation=25); ax.legend(fontsize=7)
    axes[0].set(ylabel="Mean forecast minus H-month prevalence")
    axes[1].set(ylabel="Brier score on eligible policy-generated snapshots")
    savefig(fig,figures/"pd_feedback.png")


def report(config, settings, pd_model, results, models, figures, identity):
    from experiments.compare_policies import load_specs
    settings={**settings,"pd_horizon_months":int(pd_model.metadata["horizon_months"])}
    diagnostics(settings, results)
    specs=load_specs(config,settings,results,models,identity)
    summary_figures(results,figures)
    convergence(models,figures)
    policy_slices(config,settings,pd_model,specs,figures,results)
    representative_trajectories(results,figures,config,settings)
