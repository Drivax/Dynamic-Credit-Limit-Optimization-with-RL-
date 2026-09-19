"""Paired customer bootstrap, with PPO training-seed resampling above customers."""
import numpy as np
import pandas as pd
from .policy_engine import aggregate_episodes

KEY_METRICS = ("cumulative_reward", "net_economic_value", "credit_loss", "defaulted")


def matrix(frame, metric):
    result = frame.pivot(index="policy_seed", columns="customer_id", values=metric).sort_index().sort_index(axis=1)
    if result.isna().any().any():
        raise ValueError("Incomplete seed/customer evaluation panel")
    return result


def interval(values, repetitions, seed):
    """Shared customer resample across all sampled policy seeds, never monthly rows."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        customers = rng.integers(values.shape[1], size=values.shape[1])
        seeds = rng.integers(values.shape[0], size=values.shape[0])
        draws.append(values[np.ix_(seeds, customers)].mean())
    return np.quantile(draws, [.025, .975])


def summarize(episodes, settings):
    keys = ["policy", "policy_seed", "scenario", "information_set"]
    seed_rows = [{**dict(zip(keys, key)), **aggregate_episodes(group)} for key, group in episodes.groupby(keys)]
    seed_table = pd.DataFrame(seed_rows)
    rows = []
    for (policy, scenario, information), group in episodes.groupby(["policy", "scenario", "information_set"]):
        row = dict(policy=policy, scenario=scenario, information_set=information,
                   unique_customers=group.customer_id.nunique(), training_seeds=group.policy_seed.nunique(),
                   **aggregate_episodes(group))
        for metric in KEY_METRICS:
            values = matrix(group, metric).to_numpy()
            lower, upper = interval(values, settings["evaluation"]["bootstrap_repetitions"], settings["evaluation"]["bootstrap_seed"])
            row[metric+"_lower"], row[metric+"_upper"] = lower, upper
            row[metric+"_seed_sd"] = values.mean(axis=1).std(ddof=1) if len(values) > 1 else 0.
        r = settings["risk_constraints"]
        row["default_alert"] = row["default_rate"] > r["max_default_rate"]
        row["loss_alert"] = row["credit_loss_rate"] > r["max_credit_loss_rate"]
        row["high_risk_exposure_alert"] = row["high_risk_exposure_share"] > r["max_high_risk_exposure_share"]
        rows.append(row)
    return pd.DataFrame(rows), seed_table


def paired_comparisons(episodes, settings):
    rows, distributions = [], []
    for scenario, frame in episodes.groupby("scenario"):
        for reference in ("Static", "MyopicEconomic"):
            base = frame[frame.policy == reference]
            if base.empty:
                continue
            for policy, group in frame.groupby("policy"):
                if policy == reference:
                    continue
                for metric in KEY_METRICS:
                    a, b = matrix(group, metric), matrix(base, metric)
                    if not a.columns.equals(b.columns):
                        raise ValueError("Paired customers do not match")
                    differences = a.to_numpy()-b.to_numpy()
                    lo, hi = interval(differences, settings["evaluation"]["bootstrap_repetitions"], settings["evaluation"]["bootstrap_seed"])
                    rows.append(dict(policy=policy, reference=reference, scenario=scenario, metric=metric,
                        difference=differences.mean(), lower=lo, upper=hi,
                        fraction_customers_positive=float((differences.mean(axis=0) > 0).mean()),
                        information_set=group.information_set.iloc[0]))
                    distributions.extend(dict(policy=policy, reference=reference, scenario=scenario, metric=metric,
                        customer_id=customer, seed_mean_difference=float(value))
                        for customer, value in zip(a.columns, differences.mean(axis=0)))
    return pd.DataFrame(rows), pd.DataFrame(distributions)


def robustness(summary):
    out = []
    for policy, group in summary.groupby("policy"):
        row = dict(policy=policy, information_set=group.information_set.iloc[0])
        for r in group.itertuples():
            row[f"{r.scenario}_value"] = r.net_economic_value
            row[f"{r.scenario}_default_rate"] = r.default_rate
        main = group[group.scenario.isin(["baseline", "mild_stress", "severe_stress", "recovery"])]
        if not main.empty:
            row["worst_scenario_value"] = main.net_economic_value.min()
        out.append(row)
    return pd.DataFrame(out)
