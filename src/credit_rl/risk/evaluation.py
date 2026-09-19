"""Probability diagnostics and customer-cluster bootstrap (conditional on macro path)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def calibration_table(y, p, bins=10, quantile=False):
    y, p = np.asarray(y), np.asarray(p)
    frame = pd.DataFrame({"target": y, "predicted_pd": p})
    if quantile:
        # Equal predictions stay together; a constant baseline has one bucket.
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins+1)))
        if len(edges) == 1:
            frame["bucket"] = 0
        else:
            frame["bucket"] = pd.cut(p, edges, include_lowest=True, labels=False)
    else:
        frame["bucket"] = np.minimum((np.asarray(p)*bins).astype(int), bins-1)
    return frame.groupby("bucket", observed=True).agg(n_obs=("target", "size"),
        mean_predicted_pd=("predicted_pd", "mean"), observed_default_rate=("target", "mean")).reset_index()


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    both = len(np.unique(y)) == 2
    cal = calibration_table(y, p)
    return dict(n_obs=len(y), n_defaults=int(y.sum()), default_rate=float(y.mean()),
        mean_predicted_pd=float(p.mean()), roc_auc=float(roc_auc_score(y, p)) if both else np.nan,
        pr_auc=float(average_precision_score(y, p)) if both else np.nan,
        brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
        ece=float(np.average(abs(cal.mean_predicted_pd-cal.observed_default_rate), weights=cal.n_obs)))


def cluster_intervals(frame, p, repetitions=200, seed=55):
    """Resample whole customers with replacement; duplicate clusters keep multiplicity."""
    rng = np.random.default_rng(seed)
    groups = list(frame.reset_index(drop=True).groupby("customer_id", sort=True).indices.values())
    y, p = frame.target.to_numpy(), np.asarray(p)
    values = {"roc_auc": [], "brier": [], "pr_auc": [], "log_loss": []}
    for _ in range(repetitions):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        yy, pp = y[indices], p[indices]
        values["brier"].append(brier_score_loss(yy, pp))
        values["log_loss"].append(log_loss(yy, pp, labels=[0, 1]))
        if len(np.unique(yy)) == 2:
            values["roc_auc"].append(roc_auc_score(yy, pp))
            values["pr_auc"].append(average_precision_score(yy, pp))
    result = {}
    for name, vals in values.items():
        lo, hi = np.quantile(vals, [.025, .975]) if vals else (np.nan, np.nan)
        result.update({name+"_lower": float(lo), name+"_upper": float(hi), name+"_bootstrap_valid": len(vals)})
    return result


def grouped_metrics(frame, p, column):
    frame = frame.assign(predicted_pd=np.asarray(p))
    return pd.DataFrame([{column: value, **metrics(group.target, group.predicted_pd)}
                         for value, group in frame.groupby(column, observed=True)])


def distribution_summary(frame, names):
    return pd.DataFrame([dict(feature=name, mean=frame[name].mean(),
        q10=frame[name].quantile(.1), median=frame[name].median(), q90=frame[name].quantile(.9),
        missing_fraction=frame[name].isna().mean()) for name in names])
