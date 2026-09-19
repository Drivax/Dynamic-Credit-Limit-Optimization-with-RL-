"""Machine-readable diagnostics and figures derived only from saved predictions."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluation import calibration_table, grouped_metrics


def figures(predictions, comparison, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    test = predictions[predictions["sample"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for name, group in test.groupby("model"):
        cal = calibration_table(group.target, group.predicted_pd, quantile=True)
        axes[0].plot(cal.mean_predicted_pd, cal.observed_default_rate, "o-", label=name)
        axes[1].plot(range(1, len(cal)+1), cal.observed_default_rate, "o-", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[0].set(xlabel="Mean predicted H-month PD", ylabel="Observed default frequency", title="Test reliability (quantile buckets)")
    axes[1].set(xlabel="Risk bucket (ties kept together)", ylabel="Observed default frequency", title="Risk ordering")
    axes[0].legend(fontsize=7)
    fig.savefig(destination/"calibration_deciles.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, name in zip(axes, ("logistic_calibrated", "boosting_calibrated")):
        for label in (0, 1):
            values = test[(test.model == name) & (test.target == label)].predicted_pd
            ax.hist(values, bins=np.linspace(0, 1, 31), density=True, alpha=.5, label=f"Y={label}")
        ax.set(title=name, xlabel="Predicted PD", ylabel="Density")
        ax.legend()
    fig.savefig(destination/"pd_distribution.png", dpi=150)
    plt.close(fig)
    selected = comparison[comparison.model.isin(["constant", "logistic_calibrated", "boosting_calibrated"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for name, group in selected.groupby("model"):
        axes[0].plot(group["sample"], group.brier, "o-", label=name)
        axes[1].plot(group["sample"], group.mean_predicted_pd-group.default_rate, "o-", label=name)
    for ax in axes:
        ax.tick_params(axis="x", rotation=70)
        ax.legend(fontsize=7)
    axes[0].set(ylabel="Brier score", title="Temporal, macro and policy diagnostics")
    axes[1].axhline(0, color="black", alpha=.3)
    axes[1].set(ylabel="Mean PD minus default prevalence", title="Calibration bias")
    fig.savefig(destination/"stress_policy_comparison.png", dpi=150)
    plt.close(fig)
    time = predictions[(predictions.model == "logistic_calibrated") & predictions["sample"].isin(["validation", "test", "oot"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for sample, group in time.groupby("sample"):
        table = grouped_metrics(group, group.predicted_pd, "observation_month")
        axes[0].plot(table.observation_month, table.mean_predicted_pd, label=f"{sample}: predicted")
        axes[0].plot(table.observation_month, table.default_rate, "--", label=f"{sample}: observed")
        axes[1].plot(table.observation_month, table.brier, label=sample)
    axes[0].set(xlabel="Simulation calendar month", ylabel="H-month risk", title="Risk and calibration over time")
    axes[1].set(xlabel="Simulation calendar month", ylabel="Brier score")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=8)
    fig.savefig(destination/"temporal_drift.png", dpi=150)
    plt.close(fig)
    stress = predictions[predictions.model == "logistic_calibrated"]
    table = grouped_metrics(stress, stress.predicted_pd, "macro_state")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    axes[0].bar(table.macro_state-.15, table.mean_predicted_pd, width=.3, label="Predicted")
    axes[0].bar(table.macro_state+.15, table.default_rate, width=.3, label="Observed")
    axes[1].bar(table.macro_state, table.brier)
    axes[0].set(xlabel="Observed regime: 0 expansion, 1 normal, 2 stress", ylabel="PD / prevalence")
    axes[1].set(xlabel="Observed regime", ylabel="Brier score")
    axes[0].legend()
    fig.suptitle("Pooled diagnostic samples; repeated paired customers are not independent")
    fig.savefig(destination/"macro_regime.png", dpi=150)
    plt.close(fig)


def trajectory_figure(frame, destination):
    ids = list(frame.customer_id.unique())[:3]
    fig, axes = plt.subplots(len(ids), 2, figsize=(11, 3*len(ids)), squeeze=False, layout="constrained")
    for axes_row, identity in zip(axes, ids):
        group = frame[frame.customer_id == identity]
        axes_row[0].plot(group.month, group.predicted_pd, label="Forecast over H months")
        axes_row[0].plot(group.month, group.p_default_true, "--", label="Closing monthly hazard (diagnostic)")
        axes_row[0].set(title=identity, xlabel="Month", ylabel="Probability; different horizons")
        axes_row[1].plot(group.month, group.utilization, label="Utilization")
        axes_row[1].plot(group.month, group.payment_ratio, label="Payment ratio")
        axes_row[1].set(xlabel="Month", ylabel="Observed ratio")
        for ax in axes_row:
            ax.legend(fontsize=7)
    fig.suptitle("SIMULATOR-ONLY diagnostic: curves are not a calibration comparison")
    fig.savefig(destination, dpi=150)
    plt.close(fig)
