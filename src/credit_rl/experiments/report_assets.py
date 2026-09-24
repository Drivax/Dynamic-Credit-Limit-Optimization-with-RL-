"""Figures and Markdown tables derived solely from canonical measured outputs."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown(frame):
    def cell(value):
        return f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def report(output):
    output = Path(output)
    results, figures = output / "results", output / "figures"
    figures.mkdir(exist_ok=True)
    summary = pd.read_csv(results / "summary.csv")
    paired = pd.read_csv(results / "paired_comparisons.csv")
    pd_metrics = pd.read_csv(results / "pd/metrics.csv")
    from credit_rl.risk.reporting import figures as pd_figures, trajectory_figure
    pd_figures(pd.read_csv(results / "pd/predictions.csv.gz"), pd_metrics, figures / "pd")
    trajectory_figure(pd.read_csv(results / "pd/representative_trajectories_DIAGNOSTIC_ONLY.csv"),
                      figures / "pd/representative_trajectories.png")
    columns = ["policy", "net_economic_value", "revenue", "credit_loss", "default_rate", "mean_limit", "mean_utilization"]
    sections = []
    for scenario, name in (("baseline", "Baseline"), ("severe_stress", "Stress")):
        frame = summary[summary.scenario == scenario][columns]
        frame.to_csv(results / f"{scenario}_results.csv", index=False)
        sections.append(f"### {name}\n\n" + markdown(frame))
    comparisons = paired[(paired.policy == "PPO") & (paired.reference.isin(["MyopicEconomic", "Static"]))]
    sections.append("### Paired PPO differences (95% bootstrap)\n\n" + markdown(
        comparisons[["scenario", "reference", "metric", "difference", "lower", "upper"]]))
    sections.append("### PD test performance\n\n" + markdown(pd_metrics[pd_metrics["sample"] == "test"]
        [["model", "roc_auc", "pr_auc", "brier", "log_loss"]]))
    (results / "tables.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for scenario, ax in zip(("baseline", "severe_stress"), axes):
        s = summary[summary.scenario == scenario]
        for index, row in enumerate(s.itertuples()):
            ax.scatter(row.default_rate, row.net_economic_value, label=row.policy,
                       marker=["o", "s", "^", "x", "+"][index], s=55)
        ax.set(xlabel="Default fraction / initial customers", ylabel="Net value (EUR / customer)", title=scenario)
        ax.legend(fontsize=8, loc="best")
    fig.savefig(figures / "risk_value.png", dpi=150)
    plt.close(fig)
    pivot = summary.pivot(index="policy", columns="scenario", values="net_economic_value")
    ax = pivot.plot.bar(figsize=(9, 5), rot=15, ylabel="Net value (EUR / customer)", title="Paired macro scenarios")
    ax.figure.tight_layout()
    ax.figure.savefig(figures / "baseline_stress.png", dpi=150)
    plt.close(ax.figure)
    histories = pd.read_csv(results / "trajectories.csv.gz")
    # Representative chosen by ID, never by economic outcome.
    identity = sorted(histories.customer_id.unique())[0]
    selected = histories[(histories.customer_id == identity) & (histories.scenario == "baseline")]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for (policy, seed), frame in selected.groupby(["policy", "policy_seed"]):
        frame = frame.sort_values("month")
        label = f"{policy} {seed}" if policy == "PPO" else policy
        axes[0].plot(frame.month, frame.credit_limit, label=label)
        axes[1].plot(frame.month, frame.reward.cumsum(), label=label)
    axes[0].set(ylabel="Limit (EUR)", title=f"Paired customer {identity}; all PPO seeds")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Month", ylabel="Cumulative training reward (EUR)")
    fig.savefig(figures / "paired_trajectory.png", dpi=150)
    plt.close(fig)
    ppo = histories[(histories.policy == "PPO") & (histories.month > 0)].copy()
    ppo["pd_bin"] = pd.cut(ppo.actor_pd, np.linspace(0, 1, 6))
    ppo["util_bin"] = pd.cut(ppo.opening_utilization, [0, .25, .5, .75, 1, np.inf], include_lowest=True)
    grid = ppo.groupby(["pd_bin", "util_bin"], observed=False).requested_action.mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    im = ax.imshow(grid.to_numpy(), origin="lower", vmin=-.2, vmax=.2, cmap="coolwarm")
    ax.set(xticks=range(len(grid.columns)), xticklabels=[str(x) for x in grid.columns],
           yticks=range(len(grid.index)), yticklabels=[str(x) for x in grid.index],
           xlabel="Opening utilization bin", ylabel="Actor PD bin",
           title="Observed PPO requested changes; all seeds and scenarios")
    fig.colorbar(im, ax=ax, label="Mean requested fractional limit change")
    ax.tick_params(axis="x", labelrotation=20)
    fig.savefig(figures / "ppo_policy_map.png", dpi=150)
    plt.close(fig)
    # Calibration is already regenerated by the PD reporting pipeline.
    metadata = dict(representative_customer=identity, policy_map="empirical bins; empty cells missing; descriptive only")
    (results / "figure_metadata.json").write_text(json.dumps(metadata, indent=2))


def update_docs(output, docs):
    fragment = (Path(output) / "results/tables.md").read_text(encoding="utf-8")
    start, end = "<!-- canonical-results:start -->", "<!-- canonical-results:end -->"
    for filename in docs:
        path = Path(filename)
        text = path.read_text(encoding="utf-8")
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"Expected one result block in {path}")
        before, rest = text.split(start)
        _, after = rest.split(end)
        path.write_text(before + start + "\n\n" + fragment + "\n" + end + after, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/main/standard"))
    parser.add_argument("--update-docs", action="store_true")
    args = parser.parse_args()
    report(args.output)
    if args.update_docs:
        update_docs(args.output, ["README.md", "docs/technical_paper.md"])


if __name__ == "__main__":
    main()
