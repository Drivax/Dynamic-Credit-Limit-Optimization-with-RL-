"""Clean-artifact integration and reproducibility, including actual PPO fitting."""
import json

import numpy as np
import pandas as pd
import pytest

from credit_rl.experiments.main_evaluation import run, verify
from credit_rl.experiments.report_assets import update_docs
from credit_rl.evaluation.policy_engine import episode_metrics, aggregate_episodes


def test_end_to_end_fresh_runs_and_frozen_replay(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run(output=first)
    run(output=second)
    for filename in ("summary.csv", "episode_metrics.csv", "paired_comparisons.csv", "pd/metrics.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(first / "results" / filename),
                                      pd.read_csv(second / "results" / filename), check_exact=True)
    before = pd.read_csv(first / "results/summary.csv")
    run(output=first, stage="evaluate")
    pd.testing.assert_frame_equal(before, verify(first), check_exact=True)
    assert (first / "figures/ppo_policy_map.png").stat().st_size > 1000
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["run"]["policy"]["ppo"]["seeds"] == [101]
    assert manifest["package_sha256"] and manifest["timestamp"]
    doc = tmp_path / "report.md"
    doc.write_text("Intro\n<!-- canonical-results:start -->\nstale\n<!-- canonical-results:end -->\nEnd")
    update_docs(first, [doc])
    assert "stale" not in doc.read_text() and "PD test performance" in doc.read_text()
    corrupted = before.copy()
    corrupted.loc[0, "net_economic_value"] = np.inf
    corrupted.to_csv(first / "results/summary.csv", index=False)
    with pytest.raises(ValueError, match="Nonfinite"):
        verify(first)


def test_manual_episode_accounting_and_population_denominators():
    # Two observed months; all money totals can be checked without simulation.
    rows = []
    for month in range(3):
        rows.append(dict(customer_id="a", month=month, balance=100., credit_limit=200.,
            utilization=.5, reward_interest_income=10., reward_fee_income=2.,
            reward_credit_loss=30. if month == 2 else 0., reward_funding_cost=1.,
            reward_capital_cost=2., reward_constraint_penalty=3., reward=-24. if month == 2 else 6.,
            defaulted=month == 2, months_delinquent=0, effective_action=0., requested_action=0.,
            decision_pd=.2, exposure=100., guardrail_blocked=False, opening_delinquency=0))
    settings = {"ppo": {"gamma": .5}, "risk_constraints": {"high_pd": .6},
                "guardrails": {"severe_delinquency_months": 3}}
    e = episode_metrics(pd.DataFrame(rows), settings)
    assert e["revenue"] == 24 and e["credit_loss"] == 30
    assert e["net_economic_value"] == -8 and e["cumulative_reward"] == -18
    assert e["discounted_reward"] == -6
    assert e["mean_utilization"] == .5 and e["mean_limit"] == 200
    survivor = {**e, "customer_id": "b", "defaulted": 0, "credit_loss": 0}
    totals = aggregate_episodes(pd.DataFrame([e, survivor]))
    assert totals["default_rate"] == .5 and totals["credit_loss_rate"] == .15
