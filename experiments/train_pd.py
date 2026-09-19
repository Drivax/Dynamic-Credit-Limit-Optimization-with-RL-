"""Generate, fit, calibrate and evaluate the point-in-time longitudinal PD pipeline."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from credit_rl.config import DGP_VERSION, SimulationConfig
from credit_rl.risk.dataset import assert_split_integrity, build_dataset, sample_summary
from credit_rl.risk.evaluation import (calibration_table, cluster_intervals, distribution_summary,
                                      grouped_metrics, metrics)
from credit_rl.risk.features import FEATURE_NAMES, classify_column, feature_matrix, build_features
from credit_rl.risk.features import ALLOWED_AT_DECISION_TIME
from credit_rl.risk.generation import generate_cohort
from credit_rl.risk.longitudinal import LongitudinalPDModel, train_models
from credit_rl.risk.reporting import figures, trajectory_figure
from credit_rl.risk.settings import load_settings
from credit_rl.simulation.macro import MacroPath, MacroProcess
from .common import write_manifest


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def evaluate(output, settings):
    results, model_dir, fig_dir = (output/"results"/"pd", output/"models"/"pd", output/"figures"/"pd")
    models = {name: LongitudinalPDModel.load(model_dir/f"{name}.joblib") for name in
              ("logistic", "logistic_calibrated", "boosting", "boosting_calibrated")}
    metadata = json.loads((results/"metadata.json").read_text(encoding="utf-8"))
    baseline = metadata["training_prevalence"]
    comparisons, predictions, deciles, reliability, subgroups, periods, distributions = [], [], [], [], [], [], []
    samples = metadata["evaluation_samples"]
    for sample in samples:
        frame = pd.read_csv(results/f"dataset_{sample}.csv.gz")
        probabilities = {"constant": np.full(len(frame), baseline)}
        for name, model in models.items():
            # A later calibration cohort cannot validate an earlier period prospectively.
            if sample == "validation" and name.endswith("_calibrated"):
                continue
            probabilities[name] = model.predict_proba(feature_matrix(frame, model.feature_names))
        distributions.append(distribution_summary(frame, FEATURE_NAMES).assign(sample=sample))
        for name, p in probabilities.items():
            comparisons.append(dict(model=name, sample=sample, **metrics(frame.target, p),
                **cluster_intervals(frame, p, settings["evaluation"]["bootstrap_repetitions"], settings["seeds"]["bootstrap"])))
            view = frame[["customer_id", "month", "observation_month", "target", "macro_state", "utilization", "months_delinquent", "income"]].copy()
            view["predicted_pd"], view["model"], view["sample"] = p, name, sample
            predictions.append(view)
            deciles.append(calibration_table(frame.target, p, quantile=True).assign(model=name, sample=sample))
            reliability.append(calibration_table(frame.target, p).assign(model=name, sample=sample))
            periods.append(grouped_metrics(frame, p, "observation_month").assign(model=name, sample=sample))
            view["utilization_bucket"] = pd.cut(view.utilization, [0, .5, .9, 1., np.inf], include_lowest=True)
            view["income_bucket"] = pd.cut(view.income, [0, 2000, 4000, np.inf], include_lowest=True)
            view["delinquency_bucket"] = np.minimum(view.months_delinquent, 3)
            for column in ("macro_state", "utilization_bucket", "income_bucket", "delinquency_bucket"):
                table = grouped_metrics(view, p, column).rename(columns={column: "group"})
                table["group"] = table["group"].astype(str)
                subgroups.append(table.assign(group_type=column, sample=sample, model=name))
        print(f"Evaluated {sample}: {len(frame)} snapshots", flush=True)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(results/"metrics.csv", index=False)
    (results/"metrics.json").write_text(comparison.to_json(orient="records", indent=2), encoding="utf-8")
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(results/"predictions.csv.gz", index=False)
    for name, tables in (("risk_deciles", deciles), ("reliability", reliability), ("subgroups", subgroups),
                         ("metrics_by_month", periods), ("feature_distributions", distributions)):
        pd.concat(tables, ignore_index=True).to_csv(results/f"{name}.csv", index=False)
    figures(pred, comparison, fig_dir)
    return comparison


def run(config, settings, output):
    started = perf_counter()
    results, model_dir, fig_dir = (output/"results"/"pd", output/"models"/"pd", output/"figures"/"pd")
    for path in (results, model_dir, fig_dir):
        path.mkdir(parents=True, exist_ok=True)
    h, span = settings["target"]["horizon_months"], config.environment.horizon+1
    macro_process = MacroProcess(config.macro)
    # One shared calendar path, not independent macro draws per customer.
    calendar = macro_process.generate(span*5, np.random.default_rng(settings["seeds"]["macro"]))
    calendar.save(results/"macro_calendar.json")
    samples, histories, diagnostics, counts, audits, schema = {}, {}, {}, {}, [], {}
    cohort_names = ("train", "validation", "calibration", "test", "oot")
    for index, name in enumerate(cohort_names):
        start = index*span
        macro = MacroPath(calendar.states[start:start+span], f"calendar_{name}")
        raw, diag = generate_cohort(config, customers=settings["split"][name], cohort_index=index,
            cohort_start=start, seeds=settings["seeds"], macro_path=macro,
            action_probabilities=settings["behavior_policy"]["action_probabilities"])
        frame, audit = build_dataset(raw, h)
        frame["partition"] = name
        samples[name], histories[name], diagnostics[name] = frame, raw, diag
        audits.append(audit.assign(sample=name))
        counts[name] = {**sample_summary(frame), "generated_customers": settings["split"][name],
                        "realized_default_events": int(raw.defaulted.sum())}
        raw.to_csv(results/f"trajectories_{name}.csv.gz", index=False)
        frame.to_csv(results/f"dataset_{name}.csv.gz", index=False)
        schema.update({column: classify_column(column) for column in raw.columns})
        print(f"Generated {name}: {len(frame)} eligible snapshots", flush=True)
    assert_split_integrity(samples)
    for scenario, policy in (("baseline", "behavior"), ("mild_stress", "behavior"), ("severe_stress", "behavior"),
                              ("baseline", "static"), ("baseline", "always_increase"), ("baseline", "always_decrease")):
        name = f"{scenario}_{policy}"
        raw, diag = generate_cohort(config, customers=settings["split"]["oot"], cohort_index=4,
            cohort_start=4*span, seeds=settings["seeds"], macro_path=macro_process.scenario(scenario, span-1), policy=policy,
            action_probabilities=settings["behavior_policy"]["action_probabilities"])
        frame, audit = build_dataset(raw, h)
        frame["partition"] = name
        audits.append(audit.assign(sample=name))
        samples[name] = frame
        counts[name] = {**sample_summary(frame), "generated_customers": settings["split"]["oot"],
                        "realized_default_events": int(raw.defaulted.sum())}
        frame.to_csv(results/f"dataset_{name}.csv.gz", index=False)
        raw.to_csv(results/f"trajectories_{name}.csv.gz", index=False)
        print(f"Generated paired {name}", flush=True)
    pd.DataFrame(counts).T.rename_axis("sample").to_csv(results/"sample_counts.csv")
    pd.concat(audits).to_csv(results/"censoring_audit.csv", index=False)
    write_json(results/"column_classification.json", schema)
    metadata = dict(dgp_version=DGP_VERSION, simulation=asdict(config), configuration=settings,
        horizon_months=h, training_population="synthetic longitudinal entrants",
        behavior_policy=settings["behavior_policy"], training_scenario="shared Markov calendar",
        periods=counts, seeds=settings["seeds"], training_prevalence=float(samples["train"].target.mean()),
        evaluation_samples=[name for name in samples if name not in ("train", "calibration")],
        selected_model=settings["evaluation"]["selected_model"],
        selection="Predeclared configurable reference, no test-driven winner selection")
    models, training_seconds = train_models(samples["train"], samples["calibration"], settings, metadata)
    for name, model in models.items():
        path = model_dir/f"{name}.joblib"
        model.save(path)
        x = feature_matrix(samples["test"], model.feature_names)
        np.testing.assert_array_equal(model.predict_proba(x), LongitudinalPDModel.load(path).predict_proba(x))
        write_json(model_dir/f"{name}.metadata.json", model.metadata)
    write_json(results/"metadata.json", metadata)
    # Standardized logistic coefficients, including train-derived missing indicators.
    estimator = models["logistic"].estimator
    names = estimator[0].get_feature_names_out(models["logistic"].feature_names)
    pd.DataFrame({"feature": names, "coefficient_standardized": estimator[-1].coef_[0]}).to_csv(results/"logistic_coefficients.csv", index=False)
    distributions = distribution_summary(samples["train"], FEATURE_NAMES)
    distributions.to_csv(results/"train_feature_distributions.csv", index=False)
    selected = models[settings["evaluation"]["selected_model"]]
    x = feature_matrix(samples["test"], selected.feature_names).to_numpy()
    selected.predict_array(x[:1])
    start = perf_counter()
    for _ in range(200):
        selected.predict_array(x[:1])
    single = (perf_counter()-start)/200
    start = perf_counter()
    selected.predict_array(x)
    batch = perf_counter()-start
    benchmark = dict(training_seconds=training_seconds, single_feature_row_ms=single*1000,
        batch_rows=len(x), batch_seconds=batch, batch_rows_per_second=len(x)/batch)
    prefix = histories["test"].iloc[:1][list(ALLOWED_AT_DECISION_TIME)].to_dict("records")
    online = {}
    for name, model in models.items():
        model.predict_history(prefix)
        start = perf_counter()
        for _ in range(100):
            model.predict_history(prefix)
        online[name] = dict(history_prediction_ms=1000*(perf_counter()-start)/100)
        start = perf_counter()
        model.predict_proba(feature_matrix(samples["test"], model.feature_names))
        online[name]["batch_rows_per_second"] = len(x)/(perf_counter()-start)
    benchmark["online_models"] = online
    # Save manual audit material including full prefixes and actual future event rows.
    raw = histories["test"]
    ids = raw.customer_id.unique()[:3]
    representatives = raw[raw.customer_id.isin(ids)].copy()
    f = build_features(representatives)
    representatives["predicted_pd"] = selected.predict_proba(feature_matrix(f, selected.feature_names))
    representatives = representatives.merge(diagnostics["test"], on=["customer_id", "month"], how="left")
    representatives.to_csv(results/"representative_trajectories_DIAGNOSTIC_ONLY.csv", index=False)
    trajectory_figure(representatives, fig_dir/"representative_trajectories.png")
    snapshots = samples["test"][samples["test"].customer_id.isin(ids)].copy()
    snapshots["predicted_pd"] = selected.predict_proba(feature_matrix(snapshots, selected.feature_names))
    snapshots.to_csv(results/"snapshot_audit.csv", index=False)
    comparison = evaluate(output, settings)
    benchmark["pipeline_seconds"] = perf_counter()-started
    write_json(results/"benchmark.json", benchmark)
    write_manifest(results/"manifest.json", config, settings, benchmark=benchmark,
        target="first default in (t,t+H], active at t", protocol="disjoint chronological cohorts; mature labels; separate calibration")
    print(comparison[["model", "sample", "roc_auc", "pr_auc", "brier", "log_loss"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pd_model.yaml"))
    parser.add_argument("--simulation", type=Path, default=Path("configs/simulation.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    simulation = SimulationConfig.from_yaml(args.simulation)
    settings = load_settings(args.config, simulation)
    with threadpool_limits(limits=1):
        if args.evaluate_only:
            evaluate(args.output, settings)
        else:
            run(simulation, settings, args.output)


if __name__ == "__main__":
    main()
