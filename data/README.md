# Data provenance and local files

All customer data found in the original repository was synthetic. The generator
uses NumPy distributions and assumed regional macro values; it does not load a
public banking dataset, private customer data, names, addresses or account numbers.
Geographical labels and ages are synthetic attributes, not identified individuals.

No external dataset or local model artifact is required. Superseded row-based
prototype data, models and reports were removed after their role was audited; they
remain recoverable from Git history.

New experiments generate their initial populations in memory using
`credit_rl.simulation.synthetic_snapshot.generate_synthetic_portfolio` (the preserved
generator), and save diagnostic trajectories under `outputs/results/`. Those outputs
are ignored by default. No input downloads are required. The old `true_pd` and
`default_next_month` columns are used for snapshot model training only, never as
longitudinal transition inputs or agent features.

No raw/processed datasets are required currently, so no empty directories are added.
If future work adds external inputs under `data/raw/` or `data/processed/`, both are
ignored by default; record source, license, consent and permitted uses before sharing.

## Longitudinal diagnostic data

`experiments.dgp_sanity` retains the original synthetic snapshot generator, discards
its labels when initializing the environment, and generates longitudinal outcomes
with DGP 2.0. `observed_learning_check.csv.gz` uses opening observations and a
next-month default label, with customer-level train/test separation. Do not join
`DIAGNOSTIC_ONLY` histories or `initial_customers.json` into training features:
they contain hidden types, simulator hazards, and realized within-month shocks.
Macro and shock paths are reusable for controlled paired experiments. All exported
data remain synthetic and uncalibrated; see `docs/dgp.md`.
