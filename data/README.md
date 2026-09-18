# Data provenance and local files

All customer data found in the original repository was synthetic. The generator
uses NumPy distributions and assumed regional macro values; it does not load a
public banking dataset, private customer data, names, addresses or account numbers.
Geographical labels and ages are synthetic attributes, not identified individuals.

The original local `artifacts/sample_clients.csv` is preserved at
`outputs/legacy/artifacts/sample_clients.csv` along with old model files. It was
ignored by Git and remains ignored. No original dataset was deleted or duplicated
as part of the migration. Historical tracked prediction/decision CSVs now live in
`outputs/legacy/results/` and remain available for versioning.

New experiments generate their initial populations in memory using
`credit_rl.simulation.synthetic_snapshot.generate_synthetic_portfolio` (the preserved
generator), and save diagnostic trajectories under `outputs/results/`. Those outputs
are ignored by default. No input downloads are required. The old `true_pd` and
`default_next_month` columns are used for snapshot model training only, never as
longitudinal transition inputs or agent features.

No raw/processed datasets are required currently, so no empty directories are added.
If future work adds external inputs under `data/raw/` or `data/processed/`, both are
ignored by default; record source, license, consent and permitted uses before sharing.
