# Repository audit for longitudinal PD modeling

The source, configuration, tests, experiment entry points, documentation and output
inventory were inspected recursively before substantive changes. No notebooks were
present outside the environment. The working tree was clean. The local `.venv`
provides Python 3.12 and the existing sklearn/joblib/matplotlib experiment dependencies.

## Relevant initial state

- `simulation/customer.py` separates immutable observable state from persistent
  creditworthiness, spending/payment propensities and income stability.
- `CreditDGP.step` applies the action, realizes income/payment/spending, updates
  delinquency and score, computes the hidden conditional hazard, then draws default.
  It uses the opening macro state; the next macro state is revealed for the next decision.
- Episodes last at most 24 monthly transitions, ending on first default. Default is
  a Bernoulli event under the structural hazard, not a deterministic 90-DPD definition.
- `get_history` excludes privileged quantities. Diagnostic exports include hidden
  hazard, shocks, income events and latent traits and must remain outside modeling.
- `risk/training.py` fits a GradientBoostingClassifier to independent synthetic
  snapshots with `default_next_month`; its split is row-based because those data have
  one row per independent snapshot. It is not a valid longitudinal validation protocol.
- `SnapshotPDModel` maps observed state to this snapshot model. `ObservedLogisticPD`
  is an assumed-coefficient fallback, not a fitted or calibrated horizon PD.
- Existing PPO/trajectory scripts trained the snapshot classifier internally.
- Core tests already covered absorbing default, accounting, hidden-state isolation,
  common random numbers, macro/shock timing, reproducibility and Gymnasium compatibility.
- Existing result/model/figure folders contain DGP, PPO and snapshot diagnostics;
  they are not evidence for an H-month longitudinal estimator.

## Leakage inventory

Searches included `p_default_true`, `latent_*`, `future_*`, `next_*`, `true_pd`,
`shift`, `rolling`, numeric selection and splitting code.

| Quantity | Treatment |
|---|---|
| `p_default_true`, `true_pd`, `true_default_probability` | Simulator-only; forbidden as predictors |
| Creditworthiness, spending/payment propensities, income stability | Hidden traits; forbidden |
| `shock_*`, adverse-income event, spending elasticity | Privileged realized diagnostics; forbidden |
| `future_*`, `next_*` | Forbidden even when numeric |
| `defaulted`, `realized_default`, `default_next_month`, horizon target | Label/eligibility only |
| Full macro path and future actions | Generation only; never predictors |
| `predicted_pd`, decision PD, action/reward/exposure columns | Excluded to avoid circular predictions and ambiguous timing |
| Customer identity, month, cohort/split/end dates | Indexing/labeling/splitting only |
| Initial income/score anchors | Excluded from features; internal transition bookkeeping |
| Current score, income, balance, payment, utilization, current macro factors | Observable at closing t, before decision t→t+1 |

The synthetic initializer still computes snapshot labels internally; initialization
selects only explicit initial observed fields. Those labels are neither trajectory
labels nor risk features. Historical archived workflows remain separate.

No DGP coefficients were changed to improve predictive results. New numeric columns
cannot enter X implicitly: all training and scoring use a reviewed feature schema,
and dataframe scoring rejects additional or reordered columns.
