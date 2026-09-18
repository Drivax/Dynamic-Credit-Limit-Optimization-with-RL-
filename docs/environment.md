# Environment API — DGP 2.0

The complete structural specification, equations, causal DAG, macro assumptions and
information boundaries are in [dgp.md](dgp.md). The historical Sprint 1 specification
is retained in [environment_v1.md](environment_v1.md).

`CreditLimitEnv` follows one customer for up to 24 monthly transitions. Actions are
multipliers `[0.8,0.9,1,1.1,1.2]` applied to current limit, with monthly change caps
and absolute bounds. Default terminates; surviving the horizon truncates. Calling
step after either event raises. Default loss is charged once.

`reset(seed=..., options=...)` accepts either a portfolio `customer_index` or an
explicit `initial_state` and `traits`. Optional `macro_path` and `shock_path` provide
exact paired experiments; lengths and customer identity are validated. Old v1
`macro_state` reset options are replaced by `MacroPath.constant(...)`.

The float32 Box observation has 21 entries, ordered by `OBSERVATION_NAMES`:
month fraction, limit, balance, utilization, payment ratio, delinquency indicator,
consecutive delinquency count, income, behavioral score, current credit stress,
imperfect predicted PD, latest spending, recent late-payment fraction, tenure,
terminal default indicator, observed income log change, current macro income growth,
current macro spending growth, expansion/normal/stress one-hot indicators.
Ratios use x/(1+x); signed factors are rescaled to [0,1]. See `envs/observation.py`
for exact transforms. No latent trait, future shock, future macro or true hazard is
included. Previous 11- and 15-feature model checkpoints are incompatible.

`info` includes realized reward components and requested/effective limit changes.
`get_history()` exports policy-safe realized state history. `record_history=False`
avoids recording overhead. `record_diagnostics=True` explicitly enables privileged
`get_diagnostics()` exports of true hazard, realized shocks and transition details;
`get_latent_diagnostics()` exposes traits separately for research. These interfaces
are not a security boundary against arbitrary Python introspection.

Principal evolves as B'=B-P+C; interest and fees are economic reward proxies and
are not capitalized. Limit cuts never erase debt. Income is stochastic, delinquency
can persist/cure/worsen, and macro paths are exogenous. A pure hidden DGP generates
default independently of the observable PD estimator. The preserved snapshot GB
adapter uses current observable features and array inference; its old snapshot
labels are not longitudinal calibration evidence.

Run `python -m pytest` and `python -m experiments.dgp_sanity --customers 5000 --seed 42`.
See [sprint2_report.md](sprint2_report.md) for measured outcomes and limitations.
