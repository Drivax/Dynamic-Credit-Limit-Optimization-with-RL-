# Environment API

`CreditLimitEnv` follows one customer for up to 24 monthly transitions. Actions are configured limit multipliers, subject to monthly and absolute bounds. Default terminates; survival to the horizon truncates. Calling step after either raises.

`reset(seed=..., options=...)` accepts a portfolio `customer_index` or an explicit `initial_state` and hidden `traits` pair for controlled research. Optional immutable `macro_path` and `shock_path` inputs support paired experiments. These inputs are simulation setup, never policy observations. Unknown options and inconsistent paths fail.

The 21 float32 observation entries are ordered by `OBSERVATION_NAMES`: elapsed month, limit, balance, utilization, payment ratio, delinquency indicator/count, income, behavioral score, macro credit stress, predicted PD, latest spending, late-payment fraction, tenure, terminal default indicator, income log change, macro income/spending growth and expansion/normal/stress indicators. Ratios are bounded as x/(1+x); signed factors are rescaled. No hidden trait, hazard, future shock or future macro is included.

Pass a trusted `LongitudinalPDModel.load(path)` as `pd_model`. The environment stores a separate seven-row observable history, reset for each episode. It calls the stateless `predict_history` interface with only those fields, before the next action. Array preprocessing avoids Pandas in each step. The model can be shared safely between multiple environments because history lives in the environment, not the model.

`info.decision_pd` is the opening forecast used for the completed transition's reward; `info.predicted_pd` is the closing forecast for the next decision. A terminal default has sentinel PD=1 with the longitudinal model and is never an active decision. With no artifact, `ObservedLogisticPD` is the explicitly uncalibrated fallback proxy. It does not have the longitudinal model's 12-month forecast interpretation.

`record_history=False` disables full exported history, but preserves the bounded risk feature history. `record_diagnostics=True` enables researcher-only true hazard, realized shocks and transition diagnostics. Hidden traits are a separate export. These Python APIs express an information contract, not an introspection security sandbox.

Default is generated solely by `CreditDGP`. The supplied estimator affects policy decisions and economic reward proxies but is not passed to the transition mechanism. Principal obeys B'=B-P+C. A limit cut never forgives debt; credit loss is charged once.

See [PD methodology](pd_model.md), [DGP specification](dgp.md) and [measured results](pd_results.md). The executable integration check is `python -m experiments.pd_env_smoke` after the PD training pipeline.
