# Robust policy evaluation under model uncertainty

## Audit and uncertainty contract

The audited package already separates hidden DGP dynamics, public observations,
point-in-time PD estimation and policy decisions. A common evaluator and indexed
shock streams support paired comparisons. The nominal benchmark found a constant
PPO contraction policy, not a demonstrated planning advantage. Its uncertainty
intervals conditioned on a single DGP and macro path; risk alerts were ex post.
The PD modeling behavior policy samples actions with probabilities
`[0.1, 0.2, 0.4, 0.2, 0.1]`; forecasts can shift under other policies.

The robustness study reuses the evaluator rather than adding a separate simulator.
It preserves the reward, default law's functional form and nominal trained artifacts.
It distinguishes:

| Source | Treatment |
|---|---|
| Aleatoric randomness | Customer/month/channel-indexed income, spending, payment and default shocks |
| Customer heterogeneity | Persistent latent traits drawn once from the nominal initial population |
| PD misspecification | Frozen predictive model; actor-only score distortions and operational lag |
| Environment uncertainty | Declared distributions over behavioral and hazard coefficients |
| Macro uncertainty | Shared stress onset, duration, intensity and recovery paths |
| Policy-training uncertainty | All five saved PPO seeds with and without explicit PD |

The worlds vary post-initialization dynamics. Initial population and latent-trait
distribution are held fixed; this does not explore every possible population shift.

## Prespecified worlds

`configs/robustness.yaml` defines ranges **before standard evaluation**. Independent
uniform draws within the following absolute ranges are transparent synthetic
assumptions, not empirically estimated priors or plausible-region guarantees.

| Parameter | Nominal | Range | Interpretation |
|---|---:|---:|---|
| Spending-limit elasticity | 0.15 | 0.05–0.30 | Weak to stronger purchase response; heterogeneous loading/cap unchanged |
| Monthly income sigma | 0.025 | 0.018–0.040 | Moderate to higher income volatility |
| Payment persistence | 0.50 | 0.25–0.75 | Convex memory of the previous payment ratio |
| Missed-payment delinquency coefficient | 0.60 | 0.35–0.90 | Persistence of missed payments, preserving adverse sign |
| Payment/income cap | 0.50 | 0.40–0.60 | Affordability cap on principal repayment |
| Default intercept | −6.0 | −6.35–−5.65 | Baseline hazard odds multiply by about 0.70–1.42 |
| Default utilization coefficient | 1.20 | 0.85–1.65 | Exposure relative to available credit |
| Default delinquency coefficient | 0.65 | 0.45–0.85 | Consecutive delinquency remains adverse |
| Default debt/income coefficient | 0.45 | 0.30–0.65 | Debt burden sensitivity |
| Default macro-stress coefficient | 0.90 | 0.60–1.30 | Macro sensitivity of the hidden hazard |

The nominal world retains normal macro. Random worlds independently sample stress
onset in decisions 4–12, duration 3–10, recovery 2–6 (inclusive integer uniforms),
and intensity 0.5–1.8. A linear decline over recovery precedes normal conditions.
Paths are clipped by the existing 24-decision horizon, never extended into future
observations. Every path is validated by the existing `MacroState` bounds.

Standard uses **32 randomized worlds and 64 fresh customers**. It also runs 20
one-at-a-time endpoint worlds around a common stress anchor (onset 6, duration 8,
recovery 4, intensity 1), that anchor itself, nominal, and five macro-only cases:
early onset 3; late onset 12; duration 14; recovery 8; unexpected severe stress
after 12 stable decisions (duration 8, intensity 2, recovery 2). These diagnostic
cases are not pooled into the randomized-world distribution.

Nominal, random_000 and the unexpected-shock world are prespecified sensor anchors;
none is chosen because of its results. Eight nonidentity sensor variants run on
these three worlds. Standard therefore has 59 physical worlds and 83 world/signal
jobs. Smoke and full profiles are explicitly configured, not run inside unit tests.

## Frozen policy and information boundary

All deployable rules retain their **nominal** configuration and validation-selected
thresholds. In particular, MyopicEconomic does not receive shifted true elasticities
or risk coefficients. The environment alone receives each evaluation-world config.
The separately labeled SIMULATOR-ONLY ORACLE sees the true world through its existing
privileged adapter; it is neither deployable nor an optimal value bound.

The same 64 observed initial states, hidden traits and exogenous shocks are reused
across all policies and worlds. `ROBUST_TEST` IDs are distinct from PD and RL
train/validation/test IDs. Every result records world ID, seed, signal, policy seed
and information set. Worker processes load frozen models; no training is invoked.

## Actor-only information shifts

For current forecast p, the miscalibration/noise transform is

`p_supplied = sigmoid(a + b*logit(clip(p, 1e-7, 1-1e-7)) + sigma*epsilon)`.

Cases: intercept a=−0.7 or +0.7; slope b=0.65 or 1.35; Gaussian logit noise sigma=0.3
or 0.8. Unspecified parameters use identity defaults. Exact identity preserves
endpoints without numerical clipping. Noise uses a customer-specific RNG independent
of DGP shocks and shared across policies. This is exogenous measurement noise, not
re-estimation of the PD model.

One-month lag returns the previous decision's forecast, using current PD only at
initialization. It never reads a future forecast. Reporting error multiplies income
and balance by mean-one lognormal noise (sigma=0.05); utilization shares the balance
error, preserving balance/limit consistency. The actor's 21-vector is transformed
through the existing bounded representation. Other features remain untouched.

Transforms occur after the true public observation is generated. Neither the DGP,
PD estimator, reward PD nor risk-metric forecasts change directly. No-PD PPO masks
the coordinate **after** the sensor transform, so noise cannot reintroduce the
ablated signal. Resulting actions may of course change later states and forecasts.

## Metrics, risk and uncertainty

Economic value, reward components, exposure, delinquency, requested/effective actions,
limit changes and direction reversals reuse the documented nominal denominators.
For each risk metric x and threshold c, excess is `max(0,x-c)`. We report fraction
of worlds with excess >0, mean excess over all worlds, mean conditional on violation
and maximum excess. Thresholds remain the declared synthetic defaults (0.75 default
incidence, 0.70 losses/initial balance, 0.60 high-risk exposure share).

Performance distributions average training seeds within each world, then report
mean, median, sample standard deviation, p10/p25/p75/p90 and worst observed world.
Worst means minimum value/reward or maximum risk. It is not an adversarial optimum.
Rank comparisons report the empirical frequency of positive paired world-level value
differences and ties separately, not a universal probability of superiority.

Because the same customers and shocks cross all worlds, uncertainty uses a
**crossed world × customer × training-seed bootstrap** (400 replicates): resample
worlds, resample customer identities once shared across worlds, and resample policy
seeds. Paired differences are formed before resampling. Customer-month rows are never
independent bootstrap units. Conditional intervals hold the sampled worlds fixed.
Between-world SD of mean value, within-world customer SD and training-seed SD are
descriptive quantities, not an identified causal/additive variance decomposition.

The empirical value/default frontier uses available policies only. Point dominance
requires no lower value and no higher default incidence with at least one practical
improvement (EUR 1/client or 0.5 percentage points). Interval-supported dominance
also requires paired value CI lower >=0 and default CI upper <=0. These exploratory
comparisons have no multiple-comparison familywise guarantee.

## Execution and provenance

`python -m experiments.robustness --profile standard --stage all` loads saved nominal
models, fixes worlds, evaluates paired jobs and generates statistics/figures. Separate
`evaluate` and `report` stages support inspection. Three spawned Windows-compatible
processes evaluate independent jobs, each using one Torch/BLAS thread. Immutable seeds
do not depend on completion order. Job completion metadata is written last; completed
jobs can be reused only with the same experiment fingerprint.

`outputs/experiments/robustness_standard` stores timestamps, commit, exact YAML copies,
configuration, world definitions, PD/model hashes and source hashes. Detailed episodes,
timings, actions, monthly outcomes and representative full traces are under
`outputs/results/robustness/standard`; figures are under the matching figures directory.

Limitations: uncalibrated synthetic ranges and independent parameter draws; limited
world/customer counts; fixed initialization distribution and economic assumptions;
finite-horizon default-heavy portfolios; no regulatory feasibility guarantee;
extrapolation of frozen PD and surrogate myopic dynamics; Monte Carlo error and no
evidence of real-world banking robustness. No ranges are retuned after final outcomes.
