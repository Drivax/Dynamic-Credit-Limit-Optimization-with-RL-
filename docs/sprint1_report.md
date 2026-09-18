# Sprint 1 technical report

Implemented locally: a longitudinal customer environment, a packaged research
workflow, preservation of historical work, meaningful tests and executed diagnostics.
No simulator/reward tuning to favor PPO, full PPO optimization, cloud/API/frontend,
constrained RL or history rewrite was performed.

## A. Repository audit

The old environment stepped through independent dataset rows; actions never updated
the next state of the same customer. Defaults did not terminate customer episodes.
The reward combined weighted expected credit loss with realized credit loss. The
README described sequential behavior that did not exist. Evaluation had unused
stratification, reversed risk labels, unordered synthetic rows called time series,
prefit-risk-model contamination across some folds and test-set HPO over different
reward definitions. Diagnostic scripts contained duplicated and incomplete code.
There were no tests, notebooks, packaging metadata or external raw datasets.

See [repository_audit.md](repository_audit.md) for file classification and findings.
`pre_migration_inventory.csv` contains original hashes; `migration_verification.json`
confirms **48 original files preserved byte-for-byte**, with only nine disposable
cache/IDE files removed (plus the intentionally updated ignore configuration).

## B. File operations

| Operation | Important paths |
|---|---|
| CREATED | `pyproject.toml`, `configs/{simulation,experiments}.yaml` |
| CREATED | `src/credit_rl/{config,reward}.py`, `envs/credit_limit_env.py` |
| CREATED | `simulation/{customer,dynamics,simulator}.py`, `risk/pd_model.py` |
| CREATED | `policies/baselines.py`, `evaluation/metrics.py`, `utils/seeding.py`, package namespaces |
| CREATED / REUSED | `simulation/synthetic_snapshot.py` and `risk/training.py`, directly preserving useful generator/trainer implementations |
| CREATED | `experiments/{common,trajectory_sanity,train_ppo}.py`, experiment/archive documentation |
| CREATED | Four test modules plus shared fixtures; `data/README.md`; environment, audit and sprint documentation |
| MODIFIED | Root `README.md` replaced by scientifically qualified current documentation; `.gitignore` expanded |
| MOVED | Original `credit_limit_rl/`, training/retraining/diagnostic scripts, `app.py`, old README/requirements/action-collapse notes to `experiments/legacy/` |
| MOVED | All original `results/` to `outputs/legacy/results/`; ignored `artifacts/` to `outputs/legacy/artifacts/` |
| DELETED | Original root/package bytecode caches and `.vscode/settings.json`; no original dataset, model or result |
| UNTRACKED | 8,521 `.venv` files captured by an intervening local commit removed from Git's index; local environment preserved |

New results, figures and models are generated under `outputs/` and ignored. Historical
tracked results remain tracked. New source/documentation is left locally reviewable;
the `.venv` index removals are staged. This task did not create a commit or push.

## C. Architecture

`credit_rl` uses a proper `src/` layout and editable installation. `config` contains
typed, validated DGP/economic parameters. `simulation` owns state/traits/transitions
and trajectory collection. `envs` owns Gymnasium lifecycle, observed information and
history. `risk` owns an observed-feature interface, fallback estimate, GB adapter and
preserved trainer. `reward` exposes named economic components. `policies` provides
static, constant-action, risk-threshold and SB3 adapters. `evaluation` summarizes
episode return/default incidence. `utils` separates random streams.

Core dependencies are NumPy, Pandas, Gymnasium and PyYAML. Experiment extras add
sklearn/plotting/joblib; RL adds SB3; development adds pytest. The old Streamlit,
Plotly and Optuna dependencies remain only in the frozen archival requirements.

## D. Longitudinal environment

S_t is an immutable CustomerState snapshot owned by a mutable environment, plus
separately held persistent CustomerTraits. State includes identity/month, limit,
balance, payment ratio, delinquency duration/history, income, behavioral score,
tenure, spend, macro and default status. Utilization is derived, avoiding stale copies.

A_t selects one of [0.8,0.9,1,1.1,1.2]. L_(t+1)=clip(L_t*A_t,500,15000). The
transition calculates payment, spending, new balance, delinquency, score and default,
then replaces the state for the **same customer**. An action therefore changes future
available credit, exposure and behavior. `step` never reads the portfolio.

An episode lasts up to 24 transitions. Default terminates; surviving to the horizon
truncates. Post-terminal calls fail instead of generating activity. Observations match
a named 15-dimensional float32 [0,1] space. Histories include initialization and full
monthly transition outcomes, with action/reward attached to the resulting state row.

## E. Data-generating process

- Three traits sampled once: hidden creditworthiness (partly related to observed
  score), spending propensity and payment propensity.
- Payment mixes last month's payment fraction with a logistic response to traits,
  post-action utilization, debt/income, prior delinquency, current macro and shocks.
  Missed-payment probability also depends on persistent delinquency and traits.
- Delinquency increments on insufficient payment and resets on cure; six-month
  late-payment history rolls forward. Behavioral score mean-reverts with payment,
  delinquency and noise effects.
- Spending demand combines previous spending, income, propensity, a limit-change
  elasticity, macro and lognormal shocks. Available credit caps actual purchases.
  B'=B-payment+spend. Limit cuts may leave existing debt above the limit; it is never
  forgiven or increased through purchases while no headroom exists.
- Macro is a two-state Markov chain with 4% monthly entry to stress and 20% exit,
  sampled independently of policy actions using a separate RNG stream.
- Hidden monthly default probability is a logistic function of closing utilization,
  debt burden, delinquency, payment weakness, latent creditworthiness and current
  macro. A separate uniform draw realizes default. Zero closing debt has zero default
  probability. Default is terminal and closing principal is loss exposure.

All principal behavioral coefficients are centralized. Full formulas and ordering
are in [environment.md](environment.md). They are assumptions, not calibrated banking
relationships. Income is currently fixed, and there is no complete arrears ledger.

## F. PD separation and leakage checks

The hidden DGP takes CustomerState and CustomerTraits; it has no PD-model dependency.
The predictor receives only ObservedRiskFeatures. Snapshot targets/true_pd are stripped
at ingestion. Neither observations, info nor trajectory history contains simulator
probabilities, latent traits or future shocks. Controlled experiment manifests may
record traits for reproducibility; policies never receive those manifests.

The default `ObservedLogisticPD` is an imperfect hand-specified proxy. Executed sanity
and PPO runs use the retained GradientBoosting trainer through SnapshotPDModel, with
observed features refreshed monthly. Its old snapshot targets come from a different
equation; it is not trained on defaults generated from its own predictions.

Tests change all predicted PDs from 0 to 1 under identical actions/seeds and verify
identical state/default paths. Other tests poison future labels and change latent
traits while holding the initial observation fixed. These verify the direct data
boundary, not empirical risk-model accuracy. Longitudinal PD calibration remains open.

## G. Reward

For opening B, closing E=B', purchases W', opening predicted PD p and default I:

| Component | Formula |
|---|---|
| Interest | (1-I) * B * 0.18/12 |
| Fees | W' * 0.012 |
| Realized credit loss | I * 0.55 * E |
| Funding | E * 0.03/12 |
| Capital proxy | 1.5 * 0.08 * p * E |
| Soft risk penalty | 25 * max(0,p-0.12) |
| Reward | Interest + fees - loss - funding - capital - penalty |

All coefficients are configurable. Expected-loss subtraction was removed to avoid
double-counting realized credit loss. No score-tier or batch-level pseudo-portfolio
constraint is presented as a hard constraint. `info.reward_components` contains all
six components and total. Interest/fees are simplified cash proxies; capital is not
a regulatory formula. Outstanding assets receive no liquidation value at truncation.

## H. Tests and local validation

**42 tests passed, no warnings**, using Python 3.12.14 and pytest 9.1.1:

- Environment: Gymnasium checker, observation/action spaces, all five recursive
  actions, customer persistence, history copies, reset, terminal/default precedence,
  horizon, invalid actions, long-run invariants, limit saturation and over-limit debt.
- Dynamics: payment/spend accounting, no debt forgiveness, previous-state dependence,
  default absorption, zero-debt hazard, delinquency persistence/cure, latent effects,
  and statistically worse payment/default under stress.
- Reproducibility: same seeds/actions, different seeds, isolation from global NumPy,
  different action futures with unchanged macro paths, predicted-PD independence,
  hidden-state non-disclosure and ignored future labels.
- Economics/risk/configuration: hand-calculated rewards including default loss,
  observed-feature allowlist/refresh, invalid PD rejection, episode-level metric
  denominators, YAML/default agreement and invalid parameter rejection.

Executed commands:

```shell
.venv\Scripts\python.exe -m pip install -e . --no-build-isolation --no-deps
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m experiments.trajectory_sanity
.venv\Scripts\python.exe -m experiments.train_ppo --timesteps 256
```

Package imports resolve to `src/credit_rl` from the project root. `pip check` found no
broken dependencies. There was no pre-existing lint/format tooling to run. SB3's env
checker and a real train/save/reload/evaluate path passed. Repeating the complete
sanity experiment reproduced **all seven CSV outputs byte-for-byte**. Version and
source-hash manifests accompany outputs; identical results across other library
versions/hardware are not promised.

## I. Sanity experiment and manual inspection

Saved under `outputs/results/trajectory_sanity/`: trajectories, policy summary,
latent-risk trajectories, 400 macro trial records, macro summary, PD validation and
feature importance, manifest and reproducibility verification. Fitted PD model is
under `outputs/models/trajectory_sanity/`. Five figures under
`outputs/figures/trajectory_sanity/` show three customer policy comparisons, hidden
trait comparison and normal/stress diagnostic. Figures were visually inspected.

| Six-customer diagnostic policy | Mean episode return (EUR) | Default incidence |
|---|---:|---:|
| Static | -2327.11 | 4/6 |
| Repeated +10% | -2142.73 | 4/6 |
| Repeated -10% | -1638.72 | 4/6 |
| Risk threshold | -1237.86 | 4/6 |

These are tiny paired mechanism checks, not a policy-ranking experiment. All policies
happen to have the same default months for these six seeds; balances and losses differ.

- **A, static:** customer 2 keeps a EUR 4567 limit for 24 months while balance,
  utilization, predicted PD and accumulated return evolve.
- **B, repeated increases:** the same customer's limit compounds to EUR 15000;
  future balance, utilization and return differ from static.
- **C, repeated reductions:** its limit reaches EUR 500, constraining spending and
  balance. Other customer paths demonstrate debt persisting above reduced limits.
- **D, risky customer:** under identical observed initial state/shocks, weaker hidden
  traits produce lower payments and default at month 3 versus month 21 in the reference
  trace. The reference subsequently accumulates six consecutive delinquent months.
  This single comparison illustrates mechanics; stochastic dominance is not implied.
- **E, macro stress:** for one fixed customer/traits and 200 paired runs per regime,
  first-month mean payment ratio falls from 0.3177 to 0.2391; delinquency rises from
  13.0% to 23.5%; horizon default incidence rises from 94% to 100%.

The high conditional default incidence exposes an uncalibrated scenario, not a
realistic portfolio forecast. Stress return is less negative (-3191.19 versus
-4461.28) despite worse risk, because spending/exposure and survival duration change.
This is retained transparently rather than tuning away the result.

The separate **256-step PPO smoke test** saved/reloaded successfully and evaluated
30 held-out customers: mean returns PPO -1717.39, static -3375.60, threshold -1748.56.
This tiny run does not establish PPO superiority. Historical negative PPO results
are preserved and remain explicitly separated from these new outputs.

## J. Remaining technical debt

Behavioral/default coefficients and latent distributions lack empirical calibration.
Snapshot-trained PD has distribution/horizon mismatch and must be validated on new
trajectories. Delinquency is a principal-payment proxy without aging arrears, recoveries
or regulatory semantics. Income is fixed; macro chains are independent across episodes.
Default rates may be excessive for plausible populations. The reward is an economic
proxy with finite-horizon censoring and no terminal asset valuation/customer welfare.
Only simple baselines and smoke PPO runs exist; no confidence intervals or robust
generalization claims are warranted. Initialization reconstructs missing history by
assumption. Archived scripts are evidence, not maintained compatible entry points.
The accidentally committed virtualenv remains in existing Git history; only future
tracking was corrected, without a destructive history rewrite.

## K. Prioritized next sprint (not implemented)

1. Define target population, economic objective, monthly default/delinquency semantics
   and acceptable calibration evidence before choosing parameter values.
2. Perform sensitivity and identifiability analysis on payment/default coefficients;
   characterize default incidence, utilization and delinquency duration distributions.
3. Build independent DGP trajectory datasets split by customer and seed; fit/calibrate
   a PD estimator on observed histories with an explicit prediction horizon.
4. Add a coherent arrears/recovery ledger and terminal exposure valuation; audit reward
   timing and incentives before interpreting total return as business value.
5. Add stronger transparent non-RL policies (risk/affordability rules and a short-horizon
   look-ahead benchmark) under the same information and action constraints.
6. Establish held-out multi-seed policy evaluation with paired uncertainty intervals,
   survival-aware risk metrics and predeclared model-selection rules.
7. Introduce shared macro scenario paths for cohort stress tests and distribution-shift
   checks; retain explicit distinction between causal assumptions and fitted evidence.
8. Only then run a fixed-budget PPO comparison, reporting negative results and testing
   robustness without changing reward definitions to favor a policy.
