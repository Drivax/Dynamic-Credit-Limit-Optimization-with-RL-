# Off-policy evaluation with simulator validation

## Estimand and information boundary

This experiment estimates a target policy's **undiscounted 24-month or first-default
episode value**, separately for net economic value and the raw cumulative reward.
OPE gamma is 1, whereas the trained PPO used gamma 0.98. This is deliberate: OPE is
checking the reported cumulative KPIs, not its training objective. Monte Carlo uses
the same estimand and nominal environment.

Logged fields are customer ID, decision month, 21 public observation coordinates,
requested action, reward, net economic value, next public observation, terminated,
truncated, chosen-action propensity and the full five-action behavior distribution.
There are no latent traits, hidden hazards or future shocks in this dataset. The next
observation is a recorded transition outcome, not an input available to the actor at t.

The behavior policy conditions only on observed information and randomizes independently
of the world's shocks. Thus the known action propensity supplies sequential action
randomization; hidden customer heterogeneity does not introduce an unlogged behavior
confounder in this constructed experiment. This is not a claim about bank histories.

## Behavior and target policies

At each decision, the behavior distribution is

`mu(a|o) = 0.4 I[a=PDThreshold(o)] + 0.2 I[a=Static(o)]`
`          + 0.2 I[a=MyopicEconomic(o)] + 0.2/5`.

Every requested command therefore has propensity at least 0.04. Commands that are
projected to the same effective limit are still distinct logged actions: their
probabilities must not be silently merged. Environment guardrails apply normally.
The stored propensity is exactly the probability of the sampled requested action.
The evaluation explicitly verifies sums and positive behavior support.

Targets are Static, AlwaysDecrease20, PDThreshold, UtilizationPD, MyopicEconomic,
all five PPO seeds with and without PD, Behavior itself, and SoftPDThreshold.
The latter is the declared stochastic target
`pi_soft = 0.8 mu + 0.2 delta_PDThreshold`, a modest behavior shift used to study
overlap. It is not a new trained algorithm. Oracle policies are excluded from OPE.
Deterministic PPO evaluation uses its argmax action probability (0 or 1), not its
stochastic training distribution. No target is chosen from the OPE results.

## Episode importance sampling

For logged episode i, length T_i, known behavior mu and target pi:

`w_i = product(t=0..T_i-1) pi(a_it|o_it) / mu(a_it|o_it)`

`IS = (1/N) sum_i w_i G_i`

`WIS = sum_i w_i G_i / sum_i w_i`

`ESS = (sum_i w_i)^2 / sum_i w_i^2`.

The likelihood ratio uses the entire observed episode, including the last decision
before termination or horizon truncation. No action is invented after default.
Products are accumulated in log space. Any zero target propensity makes the episode
weight exactly zero. Overflow is rejected rather than silently truncated.

Ordinary IS is the standard full-support likelihood-ratio estimator; large finite
sample errors are still possible. WIS is self-normalized and generally biased at
finite N. If all weights are zero, IS numerically equals zero, WIS is undefined and
ESS is zero. These outcomes are retained as failures, not replaced by an MC value.

The identity check pi=mu gives all weights exactly one, ESS=N and both estimates
equal the sample mean. It validates implementation; it does not validate a distant
deterministic target's overlap.

## Clipping, overlap and uncertainty

The same data are evaluated unclipped and with the declared cap `min(w,20)`.
Both IS and WIS clipping results are exported separately. Clipping trades variance
for bias and can inflate ESS; it cannot establish reliable support. No hidden
clipping is applied to the primary estimates.

Support summaries include minimum, p10 and median target-weighted behavior propensity,
`sum_a pi(a|o) mu(a|o)`, plus mean target mass on commands with mu<0.1 and on mu=0.
For deterministic targets, this is exactly mu(a_target|o). Positive per-decision
support does not ensure any whole logged trajectory follows a deterministic target.
Zero-weight fraction, nonzero trajectory count, maximum weight, p99 and ESS expose
this sequential overlap problem. ESS<30 is an explicit diagnostic warning; ESS>=30
is **not** a reliability guarantee.

The standard run uses **five independently generated logged cohorts of 600 episodes**,
with independent IDs, population, traits, shocks and behavior draws. The common target
MC cohort has **2,000 different customers**, disjoint from all logged and RL/PD cohorts.
The MC cohort shares exogenous inputs across targets, but never with logged episodes.
All target seeds are retained. Unit tests use small analytic cases and never launch
these budgets.

Within each logged replicate, 400 bootstrap samples resample complete episodes to
give IS/WIS percentile intervals. Replicate means, SD and RMSE versus the independent
MC mean are exported. The reported mean error is a finite-replicate estimate relative
to an **uncertain MC reference**, not known population bias. MC standard errors and
customer-bootstrap intervals are saved. Coverage of the MC point across five runs
is descriptive, not a reliable nominal-coverage estimate and not coverage of an
exact simulator expectation. Undefined WIS replicates and invalid bootstrap counts
are retained. Resampling cannot recover rare trajectories absent from the log, so
small-ESS bootstrap intervals can be severely misleading.

## Reproduction and artifacts

`python -m experiments.off_policy_evaluation --profile standard --stage all` loads
frozen policies, simulates the independent MC cohort, generates logs, verifies
propensities, calculates weights/estimates/support and generates figures. `evaluate`
and `report` can run separately. Smoke/standard/full budgets are in `configs/ope.yaml`.
Completed MC/log files are reusable under the immutable run fingerprint.

`outputs/results/ope/standard` contains logged CSV.GZ trajectories, MC episode CSVs,
`mc_summary.csv`, per-replicate `estimates.csv`, `summary.csv`, support, full episode
weights, timings and runtime. `outputs/experiments/ope_standard` stores the timestamp,
commit, exact configs, hashes, seeds, target identities and nominal world parameters.
Figures show estimates against MC, positive importance weights with zero fractions,
ESS and error versus support. PPO displays average all training seeds, not a best seed.

## Deliberate limits

No Doubly Robust estimator is claimed: it would require an additional validated
outcome/value model and cross-fitting design. No offline RL or behavior cloning is
implemented. The full-trajectory estimator is deliberately simple and exposes long-
horizon overlap failure; per-decision and model-based estimators remain future work.
Synthetic known propensities, no hidden behavior confounding, a fixed reward model,
few logged replicates and finite MC precision strongly limit generalization to real
logged credit decisions. Low ESS is a substantive negative result, not a reason to
hide an estimator or retune the behavior policy after evaluation.
