# Phase B: predeclared information and planning protocol

Phase A artifacts, its reports and canonical checkpoints are read-only. DGP,
reward, economic coefficients, canonical PPO budget/hyperparameters and checkpoint
selection are unchanged. This protocol is fixed before Phase B test evaluation.

## Information audit

X_t contains CustomerState (including income/score reversion anchors and ordered
late history), four persistent traits, elapsed time, the seven-row observable
history used by the frozen PD estimator, and the macro process context.
O_t is exactly `build_observation`'s bounded float32 vector of 21 quantities.
H_t=(O_0,A_0,...,O_t) includes only decisions and observations already available.
X_t -> O_t loses traits, reversion anchors, ordered late history, individual past
observations/actions, and details compressed by the PD estimator. PD and late
counts already encode some history: a feed-forward actor is not wholly memoryless
with respect to the underlying raw data. Initial income and score were observed
at enrollment and can be recovered from a retained history; they are not inherently
unobservable traits. Most monotone bounded transforms are invertible in theory;
float32 precision and aggregation still matter.

Future shocks and future macro paths are not features for ANY planner. Conditional
MC targets use the same fixed baseline/stress scenario protocol as Phase A.
Scenario labels are metadata, not predictors; fitted planners average over the
training mixture conditional on their allowed features. The fixed-scenario MC
teacher therefore has additional conditioning information relative to the fitted
planners and is not a same-information upper bound.

## Estimators and populations

The 72 Phase A states establish mechanisms but are too few to fit and validate
several information sets. Expand that diagnostic protocol with independently
generated Phase B customers. A new population seed changes initial profiles,
traits and exogenous paths, not DGP parameters. Disjoint train/validation/test
identity namespaces are retained across paired baseline/stress scenarios. No
Phase A target enters training or selection. The final test has never selected
canonical PPO checkpoints, PD models, planner horizons or model complexity.

Per customer/scenario, uniformly sample up to the declared number of active
visits. Assign each customer one source policy from Static, MyopicEconomic,
uniform legal Random, and frozen PPO. All sampled states are reachable. Weights
restore active visitation within the sampled source trajectories. Save public
observation/action prefixes separately from privileged fields and MC targets.

Use existing `structural.rollout`, `ShockPath`, frozen PD, constraints and reward.
Each initial request shares independent hypothetical shock channels with all
other requests. Main Q targets use frozen PPO seed 101 deterministic continuation,
gamma .98, horizons 6/12/remaining. Split the draws into selection and evaluation
halves on test; training uses train draws only. The MC teacher is selected on one
half and evaluated on the other, so held-out regret can be negative.

Fit simple histogram gradient boosting regressors to five action values, using
hold-relative targets to focus on action differences, plus a common hold value.
Select leaf complexity and horizon only on validation hold-relative Q prediction
error; no test-driven tuning. Fixed iteration budgets and minimum leaf counts
regularize fitting. The feature sets are F0=current 21D; F1=current without PD;
F2=current without repayment/delinquency/score/spending predictors; F3=current
plus 3 months; F4=plus 6 months; F5=all available observable history summarized.
History includes lags, rolling means/volatility/trends, previous PD/payment/limits
and previous requested actions. F2 still uses the public lending feasibility mask,
which necessarily reveals whether an increase is allowed.

ObservationPlanner and HistoryPlanner consume only public arrays. The separately
typed FullStatePlanner adds traits, anchors, ordered late history and the exact
seven available PD-history rows. It is a learned, privileged approximation to Q, NOT
Q*, a guaranteed upper bound, or proof of the economic value of latent information.
All learned planners are evaluated by the existing `evaluate_policy` engine on
the same untouched customers/shocks. Repeated greedy execution differs from
their fixed-continuation training target; report this approximation explicitly.

## Comparisons and uncertainty

Report canonical monetary components, default incidence, exposure, purchases and
requested/effective action distributions. J denotes discounted raw EUR reward;
net economic value remains undiscounted. Pair all policy differences by customer;
reuse the existing crossed customer/seed bootstrap. Intervals condition on fitted
PD/planners, training dataset, MC targets and fixed scenarios. They exclude model
misspecification and planner refitting uncertainty.

Report both same-state regret against the fitted FullStatePlanner and independently
selected MC full-state reference. No forced nonnegative truncation of held-out
regret. Separate raw requested agreement from admitted-limit equivalence. PPO
counterfactual regret uses actual frozen-PPO visits and the named continuation;
it is a first-action opportunity, not the sum of attainable episode improvements.

Latent regressions use train-only fitting and test R2/RMSE. Grouped permutation
importance evaluates held-out action-value prediction errors, not causal effects;
correlated/physically implausible permutations limit interpretation.

## PPO audit

Read all saved checkpoints/logs and validation selection records. The original
rollout advantages were not persisted. Instrument a separate exact replay with
the original population, seeds, optimizer, reward scaling and validation callback;
save raw GAE advantages/returns, probabilities/logits/entropy, state/action coverage,
and gradients before clipping. Never overwrite canonical artifacts. Compare replay
weights with saved weights and report any mismatch; do not label reconstruction
as an original log unless equivalence is verified. No tuning, extra training
budget, recurrent PPO or alternative algorithm is introduced.

Compare critics in raw EUR (undo .001 scaling) with independent MC returns,
including stochastic continuation when available: training values correspond to
the stochastic actor, whereas canonical evaluation is deterministic. Report both
estimands rather than attributing their difference entirely to critic error.

## Decision gate

Empirical gaps are descriptive, not a causal additive decomposition. Positive
paired 95% intervals for History-Observation, Observation-PPO and Full-History
respectively motivate memory, optimization/representation, and observability
experiments. Multiple supported gaps motivate a factorial experiment. Report
effect sizes and multiplicity/approximation limits. If evidence is inconclusive
or the privileged regressor is too inaccurate, classify as unresolved, not as
proof that information is inaccessible. Recommend only; do not implement Phase C.
