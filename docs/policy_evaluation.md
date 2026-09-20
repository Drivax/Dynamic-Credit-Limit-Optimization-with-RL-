# Policy evaluation under partially observed credit risk

## Decision problem and information contract

For customer i, persistent hidden characteristics Z_i, observable macro M_t and
indexed exogenous shocks ε_i,t drive the transition:

\[
 S_{i,t+1}=f(S_{i,t},A_{i,t},M_t,Z_i,\epsilon_{i,t}),\qquad
 \max_\pi E[\sum_{t=0}^{T_i-1}\gamma^t R_{i,t}],\quad \gamma=0.98.
\]

An episode ends at first default or 24 transitions. The terminal nondefault loan
has no residual asset valuation. This finite-horizon approximation affects incentives.

The policy receives exactly the 21 coordinates in `OBSERVATION_NAMES`: elapsed
fraction, current limit, balance, utilization, payment ratio, delinquency indicator
and consecutive count, income, behavioral score, current macro stress, predicted
12-month PD, recent spending, six-month late count, tenure, terminal default flag,
income change, macro income/spending growth and three regime indicators. All use
fixed transforms bounded in [0,1]. The recurrent history is represented by observed
behavioral summaries and the separately trained history-based PD, not an RNN.

Deployable `act(observation)` implementations receive no customer traits, private
hazard, full future paths or default outcome. Hidden state appears only in simulation
setup and a separately labeled oracle. This is a code-level information contract,
not a security sandbox against arbitrary Python introspection.

## Actions and constraints

The requested action is one of −20%,−10%,0,+10%,+20%. The environment bounds the
resulting limit to EUR 500–15,000 and respects configured monthly caps. In this
benchmark, **three or more consecutive delinquency months prohibit increases**:
an increase request becomes maintain. Decreases do not erase debt. Default ends the
episode and prevents another action. Requested and effective changes, plus guardrail
blocks, are exported. The guardrail is a configurable environment admission rule;
it does not change customer behavior coefficients or the hidden default law.

Rules and Random choose feasible actions. PPO uses the ordinary SB3 categorical
action space, with environment projection for infeasible requests. No separate
masking dependency or arbitrary invalid-action reward is added. At a binding limit,
two requests that induce the same effective action have the same transition/reward
under the same shocks. Requested-action churn and effective churn are distinguished.

Experimental portfolio alerts, not regulatory constraints, are declared in YAML:
default incidence >0.75, losses/initial balances >0.70, high-risk exposure share >0.60,
with high risk defined as predicted PD ≥0.60. These are reported ex post and do not
turn PPO into a constrained-RL algorithm. The hard delinquency rule applies to all
policies; soft alerts may be violated.

## Reward, units and economic KPIs

For opening balance B_t, realized next purchases C, closing exposure B', realized
default D and opening horizon PD p_t, the code computes:

\[
 R_t=(1-D)B_t\,APR/12+C\,fee-D\,LGD\,B'
 -funding\_rate\,B'/12-capital\_weight\,rwa\_factor\,p_tB'
 -constraint\_weight\max(0,p_t-threshold).
\]

Defaults remove interest income for that transition; interchange remains. Principal
obeys B'=B−payment+purchases, with no capitalized interest. All terms are EUR per
transition. Assumed APR=18%, fee=1.2%, LGD=55%, annual funding=3%, capital weight=1.5,
RWA factor=0.08, penalty weight=25 and threshold=0.12 remain unchanged. The capital
charge is a synthetic monthly proxy using the supplied PD horizon; it is not an
annual regulatory capital formula. Its large magnitude is visible in the empirical
audit and must not be confused with a proven cost of capital.

The report distinguishes:

- **Revenue:** summed recognized interest plus purchase fees.
- **Net economic value:** revenue minus realized credit losses and funding costs.
  This is a simplified cash/economic proxy before synthetic capital and penalties.
- **Cumulative reward:** net economic value minus capital charges and penalties.
- **Discounted reward:** Σ0.98^t R_t, the validation objective used for checkpoints.

These are separate columns; maximizing reward need not maximize the economic proxy.
No terminal loan value, customer utility, operational change cost or recovery timing
is modeled. A churn penalty is not introduced without an empirical cost assumption.

## Implemented policies

All policies use the same action space, admission rules and actual transition engine.

| Policy | Exact requested decision logic |
|---|---|
| Static | Maintain |
| AlwaysDecrease | −10% when feasible, otherwise maintain |
| AlwaysDecrease20 | −20% when feasible, otherwise maintain |
| AlwaysIncrease | +10% when feasible, otherwise maintain |
| Random | Uniform among feasible actions; separate seeded generator per customer |
| PDThreshold | PD ≥high: −10%; PD <low: +10%; otherwise maintain; respect guardrails |
| UtilizationPD | PD ≥high or delinquency ≥3: −10%; PD <low, utilization ≥0.75 and no delinquency: +10%; otherwise maintain |
| MyopicEconomic | Maximize the explicitly specified observable one-step expected reward surrogate over feasible actions |
| PPO | Deterministic argmax of the trained SB3 policy distribution at evaluation |
| PPO_without_PD | Same architecture/budget/seeds; PD coordinate zeroed for the actor during training and evaluation |

Threshold candidates are (0.1,0.5), (0.2,0.6), (0.3,0.7). A single pair is selected
using the mean of both rules' validation discounted reward. All alternatives remain
in `threshold_validation.csv`. AlwaysDecrease20 was added after observing nearly
constant behavior on validation and before opening the final test; it directly
checks whether PPO's strongest reduction can be replicated by a trivial rule.

### Observable myopic surrogate

For each admitted L', repayment is approximated as
`min(B*q_observed, 0.5*income)`, remaining principal B_r=B−payment, and purchases as
`min(max(0,L'−B_r), spending_t*(L'/L)^0.15*(1+observed_macro_spending_growth))`.
Then B'=B_r+purchases. This is an approximation, not a call to the hidden DGP.

The H-month forecast is converted to a monthly reference hazard by
`q=1−(1−PD)^(1/H)`, assuming a flat conditional hazard. An explicit action/state
surrogate adjusts its log odds by
`1.0*(B'/L'−B/L)+0.3*(B'−B)/income`. Coefficients are predeclared assumptions,
not fitted using test data and not identified causal effects. Expected reward uses
q' for default loss and surviving interest, and the original H-month PD for the
same capital/penalty terms as PPO. Ties prefer the smallest absolute limit change.
The baseline optimizes one-step training reward, not a multi-period continuation.

### SIMULATOR-ONLY ORACLE

`SIMULATOR_ONLY_ORACLE_Myopic` receives hidden traits via an explicit privileged
adapter. It evaluates each feasible action using 12 independent hypothetical
behavioral shock draws shared across candidate actions, holds the currently observed
macro state fixed and integrates default loss using the resulting true conditional
hazard. It never reads the realized future evaluation shocks or future macro path.

This is **not deployable**, not a same-information baseline and not a rigorous upper
bound: it remains myopic and Monte Carlo approximate. Its gap to a deployable policy
can be negative. Oracle rows have their own information-set label and are excluded
from the main deployable results/figures.

## Populations and controlled exogenous inputs

`EvaluationScenario` bundles immutable initial observed state, persistent traits,
shock path and macro path. Customer IDs use disjoint prefixes RL_train_,
RL_validation_, RL_test_, RL_oot_. SeedSequence namespaces 11/22/33/44 and master
seed 73000 separately generate initial populations and trait draws. Stable shock
hashing further includes identity/month/channel. Subsets preserve population prefixes.

Default sizes are 1,500 training, 100 validation, 300 test and 300 supplementary
independent Markov-cohort customers. The same customer and shocks are replayed under
every policy and macro intervention. No Random-policy or oracle hypothetical draw
can shift the evaluation world's random numbers. Named baseline/mild/severe/recovery
paths are exactly the existing macro-engine scenarios.

At a 24-month horizon, baseline stays normal. Mild stress is normal for decisions
0–5, stress with severity 0.65 for 6–11, then normal. Severe stress is normal for
0–3, stress with severity 1.6 for 4–19, then normal. Recovery starts in stress at
severity 1.3 for 0–5, weakens to 0.6 for 6–11, becomes normal for 12–17 and expansion
for 18–23. Recovery therefore does not simply append improvement to the severe
scenario: early exposure and default timing differ, so cumulative losses need not
be ordered by the scenario names.

Training uses fixed per-customer Markov paths; validation uses baseline. The
supplementary `oot_markov` sample is a fresh independent synthetic cohort/path draw,
**not an empirical calendar out-of-time test**. The main robustness evidence is
held-out customers under the explicit macro interventions. Changing survival changes
which later states remain observable; active-month summaries have this selection effect.

The PD artifact is frozen, identified by SHA-256 and not retrained. RL cohorts use
different seeds and namespaces from the PD-modeling cohorts. Scenario setup contains
latent fields but they are never supplied to the deployable actor.

## PPO architecture, selection and training

The `configs/policy_evaluation.yaml` file centralizes all experiment parameters.
SB3 PPO uses separate actor/critic MLPs, each **[64,64], tanh**, categorical actions,
CPU, one Torch thread and deterministic Torch algorithms. Defaults:

| Parameter | Value |
|---|---:|
| Total transitions per main agent | 32,768 |
| Main training seeds | 101,202,303,404,505 |
| Discount γ / GAE λ | 0.98 / 0.95 |
| Rollout n_steps / minibatch | 512 / 128 |
| Epochs / clipping | 5 / 0.2 |
| Value coefficient / gradient norm | 0.5 / 0.5 |
| Observation normalization | Fixed public transforms; no learned statistics |
| Reward units for optimization | kEUR = EUR×0.001, fixed positive scaling |
| Validation interval | 8,192 transitions, plus initialization and final update |

Two pilot settings, learning-rate/entropy (0.0003,0.01) and (0.0001,0.02), use 4,096
transitions on pilot seed 17. The maximum validation discounted reward selects the
setting; ties retain the first. Main runs train five agents with PD and five without
PD at equal budgets. All ten runs and both pilots remain saved; no best-seed result
is substituted for the multi-seed aggregate.

For each agent, the checkpoint with maximum mean validation discounted raw reward
is selected, earliest on an exact tie. Initialization is eligible so training is not
assumed beneficial. At rollout boundaries the callback precedes SB3's final update;
that last update is evaluated explicitly at training end. Checkpoint records may
therefore contain two entries at the final timestep. Selection never uses test,
stress or supplementary cohort outcomes. All selected models are saved/reloaded and
checked for deterministic prediction parity.

Episode monitor files record kEUR returns. `progress.csv` records value/policy loss,
entropy loss, approximate KL and explained variance; `validation.csv` records raw
EUR validation outcomes. Checkpoints and final models are retained. Metadata contains
all train/validation IDs, settings, selected step, timing and the experiment identity.

## Metrics, pairing and uncertainty

The evaluator has a single `evaluate_policy` path for every policy, with factories
only at construction. Detailed files identify customer, month, partition, scenario,
policy, policy seed and information set. Full histories include initialization;
transition metrics exclude that row. Economic values are reported per initial customer.

Default incidence is defaults/initial customers, not defaults/active months. Credit
loss rate is total realized loss/total **initial balances**. EAD is exposure summed
on default divided by defaults. Delinquency includes ever-delinquent customers and
delinquent active-month fraction. High-risk exposure share is ΣB'·1(PD_t≥0.6)/ΣB'.
Exposure/utilization means average each customer's active-month mean, then customers.
Action fractions and mean requested/effective changes use active transition counts.
Limit growth uses final/initial limit−1. Reversals count sign changes between successive
nonzero effective adjustments; no-change periods do not create a reversal.

Main intervals use 300 bootstrap replicates. A replicate samples customers jointly
across policy seeds, and samples PPO training seeds with replacement. Baselines have
one deterministic instance (Random has fixed customer RNG). Paired differences versus
Static and Myopic use the same customer resample and subtract customer outcomes before
aggregation. Full customer-level difference distributions and fraction positive are
saved. There are no monthly-row bootstrap tests or multiple p-value rankings.

Multi-seed tables include individual seed means and their standard deviation. Zero
seed variability can legitimately occur when deterministic policies collapse to the
same actions; it does not imply zero uncertainty about customer populations or DGPs.
Intervals remain conditional on the specified macro scenarios and frozen PD model.

## Robustness, information ablation and policy feedback

All main policies/seeds face baseline, mild stress, severe stress and recovery.
The robustness table retains scenario-specific value/default incidence and worst
scenario value without constructing a composite score. Experimental risk-alert flags
remain visible.

The without-PD ablation removes only the explicit actor PD coordinate. The reward
still uses the same frozen PD estimate, preserving the objective. Other observables
retain risk information, so this is not removal of all risk information.
An additional evaluation halves the actor-visible PD for PPO, PDThreshold and Myopic,
while actual transitions, reported true model forecasts and reward inputs stay unchanged.
This isolates decision sensitivity to a biased information channel; it does not
simulate retraining, a new risk model or altered default laws.

Controlled observation slices vary PD, utilization, delinquency and current macro,
with other state quantities held fixed or changed consistently with balance/limit.
They may be off the observed data manifold and are descriptive, not causal analyses.
PPO heatmaps average requested actions over all five seeds. Increasing-PD/increasing-
limit violations are counted rather than repaired by imposed monotonicity.

PD calibration under policy-generated states is evaluated against the actual next
H-month default target with the same strict administrative/censoring rules as PD
training. Only eligible active snapshots enter. Model forecasts are not refit.
Feature means/quantiles are compared with saved PD-training distributions. This
measures the feedback loop PD→policy→actions→states→PD inputs. It is a diagnostic of
covariate and outcome shift, not an off-policy correction or new calibration claim.

Representative trajectories are selected by initial observable utilization and score,
before examining outcomes. Static, Myopic and every PPO seed share the shocks. Limits,
balances, utilization, PD, delinquency, reward and cumulative value are plotted. A
positive difference alone is not evidence that PPO learned continuation value: the
same effect must not be reproducible by a constant action rule.

`constant_rule_equivalence.csv` explicitly joins customer/month trajectories against
AlwaysDecrease20 and records unmatched rows and maximum differences in effective
actions, balance, limit, reward and PD. Requested actions can differ at the minimum
limit even when the effective transition is the same.

## Reproduction and limitations

From the repository root, with PD artifacts and RL extras installed:

```shell
python -m experiments.compare_policies --stage audit
python -m experiments.compare_policies --stage smoke
python -m experiments.compare_policies --stage train
python -m experiments.compare_policies --stage evaluate
python -m experiments.compare_policies --stage figures
```

`--stage all` composes audit/train/evaluate/figures. `--output` isolates another run.
Completed main agents are reusable only when configuration and PD artifact identity
match. Saved data and figures live under `outputs/{results,models,figures}/policy_evaluation`.

Limitations include synthetic DGP/portfolio/economic parameters, incomplete terminal
valuation, principal-only repayment accounting, no customer utility, fixed small action
menu, simplified recovery, fixed training trajectories, small validation population,
limited training budget, one pilot seed, unvalidated H-month capital proxy, surrogate
myopic dynamics, macro-path uncertainty outside bootstrap and policy-induced PD shift.
An oracle is not an attainable performance bound. Empirical deployment conclusions
require independently governed real-world validation.
