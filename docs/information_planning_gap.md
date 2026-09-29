# Information and planning gap

This is Phase B of the unchanged canonical simulator. Phase A established a
state-dependent, multi-step opportunity under privileged full-state conditioning;
it did not demonstrate attainable improvement by the observable PPO actor.
The present measurements are generated from `outputs/main/information_gap`. Smoke runs
are software checks, not scientific evidence. No DGP, reward, economic coefficient,
PPO hyperparameter or canonical checkpoint has been changed.

## 1. What information does PPO actually observe?

O_t is exactly the 21 float32 entries of `build_observation`: time, limit, balance,
utilization, payment, delinquency flag/count, income, score, macro stress, PD,
spend, late-payment fraction, tenure, default, income change, macro growth and
regime indicators. X_t additionally contains four traits, income/score reversion
anchors, ordered late history and the PD model's seven-row history. H_t retains
only public observation/action prefixes. Anchors were observed at enrollment;
they should not be confused with genuinely hidden persistent traits. PD itself
already compresses observable history. No fitted planner receives future shocks,
future macro paths, outcomes, customer identities or scenario labels.

See [the predeclared protocol](information_gap_protocol.md), `feature_names.json`
and the explicit public/privileged classes in `policies/information.py`.
Train, validation and test customers are disjoint, with new populations unrelated
to canonical fitting/selection. Phase A targets are not reused to fit these models.
Training targets expand the same simulator/CRN protocol. Test targets use separate
draws and split-draw reference selection. Public planners receive only their arrays;
FullStatePlanner alone receives a privileged snapshot, without future information.

| role | customers | states |
| --- | --- | --- |
| test | 80 | 318 |
| train | 160 | 640 |
| validation | 48 | 188 |

The two scenarios reuse identities and exogenous shocks for pairing. The standard
protocol uses 64 first-action MC draws per state and 32 independent stochastic
critic draws. State samples restore source-trajectory visitation through inverse
inclusion weights; whole-episode J gives each initial customer equal weight.

## 2. How much planning value exists with current observation only?

ObservationPlanner minus MyopicEconomic: baseline: +2647.79 EUR, 95% CI [1917.54, 3367.28] (positive interval). severe_stress: +2270.36 EUR, 95% CI [1793.71, 2818.32] (positive interval).

J is discounted raw reward in EUR per initial customer. Net economic value is
undiscounted and excludes the capital proxy and PD penalty. All policies below
use the same test customers and shock paths, including defaults and early exit.

| scenario | policy | discounted_reward | net_economic_value | default_rate |
| --- | --- | --- | --- | --- |
| baseline | AlwaysDecrease20 | -2102.25 | -746.14 | 0.68 |
| baseline | FullStatePlanner | -2078.85 | -578.51 | 0.61 |
| baseline | HistoryPlanner | -2100.50 | -613.22 | 0.60 |
| baseline | MyopicEconomic | -4705.91 | -1364.31 | 0.53 |
| baseline | ObservationPlanner | -2058.12 | -663.04 | 0.60 |
| baseline | PPO | -2102.25 | -746.14 | 0.68 |
| severe_stress | AlwaysDecrease20 | -2545.69 | -1200.57 | 0.95 |
| severe_stress | FullStatePlanner | -2608.84 | -1128.93 | 0.93 |
| severe_stress | HistoryPlanner | -2621.15 | -1171.98 | 0.95 |
| severe_stress | MyopicEconomic | -4846.57 | -2180.11 | 0.84 |
| severe_stress | ObservationPlanner | -2576.22 | -1165.73 | 0.95 |
| severe_stress | PPO | -2545.69 | -1200.57 | 0.95 |

The conditional paired customer/PPO-seed bootstrap intervals are:

| scenario | gap | difference | lower | upper |
| --- | --- | --- | --- | --- |
| baseline | observable_planning | 2647.79 | 1917.54 | 3367.28 |
| baseline | history | -42.37 | -83.96 | -4.07 |
| baseline | privileged | 21.65 | -12.41 | 61.50 |
| baseline | ppo_gap | 44.13 | 9.12 | 87.13 |
| severe_stress | observable_planning | 2270.36 | 1793.71 | 2818.32 |
| severe_stress | history | -44.94 | -86.87 | -7.09 |
| severe_stress | privileged | 12.31 | -60.55 | 73.64 |
| severe_stress | ppo_gap | -30.53 | -65.38 | -3.85 |

These are empirical benchmark gaps, not an additive causal identity. Confidence
intervals condition on fitted PD/planners, sampled training states, MC targets and
fixed scenarios; they do not include refitting or simulator misspecification.
All canonical components, purchases and action distributions are saved separately
in `benchmark_values.csv`, `episode_metrics.csv` and `action_distribution.csv`.

In particular, net-value improvements need not improve the training objective:
reward additionally charges the unchanged capital proxy and PD penalty, and
discounting also differs. The net-value paired intervals are preserved in
`information_gap.csv`; objective disagreement must not be labelled PPO failure.

## 3. How much additional value does observable history provide?

HistoryPlanner minus ObservationPlanner: baseline: -42.37 EUR, 95% CI [-83.96, -4.07] (negative interval). severe_stress: -44.94 EUR, 95% CI [-86.87, -7.09] (negative interval).

F3 and F4 add three/six prior observations/actions plus window mean, volatility and
trend; F5 adds whole-prefix summaries and enrollment observations. Each episode
gets fresh planner memory. F1 removes PD; F2 removes payment/delinquency/score/spend
variables, while retaining the public feasibility mask. Thus F2 is an information
ablation with a declared guardrail side channel. No latent estimates enter PPO.

| scenario | policy | discounted_reward |
| --- | --- | --- |
| baseline | History3 | -2076.71 |
| baseline | History6 | -2093.12 |
| baseline | HistoryPlanner | -2100.50 |
| baseline | ObservationPlanner | -2058.12 |
| baseline | WithoutBehavior | -2076.00 |
| baseline | WithoutPD | -2055.99 |
| severe_stress | History3 | -2581.31 |
| severe_stress | History6 | -2615.11 |
| severe_stress | HistoryPlanner | -2621.15 |
| severe_stress | ObservationPlanner | -2576.22 |
| severe_stress | WithoutBehavior | -2597.67 |
| severe_stress | WithoutPD | -2555.26 |

Latent predictability is assessed by train-only regressors and held-out weighted
R2/RMSE. Negative R2 is retained; prediction accuracy is not decision value.

| feature_set | latent | r2 | rmse | r2_gain_over_current |
| --- | --- | --- | --- | --- |
| F0 | latent_creditworthiness | 0.70 | 0.50 | 0.00 |
| F0 | latent_spending_propensity | 0.19 | 0.25 | 0.00 |
| F0 | latent_payment_propensity | 0.35 | 0.09 | 0.00 |
| F0 | latent_income_stability | 0.02 | 0.12 | 0.00 |
| F5 | latent_creditworthiness | 0.67 | 0.53 | -0.03 |
| F5 | latent_spending_propensity | 0.32 | 0.23 | 0.14 |
| F5 | latent_payment_propensity | 0.43 | 0.08 | 0.08 |
| F5 | latent_income_stability | 0.06 | 0.12 | 0.04 |

History can improve latent prediction without improving the policy objective.
The two claims are tested separately; no equivalence between prediction and
economic usefulness is assumed.

Grouped permutation importance is in `feature_importance.csv` and its figure.
It measures changes in held-out action-difference MSE, not causality; correlated
features and off-manifold permutations limit interpretation.

The three largest mean grouped importance values for each information set are:

| feature_set | group | relative_mse_increase |
| --- | --- | --- |
| F0 | other current | 17652.62 |
| F0 | utilization/balance | 7044.68 |
| F0 | PD | 6782.24 |
| F5 | historical trends | 9743.40 |
| F5 | utilization/balance | 6041.80 |
| F5 | other current | 2692.18 |

The 'other current' group contains time-to-horizon, spending, tenure and the
default indicator. Their grouped importance does not identify an individual
feature effect or show that PPO ignores that feature.

## 4. How much value requires privileged latent information?

FullStatePlanner minus HistoryPlanner: baseline: +21.65 EUR, 95% CI [-12.41, 61.50] (interval includes zero). severe_stress: +12.31 EUR, 95% CI [-60.55, 73.64] (interval includes zero).

FullStatePlanner adds the four traits, anchors, ordered late indicators and exact
available PD-history rows. It is a supervised approximation to continuation Q,
not Q*, not deployable, and not a guaranteed upper bound. Full-minus-history also
contains representation and estimation differences; it cannot prove latents are
unrecoverable. In particular, a nonsignificant gap does not establish equivalence.

Complexity and target horizon were selected only by validation relative-Q MSE:

| feature_set | horizon | leaves | validation_relative_mse |
| --- | --- | --- | --- |
| F0 | 12.00 | 7.00 | 20700.40 |
| F1 | 24.00 | 15.00 | 23499.08 |
| F2 | 12.00 | 7.00 | 24495.73 |
| F3 | 6.00 | 7.00 | 23820.86 |
| F4 | 6.00 | 15.00 | 22901.46 |
| F5 | 6.00 | 15.00 | 22979.23 |
| Full | 6.00 | 7.00 | 22427.99 |

Held-out approximation error is material to interpretation:

| feature_set | horizon | test_relative_mse | test_relative_rmse | target_relative_rms |
| --- | --- | --- | --- | --- |
| F0 | 12.00 | 12320.85 | 111.00 | 187.98 |
| F1 | 24.00 | 12162.67 | 110.28 | 189.95 |
| F2 | 12.00 | 14308.67 | 119.62 | 187.98 |
| F3 | 6.00 | 11704.00 | 108.19 | 149.62 |
| F4 | 6.00 | 12090.44 | 109.96 | 149.62 |
| F5 | 6.00 | 11991.82 | 109.51 | 149.62 |
| Full | 6.00 | 10314.25 | 101.56 | 149.62 |

Different selected horizons and finite regressors confound pure information
comparisons. Repeated greedy execution also differs from the fixed PPO101
continuation used for targets. MC teachers condition on fixed macro scenarios,
which are not supplied as future features. These are explicit limitations of
the benchmark hierarchy, not evidence for an inaccessible-information bound.

## 5. Where does PPO sit relative to these benchmarks?

ObservationPlanner minus PPO: baseline: +44.13 EUR, 95% CI [9.12, 87.13] (positive interval). severe_stress: -30.53 EUR, 95% CI [-65.38, -3.85] (negative interval).

Observation-minus-PPO above is the direct same-current-information comparison.
The canonical contraction result remains intact. `policy_agreement.csv` reports
requested and effective agreement with FullStatePlanner. `planner_regret.csv`
reports weighted mean/median/P90/P95 and customer-cluster intervals against both
the fitted full-state planner and split-draw MC teacher, by PD, utilization,
macro and remaining horizon. Negative held-out regret is intentionally retained.
Confusion CSVs preserve exact admitted-limit changes, including floor/cap aliases;
the effective-action figure groups these only by direction for readability.

## 6. Does PPO receive the correct long-horizon learning signal?

Original GAE buffers were not saved. An isolated replay uses canonical Markov
training populations, baseline validation, identical budget, seeds and optimizer.
It reads actual completed rollout buffers and actual normalized minibatches,
without consuming RNG draws. Replay verification:

| seed | protocol_version | training_macro | selected_weights_identical | final_weights_identical |
| --- | --- | --- | --- | --- |
| 101.00 | 2.00 | markov | True | True |
| 202.00 | 2.00 | markov | True | True |
| 303.00 | 2.00 | markov | True | True |

Raw advantages averaged over collected transitions (scaled reward units):

| action | count | mean_raw_gae |
| --- | --- | --- |
| 0.00000 | 79785.00000 | -0.00478 |
| 1.00000 | 7944.00000 | -0.14250 |
| 2.00000 | 4357.00000 | -0.35817 |
| 3.00000 | 3294.00000 | -0.52851 |
| 4.00000 | 2924.00000 | -0.65511 |

After actual minibatch normalization (all optimization epochs):

| action | count | mean_normalized_advantage |
| --- | --- | --- |
| 0.00000 | 398925.00000 | 0.01305 |
| 1.00000 | 39720.00000 | -0.01148 |
| 2.00000 | 21785.00000 | -0.06579 |
| 3.00000 | 16470.00000 | -0.09788 |
| 4.00000 | 14620.00000 | -0.11648 |

`ppo_normalized_advantages.csv` additionally records the actual per-minibatch
normalization used by PPO, including repeated training epochs. Action-conditional
raw GAE means reflect state occupancy and baseline error, not causal action effects.
At predetermined rollout boundaries, `advantage_alignment.csv` compares the
sampled GAE with independent full-horizon stochastic-policy MC advantages at the
same training state and pre-update actor. There are 12 matched probes;
9 have |MC mean| > 1.96 MC SE, with sign agreement 55.6% among
those probes. This small audit conditions on the realized training macro path.
Single-trajectory GAE, lambda=.95, finite rollouts, critic bootstrapping and MC
conditional expectations are different estimands; disagreement does not uniquely
identify optimization failure or a reward-scaling error.

## 7. Is the critic accurate enough?

Critic outputs are divided by the unchanged reward scale .001. Independent MC
uses the selected stochastic actor, including a probability-weighted first action,
and remaining finite horizon. Shared private action uniforms preserve CRN across
counterfactual branches. This avoids confusing a stochastic critic with the
deterministic policy used for headline evaluation. The separately labelled
deterministic comparison uses PPO101 continuation.

| seed | scenario | states | bias | rmse | r2 | mean_mc_se |
| --- | --- | --- | --- | --- | --- | --- |
| 101.00 | baseline | 40.00 | -25.82 | 839.54 | 0.65 | 49.76 |
| 101.00 | severe_stress | 40.00 | 513.90 | 1104.95 | 0.44 | 77.10 |
| 202.00 | baseline | 40.00 | 24.12 | 833.21 | 0.67 | 50.83 |
| 202.00 | severe_stress | 40.00 | 369.04 | 936.46 | 0.62 | 79.04 |
| 303.00 | baseline | 40.00 | -9.09 | 785.87 | 0.69 | 49.44 |
| 303.00 | severe_stress | 40.00 | 411.50 | 898.61 | 0.63 | 73.74 |

`critic_states.csv` and `critic_diagnostics.csv` retain PD, utilization, horizon,
MC-preferred action and latent-profile diagnostics. Latents here are diagnostic
labels only. MC uncertainty contributes to measured RMSE and depresses R2;
the selected-action subgroup is noisy and is not a causal attribution.

## 8. Does exploration collapse?

The first saved checkpoint with at least 95% deterministic contraction on the
fixed validation observation panel is:

| seed | timesteps | contraction_share |
| --- | --- | --- |
| 101.00 | 8192.00 | 1.00 |
| 202.00 | 8192.00 | 1.00 |
| 303.00 | 8192.00 | 1.00 |

This bounds the time of behavioral collapse at checkpoint resolution; it is not
an exact transition time. The training-action figure gives finer rollout resolution.
Compare the validation trajectory before and after that point: a flat trajectory
with persistent contraction indicates that checkpoint selection retains an already
collapsed actor, rather than creating that behavior by selecting among different
final policies.

Actual rollout behavior at the first and final updates:

| seed | stage | timesteps | entropy | action_fraction_0 | action_fraction_3 | action_fraction_4 |
| --- | --- | --- | --- | --- | --- | --- |
| 101.0000 | first | 512.0000 | 1.6094 | 0.2168 | 0.2305 | 0.1953 |
| 202.0000 | first | 512.0000 | 1.6094 | 0.2051 | 0.1895 | 0.2148 |
| 303.0000 | first | 512.0000 | 1.6094 | 0.2227 | 0.2051 | 0.2051 |
| 101.0000 | last | 32768.0000 | 0.0856 | 0.9922 | 0.0020 | 0.0000 |
| 202.0000 | last | 32768.0000 | 0.2662 | 0.9453 | 0.0000 | 0.0098 |
| 303.0000 | last | 32768.0000 | 0.0655 | 0.9922 | 0.0000 | 0.0000 |

`ppo_action_training.png` shows collected requests; `ppo_entropy_training.png`
shows entropy and the original validation trajectory. Checkpoint distributions
are evaluated on the same held-out validation observation panel. State/action
coverage is conditional on PD, utilization and elapsed-horizon buckets in
`exploration.csv`; absent rows mean no visits. Gradient norms are recorded before
the unchanged clipping operation. Original SB3 KL, clipping and explained-variance
logs are retained in the training table; SB3 logs training statistics from the
preceding update at the following rollout row. Deterministic action collapse
does not imply zero stochastic exploration. Selection uses maximum validation
discounted reward, earliest on ties, including final post-update evaluation.

## 9. Where is PPO regret concentrated?

These are genuinely PPO101-visited held-out states, never a synthetic grid:

| scenario | mean | median | p90 | p95 | lower | upper | weighted_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 6.38 | 0.00 | 35.12 | 60.54 | -4.32 | 19.63 | 2042.07 |
| severe_stress | 4.37 | 0.00 | 17.97 | 35.97 | 0.00 | 10.00 | 642.43 |

The largest weighted regional contributions are shown below. This exploratory
ranking is computed after evaluation and does not select policies or hyperparameters.

| scenario | pd_bucket | utilization_bucket | states | mean | lower | upper | weighted_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2 | 2 | 9.00 | 53.66 | 0.00 | 129.06 | 1824.43 |
| baseline | 1 | 2 | 3.00 | 42.23 | 0.00 | 102.34 | 1309.04 |
| baseline | 0 | 1 | 5.00 | 0.56 | -0.36 | 1.44 | 33.76 |
| severe_stress | 2 | 2 | 12.00 | 12.47 | 0.42 | 29.31 | 642.43 |
| severe_stress | 0 | 0 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| severe_stress | 0 | 1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |

These magnitudes need not reproduce Phase A's pooled-policy planning opportunity:
they condition on PPO visitation, use a first-action PPO comparison rather than
the myopic/full-horizon disagreement estimand, and have a different independent
sample and MC budget. A small PPO-occupancy regret does not erase Phase A's result.

The PD x utilization map and full CSV retain regional estimates. PD buckets are
<.2, [.2,.6), >=.6; utilization buckets are <.5, [.5,1), >=1. The teacher selects
on one half of CRN draws and evaluates on the other. Inverse inclusion weights
restore active visits in sampled source trajectories. Weighted total regret is
a sum of first-action counterfactual opportunities, **not** attainable cumulative
episode improvement. PPO202/303 same-state comparisons are not claimed to be
their own visitation distributions.

## 10. What is the dominant bottleneck?

**unresolved with current evidence.** The directional gate alone suggests primarily optimization-limited. However, statistically signed scenario reversals in ppo_gap prevent a scenario-general dominant-bottleneck claim; the overall classification is therefore unresolved.

The mechanical gate tests whether the paired 95% interval
for History-Observation, Observation-PPO or Full-History is wholly positive in
either named scenario. It retains scenario dependence and does not correct for
multiple comparisons. Supported gaps motivate experiments, not exclusive causal
diagnoses; nonsignificance does not establish equivalence or inaccessible value.

Recommendation: **A PPO optimization and representation experiment. Stratify by macro scenario: the supported gap reverses sign across scenarios.** No next-phase intervention is implemented.
Finite test cohorts, fixed continuation, horizon selection, supervised approximation
and limited MC precision remain potential explanations. The saved evidence must
be read jointly with these limitations rather than interpreting an approximate
planner as an information-theoretic bound.

## Reproduction and integrity

```shell
python -m credit_rl.experiments.information_gap
python -m credit_rl.experiments.information_ppo
python -m credit_rl.experiments.information_report --update-docs
```

The canonical standard artifacts must already exist. Use `--profile smoke`,
`--canonical-profile smoke`, a separate `--canonical` smoke folder and a separate
`--output` for software checks. All figures regenerate from CSV with the report
command alone. `protected_artifacts.json` verifies Phase A files and canonical
models remain byte-identical. Cached runs require matching protocol, canonical
profile, model and scientific-source identity. `information_gap --stage evaluate`
reuses fitted planners to reproduce the scientific tables without refitting.
Raw feature prefixes, MC draws and replay checkpoints stay local; compact tables,
figures and manifests are the reviewable scientific outputs.
