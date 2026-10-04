# Policy initialization and preservation — Phase D

The 64x64 public-state imitation passed the frozen validation gate before PPO. Only actor weights differ within paired runs; the random critic and canonical PPO objective are shared. The experiment distinguishes retention of nonconstant decisions, preservation of the fitted teacher boundary, and realized economic performance. These are not interchangeable outcomes.

## 1. Motivation from Phase C

Phase C found early greedy contraction, later stochastic concentration and partial late recovery. Critic, entropy and actor-capacity interventions did not identify a unique mechanism. Its imitation was only partial and did not reduce teacher regret; Phase D therefore first qualifies a new initialization.

## 2. Experimental design

The 64x64 public-state imitation passed the frozen validation gate before PPO. Only actor weights differ within paired runs; the random critic and canonical PPO objective are shared. The experiment distinguishes retention of nonconstant decisions, preservation of the fitted teacher boundary, and realized economic performance. These are not interchangeable outcomes.

Five paired seeds (101, 202, 303, 404, 505), independent 32,768 and 262,144-step runs, unchanged canonical training population and baseline-only checkpoint selection. Natural imitation visits come from equally allocated PPO, F0 and myopic customer trajectories, with 600/150/150 new train/validation/test customers per macro scenario. Both macro versions retain the same customer partition. Only the 21 public features enter the actor. All visits are retained without class balancing. Six hundred epochs, Adam .001, batch 256 and validation cross entropy select the imitation checkpoint. The teacher is frozen Phase B F0. The protocol and its SHA256 were saved before collection. No optional anchored loss or critic warm start was run.

Selected checkpoints retain canonical pre-pending-update timing at validation boundaries, plus a final post-update evaluation. Temporal snapshots are explicitly post-update. Stochastic evaluation uses local customer/month uniforms shared across arms. Endogenous trajectories and termination times can diverge. The primary estimand is the imitation-minus-random difference for validation-selected policies on held-out customers. Seed/customer bootstrap intervals are descriptive, unadjusted for correlated endpoints.

## 3. Imitation qualification

The gate uses validation only: significant effective-agreement gain, significant fitted-teacher-regret reduction, diversity >=.05 and at least two effective actions with >=1% share in at least four seeds, separately in both scenarios. The 64x64 architecture passed; the 128x128 fallback was not used.

| scenario | agreement_gain | agreement_low | agreement_high | regret_change | regret_low | regret_high | diverse_seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.1320 | 0.0871 | 0.1790 | -5.1759 | -7.1471 | -3.4615 | 5.0000 |
| severe_stress | 0.0499 | 0.0120 | 0.0858 | -2.5420 | -4.0859 | -0.9753 | 5.0000 |

Test fidelity was measured only after architecture selection and short PPO training. See `imitation_test_quality.csv`; it did not change the decision. Higher agreement does not certify the teacher as optimal.

Independent test fidelity (seed means):

| experiment_id | scenario | requested_accuracy | effective_accuracy | teacher_regret | diversity |
| --- | --- | --- | --- | --- | --- |
| Imitation_64x64 | baseline | 0.7742 | 0.7761 | 3.2058 | 0.4295 |
| Imitation_64x64 | severe_stress | 0.8688 | 0.8706 | 2.5726 | 0.2099 |


## 4. Initial decision boundary

| experiment_id | scenario | diversity | teacher_regret | agreement | entropy |
| --- | --- | --- | --- | --- | --- |
| ImitationInit | baseline | 0.3846 | 3.3139 | 0.8222 | 0.4322 |
| ImitationInit | severe_stress | 0.2142 | 2.4537 | 0.8711 | 0.2636 |
| RandomInit | baseline | 0.3197 | 158.4917 | 0.1472 | 1.6094 |
| RandomInit | severe_stress | 0.2437 | 233.7433 | 0.0887 | 1.6094 |

These are fixed natural-validation-state means, not whole-policy values. Both deterministic and stochastic t=0 economic evaluations were saved before the first PPO update in each run. Actor probabilities/logits and greedy actions match the supervised actor numerically; critic tensors remain identical within pairs.

| experiment_id | scenario | mode | discounted_reward | net_economic_value |
| --- | --- | --- | --- | --- |
| ImitationInit | baseline | deterministic | -1936.0561 | -578.0920 |
| ImitationInit | baseline | stochastic | -1924.1247 | -564.8446 |
| ImitationInit | severe_stress | deterministic | -2402.4885 | -1070.4812 |
| ImitationInit | severe_stress | stochastic | -2407.6651 | -1071.3135 |
| RandomInit | baseline | deterministic | -3996.1077 | -1220.5637 |
| RandomInit | baseline | stochastic | -3622.8434 | -1137.5158 |
| RandomInit | severe_stress | deterministic | -4426.0032 | -2073.6235 |
| RandomInit | severe_stress | stochastic | -4203.3302 | -2045.4944 |


## 5. Random-init PPO trajectory

| budget | experiment_id | scenario | diversity | teacher_regret | agreement | deterministic_contraction | stochastic_contraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32768.0000 | RandomInit | baseline | 0.0000 | 8.4897 | 0.6182 | 1.0000 | 0.9571 |
| 32768.0000 | RandomInit | severe_stress | 0.0000 | 4.9956 | 0.7773 | 1.0000 | 0.9539 |
| 262144.0000 | RandomInit | baseline | 0.2373 | 8.6139 | 0.5764 | 0.7627 | 0.7253 |
| 262144.0000 | RandomInit | severe_stress | 0.0863 | 5.1502 | 0.7605 | 0.9137 | 0.8726 |

The random actor is evaluated before any update as well. Early greedy and stochastic contraction must be distinguished: the former describes argmax decisions, the latter the mean contraction probability. `collapse_timing.csv` reports first crossings and censoring, not an absorbing-state assumption.

## 6. Imitation-init PPO trajectory

| budget | experiment_id | scenario | diversity | teacher_regret | agreement | deterministic_contraction | stochastic_contraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32768.0000 | ImitationInit | baseline | 0.3234 | 4.8679 | 0.7412 | 0.6766 | 0.6592 |
| 32768.0000 | ImitationInit | severe_stress | 0.2258 | 3.7556 | 0.8315 | 0.7742 | 0.7683 |
| 262144.0000 | ImitationInit | baseline | 0.3568 | 9.3069 | 0.6330 | 0.6432 | 0.6185 |
| 262144.0000 | ImitationInit | severe_stress | 0.1702 | 7.0276 | 0.7773 | 0.8298 | 0.8121 |

Compare these final policies with initialization and the separately selected checkpoints. An actor can remain state-dependent while losing some teacher agreement. Random initialization is not the only trajectory compatible with the unchanged PPO objective.

## 7. Does PPO preserve a useful policy?

Preservation loss is fitted teacher regret at t minus regret at t=0 on the same states. Positive values indicate teacher-boundary erosion; they do not alone demonstrate worse simulator performance.

| budget | scenario | metric | mean | low | high |
| --- | --- | --- | --- | --- | --- |
| 32768.0000 | baseline | preservation_loss | 1.5540 | 0.7943 | 2.2492 |
| 32768.0000 | baseline | stochastic_preservation_loss | 0.5884 | -0.1357 | 1.2573 |
| 32768.0000 | severe_stress | preservation_loss | 1.3019 | 0.0133 | 2.4010 |
| 32768.0000 | severe_stress | stochastic_preservation_loss | 0.7504 | -0.3102 | 1.6974 |
| 262144.0000 | baseline | preservation_loss | 5.9930 | 3.2429 | 9.1410 |
| 262144.0000 | baseline | stochastic_preservation_loss | 5.7043 | 2.9035 | 8.8778 |
| 262144.0000 | severe_stress | preservation_loss | 4.5740 | 2.3830 | 7.6106 |
| 262144.0000 | severe_stress | stochastic_preservation_loss | 4.5771 | 2.1144 | 7.8316 |

The four interpretive quadrants are retained: teacher preserved/economics preserved; teacher lost/economics improved; teacher preserved/economics worsened; teacher lost/economics worsened. Near-zero confidence intervals are inconclusive, not proof of equivalence. Reward and NEV can place a trajectory in different quadrants.

Final imitation-initialized policy minus its own initialization on fixed validation customers (economic changes, EUR/customer; paired seed/customer intervals):

| budget | scenario | metric | mean | low | high |
| --- | --- | --- | --- | --- | --- |
| 32768.0000 | baseline | discounted_reward | 33.3049 | -0.5445 | 88.5727 |
| 32768.0000 | baseline | net_economic_value | 24.4151 | -18.9574 | 92.7527 |
| 32768.0000 | severe_stress | discounted_reward | -13.2492 | -28.2185 | -0.0753 |
| 32768.0000 | severe_stress | net_economic_value | -1.5737 | -19.8356 | 11.9624 |
| 262144.0000 | baseline | discounted_reward | 82.5638 | 27.2028 | 156.7880 |
| 262144.0000 | baseline | net_economic_value | 107.7017 | 38.5390 | 200.6101 |
| 262144.0000 | severe_stress | discounted_reward | -14.0971 | -48.4007 | 11.4693 |
| 262144.0000 | severe_stress | net_economic_value | -13.4255 | -53.0918 | 20.5805 |


## 8. When does policy drift begin?

| scenario | timesteps | kl_initial | action_flip | preservation_loss | deterministic_contraction |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6154 |
| baseline | 512.0000 | 0.0047 | 0.0235 | 0.0352 | 0.6101 |
| baseline | 1024.0000 | 0.0128 | 0.0390 | 0.1060 | 0.6072 |
| baseline | 2048.0000 | 0.0232 | 0.0567 | 0.3270 | 0.6062 |
| baseline | 4096.0000 | 0.0486 | 0.0833 | 0.3374 | 0.6129 |
| baseline | 8192.0000 | 0.0963 | 0.1278 | 0.7420 | 0.6316 |
| severe_stress | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7858 |
| severe_stress | 512.0000 | 0.0044 | 0.0182 | 0.0059 | 0.7792 |
| severe_stress | 1024.0000 | 0.0111 | 0.0274 | 0.1390 | 0.7757 |
| severe_stress | 2048.0000 | 0.0231 | 0.0440 | 0.4024 | 0.7764 |
| severe_stress | 4096.0000 | 0.0422 | 0.0596 | 0.4900 | 0.7697 |
| severe_stress | 8192.0000 | 0.0727 | 0.0847 | 0.6226 | 0.7636 |

KL is KL(pi_t || pi_0); the previous-snapshot KL and flip fraction are in `policy_drift.csv`. Intervals between snapshots localize changes, not the exact optimization step.

## 9. Which action regions are lost first?

| budget | scenario | teacher_region | eligible_states | boundary_survival |
| --- | --- | --- | --- | --- |
| 32768.0000 | baseline | contraction | 6101.0000 | 0.9087 |
| 32768.0000 | baseline | hold | 1142.0000 | 0.8231 |
| 32768.0000 | baseline | increase | 597.0000 | 0.2194 |
| 32768.0000 | severe_stress | contraction | 4539.0000 | 0.9339 |
| 32768.0000 | severe_stress | hold | 387.0000 | 0.9354 |
| 32768.0000 | severe_stress | increase | 100.0000 | 0.2500 |
| 262144.0000 | baseline | contraction | 6101.0000 | 0.8413 |
| 262144.0000 | baseline | hold | 1142.0000 | 0.2925 |
| 262144.0000 | baseline | increase | 597.0000 | 0.3132 |
| 262144.0000 | severe_stress | contraction | 4539.0000 | 0.9172 |
| 262144.0000 | severe_stress | hold | 387.0000 | 0.2972 |
| 262144.0000 | severe_stress | increase | 100.0000 | 0.3700 |

Survival conditions on initial requested-action agreement. Ineligible states are missing, not failures. Low/medium/high confidence uses validation effective-action Q-gap terciles. The CSV retains counts and confidence strata; sparse strata should not be treated as equally precise evidence.

## 10. Advantage pressure on teacher-preferred actions

Actual rollout files retain raw GAE, returns, pre-update critic values and observations; minibatch files retain actual normalized advantage sums by sampled action. Diagnostic joins label teacher-preferred actions after training. No labels or Q values enter PPO. Fixed-panel probes below use the previous snapshot and independent simulated continuation to estimate preceding teacher-action GAE; they are not the actual minibatch signal for those fixed states.

| experiment_id | scenario | teacher_action | preceding_teacher_gae | preceding_teacher_mc_advantage |
| --- | --- | --- | --- | --- |
| ImitationInit | baseline | 0.0000 | -2273.4340 | -2.9573 |
| ImitationInit | baseline | 2.0000 | 200.6533 | 0.2278 |
| ImitationInit | baseline | 4.0000 | 269.3514 | -1.9316 |
| ImitationInit | severe_stress | 0.0000 | -768.5428 | 2.2696 |
| RandomInit | baseline | 0.0000 | -2837.4305 | 257.3460 |
| RandomInit | baseline | 2.0000 | 598.0005 | 6.7237 |
| RandomInit | baseline | 4.0000 | 808.0273 | 10.1532 |
| RandomInit | severe_stress | 0.0000 | -779.4123 | 203.1908 |

Do not infer causal update pressure from the sign of raw GAE alone: PPO centers and scales advantages per minibatch. Probe and actual-rollout distributions differ.

The original random MC panel lacked stress hold/increase coverage. An explicitly post-outcome, validation-only diagnostic supplement samples up to four states per available teacher action and scenario. It changes neither the original bank nor any headline estimand. This is exploratory coverage repair, not preregistered confirmation. Missing action support remains missing.

| scenario | teacher_action | available_states | selected_states |
| --- | --- | --- | --- |
| baseline | 0.0000 | 1179.0000 | 4.0000 |
| baseline | 1.0000 | 228.0000 | 4.0000 |
| baseline | 2.0000 | 303.0000 | 4.0000 |
| baseline | 3.0000 | 48.0000 | 4.0000 |
| baseline | 4.0000 | 149.0000 | 4.0000 |
| severe_stress | 0.0000 | 897.0000 | 4.0000 |
| severe_stress | 1.0000 | 97.0000 | 4.0000 |
| severe_stress | 2.0000 | 109.0000 | 4.0000 |
| severe_stress | 3.0000 | 17.0000 | 4.0000 |
| severe_stress | 4.0000 | 34.0000 | 4.0000 |

| scenario | teacher_action | preceding_teacher_gae | preceding_teacher_mc_advantage | critic_error |
| --- | --- | --- | --- | --- |
| baseline | 0.0000 | -1430.6806 | 12.1872 | 1502.1882 |
| baseline | 1.0000 | -175.2334 | -8.4902 | 136.6083 |
| baseline | 2.0000 | 368.6304 | 1.3818 | -483.9721 |
| baseline | 3.0000 | 348.7908 | -3.3544 | -418.0367 |
| baseline | 4.0000 | 289.4538 | -0.1301 | -345.5585 |
| severe_stress | 0.0000 | -1188.6634 | 0.6124 | 1267.1339 |
| severe_stress | 1.0000 | -55.1761 | 11.0811 | 26.3297 |
| severe_stress | 2.0000 | 354.3066 | -0.0309 | -429.9420 |
| severe_stress | 3.0000 | 369.0464 | 3.7699 | -435.4915 |
| severe_stress | 4.0000 | 318.8850 | -0.2483 | -389.0229 |


Actual normalized PPO advantages on sampled teacher-matching actions, steps 0–8,192; 95% seed-cluster bootstrap intervals. This is Markov training occupancy, not separate baseline/stress evaluation populations. Repeated samples across five epochs are not treated as independent. Labels are joined after the diagnostic replay; all selected and final tensors must equal the original runs.

| experiment_id | teacher_action | seed_count | normalized_mean | low | high | negative_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| ImitationInit | 0.0000 | 5.0000 | -0.1149 | -0.1399 | -0.0970 | 0.4466 |
| ImitationInit | 1.0000 | 5.0000 | -0.0382 | -0.0741 | -0.0038 | 0.3881 |
| ImitationInit | 2.0000 | 5.0000 | 0.3427 | 0.3215 | 0.3656 | 0.2487 |
| ImitationInit | 3.0000 | 5.0000 | 0.1694 | 0.1042 | 0.2300 | 0.3748 |
| ImitationInit | 4.0000 | 5.0000 | 0.1346 | 0.0198 | 0.2477 | 0.3927 |
| RandomInit | 0.0000 | 5.0000 | 0.0308 | 0.0145 | 0.0475 | 0.4140 |
| RandomInit | 1.0000 | 5.0000 | 0.1274 | 0.0788 | 0.1577 | 0.3400 |
| RandomInit | 2.0000 | 5.0000 | 0.3733 | 0.3259 | 0.4363 | 0.2431 |
| RandomInit | 3.0000 | 5.0000 | 0.4424 | 0.3472 | 0.5414 | 0.2113 |
| RandomInit | 4.0000 | 5.0000 | 0.4396 | 0.3606 | 0.5157 | 0.2359 |


All three non-contraction teacher-preferred actions have positive mean normalized advantages with seed-cluster intervals above zero. These observations do not support uniformly negative average PPO pressure on hold/+10%/+20%. Individual negative advantages remain common; conditional means over visited states neither determine every held-out decision nor isolate the effect of shared actor parameters, clipping and changes in occupancy.

## 11. Short-budget results

Imitation minus random, held-out customers, validation-selected checkpoint; 95% paired seed/customer intervals. Units are EUR per customer.

| mode | metric | scenario | mean | low | high |
| --- | --- | --- | --- | --- | --- |
| deterministic | discounted_reward | baseline | 99.3344 | 41.9437 | 165.5620 |
| deterministic | discounted_reward | severe_stress | -35.9209 | -66.7828 | -5.5938 |
| deterministic | net_economic_value | baseline | 214.6763 | 143.2867 | 300.8763 |
| deterministic | net_economic_value | severe_stress | 72.4043 | 44.8037 | 105.6874 |
| stochastic | discounted_reward | baseline | 267.3137 | 181.0123 | 363.1647 |
| stochastic | discounted_reward | severe_stress | 241.1430 | 159.7717 | 333.1026 |
| stochastic | net_economic_value | baseline | 222.7990 | 142.5444 | 319.2445 |
| stochastic | net_economic_value | severe_stress | 195.4470 | 143.2514 | 247.8780 |

The final-policy estimates are retained separately in `statistical_comparisons.csv`. Temporal snapshots were never ranked using test outcomes.

## 12. Long-budget results

Imitation minus random, held-out customers, validation-selected checkpoint; 95% paired seed/customer intervals. Units are EUR per customer.

| mode | metric | scenario | mean | low | high |
| --- | --- | --- | --- | --- | --- |
| deterministic | discounted_reward | baseline | 75.5848 | 18.2889 | 153.4753 |
| deterministic | discounted_reward | severe_stress | -19.8631 | -50.2052 | 7.5360 |
| deterministic | net_economic_value | baseline | 155.0456 | 74.1619 | 255.4068 |
| deterministic | net_economic_value | severe_stress | 29.9000 | -3.1661 | 67.3494 |
| stochastic | discounted_reward | baseline | 89.7084 | 35.3938 | 152.5431 |
| stochastic | discounted_reward | severe_stress | 12.9422 | -21.9476 | 49.0864 |
| stochastic | net_economic_value | baseline | 150.0632 | 83.4698 | 229.8119 |
| stochastic | net_economic_value | severe_stress | 44.2843 | 13.8654 | 80.4017 |

The final-policy estimates are retained separately in `statistical_comparisons.csv`. Temporal snapshots were never ranked using test outcomes.

## 13. Baseline vs severe-stress interaction

Stress-minus-baseline treatment effect, matched seed/customer resampling:

| budget | metric | mean | low | high |
| --- | --- | --- | --- | --- |
| 32768.0000 | discounted_reward | -135.2553 | -208.2227 | -74.5046 |
| 262144.0000 | discounted_reward | -95.4479 | -170.4479 | -34.2057 |

A positive NEV difference alongside a negative reward difference is possible because reward includes capital and constraint penalties beyond NEV. Neither endpoint replaces the other.

## 14. Discovery vs preservation

The short-budget baseline comparison supports an initialization-sensitive discovery limitation: a useful observable boundary can remain nonconstant under the same PPO objective that yields contraction from random initialization. This does not establish exact teacher-boundary preservation or universal economic improvement under stress. Erosion and retention can coexist. Long-run similarity is descriptive: no equivalence margin was preregistered, so absence of a significant difference cannot establish LongRunConvergence.

## 15. Mechanistic interpretation

The randomized paired comparison changes actor initialization only. It identifies sensitivity to that initial condition, not a unique decomposition into critic error, optimization curvature or advantage noise. Critic errors, actual action-conditioned GAE, clipping, entropy, KL and preclip gradient norms are retained in each run. Independent counterfactual comparisons use H=12, the same frozen canonical continuation, common action draws and separate selection/evaluation banks. They measure first-action opportunity, not lifetime value or an oracle policy. No post-outcome PPO tuning or auxiliary objective was introduced.

## 16. Limitations

Synthetic credit environment; fitted public teacher; five training seeds; one new customer cohort; natural mixture rather than target-policy occupancy; finite and small MC panel; unadjusted correlated intervals; selected-policy and post-update timing differences; baseline-only canonical selection; deterministic/stochastic estimand differences. Confidence-gap estimates inherit teacher uncertainty. The fixed-panel GAE probes are model-based diagnostics, not measured gradients of those exact states. Imitation uses 13,537 extra labeled training visits and 600 supervised epochs outside the PPO step budget; improved PPO-step sample efficiency is not an end-to-end cost-matched learning comparison. Epoch selection and qualification share validation data, so qualification intervals can be optimistic; independent test fidelity is reported without reopening model selection. Pooled boundary survival weights initially agreeing states; sparse strata require their denominators. Test data were not used to qualify the initialization or choose checkpoints. The A/B/C and canonical artifact protection manifest is checked separately. These conclusions apply to this protocol, not deployed lending decisions.

## 17. Decision gate for next phase

**InitializationSensitive, ScenarioDependent, DiscoveryLimited, Unresolved**.

DiscoveryLimited, when present, is restricted to short-budget baseline behavior; not a uniquely identified universal cause. Teacher fidelity can erode without full collapse; stress reward and net economic value are different objectives. No equivalence margin was preregistered for long-run convergence.

PPO can retain useful observable state dependence after imitation initialization in this experiment. That does not imply zero erosion of the initial boundary, nor uniform reward gains under macro stress. Phase D ends here; no further phase or economic-model change is implemented.

## Main figures

Seed ranges on trajectory plots are descriptive, not confidence intervals. Decision maps are projections of observed states, not controlled two-dimensional decision slices.

![teacher vs initial](../outputs/main/policy_initialization/figures/teacher_vs_initial.png)

![teacher regret over time](../outputs/main/policy_initialization/figures/teacher_regret_over_time.png)

![contraction share over time](../outputs/main/policy_initialization/figures/contraction_share_over_time.png)

![random vs imitation](../outputs/main/policy_initialization/figures/random_vs_imitation.png)

![boundary survival](../outputs/main/policy_initialization/figures/boundary_survival.png)

![validation economics](../outputs/main/policy_initialization/figures/validation_economics.png)

![baseline vs stress](../outputs/main/policy_initialization/figures/baseline_vs_stress.png)

![decision maps over time](../outputs/main/policy_initialization/figures/decision_maps_over_time.png)
