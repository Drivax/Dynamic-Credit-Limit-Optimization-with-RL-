| Question | Measured answer |
| --- | --- |
| Q1: Is -20% immediately optimal everywhere? | No: weighted share 37.6%. |
| Q2: Does it remain universally optimal at long horizon? | No: 60.1% with contraction continuation; see other continuations. |
| Q3: Can an increase be better? | An increase wins in 33.1% at H1 and 23.2% at remaining horizon. |
| Q4: Are those states visited? | Yes: all 72 states are from simulated held-out trajectories, not a fabricated grid. |
| Q5: Do effects persist? | Yes: action_persistence.csv separates future rewards; persistent_state_effects.csv reports lagged exposure/purchase/survival effects. |
| Q6: Does the choice depend on state? | Yes: the largest remaining-horizon share is 60.1% under contraction continuation, far from all sampled visitation. |
| Q7: Does the preferred action depend on horizon? | Estimated one-step/remaining-horizon disagreement 43.6%; phase maps are descriptive. |
| Q8: Is myopic selection equivalent to multistep selection? | No for this full-state diagnostic: held-out gain EUR 58.47; held-out immediate sacrifice EUR 17.45 [7.84, 27.06]. This differs from the MyopicEconomic surrogate policy. |
| Q9: What PPO value comes from state-dependent effective decisions? | The canonical PPO/AlwaysDecrease20 effective equivalence remains; PPO has not demonstrated exploitation of this opportunity. See per-seed policy_collapse.csv. |
| Q10: Does the result justify RL? | It motivates studying observable-information planning, but does not validate PPO or establish that RL is necessary; privileged lookahead is not a fair-information optimum. |
