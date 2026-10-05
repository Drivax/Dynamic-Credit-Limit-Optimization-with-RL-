Primary budget: **32,768 steps**. Values are synthetic EUR per customer; diversity is one minus the largest requested-action share. Default and diversity below refer to nominal.

| Policy | Baseline NEV | Stress NEV | Baseline reward | Stress reward | Default | Diversity | Worst-world NEV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static | -1211.2 | -2227.9 | -4269.0 | -4590.4 | 66.7% | 0.000 | -2227.9 |
| PDThreshold | -783.5 | -1800.5 | -3224.7 | -3755.3 | 64.0% | 0.507 | -1800.5 |
| MyopicEconomic | -1157.7 | -2146.2 | -4526.2 | -4756.3 | 56.0% | 0.498 | -2146.2 |
| AlwaysDecrease20 | -787.6 | -1245.7 | -2107.0 | -2545.0 | 75.3% | 0.313 | -1245.7 |
| CanonicalPPO | -787.6 | -1245.7 | -2107.0 | -2545.0 | 75.3% | 0.000 | -1245.7 |
| ObservationPlanner | -618.2 | -1201.1 | -2036.3 | -2556.0 | 67.3% | 0.435 | -1201.1 |
| BCInitPPO | -586.9 | -1182.3 | -2024.1 | -2567.8 | 65.3% | 0.467 | -1182.3 |
| BCRegularizedPPO | -586.8 | -1184.6 | -2010.4 | -2562.4 | 64.9% | 0.407 | -1184.6 |

Incremental NEV of BCRegularizedPPO over AlwaysDecrease20 (paired 95% intervals):

| Budget | World | Difference | 95% interval | Seeds | Customers |
|---:|---|---:|---:|---:|---:|
| 32,768 | nominal | 200.8 | [120.7, 293.0] | 5 | 150 |
| 32,768 | severe_stress | 61.1 | [32.8, 96.7] | 5 | 150 |
| 32,768 | worst_world | 61.1 | [32.3, 97.4] | 5 | 150 |
| 262,144 | nominal | 238.9 | [152.6, 338.7] | 5 | 150 |
| 262,144 | severe_stress | 59.6 | [31.1, 94.5] | 5 | 150 |
| 262,144 | worst_world | 59.6 | [31.9, 91.8] | 5 | 150 |

Validation selected **decay**, β₀=0.05. Mean validation reward: -1894.02; BC0: -1890.94. The latter comparison is descriptive and did not trigger additional tuning.

Intervals are marginal, conditional on the synthetic simulator, and do not establish real-bank population effects. Full intervals, stress interactions, both budgets and all five worlds are retained in the supporting CSVs.

Final classification: **SequentialValue: conditional; Learnability: initialization-sensitive; Robustness: scenario-dependent; ComplexityValue: conditionally justified.**

Regularization versus initialization alone (primary budget; paired 95% intervals):

| Scenario | NEV difference | Reward difference |
|---|---:|---:|
| nominal | 0.2 [-33.4, 42.5] | 13.7 [-10.2, 42.5] |
| severe_stress | -2.3 [-18.0, 13.0] | 5.4 [-6.4, 21.1] |

Learned sequential control adds net economic value over contraction across the declared worlds, but does not establish a reward improvement in every world. Its complexity is justified conditionally on the economic objective and synthetic setting, not by universal policy superiority. The additional BC regularizer does not establish an incremental NEV gain over initialization alone in both macro scenarios.
