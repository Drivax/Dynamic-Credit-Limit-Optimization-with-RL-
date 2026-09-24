### Baseline

| policy | net_economic_value | revenue | credit_loss | default_rate | mean_limit | mean_utilization |
| --- | --- | --- | --- | --- | --- | --- |
| AlwaysDecrease20 | -779.428 | 385.293 | 1105.889 | 0.713 | 2794.781 | 1.098 |
| MyopicEconomic | -1285.245 | 1066.159 | 2191.299 | 0.570 | 11174.149 | 0.473 |
| PDThreshold | -880.473 | 862.116 | 1614.304 | 0.630 | 7189.632 | 0.708 |
| PPO | -779.428 | 385.293 | 1105.889 | 0.713 | 2794.781 | 1.098 |
| Static | -1379.340 | 1012.640 | 2237.257 | 0.640 | 7312.645 | 0.661 |

### Stress

| policy | net_economic_value | revenue | credit_loss | default_rate | mean_limit | mean_utilization |
| --- | --- | --- | --- | --- | --- | --- |
| AlwaysDecrease20 | -1280.840 | 301.699 | 1532.196 | 0.953 | 3423.447 | 1.074 |
| MyopicEconomic | -2430.560 | 596.680 | 2926.816 | 0.873 | 10290.866 | 0.495 |
| PDThreshold | -1953.059 | 477.893 | 2350.705 | 0.917 | 6452.382 | 0.736 |
| PPO | -1280.840 | 301.699 | 1532.196 | 0.953 | 3423.447 | 1.074 |
| Static | -2494.226 | 538.300 | 2940.479 | 0.913 | 7312.645 | 0.647 |

### Paired PPO differences (95% bootstrap)

| scenario | reference | metric | difference | lower | upper |
| --- | --- | --- | --- | --- | --- |
| baseline | Static | cumulative_reward | 3225.122 | 2821.154 | 3706.490 |
| baseline | Static | net_economic_value | 599.912 | 393.436 | 796.347 |
| baseline | Static | credit_loss | -1131.368 | -1301.271 | -955.382 |
| baseline | Static | defaulted | 0.073 | 0.037 | 0.115 |
| baseline | MyopicEconomic | cumulative_reward | 3156.257 | 2653.905 | 3670.037 |
| baseline | MyopicEconomic | net_economic_value | 505.817 | 282.939 | 706.396 |
| baseline | MyopicEconomic | credit_loss | -1085.410 | -1283.859 | -880.017 |
| baseline | MyopicEconomic | defaulted | 0.143 | 0.105 | 0.193 |
| severe_stress | Static | cumulative_reward | 2747.200 | 2505.704 | 3018.580 |
| severe_stress | Static | net_economic_value | 1213.386 | 1106.940 | 1331.399 |
| severe_stress | Static | credit_loss | -1408.284 | -1537.349 | -1288.953 |
| severe_stress | Static | defaulted | 0.040 | 0.020 | 0.060 |
| severe_stress | MyopicEconomic | cumulative_reward | 3012.293 | 2686.070 | 3330.624 |
| severe_stress | MyopicEconomic | net_economic_value | 1149.720 | 994.173 | 1294.821 |
| severe_stress | MyopicEconomic | credit_loss | -1394.620 | -1543.396 | -1224.009 |
| severe_stress | MyopicEconomic | defaulted | 0.080 | 0.052 | 0.112 |

### PD test performance

| model | roc_auc | pr_auc | brier | log_loss |
| --- | --- | --- | --- | --- |
| constant | 0.500 | 0.458 | 0.249 | 0.691 |
| logistic | 0.842 | 0.823 | 0.162 | 0.487 |
| logistic_calibrated | 0.842 | 0.823 | 0.162 | 0.486 |
| boosting | 0.846 | 0.831 | 0.161 | 0.485 |
| boosting_calibrated | 0.846 | 0.831 | 0.160 | 0.483 |
