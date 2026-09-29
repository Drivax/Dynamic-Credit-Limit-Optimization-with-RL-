# Phase C implementation and technical corrections

The experiment matrix in `configs/ppo_diagnostics.yaml` is frozen in each output's
`preregistration.json`. No treatment, seed, budget, test population or confirmation
eligibility threshold was changed after observing outcomes. Training stays in the
existing DGP with the existing economic reward. This note distinguishes technical
implementation work from changes to the scientific design.

- PPO overrides now reach both the training environment and checkpoint evaluator.
  This is necessary for actual reward-scale and training-discount treatments.
  With no override, selected and final canonical weights reproduce exactly.
- Larger actor/critic experiments copy the unchanged component from the matching
  seed's canonical initial checkpoint. Loading an SB3 checkpoint can reset random
  generators, so Python, NumPy and Torch generator states are preserved around
  this load. Reference loading therefore does not accidentally change the
  unchanged network or introduce an extra training RNG reset. Initializing a
  different-sized network itself consumes a different number of parameter draws;
  identical training action uniforms across architectures are not asserted.
  Customer paths and held-out evaluation shocks remain paired.
- C3 uses a separate Adam optimizer for five ordered, critic-only passes over
  the current rollout returns. It adds no actor updates and consumes no sampling
  RNG. The original joint PPO training call remains unchanged.
- Actual rollout GAE is matched to the checkpoint **before** the rollout. A model
  saved after that update would not be its behavior critic. Independent frozen-
  state MC probes are explicitly different from actual finite-rollout GAE.
- Canonical validation selection at a rollout boundary happens before its pending
  PPO update; fine temporal snapshots happen after the update. The final callback
  additionally evaluates after the final update. Thus `selected` and `time_8192`
  can have the same step label but different weights. MC summaries retain
  `checkpoint_kind` and do not deduplicate or pool those two measurements. This
  correction to diagnostic aggregation changes no checkpoint or economic result.
  Dense rollout summaries (including critic error against GAE returns) refer to
  the behavior policy before the update; fixed-panel summaries refer to the
  policy after the update. The two are aligned by the completed rollout's step
  count rather than presented as simultaneous estimates from identical weights.
- Public teacher regret collapses legal requests with the same effective limit
  at floors, caps and delinquency guards. The earlier requested-action quantity
  is retained as `requested_teacher_regret`. Cached Phase C state tables were
  upgraded to contain both quantities; this did not retrain or select policies.
  Phase A/B reports and caches were not rewritten.
- A pandas read-only NumPy view initially prevented regional C4 summaries from
  completing. Copying that boolean mask fixes the exception without changing
  the Monte Carlo draws, fitted predictor or estimand.
- Selected actors remain the primary result. Final actors at every budget are
  additional diagnostics of recovery that might be hidden by earliest-best
  validation selection. No final actor replaces a selected actor on test results.
- `mc_action_opportunity.csv` and centered-GAE calibration are supplemental
  descriptive analyses. They are not factors, tuning criteria or inputs to the
  frozen confirmation selection. An estimated advantage above two MC standard
  errors is a diagnostic count, not a multiplicity-adjusted significance claim.

Each trained PPO run records its expanded settings, source hashes, versions,
reward/PD/checkpoint hashes and timestamp. Evaluation, imitation and the supervised
MC benchmark have separate provenance files. `ppo_verify` checks immutable inputs,
snapshot hashes, original canonical weights, common customer IDs, completion, and
byte-identical regeneration of CSV-derived tables and all thirteen figures.

All work is local. The ignored raw buffers, checkpoints and Monte Carlo caches
remain available in the Phase C output directories; compact tables and figures
are explicitly unignored for review. There is no PR, push, deployment or Phase D.
