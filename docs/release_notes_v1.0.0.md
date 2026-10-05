# v1.0.0 — Final research release

Prepared release notes; no tag or GitHub release has been published by this task.

## What this release contains

- A synthetic longitudinal credit environment with partially observed sequential decisions.
- A calibrated PD pipeline, rule-based controls and canonical PPO.
- PPO collapse, structural and information diagnostics.
- Observable-state behavior-cloning initialization and BC-regularized PPO.
- Macro stress, designed distribution shifts and off-policy evaluation.
- Frozen tables and figures, reproducibility manifests, tests and CI configuration.

## Main finding

Learned state-dependent control improves synthetic NEV over constant contraction across the declared worlds, but benefits are objective- and scenario-dependent. Additional BC regularization does not establish an incremental NEV gain over BC initialization alone. All schedules, seeds and negative results are retained.

SequentialValue: conditional; Learnability: initialization-sensitive; Robustness: scenario-dependent; ComplexityValue: conditionally justified.

## Scope

Fully synthetic methodological study. No real-bank calibration, deployment or regulatory claim. Confidence intervals are conditional on the declared simulator and evaluation design. Reward and NEV are different objectives.

## Reproduction and assets

A source checkout supports the portable smoke and the frozen documentation renderer. Standard reproduction requires additional frozen local research artifacts; these are not bundled as release assets. See [artifact inventory](release_assets.md). No raw-output archive or PDF is attached. The [scientific manifest](../outputs/main/final_manifest.json) identifies the measured products; the [packaging audit](release_audit.md) records release preparation separately.
