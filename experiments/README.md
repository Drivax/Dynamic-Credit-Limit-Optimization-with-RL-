# Reproducible experiments

Run from the repository root after installing `.[dev,experiments]`.

## Longitudinal risk

```shell
python -m experiments.train_pd
python -m experiments.train_pd --evaluate-only
python -m experiments.pd_env_smoke
```

Training owns generation, point-in-time labels/features, customer-disjoint temporal cohorts, base models, dedicated calibration, cluster-bootstrap evaluation, stress/policy replays, artifacts and figures. Settings are in `configs/pd_model.yaml`; the DGP uses `configs/simulation.yaml` and its sibling macro configuration. Use `--output` for a separate run. Outputs go to `{models,results,figures}/pd` below it. Evaluation-only requires completed training outputs and never refits.

The environment smoke test loads the bundled artifact, runs ten customer episodes and compares every active online PD against offline feature construction. Use `--model` to select a different trusted local artifact.

## DGP checks

```shell
python -m experiments.dgp_sanity --customers 100 --seed 42 --output outputs/pd_dgp_check
```

This checked small run exercises structural checks, paired macro/policy simulations and figures. Larger populations improve precision. Simulator-only exports contain hidden hazards/traits/shocks and must never be used as PD features.

## Controlled policy benchmark

Install `.[dev,experiments,rl]`, generate the PD artifact above, then run:

```shell
python -m experiments.compare_policies --stage all
```

The stages `audit`, `smoke`, `train`, `evaluate`, and `figures` can also run separately. `train` selects thresholds and PPO hyperparameters on validation only, then trains five seeds with PD and five without PD. `evaluate` uses the frozen selected checkpoints on shared held-out customer scenarios; `figures` reads those traces without fitting. All settings are in `configs/policy_evaluation.yaml`. Completed training runs are reused only when their input fingerprint matches. Use a separate `--output` directory for different configurations.

Outputs live under `outputs/{models,results,figures}/policy_evaluation`. Large model files and trajectories remain local; compact results and report figures are versionable. See [methodology](../docs/policy_evaluation.md), [reward audit](../docs/policy_audit.md), and [results](../docs/policy_results.md).

## World robustness and logged-policy evaluation

These experiments require the completed nominal policy benchmark's selected agents and frozen PD artifact. They never retrain PPO automatically.

```shell
python -m experiments.robustness --profile smoke --stage evaluate
python -m experiments.robustness --profile smoke --stage report
python -m experiments.robustness --profile standard --stage evaluate
python -m experiments.robustness --profile standard --stage report
python -m experiments.off_policy_evaluation --profile smoke --stage evaluate
python -m experiments.off_policy_evaluation --profile smoke --stage report
python -m experiments.off_policy_evaluation --profile standard --stage evaluate
python -m experiments.off_policy_evaluation --profile standard --stage report
```

`--stage all` composes evaluation and reporting. `full` is the larger configured budget; standard is the measured local study. Robustness uses three independent Windows worker processes with one computation thread each. Configs are `configs/robustness.yaml` and `configs/ope.yaml`; registry snapshots live in `outputs/experiments/{robustness,ope}_<profile>`. Tables and figures are under `outputs/{results,figures}/{robustness,ope}/<profile>`. See [world methodology](../docs/robustness.md) and [OPE methodology](../docs/off_policy_evaluation.md).

## PPO integration smoke

With the optional `rl` dependencies installed:

```shell
python -m experiments.train_ppo --timesteps 256 --output outputs/pd_ppo_check
```

The default PD artifact is `outputs/models/pd/logistic_calibrated.joblib`; override with `--pd-model`. This command loads the longitudinal risk model and checks Gymnasium/SB3 compatibility, including saving/reloading PPO. It is a short integration check, not a policy performance claim. Risk thresholds and reward coefficients need economic validation for the supplied PD horizon.

`trajectory_sanity.py` remains an independent snapshot-adapter diagnostic; its snapshot targets and probabilities are not the longitudinal PD experiment. `legacy/` is isolated from the active package and is not part of reproduction.


## Risk-constrained portfolios

The portfolio study requires the frozen calibrated PD model, and trains its own 34-feature portfolio PPO agents. It does not require the independent-customer PPO checkpoints.

```shell
python -m experiments.portfolio_constraints --profile smoke --stage train
python -m experiments.portfolio_constraints --profile smoke --stage evaluate
python -m experiments.portfolio_constraints --profile smoke --stage ope
python -m experiments.portfolio_constraints --profile smoke --stage report
python -m experiments.portfolio_constraints --profile standard --stage train
python -m experiments.portfolio_constraints --profile standard --stage evaluate
python -m experiments.portfolio_constraints --profile standard --stage tail
python -m experiments.portfolio_constraints --profile standard --stage ope
python -m experiments.portfolio_constraints --profile standard --stage report
```

`--stage all` composes these stages. Three PPO formulations use every configured seed: structural guards only, EL penalty, and hard admission. Lambda and the risk-rule buffer are selected on validation only. Main held-out cases cross three risk budgets with normal/stress macro; separate cases cover PD errors, buffers, sizes, ordering and DGP worlds. The tail stage uses independent portfolios, while OPE logs complete coupled portfolios under a common hard constraint kernel.

Configs are `configs/portfolio.yaml` and `configs/constrained_policy.yaml`. Outputs are `outputs/{models,results,figures}/portfolio/<profile>` and `outputs/experiments/portfolio/<profile>`. Raw job completion files support resumable evaluation with frozen identities; model hashes are checked before evaluation. Preserve a completed run before changing its inputs. The full profile is a larger configured budget, not an automatically executed unit test.

[Portfolio equations and information boundary](../docs/portfolio_decisioning.md) explain why a hard admission rule cannot erase an inherited risk shortfall or guarantee next-month realized losses. The monthly predicted constraint, the independent hidden-DGP conditional-loss diagnostic and realized loss are separately recorded.
