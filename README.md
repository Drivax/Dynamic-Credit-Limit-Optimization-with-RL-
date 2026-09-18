# Dynamic credit-limit optimization under credit risk

A simulation and sequential decision-making framework for dynamic credit-limit
optimization. Each episode follows the **same customer over multiple months**.
An action changes their credit limit, which affects subsequent payment, spending,
utilization, exposure and risk. Persistent hidden customer traits and exogenous
macro conditions make this a partially observed sequential decision problem.

```text
customer state -> imperfect risk estimate -> decision policy -> limit adjustment
      ^                                                        |
      |              new behavior, exposure and economics <----+
      +----------------------- next month ---------------------+
```

This is a research simulator with **assumed, uncalibrated behavioral dynamics**.
It is not a validated representation of real bank customers, a lending recommendation
system, or evidence that RL outperforms simpler policies. The priority is defensible
experimental mechanics and honest comparisons.

## Run locally

Python 3.11+ is required. From the repository root:

```shell
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,experiments]"
python -m pytest
python -m experiments.trajectory_sanity
```

If the Windows `python` Store alias is unavailable, use your installed Python's full
path to create the environment. Activation is optional: use `.venv\Scripts\python.exe`
directly. On Linux/macOS activate with `source .venv/bin/activate`.

Optional PPO integration (no hyperparameter search):

```shell
python -m pip install -e ".[rl]"
python -m experiments.train_ppo --timesteps 256
```

See [experiments](experiments/README.md) for configs, output files and the limits of
these smoke runs. All new generated figures/models/results go under ignored `outputs/`.

## What changed in Sprint 1

- One episode is a customer trajectory, normally 24 months, ending early on default.
- Limit adjustments [-20%, -10%, 0%, +10%, +20%] compound on the current limit.
- Payments depend on prior payment, debt burden, utilization, delinquency and hidden
  traits. Spending depends on prior spending, income, available credit and shocks.
- Default uses a separate hidden DGP, **not the model's predicted PD**. Default is
  terminal; the horizon is a Gymnasium truncation.
- Default loss is charged once. Named interest, fee, funding, capital and soft-risk
  components make reward economics inspectable.
- Observation-only baselines, trajectory dataframes, seed-isolated random streams,
  Gymnasium checks and targeted tests support reproducible comparisons.

The original synthetic generator and Gradient Boosting PD trainer are retained.
A clean adapter updates their observed features monthly. This PD model has a known
distribution mismatch with the new DGP; its snapshot validation metrics do not imply
longitudinal calibration. A lightweight observable-only logistic proxy is also provided.

## Structure

```text
configs/                      # complete simulator assumptions and run settings
src/credit_rl/
  config.py                   # validated parameter dataclasses and YAML loading
  envs/                       # Gymnasium API, observation boundary, history
  simulation/                 # customers, traits, transitions, trajectory runner
  risk/                       # PD interface, adapter and preserved GB training
  reward.py                   # named monthly economic components
  policies/                   # static, constant, risk-threshold and SB3 adapters
  evaluation/                 # customer-level return/default summaries
  utils/                      # local RNG streams and training seed utility
experiments/                  # trajectory sanity checks and short PPO integration
experiments/legacy/           # frozen row-based prototype, including original README
tests/                        # transition, economics, leakage and reproducibility tests
data/README.md                # synthetic-data provenance
outputs/legacy/               # preserved historical results and local artifacts
docs/                        # environment specification, audit and sprint report
```

No notebooks or external raw datasets were present; empty placeholder directories
were deliberately omitted. The package uses a `src/` layout and editable installation,
without `sys.path` manipulation. Core simulation does not require Torch or sklearn.

## Scientific status

The original README's policy snapshot showed PPO underperforming static (-1591.90
versus -1520.64 average reward); some other saved runs showed all-increase policies
with improved one-step rewards. These historical results and their conflicting
snapshots remain in the archive. They came from a row-based environment and must
not be presented as longitudinal results. The old architecture could not support its
claimed customer-level sequential behavior.

Current diagnostic runs establish mechanics, not policy superiority. Remaining gaps
include empirical calibration, PD validation on independent trajectories, stronger
baselines, multi-seed uncertainty, a full billing/arrears ledger, terminal exposure
valuation, and correlated portfolio macro scenarios. No PPO optimization, constrained
RL or infrastructure work is part of this sprint.

Read the [environment specification](docs/environment.md), [repository audit](docs/repository_audit.md),
[sprint report](docs/sprint1_report.md), and [data provenance](data/README.md).
