# Experiments

Install from the project root: `python -m pip install -e ".[dev,experiments]"`.
Then run `python -m experiments.trajectory_sanity`.

For the full Sprint 2 diagnostic, run
`python -m experiments.dgp_sanity --customers 5000 --seed 42`.
It runs one Markov cohort and six paired policy/scenario cohorts on the same initial
customers and indexed shocks. Horizon is in `configs/simulation.yaml`; macro effects,
matrix and phases are in `configs/macro_scenarios.yaml` (`--macro-config` overrides).
Results are in `outputs/results/dgp/`; six figures are in `outputs/figures/dgp/`.
The command fails if directional, stress, learnability or nondegeneracy checks fail.

The learnability check uses opening observations and next-month defaults, splitting
by customer. `DIAGNOSTIC_ONLY` files and `initial_customers.json` contain privileged
information and must not become policy features. Initial states, macro paths, named
shocks/seeds, config and source hashes support replay. Use `--output` for separate runs.
The older trajectory check now writes to `trajectory_sanity_v2` output subdirectories.

This fits the preserved snapshot PD classifier on 5,000 synthetic customers,
evaluates static, repeated +10%, repeated -10%, and risk-threshold policies on six
different customers, and saves trajectories, episode-level summaries, fitted PD
model, five diagnostic figures, and a run manifest. Paired policy runs share seeds.
The latent-risk comparison holds observed initial conditions fixed while changing
hidden creditworthiness/payment propensity. The macro check runs 200 paired
trajectories of a fixed customer in locked normal/stress regimes.

Inputs are `configs/simulation.yaml` and `configs/experiments.yaml`. Both can be
overridden with `--config`/`--experiments`. `--output` changes the output root.
Named +10%/-10% diagnostic policies require those multipliers in the configuration.
Re-running a command replaces that command's diagnostic outputs. Use a separate
`--output outputs/run-name` to retain multiple runs. Manifests contain the complete
resolved config, Python/package versions, source hashes and experiment settings.

For a small PPO integration test:

```shell
python -m pip install -e ".[dev,experiments,rl]"
python -m experiments.train_ppo --timesteps 256
```

This checks SB3 compatibility, trains briefly, saves/reloads a new model and evaluates
30 held-out customers against static and risk-threshold policies with common seeds.
Results are under `outputs/results/ppo_smoke_v2/`. The configurable default budget is
4,096 steps. These are smoke tests, not performance claims. Historical 11-feature
PPO checkpoints, and Sprint 1 15-feature checkpoints, cannot be loaded into the new
21-feature environment. PPO tuning is outside Sprint 2.

`legacy/` contains the original row-based workflow and its historical README.
Read `legacy/ARCHIVE.md` before interpreting or attempting to execute it.
