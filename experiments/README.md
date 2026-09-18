# Experiments

Install from the project root: `python -m pip install -e ".[dev,experiments]"`.
Then run `python -m experiments.trajectory_sanity`.

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
Results are under `outputs/results/ppo_smoke/`. The configurable default budget is
4,096 steps. These are smoke tests, not performance claims. Historical 11-feature
PPO checkpoints cannot be loaded into the new 15-feature environment.

`legacy/` contains the original row-based workflow and its historical README.
Read `legacy/ARCHIVE.md` before interpreting or attempting to execute it.
