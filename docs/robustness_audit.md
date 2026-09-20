# Audit before the uncertainty and OPE experiments

The recursive audit covered active `src/credit_rl`, experiments, configs, tests,
outputs and technical docs. Archived experiment code is isolated and not imported.
The working tree includes the completed policy benchmark and report; these are
preserved rather than reset.

* The DGP is RNG-free conditional on persistent customer traits, indexed monthly
  shocks and current macro. The next macro state is stored only after the transition;
  it is not used by that transition's default hazard.
* The 21-vector observation is an explicit public projection. The frozen logistic
  calibrated PD reads seven public historical snapshots. Its 12-month target is
  generated under a stochastic behavior policy, not under every evaluated policy.
* Requested commands are projected through limit caps and the optional severe-
  delinquency guardrail. Lower limits do not forgive debt. Recorded reward components
  preserve the identity between economic value, capital, penalties and reward.
* The existing evaluator supplies one shared path for all policies and returns
  episode histories with customer/month/policy/seed tags. A public transition callback
  is sufficient for OPE logging; a separate simulation loop is unnecessary.
* Saved PPO policies were selected on validation; all five seeds with and without PD
  remain available. No new training is needed to test nominal-policy misspecification.
* The completed nominal benchmark reports 108 passing tests and pathwise-equivalent
  PPO/constant -20% outcomes. Its conditional bootstrap does not cover DGP-world or
  common macro uncertainty. The claim is correctly limited to that experiment.
* `baseline_specs(config,...)` constructs the myopic policy using that configuration.
  Passing a shifted evaluation config here would silently grant it correct behavioral
  coefficients unavailable at nominal training. The new runner explicitly constructs
  all deployable policies with the nominal config, then shifts only the environment.
* The base config validates behavioral bounds but does not enforce all hidden-hazard
  coefficient signs. The world perturbation function adds that check without changing
  nominal dynamics.
* Existing risk alerts are ex post. The new study estimates their frequency and
  magnitude across worlds; it does not label this constrained optimization.
* Existing manifests record sources/configuration but the nominal resume key checks
  configuration and PD artifact rather than all source/model changes. New studies
  include core source hashes and each selected PPO artifact in the cache identity,
  and save exact YAML snapshots and world definitions before evaluation.

Ranges, anchors, OAT endpoints, macro timing, sensor severities, OPE behavior weights,
clipping, sample sizes and bootstrap schemes are specified in the two experiment
YAML files before final outcomes are inspected. No experiment changes nominal DGP
coefficients or refits PD/PPO to improve the reported rankings.
