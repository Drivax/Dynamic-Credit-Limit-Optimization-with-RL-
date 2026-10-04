# Phase D execution and reproducibility

All commands run locally from the repository root with `.venv/Scripts/python.exe`.
They write exclusively to the dedicated Phase D outputs and the explicitly
published Phase D documentation. No PR, push, environment/reward change or
teacher refit is part of this workflow.

```powershell
.venv\Scripts\python.exe -m credit_rl.experiments.policy_initialization --stage audit
.venv\Scripts\python.exe -m credit_rl.experiments.policy_initialization --stage imitate
.venv\Scripts\python.exe -m credit_rl.experiments.policy_initialization --stage train --workers 3
.venv\Scripts\python.exe -m credit_rl.experiments.initialization_finalize --publish
```

Use `--profile smoke` for smoke outputs. Smoke deliberately exercises transfer
and training even if its three-epoch imitation does not qualify; every such run
is marked `smoke_unqualified`. These are engineering checks, not scientific
evidence. Standard training refuses unqualified imitation. The single fallback
architecture is attempted only if the frozen validation criterion fails.

The first command records the Phase C audit, scientific-input hashes and complete
canonical/Phase A/B/C standard artifact hashes. A separate supplemental manifest
protects the pre-existing smoke outputs. Historical README and technical-paper
prefix hashes permit only append-only publication. The 21D actor inputs and each
customer-disjoint partition are stored separately. Test arrays are not opened
by imitation fitting or qualification.

The `train` command runs all short-budget pairs, analyzes their temporal snapshots
and held-out endpoints, then runs the extended-budget pairs. Parallel workers
operate on distinct seed pairs. Completed run markers prevent unnecessary
retraining. An interrupted incomplete run restarts from its fixed initial seed;
partially written diagnostics are not counted as completed results. Do not run
two copies of the workflow on the same output directory at once.

`InitializedPPO.train` delegates to unchanged SB3 PPO and only then calls the
diagnostic recorder. Actor and critic are separate networks. Loading reference
models preserves NumPy, Python and Torch RNG state. Transfer changes only the
actor network and action head; no critic pretraining or optimizer-state transfer
is used. Canonical validation selection remains unchanged, including boundary
evaluations before the pending update and the final evaluation after training.
The `temporal_*` files always identify post-update models (except t=0).

Raw rollout files retain the first 8,192 steps. Actual minibatch normalized
advantages are summarized by sampled action; teacher conditioning is added to
the raw rollout observations only after training. Fixed-state teacher-action GAE
probes use the preceding stored snapshot, which can be several PPO updates back.
They must not be described as actual minibatch gradients for those fixed states.
Normalized log-probabilities are the recorded logits. Sampled stochastic
evaluation uses local customer/month uniforms; training RNG is not consumed.

`initialization_finalize` rebuilds aggregation tables, adds actual-advantage and
sample-efficiency summaries, writes the report, generates figures exclusively
from CSVs, then verifies hashes and numerical controls. Verification checks
canonical selected/final weight equality for short-budget controls, identical
paired critics, short/long prefix equality, snapshot labels and hashes, split
separation, original document prefixes, and byte-identical figure regeneration.

All confidence intervals resample paired seeds and customer clusters; visit-level
endpoints retain the natural visit weighting. Diversity uses paired seed-only
intervals. A selected endpoint is compared at its selected checkpoint even when
the two arms select different steps (`timesteps=-1` in paired summary tables).
Temporal intervals are descriptive and are not independent repeated tests.
First-passage records retain censoring and recrossings. No equivalence margin was
preregistered, so a confidence interval spanning zero does not prove preservation
or long-run convergence.

Validation:

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pytest --cov=credit_rl --cov-report=term-missing --cov-fail-under=70
.venv\Scripts\python.exe -m credit_rl.experiments.initialization_verify --profile smoke
.venv\Scripts\python.exe -m credit_rl.experiments.initialization_verify
```

The integration tests replay a tiny paired training experiment twice and compare
final tensors, recompute saved smoke analyses in a temporary directory, and
regenerate the CSV figures. They do not overwrite Phase A/B/C outputs. All
negative results remain in the run and analysis directories.
