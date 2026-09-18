# Frozen row-based prototype archive

The other files here are preserved historical source, not current environment
documentation. Their methodological claims and existing bugs have not been silently
rewritten. The former top-level package, trainer, Streamlit demo, HPO/retraining and
action-collapse diagnostics were moved here together. `QUICK_FIX_HARD_PD_CONSTRAINTS.py`
is an incomplete integration sketch with missing imports, not an executable experiment.

Historical tracked outputs are in `../../outputs/legacy/results/`; previously ignored
local models/sample data/HPO artifacts are in `../../outputs/legacy/artifacts/`.
Their contents were preserved. Original relative `artifacts/` and `results/` paths
inside archival scripts refer to their former working directory and are not migrated
entry points. Reproduce the exact old layout from pre-sprint Git history if needed,
and supply the archived local assets separately. Do not run the archive as the new
workflow or assume its dependencies/results validate longitudinal-v1.

Known issues: independent rows per step; limit updates never fed into the next
state; defaults did not terminate customer trajectories; expected and realized loss
both subtracted; heterogeneous PD thresholds were soft penalties, not hard constraints;
unused risk stratification, reversed score-segment labels, unordered synthetic rows
called time-series evaluation; prefit risk model reused across folds; HPO selected
on test rewards while changing the reward objective; pruning callbacks were absent;
action-logit diagnostics used an invalid context manager and could silently substitute
zeros. Action logits were not estimated Q-values. Proposed action-diversity rewards
would not establish economic validity and were not carried into the new simulator.

The original README reported static -1520.64 versus PPO -1591.90 mean reward. Other
local JSON runs reported PPO improvements with all +20% actions. These are different
historical outputs, and their provenance is insufficient to reconcile the snapshots.
Both are retained; neither result describes the new longitudinal environment.
