# Sprint 1 repository audit and migration

The audit was performed before source changes, using a recursive inventory, tracked
file list, full module/script inspection and saved data/metrics inspection. Initial
working tree was clean. `pre_migration_inventory.csv` records original paths, byte
sizes and SHA-256 hashes (including ignored local artifacts, excluding Git internals).
There was no AGENTS.md, test suite, notebook, packaging metadata or lint configuration.

## Original contents and disposition

| Classification | Original files | Decision |
|---|---|---|
| Core prototype | `credit_limit_rl/{config,data,env,risk,evaluation,hpo,__init__}.py` | Archive whole package; carry synthetic generator and GB trainer into current package |
| Training/experiments | `train.py`, `retrain_fixed.py`, `retrain_optimized_weights.py`, `analyze_action_collapse.py` | Preserve under `experiments/legacy/`; replace active entry points with small longitudinal experiments |
| Presentation | `app.py` (Streamlit), original README | Preserve in archive; no new frontend |
| Exploratory/duplicated/dead sketch | `QUICK_FIX_HARD_PD_CONSTRAINTS.py`, `ACTION_COLLAPSE_ROOT_CAUSE_AND_SOLUTIONS.md` | Archive; sketch missing imports and repeats implemented threshold logic; proposed fixes are not validated findings |
| Generated tracked results | `results/` CSV, JSON and PNG files | Move intact to `outputs/legacy/results/` |
| Generated ignored local assets | `artifacts/` four PPO ZIPs, GB joblib, sample CSV, metrics, HPO parameters/trials | Move intact to ignored `outputs/legacy/artifacts/` |
| Dependencies | `requirements.txt` | Preserve legacy requirements; new minimal pyproject with core, experiments, RL and dev extras |
| Cache/IDE metadata | root/package `__pycache__`, `.vscode/settings.json` | Delete disposable caches and local IDE preferences; ignore recurrence |
| Notebooks/external datasets | None | No invented notebook/raw-data directories |

No useful research result, original dataset, model or negative finding was deleted.
The active generator/PD trainer initially reused the original source directly. The
legacy copy remains immutable reference material; active interfaces evolve separately.

## Scientific and implementation findings

1. `CreditLimitEnv.reset` sampled an array of unrelated customer indices; `step`
   advanced through it. The computed new limit/spend never became the next customer
   state. Episodes modeled row batches, not monthly customer decisions. README text
   claiming customer transitions was unsupported by the implementation.
2. Default draws used an existing synthetic `true_pd` modified by utilization, not
   predicted PD directly. That separation was useful, but defaulted customers did
   not persist or terminate, and there was no evolving balance/payment state.
3. Reward subtracted a weighted predicted expected loss plus realized LGD loss.
   This double counted loss in expectation; capital and score-tier penalties further
   changed optimization. Score-tier penalties were called hard constraints but were
   ordinary soft reward penalties. Portfolio penalty was a row-batch running average.
4. `kfold_split` computed risk quartiles then ignored them. Risk-regime labels
   interpreted low scores as low risk and high scores as high risk (reversed).
   Time-series splits used random synthetic row order absent actual dates. The risk
   model was fitted before some cross-validation folds, contaminating their independence.
5. HPO selected on the named test set and varied both reward weights and PPO parameters.
   Rewards across different objective definitions are not directly comparable.
   A pruning object existed without trial reports/callbacks that would activate it.
6. The action-collapse script used an invalid context manager/feature input path and
   caught errors by substituting zero logits. Logits were also discussed as action
   value approximations. Its numerical diagnostics should be treated cautiously.
7. Retraining scripts repeated model construction, batch-size alignment and parameter
   loading. The quick-fix script duplicated threshold/reward code but was incomplete.
   Action diversity and favorable PPO outcomes were proposed as targets without
   evidence of economic validity. None was adopted as a requirement.
8. The historical README's negative PPO snapshot conflicted with local JSON runs that
   reported positive relative one-step return and a 100% +20% action distribution.
   All outputs remain; they are not reconciled or relabeled as longitudinal findings.
9. The original generator creates synthetic demographic/financial fields. No customer
   PII or external/raw banking dataset was found. The sample dataset remains local
   and ignored; source/status are documented in `data/README.md`.
10. Dependencies mixed core simulation, visualization, Streamlit, PPO and HPO. The
    new package makes sklearn/plotting and Torch optional. Streamlit/Plotly/Optuna
    are retained only in archived requirements, since no active path uses them.

## Migration choices

Use `src/credit_rl` with editable installation, a separate transition layer, an
observed-feature PD interface, customer snapshots plus hidden traits, componentized
reward, and paired trajectory evaluation. Preserve the action set, limit bounds,
synthetic generator, GB training/diagnostics, financial coefficients, baseline
concepts and a minimal PPO integration. Do not retain flawed evaluation modes as
compatibility wrappers. The old model observation schema is deliberately incompatible.

Two configuration files suffice: full DGP/economics in `simulation.yaml`, run settings
in `experiments.yaml`. No empty modules beyond documented package namespaces, unused
notebooks, infrastructure or complex calibration framework were added.

During the work an intervening local commit, `6856fc7` (`refactoring`), captured the
initial moves and 8,521 files from the newly created `.venv`. The local environment
was retained, but its index entries were removed with `git rm --cached`; `.gitignore`
now excludes it. This removes it from the next commit without rewriting history.
No commit, push or history rewrite was performed by this task.
