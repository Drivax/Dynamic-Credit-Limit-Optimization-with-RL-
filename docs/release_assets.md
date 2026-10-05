# Artifact availability and retention

## Keep in Git

- All package source, tests and scientific configurations: historical entry points remain necessary for reproducibility.
- Final headline and paired tables, robustness, counterfactual/OPE summaries, compact supporting tables, preregistration, classification and selection under `outputs/main/final/`.
- Ten final figures; the README displays only decision process, structural action map, initialization and incremental value.
- `outputs/main/final_manifest.json`, frozen-product hashes and historical protection map. The latter is about 1.9 MB but identifies 12,665 protected files and is necessary for auditing.
- The compact initialization `training_trajectory.csv` (about 109 KB), now explicitly included so its final figure has a public data source.
- Existing historical diagnostic tables and figures referenced by methodological reports. Negative results remain intact.

## Regenerate locally

Raw per-customer episodes (about 34.5 MB for the final suite), visit trajectories, intermediate Monte Carlo banks, training-update dumps, raw action distributions, caches, coverage data, temporary manifests and smoke artifacts do not belong in a new source archive. Ignore rules exclude future generated copies and all new local validation logs.

Two historical dumps are already tracked: `outputs/main/ppo_diagnostics/training_updates.csv` (5.52 MB) and `action_distribution.csv` (4.59 MB). They are reconstructible by the PPO diagnostic analysis, but the current protection map and historical report reference them. They are **retained unchanged**, with ignore rules added for future untracked copies. `.gitignore` does not untrack existing files. Removing them from the index would be a separate coordinated artifact migration, not an accomplished cleanup in this local-only task.

`action_dominance.csv` (2.79 MB), `action_gaps.csv` (1.14 MB) and `gap_summary.csv` (1.14 MB) support structural analysis and reporting and are also retained. Historical and final figures may intentionally show the same evidence; their provenance paths are preserved rather than deduplicated blindly.

## Release asset if needed

The frozen PD model, public teacher, imitation actors, selected PPO checkpoints, evaluation cells and verification metadata are local research artifacts. A complete standard replay depends on them, including their original relative paths. They are not all tracked or included in the prepared release. No downloadable artifact bundle is claimed.

If an artifact bundle is published later, it should be one checksummed archive with an explicit manifest and restoration paths, not hundreds of separate uploads. A compact final-results archive and manifest are optional; the checked-in tables already support reading the study. No PDF was generated solely for release packaging.

## Reproduction levels

1. **Read results:** tables, figures and paper are included in the source tree.
2. **Render documentation:** `python -m credit_rl.experiments.final_report --documents-only` needs the compact measured text and classification, not historical models.
3. **Check integration:** the final smoke builds isolated fixtures when historical models are unavailable; its output is not scientific evidence.
4. **Replay standard:** requires the complete frozen research artifacts and protection map. The recorded replay used Windows; historical manifests contain Windows-style relative paths, and a POSIX standard replay has not been validated. Do not run historical generators into protected output directories merely to fill missing files. The historical protocols describe their original generation; they are not an automatic bootstrap of the published result set.

## Git history

Historical Git cleanup is **recommended as a separately authorized operation before public release**. The current tracked tree is about 47 MB; local packed objects occupy about 85 MiB and loose objects about 29 MiB. The history still contains an old `.venv/` installation: NumPy OpenBLAS (20.59 MB), SciPy OpenBLAS (20.26 MB), PIL AVIF (7.89 MB) and HiGHS (6.71 MB) binaries are among its largest blobs. These dependencies are regenerable and do not belong in source history. The two PPO dumps above are additional current-tree candidates for an artifact migration. Deleting current files would not remove old blobs. No history rewrite, index removal, garbage collection or force-push was performed. Git's own temporary object, if present, is left to Git maintenance.
