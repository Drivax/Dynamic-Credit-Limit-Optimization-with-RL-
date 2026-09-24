# Technical and scientific finalization audit

## Scope and inventory

The initial Git working tree was clean. The inventory contained 292 tracked files: 53 package source files, 11 test files, 32 experiment/archive files, nine configs, 22 documentation files, 161 output artifacts, and four root/data documents. No tracked notebook, bytecode cache or AGENTS.md was present. Python files were parsed recursively; imports, dependency declarations, random streams, point-in-time features, targets, evaluation denominators, bootstrap units and experiment entry points were inspected. `repository_inventory.csv` classifies the final publishable files by role and size.

| Category | Treatment |
|---|---|
| Core source | Retain simulator, risk, policies, portfolio and evaluation mechanisms |
| Tests | Retain scientific tests; add complete fresh-run reproducibility, manual accounting and finite-horizon regression |
| Configuration | Retain scientific parameters; add declared smoke/standard orchestration profiles |
| Experiments | Move PD orchestration and shared provenance into the installed package, retain compatible wrappers |
| Documentation | Rewrite README around the canonical experiment; create self-contained technical paper |
| Input data | Generated synthetic snapshots; no external dataset required |
| Canonical results/figures | Regenerate in outputs/main/standard; retain small tables, figures and manifests |
| Models/raw histories | Reproducibly generated, ignored by Git |
| Supplementary results | Preserve separate portfolio, robustness and OPE protocols and their recorded provenance |
| Archive | Retain isolated experiments/legacy and outputs/legacy with explicit historical disclaimer |
| Temporary files | Ignore tool caches, coverage output, validation environments and clean-source snapshot |

The largest tracked initial file was an archived policy decision CSV (3.48 MB). It is not a canonical input. Archived source contains known invalid row-based methodology and incomplete sketches, explicitly documented in `experiments/legacy/ARCHIVE.md`; it remains isolated from imports, lint and reproduction. This material was not silently reclassified as scientific evidence. Large ignored local studies and checkpoints were not deleted merely for being generated, because their existing audit reports reference them.

## Findings and changes

**Scientific correction:** the customer PPO training adapter returned horizon truncation directly to SB3. SB3 treats time-limit truncations as interrupted continuing tasks and bootstraps value. The stated customer objective ends at the economic horizon. Training now returns terminal completion at either default or that horizon, with a regression test that rules out an accidental default. Public Gymnasium API behavior remains unchanged. Portfolio training already had a finite terminal horizon. Supplementary customer checkpoint identity now includes the trainer source hash so old selected agents cannot be reused silently after this correction.

No DGP coefficient, default mechanism, model class, policy family, action space or test population was tuned to improve outcomes. Existing information-boundary tests passed. No additional latent/future leakage was found in the active PD or policy path. Historical archived bugs remain explicitly outside that path.

**Reproduction:** the initial repository had multiple staged studies but no self-contained canonical customer command. The packaged pipeline generates PD data, fits existing estimators, calibrates separately, trains every configured PPO seed, evaluates the same holdouts, runs crossed paired bootstrap, writes measured tables and generates figures. It rejects changed experiment settings or scientific sources in an existing run directory and verifies frozen model hashes. Rendering changes may reuse fitted models. JSON manifests include expanded configs, versions, UTC timestamps, source hashes and optional Git commit. Model paths are relative to the run directory.

**Dependencies and import hygiene:** pyproject.toml remains the dependency authority. threadpoolctl is now explicitly declared because the experiments import it directly. Ruff and pytest-cov join the development extra. Twenty-two unused imports and one unused assignment were removed; portfolio exports were made explicit. No active heavy dependency was removed because each declared numerical/ML/RL library is used. No sys.path injection or machine-specific source path was found in the active package or current public documentation.

**Tests/CI:** the initial 138 tests passed, with 64.90% package coverage. Added tests run two independently fitted end-to-end smoke experiments, compare deterministic CSV outputs, replay frozen models, validate report synchronization and failure on nonfinite metrics, check manually computable accounting, and prevent horizon bootstrapping. The GitHub Actions workflow installs extras, lints, runs coverage with a 70% total threshold and executes a separate smoke command on Ubuntu/Windows Python 3.12. No source module is excluded from coverage to obtain that threshold. YAML syntax, commands and trigger/matrix fields were validated locally; hosted Actions has not been dispatched by this local task.

**Reporting:** README and paper share a generated result block. DGP equations, reward components, target maturity, policy information, bootstrap units and hyperparameters were checked against source/configuration. Main figures were regenerated and visually inspected. Table values and file links are checked against canonical outputs. No literature citation, affiliation, venue or DOI was invented.

## Preserved scientific boundaries

PD fitting uses five distinct temporal/customer cohorts, with separate calibration. Policy training/validation/test use separate identities and random namespaces from PD cohorts. The fixed calibrated logistic estimator is not selected on test performance. All three PPO seeds are included; checkpoints use validation only. Common random numbers preserve initial states, traits and exogenous shocks across policies. Bootstrap resamples whole customers and training seeds, not monthly rows. Reported net value excludes the two soft penalties included in PPO reward; default incidence and monetary losses are separate outcomes.

The canonical result is negative for a strong planning interpretation: all PPO seeds match constant maximum-contraction effective trajectories to numerical precision, despite improving net value over MyopicEconomic. Default incidence rises. This finding is retained in README, paper and constant_equivalence.json.

## Validation environment and remaining limitations

Python 3.12.14 on Windows was used for measured results. A separate virtual environment was installed from scratch, then pointed at a clean source snapshot containing only publishable files, without ignored models or generated datasets. The full test suite and a fresh smoke experiment ran there. Numerical reproducibility is verified on the same software/platform, not asserted bit-for-bit across arbitrary library versions or CPUs.

Supplementary reporting modules remain less covered than the scientific core. Archived prototype code remains intentionally non-executable historical context. The complete supplementary portfolio/robustness/OPE studies were preserved but not rerun during this canonical-customer finalization. Their existing reports describe frozen source/configuration identities and must not be confused with newly generated results. No full profile or real-bank validation is claimed.
