# v1.0.0 packaging audit

## Scope

Local preparation only: no commit, tag, push, repository rename, remote metadata change or GitHub release. The working tree intentionally remains uncommitted. The scientific study is complete; no new standard experiment, tuning, model, DGP, reward or policy change was made.

## Presentation

The README leads with four findings, the five central policies, primary-budget paired NEV intervals and the negative regularization result. Its measured block is a generated subset of the full CSV-derived paper block. The paper retains every policy, both budgets and the original conditional classification. The `--documents-only` renderer publishes frozen text without scientific inputs or computations.

Four figures are shown: decision process, structural action map, initialization trajectories and incremental economic value. The diagram's whitespace is reduced; the structural map uses two rows with explicit support counts; policy names are consistent; incremental axes identify synthetic EUR per customer. All plot data remain unchanged. The other figures stay available in the paper.

## Packaging and cleanup

MIT license and original third-party attribution are included in the successfully built wheel. Citation metadata uses the existing Git author name, version 1.0.0 and actual repository URL; no release date, DOI or affiliation is invented. The changelog and release notes identify the release as prepared and unreleased.

Ignore rules exclude raw trajectories, checkpoints, caches, smoke output, coverage data and temporary release work. No protected scientific file was deleted. The two large tracked PPO dumps remain present because the historical report and hash map depend on them; ignore rules cannot remove already-tracked files. The compact initialization trajectory is explicitly publishable as a figure source.

See [retention decisions](release_assets.md) and [file inventory](release_inventory.csv). Historical Git cleanup is recommended separately because an old `.venv/` remains in Git history. No rewrite or Git-object cleanup was attempted.

## Validation

- Ruff: passed. Git whitespace check: passed.
- Full suite: **190 passed**; eight final renderer/integration regression tests also passed.
- Final combined coverage: **77.45%**, above the preceding 76.93% and configured CI floor of 75.13%. Renderer line data was cleared before regression coverage was appended; unchanged modules retain full-suite coverage.
- Clean-source final smoke and smoke report: passed without local historical models; the fixture explicitly records `scientific_evidence: false`.
- Wheel: `credit_rl-1.0.0-py3-none-any.whl` built without dependency downloads; package version and both license notices verified.
- Main local Markdown links exist and are publishable. README and paper measured blocks match the same frozen result text; the README is a primary-budget subset.
- All scientific CSVs, classification, simulator/learner code and configurations remain unchanged. All **12,665** protected historical files pass SHA-256 verification.
- The ten final figures regenerate identically with the polished renderer. The scientific manifest was refreshed; its pre-packaging version is preserved under `outputs/main/final/historical_documents/scientific_manifest.json`.
- The temporary clean-source copy was removed after validation; compact audits remain and raw logs/wheel/coverage are ignored under `outputs/release/`.

The [release manifest](release_manifest.json) records hashes and these checks. The earlier [scientific validation](../outputs/main/final/validation.json) remains separate. Hosted CI and a Linux standard replay have not been executed. Standard reproduction still needs the frozen local research artifacts described in the inventory.

## Publication status

**NOT READY TO TAG**: local changes are uncommitted and hosted CI has not run. No tag or release was created. The [publication checklist](release_checklist.md) lists the remaining GitHub steps, including metadata, optional rename/social preview/pinning and the release gate.
