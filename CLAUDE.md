# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

`AGENTS.md` (imported above) is the authoritative operating manual: methodology, notebook writing rules (pt-br impersonal markdown above every code cell, en-US identifiers, pt-br comments), storage, secrets and reproducibility rules all live there. `PLAN.md` defines each stage's inputs/outputs. This file only adds the practical commands and the cross-file architecture.

## Commands

All quality gates target `src/` and `tests/` only (never notebooks). Use the repo `.venv` via `uv`:

```bash
uv sync --locked                      # install project + dev deps from uv.lock
uv run ruff check src tests           # lint
uv run ruff format --check src tests  # format check (CI enforces; `ruff format src tests` to fix)
uv run mypy src tests                 # type check
uv run pytest tests                   # full test suite
uv run pytest tests/test_trainer.py::test_name -q   # single test
python .opencode/tools/scripts/check_notebook.py    # notebook conventions (naming, markdown-above-code, non-empty cells, no outputs)
```

CI (`.github/workflows/ci.yml`) runs exactly these on every push/PR. Notebooks must be committed with outputs stripped or the `notebook-conventions` job fails.

Kaggle publishing (local only, not run by notebooks): `python scripts/push_kaggle_notebooks.py [NN ...] [--run|--dry-run]`.

## Architecture

### How code reaches the runtime

Notebooks are written here but executed on Colab/Kaggle. The first cell of every notebook downloads `src/bootstrap.py` (stdlib-only) from the **public GitHub repo's `main` branch**, which extracts `src/`, `data/external/` and `requirements-runtime.txt` into the workspace and records the commit SHA in `src/.tcc_commit`. Consequence: **changes to `src/` only take effect in notebooks after they are pushed to `main`**. `bootstrap.py` must stay stdlib-only because it runs before any dependency is installed.

Standard notebook prologue after bootstrap:
1. `src.setup.install_runtime_requirements()` — installs `requirements-runtime.txt` pins; auto-restarts the kernel if an already-loaded compiled module (numpy/torch/…) diverges (ABI). Transformers notebooks also call `remove_incompatible_torchaudio()`.
2. `src.setup.initialize()` → `RuntimeContext` (platform, workspace, `storage_paths`, `config`, `seed`).

When adding a pinned runtime dependency, update both `pyproject.toml`/`uv.lock` (local/CI) and `requirements-runtime.txt` (Colab/Kaggle).

### Platform & storage abstraction

- `src/io.py` is the only place platform specifics live: `detect_platform()`, `mount_drive()`, `ensure_storage_root()`, `resolve_storage_paths()`. On Colab Drive is mounted natively; on Kaggle a `DriveClient` talks to the Drive API (token from `GDRIVE_TOKEN`/`GDRIVE_TOKEN_FILE`) with a local cache under `/kaggle/working/drive`. Writes go through `persist_bytes`/`persist_json`/`persist_file` and reads through `ensure_local_copy`/`path_exists` so the same code works on both — use these helpers rather than raw `open()` on storage paths.
- `src/config.yaml` (loaded by `src/config.py::get_config`) is the single source of truth; `src/utils.py::get_secret` resolves secrets (env vars, Kaggle secrets dataset).

### Pipeline data flow (per ground-truth source)

Ground-truth sources (`ground_truth.sources.*` in config: mapbiomas, alphaearth, emater) each have their own year, Sentinel-2 mosaic/composite, masks, patches and `manifest_<source>.parquet`. Stages 01–08 produce artifacts for **every** enabled source; `ground_truth.chosen_source` (manual decision recorded in stage 04) only selects which set stages 09+ consume. Source-specific logic is isolated (e.g. `src/data/emater_masks.py`, `gee_client.py`) — adding a source must not touch another source's code path.

### Training protocol & cache invalidation

- `src/trainer.py` holds the single protocol shared by U-Net (`src/models/unet.py`) and SegFormer (`src/models/segformer.py`): same loss (`src/losses.py`: Dice+Focal+Boundary, ignore label `-1`), optimizer, scheduler, seeds and folds. Only the model differs — keep it that way when changing training.
- Each fold's metrics store `manifest_fingerprint_hash`, `split_fingerprint_hash` and `config_hash` (`trainer.config_hash()`); `fold_training_done()` compares them so a fold is retrained automatically when the dataset, split or config changes. Downstream stages (evaluation, XAI, synthesis `15`, monograph `17`) follow the same fingerprint-reuse pattern — when changing an artifact's inputs, make sure its fingerprint covers them.
- `run_metadata`/`save_run_metadata` write `run.meta.json` (versions, seeds, config hash, commit SHA, determinism settings).

### Figures and inference

All figure renderers use `src/figstyle.py` (final print size in cm, PNG 300 dpi + PDF) and return `dict[str, bytes]`; `figstyle.persist_figure` writes every format next to the canonical PNG. Stage 11 persists per-patch probability histograms (`src/thresholds.py`) so stage 12 can run threshold-independent analysis (AP, leave-one-fold-out calibrated threshold) without re-inference. Patch-level tests in `comparative.py` are descriptive; primary inference is block/fold-level (`src/spatial_inference.py`).

### Downstream modules

`src/evaluation.py` (11, also per-patch probability histograms) → `src/comparative.py` (12; per-patch tests descriptive, primary inference in `src/spatial_inference.py`, threshold analysis in `src/thresholds.py`) → `src/xai/{gradcam,attention_rollout,common}.py` (13–14) → `src/synthesis.py` (15) → `src/release.py` (16, HF Hub) → `src/monograph.py` (17, read-only rendering of earlier artifacts; editorial parameters live in `src/figstyle.py`, not in `config.yaml`). Each notebook stage maps to one of these modules plus `src/data/*` for stages 01–08; `tests/test_<module>.py` mirrors each module.

## Repo notes

- Docs: `README.md` (overview, status, reference results), `PLAN.md` (per-stage roadmap), `AGENTS.md` (rules), `docs/REPRODUCIBILITY.md` (step-by-step, invalidation matrix, troubleshooting), `docs/DECISIONS.md` (decisions and limitations). Keep them aligned when behavior, schema versions or artifacts change.

- `tcc-docs/` is a read-only git submodule.
- `.opencode/` holds the OpenCode agents/commands and the notebook checker used by CI (`.opencode/tools/scripts/check_notebook.py`).
- Archived baselines (`*_v1` folders on Drive, e.g. `models/unet/emater_v1`) are never overwritten.
