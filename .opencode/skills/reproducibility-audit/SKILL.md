---
name: reproducibility-audit
description: Use when checking a run, setup, or artifact set of this research for computational reproducibility. Covers fixed seeds (python/numpy/torch/cuda), src/config.yaml as the single source of truth, manifest_<fonte>.parquet data versioning, committed uv.lock, and per-run environment logging without exposing secrets.
---

# Reproducibility audit

## What this skill covers

A checklist to verify that a run, configuration, or artifact set can be reproduced independently and traced to its exact environment.

## When to use it

Use before publishing artifacts, after changing training/inference code, or when reviewing a run for the thesis.

## Checklist

### Seeding
- [ ] `python`, `numpy`, `torch`, and `cuda` seeds fixed (`random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`).
- [ ] Seeds propagated into DataLoaders / samplers and any `generator` arguments.
- [ ] Nondeterminism sources identified and controlled where documented.
- [ ] Deterministic flags set: `torch.backends.cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)` where supported; residual nondeterminism documented.

### Configuration
- [ ] All parameters resolved from `src/config.yaml` via `src/config.py`; nothing hardcoded.
- [ ] Config schema matches what the code consumes (no stale keys, no silent defaults).

### Dependencies
- [ ] `uv.lock` committed and in sync with `pyproject.toml` (`uv lock --check`).
- [ ] Python and key library versions recorded.
- [ ] Runtime library versions pinned at notebook execution (each notebook's dependencies cell calls `src.setup.install_runtime_requirements()`, which restarts the kernel on ABI mismatch; Colab/Kaggle default envs differ) and logged per run.

### Data and artifacts
- [ ] `manifest_<fonte>.parquet` registers every dataset version used, including each patch's `valid_fraction`; patches below `data.min_valid_fraction` are discarded and remaining invalid pixels use the ignore label `-1`.
- [ ] All artifacts stored under the `tcc/` folder at the root of Google Drive (`MyDrive/tcc/`), organized in subfolders; nothing persisted outside it.
- [ ] Notebooks create `tcc/` (and any needed subfolder) when absent and reuse it when present; no pre-created layout assumed.
- [ ] Storage operations idempotent: derived artifacts are reused only when the input fingerprint/config hash matches and regenerated in place otherwise; historical baselines are archived with a `_v1` suffix (e.g., `emater_v1`, `archive_v1`).
- [ ] Notebooks committed with cleared outputs (no saved cell outputs/platform paths).
- [ ] Inputs/outputs of each stage resolvable to unique identifiers.

### Storage and platform
- [ ] Notebooks are platform-agnostic: run correctly on Google Colab and Kaggle (same protocol), no hardcoded platform paths or APIs in cells.
- [ ] Platform detection, Drive mounting, and path resolution centralized in `src/` (e.g., `io.py`, `config.py`).
- [ ] Storage paths resolved from `src/config.yaml` via `src/config.py`; nothing hardcoded in notebooks.

### Environment logging
- [ ] Every run logs versions, seeds, config hash, commit SHA (git or `src/.tcc_commit`), and dataset manifest fingerprint.
- [ ] Per-fold metrics store `manifest_fingerprint_hash` and `config_hash`, so stale artifacts trigger automatic retraining.
- [ ] Secrets (GEE, Hugging Face, Kaggle) never appear in logs, config, or code — environment variables only.
- [ ] Authentication is executed inside notebook code cells (e.g., GEE via OAuth with the primary account using `ee.Authenticate()`), reading credentials from environment variables — no external or manual authentication step.
