---
name: notebook-stage
description: Use when creating, editing, or reviewing a Jupyter notebook stage for this coffee-segmentation research. Encodes PLAN.md stage ordering, AGENTS.md cell conventions (pt-br impersonal markdown immediately above every code cell, en-US identifiers, one responsibility per cell, pt-br technical comments) and resolution of all paths from src/config.py. Also encodes platform-agnostic execution (same notebook runs correctly on Colab and Kaggle, same protocol) and canonical storage under the `tcc/` folder at the root of Google Drive (`MyDrive/tcc/`).
---

# Notebook stage

## What this skill covers

Conventions for building an isolated Jupyter notebook (`.ipynb`) for one research stage in this repository.

## When to use it

Use when a stage notebook must be created, updated, or reviewed. The authoritative stage list, inputs, and outputs live in `PLAN.md`.

## Rules

1. **Naming and order**: `NN_verb_snake_case.ipynb` inside `notebooks/`, executed in numeric order per `PLAN.md`.
2. **Isolation and single responsibility**: a notebook contains only the cells required for its stage; each cell (markdown or code) has exactly one functional responsibility.
3. **Markdown above every code cell**: mandatory, written exclusively in pt-br, neutral and impersonal (no "eu", "nós", "meu").
4. **Code cells**: identifiers in en-US; short pt-br technical comments above or to the right of critical instructions.
5. **Paths**: resolve every input/output from `src/config.py` — never hardcode paths. `src/config.yaml` is the single source of truth for configuration.
6. **Platform-agnostic**: the notebook must run **correctly** on **Google Colab and Kaggle**, executing the same protocol (same code, seeds, config, processing). No platform-specific APIs, magic commands, or hardcoded platform paths in cells; environment detection, Drive mounting and path resolution live in `src/` only.
7. **Canonical storage**: every datum, file, model, metric, figure, or log is stored under the **`tcc/` folder at the root of Google Drive** (`MyDrive/tcc/`). The notebook accesses it when it exists or **creates it — and any required subfolder — when it does not**; it never assumes a pre-created layout and never writes outside `tcc/`.
8. **Reuse (DRY)**: import reusable logic from `src/`; the notebook is only the orchestration/visualization layer and never re-implements or duplicates existing code. Apply Single Responsibility, KISS and YAGNI to every cell — one concern per cell, no speculative or dead code.
9. **Reproducibility**: respect fixed seeds (python/numpy/torch/cuda), deterministic torch flags (`cudnn.deterministic`, `benchmark=False`, `use_deterministic_algorithms` where supported), pinned runtime versions installed by the dependencies cell via `src.setup.install_runtime_requirements()` (auto kernel restart on ABI mismatch), and per-run environment logging.
10. **Authentication in code cells**: authentication steps (e.g., GEE via OAuth with the primary account using `ee.Authenticate()`) are executed inside the notebook's own code cells with the required libraries; credentials come from environment variables only and are never committed. No external/manual authentication step.
11. **Notebook JSON**: valid nbformat 4.x — `cells` array with `cell_type` (`markdown` | `code`) and `metadata` with `kernelspec` and `language_info`. Code cells must have empty `outputs` (cleared before commit).

## Checklist

- [ ] Filename matches `NN_verb_snake_case.ipynb`.
- [ ] Stage number and purpose match `PLAN.md`.
- [ ] Every code cell is preceded by a pt-br impersonal markdown cell.
- [ ] Identifiers are en-US; comments are pt-br.
- [ ] No hardcoded paths; everything resolved via `src/config.py`.
- [ ] Notebook is platform-agnostic: runs correctly on Colab and Kaggle (same protocol), no platform-specific paths/APIs.
- [ ] `src/` delivered by the first-cell bootstrap (`src/bootstrap.py`) and, when present, by the `MyDrive/tcc/repo/` mirror; Drive accessed via `mount_drive()`/`ensure_storage_root()` only.
- [ ] Ground-truth sources (MapBiomas, AlphaEarth, ...) each have their own code cell and Drive subfolder (`data/.../<source>/`).
- [ ] Dataset stages respect `data.min_valid_fraction`; invalid pixels use the ignore label `-1` and are excluded from loss/metrics.
- [ ] All outputs stored under the Drive `tcc/` root; the notebook creates `tcc/` and subfolders when missing.
- [ ] Storage operations idempotent (derived artifacts reused only with matching fingerprint/config hash; historical baselines archived as `*_v1`).
- [ ] No re-implementation of logic that belongs in `src/`.
- [ ] Cells follow DRY/Single Responsibility/KISS/YAGNI; no duplicated or dead code.
- [ ] Notebook committed with cleared outputs.
