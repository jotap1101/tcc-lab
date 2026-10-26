"""Testes de src/evaluation.py (avaliação por pixel e artefatos do estágio 11)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import evaluation
from src.models.unet import UNet
from src.trainer import weights_path

TEST_SOURCE = "alphaearth"
PATCH_SIZE = 8
FOLD_COUNT = 3


def _storage_paths(tmp_path: Path) -> dict[str, Path]:
    """Caminhos de armazenamento mínimos para os testes do estágio 11."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "models": tmp_path / "models",
        "artifacts_metrics": tmp_path / "artifacts" / "metrics",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
    }


def _small_model() -> UNet:
    """Modelo U-Net pequeno, adequado aos patches de 8x8 dos testes."""
    return UNet(in_channels=4, channels=[4, 8], out_channels=1)


def _write_dataset(tmp_path: Path) -> tuple[dict[str, Path], pd.DataFrame]:
    """Escreve manifesto, patches e estatísticas sintéticas no tmp_path."""
    paths = _storage_paths(tmp_path)
    processed = paths["data_processed"]
    processed.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(6):
        image_path = f"data/processed/patches/images/h/tile_r{index:04d}.npy"
        mask_path = f"data/processed/patches/masks/h/tile_r{index:04d}.npy"
        image = np.full((4, PATCH_SIZE, PATCH_SIZE), 100.0, dtype=np.float32)
        mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        if index % 2 == 0:
            mask[2:6, 2:6] = 1
        for relative in (image_path, mask_path):
            (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp_path / image_path, image)
        np.save(tmp_path / mask_path, mask)
        records.append(
            {
                "patch_id": f"tile_r{index:04d}_c0000",
                "tile_id": "tile_310044_2023",
                "fold": index % FOLD_COUNT,
                "row": index,
                "col": 0,
                "bbox": f"{index * 8},{0},{index * 8 + 8},{8}",
                "coffee_ratio": 0.25 if index % 2 == 0 else 0.0,
                "valid_fraction": 1.0,
                "mask_source": TEST_SOURCE,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    manifest = pd.DataFrame(records)
    manifest.to_parquet(processed / f"manifest_{TEST_SOURCE}.parquet", index=False)
    stats = {
        "bands": ["B2", "B3", "B4", "B8"],
        "per_band": {
            band: {"mean": 100.0, "std": 10.0, "nan_fraction": 0.0}
            for band in ["B2", "B3", "B4", "B8"]
        },
    }
    (processed / f"normalization_stats_{TEST_SOURCE}.json").write_text(
        json.dumps(stats, ensure_ascii=False), encoding="utf-8"
    )
    return paths, manifest


def _write_weights(paths: dict[str, Path], fold: int) -> Path:
    """Persiste pesos sintéticos da dobra com a arquitetura do modelo de teste."""
    target = weights_path(paths, "unet", TEST_SOURCE, fold)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_small_model().state_dict(), target)
    return target


@pytest.fixture
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ambiente local com dataset, pesos e o modelo pequeno injetado."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    monkeypatch.setattr(evaluation, "build_model", lambda model_name: _small_model())
    monkeypatch.setattr(evaluation, "fold_count", lambda: FOLD_COUNT)
    paths, manifest = _write_dataset(tmp_path)
    for fold in range(FOLD_COUNT):
        _write_weights(paths, fold)
    return paths, manifest


def test_artifact_paths_follow_schema() -> None:
    """Os artefatos de avaliação devem viver sob artifacts/metrics/<modelo>/<fonte>/evaluation."""
    paths = _storage_paths(Path("/tmp"))
    assert str(evaluation.evaluation_metrics_path(paths, "unet", TEST_SOURCE, 2)).endswith(
        "artifacts/metrics/unet/alphaearth/evaluation/fold_2.json"
    )
    assert str(evaluation.predictions_path(paths, "unet", TEST_SOURCE, 2)).endswith(
        "artifacts/metrics/unet/alphaearth/evaluation/predictions_fold_2.npz"
    )
    assert str(evaluation.patch_metrics_path(paths, "unet", TEST_SOURCE, 2)).endswith(
        "artifacts/metrics/unet/alphaearth/evaluation/patch_metrics_fold_2.parquet"
    )
    assert str(evaluation.evaluation_summary_path(paths, "unet", TEST_SOURCE)).endswith(
        "artifacts/metrics/unet/alphaearth/evaluation/summary.json"
    )


def test_evaluate_fold_persists_artifacts(environment) -> None:
    """evaluate_fold deve persistir métricas, predições e métricas por patch."""
    paths, _ = environment
    result = evaluation.evaluate_fold(
        "unet", paths, 0, torch.device("cpu"), source_name=TEST_SOURCE
    )

    assert not result.skipped
    assert result.n_patches == 2
    assert set(result.metrics) == set(evaluation.METRIC_KEYS)
    assert set(result.counts) == set(evaluation.COUNT_KEYS)
    assert result.metrics_path.is_file()
    assert result.predictions_path.is_file()
    assert result.patch_metrics_path.is_file()

    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["model"] == "unet"
    assert payload["fold"] == 0
    assert payload["n_patches"] == 2

    patch_metrics = pd.read_parquet(result.patch_metrics_path)
    assert len(patch_metrics) == 2
    assert {"patch_id", "fold", "tp", "fp", "fn", "iou", "coffee_ratio"} <= set(
        patch_metrics.columns
    )

    with np.load(result.predictions_path) as data:
        assert data["predictions"].shape == (2, PATCH_SIZE, PATCH_SIZE)
        assert len(data["patch_ids"]) == 2


def test_evaluate_fold_skips_when_current(environment) -> None:
    """Reexecuções devem reutilizar os artefatos sem reavaliar."""
    paths, _ = environment
    first = evaluation.evaluate_fold("unet", paths, 0, torch.device("cpu"), source_name=TEST_SOURCE)
    mtime = first.metrics_path.stat().st_mtime_ns

    second = evaluation.evaluate_fold(
        "unet", paths, 0, torch.device("cpu"), source_name=TEST_SOURCE
    )

    assert second.skipped
    assert second.metrics_path.stat().st_mtime_ns == mtime


def test_evaluate_fold_retrains_when_weights_change(environment) -> None:
    """Novos pesos (retreino) devem invalidar a avaliação anterior."""
    paths, _ = environment
    evaluation.evaluate_fold("unet", paths, 0, torch.device("cpu"), source_name=TEST_SOURCE)
    weights = weights_path(paths, "unet", TEST_SOURCE, 0)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    key = next(iter(state))
    state[key] = state[key] + 1.0
    torch.save(state, weights)

    result = evaluation.evaluate_fold(
        "unet", paths, 0, torch.device("cpu"), source_name=TEST_SOURCE
    )

    assert not result.skipped


def test_evaluate_fold_ignores_invalid_pixels(environment) -> None:
    """Pixels inválidos (-1) devem ser ignorados sem quebrar a avaliação."""
    paths, manifest = environment
    image_file = Path(paths["data_processed"]).parent.parent / str(manifest.loc[0, "image_path"])
    image = np.load(image_file)
    image[:] = np.nan
    np.save(image_file, image)

    result = evaluation.evaluate_fold(
        "unet", paths, 0, torch.device("cpu"), source_name=TEST_SOURCE
    )

    patch_metrics = pd.read_parquet(result.patch_metrics_path)
    invalid_patch = patch_metrics[patch_metrics["patch_id"] == manifest.loc[0, "patch_id"]]
    assert int(invalid_patch["tp"].iloc[0]) == 0
    assert int(invalid_patch["fp"].iloc[0]) == 0
    assert int(invalid_patch["fn"].iloc[0]) == 0


def test_summarize_model_aggregates_folds(environment) -> None:
    """O resumo deve somar as contagens e reportar média/desvio por dobra."""
    paths, _ = environment
    evaluations = evaluation.evaluate_model(
        paths, "unet", torch.device("cpu"), source_name=TEST_SOURCE
    )

    summary = evaluation.summarize_model(paths, "unet", TEST_SOURCE, evaluations)

    assert summary["n_folds"] == FOLD_COUNT
    assert summary["n_patches"] == 6
    assert sum(item["n_patches"] for item in summary["per_fold"]) == 6
    assert set(summary["aggregate"]["metrics"]) == set(evaluation.METRIC_KEYS)
    assert set(summary["fold_mean"]) == set(evaluation.METRIC_KEYS)
    assert set(summary["fold_std"]) == set(evaluation.METRIC_KEYS)
    assert evaluation.evaluation_summary_path(paths, "unet", TEST_SOURCE).is_file()


def test_save_evaluation_figure(environment) -> None:
    """A figura de avaliação deve ser persistida como PNG."""
    paths, _ = environment
    evaluations = [
        evaluation.evaluate_fold("unet", paths, fold, torch.device("cpu"), source_name=TEST_SOURCE)
        for fold in range(FOLD_COUNT)
    ]
    summary = evaluation.summarize_model(paths, "unet", TEST_SOURCE, evaluations)

    figure = evaluation.save_evaluation_figure(paths, "unet", TEST_SOURCE, summary)

    assert figure.is_file()
    assert figure.read_bytes().startswith(b"\x89PNG")


def test_load_predictions_maps_patch_ids(environment) -> None:
    """As predições carregadas devem ser indexadas pelo patch_id do manifesto."""
    paths, _ = environment
    evaluation.evaluate_fold("unet", paths, 0, torch.device("cpu"), source_name=TEST_SOURCE)

    predictions = evaluation.load_predictions(paths, "unet", TEST_SOURCE, 0)

    assert set(predictions) == {"tile_r0000_c0000", "tile_r0003_c0000"}
    assert next(iter(predictions.values())).shape == (PATCH_SIZE, PATCH_SIZE)
