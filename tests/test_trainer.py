"""Testes de src/trainer.py (protocolo de treino e artefatos por dobra)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import get_config
from src.data.dataset import build_loaders, fold_indices
from src.models.unet import UNet
from src.trainer import (
    FoldResult,
    fold_training_done,
    history_path,
    metrics_path,
    run_metadata,
    save_run_metadata,
    train_fold,
    weights_path,
)

TEST_SOURCE = "alphaearth"


def _storage_paths(tmp_path) -> dict:
    """Caminhos de armazenamento mínimos para os testes do estágio 09."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "data_processed_ground_truth": tmp_path / "data" / "processed" / "ground_truth",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "models": tmp_path / "models",
        "artifacts_metrics": tmp_path / "artifacts" / "metrics",
        "artifacts_runs": tmp_path / "artifacts" / "runs",
    }


def _training_config() -> dict:
    """Configuração de treino reduzida para os testes (1 época, sem aumentação)."""
    config = copy.deepcopy(get_config())
    config["loss"] = {"dice_weight": 1.0, "focal_weight": 1.0, "boundary_weight": 1.0}
    config["augmentation"] = {"enabled": False}
    config["training"] = {
        "epochs": 1,
        "batch_size": 2,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "num_workers": 0,
    }
    config["reproducibility"] = {"seed": 42}
    config["splits"] = {"fold_count": 3}
    return config


def _write_dataset(tmp_path) -> dict:
    """Escreve manifesto, patches e estatísticas sintéticas no tmp_path."""
    paths = _storage_paths(tmp_path)
    stats = {
        "bands": ["B2", "B3", "B4", "B8"],
        "per_band": {
            band: {"mean": 100.0, "std": 10.0, "nan_fraction": 0.0}
            for band in ["B2", "B3", "B4", "B8"]
        },
    }
    processed = paths["data_processed"]
    processed.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(6):
        image_path = f"data/processed/patches/images/h/tile_r{index:04d}.npy"
        mask_path = f"data/processed/patches/masks/h/tile_r{index:04d}.npy"
        image = np.full((4, 8, 8), 100.0, dtype=np.float32)
        mask = np.zeros((8, 8), dtype=np.uint8)
        if index % 2 == 0:
            mask[2:6, 2:6] = 1
        for relative in (image_path, mask_path):
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp_path / image_path, image)
        np.save(tmp_path / mask_path, mask)
        records.append(
            {
                "patch_id": f"tile_r{index:04d}_c0000",
                "tile_id": "tile_310044_2023",
                "fold": index % 3,
                "row": index,
                "col": 0,
                "bbox": f"{index * 8},{0},{index * 8 + 8},{8}",
                "coffee_ratio": 0.25,
                "mask_source": "alphaearth",
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    pd.DataFrame(records).to_parquet(processed / f"manifest_{TEST_SOURCE}.parquet", index=False)
    (processed / f"normalization_stats_{TEST_SOURCE}.json").write_text(
        json.dumps(stats, ensure_ascii=False), encoding="utf-8"
    )
    return paths


def _small_model() -> UNet:
    """Modelo U-Net pequeno, adequado aos patches de 8x8 dos testes."""
    return UNet(in_channels=4, channels=[4, 8], out_channels=1)


def test_artifact_paths_follow_schema() -> None:
    """Os caminhos de artefatos devem seguir o schema do PLAN.md."""
    paths = _storage_paths(Path("/tmp"))
    assert str(weights_path(paths, "unet", TEST_SOURCE, 2)).endswith(
        "models/unet/alphaearth/fold_2.pt"
    )
    assert str(metrics_path(paths, "unet", TEST_SOURCE, 2)).endswith(
        "artifacts/metrics/unet/alphaearth/fold_2.json"
    )
    assert str(history_path(paths, "unet", TEST_SOURCE, 2)).endswith(
        "artifacts/runs/unet/alphaearth/fold_2.json"
    )


def test_fold_training_done_false_when_missing(tmp_path) -> None:
    """Sem artefatos, a dobra deve ser considerada não treinada."""
    paths = _storage_paths(tmp_path)
    assert not fold_training_done(paths, "unet", TEST_SOURCE, 0)


def test_train_fold_persists_artifacts(tmp_path, monkeypatch) -> None:
    """train_fold deve treinar, persistir artefatos e reportar o resumo."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    monkeypatch.setattr("src.trainer.get_config", lambda: _training_config())
    paths = _write_dataset(tmp_path)

    result = train_fold(
        _small_model(), "unet", paths, fold=0, device=torch.device("cpu"), source_name=TEST_SOURCE
    )

    assert isinstance(result, FoldResult)
    assert result.fold == 0
    assert result.epochs == 1
    assert not result.skipped
    assert weights_path(paths, "unet", TEST_SOURCE, 0).is_file()
    assert metrics_path(paths, "unet", TEST_SOURCE, 0).is_file()
    assert history_path(paths, "unet", TEST_SOURCE, 0).is_file()
    assert fold_training_done(paths, "unet", TEST_SOURCE, 0)

    payload = json.loads(metrics_path(paths, "unet", TEST_SOURCE, 0).read_text(encoding="utf-8"))
    assert payload["model"] == "unet"
    assert payload["fold"] == 0
    assert set(payload["val"]) == {
        "loss",
        "dice",
        "focal",
        "boundary",
        "iou",
        "f1",
        "precision",
        "recall",
    }


def test_train_fold_handles_invalid_pixels(tmp_path, monkeypatch) -> None:
    """Pixels inválidos (-1) devem ser ignorados sem quebrar o treino/validação."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    monkeypatch.setattr("src.trainer.get_config", lambda: _training_config())
    paths = _write_dataset(tmp_path)
    manifest = pd.read_parquet(paths["data_processed"] / f"manifest_{TEST_SOURCE}.parquet")
    image_file = tmp_path / str(manifest.loc[0, "image_path"])
    image = np.load(image_file)
    image[:, :4, :] = np.nan
    np.save(image_file, image)

    result = train_fold(
        _small_model(), "unet", paths, fold=0, device=torch.device("cpu"), source_name=TEST_SOURCE
    )

    assert not result.skipped
    assert result.val_metrics["iou"] >= 0.0


def test_train_fold_skips_when_done(tmp_path, monkeypatch) -> None:
    """Reexecuções devem reutilizar os artefatos sem retreinar."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    monkeypatch.setattr("src.trainer.get_config", lambda: _training_config())
    paths = _write_dataset(tmp_path)
    train_fold(
        _small_model(), "unet", paths, fold=0, device=torch.device("cpu"), source_name=TEST_SOURCE
    )

    mtime = weights_path(paths, "unet", TEST_SOURCE, 0).stat().st_mtime_ns
    result = train_fold(
        _small_model(), "unet", paths, fold=0, device=torch.device("cpu"), source_name=TEST_SOURCE
    )

    assert result.skipped
    assert weights_path(paths, "unet", TEST_SOURCE, 0).stat().st_mtime_ns == mtime


def test_train_fold_retrains_when_dataset_fingerprint_changes(tmp_path, monkeypatch) -> None:
    """Artefatos de uma versão anterior do dataset não devem ser reutilizados."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    monkeypatch.setattr("src.trainer.get_config", lambda: _training_config())
    paths = _write_dataset(tmp_path)
    train_fold(
        _small_model(), "unet", paths, fold=0, device=torch.device("cpu"), source_name=TEST_SOURCE
    )

    metrics_file = metrics_path(paths, "unet", TEST_SOURCE, 0)
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    payload["manifest_fingerprint_hash"] = "fingerprint-antigo"
    metrics_file.write_text(json.dumps(payload), encoding="utf-8")

    result = train_fold(
        _small_model(), "unet", paths, fold=0, device=torch.device("cpu"), source_name=TEST_SOURCE
    )
    assert not result.skipped
    refreshed = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert refreshed["manifest_fingerprint_hash"] != "fingerprint-antigo"


def test_train_fold_retrains_when_split_changes(tmp_path, monkeypatch) -> None:
    """Mudança apenas na atribuição das dobras deve forçar o retreino."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    monkeypatch.setattr("src.trainer.get_config", lambda: _training_config())
    paths = _write_dataset(tmp_path)
    train_fold(
        _small_model(), "unet", paths, fold=0, device=torch.device("cpu"), source_name=TEST_SOURCE
    )

    metrics_file = metrics_path(paths, "unet", TEST_SOURCE, 0)
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    payload["split_fingerprint_hash"] = "split-antigo"
    metrics_file.write_text(json.dumps(payload), encoding="utf-8")

    result = train_fold(
        _small_model(), "unet", paths, fold=0, device=torch.device("cpu"), source_name=TEST_SOURCE
    )
    assert not result.skipped
    refreshed = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert refreshed["split_fingerprint_hash"] != "split-antigo"


def test_commit_sha_falls_back_to_bootstrap_file(tmp_path, monkeypatch) -> None:
    """Sem git, o SHA deve vir do arquivo gravado pelo bootstrap."""
    from src import trainer

    class _FailedProcess:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(trainer.subprocess, "run", lambda *args, **kwargs: _FailedProcess())
    commit_file = tmp_path / ".tcc_commit"
    commit_file.write_text("abc123def456\n", encoding="utf-8")
    assert trainer._commit_sha(commit_file) == "abc123def456"
    assert trainer._commit_sha(tmp_path / "missing") == "unknown"


def test_train_fold_force_retrains(tmp_path, monkeypatch) -> None:
    """force=True deve retreinar mesmo com artefatos existentes."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    monkeypatch.setattr("src.trainer.get_config", lambda: _training_config())
    paths = _write_dataset(tmp_path)
    train_fold(
        _small_model(), "unet", paths, fold=0, device=torch.device("cpu"), source_name=TEST_SOURCE
    )

    result = train_fold(
        _small_model(),
        "unet",
        paths,
        fold=0,
        device=torch.device("cpu"),
        force=True,
        source_name=TEST_SOURCE,
    )
    assert not result.skipped


def test_run_metadata_and_save(tmp_path, monkeypatch) -> None:
    """O metadata de execução deve conter a configuração do protocolo."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    monkeypatch.setattr("src.trainer.get_config", lambda: _training_config())
    paths = _write_dataset(tmp_path)

    meta = run_metadata("unet", paths, TEST_SOURCE)
    assert meta["model"] == "unet"
    assert meta["source_name"] == TEST_SOURCE
    assert meta["fold_count"] == int(get_config()["splits"]["fold_count"])
    assert meta["seed"] == 42
    assert meta["epochs"] == 1
    assert meta["platform"] == "local"
    assert len(meta["config_hash"]) == 12
    assert "commit_sha" in meta
    assert "manifest_fingerprint_hash" in meta
    assert "split_fingerprint_hash" in meta
    assert meta["model_config"]["segformer_variant"]
    assert isinstance(meta["deterministic_algorithms"]["enabled"], bool)
    assert isinstance(meta["deterministic_algorithms"]["warn_only"], bool)
    assert "environment" in meta

    saved = save_run_metadata(paths, "unet", TEST_SOURCE)
    assert saved.is_file()
    assert json.loads(saved.read_text(encoding="utf-8"))["model"] == "unet"


def test_build_loaders_works_with_training_config(tmp_path) -> None:
    """build_loaders deve funcionar com a configuração de treino dos testes."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _write_dataset(tmp_path)
    train_loader, val_loader = build_loaders(
        paths, TEST_SOURCE, fold=0, batch_size=2, seed=42, num_workers=0
    )
    manifest = pd.read_parquet(paths["data_processed"] / f"manifest_{TEST_SOURCE}.parquet")
    train_idx, val_idx = fold_indices(manifest, fold=0)
    assert len(train_idx) == 4
    assert len(val_idx) == 2
    assert len(list(train_loader)) == 2
    assert len(list(val_loader)) == 1
    monkeypatch.undo()
