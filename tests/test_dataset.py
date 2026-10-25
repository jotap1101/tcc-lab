"""Testes de src/data/dataset.py (dataset, normalização e carregadores)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import (
    PatchDataset,
    build_loaders,
    fold_indices,
    load_normalization_stats,
    normalize_patch,
)

TEST_SOURCE = "alphaearth"


def _stats() -> dict:
    """Estatísticas de normalização sintéticas (média 100, desvio 10 por banda)."""
    bands = ["B2", "B3", "B4", "B8"]
    return {
        "bands": bands,
        "per_band": {band: {"mean": 100.0, "std": 10.0, "nan_fraction": 0.0} for band in bands},
    }


def _storage_paths(tmp_path) -> dict:
    """Caminhos de armazenamento mínimos para os testes do estágio 09."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "models": tmp_path / "models",
        "artifacts_metrics": tmp_path / "artifacts" / "metrics",
        "artifacts_runs": tmp_path / "artifacts" / "runs",
    }


def _write_patch(path: Path, array: np.ndarray) -> None:
    """Persiste um patch sintético como .npy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _manifest_with_patches(tmp_path) -> tuple[dict, pd.DataFrame, dict]:
    """Manifesto sintético com 6 patches (2 por dobra) e estatísticas de normalização."""
    paths = _storage_paths(tmp_path)
    root = tmp_path
    records = []
    patch_size = 8
    for index in range(6):
        patch_id = f"tile_r{index:04d}_c0000"
        image_path = f"data/processed/patches/images/h/{patch_id}.npy"
        mask_path = f"data/processed/patches/masks/h/{patch_id}.npy"
        image = np.full((4, patch_size, patch_size), 100.0, dtype=np.float32)
        mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
        if index % 2 == 0:
            mask[2:6, 2:6] = 1
        _write_patch(root / image_path, image)
        _write_patch(root / mask_path, mask)
        records.append(
            {
                "patch_id": patch_id,
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
    manifest = pd.DataFrame(records)
    processed = paths["data_processed"]
    processed.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(processed / f"manifest_{TEST_SOURCE}.parquet", index=False)
    stats = _stats()
    (processed / f"normalization_stats_{TEST_SOURCE}.json").write_text(
        json.dumps(stats, ensure_ascii=False), encoding="utf-8"
    )
    return paths, manifest, stats


def test_normalize_patch_imputes_nan_and_z_scores() -> None:
    """NaN deve ser imputado pela média e o patch padronizado por banda."""
    image = np.array(
        [
            [[100.0, np.nan], [120.0, 80.0]],
            [[100.0, 100.0], [100.0, 100.0]],
            [[100.0, 100.0], [100.0, 100.0]],
            [[100.0, 100.0], [100.0, 100.0]],
        ],
        dtype=np.float32,
    )
    normalized = normalize_patch(image, _stats())
    assert normalized.shape == image.shape
    assert np.isfinite(normalized).all()
    assert abs(float(normalized[0, 0, 0]) - 0.0) < 1e-5  # média => 0
    assert abs(float(normalized[0, 0, 1]) - 0.0) < 1e-5  # NaN imputado com a média
    assert abs(float(normalized[0, 1, 0]) - 2.0) < 1e-5  # 120 => (120-100)/10


def test_normalize_patch_zeroes_band_without_stats() -> None:
    """Banda sem estatística finita deve ser zerada."""
    stats = {
        "bands": ["B2", "B3"],
        "per_band": {
            "B2": {"mean": None, "std": None, "nan_fraction": 1.0},
            "B3": {"mean": 10.0, "std": 2.0, "nan_fraction": 0.0},
        },
    }
    image = np.full((2, 4, 4), 5.0, dtype=np.float32)
    normalized = normalize_patch(image, stats)
    assert (normalized[0] == 0.0).all()
    assert abs(float(normalized[1, 0, 0]) - (5.0 - 10.0) / 2.0) < 1e-5


def test_fold_indices_splits_by_fold() -> None:
    """A dobra de validação deve ficar de fora do treino."""
    manifest = pd.DataFrame({"fold": [0, 0, 1, 1, 2, 2]})
    train_idx, val_idx = fold_indices(manifest, fold=1)
    assert list(train_idx) == [0, 1, 4, 5]
    assert list(val_idx) == [2, 3]


def test_fold_indices_requires_validation_patches() -> None:
    """Dobra sem patches deve falhar com mensagem clara."""
    manifest = pd.DataFrame({"fold": [0, 0]})
    with pytest.raises(ValueError, match="sem patches"):
        fold_indices(manifest, fold=2)


def test_load_normalization_stats_requires_stage08(tmp_path) -> None:
    """Sem as estatísticas do estágio 08, a carga deve falhar."""
    paths = _storage_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="estágio 08"):
        load_normalization_stats(paths, TEST_SOURCE)


def test_patch_dataset_getitem_shapes(tmp_path) -> None:
    """getitem deve devolver imagem normalizada (C,H,W) e máscara (1,H,W)."""
    paths, manifest, stats = _manifest_with_patches(tmp_path)
    dataset = PatchDataset(manifest, paths, stats)
    image, mask = dataset[0]
    assert image.shape == (4, 8, 8)
    assert mask.shape == (1, 8, 8)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}


def test_patch_dataset_marks_invalid_pixels_with_ignore_label(tmp_path) -> None:
    """Pixels sem observação válida devem receber rótulo -1 (ignorado na perda)."""
    paths, manifest, stats = _manifest_with_patches(tmp_path)
    image_path = tmp_path / str(manifest.loc[0, "image_path"])
    image = np.full((4, 8, 8), 100.0, dtype=np.float32)
    image[:, 0, :] = np.nan
    _write_patch(image_path, image)

    dataset = PatchDataset(manifest, paths, stats)
    image_tensor, mask = dataset[0]
    assert torch.isfinite(image_tensor).all()
    assert (mask[0, 0, :] == -1.0).all()
    assert (mask[0, 1:, :] >= 0.0).all()


def test_build_loaders_returns_train_and_val(tmp_path) -> None:
    """build_loaders deve dividir treino/validação pela dobra e embaralhar o treino."""
    paths, manifest, _ = _manifest_with_patches(tmp_path)
    train_loader, val_loader = build_loaders(
        paths, TEST_SOURCE, fold=0, batch_size=2, seed=42, num_workers=0
    )
    train_idx, val_idx = fold_indices(manifest, fold=0)
    assert len(train_idx) == 4  # dobras 1 e 2
    assert len(val_idx) == 2  # dobra 0
    assert len(list(train_loader)) == 2  # 4 patches / batch 2
    assert len(list(val_loader)) == 1  # 2 patches / batch 2
    images, masks = next(iter(train_loader))
    assert images.shape == (2, 4, 8, 8)
    assert masks.shape == (2, 1, 8, 8)


def test_build_loaders_requires_folds(tmp_path, monkeypatch) -> None:
    """Sem a coluna fold, a construção dos carregadores deve falhar."""
    import pandas as pd

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths, _, stats = _manifest_with_patches(tmp_path)
    manifest = pd.read_parquet(paths["data_processed"] / f"manifest_{TEST_SOURCE}.parquet")
    manifest["fold"] = None
    manifest.to_parquet(paths["data_processed"] / f"manifest_{TEST_SOURCE}.parquet", index=False)
    with pytest.raises(ValueError, match="estágio 07"):
        build_loaders(paths, TEST_SOURCE, fold=0, batch_size=2, seed=42, num_workers=0)
