"""Testes de src/data/spatial_split.py (divisão espacial k-fold do manifesto)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src.config import get_config
from src.data.patch_generation import manifest_path
from src.data.spatial_split import (
    assign_spatial_folds,
    bbox_centroid,
    save_split_figure,
    spatial_fold_centroids,
    split_figure_file_name,
    split_fingerprint_hash,
    verify_split,
)

TEST_SOURCE = "alphaearth"


def _storage_paths(tmp_path) -> dict:
    """Caminhos de armazenamento mínimos para os testes do estágio 07."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "data_processed_ground_truth": tmp_path / "data" / "processed" / "ground_truth",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
    }


def _patch_config(monkeypatch, **overrides) -> None:
    """Configura um fold_count próprio para os testes."""
    config = copy.deepcopy(get_config())
    config["splits"]["fold_count"] = overrides.get("fold_count", 5)
    monkeypatch.setattr("src.data.spatial_split.get_config", lambda: config)


def _write_manifest(tmp_path) -> tuple[dict, Path]:
    """Escreve um manifesto sintético com um grid espacial de patches."""
    import pandas as pd

    paths = _storage_paths(tmp_path)
    patch_size = 512
    records = []
    for row in range(8):
        for col in range(5):
            min_x = col * patch_size
            min_y = row * patch_size
            records.append(
                {
                    "patch_id": f"tile_r{row:04d}_c{col:04d}",
                    "tile_id": "tile_310044_2023",
                    "fold": None,
                    "row": row,
                    "col": col,
                    "bbox": f"{min_x},{min_y},{min_x + patch_size},{min_y + patch_size}",
                    "coffee_ratio": 0.1 + 0.01 * (row + col),
                    "mask_source": "alphaearth",
                    "image_path": "data/processed/patches/images/h/img.npy",
                    "mask_path": "data/processed/patches/masks/h/mask.npy",
                }
            )
    manifest_file = manifest_path(paths, TEST_SOURCE)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(manifest_file, index=False)
    return paths, manifest_file


def test_artifact_file_names_derive_from_config() -> None:
    """O nome da figura deve derivar da região, ano e fonte."""
    config = get_config()
    assert split_figure_file_name(TEST_SOURCE).endswith(".png")
    assert config["aoi"]["region_code"] in split_figure_file_name(TEST_SOURCE)


def test_bbox_centroid_parses_bbox() -> None:
    """O centroide deve ser o ponto médio da bbox registrada no manifesto."""
    assert bbox_centroid("100.00,200.00,300.00,400.00") == (200.0, 300.0)


def test_spatial_fold_centroids_deterministic() -> None:
    """A divisão deve ser idêntica para a mesma semente e entradas."""
    rng = np.random.default_rng(0)
    centroids = rng.uniform(0, 100, size=(50, 2))
    first = spatial_fold_centroids(centroids, k=5, seed=42)
    second = spatial_fold_centroids(centroids, k=5, seed=42)
    np.testing.assert_array_equal(first, second)


def test_spatial_fold_centroids_groups_contiguous_regions() -> None:
    """Regiões espacialmente separadas devem cair em dobras distintas."""
    west = np.array([[1000.0, 1000.0], [1200.0, 1000.0], [1100.0, 1200.0]])
    east = np.array([[90000.0, 90000.0], [92000.0, 90000.0], [91000.0, 92000.0]])
    centroids = np.vstack([west, east])
    labels = spatial_fold_centroids(centroids, k=2, seed=42)
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_spatial_fold_centroids_requires_enough_patches() -> None:
    """Com menos patches que dobras, a divisão deve falhar com mensagem clara."""
    centroids = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    with pytest.raises(ValueError, match="Patches insuficientes"):
        spatial_fold_centroids(centroids, k=5, seed=42)


def test_assign_spatial_folds_fills_column(tmp_path, monkeypatch) -> None:
    """A divisão deve preencher a coluna fold do manifesto com dobras válidas."""
    import pandas as pd

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths, manifest_file = _write_manifest(tmp_path)
    result = assign_spatial_folds(paths, TEST_SOURCE)
    assert result == manifest_file
    frame = pd.read_parquet(manifest_file)
    assert frame["fold"].notna().all()
    assert set(frame["fold"].unique()) <= set(range(5))


def test_assign_spatial_folds_idempotent(tmp_path, monkeypatch) -> None:
    """Reexecuções devem reutilizar a divisão sem regravar o manifesto."""
    import pandas as pd

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths, manifest_file = _write_manifest(tmp_path)
    assign_spatial_folds(paths, TEST_SOURCE)
    first = pd.read_parquet(manifest_file)
    mtime = manifest_file.stat().st_mtime_ns

    assign_spatial_folds(paths, TEST_SOURCE)
    assert manifest_file.stat().st_mtime_ns == mtime
    second = pd.read_parquet(manifest_file)
    assert list(first["fold"]) == list(second["fold"])


def test_assign_spatial_folds_requires_manifest(tmp_path, monkeypatch) -> None:
    """Sem o manifesto do estágio 06, a divisão deve falhar."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="estágio 06"):
        assign_spatial_folds(paths, TEST_SOURCE)


def test_assign_spatial_folds_recomputes_on_config_change(tmp_path, monkeypatch) -> None:
    """Mudanças na configuração de dobras devem recomputar e reutilizar depois."""
    import pandas as pd

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    _patch_config(monkeypatch, fold_count=3)
    paths, manifest_file = _write_manifest(tmp_path)
    assign_spatial_folds(paths, TEST_SOURCE)
    frame = pd.read_parquet(manifest_file)
    assert len(set(frame["fold"].unique())) == 3

    mtime = manifest_file.stat().st_mtime_ns
    assign_spatial_folds(paths, TEST_SOURCE)
    assert manifest_file.stat().st_mtime_ns == mtime


def test_verify_split_reports_per_fold(tmp_path, monkeypatch) -> None:
    """A verificação deve reportar contagens consistentes por dobra."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths, _ = _write_manifest(tmp_path)
    assign_spatial_folds(paths, TEST_SOURCE)
    stats = verify_split(paths, TEST_SOURCE)
    assert stats["fold_count"] == 5
    assert stats["n_patches"] == 40
    assert len(stats["per_fold"]) == 5
    assert sum(row["n_patches"] for row in stats["per_fold"]) == 40


def test_split_fingerprint_hash_tracks_assignment(tmp_path, monkeypatch) -> None:
    """O hash da divisão deve mudar quando a atribuição patch→dobra muda."""
    import pandas as pd

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths, manifest_file = _write_manifest(tmp_path)
    assign_spatial_folds(paths, TEST_SOURCE)
    first = split_fingerprint_hash(paths, TEST_SOURCE)

    frame = pd.read_parquet(manifest_file)
    frame.loc[0, "fold"] = (int(frame.loc[0, "fold"]) + 1) % 5
    frame.to_parquet(manifest_file, index=False)

    assert split_fingerprint_hash(paths, TEST_SOURCE) != first


def test_save_split_figure_idempotent(tmp_path, monkeypatch) -> None:
    """A figura da divisão deve ser persistida uma única vez."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths, _ = _write_manifest(tmp_path)
    assign_spatial_folds(paths, TEST_SOURCE)
    first = save_split_figure(paths, TEST_SOURCE)
    second = save_split_figure(paths, TEST_SOURCE)
    assert first == second
    assert first.is_file()
    assert first.read_bytes().startswith(b"\x89PNG")
