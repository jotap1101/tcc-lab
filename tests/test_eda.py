"""Testes de src/data/eda.py (estatísticas de normalização e sanidade do dataset)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.config import get_config
from src.data.eda import (
    compute_normalization_stats,
    eda_figure_file_name,
    normalization_stats_is_current,
    normalization_stats_path,
    render_eda_figure,
    run_sanity_checks,
    save_eda_figure,
)
from src.data.patch_generation import manifest_path, patch_fingerprint_hash

TEST_SOURCE = "alphaearth"


def _storage_paths(tmp_path) -> dict:
    """Caminhos de armazenamento mínimos para os testes do estágio 08."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "data_processed_ground_truth": tmp_path / "data" / "processed" / "ground_truth",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "data_processed_patches_images": tmp_path / "data" / "processed" / "patches" / "images",
        "data_processed_patches_masks": tmp_path / "data" / "processed" / "patches" / "masks",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
    }


def _patch_config(monkeypatch, **overrides) -> None:
    """Configura um patch_size pequeno (e outras chaves) para os testes."""
    import copy

    config = copy.deepcopy(get_config())
    config["data"]["patch_size"] = 2
    for key, value in overrides.items():
        config["data"][key] = value
    monkeypatch.setattr("src.data.eda.get_config", lambda: config)


def _write_manifest_with_patches(tmp_path, monkeypatch, patch_count: int = 3) -> dict:
    """Escreve um manifesto e os patches de imagem/máscara correspondentes."""
    import pandas as pd

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    _patch_config(monkeypatch)
    paths = _storage_paths(tmp_path)

    bands = get_config()["data"]["bands"]
    root = tmp_path
    records = []
    for index in range(patch_count):
        pid = f"tile_r{index:04d}_c0000"
        image = np.random.default_rng(index).uniform(0, 1, size=(len(bands), 2, 2))
        mask = np.zeros((2, 2), dtype=np.uint8)
        mask[0, 0] = 1
        image_path = paths["data_processed_patches_images"] / f"{pid}.npy"
        mask_path = paths["data_processed_patches_masks"] / f"{pid}.npy"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(image_path, image.astype(np.float32))
        np.save(mask_path, mask)
        records.append(
            {
                "patch_id": pid,
                "tile_id": "tile_310044_2023",
                "fold": index % 2,
                "row": index,
                "col": 0,
                "bbox": f"{index * 2},0,{index * 2 + 2},2",
                "coffee_ratio": 0.25 + 0.1 * index,
                "mask_source": "alphaearth",
                "image_path": str(image_path.relative_to(root)),
                "mask_path": str(mask_path.relative_to(root)),
            }
        )
    manifest_file = manifest_path(paths, TEST_SOURCE)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(manifest_file, index=False)
    return paths


def test_eda_figure_file_name_derives_from_config() -> None:
    """O nome da figura deve derivar da região e do ano de referência."""
    config = get_config()
    assert eda_figure_file_name(TEST_SOURCE).endswith(".png")
    assert config["aoi"]["region_code"] in eda_figure_file_name(TEST_SOURCE)


def test_normalization_stats_path_points_to_processed(tmp_path) -> None:
    """As estatísticas devem residir em data/processed/."""
    paths = _storage_paths(tmp_path)
    target = normalization_stats_path(paths, TEST_SOURCE)
    assert target.parent == paths["data_processed"]
    assert target.suffix == ".json"


def test_compute_normalization_stats_consistent_with_arrays(tmp_path, monkeypatch) -> None:
    """As estatísticas devem reproduzir média/desvio calculados sobre os arrays."""
    paths = _write_manifest_with_patches(tmp_path, monkeypatch)
    stats = compute_normalization_stats(paths, TEST_SOURCE)
    assert stats["n_patches"] == 3
    assert stats["per_band"]["B2"]["nan_fraction"] == 0.0
    assert 0.0 < stats["per_band"]["B2"]["mean"] < 1.0
    assert stats["per_band"]["B2"]["std"] >= 0.0


def test_compute_normalization_stats_idempotent(tmp_path, monkeypatch) -> None:
    """Reexecuções devem reutilizar as estatísticas sem regravar o arquivo."""
    paths = _write_manifest_with_patches(tmp_path, monkeypatch)
    stats_path = normalization_stats_path(paths, TEST_SOURCE)
    compute_normalization_stats(paths, TEST_SOURCE)
    mtime = stats_path.stat().st_mtime_ns

    compute_normalization_stats(paths, TEST_SOURCE)
    assert stats_path.stat().st_mtime_ns == mtime
    assert normalization_stats_is_current(paths, TEST_SOURCE)


def test_compute_normalization_stats_invalidates_on_input_change(tmp_path, monkeypatch) -> None:
    """Mudanças nas entradas devem tornar as estatísticas persistidas obsoletas."""
    import copy

    paths = _write_manifest_with_patches(tmp_path, monkeypatch)
    compute_normalization_stats(paths, TEST_SOURCE)
    assert normalization_stats_is_current(paths, TEST_SOURCE)

    config = copy.deepcopy(get_config())
    config["data"]["bands"] = ["B2", "B3", "B4", "B8", "B11"]
    monkeypatch.setattr("src.data.eda.get_config", lambda: config)
    assert not normalization_stats_is_current(paths, TEST_SOURCE)


def test_run_sanity_checks_reports_dataset(tmp_path, monkeypatch) -> None:
    """A verificação deve reportar contagens, café, dobras e amostra consistente."""
    paths = _write_manifest_with_patches(tmp_path, monkeypatch)
    checks = run_sanity_checks(paths, TEST_SOURCE)
    assert checks["n_patches"] == 3
    assert checks["fold"]["present"] is True
    assert checks["sample"]["expected_shape"] == [4, 2, 2]
    assert checks["sample"]["mask_values"] == [0, 1]
    assert checks["issues"] == []


def test_run_sanity_checks_detects_non_binary_mask(tmp_path, monkeypatch) -> None:
    """Valores não binários na máscara devem ser reportados como problema."""
    import pandas as pd

    paths = _write_manifest_with_patches(tmp_path, monkeypatch)
    records = pd.read_parquet(manifest_path(paths, TEST_SOURCE))
    mask_path = tmp_path / records.loc[0, "mask_path"]
    np.save(mask_path, np.full((2, 2), 2, dtype=np.uint8))

    checks = run_sanity_checks(paths, TEST_SOURCE)
    assert any("não binários" in issue for issue in checks["issues"])


def test_save_eda_figure_idempotent(tmp_path, monkeypatch) -> None:
    """A figura de resumo deve ser persistida uma única vez."""
    paths = _write_manifest_with_patches(tmp_path, monkeypatch)
    first = save_eda_figure(paths, TEST_SOURCE)
    second = save_eda_figure(paths, TEST_SOURCE)
    assert first == second
    assert first.is_file()
    assert first.read_bytes().startswith(b"\x89PNG")


def test_render_eda_figure_returns_png(tmp_path, monkeypatch) -> None:
    """A renderização da figura deve devolver bytes PNG."""
    import pandas as pd

    paths = _write_manifest_with_patches(tmp_path, monkeypatch)
    manifest = pd.read_parquet(manifest_path(paths, TEST_SOURCE))
    stats = compute_normalization_stats(paths, TEST_SOURCE)
    png = render_eda_figure(manifest, stats, paths)
    assert png["png"].startswith(b"\x89PNG")
    assert png["pdf"].startswith(b"%PDF")


def test_compute_normalization_stats_requires_manifest(tmp_path, monkeypatch) -> None:
    """Sem o manifesto do estágio 06, a computação deve falhar."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    with pytest.raises(FileNotFoundError, match="estágio 06"):
        compute_normalization_stats(paths, TEST_SOURCE)


def test_normalization_stats_json_is_valid(tmp_path, monkeypatch) -> None:
    """O arquivo persistido deve ser JSON válido com o fingerprint."""
    paths = _write_manifest_with_patches(tmp_path, monkeypatch)
    stats = compute_normalization_stats(paths, TEST_SOURCE)
    stats_path = normalization_stats_path(paths, TEST_SOURCE)
    loaded = json.loads(stats_path.read_text(encoding="utf-8"))
    assert loaded["fingerprint"]["fingerprint_hash"] == patch_fingerprint_hash(paths, TEST_SOURCE)
    assert loaded == stats
