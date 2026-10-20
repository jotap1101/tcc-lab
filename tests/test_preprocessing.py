"""Testes de src/data/preprocessing.py (composite alinhado e normalizado)."""

import numpy as np
import pytest

from src.config import get_config
from src.data.mask_utils import source_mask_path
from src.data.preprocessing import (
    build_preprocessed_composite,
    composite_file_name,
    composite_path,
    composite_preview_file_name,
    load_aligned_mosaic,
    mosaic_path,
    normalize_reflectance,
    save_composite_preview,
    verify_composite,
)

TEST_SOURCE = str(get_config()["ground_truth"]["chosen_source"])


def _write_geotiff(path, array, transform, crs="EPSG:31983") -> None:
    """Escreve um GeoTIFF de uma ou mais bandas para uso nos testes."""
    import rasterio

    if array.ndim == 2:
        array = array[None, :, :]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array)


def _storage_paths(tmp_path) -> dict:
    """Caminhos de armazenamento mínimos para os testes do estágio 05."""
    return {
        "data_raw_sentinel2": tmp_path / "data" / "raw" / "sentinel2",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
    }


def _write_inputs(tmp_path, monkeypatch, mask_offset=0.0):
    """Escreve mosaico (4 bandas) e máscara da fonte em grids possivelmente distintos."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)

    mosaic = np.zeros((4, 4, 4), dtype=np.uint16)
    mosaic[:, 0, 0] = [2000, 3000, 4000, 5000]
    mosaic_file = mosaic_path(paths)
    mosaic_file.parent.mkdir(parents=True)
    _write_geotiff(mosaic_file, mosaic, from_origin(0, 4, 10, 10))

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    mask_file = source_mask_path(TEST_SOURCE, paths)
    mask_file.parent.mkdir(parents=True)
    _write_geotiff(mask_file, mask, from_origin(mask_offset, 4, 10, 10))

    return paths, mosaic_file, mask_file


def _mask_transform(mask_path) -> np.ndarray:
    """Retorna os parâmetros da transform da máscara como array numpy."""
    import rasterio

    with rasterio.open(mask_path) as src:
        transform = src.transform
    return np.array(transform.to_gdal())


def test_artifact_file_names_derive_from_config() -> None:
    """Os nomes dos artefatos devem derivar da região e do ano de referência."""
    config = get_config()
    assert composite_file_name().startswith("composite_")
    assert composite_preview_file_name().endswith(".png")
    assert config["aoi"]["region_code"] in composite_file_name()


def test_composite_file_name_and_path_accept_year(tmp_path) -> None:
    """O nome e o caminho do composite devem aceitar um ano explícito."""
    paths = _storage_paths(tmp_path)
    assert composite_file_name(2018) == "composite_310044_2018"
    assert composite_path(paths, 2018).name == "composite_310044_2018.tif"


def test_mosaic_path_accepts_period(tmp_path) -> None:
    """O caminho do mosaico deve aceitar um período explícito (ex.: ano da Emater)."""
    paths = _storage_paths(tmp_path)
    target = mosaic_path(paths, "2018-01-01", "2018-12-31")
    assert target.name == "sentinel2_310044_2018-01-01_2018-12-31.tif"
    assert target.parent == paths["data_raw_sentinel2"]


def test_mosaic_path_points_to_raw_sentinel2(tmp_path) -> None:
    """O caminho do mosaico deve apontar para data/raw/sentinel2/."""
    paths = _storage_paths(tmp_path)
    target = mosaic_path(paths)
    assert target.parent == paths["data_raw_sentinel2"]
    assert target.suffix == ".tif"


def test_composite_path_points_to_processed_composites(tmp_path) -> None:
    """O composite deve residir em data/processed/composites/."""
    paths = _storage_paths(tmp_path)
    target = composite_path(paths)
    assert target.parent == paths["data_processed_composites"]
    assert target.suffix == ".tif"


def test_normalize_reflectance_scales_and_clips() -> None:
    """A normalização deve aplicar a escala 1e-4 e saturar em [0, 1]."""
    values = np.array([0, 5000, 10000, 20000, -1000], dtype=np.int64)
    normalized = normalize_reflectance(values)
    assert normalized.dtype == np.float32
    assert normalized.tolist() == [0.0, 0.5, 1.0, 1.0, 0.0]


def test_normalize_reflectance_preserves_nan() -> None:
    """Pixels sem dado (NaN) devem ser preservados pela normalização."""
    values = np.array([[np.nan, 5000.0]])
    normalized = normalize_reflectance(values)
    assert np.isnan(normalized[0, 0])
    assert normalized[0, 1] == pytest.approx(0.5)


def test_load_aligned_mosaic_same_grid(tmp_path, monkeypatch) -> None:
    """Com grids idênticos, o mosaico é lido diretamente e marcado como alinhado."""
    _, mosaic, mask = _write_inputs(tmp_path, monkeypatch)
    composite = load_aligned_mosaic(mosaic, mask)
    assert composite.aligned is True
    assert composite.array.shape == (4, 4, 4)
    assert composite.array[0, 0, 0] == 2000


def test_load_aligned_mosaic_different_grid(tmp_path, monkeypatch) -> None:
    """Com grids distintos, o mosaico é reamostrado para o grid da máscara."""
    _, mosaic, mask = _write_inputs(tmp_path, monkeypatch, mask_offset=5.0)
    composite = load_aligned_mosaic(mosaic, mask)
    assert composite.aligned is False
    assert composite.array.shape == (4, 4, 4)
    np.testing.assert_allclose(composite.profile["transform"].to_gdal(), _mask_transform(mask))


def test_build_preprocessed_composite_idempotent(tmp_path, monkeypatch) -> None:
    """O composite deve ser criado uma vez e reutilizado em reexecuções."""
    paths, _, _ = _write_inputs(tmp_path, monkeypatch)
    first = build_preprocessed_composite(paths)
    second = build_preprocessed_composite(paths)
    assert first == second
    assert first.is_file()
    assert first.suffix == ".tif"


def test_build_preprocessed_composite_requires_mosaic(tmp_path, monkeypatch) -> None:
    """Sem o mosaico do estágio 01, a construção deve falhar com FileNotFoundError."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask_file = source_mask_path(TEST_SOURCE, paths)
    mask_file.parent.mkdir(parents=True)
    _write_geotiff(mask_file, mask, from_origin(0, 4, 10, 10))
    with pytest.raises(FileNotFoundError, match="estágio 01"):
        build_preprocessed_composite(paths)


def test_build_preprocessed_composite_requires_mask(tmp_path, monkeypatch) -> None:
    """Sem a máscara de referência do ano, a construção deve falhar."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    mosaic = np.zeros((4, 4, 4), dtype=np.uint16)
    mosaic_file = mosaic_path(paths)
    mosaic_file.parent.mkdir(parents=True)
    _write_geotiff(mosaic_file, mosaic, from_origin(0, 4, 10, 10))
    with pytest.raises(FileNotFoundError, match="Máscara de referência"):
        build_preprocessed_composite(paths)


def test_verify_composite_reports(tmp_path, monkeypatch) -> None:
    """A verificação deve reportar grid, CRS, bandas, alinhamento e intervalo."""
    paths, _, _ = _write_inputs(tmp_path, monkeypatch)
    build_preprocessed_composite(paths)
    stats = verify_composite(paths)
    assert stats["shape"] == [4, 4]
    assert stats["crs"] == "EPSG:31983"
    assert stats["bands"] == 4
    assert stats["aligned_to_mask"] is True
    vmin, vmax = stats["value_range"]
    assert vmin >= 0.0
    assert vmax <= 1.0


def test_save_composite_preview_idempotent(tmp_path, monkeypatch) -> None:
    """A figura do composite deve ser persistida uma única vez."""
    paths, _, _ = _write_inputs(tmp_path, monkeypatch)
    build_preprocessed_composite(paths)
    first = save_composite_preview(paths)
    second = save_composite_preview(paths)
    assert first == second
    assert first.is_file()
    assert first.read_bytes().startswith(b"\x89PNG")
