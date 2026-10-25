"""Testes de src/data/patch_generation.py (patches 512x512 + manifesto por fonte)."""

import copy
from pathlib import Path

import numpy as np
import pytest

from src.config import get_config
from src.data.mask_utils import source_mask_path
from src.data.patch_generation import (
    generate_patches,
    grid_dims,
    image_patches_dir,
    load_manifest,
    manifest_has_folds,
    manifest_path,
    patch_bbox,
    patch_fingerprint_hash,
    patch_id,
    patch_montage_file_name,
    require_manifest,
    save_patch_montage,
    verify_manifest,
)
from src.data.preprocessing import composite_path

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


def _patch_config(monkeypatch, **overrides) -> None:
    """Configura um patch_size pequeno (e outras chaves) para os testes."""
    config = copy.deepcopy(get_config())
    config["data"]["patch_size"] = 2
    for key, value in overrides.items():
        config["data"][key] = value
    monkeypatch.setattr("src.data.patch_generation.get_config", lambda: config)


def _storage_paths(tmp_path) -> dict:
    """Caminhos de armazenamento mínimos para os testes do estágio 06."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
    }


def _write_inputs(tmp_path, monkeypatch, coffee_ratio=0.5) -> tuple[dict, Path, Path]:
    """Escreve composite (4 bandas) e máscara da fonte em grid comum 4x4."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    _patch_config(monkeypatch)
    paths = _storage_paths(tmp_path)

    composite = np.zeros((4, 4, 4), dtype=np.uint16)
    composite[0, 0, 0] = 2000
    composite_file = composite_path(paths)
    composite_file.parent.mkdir(parents=True)
    _write_geotiff(composite_file, composite, from_origin(0, 4, 10, 10))

    mask = np.zeros((4, 4), dtype=np.uint8)
    coffee_pixels = int(coffee_ratio * 4)
    mask[0, :coffee_pixels] = 1
    mask_file = source_mask_path(TEST_SOURCE, paths)
    mask_file.parent.mkdir(parents=True)
    _write_geotiff(mask_file, mask, from_origin(0, 4, 10, 10))

    return paths, composite_file, mask_file


def test_artifact_file_names_derive_from_config() -> None:
    """Os nomes dos artefatos devem derivar da região, ano e fonte."""
    config = get_config()
    assert patch_montage_file_name(TEST_SOURCE).endswith(".png")
    assert config["aoi"]["region_code"] in patch_montage_file_name(TEST_SOURCE)


def test_manifest_path_points_to_processed(tmp_path) -> None:
    """O manifesto da fonte deve residir em data/processed/."""
    paths = _storage_paths(tmp_path)
    target = manifest_path(paths, TEST_SOURCE)
    assert target.parent == paths["data_processed"]
    assert target.suffix == ".parquet"
    assert TEST_SOURCE in target.name


def test_require_manifest_raises_when_absent(tmp_path) -> None:
    """require_manifest deve falhar quando o manifesto da fonte está ausente."""
    paths = _storage_paths(tmp_path)
    with pytest.raises(FileNotFoundError):
        require_manifest(paths, TEST_SOURCE)


def test_load_manifest_reads_and_validates(tmp_path, monkeypatch) -> None:
    """load_manifest deve ler o manifesto persistido e falhar se vazio."""
    import pandas as pd

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    manifest_file = manifest_path(paths, TEST_SOURCE)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patch_id": ["p1"]}).to_parquet(manifest_file, index=False)
    assert len(load_manifest(paths, TEST_SOURCE)) == 1

    pd.DataFrame(columns=["patch_id"]).to_parquet(manifest_file, index=False)
    with pytest.raises(ValueError):
        load_manifest(paths, TEST_SOURCE)


def test_manifest_has_folds() -> None:
    """manifest_has_folds deve indicar a presença de fold sem valores nulos."""
    import pandas as pd

    assert manifest_has_folds(pd.DataFrame({"fold": [0, 1, 2]}))
    assert not manifest_has_folds(pd.DataFrame({"fold": [0, None]}))
    assert not manifest_has_folds(pd.DataFrame({"patch_id": ["p1"]}))


def test_patch_id_is_deterministic() -> None:
    """O identificador do patch deve ser determinístico e conter linha e coluna."""
    assert patch_id("tile_310044_2023", 3, 7) == "tile_310044_2023_r0003_c0007"


def test_patch_bbox_derived_from_affine() -> None:
    """A bbox geográfica deve derivar da transformação afim (independente de versão)."""
    from rasterio.transform import Affine

    transform = Affine(10.0, 0.0, 300000.0, 0.0, -10.0, 7780000.0)
    assert patch_bbox(transform, row=1, col=2, patch_size=2) == (
        "300020.00,7779970.00,300040.00,7779990.00"
    )


def test_grid_dims_drops_partial_edges() -> None:
    """Bordas parciais devem ser descartadas pelo grid determinístico."""
    assert grid_dims(4, 4, 2) == (2, 2)
    assert grid_dims(5, 7, 2) == (2, 3)


def test_generate_patches_creates_manifest_and_patches(tmp_path, monkeypatch) -> None:
    """A geração deve persistir patches e manifesto da fonte com o schema esperado."""
    import pandas as pd

    paths, _, _ = _write_inputs(tmp_path, monkeypatch)
    manifest = generate_patches(paths, TEST_SOURCE)
    assert manifest.is_file()

    frame = pd.read_parquet(manifest)
    expected_columns = {
        "patch_id",
        "tile_id",
        "fold",
        "row",
        "col",
        "bbox",
        "coffee_ratio",
        "mask_source",
        "image_path",
        "mask_path",
    }
    assert set(frame.columns) >= expected_columns
    assert len(frame) == 4
    assert (frame["mask_source"] == TEST_SOURCE).all()
    image_dir = image_patches_dir(paths, TEST_SOURCE, patch_fingerprint_hash(paths, TEST_SOURCE))
    assert image_dir.is_dir()
    assert (image_dir / f"{frame.loc[0, 'patch_id']}.npy").is_file()


def test_generate_patches_discards_patches_without_valid_pixels(tmp_path, monkeypatch) -> None:
    """Patches com fração de pixels válidos abaixo do mínimo devem ser descartados."""
    import pandas as pd
    from rasterio.transform import from_origin

    paths, composite_file, _ = _write_inputs(tmp_path, monkeypatch)
    composite = np.zeros((4, 4, 4), dtype=np.float32)
    composite[:, 2:, :] = np.nan  # metade inferior (fora do AOI) sem observação
    _write_geotiff(composite_file, composite, from_origin(0, 4, 10, 10))
    _patch_config(monkeypatch, min_valid_fraction=0.5)

    manifest = generate_patches(paths, TEST_SOURCE)
    frame = pd.read_parquet(manifest)
    assert len(frame) == 2
    assert set(frame["row"].tolist()) == {0}
    assert (frame["valid_fraction"] == 1.0).all()


def test_generate_patches_filters_by_coffee_ratio(tmp_path, monkeypatch) -> None:
    """Patches abaixo da proporção mínima de café devem ser descartados."""
    import pandas as pd

    paths, _, _ = _write_inputs(tmp_path, monkeypatch, coffee_ratio=0.5)
    _patch_config(monkeypatch, coffee_min_ratio=0.5)
    manifest = generate_patches(paths, TEST_SOURCE)
    frame = pd.read_parquet(manifest)
    assert len(frame) == 1
    assert frame.loc[0, "row"] == 0
    assert frame.loc[0, "col"] == 0


def test_generate_patches_idempotent(tmp_path, monkeypatch) -> None:
    """Reexecuções devem reutilizar o manifesto e não regravar patches."""
    paths, _, _ = _write_inputs(tmp_path, monkeypatch)
    generate_patches(paths, TEST_SOURCE)
    image_dir = image_patches_dir(paths, TEST_SOURCE, patch_fingerprint_hash(paths, TEST_SOURCE))
    patches = sorted(image_dir.glob("*.npy"))
    mtimes = {path: path.stat().st_mtime_ns for path in patches}

    manifest = generate_patches(paths, TEST_SOURCE)
    assert manifest.is_file()
    for path in patches:
        assert path.stat().st_mtime_ns == mtimes[path]


def test_generate_patches_versions_stale_inputs(tmp_path, monkeypatch) -> None:
    """Mudanças nas entradas criam nova subpasta; patches antigos são preservados."""
    import pandas as pd
    from rasterio.transform import from_origin

    paths, composite_file, mask_file = _write_inputs(tmp_path, monkeypatch)
    generate_patches(paths, TEST_SOURCE)
    first_hash = patch_fingerprint_hash(paths, TEST_SOURCE)
    first_image_dir = image_patches_dir(paths, TEST_SOURCE, first_hash)
    stale_patch = sorted(first_image_dir.glob("*.npy"))[0]

    _write_geotiff(composite_file, np.zeros((4, 5, 5), dtype=np.uint16), from_origin(0, 5, 10, 10))
    _write_geotiff(mask_file, np.zeros((5, 5), dtype=np.uint8), from_origin(0, 5, 10, 10))

    generate_patches(paths, TEST_SOURCE)
    second_hash = patch_fingerprint_hash(paths, TEST_SOURCE)
    assert second_hash != first_hash
    assert stale_patch.is_file()
    assert image_patches_dir(paths, TEST_SOURCE, second_hash).is_dir()
    regenerated = pd.read_parquet(manifest_path(paths, TEST_SOURCE))
    assert second_hash in regenerated.loc[0, "image_path"]


def test_generate_patches_requires_composite(tmp_path, monkeypatch) -> None:
    """Sem o composite do estágio 05, a geração deve falhar."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    _patch_config(monkeypatch)
    paths = _storage_paths(tmp_path)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask_file = source_mask_path(TEST_SOURCE, paths)
    mask_file.parent.mkdir(parents=True)
    _write_geotiff(mask_file, mask, from_origin(0, 4, 10, 10))
    with pytest.raises(FileNotFoundError, match="estágio 05"):
        generate_patches(paths, TEST_SOURCE)


def test_generate_patches_requires_mask(tmp_path, monkeypatch) -> None:
    """Sem a máscara da fonte do estágio 02, a geração deve falhar."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    _patch_config(monkeypatch)
    paths = _storage_paths(tmp_path)
    composite = np.zeros((4, 4, 4), dtype=np.uint16)
    composite_file = composite_path(paths)
    composite_file.parent.mkdir(parents=True)
    _write_geotiff(composite_file, composite, from_origin(0, 4, 10, 10))
    with pytest.raises(FileNotFoundError, match="estágio 02"):
        generate_patches(paths, TEST_SOURCE)


def test_generate_patches_requires_aligned_grid(tmp_path, monkeypatch) -> None:
    """Grids divergentes entre composite e máscara devem ser rejeitados."""
    from rasterio.transform import from_origin

    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    _patch_config(monkeypatch)
    paths = _storage_paths(tmp_path)

    composite_file = composite_path(paths)
    composite_file.parent.mkdir(parents=True)
    _write_geotiff(composite_file, np.zeros((2, 4, 4), dtype=np.uint16), from_origin(0, 4, 10, 10))
    mask_file = source_mask_path(TEST_SOURCE, paths)
    mask_file.parent.mkdir(parents=True)
    _write_geotiff(mask_file, np.zeros((4, 4), dtype=np.uint8), from_origin(5, 9, 10, 10))

    with pytest.raises(ValueError, match="grids distintos"):
        generate_patches(paths, TEST_SOURCE)


def test_verify_manifest_reports(tmp_path, monkeypatch) -> None:
    """A verificação deve reportar contagem, café, formato e fontes."""
    paths, _, _ = _write_inputs(tmp_path, monkeypatch)
    generate_patches(paths, TEST_SOURCE)
    stats = verify_manifest(paths, TEST_SOURCE)
    assert stats["n_patches"] == 4
    assert stats["patch_shape"] == [4, 2, 2]
    assert stats["coffee_ratio"]["max"] > 0.0
    assert stats["mask_source"] == [TEST_SOURCE]


def test_save_patch_montage_idempotent(tmp_path, monkeypatch) -> None:
    """A figura do mosaico deve ser persistida uma única vez."""
    paths, _, _ = _write_inputs(tmp_path, monkeypatch)
    generate_patches(paths, TEST_SOURCE)
    first = save_patch_montage(paths, TEST_SOURCE)
    second = save_patch_montage(paths, TEST_SOURCE)
    assert first == second
    assert first.is_file()
    assert first.read_bytes().startswith(b"\x89PNG")
