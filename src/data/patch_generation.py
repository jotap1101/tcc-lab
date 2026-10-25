"""Geração de patches 512x512 e manifesto de dados por fonte (estágio 06).

Corta o composite normalizado do ano da fonte (estágio 05) e a máscara binária
da própria fonte (estágio 02), ambos sobre o mesmo grid, em patches de
`data.patch_size` com sobreposição nula (bordas parciais descartadas), descarta
patches com fração de pixels válidos abaixo de `data.min_valid_fraction` (área
fora do AOI), filtra pela proporção mínima de café (`data.coffee_min_ratio`) e
registra cada patch em um manifesto Parquet por fonte
(`data/processed/manifest_<fonte>.parquet`) com paths relativos à raiz tcc/. Cada
fonte habilitada tem o seu próprio conjunto de patches e manifesto (mesmo
composite por ano quando compartilham o ano).
A geração é determinística e idempotente: reexecuções reutilizam o manifesto
vigente (fingerprint de configuração e entradas) e os patches já persistidos,
escrevendo apenas o que faltar.
"""

from __future__ import annotations

import hashlib
import io as stdlib_io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.mask_utils import reference_year, source_mask_path
from src.data.preprocessing import composite_file_name, composite_path
from src.data.raster_utils import same_grid

MANIFEST_SCHEMA_VERSION = 2
BBOX_SCHEMA_VERSION = 1  # versão do cálculo de `bbox` (invalida o manifesto, não os patches)


def image_patches_dir(
    storage_paths: dict[str, Path], source_name: str, fingerprint_hash: str
) -> Path:
    """Diretório canônico dos patches de imagem da fonte, versionado por fingerprint."""
    return storage_paths["data_processed_patches"] / source_name / "images" / fingerprint_hash


def mask_patches_dir(
    storage_paths: dict[str, Path], source_name: str, fingerprint_hash: str
) -> Path:
    """Diretório canônico dos patches de máscara da fonte, versionado por fingerprint."""
    return storage_paths["data_processed_patches"] / source_name / "masks" / fingerprint_hash


def manifest_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho canônico do manifesto de dados da fonte (data/processed/)."""
    return storage_paths["data_processed"] / f"manifest_{source_name}.parquet"


def manifest_meta_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho do metadata do manifesto da fonte (fingerprint para idempotência)."""
    return storage_paths["data_processed"] / f"manifest_{source_name}.meta.json"


def require_manifest(storage_paths: dict[str, Path], source_name: str) -> None:
    """Falha com mensagem clara quando o manifesto da fonte do estágio 06 está ausente."""
    from src import io

    manifest = manifest_path(storage_paths, source_name)
    if not io.path_exists(manifest):
        raise FileNotFoundError(f"Manifesto do estágio 06 não encontrado: {manifest}")


def load_manifest(storage_paths: dict[str, Path], source_name: str) -> pd.DataFrame:
    """Carrega o manifesto persistido da fonte e valida que não está vazio."""
    from src import io

    local_manifest = io.ensure_local_copy(manifest_path(storage_paths, source_name))
    manifest = pd.read_parquet(local_manifest)
    if manifest.empty:
        raise ValueError(f"Manifesto vazio da fonte {source_name}; execute o estágio 06 antes.")
    return manifest


def manifest_has_folds(manifest: pd.DataFrame) -> bool:
    """Indica se a divisão espacial (coluna fold) está presente e sem valores nulos."""
    return "fold" in manifest.columns and not manifest["fold"].isna().any()


def patch_id(tile_id: str, row: int, col: int) -> str:
    """Identificador determinístico e único de um patch no grid do tile."""
    return f"{tile_id}_r{row:04d}_c{col:04d}"


def patch_bbox(transform: Any, row: int, col: int, patch_size: int) -> str:
    """Bounding box geográfica (minx,miny,maxx,maxy) de um patch no CRS do tile.

    Calculada diretamente da transformação afim (atributos a..f), sem depender
    de helpers de janela do rasterio — compatível com qualquer versão. ``row`` e
    ``col`` são índices do grid de patches; a origem em pixels é
    ``(col * patch_size, row * patch_size)``.
    """
    left_px = col * patch_size
    top_px = row * patch_size
    right_px = left_px + patch_size
    bottom_px = top_px + patch_size
    left = transform.c + transform.a * left_px + transform.b * top_px
    top = transform.f + transform.d * left_px + transform.e * top_px
    right = transform.c + transform.a * right_px + transform.b * bottom_px
    bottom = transform.f + transform.d * right_px + transform.e * bottom_px
    return f"{left:.2f},{bottom:.2f},{right:.2f},{top:.2f}"


def grid_dims(height: int, width: int, patch_size: int) -> tuple[int, int]:
    """Dimensões do grid de patches, descartando as bordas parciais."""
    return height // patch_size, width // patch_size


def _storage_root(storage_paths: dict[str, Path]) -> Path:
    """Raiz canônica tcc/, derivada dos caminhos de armazenamento resolvidos."""
    return storage_paths["data_processed"].parent.parent


def resolve_from_root(relative_path: str, storage_paths: dict[str, Path]) -> Path:
    """Resolve um path relativo (registrado no manifesto) para o caminho canônico."""
    return _storage_root(storage_paths) / relative_path


def _file_size(path: Path, storage_paths: dict[str, Path]) -> int | None:
    """Tamanho em bytes de uma entrada, consultado remotamente no Kaggle."""
    from src import io

    if io.detect_platform() == "kaggle":
        return io.get_drive_client().size(str(path.relative_to(io.mount_drive())))
    return path.stat().st_size if path.is_file() else None


def manifest_fingerprint(storage_paths: dict[str, Path], source_name: str) -> dict[str, Any]:
    """Fingerprint determinístico do manifesto da fonte (config + identidade das entradas).

    Permite decidir, sem reprocessar, se o manifesto e os patches persistidos
    ainda correspondem às entradas atuais (composite do ano da fonte e máscara
    da própria fonte).
    """
    config = get_config()
    year = reference_year(source_name)
    composite = composite_path(storage_paths, year)
    mask = source_mask_path(source_name, storage_paths)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "region_code": config["aoi"]["region_code"],
        "source_name": source_name,
        "year": year,
        "bands": list(config["data"]["bands"]),
        "patch_size": int(config["data"]["patch_size"]),
        "coffee_min_ratio": float(config["data"]["coffee_min_ratio"]),
        "min_valid_fraction": float(config["data"]["min_valid_fraction"]),
        "mask_source": source_name,
        "tile_id": composite_file_name(year),
        "composite": {"name": composite.name, "size": _file_size(composite, storage_paths)},
        "mask": {"name": mask.name, "size": _file_size(mask, storage_paths)},
    }


def _canonical_fingerprint_blob(storage_paths: dict[str, Path], source_name: str) -> str:
    """Blob canônico (ordenado) do fingerprint, base do hash de versionamento."""
    return json.dumps(
        manifest_fingerprint(storage_paths, source_name), sort_keys=True, ensure_ascii=False
    )


def patch_fingerprint_hash(storage_paths: dict[str, Path], source_name: str) -> str:
    """Hash estável do fingerprint; versiona a subpasta de patches persistidos."""
    blob = _canonical_fingerprint_blob(storage_paths, source_name)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def manifest_is_current(storage_paths: dict[str, Path], source_name: str) -> bool:
    """Indica se o manifesto persistido da fonte está vigente frente às entradas atuais."""
    from src import io

    if not io.path_exists(manifest_path(storage_paths, source_name)):
        return False
    if not io.path_exists(manifest_meta_path(storage_paths, source_name)):
        return False
    local_meta = io.ensure_local_copy(manifest_meta_path(storage_paths, source_name))
    try:
        stored = json.loads(local_meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if stored.get("bbox_schema_version") != BBOX_SCHEMA_VERSION:
        return False
    return stored.get("fingerprint_hash") == patch_fingerprint_hash(storage_paths, source_name)


def _require_dependency(label: str, path: Path, storage_paths: dict[str, Path]) -> None:
    """Falha com mensagem clara quando uma dependência de estágio anterior falta."""
    from src import io

    if not io.path_exists(path):
        raise FileNotFoundError(f"{label} não encontrado: {path}")


def _check_same_grid(composite: Any, mask: Any) -> None:
    """Valida que composite e máscara compartilham o mesmo grid (pixel a pixel)."""
    if not same_grid(composite, mask):
        raise ValueError("Composite e máscara final em grids distintos; reexecute o estágio 05.")


def _ensure_remote_dirs(dirs: list[Path]) -> None:
    """Garante a existência remota das pastas de destino no Kaggle."""
    from src import io

    if io.detect_platform() != "kaggle":
        return
    client = io.get_drive_client()
    drive_root = io.mount_drive()
    for dir_path in dirs:
        client.ensure_folder(str(dir_path.relative_to(drive_root)))


def _write_patch(target_path: Path, array: np.ndarray) -> None:
    """Persiste um patch localmente e no Drive canônico (upload no Kaggle)."""
    from src import io

    buffer = stdlib_io.BytesIO()
    np.save(buffer, array)
    io.persist_bytes(target_path, buffer.getvalue())


def _write_manifest(
    path: Path, records: list[dict[str, Any]], storage_paths: dict[str, Path]
) -> None:
    """Persiste o manifesto em Parquet no Drive canônico."""
    from src import io

    buffer = stdlib_io.BytesIO()
    pd.DataFrame(records).to_parquet(buffer, index=False)
    io.persist_bytes(path, buffer.getvalue())


def _write_metadata(path: Path, payload: dict[str, Any], storage_paths: dict[str, Path]) -> None:
    """Persiste o metadata (fingerprint) do manifesto no Drive canônico."""
    from src import io

    io.persist_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def generate_patches(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Garante os patches e o manifesto da fonte no caminho canônico (idempotente).

    Reutiliza o manifesto quando o fingerprint vigente coincide com o
    persistido; caso contrário, corta o composite do ano da fonte (estágio 05)
    e a máscara binária da própria fonte (estágio 02) em patches determinísticos,
    escrevendo apenas os que faltam, e registra tudo no manifesto Parquet.
    """
    from src import io

    manifest = manifest_path(storage_paths, source_name)
    if manifest_is_current(storage_paths, source_name):
        print(f"Manifesto {source_name} já existente e atual (reutilizado): {manifest}")
        return manifest

    year = reference_year(source_name)
    composite = composite_path(storage_paths, year)
    mask = source_mask_path(source_name, storage_paths)
    _require_dependency("Composite normalizado (estágio 05)", composite, storage_paths)
    _require_dependency(f"Máscara binária da fonte {source_name} (estágio 02)", mask, storage_paths)

    composite_local = io.ensure_local_copy(composite)
    mask_local = io.ensure_local_copy(mask)

    fingerprint_hash = patch_fingerprint_hash(storage_paths, source_name)
    image_dir = image_patches_dir(storage_paths, source_name, fingerprint_hash)
    mask_dir = mask_patches_dir(storage_paths, source_name, fingerprint_hash)
    _ensure_remote_dirs([image_dir, mask_dir])

    import rasterio
    from rasterio.windows import Window

    config = get_config()
    patch_size = int(config["data"]["patch_size"])
    coffee_min_ratio = float(config["data"]["coffee_min_ratio"])
    min_valid_fraction = float(config["data"]["min_valid_fraction"])
    tile_id = composite_file_name(year)
    mask_source = source_name
    root = _storage_root(storage_paths)

    records: list[dict[str, Any]] = []
    written_patches = 0
    reused_patches = 0
    discarded_invalid = 0
    with rasterio.open(composite_local) as comp, rasterio.open(mask_local) as m:
        _check_same_grid(comp, m)
        rows, cols = grid_dims(int(m.height), int(m.width), patch_size)
        if rows == 0 or cols == 0:
            raise ValueError(f"Raster menor que um patch ({patch_size}px); ajuste data.patch_size.")
        for row in range(rows):
            for col in range(cols):
                window = Window(col * patch_size, row * patch_size, patch_size, patch_size)
                image_win = comp.read(window=window)
                mask_win = m.read(1, window=window)
                # Pixels fora do AOI chegam como NaN em todas as bandas.
                valid_pixels = np.isfinite(image_win).all(axis=0)
                valid_fraction = float(np.mean(valid_pixels))
                # Patches sem nenhum pixel válido são sempre descartados.
                if valid_fraction <= 0.0 or valid_fraction < min_valid_fraction:
                    discarded_invalid += 1
                    continue
                # A proporção de café é medida apenas na área válida do patch.
                coffee_ratio = float(np.mean(mask_win[valid_pixels] > 0))
                if coffee_ratio < coffee_min_ratio:
                    continue
                pid = patch_id(tile_id, row, col)
                image_target = image_dir / f"{pid}.npy"
                mask_target = mask_dir / f"{pid}.npy"
                if io.path_exists(image_target) and io.path_exists(mask_target):
                    reused_patches += 1
                else:
                    _write_patch(image_target, image_win)
                    _write_patch(mask_target, mask_win)
                    written_patches += 1
                records.append(
                    {
                        "patch_id": pid,
                        "tile_id": tile_id,
                        "fold": None,
                        "row": row,
                        "col": col,
                        "bbox": patch_bbox(m.transform, row, col, patch_size),
                        "coffee_ratio": coffee_ratio,
                        "valid_fraction": valid_fraction,
                        "mask_source": mask_source,
                        "image_path": str(image_target.relative_to(root)),
                        "mask_path": str(mask_target.relative_to(root)),
                    }
                )

    _write_manifest(manifest, records, storage_paths)
    meta = {
        "fingerprint_hash": fingerprint_hash,
        "bbox_schema_version": BBOX_SCHEMA_VERSION,
        "grid": {"rows": rows, "cols": cols},
        "n_patches": len(records),
        "discarded_invalid": discarded_invalid,
        "min_valid_fraction": min_valid_fraction,
    }
    _write_metadata(manifest_meta_path(storage_paths, source_name), meta, storage_paths)
    print(
        f"Patches {source_name} gerados: {written_patches} novos, {reused_patches} reutilizados, "
        f"{discarded_invalid} descartados por pixels válidos < {min_valid_fraction:.0%}."
    )
    print(f"Manifesto salvo em: {manifest}")
    return manifest


def verify_manifest(storage_paths: dict[str, Path], source_name: str) -> dict[str, Any]:
    """Verifica o manifesto da fonte persistido: contagem, café e formato dos patches."""
    from src import io

    manifest = load_manifest(storage_paths, source_name)
    with_coffee = manifest["coffee_ratio"] > 0
    ratios = manifest["coffee_ratio"]
    valid = manifest.get("valid_fraction")
    first_image = io.ensure_local_copy(
        resolve_from_root(str(manifest.loc[0, "image_path"]), storage_paths)
    )
    return {
        "n_patches": int(len(manifest)),
        "n_with_coffee": int(with_coffee.sum()),
        "coffee_ratio": {
            "min": float(ratios.min()),
            "mean": float(ratios.mean()),
            "max": float(ratios.max()),
        },
        "valid_fraction": (
            {
                "min": float(valid.min()),
                "mean": float(valid.mean()),
                "max": float(valid.max()),
            }
            if valid is not None
            else None
        ),
        "patch_shape": list(np.load(first_image).shape),
        "mask_source": sorted(manifest["mask_source"].dropna().unique().tolist()),
        "tile_id": sorted(manifest["tile_id"].dropna().unique().tolist()),
    }


def patch_montage_file_name(source_name: str) -> str:
    """Nome estável da figura do mosaico de amostras da fonte, derivado da configuração."""
    config = get_config()
    return (
        f"patch_montage_{config['aoi']['region_code']}_{reference_year(source_name)}"
        f"_{source_name}.png"
    )


def save_patch_montage(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Persiste a figura do mosaico de amostras da fonte (regenerada a cada execução).

    A figura é derivada do manifesto vigente; regenerá-la evita manter amostras
    desatualizadas após mudanças no dataset do estágio 06.
    """
    from src import io

    figure_path = storage_paths["artifacts_figures"] / patch_montage_file_name(source_name)
    manifest = load_manifest(storage_paths, source_name)
    samples = manifest.sort_values("coffee_ratio", ascending=False).head(3)
    io.persist_bytes(figure_path, render_patch_montage(samples, storage_paths))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_patch_montage(manifest: pd.DataFrame, storage_paths: dict[str, Path]) -> bytes:
    """Renderiza o mosaico RGB + máscara das amostras mais cafeeiras."""
    from matplotlib.colors import ListedColormap

    from src import io
    from src.data.mask_comparison import COFFEE_COLORS, display_array
    from src.data.raster_utils import percentile_stretch, render_figure

    bands = get_config()["data"]["bands"]
    band_index = {name: i for i, name in enumerate(bands)}

    def _build(plt: Any) -> Any:
        figure, axes = plt.subplots(len(manifest), 2, figsize=(10, 5 * len(manifest)))
        axes = np.atleast_2d(axes)
        for row, (_, record) in enumerate(manifest.iterrows()):
            image = np.load(
                io.ensure_local_copy(resolve_from_root(str(record["image_path"]), storage_paths))
            )
            mask = (
                np.load(
                    io.ensure_local_copy(resolve_from_root(str(record["mask_path"]), storage_paths))
                )
                > 0
            )
            rgb = np.stack(
                [
                    percentile_stretch(image[band_index["B4"]]),
                    percentile_stretch(image[band_index["B3"]]),
                    percentile_stretch(image[band_index["B2"]]),
                ],
                axis=-1,
            )
            axes[row, 0].imshow(rgb)
            axes[row, 0].set_title(f"{record['patch_id']} — café {record['coffee_ratio']:.2%}")
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            axes[row, 1].imshow(
                display_array(mask),
                cmap=ListedColormap(COFFEE_COLORS),
                vmin=0,
                vmax=1,
            )
            axes[row, 1].set_title("Máscara")
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])
        figure.tight_layout()
        return figure

    return render_figure(_build)
