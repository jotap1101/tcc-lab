"""Divisão espacial k-fold dos patches por fonte (estágio 07).

Atribui cada patch registrado no manifesto de uma fonte do estágio 06 a uma das
k dobras (`splits.fold_count`) por proximidade geográfica: os centroides das
bboxes são agrupados com k-means determinístico (implementado em numpy puro,
k-means++ com semente fixa) e os grupos são rotulados de forma determinística
pela posição dos centros. A contiguidade espacial dos grupos reduz o vazamento
por autocorrelação espacial entre treino e validação nos estágios 09/10. A
divisão é gravada na coluna `fold` do manifesto de cada fonte e versionada em
split_<fonte>.meta.json (fingerprint do manifesto + dobras + semente); execuções
repetidas reutilizam a divisão vigente sem reprocessamento. A implementação em
numpy puro evita dependência do scikit-learn (cuja importação de numpy.testing
é sensível a versões do numpy).
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
from src.data.mask_utils import reference_year
from src.data.patch_generation import (
    BBOX_SCHEMA_VERSION,
    load_manifest,
    manifest_path,
    patch_fingerprint_hash,
    require_manifest,
)

SPLIT_SCHEMA_VERSION = 2


def split_meta_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho do metadata da divisão da fonte (fingerprint para idempotência)."""
    return storage_paths["data_processed"] / f"split_{source_name}.meta.json"


def fold_count() -> int:
    """Número de dobras da validação cruzada, definido em config.yaml."""
    return int(get_config()["splits"]["fold_count"])


def split_seed() -> int:
    """Semente da divisão espacial, herdada da reprodutibilidade global."""
    return int(get_config()["reproducibility"]["seed"])


def bbox_centroid(bbox: str) -> tuple[float, float]:
    """Centroide (x, y) de uma bbox registrada como 'minx,miny,maxx,maxy'."""
    min_x, min_y, max_x, max_y = (float(value) for value in bbox.split(","))
    return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0


def _kmeans_plus_plus_init(centroids: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Inicializa os centros do k-means com o algoritmo k-means++ (determinístico).

    O primeiro centro é sorteado uniformemente e cada centro seguinte é sorteado
    com probabilidade proporcional à distância ao quadrado até o centro mais
    próximo, dispersando os centros iniciais sobre o domínio.
    """
    centers = [centroids[int(rng.integers(len(centroids)))]]
    for _ in range(1, k):
        distances = np.min(
            np.linalg.norm(centroids[:, None, :] - np.asarray(centers)[None, :, :], axis=2) ** 2,
            axis=1,
        )
        total = float(distances.sum())
        # Fallback uniforme quando as distâncias somam zero (pontos coincidentes).
        if not np.isfinite(total) or total <= 0.0:
            pick = int(rng.integers(len(centroids)))
        else:
            pick = int(rng.choice(len(centroids), p=distances / total))
        centers.append(centroids[pick])
    return np.asarray(centers, dtype=np.float64)


def spatial_fold_centroids(centroids: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Agrupa os centroides em k regiões espaciais e rotula de forma determinística.

    Aplica k-means determinístico em numpy puro (k-means++ com semente fixa)
    sobre as coordenadas dos centroides e reordena os rótulos pela posição dos
    centros (x, depois y), tornando o número da dobra estável e independente da
    ordem dos patches no manifesto.
    """
    if len(centroids) < k:
        raise ValueError(
            f"Patches insuficientes para {k} dobras ({len(centroids)}); reduza splits.fold_count."
        )
    rng = np.random.default_rng(seed)
    points = centroids.astype(np.float64, copy=False)
    centers = _kmeans_plus_plus_init(points, k, rng)
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(100):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centers = np.asarray(
            [
                points[new_labels == cluster].mean(axis=0)
                if (new_labels == cluster).any()
                else centers[cluster]
                for cluster in range(k)
            ],
            dtype=np.float64,
        )
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            centers, labels = new_centers, new_labels
            break
        centers, labels = new_centers, new_labels
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    relabel = {old: new for new, old in enumerate(order)}
    return np.array([relabel[int(label)] for label in labels])


def split_meta_payload(storage_paths: dict[str, Path], source_name: str) -> dict[str, Any]:
    """Payload do metadata da divisão da fonte (fingerprint + dobras + semente)."""
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "source_name": source_name,
        "manifest_fingerprint_hash": patch_fingerprint_hash(storage_paths, source_name),
        "bbox_schema_version": BBOX_SCHEMA_VERSION,
        "fold_count": fold_count(),
        "seed": split_seed(),
    }


def split_is_current(storage_paths: dict[str, Path], source_name: str) -> bool:
    """Indica se a divisão persistida da fonte está vigente frente às entradas atuais."""
    from src import io

    if not io.path_exists(manifest_path(storage_paths, source_name)):
        return False
    if not io.path_exists(split_meta_path(storage_paths, source_name)):
        return False
    local_meta = io.ensure_local_copy(split_meta_path(storage_paths, source_name))
    try:
        stored = json.loads(local_meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    current = split_meta_payload(storage_paths, source_name)
    if any(stored.get(key) != value for key, value in current.items()):
        return False
    return bool(load_manifest(storage_paths, source_name)["fold"].notna().all())


def split_fingerprint_hash(storage_paths: dict[str, Path], source_name: str) -> str:
    """Hash estável da atribuição patch→dobra (detecta mudanças no split).

    Usado pelo trainer para não reutilizar pesos de uma divisão anterior quando
    apenas a atribuição espacial mudou (ex.: correção do cálculo de `bbox`).
    """
    manifest = load_manifest(storage_paths, source_name)
    if not manifest["fold"].notna().all():
        raise ValueError(f"Divisão {source_name} incompleta; execute o estágio 07 antes.")
    pairs = sorted(
        (str(patch_id), int(fold))
        for patch_id, fold in zip(manifest["patch_id"], manifest["fold"], strict=True)
    )
    blob = "\n".join(f"{patch_id}:{fold}" for patch_id, fold in pairs)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _write_manifest(path: Path, manifest: pd.DataFrame, storage_paths: dict[str, Path]) -> None:
    """Persiste o manifesto (com a coluna fold) no Drive canônico."""
    from src import io

    buffer = stdlib_io.BytesIO()
    manifest.to_parquet(buffer, index=False)
    io.persist_bytes(path, buffer.getvalue())


def _write_meta(path: Path, payload: dict[str, Any], storage_paths: dict[str, Path]) -> None:
    """Persiste o metadata da divisão no Drive canônico."""
    from src import io

    io.persist_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def assign_spatial_folds(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Garante a divisão espacial k-fold no manifesto da fonte (idempotente).

    Reutiliza a divisão quando o metadata persistido coincide com o atual
    (fingerprint do manifesto, dobras e semente) e a coluna `fold` está preenchida;
    caso contrário, agrupa os patches por proximidade geográfica e grava a coluna
    `fold` no manifesto do estágio 06.
    """
    manifest_file = manifest_path(storage_paths, source_name)
    if split_is_current(storage_paths, source_name):
        print(f"Divisão {source_name} já existente e atual (reutilizada): {manifest_file}")
        return manifest_file

    require_manifest(storage_paths, source_name)
    manifest = load_manifest(storage_paths, source_name)
    k = fold_count()
    seed = split_seed()
    centroids = np.array([bbox_centroid(value) for value in manifest["bbox"]])
    manifest["fold"] = spatial_fold_centroids(centroids, k, seed)
    _write_manifest(manifest_file, manifest, storage_paths)
    _write_meta(
        split_meta_path(storage_paths, source_name),
        split_meta_payload(storage_paths, source_name),
        storage_paths,
    )
    print(f"Divisão espacial k-fold salva em: {manifest_file}")
    return manifest_file


def verify_split(storage_paths: dict[str, Path], source_name: str) -> dict[str, Any]:
    """Verifica a divisão da fonte: contagens, café e localização média por dobra."""
    manifest = load_manifest(storage_paths, source_name)
    k = fold_count()
    per_fold = []
    for fold in range(k):
        group = manifest[manifest["fold"] == fold]
        centroids = np.array([bbox_centroid(value) for value in group["bbox"]])
        per_fold.append(
            {
                "fold": fold,
                "n_patches": int(len(group)),
                "coffee_ratio_mean": float(group["coffee_ratio"].mean()),
                "centroid_x": float(centroids[:, 0].mean()),
                "centroid_y": float(centroids[:, 1].mean()),
            }
        )
    return {
        "fold_count": k,
        "n_patches": int(len(manifest)),
        "n_tiles": int(manifest["tile_id"].nunique()),
        "per_fold": per_fold,
    }


def split_figure_file_name(source_name: str) -> str:
    """Nome estável da figura da divisão espacial da fonte, derivado da configuração."""
    config = get_config()
    return (
        f"split_kfold_{config['aoi']['region_code']}_{reference_year(source_name)}"
        f"_{source_name}.png"
    )


def save_split_figure(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Persiste a figura da divisão espacial da fonte (regenerada a cada execução).

    A figura é derivada do manifesto vigente; regenerá-la evita manter uma
    divisão desatualizada após mudanças no dataset do estágio 06.
    """
    from src import io

    figure_path = storage_paths["artifacts_figures"] / split_figure_file_name(source_name)
    manifest = load_manifest(storage_paths, source_name)
    io.persist_bytes(figure_path, render_split_figure(manifest))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_split_figure(manifest: pd.DataFrame) -> bytes:
    """Renderiza o mapa de centroides dos patches coloridos por dobra."""
    from src.data.raster_utils import render_figure

    centroids = np.array([bbox_centroid(value) for value in manifest["bbox"]])
    folds = manifest["fold"].astype(int)

    def _build(plt: Any) -> Any:
        figure, axis = plt.subplots(figsize=(8, 8))
        scatter = axis.scatter(
            centroids[:, 0], centroids[:, 1], c=folds, cmap="tab10", s=12, alpha=0.8
        )
        axis.set_title(f"Divisão espacial k-fold (k={int(folds.max()) + 1})")
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")
        figure.colorbar(scatter, ax=axis, label="Dobra")
        figure.tight_layout()
        return figure

    return render_figure(_build)
