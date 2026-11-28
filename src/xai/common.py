"""Utilitários compartilhados dos estágios de explicabilidade (13/14).

Centraliza a seleção determinística de patches representativos, o carregamento
de imagem/máscara normalizadas de um patch e a composição RGB usada pelos mapas
de calor. Mantém Grad-CAM (U-Net) e Attention Rollout (SegFormer) alinhados ao
mesmo protocolo de amostragem e de visualização.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src import figstyle
from src.config import get_config
from src.data.dataset import load_normalization_stats, normalize_patch
from src.data.patch_generation import load_manifest, resolve_from_root
from src.data.raster_utils import rgb_composite


@dataclass(frozen=True)
class PatchSample:
    """Patch carregado para explicabilidade: bruto, normalizado e referência."""

    patch_id: str
    fold: int
    image: np.ndarray
    mask: np.ndarray
    valid: np.ndarray
    normalized: torch.Tensor
    reference: np.ndarray


def xai_root(storage_paths: dict[str, Path]) -> Path:
    """Raiz de XAI: chave dedicada quando presente, senão derivada de artifacts/."""
    if "artifacts_xai" in storage_paths:
        return storage_paths["artifacts_xai"]
    return storage_paths["artifacts_metrics"].parent / "xai"


def weights_by_fold(summary: dict[str, Any]) -> dict[int, str]:
    """Hash dos pesos de cada dobra registrado no resumo do estágio 11."""
    return {int(item["fold"]): str(item["weights_sha256"]) for item in summary["per_fold"]}


def artifacts_are_current(
    metadata_path: Path,
    artifact_paths: tuple[Path, ...],
    fingerprint: dict[str, Any],
) -> bool:
    """Indica se metadata e artefatos existem e correspondem ao fingerprint vigente."""
    from src import io

    if not all(io.path_exists(path) for path in (metadata_path, *artifact_paths)):
        return False
    try:
        stored = json.loads(io.ensure_local_copy(metadata_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return stored.get("fingerprint") == fingerprint


def select_representative_patches(
    patch_metrics: pd.DataFrame, max_patches: int = 4
) -> list[dict[str, Any]]:
    """Seleciona patches representativos por IoU e proporção de café (determinístico).

    Prioriza patches com café e escolhe, sem repetição, o maior e o menor IoU, a
    maior proporção de café e o IoU mais próximo da mediana. Empates são
    desfeitos pelo `patch_id`, garantindo a mesma amostra em reexecuções.
    """
    required = {"patch_id", "fold", "iou", "coffee_ratio"}
    missing = required.difference(patch_metrics.columns)
    if missing:
        raise ValueError(f"Colunas ausentes nas métricas por patch: {sorted(missing)}")
    frame = patch_metrics.copy()
    if "truth_positive" in frame.columns:
        pool = frame[frame["truth_positive"]]
    elif {"tp", "fn"}.issubset(frame.columns):
        pool = frame[(frame["tp"].astype(int) + frame["fn"].astype(int)) > 0]
    else:
        pool = frame[frame["coffee_ratio"] > 0]
    if pool.empty:
        pool = frame
    if pool.empty:
        return []
    median = float(pool["iou"].median())
    frame["iou_distance"] = (frame["iou"].astype(float) - median).abs()
    pool = frame.loc[pool.index]
    candidates = (
        ("Maior IoU", "iou", False),
        ("Menor IoU", "iou", True),
        ("Maior proporção de café", "coffee_ratio", False),
        ("IoU mediano", "iou_distance", True),
    )
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for label, column, ascending in candidates:
        if len(selected) >= max_patches:
            break
        ordered = pool.sort_values([column, "patch_id"], ascending=[ascending, True])
        for row in ordered.itertuples(index=False):
            patch = str(row.patch_id)
            if patch in used:
                continue
            used.add(patch)
            selected.append(
                {
                    "patch_id": patch,
                    "label": label,
                    "fold": int(row.fold),
                    "iou": float(row.iou),
                    "coffee_ratio": float(row.coffee_ratio),
                }
            )
            break
    return selected


def load_patch_sample(
    storage_paths: dict[str, Path],
    manifest: pd.DataFrame,
    patch_id: str,
    stats: dict[str, Any],
) -> PatchSample:
    """Carrega imagem e máscara de um patch, valida pixels e normaliza a entrada."""
    from src import io

    indexed = manifest.set_index("patch_id", drop=False)
    if patch_id not in indexed.index:
        raise KeyError(f"Patch {patch_id} ausente no manifesto.")
    record = indexed.loc[patch_id]
    image = np.load(
        io.ensure_local_copy(resolve_from_root(str(record["image_path"]), storage_paths))
    )
    mask = np.load(io.ensure_local_copy(resolve_from_root(str(record["mask_path"]), storage_paths)))
    valid = np.isfinite(image).all(axis=0)
    normalized = torch.from_numpy(normalize_patch(image, stats))
    reference = (mask > 0) & valid
    return PatchSample(
        patch_id=patch_id,
        fold=int(record["fold"]),
        image=np.asarray(image),
        mask=np.asarray(mask),
        valid=valid,
        normalized=normalized,
        reference=reference,
    )


def patch_rgb(sample: PatchSample) -> np.ndarray:
    """Compõe o RGB (B4/B3/B2) do patch com fundo preto fora do AOI."""
    return rgb_composite(sample.image, sample.valid, list(get_config()["data"]["bands"]))


def load_stats_and_manifest(
    storage_paths: dict[str, Path], source_name: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Carrega estatísticas de normalização e manifesto da fonte (estágios 06/08)."""
    return load_normalization_stats(storage_paths, source_name), load_manifest(
        storage_paths, source_name
    )


def render_heatmap_figure(
    samples: list[Any], heatmaps: list[np.ndarray], method: str
) -> dict[str, bytes]:
    """Por recorte: imagem RGB, referência, mapa de relevância e sobreposição.

    ``samples`` expõe `image_rgb`, `reference`, `label`, `fold` e `iou`;
    ``heatmaps`` traz os mapas normalizados em [0, 1] na mesma ordem.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize

    if not samples:
        raise ValueError(f"Nenhuma amostra disponível para a figura {method}.")
    reference_cmap = ListedColormap(["#f5f5f5", "#2e7d32"])
    columns = ["Imagem (RGB)", "Referência (café)", method, "Sobreposição"]

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(len(samples), 4, squeeze=False)
        for row, (sample, heatmap) in enumerate(zip(samples, heatmaps, strict=True)):
            axes[row, 0].imshow(sample.image_rgb)
            axes[row, 1].imshow(
                sample.reference.astype(np.uint8), cmap=reference_cmap, vmin=0, vmax=1
            )
            axes[row, 2].imshow(heatmap, cmap="turbo", vmin=0.0, vmax=1.0)
            axes[row, 3].imshow(sample.image_rgb)
            axes[row, 3].imshow(heatmap, cmap="turbo", alpha=0.45, vmin=0.0, vmax=1.0)
            axes[row, 0].set_ylabel(
                f"{figstyle.wrap_label(sample.label)}\n(dobra {sample.fold}, IoU {sample.iou:.2f})",
                fontsize=6.5,
                labelpad=4,
            )
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])
                axis.grid(False)
                for spine in axis.spines.values():
                    spine.set_visible(False)
        for column, title in enumerate(columns):
            axes[0, column].set_title(title)
        mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="turbo")
        mappable.set_array(np.asarray([]))
        figure.colorbar(
            mappable,
            ax=axes[:, 2:].ravel().tolist(),
            orientation="horizontal",
            location="bottom",
            shrink=0.5,
            aspect=40,
            label=f"Relevância normalizada ({method}, apenas pixels válidos)",
        )

    return figstyle.render_publication(_build, height_cm=3.9 * len(samples) + 2.2)
