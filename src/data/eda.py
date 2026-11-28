"""Análise exploratória dos patches e estatísticas de normalização por fonte (estágio 08).

Calcula, para cada fonte habilitada, as estatísticas de normalização por banda
(média, desvio padrão e fração de NaN) sobre todos os pixels finitos dos patches
registrados no manifesto da fonte (estágio 06/07), persistindo-as em
data/processed/normalization_stats_<fonte>.json com fingerprint de versão —
idempotente, pois reexecuções reutilizam o arquivo vigente sem reprocessar.
Executa também verificações de sanidade do dataset de cada fonte (formas, tipos,
máscaras binárias, distribuição de café, dobras e amostra de arquivos) e gera a
figura de resumo por fonte em artifacts/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import figstyle
from src.config import get_config
from src.data.mask_utils import reference_year
from src.data.patch_generation import (
    load_manifest,
    patch_fingerprint_hash,
    require_manifest,
    resolve_from_root,
)

EDA_SCHEMA_VERSION = 1
SANITY_SAMPLE_SIZE = 10  # tamanho da amostra determinística das verificações
HIST_BINS = 32  # número de bins do histograma de reflectância da figura


def normalization_stats_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho canônico das estatísticas de normalização da fonte (data/processed/)."""
    return storage_paths["data_processed"] / f"normalization_stats_{source_name}.json"


def eda_figure_file_name(source_name: str) -> str:
    """Nome estável da figura de resumo da EDA da fonte, derivado da configuração."""
    config = get_config()
    return (
        f"eda_summary_{config['aoi']['region_code']}_{reference_year(source_name)}"
        f"_{source_name}.png"
    )


def _stats_fingerprint(storage_paths: dict[str, Path], source_name: str) -> dict[str, Any]:
    """Fingerprint das estatísticas da fonte (schema, bandas e hash dos patches)."""
    config = get_config()
    return {
        "schema_version": EDA_SCHEMA_VERSION,
        "source_name": source_name,
        "bands": list(config["data"]["bands"]),
        "fingerprint_hash": patch_fingerprint_hash(storage_paths, source_name),
    }


def _deterministic_sample(manifest: pd.DataFrame, size: int) -> pd.DataFrame:
    """Amostra determinística de patches para as verificações e a figura."""
    return manifest.sample(n=min(size, len(manifest)), random_state=0)


def _finite_or_none(value: float) -> float | None:
    """Converte o valor em float, ou None quando não é finito (ex.: banda toda NaN)."""
    return float(value) if np.isfinite(value) else None


def normalization_stats_is_current(storage_paths: dict[str, Path], source_name: str) -> bool:
    """Indica se as estatísticas persistidas da fonte estão vigentes frente aos patches."""
    from src import io

    stats_path = normalization_stats_path(storage_paths, source_name)
    if not io.path_exists(stats_path):
        return False
    local_stats = io.ensure_local_copy(stats_path)
    try:
        stored = json.loads(local_stats.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return stored.get("fingerprint") == _stats_fingerprint(storage_paths, source_name)


def compute_normalization_stats(
    storage_paths: dict[str, Path],
    source_name: str,
) -> dict[str, Any]:
    """Garante as estatísticas de normalização por banda da fonte (idempotente).

    Reutiliza o arquivo persistido quando o fingerprint vigente coincide com o
    armazenado; caso contrário, percorre todos os patches do manifesto da fonte e
    acumula soma, soma dos quadrados, contagem de pixels finitos e de NaN por
    banda, derivando média, desvio padrão e fração de NaN.
    """
    from src import io

    stats_path = normalization_stats_path(storage_paths, source_name)
    if normalization_stats_is_current(storage_paths, source_name):
        print(f"Estatísticas {source_name} já existentes e atuais (reutilizadas): {stats_path}")
        local_stats = io.ensure_local_copy(stats_path)
        return json.loads(local_stats.read_text(encoding="utf-8"))

    require_manifest(storage_paths, source_name)
    manifest = load_manifest(storage_paths, source_name)
    config = get_config()
    bands = list(config["data"]["bands"])
    n_bands = len(bands)

    sums = np.zeros(n_bands, dtype=np.float64)
    sums_sq = np.zeros(n_bands, dtype=np.float64)
    counts = np.zeros(n_bands, dtype=np.float64)
    nan_counts = np.zeros(n_bands, dtype=np.float64)
    total_counts = np.zeros(n_bands, dtype=np.float64)

    for record in manifest.itertuples():
        image = np.load(
            io.ensure_local_copy(resolve_from_root(str(record.image_path), storage_paths))
        ).reshape(n_bands, -1)
        finite = np.isfinite(image)
        total_counts += image.shape[1]
        nan_counts += (~finite).sum(axis=1)
        values = np.where(finite, image, 0.0)
        sums += values.sum(axis=1)
        sums_sq += (values * values).sum(axis=1)
        counts += finite.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
        variances = sums_sq / counts - means**2
    stds = np.sqrt(np.maximum(variances, 0.0))

    per_band: dict[str, dict[str, float | None]] = {}
    for index, band in enumerate(bands):
        per_band[band] = {
            "mean": _finite_or_none(float(means[index])),
            "std": _finite_or_none(float(stds[index])),
            "nan_fraction": float(nan_counts[index] / total_counts[index]),
        }

    payload = {
        "schema_version": EDA_SCHEMA_VERSION,
        "source_name": source_name,
        "fingerprint": _stats_fingerprint(storage_paths, source_name),
        "bands": bands,
        "n_patches": int(len(manifest)),
        "per_band": per_band,
    }
    io.persist_bytes(
        stats_path,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    print(f"Estatísticas de normalização salvas em: {stats_path}")
    return payload


def run_sanity_checks(
    storage_paths: dict[str, Path],
    source_name: str,
) -> dict[str, Any]:
    """Executa verificações de sanidade do dataset da fonte a partir do manifesto.

    Valida o manifesto (contagens, café, dobras, unicidade) e uma amostra
    determinística de patches (forma esperada, dtype, máscara binária, intervalo
    de valores e fração de NaN), acumulando uma lista de problemas encontrados.
    """
    from src import io

    require_manifest(storage_paths, source_name)
    manifest = load_manifest(storage_paths, source_name)
    config = get_config()
    bands = list(config["data"]["bands"])
    patch_size = int(config["data"]["patch_size"])
    expected_shape = (len(bands), patch_size, patch_size)

    sample = _deterministic_sample(manifest, SANITY_SAMPLE_SIZE)
    shape_issues: list[str] = []
    dtypes: set[str] = set()
    mask_values: set[int] = set()
    value_min = float("inf")
    value_max = float("-inf")
    nan_pixels = 0
    for record in sample.itertuples():
        image = np.load(
            io.ensure_local_copy(resolve_from_root(str(record.image_path), storage_paths))
        )
        mask = np.load(
            io.ensure_local_copy(resolve_from_root(str(record.mask_path), storage_paths))
        )
        if image.shape != expected_shape:
            shape_issues.append(f"{record.patch_id}: {image.shape}")
        dtypes.add(str(image.dtype))
        mask_values.update(np.unique(mask).astype(int).tolist())
        finite = image[np.isfinite(image)]
        if finite.size:
            value_min = min(value_min, float(finite.min()))
            value_max = max(value_max, float(finite.max()))
        nan_pixels += int(np.isnan(image).sum())

    coffee = manifest["coffee_ratio"]
    fold_present = "fold" in manifest.columns
    issues: list[str] = []
    fold_distribution: dict[str, int] = {}
    if fold_present:
        if manifest["fold"].isna().any():
            issues.append("coluna fold com valores ausentes")
        for fold, count in manifest["fold"].value_counts().sort_index().items():
            fold_distribution[str(int(fold))] = int(count)
    if shape_issues:
        issues.append(f"formas divergentes do esperado: {shape_issues}")
    if not mask_values.issubset({0, 1}):
        issues.append(f"máscara com valores não binários: {sorted(mask_values)}")

    sample_size = int(len(sample))
    sample_total_pixels = sample_size * len(bands) * patch_size**2

    return {
        "n_patches": int(len(manifest)),
        "n_tiles": int(manifest["tile_id"].nunique()),
        "unique_patch_ids": int(manifest["patch_id"].nunique()),
        "mask_sources": sorted(manifest["mask_source"].dropna().unique().tolist()),
        "tile_ids": sorted(manifest["tile_id"].dropna().unique().tolist()),
        "coffee_ratio": {
            "min": float(coffee.min()),
            "mean": float(coffee.mean()),
            "max": float(coffee.max()),
        },
        "fold": {
            "present": fold_present,
            "distribution": fold_distribution,
        },
        "sample": {
            "size": sample_size,
            "expected_shape": list(expected_shape),
            "shape_issues": shape_issues,
            "dtypes": sorted(dtypes),
            "mask_values": sorted(mask_values),
            "value_range": [
                _finite_or_none(value_min),
                _finite_or_none(value_max),
            ],
            "nan_fraction": float(nan_pixels / sample_total_pixels),
        },
        "issues": issues,
    }


BAND_LABELS = {"B2": "B2 (azul)", "B3": "B3 (verde)", "B4": "B4 (vermelho)", "B8": "B8 (NIR)"}
BAND_COLORS = {"B2": "#0072b2", "B3": "#009e73", "B4": "#d55e00", "B8": "#7b1fa2"}


def save_eda_figure(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Persiste a figura de resumo da EDA da fonte em PNG e PDF (regenerada a cada execução).

    A figura é derivada do manifesto e das estatísticas vigentes; regenerá-la
    garante que reexecuções após mudanças no dataset não mantenham figuras
    desatualizadas.
    """
    figure_path = storage_paths["artifacts_figures"] / eda_figure_file_name(source_name)
    manifest = load_manifest(storage_paths, source_name)
    stats = compute_normalization_stats(storage_paths, source_name)
    figstyle.persist_figure(figure_path, render_eda_figure(manifest, stats, storage_paths))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_eda_figure(
    manifest: pd.DataFrame,
    stats: dict[str, Any],
    storage_paths: dict[str, Path],
) -> dict[str, bytes]:
    """Resumo: proporção de café, balanço das dobras, estatísticas e histogramas por banda."""
    from src import io

    bands = list(get_config()["data"]["bands"])
    sample = _deterministic_sample(manifest, SANITY_SAMPLE_SIZE)
    per_band = stats["per_band"]

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(2, 2)
        axes[0, 0].hist(manifest["coffee_ratio"], bins=20, color="#7b1fa2", alpha=0.85)
        axes[0, 0].set_xlabel("Proporção de café por recorte")
        axes[0, 0].set_ylabel("Recortes")
        axes[0, 0].set_title(f"Densidade de café (n = {len(manifest)} recortes)")
        figstyle.panel_label(axes[0, 0], "(a)")

        if "fold" in manifest.columns and manifest["fold"].notna().any():
            counts = manifest["fold"].value_counts().sort_index()
            bars = axes[0, 1].bar(
                [str(int(fold)) for fold in counts.index],
                counts.values,
                color=[
                    figstyle.FOLD_COLORS[int(f) % len(figstyle.FOLD_COLORS)] for f in counts.index
                ],
            )
            for bar in bars:
                axes[0, 1].annotate(
                    f"{int(bar.get_height())}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    xytext=(0, 1.5),
                    textcoords="offset points",
                )
            axes[0, 1].set_xlabel("Dobra de validação")
            axes[0, 1].set_ylabel("Recortes")
            axes[0, 1].set_title("Recortes por dobra espacial")
        else:
            axes[0, 1].text(0.5, 0.5, "Coluna de dobra ausente", ha="center", va="center")
            axes[0, 1].set_axis_off()
        figstyle.panel_label(axes[0, 1], "(b)")

        band_names = list(per_band)
        x = np.arange(len(band_names))
        means = [per_band[band]["mean"] or 0.0 for band in band_names]
        stds = [per_band[band]["std"] or 0.0 for band in band_names]
        axes[1, 0].bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=[BAND_COLORS.get(band, "#555555") for band in band_names],
            alpha=0.9,
        )
        axes[1, 0].set_xticks(x, [BAND_LABELS.get(band, band) for band in band_names])
        axes[1, 0].set_ylabel("Reflectância (média ± desvio)")
        axes[1, 0].set_title("Estatísticas por banda")
        figstyle.panel_label(axes[1, 0], "(c)")

        for index, band in enumerate(bands):
            values: list[np.ndarray] = []
            for record in sample.itertuples():
                image = np.load(
                    io.ensure_local_copy(resolve_from_root(str(record.image_path), storage_paths))
                )
                band_values = image[index]
                values.append(band_values[np.isfinite(band_values)])
            if values:
                axes[1, 1].hist(
                    np.concatenate(values),
                    bins=HIST_BINS,
                    histtype="step",
                    linewidth=1.4,
                    density=True,
                    color=BAND_COLORS.get(band, "#555555"),
                    label=BAND_LABELS.get(band, band),
                )
        axes[1, 1].legend(loc="upper right")
        axes[1, 1].set_xlabel("Reflectância")
        axes[1, 1].set_ylabel("Densidade")
        axes[1, 1].set_title("Distribuição por banda (amostra)")
        figstyle.panel_label(axes[1, 1], "(d)")

    return figstyle.render_publication(_build, height_cm=13.5)
