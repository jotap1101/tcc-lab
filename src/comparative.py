"""Comparação estatística e mapas de erro entre U-Net e SegFormer (estágio 12).

Consolida os artefatos persistidos pelo estágio 11 (resumos por modelo/fonte,
métricas por patch e predições binárias) em uma comparação emparelhada por patch.
Para cada métrica de pixel (IoU, F1, Precisão, Recall), reporta a diferença
SegFormer − U-Net, intervalo de confiança bootstrap, teste t pareado, teste de
Wilcoxon dos postos sinalizados, correção de Holm para as quatro métricas,
tamanho de efeito (d_z de Cohen e delta de Cliff) e a contagem de patches em que
cada modelo vence. Também gera a figura comparativa e os mapas de erro
(TP/FP/FN/ignorado) de patches representativos, persistindo tudo sob a raiz
canônica tcc/ do Drive. A análise é reutilizada apenas enquanto os fingerprints
do estágio 11 permanecem vigentes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src import evaluation
from src.config import get_config
from src.data.mask_utils import reference_year
from src.data.patch_generation import (
    load_manifest,
    patch_fingerprint_hash,
    resolve_from_root,
)
from src.data.raster_utils import percentile_stretch
from src.data.source_decision import chosen_source
from src.data.spatial_split import fold_count, split_fingerprint_hash
from src.evaluation import METRIC_KEYS, MODEL_NAMES
from src.trainer import config_hash

COMPARATIVE_SCHEMA_VERSION = 1
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CONFIDENCE = 0.95
COMPARATIVE_DIR_NAME = "comparative"
MODEL_LABELS = {"unet": "U-Net", "segformer": "SegFormer"}
DIRECTION = "segformer_minus_unet"

TN_CODE, FP_CODE, FN_CODE, TP_CODE, INVALID_CODE = 0, 1, 2, 3, 4
ERROR_COLORS = ("#d9d9d9", "#e53935", "#1e88e5", "#2e7d32", "#ffffff")
ERROR_LABELS = (
    "Verdadeiro negativo",
    "Falso positivo",
    "Falso negativo",
    "Verdadeiro positivo",
    "Ignorado (fora do AOI)",
)


@dataclass(frozen=True)
class ComparativeAnalysis:
    """Resultado da comparação entre os dois modelos na fonte avaliada."""

    source_name: str
    patch_metrics: pd.DataFrame
    payload: dict[str, Any]
    summary_path: Path
    skipped: bool


@dataclass(frozen=True)
class ErrorMapSample:
    """Amostra de patch para o mapa de erro comparativo."""

    patch_id: str
    label: str
    fold: int
    image_rgb: np.ndarray
    truth: np.ndarray
    valid: np.ndarray
    predictions: dict[str, np.ndarray]
    iou: dict[str, float]


def comparative_dir(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Diretório canônico da comparação entre modelos para a fonte."""
    return storage_paths["artifacts_metrics"] / COMPARATIVE_DIR_NAME / source_name


def comparison_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho do JSON consolidado da comparação estatística."""
    return comparative_dir(storage_paths, source_name) / "comparison.json"


def comparison_figure_file_name(source_name: str) -> str:
    """Nome estável da figura comparativa do par de modelos."""
    config = get_config()
    return (
        f"comparison_{config['aoi']['region_code']}_{reference_year(source_name)}_{source_name}.png"
    )


def comparison_figure_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho da figura comparativa no diretório canônico de figuras."""
    return storage_paths["artifacts_figures"] / comparison_figure_file_name(source_name)


def error_maps_figure_file_name(source_name: str) -> str:
    """Nome estável da figura de mapas de erro do par de modelos."""
    config = get_config()
    return (
        f"error_maps_{config['aoi']['region_code']}_{reference_year(source_name)}_{source_name}.png"
    )


def error_maps_figure_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho da figura de mapas de erro no diretório canônico de figuras."""
    return storage_paths["artifacts_figures"] / error_maps_figure_file_name(source_name)


def _expected_summary_fingerprint(
    storage_paths: dict[str, Path], source_name: str
) -> dict[str, Any]:
    """Fingerprint que o estágio 11 deveria ter registrado para a fonte vigente."""
    return {
        "manifest_fingerprint_hash": patch_fingerprint_hash(storage_paths, source_name),
        "split_fingerprint_hash": split_fingerprint_hash(storage_paths, source_name),
        "config_hash": config_hash(),
        "threshold": evaluation.DEFAULT_THRESHOLD,
    }


def load_evaluation_summaries(
    storage_paths: dict[str, Path], source_name: str
) -> dict[str, dict[str, Any]]:
    """Carrega os resumos do estágio 11 dos dois modelos (falha se ausentes)."""
    return {
        model_name: evaluation.load_evaluation_summary(storage_paths, model_name, source_name)
        for model_name in MODEL_NAMES
    }


def summaries_are_current(
    storage_paths: dict[str, Path],
    source_name: str,
    summaries: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Indica se os resumos do estágio 11 correspondem ao dataset/config vigentes."""
    from src import io

    if summaries is None:
        try:
            summaries = load_evaluation_summaries(storage_paths, source_name)
        except FileNotFoundError:
            return False
    expected = _expected_summary_fingerprint(storage_paths, source_name)
    expected_folds = list(range(fold_count()))
    for model_name in MODEL_NAMES:
        summary = summaries.get(model_name)
        if summary is None or summary.get("source_name") != source_name:
            return False
        if summary.get("fingerprint") != expected:
            return False
        folds = [int(item["fold"]) for item in summary.get("per_fold", [])]
        if folds != expected_folds:
            return False
        for fold in expected_folds:
            if not io.path_exists(
                evaluation.patch_metrics_path(storage_paths, model_name, source_name, fold)
            ):
                return False
            if not io.path_exists(
                evaluation.predictions_path(storage_paths, model_name, source_name, fold)
            ):
                return False
    return True


def comparison_fingerprint(
    summaries: dict[str, dict[str, Any]], source_name: str
) -> dict[str, Any]:
    """Fingerprint da comparação: fingerprints e pesos de cada dobra do estágio 11."""
    return {
        "schema_version": COMPARATIVE_SCHEMA_VERSION,
        "source_name": source_name,
        "threshold": float(summaries["unet"]["threshold"]),
        "models": {
            model_name: {
                "evaluation_schema_version": summaries[model_name].get("schema_version"),
                "fingerprint": summaries[model_name]["fingerprint"],
                "weights_sha256": [
                    item["weights_sha256"] for item in summaries[model_name]["per_fold"]
                ],
            }
            for model_name in MODEL_NAMES
        },
    }


def _comparison_is_current(path: Path, fingerprint: dict[str, Any]) -> bool:
    """Indica se o JSON persistido existe e corresponde ao fingerprint vigente."""
    from src import io

    if not io.path_exists(path):
        return False
    try:
        stored = json.loads(io.ensure_local_copy(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return stored.get("fingerprint") == fingerprint


def _load_model_patch_metrics(
    storage_paths: dict[str, Path],
    model_name: str,
    source_name: str,
    summary: dict[str, Any],
) -> pd.DataFrame:
    """Concatena as métricas por patch das dobras de um modelo."""
    frames = [
        evaluation.load_patch_metrics(storage_paths, model_name, source_name, int(item["fold"]))
        for item in summary["per_fold"]
    ]
    frame = pd.concat(frames, ignore_index=True)
    if frame["patch_id"].duplicated().any():
        raise ValueError(f"Métricas por patch duplicadas para {model_name} na fonte {source_name}.")
    return frame


def _merge_patch_metrics(
    unet: pd.DataFrame, segformer: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    """Emparelha as métricas por patch dos dois modelos validando a consistência."""
    if set(unet["patch_id"]) != set(segformer["patch_id"]):
        raise ValueError(
            f"Conjuntos de patches divergentes entre U-Net e SegFormer na fonte {source_name}; "
            "reexecute o estágio 11 para os dois modelos."
        )
    merged = unet.merge(
        segformer, on="patch_id", suffixes=("_unet", "_segformer"), validate="one_to_one"
    )
    if len(merged) != len(unet):
        raise ValueError("Emparelhamento de patches incompleto entre os modelos.")
    folds_match = merged["fold_unet"].astype(int).to_numpy() == merged["fold_segformer"].astype(int)
    if not folds_match.all():
        raise ValueError("Dobras divergentes entre U-Net e SegFormer para os mesmos patches.")
    coffee_match = np.allclose(
        merged["coffee_ratio_unet"].to_numpy(dtype=float),
        merged["coffee_ratio_segformer"].to_numpy(dtype=float),
    )
    if not coffee_match:
        raise ValueError("Proporção de café divergente entre os modelos para os mesmos patches.")
    truth_unet = (merged["tp_unet"] + merged["fn_unet"]).to_numpy()
    truth_segformer = (merged["tp_segformer"] + merged["fn_segformer"]).to_numpy()
    if not np.array_equal(truth_unet, truth_segformer):
        raise ValueError("Máscaras de referência divergentes entre os modelos.")
    merged["fold"] = merged["fold_unet"].astype(int)
    merged["coffee_ratio"] = merged["coffee_ratio_unet"].astype(float)
    if "valid_fraction_unet" in merged.columns:
        merged["valid_fraction"] = merged["valid_fraction_unet"]
    merged["truth_positive"] = truth_unet > 0
    merged = merged.drop(
        columns=[
            "fold_unet",
            "fold_segformer",
            "coffee_ratio_unet",
            "coffee_ratio_segformer",
            "valid_fraction_unet",
            "valid_fraction_segformer",
        ],
        errors="ignore",
    )
    return merged.sort_values("patch_id").reset_index(drop=True)


def load_aligned_patch_metrics(
    storage_paths: dict[str, Path],
    source_name: str,
    summaries: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Carrega e emparelha as métricas por patch dos dois modelos pelo `patch_id`."""
    unet = _load_model_patch_metrics(storage_paths, "unet", source_name, summaries["unet"])
    segformer = _load_model_patch_metrics(
        storage_paths, "segformer", source_name, summaries["segformer"]
    )
    return _merge_patch_metrics(unet, segformer, source_name)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """Intervalo de confiança bootstrap (percentil) da média de uma amostra."""
    sample = np.asarray(values, dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    total = int(sample.size)
    if total == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    block = 256
    for start in range(0, iterations, block):
        size = min(block, iterations - start)
        indices = rng.integers(0, total, size=(size, total))
        means[start : start + size] = sample[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def cohens_dz(differences: np.ndarray) -> float | None:
    """Tamanho de efeito d_z de Cohen para amostras emparelhadas."""
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    deviation = float(np.std(values, ddof=1))
    if deviation <= 1e-12:
        return None
    return float(np.mean(values) / deviation)


def cliffs_delta(segformer: np.ndarray, unet: np.ndarray) -> float | None:
    """Delta de Cliff: P(SegFormer > U-Net) − P(SegFormer < U-Net).

    Considera todas as comparações entre patches dos dois modelos (não apenas
    os pares posicionais), conforme a definição ordinal do delta de Cliff; a
    diferença emparelhada por `patch_id` é avaliada pelos testes t e Wilcoxon.
    """
    seg, base = _finite_pairs(segformer, unet)
    if seg.size == 0 or base.size == 0:
        return None
    greater = 0
    less = 0
    block = 512
    for start in range(0, int(seg.size), block):
        chunk = seg[start : start + block, None]
        greater += int(np.count_nonzero(chunk > base[None, :]))
        less += int(np.count_nonzero(chunk < base[None, :]))
    return float(greater - less) / float(seg.size * base.size)


def holm_bonferroni(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    """Correção de Holm–Bonferroni, monotônica e limitada a 1.0."""
    finite = {key: value for key, value in pvalues.items() if value is not None}
    ordered = sorted(finite.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[key] = running
    return {
        key: (adjusted.get(key) if value is not None else None) for key, value in pvalues.items()
    }


def _finite_pairs(segformer: np.ndarray, unet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mantém apenas os pares com as duas métricas finitas."""
    seg = np.asarray(segformer, dtype=np.float64)
    base = np.asarray(unet, dtype=np.float64)
    mask = np.isfinite(seg) & np.isfinite(base)
    return seg[mask], base[mask]


def _optional_float(value: Any) -> float | None:
    """Converte em float finito ou None (para serialização JSON estável)."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def paired_metric_statistics(
    segformer: np.ndarray,
    unet: np.ndarray,
    metric: str,
    seed: int,
) -> dict[str, Any]:
    """Estatísticas emparelhadas de uma métrica (diferença SegFormer − U-Net)."""
    seg, base = _finite_pairs(segformer, unet)
    count = int(seg.size)
    result: dict[str, Any] = {
        "metric": metric,
        "n_pairs": count,
        "mean_difference": None,
        "median_difference": None,
        "ci95_low": None,
        "ci95_high": None,
        "t_statistic": None,
        "t_pvalue": None,
        "wilcoxon_statistic": None,
        "wilcoxon_pvalue": None,
        "p_value_holm": None,
        "cohens_dz": None,
        "cliffs_delta": None,
        "segformer_better": 0,
        "ties": 0,
        "unet_better": 0,
    }
    if count == 0:
        return result
    differences = seg - base
    result["mean_difference"] = float(np.mean(differences))
    result["median_difference"] = float(np.median(differences))
    result["segformer_better"] = int(np.count_nonzero(differences > 1e-12))
    result["unet_better"] = int(np.count_nonzero(differences < -1e-12))
    result["ties"] = count - result["segformer_better"] - result["unet_better"]
    low, high = bootstrap_mean_ci(differences, seed=seed)
    result["ci95_low"] = low
    result["ci95_high"] = high
    result["cohens_dz"] = cohens_dz(differences)
    result["cliffs_delta"] = cliffs_delta(seg, base)
    if count >= 2 and not np.allclose(differences, 0.0, atol=1e-12):
        t_test = scipy_stats.ttest_rel(seg, base)
        result["t_statistic"] = _optional_float(t_test.statistic)
        result["t_pvalue"] = _optional_float(t_test.pvalue)
        try:
            wilcoxon = scipy_stats.wilcoxon(differences, zero_method="wilcox")
            result["wilcoxon_statistic"] = _optional_float(wilcoxon.statistic)
            result["wilcoxon_pvalue"] = _optional_float(wilcoxon.pvalue)
        except ValueError:
            result["wilcoxon_statistic"] = None
            result["wilcoxon_pvalue"] = None
    return result


def paired_comparison(patch_metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Executa a comparação emparelhada das quatro métricas, com Holm aplicado."""
    base_seed = int(get_config()["reproducibility"]["seed"])
    results: dict[str, dict[str, Any]] = {}
    for index, metric in enumerate(METRIC_KEYS):
        results[metric] = paired_metric_statistics(
            patch_metrics[f"{metric}_segformer"].to_numpy(),
            patch_metrics[f"{metric}_unet"].to_numpy(),
            metric,
            seed=base_seed + index,
        )
    adjusted = holm_bonferroni(
        {metric: result["wilcoxon_pvalue"] for metric, result in results.items()}
    )
    for metric, result in results.items():
        result["p_value_holm"] = adjusted[metric]
    return results


def aggregate_delta(summaries: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Diferença das métricas agregadas *micro* (SegFormer − U-Net)."""
    unet = summaries["unet"]["aggregate"]["metrics"]
    segformer = summaries["segformer"]["aggregate"]["metrics"]
    return {metric: float(segformer[metric]) - float(unet[metric]) for metric in METRIC_KEYS}


def per_fold_delta(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Diferença das métricas por dobra (SegFormer − U-Net), ordenada pela dobra."""
    folds: dict[int, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        for item in summaries[model_name]["per_fold"]:
            fold = int(item["fold"])
            record = folds.setdefault(
                fold, {"fold": fold, "n_patches": int(item["n_patches"]), "metrics": {}}
            )
            if record["n_patches"] != int(item["n_patches"]):
                raise ValueError(f"Número de patches divergente entre modelos na dobra {fold}.")
            record["metrics"][model_name] = item["metrics"]
    return [
        {
            "fold": fold,
            "n_patches": record["n_patches"],
            "delta": {
                metric: float(record["metrics"]["segformer"][metric])
                - float(record["metrics"]["unet"][metric])
                for metric in METRIC_KEYS
            },
        }
        for fold, record in sorted(folds.items())
    ]


def build_comparison_payload(
    patch_metrics: pd.DataFrame,
    summaries: dict[str, dict[str, Any]],
    source_name: str,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    """Monta o payload consolidado da comparação entre os dois modelos."""
    models = {
        model_name: {
            "aggregate": summaries[model_name]["aggregate"],
            "fold_mean": summaries[model_name]["fold_mean"],
            "fold_std": summaries[model_name]["fold_std"],
            "n_patches": int(summaries[model_name]["n_patches"]),
            "per_fold": [
                {
                    "fold": int(item["fold"]),
                    "n_patches": int(item["n_patches"]),
                    "metrics": item["metrics"],
                }
                for item in summaries[model_name]["per_fold"]
            ],
        }
        for model_name in MODEL_NAMES
    }
    return {
        "schema_version": COMPARATIVE_SCHEMA_VERSION,
        "source_name": source_name,
        "threshold": float(summaries["unet"]["threshold"]),
        "direction": DIRECTION,
        "n_folds": int(summaries["unet"]["n_folds"]),
        "n_patches": int(len(patch_metrics)),
        "models": models,
        "aggregate_delta": aggregate_delta(summaries),
        "per_fold_delta": per_fold_delta(summaries),
        "paired": paired_comparison(patch_metrics),
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "confidence": BOOTSTRAP_CONFIDENCE,
        },
        "fingerprint": fingerprint,
    }


def analyze_comparison(
    storage_paths: dict[str, Path],
    source_name: str | None = None,
    force: bool = False,
) -> ComparativeAnalysis:
    """Compara U-Net e SegFormer na fonte escolhida (idempotente)."""
    from src import io

    source = source_name or chosen_source()
    summaries = load_evaluation_summaries(storage_paths, source)
    if not summaries_are_current(storage_paths, source, summaries):
        raise ValueError(
            "Artefatos do estágio 11 desatualizados ou incompletos para esta fonte; "
            "reexecute a avaliação antes da comparação."
        )
    fingerprint = comparison_fingerprint(summaries, source)
    target = comparison_path(storage_paths, source)
    patch_metrics = load_aligned_patch_metrics(storage_paths, source, summaries)
    if not force and _comparison_is_current(target, fingerprint):
        payload = json.loads(io.ensure_local_copy(target).read_text(encoding="utf-8"))
        print(
            f"Comparação já vigente (artefatos reutilizados): "
            f"IoU Δ {payload['aggregate_delta']['iou']:+.4f} (SegFormer − U-Net)"
        )
        return ComparativeAnalysis(
            source_name=source,
            patch_metrics=patch_metrics,
            payload=payload,
            summary_path=target,
            skipped=True,
        )
    payload = build_comparison_payload(patch_metrics, summaries, source, fingerprint)
    io.persist_json(target, payload)
    print(
        f"Comparação concluída: {payload['n_patches']} patches | "
        f"IoU Δ {payload['aggregate_delta']['iou']:+.4f} (SegFormer − U-Net)"
    )
    return ComparativeAnalysis(
        source_name=source,
        patch_metrics=patch_metrics,
        payload=payload,
        summary_path=target,
        skipped=False,
    )


def render_comparison_figure(payload: dict[str, Any], patch_metrics: pd.DataFrame) -> bytes:
    """Renderiza distribuições, dobras, dispersão e histograma da diferença de IoU."""
    from src.data.raster_utils import render_figure

    per_fold = payload["models"]["unet"]["per_fold"]
    coffee = patch_metrics[patch_metrics["truth_positive"]]
    if coffee.empty:
        coffee = patch_metrics

    def _build(plt: Any) -> Any:
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))

        values = [coffee[f"iou_{model_name}"].to_numpy() for model_name in MODEL_NAMES]
        axes[0, 0].boxplot(values)
        axes[0, 0].set_xticks([1, 2], [MODEL_LABELS[model_name] for model_name in MODEL_NAMES])
        axes[0, 0].set_title("IoU por patch (patches com café)")
        axes[0, 0].set_ylabel("IoU")
        axes[0, 0].set_ylim(-0.02, 1.02)

        folds = [str(int(item["fold"])) for item in per_fold]
        positions = np.arange(len(folds))
        width = 0.38
        for offset, model_name in zip((-width / 2, width / 2), MODEL_NAMES, strict=True):
            values = [item["metrics"]["iou"] for item in per_fold]
            axes[0, 1].bar(
                positions + offset,
                values,
                width=width,
                label=MODEL_LABELS[model_name],
            )
        axes[0, 1].set_xticks(positions, folds)
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[0, 1].set_title("IoU agregado por dobra")
        axes[0, 1].set_xlabel("dobra")
        axes[0, 1].set_ylabel("IoU")
        axes[0, 1].legend()

        scatter = axes[1, 0].scatter(
            coffee["iou_unet"],
            coffee["iou_segformer"],
            c=coffee["coffee_ratio"],
            cmap="viridis",
            s=16,
            alpha=0.8,
        )
        axes[1, 0].plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1)
        axes[1, 0].set_xlim(-0.02, 1.02)
        axes[1, 0].set_ylim(-0.02, 1.02)
        axes[1, 0].set_xlabel("IoU U-Net")
        axes[1, 0].set_ylabel("IoU SegFormer")
        axes[1, 0].set_title("IoU por patch: U-Net vs SegFormer")
        figure.colorbar(scatter, ax=axes[1, 0], label="coffee_ratio")

        difference = (coffee["iou_segformer"] - coffee["iou_unet"]).to_numpy()
        axes[1, 1].hist(difference, bins=30, color="#6a1b9a", alpha=0.8)
        axes[1, 1].axvline(0.0, color="black", linewidth=1)
        axes[1, 1].axvline(
            float(np.mean(difference)), color="#e53935", linewidth=1.5, label="média"
        )
        axes[1, 1].set_title("Distribuição da diferença de IoU (SegFormer − U-Net)")
        axes[1, 1].set_xlabel("diferença de IoU")
        axes[1, 1].set_ylabel("patches")
        axes[1, 1].legend()

        figure.suptitle("Comparação U-Net vs SegFormer (validação espacial)", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        return figure

    return render_figure(_build)


def save_comparison_figure(
    storage_paths: dict[str, Path],
    source_name: str,
    payload: dict[str, Any],
    patch_metrics: pd.DataFrame,
) -> Path:
    """Persiste a figura comparativa (regenerada a cada execução)."""
    from src import io

    figure_path = comparison_figure_path(storage_paths, source_name)
    io.persist_bytes(figure_path, render_comparison_figure(payload, patch_metrics))
    print(f"Figura comparativa salva em: {figure_path}")
    return figure_path


def select_error_map_patches(
    patch_metrics: pd.DataFrame, max_patches: int = 4
) -> list[dict[str, Any]]:
    """Seleciona patches representativos para os mapas de erro (determinístico).

    Prioriza patches com café e escolhe, sem repetição: maior e menor diferença de
    IoU entre os modelos, maior e menor IoU médio. Empates são desfeitos pelo
    `patch_id`, garantindo a mesma seleção em reexecuções.
    """
    frame = patch_metrics.copy()
    frame["mean_iou"] = frame[["iou_unet", "iou_segformer"]].mean(axis=1)
    frame["iou_difference"] = frame["iou_segformer"] - frame["iou_unet"]
    pool = frame[frame["truth_positive"]] if "truth_positive" in frame else frame
    if pool.empty:
        pool = frame
    if pool.empty:
        return []
    candidates = (
        ("Maior diferença (SegFormer − U-Net)", "iou_difference", False),
        ("Menor diferença (SegFormer − U-Net)", "iou_difference", True),
        ("Maior IoU médio", "mean_iou", False),
        ("Menor IoU médio", "mean_iou", True),
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
                    "iou_unet": float(row.iou_unet),
                    "iou_segformer": float(row.iou_segformer),
                    "iou_difference": float(row.iou_difference),
                }
            )
            break
    return selected


def error_code_map(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Codifica TP/FP/FN/TN/ignorado (3/1/2/0/4) para o mapa de erro."""
    pred = np.asarray(prediction, dtype=bool) & valid
    reference = np.asarray(truth, dtype=bool) & valid
    codes = np.full(valid.shape, INVALID_CODE, dtype=np.uint8)
    codes[valid & ~pred & ~reference] = TN_CODE
    codes[valid & pred & ~reference] = FP_CODE
    codes[valid & ~pred & reference] = FN_CODE
    codes[valid & pred & reference] = TP_CODE
    return codes


def _image_rgb(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compõe RGB (B4, B3, B2) com estiramento de percentis e fundo preto inválido."""
    bands = list(get_config()["data"]["bands"])
    indices = [bands.index(band) for band in ("B4", "B3", "B2") if band in bands]
    if len(indices) != 3:
        raise ValueError(f"Bandas insuficientes para o RGB: {bands}")
    rgb = np.dstack([percentile_stretch(image[index]) for index in indices])
    rgb[~valid] = 0
    return rgb


def build_error_map_samples(
    storage_paths: dict[str, Path],
    source_name: str,
    selected: list[dict[str, Any]],
) -> list[ErrorMapSample]:
    """Carrega imagem, referência e predições dos patches selecionados."""
    from src import io

    if not selected:
        raise ValueError("Nenhum patch selecionado para os mapas de erro.")
    manifest = load_manifest(storage_paths, source_name).set_index("patch_id", drop=False)
    folds = sorted({int(item["fold"]) for item in selected})
    predictions = {
        model_name: {
            fold: evaluation.load_predictions(storage_paths, model_name, source_name, fold)
            for fold in folds
        }
        for model_name in MODEL_NAMES
    }
    samples: list[ErrorMapSample] = []
    for item in selected:
        patch = str(item["patch_id"])
        fold = int(item["fold"])
        if patch not in manifest.index:
            raise KeyError(f"Patch {patch} ausente no manifesto da fonte {source_name}.")
        record = manifest.loc[patch]
        image = np.load(
            io.ensure_local_copy(resolve_from_root(str(record["image_path"]), storage_paths))
        )
        mask = np.load(
            io.ensure_local_copy(resolve_from_root(str(record["mask_path"]), storage_paths))
        )
        valid = np.isfinite(image).all(axis=0)
        truth = (mask > 0) & valid
        model_predictions: dict[str, np.ndarray] = {}
        for model_name in MODEL_NAMES:
            prediction = predictions[model_name][fold].get(patch)
            if prediction is None:
                raise KeyError(f"Predição ausente de {model_name} para o patch {patch}.")
            binary = np.asarray(prediction, dtype=bool)
            if binary.shape != truth.shape:
                raise ValueError(f"Forma de predição divergente para o patch {patch}.")
            model_predictions[model_name] = binary & valid
        samples.append(
            ErrorMapSample(
                patch_id=patch,
                label=str(item["label"]),
                fold=fold,
                image_rgb=_image_rgb(image, valid),
                truth=truth,
                valid=valid,
                predictions=model_predictions,
                iou={model_name: float(item[f"iou_{model_name}"]) for model_name in MODEL_NAMES},
            )
        )
    return samples


def render_error_maps_figure(samples: list[ErrorMapSample]) -> bytes:
    """Renderiza, por patch, RGB, referência e mapas de erro dos dois modelos."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    from src.data.raster_utils import render_figure

    if not samples:
        raise ValueError("Nenhuma amostra disponível para os mapas de erro.")
    error_cmap = ListedColormap(ERROR_COLORS)
    error_norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), error_cmap.N)
    truth_cmap = ListedColormap(["#f5f5f5", "#2e7d32", "#000000"])

    def _build(plt: Any) -> Any:
        figure, axes = plt.subplots(
            len(samples), 4, figsize=(16, 4.2 * len(samples)), squeeze=False
        )
        for row, sample in enumerate(samples):
            truth_code = np.where(sample.valid, sample.truth.astype(np.uint8), 2)
            axes[row, 0].imshow(sample.image_rgb)
            axes[row, 0].set_title(
                f"{sample.label}\n{sample.patch_id} (dobra {sample.fold})", fontsize=9
            )
            axes[row, 1].imshow(truth_code, cmap=truth_cmap, vmin=0, vmax=2)
            axes[row, 1].set_title("Referência", fontsize=9)
            for column, model_name in enumerate(MODEL_NAMES, start=2):
                codes = error_code_map(sample.predictions[model_name], sample.truth, sample.valid)
                axes[row, column].imshow(
                    codes, cmap=error_cmap, norm=error_norm, interpolation="nearest"
                )
                axes[row, column].set_title(
                    f"Erro {MODEL_LABELS[model_name]} (IoU {sample.iou[model_name]:.3f})",
                    fontsize=9,
                )
            for axis in axes[row]:
                axis.axis("off")
        handles = [
            Patch(facecolor=color, edgecolor="black", label=label)
            for color, label in zip(ERROR_COLORS, ERROR_LABELS, strict=True)
        ]
        figure.legend(handles=handles, loc="lower center", ncol=5, fontsize=9)
        figure.suptitle("Mapas de erro: U-Net vs SegFormer", fontsize=13)
        figure.tight_layout(rect=(0, 0.04, 1, 0.97))
        return figure

    return render_figure(_build)


def save_error_maps_figure(
    storage_paths: dict[str, Path],
    source_name: str,
    selected: list[dict[str, Any]],
) -> Path:
    """Persiste a figura de mapas de erro (regenerada a cada execução)."""
    from src import io

    samples = build_error_map_samples(storage_paths, source_name, selected)
    figure_path = error_maps_figure_path(storage_paths, source_name)
    io.persist_bytes(figure_path, render_error_maps_figure(samples))
    print(f"Mapas de erro salvos em: {figure_path}")
    return figure_path
