"""Comparação estatística e mapas de erro entre U-Net e SegFormer (estágio 12).

Consolida os artefatos persistidos pelo estágio 11 (resumos por modelo/fonte,
métricas por patch e predições binárias) em uma comparação emparelhada por patch.
Para cada métrica de pixel (IoU, F1, Precisão, Recall), reporta a diferença
SegFormer − U-Net, intervalo de confiança bootstrap, teste t pareado, teste de
Wilcoxon dos postos sinalizados, correção de Holm para as quatro métricas,
tamanho de efeito (d_z de Cohen e delta de Cliff) e a contagem de patches em que
cada modelo vence. Como os patches são espacialmente dependentes, esses testes
por patch são **descritivos**; a inferência adequada usa blocos espaciais e
dobras como unidades (`src.spatial_inference`). Uma análise de sensibilidade ao
limiar (precisão média e limiar calibrado leave-one-fold-out, `src.thresholds`)
verifica se a conclusão depende da calibração de cada modelo. Também gera a
figura comparativa e os mapas de erro (TP/FP/FN/ignorado) de patches
representativos, persistindo tudo sob a raiz canônica tcc/ do Drive. A análise é
reutilizada apenas enquanto os fingerprints do estágio 11 permanecem vigentes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src import evaluation, figstyle, spatial_inference, thresholds
from src.config import get_config
from src.data.mask_utils import reference_year
from src.data.patch_generation import load_manifest, resolve_from_root
from src.data.raster_utils import rgb_composite
from src.data.source_decision import chosen_source
from src.data.spatial_split import fold_count
from src.evaluation import METRIC_KEYS, MODEL_NAMES

COMPARATIVE_SCHEMA_VERSION = 2
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CONFIDENCE = 0.95
COMPARATIVE_DIR_NAME = "comparative"
MODEL_LABELS = figstyle.MODEL_LABELS
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
    expected = evaluation.summary_fingerprint(storage_paths, source_name)
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
            if not io.path_exists(
                evaluation.probabilities_path(storage_paths, model_name, source_name, fold)
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
    unet = evaluation.load_model_patch_metrics(storage_paths, "unet", source_name)
    segformer = evaluation.load_model_patch_metrics(storage_paths, "segformer", source_name)
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


def _model_threshold_analysis(
    storage_paths: dict[str, Path], model_name: str, source_name: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Precisão média, curva P×R e métricas sob limiar fixo, calibrado e ótimo (um modelo)."""
    positive: dict[int, np.ndarray] = {}
    negative: dict[int, np.ndarray] = {}
    patches: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for fold in range(fold_count()):
        histograms = evaluation.load_probability_histograms(
            storage_paths, model_name, source_name, fold
        )
        patches[fold] = (
            histograms["patch_ids"],
            histograms["positive"],
            histograms["negative"],
        )
        positive[fold] = histograms["positive"].sum(axis=0)
        negative[fold] = histograms["negative"].sum(axis=0)
    pooled_positive = np.sum(list(positive.values()), axis=0)
    pooled_negative = np.sum(list(negative.values()), axis=0)

    def _metrics(tp: Any, fp: Any, fn: Any) -> dict[str, float]:
        values = thresholds.metrics_from_counts(np.asarray(tp), np.asarray(fp), np.asarray(fn))
        return {key: float(values[key]) for key in METRIC_KEYS}

    fixed_bin = thresholds.DEFAULT_THRESHOLD_BIN
    fixed_counts = thresholds.counts_per_patch_at_bin(
        pooled_positive[None, :], pooled_negative[None, :], fixed_bin
    )
    oracle_bin = thresholds.best_threshold_bin(pooled_positive, pooled_negative, "iou")
    oracle_counts = thresholds.counts_per_patch_at_bin(
        pooled_positive[None, :], pooled_negative[None, :], oracle_bin
    )
    chosen = thresholds.leave_one_fold_out_bins(positive, negative, "iou")

    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in range(fold_count()):
        ids, pos, neg = patches[fold]
        counts = thresholds.counts_per_patch_at_bin(pos, neg, chosen[fold])
        patch_values = thresholds.metrics_from_counts(counts["tp"], counts["fp"], counts["fn"])
        for index, patch_id in enumerate(ids):
            rows.append(
                {
                    "patch_id": str(patch_id),
                    "fold": fold,
                    **{
                        f"{key}_{model_name}": float(patch_values[key][index])
                        for key in METRIC_KEYS
                    },
                }
            )
        fold_rows.append(
            {
                "fold": fold,
                "threshold": thresholds.threshold_value(chosen[fold]),
                "metrics": _metrics(counts["tp"].sum(), counts["fp"].sum(), counts["fn"].sum()),
            }
        )
    calibrated_totals = {
        key: sum(
            int(
                thresholds.counts_per_patch_at_bin(
                    patches[fold][1], patches[fold][2], chosen[fold]
                )[key].sum()
            )
            for fold in range(fold_count())
        )
        for key in ("tp", "fp", "fn")
    }
    analysis = {
        "average_precision": thresholds.average_precision(pooled_positive, pooled_negative),
        "average_precision_per_fold": {
            str(fold): thresholds.average_precision(positive[fold], negative[fold])
            for fold in range(fold_count())
        },
        "pr_curve": thresholds.precision_recall_curve(pooled_positive, pooled_negative),
        "fixed": {
            "threshold": thresholds.threshold_value(fixed_bin),
            "metrics": _metrics(
                fixed_counts["tp"].sum(), fixed_counts["fp"].sum(), fixed_counts["fn"].sum()
            ),
        },
        "oracle": {
            "threshold": thresholds.threshold_value(oracle_bin),
            "metrics": _metrics(
                oracle_counts["tp"].sum(), oracle_counts["fp"].sum(), oracle_counts["fn"].sum()
            ),
        },
        "calibrated": {
            "criterion": "iou",
            "scheme": "leave_one_fold_out",
            "aggregate": _metrics(
                calibrated_totals["tp"], calibrated_totals["fp"], calibrated_totals["fn"]
            ),
            "per_fold": fold_rows,
            "fold_mean": {
                key: float(np.mean([item["metrics"][key] for item in fold_rows]))
                for key in METRIC_KEYS
            },
            "fold_std": {
                key: float(np.std([item["metrics"][key] for item in fold_rows]))
                for key in METRIC_KEYS
            },
        },
    }
    return analysis, pd.DataFrame(rows)


def threshold_sensitivity(
    storage_paths: dict[str, Path], source_name: str, seed: int
) -> dict[str, Any]:
    """Compara os modelos sem depender do limiar fixo de 0,5 (AP e limiar calibrado).

    O limiar de cada dobra é escolhido apenas nas outras dobras (sem vazamento);
    as diferenças por patch sob esses limiares alimentam a mesma inferência por
    blocos e dobras da comparação principal.
    """
    analyses: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []
    for model_name in MODEL_NAMES:
        analyses[model_name], frame = _model_threshold_analysis(
            storage_paths, model_name, source_name
        )
        frames.append(frame)
    merged = frames[0].merge(frames[1], on=["patch_id", "fold"], validate="one_to_one")
    calibrated = analyses["segformer"]["calibrated"]["aggregate"]
    baseline = analyses["unet"]["calibrated"]["aggregate"]
    return {
        "bins": thresholds.PROBABILITY_BINS,
        "models": analyses,
        "average_precision_delta": float(
            analyses["segformer"]["average_precision"] - analyses["unet"]["average_precision"]
        ),
        "calibrated_delta": {
            metric: float(calibrated[metric] - baseline[metric]) for metric in METRIC_KEYS
        },
        "inference_calibrated": spatial_inference.spatial_inference(merged, METRIC_KEYS, seed=seed),
    }


def build_comparison_payload(
    patch_metrics: pd.DataFrame,
    summaries: dict[str, dict[str, Any]],
    source_name: str,
    fingerprint: dict[str, Any],
    storage_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Monta o payload consolidado da comparação entre os dois modelos."""
    base_seed = int(get_config()["reproducibility"]["seed"])
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
        "spatial_inference": spatial_inference.spatial_inference(
            patch_metrics, METRIC_KEYS, seed=base_seed
        ),
        "threshold_sensitivity": (
            threshold_sensitivity(storage_paths, source_name, base_seed)
            if storage_paths is not None
            else None
        ),
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
    payload = build_comparison_payload(
        patch_metrics, summaries, source, fingerprint, storage_paths=storage_paths
    )
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


def _jitter(count: int, seed: int, width: float = 0.12) -> np.ndarray:
    """Deslocamento horizontal determinístico para pontos sobrepostos."""
    return np.random.default_rng(seed).uniform(-width, width, size=count)


def render_comparison_figure(
    payload: dict[str, Any], patch_metrics: pd.DataFrame
) -> dict[str, bytes]:
    """Distribuição do IoU, IoU por dobra, dispersão pareada e diferença com IC por blocos."""
    coffee = patch_metrics[patch_metrics["truth_positive"]]
    if coffee.empty:
        coffee = patch_metrics
    block = payload["spatial_inference"]["block_level"]["iou"]

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(2, 2)

        # (a) distribuição do IoU por patch, com a mediana e todos os pontos
        axis = axes[0, 0]
        for position, model_name in enumerate(MODEL_NAMES, start=1):
            values = coffee[f"iou_{model_name}"].to_numpy()
            color = figstyle.MODEL_COLORS[model_name]
            axis.boxplot(
                values,
                positions=[position],
                widths=0.5,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.4},
                boxprops={"color": color},
                whiskerprops={"color": color},
                capprops={"color": color},
            )
            axis.scatter(
                position + _jitter(len(values), position, 0.18),
                values,
                s=7,
                color=color,
                alpha=0.45,
                linewidths=0,
                zorder=3,
            )
        axis.set_xticks([1, 2], [MODEL_LABELS[name] for name in MODEL_NAMES])
        axis.set_ylabel("IoU por recorte")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(f"IoU por recorte (n = {len(coffee)})")
        figstyle.panel_label(axis, "(a)")

        # (b) IoU agregado por dobra, cada modelo com os próprios valores
        axis = axes[0, 1]
        folds = [int(item["fold"]) for item in payload["models"]["unet"]["per_fold"]]
        positions = np.arange(len(folds))
        width = 0.38
        for offset, model_name in zip((-width / 2, width / 2), MODEL_NAMES, strict=True):
            per_fold = payload["models"][model_name]["per_fold"]
            axis.bar(
                positions + offset,
                [item["metrics"]["iou"] for item in per_fold],
                width=width,
                color=figstyle.MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )
        axis.set_xticks(positions, [str(fold) for fold in folds])
        axis.set_xlabel("Dobra de validação")
        axis.set_ylabel("IoU agregado da dobra")
        axis.set_ylim(0.0, 1.18)
        axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.set_title("IoU por dobra espacial")
        axis.legend(loc="upper center", ncol=2)
        figstyle.panel_label(axis, "(b)")

        # (c) dispersão pareada, colorida pela proporção de café
        axis = axes[1, 0]
        scatter = axis.scatter(
            coffee["iou_unet"],
            coffee["iou_segformer"],
            c=coffee["coffee_ratio"],
            cmap="viridis",
            s=11,
            alpha=0.85,
            linewidths=0,
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("IoU da U-Net")
        axis.set_ylabel("IoU do SegFormer")
        axis.set_title("Pares de recortes")
        axis.text(
            0.05, 0.93, "SegFormer melhor", fontsize=7, color="gray", transform=axis.transAxes
        )
        axis.text(0.55, 0.05, "U-Net melhor", fontsize=7, color="gray", transform=axis.transAxes)
        figure.colorbar(scatter, ax=axis, shrink=0.8, label="Proporção de café")
        figstyle.panel_label(axis, "(c)")

        # (d) diferença de IoU com a média e o IC 95% por bootstrap de blocos
        axis = axes[1, 1]
        difference = (coffee["iou_segformer"] - coffee["iou_unet"]).to_numpy()
        axis.hist(difference, bins=25, color=figstyle.MODEL_COLORS["segformer"], alpha=0.8)
        axis.axvline(0.0, color="black", linewidth=1)
        mean = float(block["mean_difference"])
        axis.axvline(mean, color=figstyle.COLOR_ACCENT, linewidth=1.5, label=f"Média {mean:+.3f}")
        if block["ci95_low"] is not None:
            axis.axvspan(
                float(block["ci95_low"]),
                float(block["ci95_high"]),
                color=figstyle.COLOR_ACCENT,
                alpha=0.18,
                label="IC 95% (blocos espaciais)",
            )
        axis.set_xlabel("Δ IoU (SegFormer − U-Net)")
        axis.set_ylabel("Recortes")
        axis.set_title("Diferença pareada")
        axis.legend(loc="upper left")
        figstyle.panel_label(axis, "(d)")

    return figstyle.render_publication(_build, height_cm=14.5)


def save_comparison_figure(
    storage_paths: dict[str, Path],
    source_name: str,
    payload: dict[str, Any],
    patch_metrics: pd.DataFrame,
) -> Path:
    """Persiste a figura comparativa em PNG e PDF (regenerada a cada execução)."""
    figure_path = figstyle.persist_figure(
        comparison_figure_path(storage_paths, source_name),
        render_comparison_figure(payload, patch_metrics),
    )
    print(f"Figura comparativa salva em: {figure_path}")
    return figure_path


def inference_figure_file_name(source_name: str) -> str:
    """Nome estável da figura de inferência estatística (floresta de IC)."""
    config = get_config()
    return (
        f"inference_{config['aoi']['region_code']}_{reference_year(source_name)}_{source_name}.png"
    )


def inference_figure_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho da figura de inferência estatística."""
    return storage_paths["artifacts_figures"] / inference_figure_file_name(source_name)


def render_inference_figure(payload: dict[str, Any]) -> dict[str, bytes]:
    """Floresta de IC 95% da diferença (SegFormer − U-Net) por nível de inferência.

    Compara o IC bootstrap por patch (descritivo, subestima a incerteza), o IC por
    blocos espaciais (inferência principal) e o IC por dobras (n = 5, t de Student).
    """
    metrics = list(METRIC_KEYS)
    spatial = payload["spatial_inference"]
    levels = (
        ("Patches (descritivo)", "#9e9e9e", lambda m: payload["paired"][m]),
        (
            "Blocos espaciais",
            figstyle.MODEL_COLORS["segformer"],
            lambda m: spatial["block_level"][m],
        ),
        ("Dobras (t, n = 5)", "#e69f00", lambda m: spatial["fold_level"][m]),
    )

    def _build(plt: Any, figure: Any) -> None:
        axis = figure.subplots()
        height = 0.22
        for row, metric in enumerate(metrics):
            for level, (label, color, getter) in enumerate(levels):
                stats = getter(metric)
                mean = stats.get("mean_difference")
                low, high = stats.get("ci95_low"), stats.get("ci95_high")
                if mean is None or low is None or high is None:
                    continue
                y = row + (level - 1) * height
                axis.errorbar(
                    float(mean),
                    y,
                    xerr=[
                        [max(0.0, float(mean) - float(low))],
                        [max(0.0, float(high) - float(mean))],
                    ],
                    fmt="o",
                    color=color,
                    capsize=3,
                    markersize=4,
                    label=label if row == 0 else None,
                )
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_yticks(range(len(metrics)), [figstyle.METRIC_LABELS[m] for m in metrics])
        axis.invert_yaxis()
        axis.set_xlabel("Diferença média (SegFormer − U-Net) com IC 95%")
        axis.set_title("Incerteza da diferença segundo a unidade de análise")
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=3)

    return figstyle.render_publication(_build, height_cm=8.5)


def save_inference_figure(
    storage_paths: dict[str, Path], source_name: str, payload: dict[str, Any]
) -> Path:
    """Persiste a figura de inferência estatística em PNG e PDF."""
    figure_path = figstyle.persist_figure(
        inference_figure_path(storage_paths, source_name), render_inference_figure(payload)
    )
    print(f"Figura de inferência salva em: {figure_path}")
    return figure_path


def threshold_figure_file_name(source_name: str) -> str:
    """Nome estável da figura de sensibilidade ao limiar."""
    config = get_config()
    return (
        f"threshold_{config['aoi']['region_code']}_{reference_year(source_name)}_{source_name}.png"
    )


def threshold_figure_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho da figura de sensibilidade ao limiar."""
    return storage_paths["artifacts_figures"] / threshold_figure_file_name(source_name)


def render_threshold_figure(payload: dict[str, Any]) -> dict[str, bytes]:
    """Curvas P×R e IoU×limiar dos modelos, com o limiar fixo (0,5) e o calibrado."""
    sensitivity = payload["threshold_sensitivity"]
    if sensitivity is None:
        raise ValueError("Comparação sem análise de sensibilidade ao limiar.")

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(1, 2)
        for model_name in MODEL_NAMES:
            analysis = sensitivity["models"][model_name]
            color = figstyle.MODEL_COLORS[model_name]
            curve = analysis["pr_curve"]
            label = f"{MODEL_LABELS[model_name]} (AP = {analysis['average_precision']:.3f})"
            axes[0].plot(curve["recall"], curve["precision"], color=color, label=label)
            fixed = analysis["fixed"]["metrics"]
            axes[0].scatter(
                [fixed["recall"]], [fixed["precision"]], color=color, marker="o", s=26, zorder=4
            )
            axes[1].plot(
                curve["threshold"], curve["iou"], color=color, label=MODEL_LABELS[model_name]
            )
            axes[1].scatter(
                [analysis["fixed"]["threshold"]],
                [fixed["iou"]],
                color=color,
                marker="o",
                s=26,
                zorder=4,
            )
            axes[1].scatter(
                [analysis["oracle"]["threshold"]],
                [analysis["oracle"]["metrics"]["iou"]],
                color=color,
                marker="*",
                s=60,
                zorder=4,
            )
        axes[0].set_xlabel("Revocação")
        axes[0].set_ylabel("Precisão")
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1.02)
        axes[0].set_title("Curva precisão × revocação")
        axes[0].legend(loc="lower left")
        axes[0].text(
            0.98,
            0.98,
            "● limiar 0,5",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
        axes[1].set_xlabel("Limiar de probabilidade")
        axes[1].set_ylabel("IoU agregado")
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].set_title("IoU em função do limiar")
        axes[1].legend(loc="lower center")
        axes[1].text(
            0.98,
            0.98,
            "● limiar 0,5\n★ limiar ótimo",
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
        figstyle.panel_label(axes[0], "(a)")
        figstyle.panel_label(axes[1], "(b)")

    return figstyle.render_publication(_build, height_cm=7.5)


def save_threshold_figure(
    storage_paths: dict[str, Path], source_name: str, payload: dict[str, Any]
) -> Path:
    """Persiste a figura de sensibilidade ao limiar em PNG e PDF."""
    figure_path = figstyle.persist_figure(
        threshold_figure_path(storage_paths, source_name), render_threshold_figure(payload)
    )
    print(f"Figura de sensibilidade ao limiar salva em: {figure_path}")
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
                image_rgb=rgb_composite(image, valid, list(get_config()["data"]["bands"])),
                truth=truth,
                valid=valid,
                predictions=model_predictions,
                iou={model_name: float(item[f"iou_{model_name}"]) for model_name in MODEL_NAMES},
            )
        )
    return samples


def render_error_maps_figure(samples: list[ErrorMapSample]) -> dict[str, bytes]:
    """Por recorte: imagem RGB, referência e mapas de erro dos dois modelos."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    if not samples:
        raise ValueError("Nenhuma amostra disponível para os mapas de erro.")
    error_cmap = ListedColormap(ERROR_COLORS)
    error_norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), error_cmap.N)
    truth_cmap = ListedColormap(["#f5f5f5", "#2e7d32", "#ffffff"])
    columns = ["Imagem (RGB)", "Referência", *[MODEL_LABELS[name] for name in MODEL_NAMES]]
    row_height_cm = 3.9

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(len(samples), 4, squeeze=False)
        for row, sample in enumerate(samples):
            truth_code = np.where(sample.valid, sample.truth.astype(np.uint8), 2)
            axes[row, 0].imshow(sample.image_rgb)
            axes[row, 1].imshow(truth_code, cmap=truth_cmap, vmin=0, vmax=2)
            for column, model_name in enumerate(MODEL_NAMES, start=2):
                codes = error_code_map(sample.predictions[model_name], sample.truth, sample.valid)
                axes[row, column].imshow(
                    codes, cmap=error_cmap, norm=error_norm, interpolation="nearest"
                )
                axes[row, column].text(
                    0.02,
                    0.02,
                    f"IoU {sample.iou[model_name]:.3f}",
                    transform=axes[row, column].transAxes,
                    fontsize=7,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                )
            axes[row, 0].set_ylabel(
                f"{figstyle.wrap_label(sample.label)}\n(dobra {sample.fold})",
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
        handles = [
            Patch(facecolor=color, edgecolor="#555555", label=label)
            for color, label in zip(ERROR_COLORS[:4], ERROR_LABELS[:4], strict=True)
        ]
        figure.legend(handles=handles, loc="outside lower center", ncol=4)

    return figstyle.render_publication(_build, height_cm=row_height_cm * len(samples) + 2.2)


def save_error_maps_figure(
    storage_paths: dict[str, Path],
    source_name: str,
    selected: list[dict[str, Any]],
) -> Path:
    """Persiste a figura de mapas de erro em PNG e PDF (regenerada a cada execução)."""
    samples = build_error_map_samples(storage_paths, source_name, selected)
    figure_path = figstyle.persist_figure(
        error_maps_figure_path(storage_paths, source_name), render_error_maps_figure(samples)
    )
    print(f"Mapas de erro salvos em: {figure_path}")
    return figure_path
