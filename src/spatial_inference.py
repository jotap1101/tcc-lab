"""Inferência estatística com dependência espacial (estágio 12).

Os 125 patches de validação não são amostras independentes: patches vizinhos
compartilham relevo, manejo e condições de imageamento, e a validação cruzada
espacial agrupa os patches em dobras contíguas. Testes que tratam cada patch
como unidade independente (t pareado, Wilcoxon) subestimam a incerteza e geram
p-valores excessivamente pequenos. Este módulo fornece a inferência adequada à
estrutura dos dados, em dois níveis:

- **blocos espaciais**: patches vizinhos (grade `block_size × block_size`)
  formam um bloco; o bootstrap por blocos reamostra blocos inteiros e o teste de
  inversão de sinais (permutação) opera sobre as diferenças médias dos blocos;
- **dobras**: a unidade é a dobra de validação (n = 5), com IC pela
  distribuição t de Student e teste de inversão de sinais exato.

Nenhuma função reavalia ou retreina modelos; todas operam sobre diferenças
pareadas já persistidas pelo estágio 11.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

SPATIAL_BLOCK_SIZE = 3  # patches por lado do bloco (≈ 15 km com patches de 5,12 km)
SENSITIVITY_BLOCK_SIZES = (1, 2, 3, 4)
BOOTSTRAP_ITERATIONS = 10000
PERMUTATION_ITERATIONS = 100000
EXACT_PERMUTATION_MAX_UNITS = 16
CONFIDENCE = 0.95

_GRID_PATTERN = re.compile(r"_r(\d+)_c(\d+)$")


def parse_patch_grid(patch_id: str) -> tuple[int, int] | None:
    """Extrai (linha, coluna) da grade do identificador `..._r0009_c0012`; None se ausente."""
    match = _GRID_PATTERN.search(str(patch_id))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def assign_blocks(patch_ids: list[str] | pd.Series, block_size: int) -> np.ndarray:
    """Rótulo inteiro do bloco espacial de cada patch (patch sem grade vira bloco próprio)."""
    if block_size < 1:
        raise ValueError("block_size deve ser >= 1.")
    keys: list[tuple[int, int] | tuple[str]] = []
    for patch_id in patch_ids:
        grid = parse_patch_grid(str(patch_id))
        if grid is None:
            keys.append((str(patch_id),))
        else:
            keys.append((grid[0] // block_size, grid[1] // block_size))
    labels: dict[Any, int] = {}
    return np.array([labels.setdefault(key, len(labels)) for key in keys], dtype=np.int64)


def block_means(values: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Média dos valores em cada bloco (ordenada pelo rótulo do bloco)."""
    values = np.asarray(values, dtype=np.float64)
    totals = np.bincount(blocks, weights=values)
    counts = np.bincount(blocks)
    return totals / counts


def cluster_bootstrap_ci(
    values: np.ndarray,
    blocks: np.ndarray,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = CONFIDENCE,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    """IC percentil da média por bootstrap de blocos (reamostra blocos inteiros)."""
    values = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(values)
    values, blocks = values[keep], blocks[keep]
    if values.size == 0:
        return (None, None)
    _, inverse = np.unique(blocks, return_inverse=True)
    totals = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(np.float64)
    n_blocks = totals.size
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    chunk = 2048
    for start in range(0, iterations, chunk):
        size = min(chunk, iterations - start)
        picks = rng.integers(0, n_blocks, size=(size, n_blocks))
        means[start : start + size] = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def sign_flip_pvalue(
    unit_values: np.ndarray,
    *,
    iterations: int = PERMUTATION_ITERATIONS,
    seed: int = 0,
) -> float | None:
    """P-valor bicaudal do teste de inversão de sinais para a média das unidades.

    Exato (enumeração dos 2^n sinais) para até `EXACT_PERMUTATION_MAX_UNITS`
    unidades; Monte Carlo com correção (k + 1)/(N + 1) acima disso. Com n unidades
    o menor p-valor exato possível é 2/2^n.
    """
    values = np.asarray(unit_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count == 0:
        return None
    observed = abs(float(values.mean()))
    if count <= EXACT_PERMUTATION_MAX_UNITS:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=count)))
        statistics = np.abs((signs * values).mean(axis=1))
        return float(np.mean(statistics >= observed - 1e-15))
    rng = np.random.default_rng(seed)
    exceed = 0
    chunk = 4096
    done = 0
    while done < iterations:
        size = min(chunk, iterations - done)
        signs = rng.choice((-1.0, 1.0), size=(size, count))
        exceed += int(np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))
        done += size
    return float((exceed + 1) / (iterations + 1))


def fold_level_statistics(fold_differences: np.ndarray, seed: int = 0) -> dict[str, Any]:
    """Inferência com a dobra como unidade (IC t de Student e inversão de sinais exata)."""
    values = np.asarray(fold_differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = int(values.size)
    result: dict[str, Any] = {
        "n_folds": count,
        "mean_difference": None,
        "std_difference": None,
        "ci95_low": None,
        "ci95_high": None,
        "p_sign_flip": None,
        "segformer_better_folds": int(np.count_nonzero(values > 1e-12)),
        "unet_better_folds": int(np.count_nonzero(values < -1e-12)),
    }
    if count == 0:
        return result
    mean = float(values.mean())
    result["mean_difference"] = mean
    if count >= 2:
        std = float(values.std(ddof=1))
        margin = (
            float(scipy_stats.t.ppf(0.5 + CONFIDENCE / 2.0, df=count - 1)) * std / np.sqrt(count)
        )
        result.update(
            {
                "std_difference": std,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
                "p_sign_flip": sign_flip_pvalue(values, seed=seed),
            }
        )
    return result


def block_level_statistics(
    differences: np.ndarray,
    patch_ids: list[str] | pd.Series,
    *,
    block_size: int = SPATIAL_BLOCK_SIZE,
    seed: int = 0,
) -> dict[str, Any]:
    """Inferência com o bloco espacial como unidade (bootstrap de blocos + inversão de sinais)."""
    differences = np.asarray(differences, dtype=np.float64)
    blocks = assign_blocks(patch_ids, block_size)
    finite = np.isfinite(differences)
    differences, blocks = differences[finite], blocks[finite]
    result: dict[str, Any] = {
        "block_size": int(block_size),
        "n_blocks": 0,
        "mean_difference": None,
        "ci95_low": None,
        "ci95_high": None,
        "p_sign_flip": None,
        "segformer_better_blocks": 0,
        "unet_better_blocks": 0,
    }
    if differences.size == 0:
        return result
    _, relabeled = np.unique(blocks, return_inverse=True)
    unit_means = block_means(differences, relabeled)
    low, high = cluster_bootstrap_ci(differences, relabeled, seed=seed)
    result.update(
        {
            "n_blocks": int(unit_means.size),
            "mean_difference": float(differences.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "p_sign_flip": sign_flip_pvalue(unit_means, seed=seed),
            "segformer_better_blocks": int(np.count_nonzero(unit_means > 1e-12)),
            "unet_better_blocks": int(np.count_nonzero(unit_means < -1e-12)),
        }
    )
    return result


def holm_adjust(pvalues: dict[str, float | None]) -> dict[str, float | None]:
    """Correção de Holm–Bonferroni (monotônica, limitada a 1,0) ignorando valores nulos."""
    finite = {key: value for key, value in pvalues.items() if value is not None}
    ordered = sorted(finite.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[key] = running
    return {key: adjusted.get(key) if value is not None else None for key, value in pvalues.items()}


def spatial_inference(
    patch_metrics: pd.DataFrame,
    metrics: tuple[str, ...],
    *,
    seed: int,
    block_size: int = SPATIAL_BLOCK_SIZE,
) -> dict[str, Any]:
    """Inferência por blocos e por dobras das diferenças (SegFormer − U-Net) por métrica.

    ``patch_metrics`` traz `patch_id`, `fold` e as colunas `<metrica>_segformer` /
    `<metrica>_unet`. A diferença por dobra usa a média dos patches da dobra (macro).
    """
    block_results: dict[str, dict[str, Any]] = {}
    fold_results: dict[str, dict[str, Any]] = {}
    sensitivity: dict[str, dict[str, Any]] = {}
    for index, metric in enumerate(metrics):
        difference = patch_metrics[f"{metric}_segformer"].to_numpy(
            dtype=np.float64
        ) - patch_metrics[f"{metric}_unet"].to_numpy(dtype=np.float64)
        block_results[metric] = block_level_statistics(
            difference, patch_metrics["patch_id"], block_size=block_size, seed=seed + index
        )
        by_fold = (
            pd.DataFrame({"fold": patch_metrics["fold"].to_numpy(), "difference": difference})
            .dropna()
            .groupby("fold")["difference"]
            .mean()
        )
        fold_results[metric] = fold_level_statistics(by_fold.to_numpy(), seed=seed + index)
    iou_difference = patch_metrics["iou_segformer"].to_numpy(dtype=np.float64) - patch_metrics[
        "iou_unet"
    ].to_numpy(dtype=np.float64)
    for size in SENSITIVITY_BLOCK_SIZES:
        stats = block_level_statistics(
            iou_difference, patch_metrics["patch_id"], block_size=size, seed=seed
        )
        sensitivity[str(size)] = {
            key: stats[key] for key in ("n_blocks", "mean_difference", "ci95_low", "ci95_high")
        }
    adjusted = holm_adjust({metric: block_results[metric]["p_sign_flip"] for metric in metrics})
    for metric in metrics:
        block_results[metric]["p_value_holm"] = adjusted[metric]
    return {
        "block_size": int(block_size),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "confidence": CONFIDENCE,
        "block_level": block_results,
        "fold_level": fold_results,
        "block_size_sensitivity_iou": sensitivity,
    }
