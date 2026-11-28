"""Análise independente de limiar a partir de histogramas de probabilidade.

O estágio 11 persiste, por patch, o histograma das probabilidades preditas
(`sigmoid(logits)`) separadas em pixels de café e de não-café válidos. Com esses
histogramas é possível, **sem nova inferência e sem retreino**, obter:

- a curva precisão × revocação e a precisão média (AP), que não dependem de
  limiar e eliminam o viés de comparar dois modelos com calibrações diferentes;
- o limiar que maximiza uma métrica e as métricas por patch sob esse limiar.

O limiar de cada dobra é escolhido nas **outras** dobras (calibração leave-one-
fold-out), de modo que o desempenho reportado não use a dobra de validação para
ajustar o limiar. A resolução do limiar é 1/`PROBABILITY_BINS`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PROBABILITY_BINS = 256
DEFAULT_THRESHOLD_BIN = PROBABILITY_BINS // 2  # limiar 0,5 (aresta exata do bin)
CALIBRATION_METRICS = ("iou", "f1")


def threshold_value(bin_index: int) -> float:
    """Limiar de probabilidade correspondente à aresta inferior do bin."""
    return float(bin_index) / PROBABILITY_BINS


def probability_bins(probabilities: Any) -> Any:
    """Índice do bin de cada probabilidade (tensor/array em [0, 1]); 1,0 cai no último bin."""
    scaled = probabilities * PROBABILITY_BINS
    return scaled.floor().clamp(0, PROBABILITY_BINS - 1).long()


def counts_at_edges(positive: np.ndarray, negative: np.ndarray) -> dict[str, np.ndarray]:
    """TP/FP/FN para cada limiar (aresta de bin) a partir dos histogramas.

    As entradas têm forma ``(..., PROBABILITY_BINS)``. O limiar ``b`` classifica
    como café os pixels com bin ``>= b``; ``b`` varia de 0 (tudo café) a
    ``PROBABILITY_BINS`` (nada café), totalizando ``PROBABILITY_BINS + 1`` pontos.
    """
    pos = np.asarray(positive, dtype=np.int64)
    neg = np.asarray(negative, dtype=np.int64)
    zeros = np.zeros((*pos.shape[:-1], 1), dtype=np.int64)
    pos_above = np.concatenate([np.cumsum(pos[..., ::-1], axis=-1)[..., ::-1], zeros], axis=-1)
    neg_above = np.concatenate([np.cumsum(neg[..., ::-1], axis=-1)[..., ::-1], zeros], axis=-1)
    total_pos = pos.sum(axis=-1, keepdims=True)
    return {"tp": pos_above, "fp": neg_above, "fn": total_pos - pos_above}


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divisão elemento a elemento com 0 onde o denominador é nulo."""
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def metrics_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> dict[str, np.ndarray]:
    """IoU, F1, precisão e revocação a partir das contagens (vetorizado)."""
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    return {
        "iou": _safe_divide(tp, tp + fp + fn),
        "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
    }


def average_precision(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Precisão média (área sob a curva P×R, soma de Riemann à direita).

    Recebe os histogramas 1D (``PROBABILITY_BINS``) já agregados; retorna None
    quando não há pixel de café.
    """
    counts = counts_at_edges(positive, negative)
    if int(counts["tp"][0] + counts["fn"][0]) == 0:
        return None
    metrics = metrics_from_counts(counts["tp"], counts["fp"], counts["fn"])
    recall = metrics["recall"]
    precision = metrics["precision"]
    # Limiares crescentes => revocação decrescente; soma (R_k − R_{k+1}) · P_k.
    return float(np.sum((recall[:-1] - recall[1:]) * precision[:-1]))


def precision_recall_curve(
    positive: np.ndarray, negative: np.ndarray, stride: int = 4
) -> dict[str, list[float]]:
    """Pontos da curva P×R (um a cada ``stride`` bins) para figuras e auditoria."""
    counts = counts_at_edges(positive, negative)
    metrics = metrics_from_counts(counts["tp"], counts["fp"], counts["fn"])
    index = np.arange(0, PROBABILITY_BINS + 1, stride)
    return {
        "threshold": [threshold_value(int(b)) for b in index],
        "precision": [float(value) for value in metrics["precision"][index]],
        "recall": [float(value) for value in metrics["recall"][index]],
        "iou": [float(value) for value in metrics["iou"][index]],
        "f1": [float(value) for value in metrics["f1"][index]],
    }


def best_threshold_bin(positive: np.ndarray, negative: np.ndarray, metric: str = "iou") -> int:
    """Bin do limiar que maximiza a métrica (empate: o mais próximo de 0,5)."""
    if metric not in CALIBRATION_METRICS:
        raise ValueError(f"Métrica de calibração inválida: {metric}.")
    counts = counts_at_edges(positive, negative)
    values = metrics_from_counts(counts["tp"], counts["fp"], counts["fn"])[metric]
    # Exclui os extremos degenerados (tudo café / nada café).
    candidates = np.arange(1, PROBABILITY_BINS)
    best = float(values[candidates].max())
    tied = candidates[np.isclose(values[candidates], best, rtol=0.0, atol=1e-12)]
    return int(tied[np.argmin(np.abs(tied - DEFAULT_THRESHOLD_BIN))])


def counts_per_patch_at_bin(
    positive: np.ndarray, negative: np.ndarray, bin_index: int
) -> dict[str, np.ndarray]:
    """TP/FP/FN de cada patch ``(n, bins)`` sob um único limiar (bin)."""
    pos = np.asarray(positive, dtype=np.int64)
    neg = np.asarray(negative, dtype=np.int64)
    tp = pos[:, bin_index:].sum(axis=1)
    fp = neg[:, bin_index:].sum(axis=1)
    fn = pos[:, :bin_index].sum(axis=1)
    return {"tp": tp, "fp": fp, "fn": fn}


def leave_one_fold_out_bins(
    fold_positive: dict[int, np.ndarray],
    fold_negative: dict[int, np.ndarray],
    metric: str = "iou",
) -> dict[int, int]:
    """Limiar de cada dobra escolhido apenas nas demais dobras (sem vazamento)."""
    folds = sorted(fold_positive)
    chosen: dict[int, int] = {}
    for fold in folds:
        others = [other for other in folds if other != fold]
        if not others:
            chosen[fold] = DEFAULT_THRESHOLD_BIN
            continue
        pooled_pos = np.sum([fold_positive[other] for other in others], axis=0)
        pooled_neg = np.sum([fold_negative[other] for other in others], axis=0)
        chosen[fold] = best_threshold_bin(pooled_pos, pooled_neg, metric)
    return chosen
