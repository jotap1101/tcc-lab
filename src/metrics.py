"""Métricas de segmentação a nível de pixel: IoU, F1, Precisão e Recall.

Calculadas sobre a classe de interesse (café) a partir de contagens de confusão
(TP/FP/FN) acumuladas por lote. ``SegmentationMetrics`` acumula as contagens ao
longo de uma época e deriva as quatro métricas no final, evitando média de
métricas por lote (que distorce quando a classe é rara). Pixels com rótulo -1
(fora do AOI) são ignorados. Classes vazias (sem positivos previstos e sem
positivos reais) são consideradas perfeitas, evitando divisão por zero.
"""

from __future__ import annotations

import torch


def _safe_ratio(numerator: int, denominator: int, fallback: float) -> float:
    """Divide sem divisão por zero, usando o fallback quando o denominador é nulo."""
    if denominator == 0:
        return fallback
    return float(numerator) / float(denominator)


def confusion_counts(pred: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int]:
    """Contagens de confusão binárias (TP, FP, FN) para a classe de interesse."""
    positive = pred & target
    false_positive = pred & ~target
    false_negative = ~pred & target
    return (
        int(positive.sum()),
        int(false_positive.sum()),
        int(false_negative.sum()),
    )


def iou(tp: int, fp: int, fn: int) -> float:
    """Intersection over Union; 1.0 quando não há positivos reais nem previstos."""
    return _safe_ratio(tp, tp + fp + fn, fallback=1.0)


def precision(tp: int, fp: int) -> float:
    """Precisão (valor preditivo positivo); 1.0 sem falsos positivos."""
    return _safe_ratio(tp, tp + fp, fallback=1.0)


def recall(tp: int, fn: int) -> float:
    """Recall (sensibilidade); 1.0 sem falsos negativos."""
    return _safe_ratio(tp, tp + fn, fallback=1.0)


def f1_score(tp: int, fp: int, fn: int) -> float:
    """F1-Score (média harmônica de precisão e recall); 1.0 em classes vazias."""
    return _safe_ratio(2 * tp, 2 * tp + fp + fn, fallback=1.0)


class SegmentationMetrics:
    """Acumulador determinístico das métricas de segmentação ao longo de uma época."""

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Acumula as contagens de confusão de um lote (threshold de 0.5).

        Pixels com rótulo -1 (fora do AOI) são ignorados em TP/FP/FN.
        """
        valid = targets >= 0
        preds = (torch.sigmoid(logits) >= 0.5) & valid
        batch_tp, batch_fp, batch_fn = confusion_counts(preds, (targets > 0.5) & valid)
        self.tp += batch_tp
        self.fp += batch_fp
        self.fn += batch_fn

    def compute(self) -> dict[str, float]:
        """Deriva IoU, F1, Precisão e Recall das contagens acumuladas."""
        return {
            "iou": iou(self.tp, self.fp, self.fn),
            "f1": f1_score(self.tp, self.fp, self.fn),
            "precision": precision(self.tp, self.fp),
            "recall": recall(self.tp, self.fn),
        }
