"""Testes de src/metrics.py (IoU, F1, Precisão e Recall a nível de pixel)."""

from __future__ import annotations

import torch

from src.metrics import (
    SegmentationMetrics,
    confusion_counts,
    f1_score,
    iou,
    precision,
    recall,
)


def test_confusion_counts_on_known_case() -> None:
    """As contagens TP/FP/FN devem ser as de um caso conhecido."""
    pred = torch.tensor([[True, True, False, False]])
    target = torch.tensor([[True, False, True, False]])
    assert confusion_counts(pred, target) == (1, 1, 1)


def test_iou_value() -> None:
    """IoU = interseção / união para contagens conhecidas."""
    assert iou(tp=2, fp=1, fn=1) == 2 / 4


def test_precision_value() -> None:
    """Precisão = TP / (TP + FP)."""
    assert precision(tp=2, fp=1) == 2 / 3


def test_recall_value() -> None:
    """Recall = TP / (TP + FN)."""
    assert recall(tp=2, fn=1) == 2 / 3


def test_f1_value() -> None:
    """F1 = 2TP / (2TP + FP + FN)."""
    assert f1_score(tp=2, fp=1, fn=1) == 4 / 6


def test_metrics_perfect_with_empty_class() -> None:
    """Sem positivos reais nem previstos, todas as métricas devem ser 1.0."""
    assert iou(0, 0, 0) == 1.0
    assert precision(0, 0) == 1.0
    assert recall(0, 0) == 1.0
    assert f1_score(0, 0, 0) == 1.0


def test_segmentation_metrics_accumulation() -> None:
    """O acumulador deve somar as contagens de vários lotes."""
    accumulator = SegmentationMetrics()
    accumulator.update(
        torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]]),
        torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
    )
    accumulator.update(
        torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]]),
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
    )
    assert accumulator.tp == 3
    assert accumulator.fp == 1
    assert accumulator.fn == 1


def test_segmentation_metrics_ignores_invalid_pixels() -> None:
    """Pixels com rótulo -1 não devem entrar em TP/FP/FN."""
    accumulator = SegmentationMetrics()
    accumulator.update(
        torch.tensor([[[[1.0, 1.0]]]]),
        torch.tensor([[[[1.0, -1.0]]]]),
    )
    assert (accumulator.tp, accumulator.fp, accumulator.fn) == (1, 0, 0)


def test_segmentation_metrics_compute_returns_all_metrics() -> None:
    """compute deve devolver IoU, F1, Precisão e Recall."""
    accumulator = SegmentationMetrics()
    accumulator.update(
        torch.tensor([[[[0.9, 0.1], [0.1, 0.1]]]]),
        torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]]),
    )
    metrics = accumulator.compute()
    assert set(metrics) == {"iou", "f1", "precision", "recall"}
    assert 0.0 <= metrics["iou"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_segmentation_metrics_threshold() -> None:
    """O limiar de decisão deve ser 0.5 sobre a sigmoide dos logits."""
    accumulator = SegmentationMetrics()
    accumulator.update(
        torch.tensor([[[[-1.0, 1.0]]]]),
        torch.tensor([[[[0.0, 1.0]]]]),
    )
    assert accumulator.tp == 1
    assert accumulator.fp == 0
    assert accumulator.fn == 0
