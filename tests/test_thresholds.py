"""Testes de src/thresholds.py (análise independente de limiar)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src import thresholds
from src.thresholds import PROBABILITY_BINS


def _histograms() -> tuple[np.ndarray, np.ndarray]:
    """Histogramas sintéticos: café concentrado em probabilidades altas, fundo em baixas."""
    positive = np.zeros(PROBABILITY_BINS, dtype=np.int64)
    negative = np.zeros(PROBABILITY_BINS, dtype=np.int64)
    positive[200], positive[100] = 80, 20
    negative[10], negative[100] = 880, 20
    return positive, negative


def test_probability_bins_cover_extremes() -> None:
    """Probabilidades 0 e 1 devem cair no primeiro e no último bin."""
    bins = thresholds.probability_bins(torch.tensor([0.0, 0.5, 0.999, 1.0]))
    assert bins.tolist() == [0, PROBABILITY_BINS // 2, PROBABILITY_BINS - 1, PROBABILITY_BINS - 1]


def test_counts_at_edges_are_consistent() -> None:
    """Nos extremos, tudo é café (limiar 0) ou nada é café (limiar máximo)."""
    positive, negative = _histograms()
    counts = thresholds.counts_at_edges(positive, negative)
    assert counts["tp"][0] == 100 and counts["fp"][0] == 900 and counts["fn"][0] == 0
    assert counts["tp"][-1] == 0 and counts["fp"][-1] == 0 and counts["fn"][-1] == 100
    middle = thresholds.DEFAULT_THRESHOLD_BIN  # bin 128: apenas o bin 200 passa
    assert (counts["tp"][middle], counts["fp"][middle], counts["fn"][middle]) == (80, 0, 20)


def test_average_precision_is_perfect_when_separable() -> None:
    """Classes perfeitamente separáveis devem render AP = 1."""
    positive = np.zeros(PROBABILITY_BINS, dtype=np.int64)
    negative = np.zeros(PROBABILITY_BINS, dtype=np.int64)
    positive[250], negative[5] = 100, 100
    assert thresholds.average_precision(positive, negative) == pytest.approx(1.0)
    assert thresholds.average_precision(np.zeros(PROBABILITY_BINS), negative) is None


def test_best_threshold_maximizes_iou() -> None:
    """O limiar ótimo (IoU 0,83) inclui o bin 100 e, no empate, fica o mais próximo de 0,5."""
    positive, negative = _histograms()
    best = thresholds.best_threshold_bin(positive, negative, "iou")
    assert best == 100
    counts = thresholds.counts_at_edges(positive, negative)
    values = thresholds.metrics_from_counts(counts["tp"], counts["fp"], counts["fn"])["iou"]
    assert values[best] == pytest.approx(values[1:PROBABILITY_BINS].max())


def test_leave_one_fold_out_does_not_use_held_out_fold() -> None:
    """A dobra com distribuição enganosa não pode influenciar o próprio limiar."""
    good_pos = np.zeros(PROBABILITY_BINS, dtype=np.int64)
    good_neg = np.zeros(PROBABILITY_BINS, dtype=np.int64)
    good_pos[200], good_neg[20] = 100, 100
    odd_pos = np.zeros(PROBABILITY_BINS, dtype=np.int64)
    odd_neg = np.zeros(PROBABILITY_BINS, dtype=np.int64)
    odd_pos[60], odd_neg[20] = 100, 100  # café só aparece em probabilidade baixa
    positive = {0: good_pos, 1: good_pos, 2: odd_pos}
    negative = {0: good_neg, 1: good_neg, 2: odd_neg}

    chosen = thresholds.leave_one_fold_out_bins(positive, negative, "iou")

    assert chosen[2] > 60  # calibrado só nas dobras boas: ignora o café em 60
    assert chosen[0] <= 60  # a dobra 0 vê a dobra 2 entre as demais


def test_counts_per_patch_at_bin_matches_pooled() -> None:
    """A soma dos patches sob um limiar deve igualar a contagem agregada."""
    positive, negative = _histograms()
    stacked_pos = np.stack([positive // 2, positive - positive // 2])
    stacked_neg = np.stack([negative // 2, negative - negative // 2])
    per_patch = thresholds.counts_per_patch_at_bin(stacked_pos, stacked_neg, 128)
    pooled = thresholds.counts_at_edges(positive, negative)
    assert int(per_patch["tp"].sum()) == int(pooled["tp"][128])
    assert int(per_patch["fp"].sum()) == int(pooled["fp"][128])
    assert int(per_patch["fn"].sum()) == int(pooled["fn"][128])
