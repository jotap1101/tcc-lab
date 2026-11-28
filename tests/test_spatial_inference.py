"""Testes de src/spatial_inference.py (inferência com dependência espacial)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import spatial_inference as si


def test_parse_patch_grid_and_blocks() -> None:
    """Patches vizinhos compartilham o bloco; identificadores sem grade ficam isolados."""
    assert si.parse_patch_grid("composite_310044_2018_r0009_c0012") == (9, 12)
    assert si.parse_patch_grid("patch_001") is None
    ids = [
        "x_r0000_c0000",
        "x_r0001_c0002",
        "x_r0003_c0000",
        "patch_001",
    ]
    blocks = si.assign_blocks(ids, block_size=3)
    assert blocks[0] == blocks[1]
    assert len({blocks[0], blocks[2], blocks[3]}) == 3


def test_cluster_bootstrap_is_wider_than_naive_for_clustered_data() -> None:
    """Dados com forte efeito de bloco devem gerar IC maior que o bootstrap por patch."""
    rng = np.random.default_rng(0)
    block_effects = rng.normal(0.0, 0.1, size=10)
    values = np.repeat(block_effects, 12) + rng.normal(0.0, 0.01, size=120)
    blocks = np.repeat(np.arange(10), 12)

    low, high = si.cluster_bootstrap_ci(values, blocks, iterations=2000, seed=1)
    naive_low, naive_high = si.cluster_bootstrap_ci(values, np.arange(120), iterations=2000, seed=1)

    assert low is not None and high is not None and naive_low is not None and naive_high is not None
    assert (high - low) > 2 * (naive_high - naive_low)


def test_sign_flip_pvalue_exact_and_bounds() -> None:
    """Com 5 unidades todas positivas, o menor p bicaudal exato é 2/32."""
    assert si.sign_flip_pvalue(np.array([0.1, 0.2, 0.3, 0.1, 0.2])) == pytest.approx(2 / 32)
    assert si.sign_flip_pvalue(np.array([0.1, -0.1, 0.1, -0.1])) == pytest.approx(1.0)
    assert si.sign_flip_pvalue(np.array([])) is None
    large = si.sign_flip_pvalue(np.full(30, 0.1), iterations=2000, seed=3)
    assert large is not None and large < 0.01


def test_fold_level_statistics() -> None:
    """A inferência por dobras deve usar IC t com n − 1 graus de liberdade."""
    stats = si.fold_level_statistics(np.array([0.10, 0.04, 0.04, 0.10, 0.04]))
    assert stats["n_folds"] == 5
    assert stats["segformer_better_folds"] == 5
    assert stats["ci95_low"] is not None and stats["ci95_low"] > 0
    assert stats["p_sign_flip"] == pytest.approx(2 / 32)


def test_spatial_inference_payload_shape() -> None:
    """O payload agrega blocos, dobras, sensibilidade e Holm para todas as métricas."""
    rng = np.random.default_rng(2)
    rows = []
    for index in range(40):
        row, col = divmod(index, 8)
        base = rng.uniform(0.3, 0.7)
        record = {"patch_id": f"p_r{row:04d}_c{col:04d}", "fold": index % 5}
        for metric in ("iou", "f1"):
            record[f"{metric}_unet"] = base
            record[f"{metric}_segformer"] = base + 0.05 + rng.normal(0, 0.01)
        rows.append(record)
    frame = pd.DataFrame(rows)

    result = si.spatial_inference(frame, ("iou", "f1"), seed=0)

    assert set(result["block_level"]) == {"iou", "f1"}
    assert result["block_level"]["iou"]["n_blocks"] >= 4
    assert result["block_level"]["iou"]["p_value_holm"] is not None
    assert set(result["block_size_sensitivity_iou"]) == {"1", "2", "3", "4"}
    assert result["fold_level"]["iou"]["n_folds"] == 5
