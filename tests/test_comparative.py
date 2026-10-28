"""Testes de src/comparative.py (comparação estatística e mapas de erro — estágio 12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import comparative, evaluation
from src.data.patch_generation import manifest_path, patch_fingerprint_hash
from src.data.spatial_split import split_fingerprint_hash
from src.metrics import f1_score, iou, precision, recall
from src.trainer import config_hash

TEST_SOURCE = "alphaearth"
PATCH_SIZE = 8
FOLD_COUNT = 5
PATCHES_PER_FOLD = 2
MODEL_NAMES = ("unet", "segformer")
THRESHOLD = 0.5


def _storage_paths(tmp_path: Path) -> dict[str, Path]:
    """Caminhos de armazenamento mínimos para os testes do estágio 12."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "models": tmp_path / "models",
        "artifacts_metrics": tmp_path / "artifacts" / "metrics",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
    }


def _metrics(counts: dict[str, int]) -> dict[str, float]:
    """Deriva as métricas de pixel a partir das contagens de confusão."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "iou": iou(tp, fp, fn),
        "f1": f1_score(tp, fp, fn),
        "precision": precision(tp, fp),
        "recall": recall(tp, fn),
    }


def _predictions(index: int, truth: np.ndarray, model_name: str) -> np.ndarray:
    """Predição sintética com erros variáveis por patch e por modelo."""
    prediction = truth.copy()
    if model_name == "unet":
        prediction[0, index % PATCH_SIZE] = True
        if index % 3 == 0:
            prediction[3, 3] = False
    else:
        prediction[index % PATCH_SIZE, PATCH_SIZE - 1] = True
        if index % 4 == 0:
            prediction[4, 4] = False
    return prediction.astype(np.uint8)


def _counts(prediction: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    """Contagens de confusão binárias entre predição e referência."""
    pred = prediction.astype(bool)
    reference = truth.astype(bool)
    return {
        "tp": int(np.count_nonzero(pred & reference)),
        "fp": int(np.count_nonzero(pred & ~reference)),
        "fn": int(np.count_nonzero(~pred & reference)),
    }


def _write_summary(
    paths: dict[str, Path],
    model_name: str,
    source: str,
    frames: dict[int, pd.DataFrame],
    fingerprint: dict,
) -> None:
    """Escreve o `summary.json` sintético do estágio 11 para o modelo."""
    per_fold: list[dict[str, Any]] = []
    total = {"tp": 0, "fp": 0, "fn": 0}
    for fold, frame in sorted(frames.items()):
        counts = {key: int(frame[key].sum()) for key in ("tp", "fp", "fn")}
        for key in total:
            total[key] += counts[key]
        per_fold.append(
            {
                "fold": fold,
                "n_patches": int(len(frame)),
                "counts": counts,
                "metrics": _metrics(counts),
                "weights_sha256": f"{model_name}_{fold:02d}",
            }
        )
    fold_values = {key: [item["metrics"][key] for item in per_fold] for key in _metrics(total)}
    payload = {
        "schema_version": 1,
        "model": model_name,
        "source_name": source,
        "threshold": THRESHOLD,
        "n_folds": len(per_fold),
        "n_patches": int(sum(item["n_patches"] for item in per_fold)),
        "per_fold": per_fold,
        "aggregate": {"counts": total, "metrics": _metrics(total)},
        "fold_mean": {key: float(np.mean(values)) for key, values in fold_values.items()},
        "fold_std": {key: float(np.std(values)) for key, values in fold_values.items()},
        "fingerprint": fingerprint,
    }
    target = evaluation.evaluation_summary_path(paths, model_name, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Escreve manifesto, patches, predições, métricas por patch e resumos sintéticos."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    records = []
    truths: dict[str, np.ndarray] = {}
    for index in range(FOLD_COUNT * PATCHES_PER_FOLD):
        patch_id = f"patch_{index:03d}"
        image = np.random.default_rng(index).uniform(0.0, 1.0, size=(4, PATCH_SIZE, PATCH_SIZE))
        mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        if index % 2 == 0:
            mask[0:2, 6:8] = 1
        truth = mask > 0
        truths[patch_id] = truth
        image_path = f"data/processed/patches/{TEST_SOURCE}/images/{patch_id}.npy"
        mask_path = f"data/processed/patches/{TEST_SOURCE}/masks/{patch_id}.npy"
        for relative in (image_path, mask_path):
            (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp_path / image_path, image.astype(np.float32))
        np.save(tmp_path / mask_path, mask)
        records.append(
            {
                "patch_id": patch_id,
                "tile_id": "tile_310044_2018",
                "fold": index % FOLD_COUNT,
                "row": index,
                "col": 0,
                "bbox": f"{index * 8},0,{index * 8 + 8},8",
                "coffee_ratio": float(truth.mean()),
                "valid_fraction": 1.0,
                "mask_source": TEST_SOURCE,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    manifest = pd.DataFrame(records)
    manifest_file = manifest_path(paths, TEST_SOURCE)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_file, index=False)

    fingerprint = {
        "manifest_fingerprint_hash": patch_fingerprint_hash(paths, TEST_SOURCE),
        "split_fingerprint_hash": split_fingerprint_hash(paths, TEST_SOURCE),
        "config_hash": config_hash(),
        "threshold": THRESHOLD,
    }
    for model_name in MODEL_NAMES:
        frames: dict[int, pd.DataFrame] = {}
        for fold in range(FOLD_COUNT):
            rows = []
            fold_records = manifest[manifest["fold"] == fold]
            predictions = []
            for record in fold_records.itertuples():
                truth = truths[str(record.patch_id)]
                prediction = _predictions(
                    int(str(record.patch_id).split("_")[1]), truth, model_name
                )
                counts = _counts(prediction, truth)
                row = {
                    "patch_id": str(record.patch_id),
                    "fold": fold,
                    "coffee_ratio": float(record.coffee_ratio),
                    "valid_fraction": 1.0,
                    **counts,
                    **_metrics(counts),
                }
                rows.append(row)
                predictions.append(prediction)
            frame = pd.DataFrame(rows)
            frames[fold] = frame
            patch_file = evaluation.patch_metrics_path(paths, model_name, TEST_SOURCE, fold)
            patch_file.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(patch_file, index=False)
            prediction_file = evaluation.predictions_path(paths, model_name, TEST_SOURCE, fold)
            np.savez_compressed(
                prediction_file,
                patch_ids=np.asarray(fold_records["patch_id"].tolist(), dtype=str),
                predictions=np.stack(predictions),
            )
        _write_summary(paths, model_name, TEST_SOURCE, frames, fingerprint)
    return paths, manifest, fingerprint


def test_artifact_paths_follow_schema(tmp_path: Path) -> None:
    """A comparação deve residir em artifacts/metrics/comparative/<fonte>/."""
    paths = _storage_paths(tmp_path)
    assert str(comparative.comparison_path(paths, TEST_SOURCE)).endswith(
        "artifacts/metrics/comparative/alphaearth/comparison.json"
    )
    assert str(comparative.comparison_figure_path(paths, TEST_SOURCE)).endswith(".png")
    assert str(comparative.error_maps_figure_path(paths, TEST_SOURCE)).endswith(".png")


def test_load_aligned_patch_metrics_pairs_by_patch_id(tmp_path, monkeypatch) -> None:
    """As métricas dos dois modelos devem ser emparelhadas pelo `patch_id`."""
    paths, manifest, _ = _write_environment(tmp_path, monkeypatch)
    summaries = comparative.load_evaluation_summaries(paths, TEST_SOURCE)

    aligned = comparative.load_aligned_patch_metrics(paths, TEST_SOURCE, summaries)

    assert len(aligned) == len(manifest)
    assert aligned["patch_id"].is_unique
    assert set(aligned["patch_id"]) == set(manifest["patch_id"])
    assert {"iou_unet", "iou_segformer", "truth_positive", "fold"} <= set(aligned.columns)
    assert aligned["truth_positive"].all()


def test_analyze_comparison_persists_and_reuses(tmp_path, monkeypatch) -> None:
    """A comparação deve ser persistida e reutilizada enquanto o fingerprint coincide."""
    paths, _, _ = _write_environment(tmp_path, monkeypatch)

    first = comparative.analyze_comparison(paths, TEST_SOURCE)
    assert not first.skipped
    assert first.summary_path.is_file()
    assert set(first.payload["paired"]) == set(comparative.METRIC_KEYS)
    mtime = first.summary_path.stat().st_mtime_ns

    second = comparative.analyze_comparison(paths, TEST_SOURCE)
    assert second.skipped
    assert second.summary_path.stat().st_mtime_ns == mtime


def test_analyze_comparison_invalidates_on_weights_change(tmp_path, monkeypatch) -> None:
    """Mudanças no hash dos pesos do estágio 11 devem invalidar a comparação."""
    paths, _, _ = _write_environment(tmp_path, monkeypatch)
    comparative.analyze_comparison(paths, TEST_SOURCE)

    summary_path = evaluation.evaluation_summary_path(paths, "unet", TEST_SOURCE)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["per_fold"][0]["weights_sha256"] = "novo_peso"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    result = comparative.analyze_comparison(paths, TEST_SOURCE)
    assert not result.skipped
    assert result.payload["models"]["unet"]["per_fold"][0]["fold"] == 0


def test_summaries_are_current_requires_predictions(tmp_path, monkeypatch) -> None:
    """A ausência de predições do estágio 11 deve invalidar os resumos."""
    paths, _, _ = _write_environment(tmp_path, monkeypatch)
    summaries = comparative.load_evaluation_summaries(paths, TEST_SOURCE)
    assert comparative.summaries_are_current(paths, TEST_SOURCE, summaries)

    evaluation.predictions_path(paths, "unet", TEST_SOURCE, 0).unlink()
    assert not comparative.summaries_are_current(paths, TEST_SOURCE, summaries)


def test_paired_statistics_reports_difference() -> None:
    """O teste emparelhado deve detectar diferença sistemática entre os modelos."""
    base = np.linspace(0.2, 0.8, 20)
    segformer = base + np.linspace(0.01, 0.09, 20)

    stats = comparative.paired_metric_statistics(segformer, base, "iou", seed=1)

    assert stats["n_pairs"] == 20
    assert stats["mean_difference"] == pytest.approx(0.05)
    assert stats["segformer_better"] == 20
    assert stats["unet_better"] == 0
    assert stats["ci95_low"] <= stats["mean_difference"] <= stats["ci95_high"]
    assert stats["wilcoxon_pvalue"] is not None and stats["wilcoxon_pvalue"] < 0.05
    assert stats["t_pvalue"] is not None and stats["t_pvalue"] < 0.05


def test_holm_bonferroni_is_monotonic() -> None:
    """A correção de Holm deve preservar a ordem e limitar os valores a 1.0."""
    adjusted = comparative.holm_bonferroni({"a": 0.01, "b": 0.04, "c": None, "d": 0.9})
    a, b, d = adjusted["a"], adjusted["b"], adjusted["d"]
    assert adjusted["c"] is None
    assert a is not None and b is not None and d is not None
    assert a == pytest.approx(0.03)
    assert b == pytest.approx(0.08)
    assert d == pytest.approx(0.9)
    assert a <= b <= d


def test_effect_sizes() -> None:
    """As medidas de efeito devem refletir diferenças e empates."""
    base = np.array([0.2, 0.4, 0.6, 0.8])
    segformer = base + 1.0
    assert comparative.cohens_dz(segformer - base) is None
    assert comparative.cliffs_delta(segformer, base) == pytest.approx(1.0)
    assert comparative.cohens_dz(np.array([0.1, 0.2, 0.3])) is not None


def test_bootstrap_mean_ci_is_deterministic() -> None:
    """O bootstrap com a mesma semente deve produzir o mesmo intervalo."""
    values = np.linspace(-0.2, 0.3, 50)
    first = comparative.bootstrap_mean_ci(values, iterations=200, seed=7)
    second = comparative.bootstrap_mean_ci(values, iterations=200, seed=7)
    assert first == second
    assert first[0] is not None and first[1] is not None
    assert first[0] <= np.mean(values) <= first[1]


def test_select_error_map_patches_is_deterministic(tmp_path, monkeypatch) -> None:
    """A seleção de patches para os mapas de erro deve ser única e determinística."""
    paths, _, _ = _write_environment(tmp_path, monkeypatch)
    summaries = comparative.load_evaluation_summaries(paths, TEST_SOURCE)
    aligned = comparative.load_aligned_patch_metrics(paths, TEST_SOURCE, summaries)

    first = comparative.select_error_map_patches(aligned)
    second = comparative.select_error_map_patches(aligned)

    assert first == second
    assert len(first) == 4
    assert len({item["patch_id"] for item in first}) == 4
    assert all(item["label"] for item in first)


def test_error_code_map_encodes_confusion() -> None:
    """O mapa de erro deve codificar TP/FP/FN/TN e ignorar pixels inválidos."""
    truth = np.array([[1, 1, 0, 0]], dtype=bool)
    prediction = np.array([[1, 0, 1, 0]], dtype=bool)
    valid = np.array([[True, True, True, False]])

    codes = comparative.error_code_map(prediction, truth, valid)

    assert codes.tolist() == [
        [
            comparative.TP_CODE,
            comparative.FN_CODE,
            comparative.FP_CODE,
            comparative.INVALID_CODE,
        ]
    ]


def test_render_and_save_error_maps(tmp_path, monkeypatch) -> None:
    """Os mapas de erro devem ser renderizados e persistidos como PNG."""
    paths, _, _ = _write_environment(tmp_path, monkeypatch)
    summaries = comparative.load_evaluation_summaries(paths, TEST_SOURCE)
    aligned = comparative.load_aligned_patch_metrics(paths, TEST_SOURCE, summaries)
    selected = comparative.select_error_map_patches(aligned)

    samples = comparative.build_error_map_samples(paths, TEST_SOURCE, selected)
    png = comparative.render_error_maps_figure(samples)
    assert png.startswith(b"\x89PNG")

    figure = comparative.save_error_maps_figure(paths, TEST_SOURCE, selected)
    assert figure.is_file()
    assert figure.read_bytes().startswith(b"\x89PNG")


def test_save_comparison_figure(tmp_path, monkeypatch) -> None:
    """A figura comparativa deve ser persistida como PNG."""
    paths, _, _ = _write_environment(tmp_path, monkeypatch)
    analysis = comparative.analyze_comparison(paths, TEST_SOURCE)

    figure = comparative.save_comparison_figure(
        paths, TEST_SOURCE, analysis.payload, analysis.patch_metrics
    )

    assert figure.is_file()
    assert figure.read_bytes().startswith(b"\x89PNG")
