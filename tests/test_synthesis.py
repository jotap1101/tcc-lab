"""Testes de src/synthesis.py (síntese dos resultados — estágio 15)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import comparative, evaluation, synthesis, trainer
from src.data.patch_generation import manifest_path
from src.metrics import f1_score, iou, precision, recall
from src.xai import attention_rollout, gradcam

TEST_SOURCE = "emater"
PATCH_SIZE = 16
FOLD_COUNT = 5


def _metrics(counts: dict[str, int]) -> dict[str, float]:
    """Deriva as métricas de pixel a partir das contagens de confusão."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "iou": iou(tp, fp, fn),
        "f1": f1_score(tp, fp, fn),
        "precision": precision(tp, fp),
        "recall": recall(tp, fn),
    }


def _storage_paths(tmp_path: Path) -> dict[str, Path]:
    """Caminhos de armazenamento mínimos para os testes do estágio 15."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "models": tmp_path / "models",
        "artifacts_metrics": tmp_path / "artifacts" / "metrics",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
        "artifacts_runs": tmp_path / "artifacts" / "runs",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Escreve um JSON auxiliar, criando o diretório pai."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _summary(
    model_name: str, source: str, frames: dict[int, pd.DataFrame], fingerprint: dict[str, Any]
) -> dict[str, Any]:
    """Monta o resumo sintético do estágio 11 para um modelo."""
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
    values = {key: [item["metrics"][key] for item in per_fold] for key in evaluation.METRIC_KEYS}
    return {
        "schema_version": 1,
        "model": model_name,
        "source_name": source,
        "threshold": evaluation.DEFAULT_THRESHOLD,
        "n_folds": len(per_fold),
        "n_patches": int(sum(item["n_patches"] for item in per_fold)),
        "per_fold": per_fold,
        "aggregate": {"counts": total, "metrics": _metrics(total)},
        "fold_mean": {key: float(np.mean(value)) for key, value in values.items()},
        "fold_std": {key: float(np.std(value)) for key, value in values.items()},
        "fingerprint": fingerprint,
    }


def _write_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_xai: bool = False
) -> dict[str, Path]:
    """Escreve manifesto, treinos, avaliações, comparação e XAI sintéticos."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    stats_payload = {
        "bands": ["B2", "B3", "B4", "B8"],
        "per_band": {
            band: {"mean": 0.0, "std": 1.0, "nan_fraction": 0.0}
            for band in ["B2", "B3", "B4", "B8"]
        },
    }
    stats_path = paths["data_processed"] / f"normalization_stats_{TEST_SOURCE}.json"
    _write_json(stats_path, stats_payload)

    records = []
    model_frames: dict[str, dict[int, pd.DataFrame]] = {
        model_name: {} for model_name in synthesis.MODEL_ORDER
    }
    truth = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    truth[3:12, 4:13] = True
    for fold in range(FOLD_COUNT):
        patch_id = f"patch_{fold:03d}"
        image = np.random.default_rng(fold).uniform(0.0, 1.0, size=(4, PATCH_SIZE, PATCH_SIZE))
        mask = truth.astype(np.uint8)
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
                "fold": fold,
                "row": fold,
                "col": 0,
                "bbox": f"{fold * PATCH_SIZE},0,{(fold + 1) * PATCH_SIZE},{PATCH_SIZE}",
                "coffee_ratio": float(truth.mean()),
                "valid_fraction": 1.0,
                "mask_source": TEST_SOURCE,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
        for model_name in synthesis.MODEL_ORDER:
            prediction = truth.copy()
            if model_name == "unet":
                prediction[0, fold] = True
            else:
                prediction[6, 6] = False
            counts = {
                "tp": int(np.count_nonzero(prediction & truth)),
                "fp": int(np.count_nonzero(prediction & ~truth)),
                "fn": int(np.count_nonzero(~prediction & truth)),
            }
            frame = pd.DataFrame(
                [
                    {
                        "patch_id": patch_id,
                        "fold": fold,
                        "coffee_ratio": float(truth.mean()),
                        "valid_fraction": 1.0,
                        **counts,
                        **_metrics(counts),
                    }
                ]
            )
            model_frames[model_name][fold] = frame
            patch_file = evaluation.patch_metrics_path(paths, model_name, TEST_SOURCE, fold)
            patch_file.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(patch_file, index=False)
            prediction_file = evaluation.predictions_path(paths, model_name, TEST_SOURCE, fold)
            np.savez_compressed(
                prediction_file,
                patch_ids=np.asarray([patch_id], dtype=str),
                predictions=np.stack([prediction.astype(np.uint8)]),
            )

    manifest = pd.DataFrame(records)
    manifest_file = manifest_path(paths, TEST_SOURCE)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_file, index=False)

    summaries: dict[str, dict[str, Any]] = {}
    fingerprint = evaluation.summary_fingerprint(paths, TEST_SOURCE)
    for model_name in synthesis.MODEL_ORDER:
        summaries[model_name] = _summary(
            model_name, TEST_SOURCE, model_frames[model_name], fingerprint
        )
        _write_json(
            evaluation.evaluation_summary_path(paths, model_name, TEST_SOURCE),
            summaries[model_name],
        )

    for model_name in synthesis.MODEL_ORDER:
        for fold in range(FOLD_COUNT):
            metrics_payload = {
                "model": model_name,
                "fold": fold,
                "epochs": 2,
                "seed": fold,
                "best_epoch": 1,
                "manifest_fingerprint_hash": fingerprint["manifest_fingerprint_hash"],
                "split_fingerprint_hash": fingerprint["split_fingerprint_hash"],
                "config_hash": fingerprint["config_hash"],
                "train": {
                    "loss": 1.0,
                    "iou": 0.5,
                    "f1": 0.6,
                    "precision": 0.7,
                    "recall": 0.8,
                },
                "val": {
                    "loss": 0.9,
                    "iou": 0.4,
                    "f1": 0.5,
                    "precision": 0.6,
                    "recall": 0.7,
                },
            }
            _write_json(trainer.metrics_path(paths, model_name, TEST_SOURCE, fold), metrics_payload)
            _write_json(
                trainer.history_path(paths, model_name, TEST_SOURCE, fold),
                {
                    "model": model_name,
                    "fold": fold,
                    "seed": fold,
                    "epochs": [
                        {
                            "epoch": 1,
                            "lr": 1e-4,
                            "train_loss": 1.0,
                            "val_loss": 0.9,
                            "val_iou": 0.4,
                            "val_f1": 0.5,
                        },
                        {
                            "epoch": 2,
                            "lr": 9e-5,
                            "train_loss": 0.8,
                            "val_loss": 0.7,
                            "val_iou": 0.5,
                            "val_f1": 0.6,
                        },
                    ],
                },
            )
            weights = trainer.weights_path(paths, model_name, TEST_SOURCE, fold)
            weights.parent.mkdir(parents=True, exist_ok=True)
            weights.write_bytes(b"weights")

    aligned = comparative.load_aligned_patch_metrics(paths, TEST_SOURCE, summaries)
    comparison = comparative.build_comparison_payload(
        aligned,
        summaries,
        TEST_SOURCE,
        comparative.comparison_fingerprint(summaries, TEST_SOURCE),
    )
    _write_json(comparative.comparison_path(paths, TEST_SOURCE), comparison)

    if with_xai:
        _write_xai(paths, manifest, summaries)
    return paths


def _write_xai(
    paths: dict[str, Path], manifest: pd.DataFrame, summaries: dict[str, dict[str, Any]]
) -> None:
    """Escreve metadados e mapas sintéticos de Grad-CAM e Attention Rollout."""
    for model_name in synthesis.MODEL_ORDER:
        selected = [manifest.iloc[0], manifest.iloc[1]]
        metadata_patches = []
        heatmaps = []
        for record in selected:
            fold = int(record["fold"])
            weights = {
                int(item["fold"]): str(item["weights_sha256"])
                for item in summaries[model_name]["per_fold"]
            }[fold]
            metadata_patches.append(
                {
                    "patch_id": str(record["patch_id"]),
                    "label": "Maior IoU",
                    "fold": fold,
                    "iou": 0.5,
                    "coffee_ratio": float(record["coffee_ratio"]),
                    "weights_sha256": weights,
                }
            )
            heatmap = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            root = paths["data_processed_patches"].parents[2]
            mask = np.load(root / str(record["mask_path"]))
            heatmap[mask > 0] = 1.0
            heatmaps.append(heatmap)
        fingerprint = {
            "schema_version": 1,
            "model": model_name,
            "source_name": TEST_SOURCE,
            "manifest_fingerprint_hash": summaries[model_name]["fingerprint"][
                "manifest_fingerprint_hash"
            ],
            "split_fingerprint_hash": summaries[model_name]["fingerprint"][
                "split_fingerprint_hash"
            ],
            "config_hash": summaries[model_name]["fingerprint"]["config_hash"],
            "threshold": summaries[model_name]["fingerprint"]["threshold"],
        }
        metadata = {
            "schema_version": 1,
            "model": model_name,
            "source_name": TEST_SOURCE,
            "n_patches": len(metadata_patches),
            "patches": metadata_patches,
            "fingerprint": fingerprint,
            "rollout": {"stage_index": 3},
            "target_layer": "encoders[-1]",
        }
        if model_name == "unet":
            metadata_file = gradcam.gradcam_metadata_path(paths, TEST_SOURCE)
            maps_file = gradcam.gradcam_maps_path(paths, TEST_SOURCE)
            key = "cams"
        else:
            metadata_file = attention_rollout.attention_rollout_metadata_path(paths, TEST_SOURCE)
            maps_file = attention_rollout.attention_rollout_maps_path(paths, TEST_SOURCE)
            key = "rollouts"
        _write_json(metadata_file, metadata)
        maps_file.parent.mkdir(parents=True, exist_ok=True)
        patch_ids = np.asarray([item["patch_id"] for item in metadata_patches], dtype=str)
        folds = np.asarray([item["fold"] for item in metadata_patches], dtype=np.int64)
        if key == "cams":
            np.savez_compressed(
                maps_file, patch_ids=patch_ids, folds=folds, cams=np.stack(heatmaps)
            )
        else:
            np.savez_compressed(
                maps_file, patch_ids=patch_ids, folds=folds, rollouts=np.stack(heatmaps)
            )


def test_run_synthesis_persists_tables_and_figures(tmp_path, monkeypatch) -> None:
    """A síntese deve gerar tabelas CSV/LaTeX, figuras e resumo sem XAI."""
    paths = _write_environment(tmp_path, monkeypatch)

    result = synthesis.run_synthesis(paths, TEST_SOURCE)

    assert not result.skipped
    assert set(result.tables) == set(synthesis.CORE_TABLES)
    assert set(result.figures) == {"training_curves", "evaluation_overview"}
    for path in list(result.tables.values()) + list(result.figures.values()):
        assert path.is_file()
    for name in synthesis.CORE_TABLES:
        assert synthesis.table_path(paths, TEST_SOURCE, name, "tex").is_file()
    assert result.figures["training_curves"].read_bytes().startswith(b"\x89PNG")
    assert result.tables_data["evaluation_aggregate"].shape[0] == 2
    assert result.tables_data["comparative_paired"].shape[0] == 4
    assert result.summary["xai"]["included"] is False
    assert result.summary["n_patches_chosen"] == FOLD_COUNT
    assert result.metadata_path.is_file()

    mtime = result.metadata_path.stat().st_mtime_ns
    reused = synthesis.run_synthesis(paths, TEST_SOURCE)
    assert reused.skipped
    assert reused.metadata_path.stat().st_mtime_ns == mtime
    assert reused.tables_data["dataset"].equals(result.tables_data["dataset"])


def test_run_synthesis_includes_xai_relevance(tmp_path, monkeypatch) -> None:
    """A síntese deve calcular relevância dentro/fora do café quando há XAI."""
    paths = _write_environment(tmp_path, monkeypatch, with_xai=True)

    result = synthesis.run_synthesis(paths, TEST_SOURCE)

    assert result.summary["xai"]["included"] is True
    assert set(result.summary["xai"]["models"]) == set(synthesis.MODEL_ORDER)
    assert "xai" in result.tables
    assert "xai_relevance" in result.figures
    xai_table = result.tables_data["xai"]
    assert len(xai_table) == 4
    assert (xai_table["relevance_inside"] > xai_table["relevance_outside"]).all()
    assert xai_table["relevance_difference"].min() == pytest.approx(1.0)
    assert result.figures["xai_relevance"].read_bytes().startswith(b"\x89PNG")


def test_run_synthesis_requires_current_comparison(tmp_path, monkeypatch) -> None:
    """Sem a comparação vigente do estágio 12, a síntese deve falhar claramente."""
    paths = _write_environment(tmp_path, monkeypatch)
    comparative.comparison_path(paths, TEST_SOURCE).unlink()

    with pytest.raises(FileNotFoundError, match="estágio 12"):
        synthesis.run_synthesis(paths, TEST_SOURCE)


def test_run_synthesis_rejects_stale_xai(tmp_path, monkeypatch) -> None:
    """Metadados de XAI com pesos divergentes devem ser rejeitados."""
    paths = _write_environment(tmp_path, monkeypatch, with_xai=True)
    metadata_file = gradcam.gradcam_metadata_path(paths, TEST_SOURCE)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata["patches"][0]["weights_sha256"] = "stale"
    _write_json(metadata_file, metadata)

    with pytest.raises(ValueError, match="estágio 13/14"):
        synthesis.run_synthesis(paths, TEST_SOURCE)


def test_render_xai_relevance_returns_png(tmp_path, monkeypatch) -> None:
    """A figura quantitativa de XAI deve ser renderizada como PNG."""
    paths = _write_environment(tmp_path, monkeypatch, with_xai=True)
    _, _, _, xai_models = synthesis.validate_inputs(paths, TEST_SOURCE)
    table = synthesis.build_xai_table(paths, TEST_SOURCE, xai_models)

    png = synthesis.render_xai_relevance(table)

    assert png.startswith(b"\x89PNG")
