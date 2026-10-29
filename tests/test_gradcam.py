"""Testes de src/xai/gradcam.py e src/xai/common.py (Grad-CAM do U-Net — estágio 13)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src import evaluation
from src.data.patch_generation import manifest_path
from src.metrics import f1_score, iou, precision, recall
from src.models.unet import UNet
from src.xai import common, gradcam

TEST_SOURCE = "alphaearth"
PATCH_SIZE = 16
FOLD_COUNT = 5


def _storage_paths(tmp_path: Path) -> dict[str, Path]:
    """Caminhos de armazenamento mínimos para os testes do estágio 13."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "models": tmp_path / "models",
        "artifacts_metrics": tmp_path / "artifacts" / "metrics",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
        "artifacts_xai": tmp_path / "artifacts" / "xai",
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


def _summary(
    source: str, frames: dict[int, pd.DataFrame], fingerprint: dict[str, Any]
) -> dict[str, Any]:
    """Monta o resumo sintético do estágio 11 para a U-Net."""
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
                "weights_sha256": f"unet_{fold:02d}",
            }
        )
    values = {key: [item["metrics"][key] for item in per_fold] for key in ("iou", "f1")}
    return {
        "schema_version": 1,
        "model": "unet",
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


def _write_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Escreve manifesto, patches, estatísticas e artefatos sintéticos do estágio 11."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    records = []
    frames: dict[int, pd.DataFrame] = {}
    stats_payload = {
        "bands": ["B2", "B3", "B4", "B8"],
        "per_band": {
            band: {"mean": 0.0, "std": 1.0, "nan_fraction": 0.0}
            for band in ["B2", "B3", "B4", "B8"]
        },
    }
    stats_path = paths["data_processed"] / f"normalization_stats_{TEST_SOURCE}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats_payload), encoding="utf-8")

    for fold in range(FOLD_COUNT):
        patch_id = f"patch_{fold:03d}"
        image = np.random.default_rng(fold).uniform(0.0, 1.0, size=(4, PATCH_SIZE, PATCH_SIZE))
        mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        mask[3:12, 4:13] = 1
        truth = mask > 0
        prediction = truth.copy()
        prediction[0, fold] = True
        if fold % 2 == 0:
            prediction[6, 6] = False
        tp = int(np.count_nonzero(prediction & truth))
        fp = int(np.count_nonzero(prediction & ~truth))
        fn = int(np.count_nonzero(~prediction & truth))
        counts = {"tp": tp, "fp": fp, "fn": fn}
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
        rows = [
            {
                "patch_id": patch_id,
                "fold": fold,
                "coffee_ratio": float(truth.mean()),
                "valid_fraction": 1.0,
                **counts,
                **_metrics(counts),
            }
        ]
        frame = pd.DataFrame(rows)
        frames[fold] = frame
        patch_file = evaluation.patch_metrics_path(paths, "unet", TEST_SOURCE, fold)
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(patch_file, index=False)
        prediction_file = evaluation.predictions_path(paths, "unet", TEST_SOURCE, fold)
        np.savez_compressed(
            prediction_file,
            patch_ids=np.asarray([patch_id], dtype=str),
            predictions=np.stack([prediction.astype(np.uint8)]),
        )

    manifest = pd.DataFrame(records)
    manifest_file = manifest_path(paths, TEST_SOURCE)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_file, index=False)
    fingerprint = evaluation.summary_fingerprint(paths, TEST_SOURCE)
    summary_file = evaluation.evaluation_summary_path(paths, "unet", TEST_SOURCE)
    summary_file.write_text(
        json.dumps(_summary(TEST_SOURCE, frames, fingerprint)), encoding="utf-8"
    )
    return paths


def _small_model() -> UNet:
    """U-Net pequena compatível com patches de 16x16."""
    return UNet(in_channels=4, channels=[4, 8], out_channels=1)


def test_default_target_layer_returns_deepest_encoder() -> None:
    """A camada-alvo padrão deve ser o último bloco do codificador."""
    model = _small_model()
    assert gradcam.default_target_layer(model) is model.encoders[-1]


def test_compute_gradcam_shape_and_range() -> None:
    """O Grad-CAM deve ter a resolução da entrada e valores em [0, 1]."""
    torch.manual_seed(0)
    conv = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.abs_()
    target_layer = conv
    model = nn.Sequential(conv, nn.ReLU())
    image = torch.rand(1, 4, PATCH_SIZE, PATCH_SIZE)
    target = torch.zeros(PATCH_SIZE, PATCH_SIZE, dtype=torch.bool)
    target[4:10, 4:10] = True

    cam = gradcam.compute_gradcam(model, image, target_mask=target, target_layer=target_layer)

    assert cam.shape == (PATCH_SIZE, PATCH_SIZE)
    assert cam.dtype == np.float32
    assert 0.0 <= float(cam.min()) <= float(cam.max()) <= 1.0
    assert float(cam.max()) > 0.0


def test_compute_gradcam_without_mask() -> None:
    """Sem máscara, o Grad-CAM deve usar todos os pixels e manter a forma."""
    torch.manual_seed(1)
    model = _small_model().eval()
    image = torch.rand(4, PATCH_SIZE, PATCH_SIZE)

    cam = gradcam.compute_gradcam(model, image)

    assert cam.shape == (PATCH_SIZE, PATCH_SIZE)


def test_select_representative_patches_is_deterministic(tmp_path, monkeypatch) -> None:
    """A seleção deve ser única, determinística e priorizar patches com café."""
    paths = _write_environment(tmp_path, monkeypatch)
    patch_metrics = evaluation.load_model_patch_metrics(paths, "unet", TEST_SOURCE)

    first = common.select_representative_patches(patch_metrics)
    second = common.select_representative_patches(patch_metrics)

    assert first == second
    assert len(first) == 4
    assert len({item["patch_id"] for item in first}) == 4


def test_load_patch_sample(tmp_path, monkeypatch) -> None:
    """O carregamento deve normalizar a imagem e derivar a referência válida."""
    paths = _write_environment(tmp_path, monkeypatch)
    stats, manifest = common.load_stats_and_manifest(paths, TEST_SOURCE)

    sample = common.load_patch_sample(paths, manifest, "patch_000", stats)

    assert sample.image.shape == (4, PATCH_SIZE, PATCH_SIZE)
    assert sample.normalized.shape == (4, PATCH_SIZE, PATCH_SIZE)
    assert sample.reference.any()
    assert not (sample.reference & ~sample.valid).any()
    assert common.patch_rgb(sample).shape == (PATCH_SIZE, PATCH_SIZE, 3)


def test_run_gradcam_persists_and_reuses(tmp_path, monkeypatch) -> None:
    """O estágio 13 deve persistir mapas/metadata/figura e reutilizar quando vigente."""
    paths = _write_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        gradcam,
        "load_model",
        lambda model_name, storage, source, fold, device: _small_model().to(device).eval(),
    )

    first = gradcam.run_gradcam(paths, TEST_SOURCE, device=torch.device("cpu"))
    assert not first.skipped
    assert len(first.samples) == 4
    assert first.maps_path.is_file()
    assert first.metadata_path.is_file()
    assert first.figure_path.is_file()
    assert first.figure_path.read_bytes().startswith(b"\x89PNG")
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["model"] == "unet"
    assert metadata["n_patches"] == 4
    assert metadata["fingerprint"]["target_layer"] == gradcam.TARGET_LAYER_NAME
    with np.load(first.maps_path) as data:
        assert data["cams"].shape == (4, PATCH_SIZE, PATCH_SIZE)
        assert len(data["patch_ids"]) == 4

    mtime = first.metadata_path.stat().st_mtime_ns
    second = gradcam.run_gradcam(paths, TEST_SOURCE, device=torch.device("cpu"))
    assert second.skipped
    assert second.samples == []
    assert second.metadata_path.stat().st_mtime_ns == mtime


def test_run_gradcam_requires_current_summary(tmp_path, monkeypatch) -> None:
    """Sem resumo vigente do estágio 11, a geração deve falhar com mensagem clara."""
    paths = _write_environment(tmp_path, monkeypatch)
    evaluation.evaluation_summary_path(paths, "unet", TEST_SOURCE).unlink()

    with pytest.raises(ValueError, match="estágio 11"):
        gradcam.run_gradcam(paths, TEST_SOURCE, device=torch.device("cpu"))


def test_render_gradcam_figure_returns_png(tmp_path, monkeypatch) -> None:
    """A renderização da figura Grad-CAM deve devolver bytes PNG."""
    paths = _write_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        gradcam,
        "load_model",
        lambda model_name, storage, source, fold, device: _small_model().to(device).eval(),
    )
    result = gradcam.run_gradcam(paths, TEST_SOURCE, device=torch.device("cpu"))

    png = gradcam.render_gradcam_figure(result.samples)

    assert png.startswith(b"\x89PNG")


def test_gradcam_dir_derives_from_artifacts_without_config_key(tmp_path: Path) -> None:
    """Sem a chave dedicada, a raiz de XAI deve ser derivada de artifacts/metrics."""
    paths = _storage_paths(tmp_path)
    paths.pop("artifacts_xai")

    target = gradcam.gradcam_dir(paths, TEST_SOURCE)

    assert target == paths["artifacts_metrics"].parent / "xai" / "gradcam" / "unet" / TEST_SOURCE
