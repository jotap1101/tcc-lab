"""Testes de src/xai/attention_rollout.py (Attention Rollout do SegFormer — estágio 14)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src import evaluation
from src.data.patch_generation import manifest_path
from src.metrics import f1_score, iou, precision, recall
from src.xai import attention_rollout

TEST_SOURCE = "alphaearth"
PATCH_SIZE = 16
FOLD_COUNT = 5
STAGE_GRID = 4


class _FakeOutput:
    """Saída mínima do backbone com atenções."""

    def __init__(self, attentions: tuple[torch.Tensor, ...]) -> None:
        self.attentions = attentions


class _FakeBackbone(nn.Module):
    """Backbone sintético com atenções não quadradas nas etapas iniciais."""

    def __init__(self, attentions: tuple[torch.Tensor, ...]) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            depths=(1, 1, 1, 2),
            sr_ratios=(8, 4, 2, 1),
            _attn_implementation="sdpa",
        )
        self.attentions = attentions

    def set_attn_implementation(self, implementation: str) -> None:
        """Registra a implementação de atenção selecionada."""
        self.config._attn_implementation = implementation

    def forward(self, pixel_values: torch.Tensor, output_attentions: bool) -> _FakeOutput:
        """Retorna as atenções sintéticas, independentemente da entrada."""
        del pixel_values, output_attentions
        return _FakeOutput(self.attentions)


class _FakeModel(nn.Module):
    """Modelo no formato do SegFormerBinary usado pelo rollout."""

    def __init__(self, attentions: tuple[torch.Tensor, ...]) -> None:
        super().__init__()
        self.model = _FakeBackbone(attentions)


def _fake_attentions(grid: int = STAGE_GRID) -> tuple[torch.Tensor, ...]:
    """Atenções sintéticas: três etapas não quadradas e duas quadradas finais."""
    tokens = grid * grid
    generator = torch.Generator().manual_seed(7)
    final = torch.softmax(torch.randn(1, 2, tokens, tokens, generator=generator), dim=-1)
    return (
        torch.rand(1, 1, 8, 2),
        torch.rand(1, 1, 8, 2),
        torch.rand(1, 1, 8, 2),
        final,
        final,
    )


def _fake_model(grid: int = STAGE_GRID) -> _FakeModel:
    """Modelo sintético compatível com o rollout da última etapa."""
    return _FakeModel(_fake_attentions(grid))


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
    """Caminhos de armazenamento mínimos para os testes do estágio 14."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "models": tmp_path / "models",
        "artifacts_metrics": tmp_path / "artifacts" / "metrics",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
    }


def _summary(
    source: str, frames: dict[int, pd.DataFrame], fingerprint: dict[str, Any]
) -> dict[str, Any]:
    """Monta o resumo sintético do estágio 11 para o SegFormer."""
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
                "weights_sha256": f"segformer_{fold:02d}",
            }
        )
    values = {key: [item["metrics"][key] for item in per_fold] for key in ("iou", "f1")}
    return {
        "schema_version": 1,
        "model": "segformer",
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
        counts = {
            "tp": int(np.count_nonzero(prediction & truth)),
            "fp": int(np.count_nonzero(prediction & ~truth)),
            "fn": int(np.count_nonzero(~prediction & truth)),
        }
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
        frames[fold] = frame
        patch_file = evaluation.patch_metrics_path(paths, "segformer", TEST_SOURCE, fold)
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(patch_file, index=False)
        prediction_file = evaluation.predictions_path(paths, "segformer", TEST_SOURCE, fold)
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
    summary_file = evaluation.evaluation_summary_path(paths, "segformer", TEST_SOURCE)
    summary_file.write_text(
        json.dumps(_summary(TEST_SOURCE, frames, fingerprint)), encoding="utf-8"
    )
    return paths


def test_evaluate_rollout_attentions_identity_is_uniform() -> None:
    """Atenções identidade com resíduo devem produzir relevância uniforme."""
    identity = torch.eye(4).repeat(1, 1, 1)

    relevance = attention_rollout.evaluate_rollout_attentions((identity, identity))

    assert relevance.shape == (4,)
    assert torch.allclose(relevance, torch.full((4,), 0.25))
    assert float(relevance.sum()) == pytest.approx(1.0)


def test_evaluate_rollout_attentions_rejects_non_square() -> None:
    """Atenções não quadradas devem ser rejeitadas com mensagem clara."""
    with pytest.raises(ValueError, match="N, N"):
        attention_rollout.evaluate_rollout_attentions((torch.rand(1, 8, 2),))


def test_last_stage_attentions_selects_last_stage() -> None:
    """A seleção deve considerar apenas as camadas da última etapa."""
    attentions = (
        torch.rand(1, 1, 8, 2),
        torch.rand(1, 1, 8, 2),
        torch.rand(1, 1, 8, 2),
        torch.rand(1, 2, 4, 4),
        torch.rand(1, 2, 4, 4),
    )

    selected = attention_rollout.last_stage_attentions(attentions, depths=(2, 1, 2))

    assert selected == list(attentions[-2:])


def test_compute_attention_rollout_uses_last_stage() -> None:
    """O mapa deve usar a última etapa, ter a resolução da entrada e [0, 1]."""
    model = _fake_model()

    rollout, info = attention_rollout.compute_attention_rollout(
        model, torch.rand(1, 4, PATCH_SIZE, PATCH_SIZE)
    )

    assert rollout.shape == (PATCH_SIZE, PATCH_SIZE)
    assert rollout.dtype == np.float32
    assert 0.0 <= float(rollout.min()) <= float(rollout.max()) <= 1.0
    assert info["rollout_scope"] == attention_rollout.ROLLOUT_SCOPE
    assert info["stage_grid"] == [STAGE_GRID, STAGE_GRID]
    assert info["stage_layers"] == 2
    assert info["stage_index"] == 3
    assert info["num_heads"] == 2
    assert info["attention_implementation"] == "eager"


def test_compute_attention_rollout_requires_attentions() -> None:
    """Sem atenções retornadas, o cálculo deve falhar com orientação clara."""
    model = _FakeModel(())

    with pytest.raises(RuntimeError, match="output_attentions"):
        attention_rollout.compute_attention_rollout(model, torch.rand(1, 4, PATCH_SIZE, PATCH_SIZE))


def test_run_attention_rollout_persists_and_reuses(tmp_path, monkeypatch) -> None:
    """O estágio 14 deve persistir mapas/metadata/figura e reutilizar quando vigente."""
    paths = _write_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        attention_rollout,
        "load_model",
        lambda model_name, storage, source, fold, device: _fake_model().to(device).eval(),
    )

    first = attention_rollout.run_attention_rollout(paths, TEST_SOURCE, device=torch.device("cpu"))
    assert not first.skipped
    assert len(first.samples) == 4
    assert first.maps_path.is_file()
    assert first.metadata_path.is_file()
    assert first.figure_path.is_file()
    assert first.figure_path.read_bytes().startswith(b"\x89PNG")
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["model"] == "segformer"
    assert metadata["n_patches"] == 4
    assert metadata["rollout"]["stage_grid"] == [STAGE_GRID, STAGE_GRID]
    assert metadata["fingerprint"]["rollout_scope"] == attention_rollout.ROLLOUT_SCOPE
    with np.load(first.maps_path) as data:
        assert data["rollouts"].shape == (4, PATCH_SIZE, PATCH_SIZE)
        assert len(data["patch_ids"]) == 4

    mtime = first.metadata_path.stat().st_mtime_ns
    second = attention_rollout.run_attention_rollout(paths, TEST_SOURCE, device=torch.device("cpu"))
    assert second.skipped
    assert second.samples == []
    assert second.metadata_path.stat().st_mtime_ns == mtime


def test_run_attention_rollout_requires_current_summary(tmp_path, monkeypatch) -> None:
    """Sem resumo vigente do estágio 11, a geração deve falhar com mensagem clara."""
    paths = _write_environment(tmp_path, monkeypatch)
    evaluation.evaluation_summary_path(paths, "segformer", TEST_SOURCE).unlink()

    with pytest.raises(ValueError, match="estágio 11"):
        attention_rollout.run_attention_rollout(paths, TEST_SOURCE, device=torch.device("cpu"))


def test_render_attention_rollout_figure_returns_png(tmp_path, monkeypatch) -> None:
    """A renderização da figura Attention Rollout deve devolver bytes PNG."""
    paths = _write_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        attention_rollout,
        "load_model",
        lambda model_name, storage, source, fold, device: _fake_model().to(device).eval(),
    )
    result = attention_rollout.run_attention_rollout(paths, TEST_SOURCE, device=torch.device("cpu"))

    png = attention_rollout.render_attention_rollout_figure(result.samples)

    assert png.startswith(b"\x89PNG")


def test_compute_attention_rollout_with_segformer_backbone() -> None:
    """O backbone real do SegFormer deve expor atenções quadradas na última etapa."""
    from src.models.segformer import build_segformer

    model = build_segformer(pretrained=False).eval()

    rollout, info = attention_rollout.compute_attention_rollout(model, torch.rand(1, 4, 64, 64))

    assert rollout.shape == (64, 64)
    assert info["attention_implementation"] == "eager"
    assert info["stage_index"] == len(model.model.config.depths) - 1
    assert info["stage_layers"] == model.model.config.depths[-1]
    side = info["stage_grid"][0]
    assert side * side == info["stage_grid"][0] * info["stage_grid"][1]
