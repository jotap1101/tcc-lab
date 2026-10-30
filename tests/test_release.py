"""Testes de src/release.py (publicação no Hugging Face Hub — estágio 16)."""

from __future__ import annotations

import hashlib
import io as stdlib_io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import evaluation, release, trainer
from src.data import eda
from src.data.patch_generation import manifest_path
from src.metrics import f1_score, iou, precision, recall

TEST_SOURCE = "emater"
PATCH_SIZE = 16
FOLD_COUNT = 5


class FakeApi:
    """API mínima em memória para exercitar o release sem rede."""

    def __init__(self) -> None:
        self.repos: dict[tuple[str, str], bool] = {}
        self.files: dict[tuple[str, str], dict[str, bytes]] = {}

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> None:
        """Registra a criação do repositório."""
        del exist_ok
        self.repos[(repo_id, repo_type)] = private

    def repo_exists(self, repo_id: str, *, repo_type: str) -> bool:
        """Indica se o repositório foi criado."""
        return (repo_id, repo_type) in self.repos

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: Path,
        commit_message: str,
        **_: Any,
    ) -> SimpleNamespace:
        """Armazena os arquivos do staging em memória e devolve um commit fictício."""
        del commit_message
        root = Path(folder_path)
        files = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
        self.files[(repo_id, repo_type)] = files
        return SimpleNamespace(commit_url=f"https://huggingface.co/{repo_id}/commit/test")

    def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
        """Lista os arquivos remotos registrados."""
        return sorted(self.files.get((repo_id, repo_type), {}))


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
    """Caminhos de armazenamento mínimos para os testes do estágio 16."""
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
    """Monta um resumo sintético do estágio 11."""
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


def _write_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Escreve manifesto, patches, estatísticas, treinos e avaliações sintéticas."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    records = []
    model_frames: dict[str, dict[int, pd.DataFrame]] = {
        model_name: {} for model_name in release.MODEL_ORDER
    }
    truth = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    truth[3:12, 4:13] = True
    for fold in range(FOLD_COUNT):
        patch_id = f"patch_{fold:03d}"
        image = np.ones((4, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        mask = truth.astype(np.uint8)
        image_path = f"data/processed/patches/{TEST_SOURCE}/images/{patch_id}.npy"
        mask_path = f"data/processed/patches/{TEST_SOURCE}/masks/{patch_id}.npy"
        for relative in (image_path, mask_path):
            (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp_path / image_path, image)
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
        for model_name in release.MODEL_ORDER:
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
    eda.compute_normalization_stats(paths, TEST_SOURCE)

    fingerprint = evaluation.summary_fingerprint(paths, TEST_SOURCE)
    summaries: dict[str, dict[str, Any]] = {}
    for model_name in release.MODEL_ORDER:
        summaries[model_name] = _summary(
            model_name, TEST_SOURCE, model_frames[model_name], fingerprint
        )
        _write_json(
            evaluation.evaluation_summary_path(paths, model_name, TEST_SOURCE),
            summaries[model_name],
        )
        for fold in range(FOLD_COUNT):
            _write_json(
                trainer.metrics_path(paths, model_name, TEST_SOURCE, fold),
                {
                    "model": model_name,
                    "fold": fold,
                    "epochs": 2,
                    "seed": fold,
                    "best_epoch": 1,
                    "train": {"loss": 1.0, "iou": 0.5, "f1": 0.6, "precision": 0.7, "recall": 0.8},
                    "val": {"loss": 0.9, "iou": 0.4, "f1": 0.5, "precision": 0.6, "recall": 0.7},
                },
            )
            _write_json(
                trainer.history_path(paths, model_name, TEST_SOURCE, fold),
                {"model": model_name, "fold": fold, "seed": fold, "epochs": []},
            )
            weights = trainer.weights_path(paths, model_name, TEST_SOURCE, fold)
            weights.parent.mkdir(parents=True, exist_ok=True)
            weights.write_bytes(f"{model_name}-{fold}".encode())

    for model_name in release.MODEL_ORDER:
        for item in summaries[model_name]["per_fold"]:
            fold = int(item["fold"])
            weights = trainer.weights_path(paths, model_name, TEST_SOURCE, fold)
            item["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()[:12]
        _write_json(
            evaluation.evaluation_summary_path(paths, model_name, TEST_SOURCE),
            summaries[model_name],
        )
    return paths


def test_resolve_repo_ids_uses_defaults_and_overrides() -> None:
    """Os IDs padrão devem derivar da conta e aceitar sobrescrita explícita."""
    defaults = release.resolve_repo_ids("alice", "emater")
    assert defaults == {
        "dataset": "alice/tcc-coffee-emater",
        "unet": "alice/tcc-coffee-unet-emater",
        "segformer": "alice/tcc-coffee-segformer-emater",
    }
    overrides = release.resolve_repo_ids(
        "alice", "emater", dataset_repo="org/data", unet_repo="org/unet", segformer_repo="org/sf"
    )
    assert overrides == {"dataset": "org/data", "unet": "org/unet", "segformer": "org/sf"}


def test_run_release_packages_and_reuses(tmp_path, monkeypatch) -> None:
    """O release deve publicar dataset/modelos e reutilizar quando vigente."""
    paths = _write_environment(tmp_path, monkeypatch)
    api = FakeApi()
    repos = release.resolve_repo_ids("alice", TEST_SOURCE)

    first = release.run_release(
        paths,
        TEST_SOURCE,
        dataset_repo=repos["dataset"],
        unet_repo=repos["unet"],
        segformer_repo=repos["segformer"],
        api=api,
    )
    assert not first.skipped
    assert first.n_patches == FOLD_COUNT
    dataset_files = api.files[(repos["dataset"], "dataset")]
    assert {"README.md", "manifest.parquet", f"normalization_stats_{TEST_SOURCE}.json"} <= set(
        dataset_files
    )
    assert "images/patch_000.npy" in dataset_files
    assert "masks/patch_000.npy" in dataset_files
    release_manifest = pd.read_parquet(stdlib_io.BytesIO(dataset_files["manifest.parquet"]))
    assert release_manifest.loc[0, "image_path"] == "images/patch_000.npy"
    assert release_manifest.loc[0, "mask_path"] == "masks/patch_000.npy"
    for model_name in release.MODEL_ORDER:
        files = api.files[(repos[model_name], "model")]
        assert {"README.md", "model_config.json", "fold_0.pt"} <= set(files)
        config = json.loads(files["model_config.json"])
        assert config["model"] == model_name
        assert config["source_name"] == TEST_SOURCE
        assert config["license"] == release.MODEL_LICENSE
    metadata = json.loads(first.release_path.read_text(encoding="utf-8"))
    assert metadata["repos"] == repos
    assert metadata["n_patches"] == FOLD_COUNT
    assert metadata["fingerprint"]["weights_sha256"]["unet"]["0"]

    second = release.run_release(
        paths,
        TEST_SOURCE,
        dataset_repo=repos["dataset"],
        unet_repo=repos["unet"],
        segformer_repo=repos["segformer"],
        api=api,
    )
    assert second.skipped
    assert second.n_patches == FOLD_COUNT


def test_run_release_requires_token(tmp_path, monkeypatch) -> None:
    """Sem HF_TOKEN e sem API injetada, o release deve falhar com orientação clara."""
    paths = _write_environment(tmp_path, monkeypatch)
    monkeypatch.setattr("src.utils.get_secret", lambda name: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        release.run_release(
            paths,
            TEST_SOURCE,
            dataset_repo="a/d",
            unet_repo="a/u",
            segformer_repo="a/s",
        )


def test_run_release_requires_current_evaluation(tmp_path, monkeypatch) -> None:
    """Sem resumo vigente do estágio 11, o release deve ser recusado."""
    paths = _write_environment(tmp_path, monkeypatch)
    evaluation.evaluation_summary_path(paths, "unet", TEST_SOURCE).unlink()

    with pytest.raises(ValueError, match="estágio 11"):
        release.run_release(
            paths,
            TEST_SOURCE,
            dataset_repo="a/d",
            unet_repo="a/u",
            segformer_repo="a/s",
            api=FakeApi(),
        )


def test_run_release_rejects_weights_diverging_from_evaluation(tmp_path, monkeypatch) -> None:
    """Pesos diferentes dos avaliados no estágio 11 não podem ser publicados."""
    paths = _write_environment(tmp_path, monkeypatch)
    weights = trainer.weights_path(paths, "unet", TEST_SOURCE, 0)
    weights.write_bytes(b"retrained")

    with pytest.raises(ValueError, match="divergem da avaliação"):
        release.run_release(
            paths,
            TEST_SOURCE,
            dataset_repo="a/d",
            unet_repo="a/u",
            segformer_repo="a/s",
            api=FakeApi(),
        )
