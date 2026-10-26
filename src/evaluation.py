"""Avaliação por pixel dos modelos treinados (estágio 11).

Executa a inferência dos pesos por dobra dos estágios 09 (U-Net) e 10
(SegFormer) sobre os patches de validação da divisão espacial e calcula IoU,
F1, Precisão e Recall em nível de pixel, ignorando os pixels fora do AOI
(rótulo -1). Persiste, por dobra, as métricas agregadas, as métricas por patch
e as predições binárias; consolida o resumo de cada modelo/fonte. Os artefatos
são reutilizados apenas quando o fingerprint do manifesto, o fingerprint do
split, o hash da configuração e o hash dos pesos coincidem (idempotência).
"""

from __future__ import annotations

import hashlib
import io as stdlib_io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import get_config
from src.data.dataset import PatchDataset, load_normalization_stats
from src.data.mask_utils import reference_year
from src.data.patch_generation import (
    load_manifest,
    manifest_has_folds,
    patch_fingerprint_hash,
    require_manifest,
)
from src.data.source_decision import chosen_source
from src.data.spatial_split import fold_count, split_fingerprint_hash
from src.metrics import f1_score, iou, precision, recall
from src.models.segformer import build_segformer
from src.models.unet import build_unet
from src.trainer import config_hash, weights_path

EVALUATION_SCHEMA_VERSION = 1
DEFAULT_THRESHOLD = 0.5
METRIC_KEYS = ("iou", "f1", "precision", "recall")
COUNT_KEYS = ("tp", "fp", "fn")
MODEL_NAMES = ("unet", "segformer")


@dataclass(frozen=True)
class FoldEvaluation:
    """Resultado da avaliação de uma dobra (ou reaproveitamento de uma existente)."""

    model: str
    source_name: str
    fold: int
    n_patches: int
    threshold: float
    counts: dict[str, int]
    metrics: dict[str, float]
    fingerprint: dict[str, Any]
    metrics_path: Path
    predictions_path: Path
    patch_metrics_path: Path
    skipped: bool


def evaluation_dir(storage_paths: dict[str, Path], model_name: str, source_name: str) -> Path:
    """Diretório canônico dos artefatos de avaliação do modelo/fonte."""
    return storage_paths["artifacts_metrics"] / model_name / source_name / "evaluation"


def evaluation_metrics_path(
    storage_paths: dict[str, Path], model_name: str, source_name: str, fold: int
) -> Path:
    """Caminho das métricas agregadas de uma dobra avaliada."""
    return evaluation_dir(storage_paths, model_name, source_name) / f"fold_{fold}.json"


def predictions_path(
    storage_paths: dict[str, Path], model_name: str, source_name: str, fold: int
) -> Path:
    """Caminho das predições binárias (patches de validação) de uma dobra."""
    return evaluation_dir(storage_paths, model_name, source_name) / f"predictions_fold_{fold}.npz"


def patch_metrics_path(
    storage_paths: dict[str, Path], model_name: str, source_name: str, fold: int
) -> Path:
    """Caminho das métricas por patch de uma dobra avaliada."""
    return (
        evaluation_dir(storage_paths, model_name, source_name)
        / f"patch_metrics_fold_{fold}.parquet"
    )


def evaluation_summary_path(
    storage_paths: dict[str, Path], model_name: str, source_name: str
) -> Path:
    """Caminho do resumo consolidado (dobras + agregado) do modelo/fonte."""
    return evaluation_dir(storage_paths, model_name, source_name) / "summary.json"


def evaluation_figure_file_name(model_name: str, source_name: str) -> str:
    """Nome estável da figura de avaliação do modelo/fonte."""
    config = get_config()
    return (
        f"evaluation_{config['aoi']['region_code']}_{reference_year(source_name)}"
        f"_{source_name}_{model_name}.png"
    )


def evaluation_figure_path(
    storage_paths: dict[str, Path], model_name: str, source_name: str
) -> Path:
    """Caminho da figura de avaliação do modelo/fonte."""
    return storage_paths["artifacts_figures"] / evaluation_figure_file_name(model_name, source_name)


def build_model(model_name: str) -> nn.Module:
    """Constrói o modelo treinado correspondente ao nome canônico."""
    if model_name == "unet":
        return build_unet()
    if model_name == "segformer":
        return build_segformer()
    raise ValueError(f"Modelo desconhecido: {model_name}. Use um de {MODEL_NAMES}.")


def _weights_digest(
    storage_paths: dict[str, Path], model_name: str, source_name: str, fold: int
) -> str:
    """Hash SHA-256 curto dos pesos da dobra (detecta retreino)."""
    from src import io

    path = weights_path(storage_paths, model_name, source_name, fold)
    if not io.path_exists(path):
        raise FileNotFoundError(
            f"Pesos de {model_name} (dobra {fold}) não encontrados: {path}. "
            "Execute o estágio 09/10 correspondente antes da avaliação."
        )
    local_weights = io.ensure_local_copy(path)
    return hashlib.sha256(local_weights.read_bytes()).hexdigest()[:12]


def evaluation_fingerprint(
    storage_paths: dict[str, Path],
    model_name: str,
    source_name: str,
    fold: int,
    threshold: float,
) -> dict[str, Any]:
    """Fingerprint que valida a reutilização dos artefatos de avaliação."""
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "threshold": float(threshold),
        "manifest_fingerprint_hash": patch_fingerprint_hash(storage_paths, source_name),
        "split_fingerprint_hash": split_fingerprint_hash(storage_paths, source_name),
        "config_hash": config_hash(),
        "weights_sha256": _weights_digest(storage_paths, model_name, source_name, fold),
    }


def _evaluation_is_current(
    metrics_file: Path,
    predictions_file: Path,
    patch_metrics_file: Path,
    fingerprint: dict[str, Any],
) -> bool:
    """Indica se os três artefatos da dobra existem e correspondem ao fingerprint."""
    from src import io

    if not all(
        io.path_exists(path) for path in (metrics_file, predictions_file, patch_metrics_file)
    ):
        return False
    try:
        stored = json.loads(io.ensure_local_copy(metrics_file).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return stored.get("fingerprint") == fingerprint


def load_model(
    model_name: str,
    storage_paths: dict[str, Path],
    source_name: str,
    fold: int,
    device: torch.device,
) -> nn.Module:
    """Constrói o modelo e carrega os pesos treinados da dobra."""
    from src import io

    model = build_model(model_name)
    local_weights = io.ensure_local_copy(weights_path(storage_paths, model_name, source_name, fold))
    state_dict = torch.load(local_weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def _persist_predictions(path: Path, patch_ids: np.ndarray, predictions: np.ndarray) -> None:
    """Persiste as predições binárias comprimidas com os identificadores dos patches."""
    from src import io

    buffer = stdlib_io.BytesIO()
    np.savez_compressed(buffer, patch_ids=np.asarray(patch_ids, dtype=str), predictions=predictions)
    io.persist_bytes(path, buffer.getvalue())


def _persist_patch_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persiste as métricas por patch da dobra em Parquet."""
    from src import io

    buffer = stdlib_io.BytesIO()
    pd.DataFrame(rows).to_parquet(buffer, index=False)
    io.persist_bytes(path, buffer.getvalue())


def _valid_fraction(record: pd.Series) -> float | None:
    """Fração válida do patch (None quando a coluna não existe ou é nula)."""
    if "valid_fraction" not in record.index:
        return None
    value = record["valid_fraction"]
    return None if pd.isna(value) else float(value)


def evaluate_fold(
    model_name: str,
    storage_paths: dict[str, Path],
    fold: int,
    device: torch.device,
    source_name: str | None = None,
    force: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> FoldEvaluation:
    """Avalia um modelo na dobra de validação espacial da fonte (idempotente).

    Reutiliza as métricas, predições e métricas por patch quando o fingerprint
    vigente coincide; caso contrário, executa a inferência sobre os patches da
    dobra e persiste os artefatos no Drive canônico.
    """
    from src import io

    source = source_name or chosen_source()
    require_manifest(storage_paths, source)
    manifest = load_manifest(storage_paths, source)
    if not manifest_has_folds(manifest):
        raise ValueError("Coluna fold ausente/incompleta; execute o estágio 07 antes.")
    fold_records = manifest[manifest["fold"].astype(int) == fold].reset_index(drop=True)
    if fold_records.empty:
        raise ValueError(f"Dobra {fold} sem patches no manifesto; execute o estágio 07.")

    metrics_file = evaluation_metrics_path(storage_paths, model_name, source, fold)
    predictions_file = predictions_path(storage_paths, model_name, source, fold)
    patch_file = patch_metrics_path(storage_paths, model_name, source, fold)
    fingerprint = evaluation_fingerprint(storage_paths, model_name, source, fold, threshold)

    if not force and _evaluation_is_current(
        metrics_file, predictions_file, patch_file, fingerprint
    ):
        payload = json.loads(io.ensure_local_copy(metrics_file).read_text(encoding="utf-8"))
        print(
            f"[{model_name}] dobra {fold} já avaliada "
            f"(artefatos reutilizados): IoU {payload['metrics']['iou']:.4f}"
        )
        return FoldEvaluation(
            model=model_name,
            source_name=source,
            fold=fold,
            n_patches=int(payload["n_patches"]),
            threshold=float(payload["threshold"]),
            counts={key: int(payload["counts"][key]) for key in COUNT_KEYS},
            metrics={key: float(payload["metrics"][key]) for key in METRIC_KEYS},
            fingerprint=fingerprint,
            metrics_path=metrics_file,
            predictions_path=predictions_file,
            patch_metrics_path=patch_file,
            skipped=True,
        )

    stats = load_normalization_stats(storage_paths, source)
    model = load_model(model_name, storage_paths, source, fold, device)
    batch_size = max(1, int(get_config()["training"]["batch_size"]))
    loader = DataLoader(
        PatchDataset(fold_records, storage_paths, stats),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    offset = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            valid = targets >= 0
            truth = (targets > 0.5) & valid
            predicted = (torch.sigmoid(logits) >= threshold) & valid
            predicted_np = predicted.squeeze(1).cpu().numpy().astype(np.uint8)
            truth_np = truth.squeeze(1).cpu().numpy()
            for index in range(predicted_np.shape[0]):
                record = fold_records.iloc[offset + index]
                tp = int(np.count_nonzero(predicted_np[index] & truth_np[index]))
                fp = int(np.count_nonzero(predicted_np[index] & ~truth_np[index]))
                fn = int(np.count_nonzero(~(predicted_np[index] > 0) & truth_np[index]))
                total_tp += tp
                total_fp += fp
                total_fn += fn
                rows.append(
                    {
                        "patch_id": str(record["patch_id"]),
                        "fold": int(fold),
                        "coffee_ratio": float(record["coffee_ratio"]),
                        "valid_fraction": _valid_fraction(record),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "iou": iou(tp, fp, fn),
                        "f1": f1_score(tp, fp, fn),
                        "precision": precision(tp, fp),
                        "recall": recall(tp, fn),
                    }
                )
                predictions.append(predicted_np[index])
            offset += predicted_np.shape[0]

    counts = {"tp": total_tp, "fp": total_fp, "fn": total_fn}
    metrics = {
        "iou": iou(total_tp, total_fp, total_fn),
        "f1": f1_score(total_tp, total_fp, total_fn),
        "precision": precision(total_tp, total_fp),
        "recall": recall(total_tp, total_fn),
    }
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "model": model_name,
        "source_name": source,
        "fold": fold,
        "threshold": float(threshold),
        "n_patches": int(len(fold_records)),
        "counts": counts,
        "metrics": metrics,
        "fingerprint": fingerprint,
    }
    _persist_predictions(
        predictions_file,
        fold_records["patch_id"].astype(str).to_numpy(),
        np.stack(predictions, axis=0),
    )
    _persist_patch_metrics(patch_file, rows)
    io.persist_json(metrics_file, payload)
    print(
        f"[{model_name}] dobra {fold} avaliada: IoU {metrics['iou']:.4f} | "
        f"F1 {metrics['f1']:.4f} | Precisão {metrics['precision']:.4f} | "
        f"Recall {metrics['recall']:.4f} ({len(fold_records)} patches)"
    )
    return FoldEvaluation(
        model=model_name,
        source_name=source,
        fold=fold,
        n_patches=int(len(fold_records)),
        threshold=float(threshold),
        counts=counts,
        metrics=metrics,
        fingerprint=fingerprint,
        metrics_path=metrics_file,
        predictions_path=predictions_file,
        patch_metrics_path=patch_file,
        skipped=False,
    )


def evaluate_model(
    storage_paths: dict[str, Path],
    model_name: str,
    device: torch.device,
    source_name: str | None = None,
    force: bool = False,
) -> list[FoldEvaluation]:
    """Avalia todas as dobras da validação espacial de um modelo na fonte."""
    source = source_name or chosen_source()
    return [
        evaluate_fold(
            model_name,
            storage_paths,
            fold,
            device,
            source_name=source,
            force=force,
        )
        for fold in range(fold_count())
    ]


def summarize_model(
    storage_paths: dict[str, Path],
    model_name: str,
    source_name: str,
    evaluations: list[FoldEvaluation],
) -> dict[str, Any]:
    """Consolida as métricas das dobras em um resumo por modelo/fonte.

    Reporta o agregado *micro* (TP/FP/FN somados entre dobras) e a média/desvio
    das métricas por dobra (macro), além das contagens e do hash dos pesos de
    cada dobra. O resumo é sempre regravado a partir das dobras avaliadas.
    """
    from src import io

    if not evaluations:
        raise ValueError("Nenhuma dobra avaliada para consolidar o resumo.")
    ordered = sorted(evaluations, key=lambda item: item.fold)
    totals = {key: sum(item.counts[key] for item in ordered) for key in COUNT_KEYS}
    aggregate_metrics = {
        "iou": iou(totals["tp"], totals["fp"], totals["fn"]),
        "f1": f1_score(totals["tp"], totals["fp"], totals["fn"]),
        "precision": precision(totals["tp"], totals["fp"]),
        "recall": recall(totals["tp"], totals["fn"]),
    }
    fold_mean = {
        key: float(np.mean([item.metrics[key] for item in ordered])) for key in METRIC_KEYS
    }
    fold_std = {key: float(np.std([item.metrics[key] for item in ordered])) for key in METRIC_KEYS}
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "model": model_name,
        "source_name": source_name,
        "threshold": ordered[0].threshold,
        "n_folds": len(ordered),
        "n_patches": sum(item.n_patches for item in ordered),
        "per_fold": [
            {
                "fold": item.fold,
                "n_patches": item.n_patches,
                "counts": dict(item.counts),
                "metrics": dict(item.metrics),
                "weights_sha256": item.fingerprint["weights_sha256"],
            }
            for item in ordered
        ],
        "aggregate": {"counts": totals, "metrics": aggregate_metrics},
        "fold_mean": fold_mean,
        "fold_std": fold_std,
        "fingerprint": {
            "manifest_fingerprint_hash": ordered[0].fingerprint["manifest_fingerprint_hash"],
            "split_fingerprint_hash": ordered[0].fingerprint["split_fingerprint_hash"],
            "config_hash": ordered[0].fingerprint["config_hash"],
            "threshold": ordered[0].threshold,
        },
    }
    io.persist_json(evaluation_summary_path(storage_paths, model_name, source_name), payload)
    return payload


def load_patch_metrics(
    storage_paths: dict[str, Path], model_name: str, source_name: str, fold: int
) -> pd.DataFrame:
    """Carrega as métricas por patch de uma dobra avaliada."""
    from src import io

    path = patch_metrics_path(storage_paths, model_name, source_name, fold)
    if not io.path_exists(path):
        raise FileNotFoundError(f"Métricas por patch não encontradas: {path}")
    return pd.read_parquet(io.ensure_local_copy(path))


def load_predictions(
    storage_paths: dict[str, Path], model_name: str, source_name: str, fold: int
) -> dict[str, np.ndarray]:
    """Carrega as predições binárias da dobra, indexadas pelo identificador do patch."""
    from src import io

    path = predictions_path(storage_paths, model_name, source_name, fold)
    if not io.path_exists(path):
        raise FileNotFoundError(f"Predições não encontradas: {path}")
    with np.load(io.ensure_local_copy(path), allow_pickle=False) as data:
        predictions = data["predictions"]
        return {
            str(patch_id): predictions[index] for index, patch_id in enumerate(data["patch_ids"])
        }


def save_evaluation_figure(
    storage_paths: dict[str, Path],
    model_name: str,
    source_name: str,
    summary: dict[str, Any],
) -> Path:
    """Persiste a figura de avaliação do modelo/fonte (regenerada a cada execução)."""
    from src import io

    frames = [
        load_patch_metrics(storage_paths, model_name, source_name, item["fold"])
        for item in summary["per_fold"]
    ]
    patch_metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    figure_path = evaluation_figure_path(storage_paths, model_name, source_name)
    io.persist_bytes(figure_path, render_evaluation_figure(summary, patch_metrics))
    print(f"Figura salva em: {figure_path}")
    return figure_path


def render_evaluation_figure(summary: dict[str, Any], patch_metrics: pd.DataFrame) -> bytes:
    """Renderiza métricas por dobra, IoU por patch e relação com café/área válida."""
    from src.data.raster_utils import render_figure

    per_fold = summary["per_fold"]
    coffee_patches = (
        patch_metrics[patch_metrics["tp"] + patch_metrics["fn"] > 0]
        if not patch_metrics.empty
        else patch_metrics
    )

    def _build(plt: Any) -> Any:
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        fold_labels = [str(item["fold"]) for item in per_fold]
        positions = np.arange(len(fold_labels))
        width = 0.8 / len(METRIC_KEYS)
        for index, key in enumerate(METRIC_KEYS):
            offset = (index - (len(METRIC_KEYS) - 1) / 2) * width
            axes[0, 0].bar(
                positions + offset,
                [item["metrics"][key] for item in per_fold],
                width=width,
                label=key,
            )
        axes[0, 0].set_xticks(positions, fold_labels)
        axes[0, 0].set_ylim(0.0, 1.05)
        axes[0, 0].set_title("Métricas por dobra (validação espacial)")
        axes[0, 0].set_xlabel("dobra")
        axes[0, 0].set_ylabel("valor")
        axes[0, 0].legend()

        if not coffee_patches.empty:
            axes[0, 1].hist(coffee_patches["iou"], bins=20, color="#1565c0", alpha=0.85)
        else:
            axes[0, 1].text(0.5, 0.5, "sem patches com café", ha="center", va="center")
            axes[0, 1].set_axis_off()
        axes[0, 1].set_title("IoU por patch (patches com café)")
        axes[0, 1].set_xlabel("IoU")
        axes[0, 1].set_ylabel("patches")

        if not coffee_patches.empty:
            axes[1, 0].scatter(
                coffee_patches["coffee_ratio"],
                coffee_patches["iou"],
                s=14,
                alpha=0.7,
                color="#2e7d32",
            )
        axes[1, 0].set_title("IoU vs proporção de café")
        axes[1, 0].set_xlabel("coffee_ratio")
        axes[1, 0].set_ylabel("IoU")
        axes[1, 0].set_ylim(-0.02, 1.02)

        if (
            not patch_metrics.empty
            and "valid_fraction" in patch_metrics
            and patch_metrics["valid_fraction"].notna().any()
        ):
            valid = patch_metrics.dropna(subset=["valid_fraction"])
            axes[1, 1].scatter(
                valid["valid_fraction"],
                valid["iou"],
                s=14,
                alpha=0.7,
                color="#f9a825",
            )
        axes[1, 1].set_title("IoU vs fração válida do patch")
        axes[1, 1].set_xlabel("valid_fraction")
        axes[1, 1].set_ylabel("IoU")
        axes[1, 1].set_ylim(-0.02, 1.02)

        figure.tight_layout()
        return figure

    return render_figure(_build)
