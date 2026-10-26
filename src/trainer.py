"""Protocolo de treino único compartilhado pelos estágios 09 e 10.

Um único `train_fold()` executa a validação cruzada espacial (k = `splits.fold_count`)
com o mesmo protocolo para U-Net (estágio 09) e SegFormer (estágio 10): mesma
perda multivariada (Dice + Focal + Boundary), otimizador Adam, scheduler
cosine annealing, sementes por dobra e métricas de pixel. Somente a arquitetura
diverge — garantia da comparação científica justa. O treino usa o conjunto de
patches da fonte escolhida manualmente (`ground_truth.chosen_source`), que possui
máscara e patches próprios independentemente da escolha. Os artefatos por dobra
(pesos, métricas e histórico) são persistidos no Drive canônico sob subpasta da
fonte e reutilizados em reexecuções (idempotência), com metadata em artifacts/runs/.
"""

from __future__ import annotations

import hashlib
import io as stdlib_io
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.config import get_config
from src.data.augmentations import SegmentAugmentation
from src.data.dataset import build_loaders
from src.data.mask_utils import reference_year
from src.data.patch_generation import patch_fingerprint_hash
from src.data.source_decision import chosen_source
from src.data.spatial_split import split_fingerprint_hash
from src.losses import SegmentationLoss
from src.metrics import SegmentationMetrics
from src.utils import set_all_seeds


@dataclass(frozen=True)
class FoldResult:
    """Resumo do treino de uma dobra (ou do reaproveitamento de uma já treinada)."""

    model: str
    source_name: str
    fold: int
    weights_path: Path
    metrics_path: Path
    history_path: Path
    best_epoch: int
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    epochs: int
    skipped: bool


def weights_path(
    storage_paths: dict[str, Path], model_name: str, source_name: str, fold: int
) -> Path:
    """Caminho canônico dos pesos do modelo em uma dobra da fonte."""
    return storage_paths["models"] / model_name / source_name / f"fold_{fold}.pt"


def metrics_path(
    storage_paths: dict[str, Path], model_name: str, source_name: str, fold: int
) -> Path:
    """Caminho canônico das métricas de validação em uma dobra da fonte."""
    return storage_paths["artifacts_metrics"] / model_name / source_name / f"fold_{fold}.json"


def history_path(
    storage_paths: dict[str, Path], model_name: str, source_name: str, fold: int
) -> Path:
    """Caminho canônico do histórico de treino (loss e métricas por época) da fonte."""
    return storage_paths["artifacts_runs"] / model_name / source_name / f"fold_{fold}.json"


def fold_training_done(
    storage_paths: dict[str, Path], model_name: str, source_name: str, fold: int
) -> bool:
    """Indica se os três artefatos de uma dobra já existem (idempotência)."""
    from src import io

    return all(
        io.path_exists(path)
        for path in (
            weights_path(storage_paths, model_name, source_name, fold),
            metrics_path(storage_paths, model_name, source_name, fold),
            history_path(storage_paths, model_name, source_name, fold),
        )
    )


def _persist_weights(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    """Persiste o state dict do modelo no Drive canônico (via buffer em memória)."""
    from src import io

    buffer = stdlib_io.BytesIO()
    torch.save(state_dict, buffer)
    io.persist_bytes(path, buffer.getvalue())


def _run_epoch(
    loader: Any,
    model: nn.Module,
    loss_fn: SegmentationLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    """Executa uma época (treino ou validação) e retorna loss e métricas médias."""
    model.train(training)
    metrics = SegmentationMetrics()
    sums = {"total": 0.0, "dice": 0.0, "focal": 0.0, "boundary": 0.0}
    batches = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        if training:
            assert optimizer is not None
            optimizer.zero_grad()
            logits = model(images)
            components = loss_fn(logits, targets)
            components["total"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                components = loss_fn(logits, targets)
        for key in sums:
            sums[key] += float(components[key].detach().cpu())
        metrics.update(logits.detach(), targets)
        batches += 1
    stats = metrics.compute()
    for key, value in sums.items():
        stats["loss" if key == "total" else key] = value / max(batches, 1)
    return stats


def train_fold(
    model: nn.Module,
    model_name: str,
    storage_paths: dict[str, Path],
    fold: int,
    device: torch.device,
    force: bool = False,
    source_name: str | None = None,
) -> FoldResult:
    """Treina um modelo em uma dobra da validação espacial da fonte (protocolo único).

    A fonte é a escolhida manualmente em config.yaml (padrão) ou a explicitada em
    ``source_name``. Reutiliza os artefatos da dobra quando já existem (a menos
    que ``force``); caso contrário, executa o protocolo completo e persiste pesos,
    métricas e histórico. A semente por dobra é ``reproducibility.seed + fold`` e,
    a cada época, as aumentações são reseedadas com ``semente + época``.
    """
    from src import io

    source = source_name or chosen_source()
    config = get_config()
    training = config["training"]
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    lr = float(training["lr"])
    weight_decay = float(training["weight_decay"])
    num_workers = int(training["num_workers"])
    fold_seed = int(config["reproducibility"]["seed"]) + fold

    weights_file = weights_path(storage_paths, model_name, source, fold)
    metrics_file = metrics_path(storage_paths, model_name, source, fold)
    history_file = history_path(storage_paths, model_name, source, fold)

    current_fingerprint = patch_fingerprint_hash(storage_paths, source)
    current_split = split_fingerprint_hash(storage_paths, source)
    current_config_hash = config_hash()

    if not force and fold_training_done(storage_paths, model_name, source, fold):
        saved = json.loads(io.ensure_local_copy(metrics_file).read_text(encoding="utf-8"))
        # Só reutiliza quando os artefatos correspondem ao dataset e à config atuais.
        if (
            saved.get("manifest_fingerprint_hash") == current_fingerprint
            and saved.get("split_fingerprint_hash") == current_split
            and saved.get("config_hash") == current_config_hash
        ):
            print(f"[{model_name}] dobra {fold} já treinada (artefatos reutilizados).")
            return FoldResult(
                model=model_name,
                source_name=source,
                fold=fold,
                weights_path=weights_file,
                metrics_path=metrics_file,
                history_path=history_file,
                best_epoch=int(saved["best_epoch"]),
                train_metrics=saved["train"],
                val_metrics=saved["val"],
                epochs=epochs,
                skipped=True,
            )
        print(
            f"[{model_name}] dobra {fold}: artefatos desatualizados "
            "(dataset/split/config mudaram); retreinando."
        )

    set_all_seeds(fold_seed)
    augmentation = SegmentAugmentation(config)
    train_loader, val_loader = build_loaders(
        storage_paths,
        source,
        fold,
        batch_size,
        fold_seed,
        num_workers,
        transform=augmentation,
    )
    model.to(device)
    loss_fn = SegmentationLoss(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_iou = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_train: dict[str, float] = {}
    best_val: dict[str, float] = {}
    history: list[dict[str, Any]] = []

    for epoch in range(epochs):
        augmentation.set_seed(fold_seed + epoch)
        # Captura o LR vigente antes do passo do scheduler (valor usado nesta época).
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        train_stats = _run_epoch(train_loader, model, loss_fn, optimizer, device, training=True)
        scheduler.step()
        val_stats = _run_epoch(val_loader, model, loss_fn, None, device, training=False)
        history.append(
            {
                "epoch": epoch + 1,
                "lr": epoch_lr,
                **{f"train_{key}": value for key, value in train_stats.items()},
                **{f"val_{key}": value for key, value in val_stats.items()},
            }
        )
        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_train = dict(train_stats)
            best_val = dict(val_stats)
        print(
            f"[{model_name}] dobra {fold} | época {epoch + 1}/{epochs} | "
            f"treino iou {train_stats['iou']:.4f} loss {train_stats['loss']:.4f} | "
            f"val iou {val_stats['iou']:.4f} loss {val_stats['loss']:.4f}"
        )

    if best_state is None:
        raise RuntimeError(f"[{model_name}] dobra {fold}: treino sem nenhuma época concluída.")
    model.load_state_dict(best_state)
    _persist_weights(weights_file, best_state)
    io.persist_json(
        metrics_file,
        {
            "model": model_name,
            "fold": fold,
            "epochs": epochs,
            "seed": fold_seed,
            "best_epoch": best_epoch,
            "manifest_fingerprint_hash": current_fingerprint,
            "split_fingerprint_hash": current_split,
            "config_hash": current_config_hash,
            "train": best_train,
            "val": best_val,
        },
    )
    io.persist_json(
        history_file,
        {"model": model_name, "fold": fold, "seed": fold_seed, "epochs": history},
    )
    print(
        f"[{model_name}] dobra {fold} concluída: melhor IoU val {best_iou:.4f} "
        f"(época {best_epoch})."
    )
    print(f"Pesos: {weights_file}")
    return FoldResult(
        model=model_name,
        source_name=source,
        fold=fold,
        weights_path=weights_file,
        metrics_path=metrics_file,
        history_path=history_file,
        best_epoch=best_epoch,
        train_metrics=best_train,
        val_metrics=best_val,
        epochs=epochs,
        skipped=False,
    )


def config_hash() -> str:
    """Hash SHA-256 curto do config.yaml vigente (fingerprint da configuração)."""
    from src.config import CONFIG_PATH

    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()[:12]


COMMIT_FILE_NAME = ".tcc_commit"


def _commit_sha(commit_file: Path | None = None) -> str:
    """SHA do commit em execução (git local ou arquivo gravado pelo bootstrap).

    No Colab/Kaggle o `src/` vem de um tarball sem `.git`; nesse caso usa o SHA
    que o bootstrap grava em `src/.tcc_commit`. Retorna 'unknown' se nenhum dos
    dois estiver disponível.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
        sha = completed.stdout.strip()
        if completed.returncode == 0 and sha:
            return sha
    except OSError:
        pass

    path = commit_file or Path(__file__).resolve().parent / COMMIT_FILE_NAME
    try:
        sha = path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    return sha or "unknown"


def _package_versions() -> dict[str, str]:
    """Versões dos pacotes principais do runtime, para o log de reprodutibilidade."""
    import importlib.metadata

    versions: dict[str, str] = {}
    for package in ("torch", "torchvision", "transformers", "numpy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "não instalado"
    return versions


def run_metadata(
    model_name: str, storage_paths: dict[str, Path], source_name: str
) -> dict[str, Any]:
    """Metadata de reprodutibilidade de uma execução de treino da fonte."""
    from src import io
    from src.data.patch_generation import patch_fingerprint_hash
    from src.data.spatial_split import fold_count

    config = get_config()
    training = config["training"]
    return {
        "model": model_name,
        "source_name": source_name,
        "year": reference_year(source_name),
        "fold_count": fold_count(),
        "platform": io.detect_platform(),
        "config_hash": config_hash(),
        "commit_sha": _commit_sha(),
        "seed": int(config["reproducibility"]["seed"]),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "lr": float(training["lr"]),
        "weight_decay": float(training["weight_decay"]),
        "num_workers": int(training["num_workers"]),
        "model_config": dict(config["model"]),
        "deterministic_algorithms": {
            "enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        },
        "manifest_fingerprint_hash": patch_fingerprint_hash(storage_paths, source_name),
        "split_fingerprint_hash": split_fingerprint_hash(storage_paths, source_name),
        "environment": _package_versions(),
    }


def save_run_metadata(storage_paths: dict[str, Path], model_name: str, source_name: str) -> Path:
    """Persiste o metadata de execução em artifacts/runs/{model}/{fonte}/run.meta.json.

    Reaproveita o arquivo apenas quando o conteúdo vigente é idêntico; caso
    contrário, sobrescreve (ex.: fingerprint do dataset ou commit SHA mudou).
    """
    from src import io

    path = storage_paths["artifacts_runs"] / model_name / source_name / "run.meta.json"
    metadata = run_metadata(model_name, storage_paths, source_name)
    if io.path_exists(path):
        stored = json.loads(io.ensure_local_copy(path).read_text(encoding="utf-8"))
        if stored == metadata:
            print(f"Metadata de execução já existente e atual (reutilizado): {path}")
            return path
    io.persist_json(path, metadata)
    print(f"Metadata de execução salvo em: {path}")
    return path
