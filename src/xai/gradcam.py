"""Grad-CAM para o U-Net (estágio 13).

Gera mapas de calor Grad-CAM para patches representativos da validação espacial
usando os pesos de cada dobra (estágio 09). A camada-alvo padrão é o bloco do
codificador mais profundo (`encoders[-1]`), cuja ativação é agregada pelos
gradientes da soma dos logits da classe café na região de referência do patch.
Os mapas são reamostrados bilinearmente para a resolução de entrada, persistidos
em `artifacts/xai/gradcam/unet/<fonte>/` e renderizados em uma figura com RGB,
referência, mapa de calor e sobreposição. A seleção de patches e a geração são
determinísticas e reutilizadas apenas quando o resumo do estágio 11, os pesos
das dobras e a configuração permanecem vigentes.
"""

from __future__ import annotations

import io as stdlib_io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.config import get_config
from src.data.mask_utils import reference_year
from src.data.source_decision import chosen_source
from src.evaluation import (
    load_evaluation_summary,
    load_model,
    load_model_patch_metrics,
    summary_is_current,
)
from src.xai import common

GRADCAM_SCHEMA_VERSION = 1
MODEL_NAME = "unet"
TARGET_LAYER_NAME = "encoders[-1]"
SCORE_REGION = "reference_coffee"


@dataclass(frozen=True)
class GradCamSample:
    """Mapa Grad-CAM de um patch representativo."""

    patch_id: str
    label: str
    fold: int
    iou: float
    coffee_ratio: float
    image_rgb: np.ndarray
    reference: np.ndarray
    cam: np.ndarray
    weights_sha256: str


@dataclass(frozen=True)
class GradCamResult:
    """Resultado da geração (ou reaproveitamento) dos mapas Grad-CAM."""

    source_name: str
    samples: list[GradCamSample]
    metadata: dict[str, Any]
    maps_path: Path
    metadata_path: Path
    figure_path: Path
    skipped: bool


def _xai_root(storage_paths: dict[str, Path]) -> Path:
    """Raiz de XAI: chave dedicada quando presente, senão derivada de artifacts/."""
    if "artifacts_xai" in storage_paths:
        return storage_paths["artifacts_xai"]
    return storage_paths["artifacts_metrics"].parent / "xai"


def gradcam_dir(
    storage_paths: dict[str, Path], source_name: str, model_name: str = MODEL_NAME
) -> Path:
    """Diretório canônico dos mapas Grad-CAM do modelo/fonte."""
    return _xai_root(storage_paths) / "gradcam" / model_name / source_name


def gradcam_maps_path(
    storage_paths: dict[str, Path], source_name: str, model_name: str = MODEL_NAME
) -> Path:
    """Caminho do arquivo compactado com os mapas Grad-CAM persistidos."""
    return gradcam_dir(storage_paths, source_name, model_name) / "gradcam_maps.npz"


def gradcam_metadata_path(
    storage_paths: dict[str, Path], source_name: str, model_name: str = MODEL_NAME
) -> Path:
    """Caminho do metadata (fingerprint e patches) dos mapas Grad-CAM."""
    return gradcam_dir(storage_paths, source_name, model_name) / "gradcam_metadata.json"


def gradcam_figure_file_name(source_name: str, model_name: str = MODEL_NAME) -> str:
    """Nome estável da figura Grad-CAM do modelo/fonte."""
    config = get_config()
    return (
        f"gradcam_{model_name}_{config['aoi']['region_code']}"
        f"_{reference_year(source_name)}_{source_name}.png"
    )


def gradcam_figure_path(
    storage_paths: dict[str, Path], source_name: str, model_name: str = MODEL_NAME
) -> Path:
    """Caminho da figura Grad-CAM no diretório canônico de figuras."""
    return storage_paths["artifacts_figures"] / gradcam_figure_file_name(source_name, model_name)


def default_target_layer(model: nn.Module) -> nn.Module:
    """Retorna a camada-alvo padrão do U-Net: o codificador mais profundo."""
    encoders = getattr(model, "encoders", None)
    if not isinstance(encoders, nn.ModuleList) or len(encoders) == 0:
        raise ValueError("Modelo sem `encoders` (ModuleList não vazio) para o Grad-CAM.")
    return encoders[-1]


def compute_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    """Calcula o mapa Grad-CAM de um patch (uma imagem) para a classe café.

    ``image`` tem forma (C, H, W) ou (1, C, H, W). Quando ``target_mask``
    (H, W) é fornecido, a pontuação é a soma dos logits nessa região; sem máscara
    (ou com máscara vazia), usa todos os pixels. Retorna um mapa float32 em
    [0, 1] na resolução da entrada, calculado com autograd (sem alterar pesos).
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError("Entrada do Grad-CAM deve ter forma (C,H,W) ou (1,C,H,W).")
    layer = target_layer or default_target_layer(model)
    captured: dict[str, torch.Tensor] = {}

    def _capture(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        captured["activation"] = output

    handle = layer.register_forward_hook(_capture)
    try:
        with torch.enable_grad():
            logits = model(image)
            activation = captured.get("activation")
            if activation is None:
                raise RuntimeError("Camada-alvo não produziu ativação no forward.")
            if target_mask is None:
                score = logits.sum()
            else:
                mask = target_mask.to(device=logits.device, dtype=torch.bool)
                if mask.dim() == 3:
                    mask = mask.squeeze(0)
                if mask.shape != logits.shape[-2:]:
                    raise ValueError(
                        f"Máscara {tuple(mask.shape)} incompatível com logits "
                        f"{tuple(logits.shape[-2:])}."
                    )
                score = logits[0, 0][mask].sum() if bool(mask.any()) else logits.sum()
            gradients = torch.autograd.grad(score, activation)[0]
    finally:
        handle.remove()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1))[0].detach()
    cam = torch.nn.functional.interpolate(
        cam[None, None],
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0].detach()
    minimum = float(cam.min())
    maximum = float(cam.max())
    if maximum <= minimum:
        return np.zeros(tuple(image.shape[-2:]), dtype=np.float32)
    normalized = (cam - minimum) / (maximum - minimum)
    return normalized.cpu().numpy().astype(np.float32)


def _weights_by_fold(summary: dict[str, Any]) -> dict[int, str]:
    """Hash dos pesos de cada dobra registrado no resumo do estágio 11."""
    return {int(item["fold"]): str(item["weights_sha256"]) for item in summary["per_fold"]}


def gradcam_fingerprint(
    summary: dict[str, Any], selected: list[dict[str, Any]], source_name: str
) -> dict[str, Any]:
    """Fingerprint que valida a reutilização dos mapas Grad-CAM."""
    weights = _weights_by_fold(summary)
    used_folds = sorted({int(item["fold"]) for item in selected})
    return {
        "schema_version": GRADCAM_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "source_name": source_name,
        "target_layer": TARGET_LAYER_NAME,
        "score_region": SCORE_REGION,
        "evaluation_schema_version": summary.get("schema_version"),
        "manifest_fingerprint_hash": summary["fingerprint"]["manifest_fingerprint_hash"],
        "split_fingerprint_hash": summary["fingerprint"]["split_fingerprint_hash"],
        "config_hash": summary["fingerprint"]["config_hash"],
        "threshold": summary["fingerprint"]["threshold"],
        "weights_sha256": {str(fold): weights[fold] for fold in used_folds},
        "patches": [
            {"patch_id": str(item["patch_id"]), "fold": int(item["fold"])} for item in selected
        ],
    }


def _artifacts_are_current(
    metadata_path: Path,
    maps_path: Path,
    figure_path: Path,
    fingerprint: dict[str, Any],
) -> bool:
    """Indica se os três artefatos existem e correspondem ao fingerprint vigente."""
    from src import io

    if not all(io.path_exists(path) for path in (metadata_path, maps_path, figure_path)):
        return False
    try:
        stored = json.loads(io.ensure_local_copy(metadata_path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return stored.get("fingerprint") == fingerprint


def _persist_maps(path: Path, samples: list[GradCamSample]) -> None:
    """Persiste os mapas Grad-CAM comprimidos com identificadores e dobras."""
    from src import io

    buffer = stdlib_io.BytesIO()
    np.savez_compressed(
        buffer,
        patch_ids=np.asarray([sample.patch_id for sample in samples], dtype=str),
        folds=np.asarray([sample.fold for sample in samples], dtype=np.int64),
        cams=np.stack([sample.cam for sample in samples], axis=0),
        iou=np.asarray([sample.iou for sample in samples], dtype=np.float64),
        coffee_ratio=np.asarray([sample.coffee_ratio for sample in samples], dtype=np.float64),
    )
    io.persist_bytes(path, buffer.getvalue())


def run_gradcam(
    storage_paths: dict[str, Path],
    source_name: str | None = None,
    device: torch.device | None = None,
    max_patches: int = 4,
    force: bool = False,
) -> GradCamResult:
    """Gera (ou reutiliza) os mapas Grad-CAM do U-Net na fonte escolhida."""
    from src import io

    source = source_name or chosen_source()
    if not summary_is_current(storage_paths, MODEL_NAME, source):
        raise ValueError(
            "Resumo do estágio 11 desatualizado ou incompleto para a U-Net; "
            "reexecute a avaliação antes do Grad-CAM."
        )
    summary = load_evaluation_summary(storage_paths, MODEL_NAME, source)
    patch_metrics = load_model_patch_metrics(storage_paths, MODEL_NAME, source)
    selected = common.select_representative_patches(patch_metrics, max_patches=max_patches)
    if not selected:
        raise ValueError(f"Nenhum patch disponível para o Grad-CAM na fonte {source}.")
    fingerprint = gradcam_fingerprint(summary, selected, source)
    maps_file = gradcam_maps_path(storage_paths, source)
    metadata_file = gradcam_metadata_path(storage_paths, source)
    figure_file = gradcam_figure_path(storage_paths, source)
    if not force and _artifacts_are_current(metadata_file, maps_file, figure_file, fingerprint):
        metadata = json.loads(io.ensure_local_copy(metadata_file).read_text(encoding="utf-8"))
        print("Mapas Grad-CAM já vigentes (artefatos reutilizados).")
        return GradCamResult(
            source_name=source,
            samples=[],
            metadata=metadata,
            maps_path=maps_file,
            metadata_path=metadata_file,
            figure_path=figure_file,
            skipped=True,
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats, manifest = common.load_stats_and_manifest(storage_paths, source)
    weights = _weights_by_fold(summary)
    models: dict[int, tuple[nn.Module, nn.Module]] = {}
    samples: list[GradCamSample] = []
    for item in selected:
        patch = str(item["patch_id"])
        fold = int(item["fold"])
        if fold not in models:
            model = load_model(MODEL_NAME, storage_paths, source, fold, device)
            models[fold] = (model, default_target_layer(model))
        model, target_layer = models[fold]
        sample = common.load_patch_sample(storage_paths, manifest, patch, stats)
        cam = compute_gradcam(
            model,
            sample.normalized.unsqueeze(0).to(device),
            target_mask=torch.from_numpy(sample.reference).to(device),
            target_layer=target_layer,
        )
        samples.append(
            GradCamSample(
                patch_id=patch,
                label=str(item["label"]),
                fold=fold,
                iou=float(item["iou"]),
                coffee_ratio=float(item["coffee_ratio"]),
                image_rgb=common.patch_rgb(sample),
                reference=sample.reference,
                cam=cam,
                weights_sha256=weights[fold],
            )
        )

    metadata = {
        "schema_version": GRADCAM_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "source_name": source,
        "device": str(device),
        "target_layer": TARGET_LAYER_NAME,
        "score_region": SCORE_REGION,
        "n_patches": len(samples),
        "patches": [
            {
                "patch_id": sample.patch_id,
                "label": sample.label,
                "fold": sample.fold,
                "iou": sample.iou,
                "coffee_ratio": sample.coffee_ratio,
                "weights_sha256": sample.weights_sha256,
            }
            for sample in samples
        ],
        "fingerprint": fingerprint,
    }
    _persist_maps(maps_file, samples)
    io.persist_json(metadata_file, metadata)
    io.persist_bytes(figure_file, render_gradcam_figure(samples))
    print(
        f"Grad-CAM: {len(samples)} patches | dobras "
        f"{sorted({sample.fold for sample in samples})} | {device}"
    )
    print(f"Figura salva em: {figure_file}")
    return GradCamResult(
        source_name=source,
        samples=samples,
        metadata=metadata,
        maps_path=maps_file,
        metadata_path=metadata_file,
        figure_path=figure_file,
        skipped=False,
    )


def render_gradcam_figure(samples: list[GradCamSample]) -> bytes:
    """Renderiza RGB, referência, Grad-CAM e sobreposição por patch."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize

    from src.data.raster_utils import render_figure

    if not samples:
        raise ValueError("Nenhuma amostra disponível para a figura Grad-CAM.")
    reference_cmap = ListedColormap(["#f5f5f5", "#2e7d32"])

    def _build(plt: Any) -> Any:
        figure, axes = plt.subplots(
            len(samples),
            4,
            figsize=(16, 4.2 * len(samples)),
            squeeze=False,
            constrained_layout=True,
        )
        for row, sample in enumerate(samples):
            axes[row, 0].imshow(sample.image_rgb)
            axes[row, 0].set_title(
                f"{sample.label}\n{sample.patch_id} (dobra {sample.fold})", fontsize=9
            )
            axes[row, 1].imshow(
                sample.reference.astype(np.uint8), cmap=reference_cmap, vmin=0, vmax=1
            )
            axes[row, 1].set_title("Referência (café)", fontsize=9)
            axes[row, 2].imshow(sample.cam, cmap="turbo", vmin=0.0, vmax=1.0)
            axes[row, 2].set_title(f"Grad-CAM (IoU {sample.iou:.3f})", fontsize=9)
            axes[row, 3].imshow(sample.image_rgb)
            axes[row, 3].imshow(sample.cam, cmap="turbo", alpha=0.45, vmin=0.0, vmax=1.0)
            axes[row, 3].set_title("Sobreposição", fontsize=9)
            for axis in axes[row]:
                axis.axis("off")
        mappable = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="turbo")
        mappable.set_array(np.asarray([]))
        figure.colorbar(
            mappable,
            ax=axes[:, 2].tolist(),
            orientation="horizontal",
            fraction=0.02,
            pad=0.02,
            label="relevância Grad-CAM",
        )
        figure.suptitle("Grad-CAM — U-Net (validação espacial)", fontsize=13)
        return figure

    return render_figure(_build)
