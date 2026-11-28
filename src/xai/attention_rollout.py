"""Attention Rollout para o SegFormer (estágio 14).

Gera mapas de relevância por Attention Rollout para patches representativos da
validação espacial usando os pesos de cada dobra (estágio 10). O rollout é
calculado sobre a última etapa do codificador hierárquico do SegFormer, única
com atenção quadrada (redução espacial 1:1), agregando as cabeças pela média,
somando a identidade (conexões residuais), normalizando cada linha e compondo as
camadas da etapa. Como o SegFormer não possui token CLS, a relevância de cada
token é a média das linhas do rollout. O mapa é reamostrado bilinearmente para a
resolução de entrada, persistido em `artifacts/xai/attention_rollout/segformer/
<fonte>/` e renderizado em uma figura com RGB, referência, mapa e sobreposição.
A seleção de patches e a geração são determinísticas e reutilizadas apenas
quando o resumo do estágio 11, os pesos das dobras e a configuração permanecem
vigentes.
"""

from __future__ import annotations

import io as stdlib_io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src import figstyle
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

ATTENTION_ROLLOUT_SCHEMA_VERSION = 2
MODEL_NAME = "segformer"
ROLLOUT_SCOPE = "last_stage"
HEAD_REDUCTION = "mean"
TOKEN_REDUCTION = "mean"
QUERY_WEIGHTING = "predicted_coffee_probability"
RESIDUAL = True
NORMALIZATION = "min_max"
ATTENTION_IMPLEMENTATION = "eager"


@dataclass(frozen=True)
class AttentionRolloutSample:
    """Mapa Attention Rollout de um patch representativo."""

    patch_id: str
    label: str
    fold: int
    iou: float
    coffee_ratio: float
    image_rgb: np.ndarray
    reference: np.ndarray
    rollout: np.ndarray
    weights_sha256: str


@dataclass(frozen=True)
class AttentionRolloutResult:
    """Resultado da geração (ou reaproveitamento) dos mapas Attention Rollout."""

    source_name: str
    samples: list[AttentionRolloutSample]
    metadata: dict[str, Any]
    maps_path: Path
    metadata_path: Path
    figure_path: Path
    skipped: bool


def attention_rollout_dir(
    storage_paths: dict[str, Path], source_name: str, model_name: str = MODEL_NAME
) -> Path:
    """Diretório canônico dos mapas Attention Rollout do modelo/fonte."""
    return common.xai_root(storage_paths) / "attention_rollout" / model_name / source_name


def attention_rollout_maps_path(
    storage_paths: dict[str, Path], source_name: str, model_name: str = MODEL_NAME
) -> Path:
    """Caminho do arquivo compactado com os mapas Attention Rollout persistidos."""
    return (
        attention_rollout_dir(storage_paths, source_name, model_name) / "attention_rollout_maps.npz"
    )


def attention_rollout_metadata_path(
    storage_paths: dict[str, Path], source_name: str, model_name: str = MODEL_NAME
) -> Path:
    """Caminho do metadata (fingerprint e patches) dos mapas Attention Rollout."""
    return (
        attention_rollout_dir(storage_paths, source_name, model_name)
        / "attention_rollout_metadata.json"
    )


def attention_rollout_figure_file_name(source_name: str, model_name: str = MODEL_NAME) -> str:
    """Nome estável da figura Attention Rollout do modelo/fonte."""
    config = get_config()
    return (
        f"attention_rollout_{model_name}_{config['aoi']['region_code']}"
        f"_{reference_year(source_name)}_{source_name}.png"
    )


def attention_rollout_figure_path(
    storage_paths: dict[str, Path], source_name: str, model_name: str = MODEL_NAME
) -> Path:
    """Caminho da figura Attention Rollout no diretório canônico de figuras."""
    return storage_paths["artifacts_figures"] / attention_rollout_figure_file_name(
        source_name, model_name
    )


def evaluate_rollout_attentions(
    attentions: Sequence[torch.Tensor],
    *,
    residual: bool = RESIDUAL,
    query_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calcula o Attention Rollout de atenções quadradas ``(cabeças, N, N)``.

    As cabeças são agregadas pela média, a identidade é somada quando
    ``residual`` é verdadeiro e cada linha é normalizada. As camadas são
    compostas por multiplicação matricial, na ordem de execução. Sem token CLS,
    a relevância final de cada token é a média das linhas do rollout; com
    ``query_weights`` (N,), é a média ponderada das linhas, o que restringe a
    explicação aos tokens de consulta que sustentam a classe-alvo. A média
    uniforme acumula relevância em tokens que recebem muita atenção
    independentemente do conteúdo (*attention sinks*), como o canto da imagem.
    """
    if not attentions:
        raise ValueError("Nenhuma atenção fornecida para o Attention Rollout.")
    rollout: torch.Tensor | None = None
    for index, attention in enumerate(attentions):
        if attention.dim() == 4 and attention.shape[0] == 1:
            attention = attention[0]
        if attention.dim() != 3 or attention.shape[-2] != attention.shape[-1]:
            raise ValueError(
                f"Atenção {index} deve ter forma (cabeças, N, N); "
                f"recebido {tuple(attention.shape)}."
            )
        matrix = attention.float().mean(dim=0)
        if residual:
            identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
            matrix = matrix + identity
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = matrix if rollout is None else matrix @ rollout
    assert rollout is not None
    if query_weights is None:
        relevance = rollout.mean(dim=0)
    else:
        weights = query_weights.to(device=rollout.device, dtype=rollout.dtype).flatten()
        if weights.numel() != rollout.shape[0]:
            raise ValueError(
                f"`query_weights` com {weights.numel()} elementos para {rollout.shape[0]} tokens."
            )
        total = weights.sum()
        relevance = (
            rollout.mean(dim=0)
            if float(total) <= 0.0
            else (weights[:, None] * rollout).sum(0) / total
        )
    return relevance / relevance.sum().clamp_min(1e-12)


def last_stage_attentions(
    attentions: Sequence[torch.Tensor], depths: Sequence[int]
) -> list[torch.Tensor]:
    """Seleciona as atenções da última etapa do codificador hierárquico."""
    if not depths:
        raise ValueError("Configuração sem `depths`; não é possível isolar a última etapa.")
    stage_layers = int(depths[-1])
    if stage_layers <= 0 or stage_layers > len(attentions):
        raise ValueError(
            f"Configuração inválida: {stage_layers} camadas na última etapa, "
            f"mas {len(attentions)} atenções disponíveis."
        )
    selected = list(attentions[-stage_layers:])
    for index, attention in enumerate(selected):
        if attention.shape[-2] != attention.shape[-1]:
            raise ValueError(
                "A última etapa do SegFormer deve ter atenção quadrada "
                f"(redução espacial 1:1); atenção {index} tem forma "
                f"{tuple(attention.shape)}."
            )
    return selected


def _ensure_eager_attention(model: nn.Module) -> str:
    """Força a implementação `eager` para que o modelo exponha as atenções."""
    config = getattr(model, "config", None)
    implementation = getattr(config, "_attn_implementation", None)
    if implementation != ATTENTION_IMPLEMENTATION:
        setter = getattr(model, "set_attn_implementation", None)
        if setter is None:
            raise ValueError(
                "Modelo sem `set_attn_implementation`; não é possível habilitar atenções."
            )
        setter(ATTENTION_IMPLEMENTATION)
        implementation = getattr(model.config, "_attn_implementation", None)
    if implementation != ATTENTION_IMPLEMENTATION:
        raise RuntimeError(
            f"Implementação de atenção efetiva {implementation!r}; "
            f"esperado {ATTENTION_IMPLEMENTATION!r}."
        )
    return str(implementation)


def _query_weights(
    logits: torch.Tensor, valid_mask: torch.Tensor | None, side: int
) -> torch.Tensor:
    """Probabilidade média de café (somente pixels válidos) em cada token de consulta."""
    probability = torch.sigmoid(logits.float())[:, :1]
    if valid_mask is not None:
        valid = valid_mask.to(device=probability.device, dtype=probability.dtype)
        valid = F.interpolate(valid[None, None], size=probability.shape[-2:], mode="nearest")
        probability = probability * valid
    return F.adaptive_avg_pool2d(probability, (side, side)).flatten()


def compute_attention_rollout(
    model: nn.Module, image: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calcula o Attention Rollout de um patch para o SegFormer.

    ``image`` tem forma (C, H, W) ou (1, C, H, W). A atenção é extraída do
    backbone com `output_attentions=True` em modo `eager`, restrita à última
    etapa e agregada por cabeça. Os tokens de consulta são ponderados pela
    probabilidade predita de café (restrita aos pixels válidos de ``valid_mask``),
    de modo que o mapa indique quais tokens sustentam as consultas de café. O mapa
    é reamostrado para a resolução da entrada e normalizado em [0, 1] apenas sobre
    os pixels válidos. Retorna o mapa float32 e seus metadados de escopo.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError("Entrada do Attention Rollout deve ter forma (C,H,W) ou (1,C,H,W).")
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise ValueError("Modelo sem backbone `model` para o Attention Rollout.")
    implementation = _ensure_eager_attention(backbone)
    with torch.no_grad():
        outputs = backbone(pixel_values=image, output_attentions=True)
    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        raise RuntimeError(
            "O SegFormer não retornou atenções; verifique a implementação `eager` "
            "e `output_attentions=True`."
        )
    depths = getattr(getattr(backbone, "config", None), "depths", None)
    if not depths:
        raise ValueError("Configuração do SegFormer sem `depths`.")
    stage_attentions = last_stage_attentions(attentions, depths)
    tokens = int(stage_attentions[0].shape[-1])
    side = int(round(math.sqrt(tokens)))
    if side * side != tokens:
        raise ValueError(
            f"Relevância com {tokens} tokens não forma uma grade quadrada; "
            "a última etapa do SegFormer deve ser espacial."
        )
    query_weights = _query_weights(outputs.logits, valid_mask, side)
    relevance = evaluate_rollout_attentions(stage_attentions, query_weights=query_weights)
    grid = relevance.reshape(side, side)[None, None]
    rollout = F.interpolate(
        grid,
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    valid = (
        torch.ones_like(rollout, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(device=rollout.device, dtype=torch.bool)
    )
    if not bool(valid.any()):
        valid = torch.ones_like(rollout, dtype=torch.bool)
    minimum = float(rollout[valid].min())
    maximum = float(rollout[valid].max())
    if maximum <= minimum:
        normalized = np.zeros(tuple(image.shape[-2:]), dtype=np.float32)
    else:
        scaled = ((rollout - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
        masked = torch.where(valid, scaled, torch.zeros_like(scaled))
        normalized = masked.cpu().numpy().astype(np.float32)
    info = {
        "rollout_scope": ROLLOUT_SCOPE,
        "stage_index": len(depths) - 1,
        "stage_layers": len(stage_attentions),
        "total_layers": len(attentions),
        "stage_grid": [side, side],
        "num_heads": int(stage_attentions[0].shape[1]),
        "head_reduction": HEAD_REDUCTION,
        "token_reduction": TOKEN_REDUCTION,
        "query_weighting": QUERY_WEIGHTING,
        "residual": RESIDUAL,
        "normalization": NORMALIZATION,
        "attention_implementation": implementation,
        "input_shape": [int(image.shape[-2]), int(image.shape[-1])],
        "upsample_mode": "bilinear",
    }
    return normalized, info


def attention_rollout_fingerprint(
    summary: dict[str, Any], selected: list[dict[str, Any]], source_name: str
) -> dict[str, Any]:
    """Fingerprint que valida a reutilização dos mapas Attention Rollout."""
    weights = common.weights_by_fold(summary)
    used_folds = sorted({int(item["fold"]) for item in selected})
    return {
        "schema_version": ATTENTION_ROLLOUT_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "model_variant": str(get_config()["model"]["segformer_variant"]),
        "source_name": source_name,
        "rollout_scope": ROLLOUT_SCOPE,
        "head_reduction": HEAD_REDUCTION,
        "token_reduction": TOKEN_REDUCTION,
        "query_weighting": QUERY_WEIGHTING,
        "residual": RESIDUAL,
        "normalization": NORMALIZATION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
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


def _persist_maps(path: Path, samples: list[AttentionRolloutSample]) -> None:
    """Persiste os mapas Attention Rollout comprimidos com identificadores e dobras."""
    from src import io

    buffer = stdlib_io.BytesIO()
    np.savez_compressed(
        buffer,
        patch_ids=np.asarray([sample.patch_id for sample in samples], dtype=str),
        folds=np.asarray([sample.fold for sample in samples], dtype=np.int64),
        rollouts=np.stack([sample.rollout for sample in samples], axis=0),
        iou=np.asarray([sample.iou for sample in samples], dtype=np.float64),
        coffee_ratio=np.asarray([sample.coffee_ratio for sample in samples], dtype=np.float64),
    )
    io.persist_bytes(path, buffer.getvalue())


def run_attention_rollout(
    storage_paths: dict[str, Path],
    source_name: str | None = None,
    device: torch.device | None = None,
    max_patches: int = 4,
    force: bool = False,
) -> AttentionRolloutResult:
    """Gera (ou reutiliza) os mapas Attention Rollout do SegFormer na fonte escolhida."""
    from src import io

    source = source_name or chosen_source()
    if not summary_is_current(storage_paths, MODEL_NAME, source):
        raise ValueError(
            "Resumo do estágio 11 desatualizado ou incompleto para o SegFormer; "
            "reexecute a avaliação antes do Attention Rollout."
        )
    summary = load_evaluation_summary(storage_paths, MODEL_NAME, source)
    patch_metrics = load_model_patch_metrics(storage_paths, MODEL_NAME, source)
    selected = common.select_representative_patches(patch_metrics, max_patches=max_patches)
    if not selected:
        raise ValueError(f"Nenhum patch disponível para o Attention Rollout na fonte {source}.")
    fingerprint = attention_rollout_fingerprint(summary, selected, source)
    maps_file = attention_rollout_maps_path(storage_paths, source)
    metadata_file = attention_rollout_metadata_path(storage_paths, source)
    figure_file = attention_rollout_figure_path(storage_paths, source)
    if not force and common.artifacts_are_current(
        metadata_file, (maps_file, figure_file), fingerprint
    ):
        metadata = json.loads(io.ensure_local_copy(metadata_file).read_text(encoding="utf-8"))
        print("Mapas Attention Rollout já vigentes (artefatos reutilizados).")
        return AttentionRolloutResult(
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
    weights = common.weights_by_fold(summary)
    models: dict[int, nn.Module] = {}
    samples: list[AttentionRolloutSample] = []
    rollout_info: dict[str, Any] | None = None
    for item in selected:
        patch = str(item["patch_id"])
        fold = int(item["fold"])
        if fold not in models:
            models[fold] = load_model(MODEL_NAME, storage_paths, source, fold, device)
        model = models[fold]
        sample = common.load_patch_sample(storage_paths, manifest, patch, stats)
        rollout, info = compute_attention_rollout(
            model,
            sample.normalized.unsqueeze(0).to(device),
            valid_mask=torch.from_numpy(sample.valid).to(device),
        )
        rollout_info = rollout_info or info
        samples.append(
            AttentionRolloutSample(
                patch_id=patch,
                label=str(item["label"]),
                fold=fold,
                iou=float(item["iou"]),
                coffee_ratio=float(item["coffee_ratio"]),
                image_rgb=common.patch_rgb(sample),
                reference=sample.reference,
                rollout=rollout,
                weights_sha256=weights[fold],
            )
        )

    metadata = {
        "schema_version": ATTENTION_ROLLOUT_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "source_name": source,
        "device": str(device),
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
        "rollout": rollout_info,
        "fingerprint": fingerprint,
    }
    _persist_maps(maps_file, samples)
    io.persist_json(metadata_file, metadata)
    figstyle.persist_figure(figure_file, render_attention_rollout_figure(samples))
    print(
        f"Attention Rollout: {len(samples)} patches | dobras "
        f"{sorted({sample.fold for sample in samples})} | {device}"
    )
    print(f"Figura salva em: {figure_file}")
    return AttentionRolloutResult(
        source_name=source,
        samples=samples,
        metadata=metadata,
        maps_path=maps_file,
        metadata_path=metadata_file,
        figure_path=figure_file,
        skipped=False,
    )


def render_attention_rollout_figure(
    samples: list[AttentionRolloutSample],
) -> dict[str, bytes]:
    """Por recorte: imagem RGB, referência, Attention Rollout e sobreposição."""
    return common.render_heatmap_figure(
        samples, [sample.rollout for sample in samples], "Attention Rollout"
    )
