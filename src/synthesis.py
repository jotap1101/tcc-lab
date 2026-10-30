"""Síntese dos resultados para a monografia (estágio 15).

Consolida os artefatos produzidos pelos estágios 06–14 em tabelas (CSV e LaTeX)
e figuras de síntese sob a raiz canônica `tcc/`. As tabelas cobrem o dataset por
fonte, o treinamento por modelo/dobra, a avaliação por dobra e agregada, a
comparação estatística emparelhada e a relevância dos mapas de XAI dentro e fora
da região de referência de café. As figuras resumem as curvas de treino, as
métricas de avaliação/comparação e a relevância quantitativa dos mapas Grad-CAM
e Attention Rollout. A síntese não retreina nem reavalia: apenas lê artefatos
vigentes e é regenerada quando qualquer entrada muda (fingerprint).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import comparative, evaluation, trainer
from src.config import get_config
from src.data.mask_utils import reference_year
from src.data.patch_generation import (
    load_manifest,
    manifest_path,
    patch_fingerprint_hash,
    require_manifest,
)
from src.data.source_decision import chosen_source
from src.data.spatial_split import fold_count
from src.xai import attention_rollout, common, gradcam

SYNTHESIS_SCHEMA_VERSION = 1
MODEL_ORDER = ("unet", "segformer")
MODEL_LABELS = {"unet": "U-Net", "segformer": "SegFormer"}
MODEL_COLORS = {"unet": "#1565c0", "segformer": "#6a1b9a"}
CORE_TABLES = (
    "dataset",
    "training",
    "evaluation_per_fold",
    "evaluation_aggregate",
    "comparative_paired",
    "comparative_per_fold",
)
FIGURE_NAMES = ("training_curves", "evaluation_overview")
XAI_COLUMNS = (
    "model",
    "patch_id",
    "label",
    "fold",
    "iou",
    "coffee_ratio",
    "relevance_inside",
    "relevance_outside",
    "relevance_difference",
    "weights_sha256",
    "explanation",
)


@dataclass(frozen=True)
class SynthesisResult:
    """Tabelas, figuras e metadados da síntese dos resultados."""

    source_name: str
    tables: dict[str, Path]
    figures: dict[str, Path]
    tables_data: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    summary_path: Path
    metadata_path: Path
    skipped: bool


def synthesis_dir(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Diretório canônico da síntese da fonte (derivado de artifacts/)."""
    root = storage_paths.get("artifacts_synthesis")
    if root is None:
        root = storage_paths["artifacts_metrics"].parent / "synthesis"
    return root / source_name


def table_path(
    storage_paths: dict[str, Path], source_name: str, table: str, suffix: str = "csv"
) -> Path:
    """Caminho de uma tabela da síntese (CSV ou LaTeX)."""
    return synthesis_dir(storage_paths, source_name) / "tables" / f"{table}.{suffix}"


def summary_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho do resumo consolidado da síntese (JSON)."""
    return synthesis_dir(storage_paths, source_name) / "summary.json"


def metadata_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho do metadata com fingerprint e inventário da síntese."""
    return synthesis_dir(storage_paths, source_name) / "synthesis_metadata.json"


def figure_file_name(kind: str, source_name: str) -> str:
    """Nome estável de uma figura de síntese."""
    config = get_config()
    return (
        f"synthesis_{kind}_{config['aoi']['region_code']}"
        f"_{reference_year(source_name)}_{source_name}.png"
    )


def figure_path(storage_paths: dict[str, Path], kind: str, source_name: str) -> Path:
    """Caminho de uma figura de síntese no diretório canônico de figuras."""
    return storage_paths["artifacts_figures"] / figure_file_name(kind, source_name)


def xai_metadata_path(storage_paths: dict[str, Path], source_name: str, model_name: str) -> Path:
    """Caminho do metadata de XAI do modelo (Grad-CAM ou Attention Rollout)."""
    if model_name == "unet":
        return gradcam.gradcam_metadata_path(storage_paths, source_name)
    if model_name == "segformer":
        return attention_rollout.attention_rollout_metadata_path(storage_paths, source_name)
    raise ValueError(f"Modelo sem XAI: {model_name}.")


def xai_maps_path(storage_paths: dict[str, Path], source_name: str, model_name: str) -> Path:
    """Caminho dos mapas de XAI persistidos do modelo."""
    if model_name == "unet":
        return gradcam.gradcam_maps_path(storage_paths, source_name)
    if model_name == "segformer":
        return attention_rollout.attention_rollout_maps_path(storage_paths, source_name)
    raise ValueError(f"Modelo sem XAI: {model_name}.")


def _read_json(path: Path) -> dict[str, Any]:
    """Lê um JSON local/canonizado pelo helper de I/O do projeto."""
    from src import io

    return json.loads(io.ensure_local_copy(path).read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    """Hash SHA-256 curto de um arquivo (detecta mudanças de entrada)."""
    from src import io

    return hashlib.sha256(io.ensure_local_copy(path).read_bytes()).hexdigest()[:12]


def _xai_is_current(metadata: dict[str, Any], summary: dict[str, Any], model_name: str) -> bool:
    """Indica se o metadata de XAI corresponde ao resumo vigente do estágio 11."""
    if metadata.get("model") != model_name:
        return False
    fingerprint = metadata.get("fingerprint", {})
    current = summary.get("fingerprint", {})
    for key in ("manifest_fingerprint_hash", "split_fingerprint_hash", "config_hash", "threshold"):
        if fingerprint.get(key) != current.get(key):
            return False
    weights = {int(item["fold"]): str(item["weights_sha256"]) for item in summary["per_fold"]}
    for patch in metadata.get("patches", []):
        if weights.get(int(patch["fold"])) != str(patch["weights_sha256"]):
            return False
    return True


def validate_inputs(
    storage_paths: dict[str, Path], source_name: str | None = None
) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    """Valida os artefatos de entrada e retorna fonte, resumos, comparação e modelos XAI.

    Exige os resumos vigentes do estágio 11 para ambos os modelos, a comparação
    do estágio 12 compatível com esses resumos e os artefatos de treino das dez
    dobras. O XAI é opcional: metadados presentes precisam estar completos e
    vigentes; modelos sem metadados são omitidos da síntese.
    """
    from src import io

    source = source_name or chosen_source()
    require_manifest(storage_paths, source)
    summaries = comparative.load_evaluation_summaries(storage_paths, source)
    if not comparative.summaries_are_current(storage_paths, source, summaries):
        raise ValueError(
            "Artefatos do estágio 11 desatualizados ou incompletos; "
            "reexecute a avaliação antes da síntese."
        )
    comparison_file = comparative.comparison_path(storage_paths, source)
    if not io.path_exists(comparison_file):
        raise FileNotFoundError(
            f"Comparação do estágio 12 não encontrada: {comparison_file}. "
            "Execute o estágio 12 antes da síntese."
        )
    comparison = _read_json(comparison_file)
    expected = comparative.comparison_fingerprint(summaries, source)
    if comparison.get("fingerprint") != expected:
        raise ValueError(
            "Comparação do estágio 12 desatualizada frente ao estágio 11; "
            "reexecute o estágio 12 antes da síntese."
        )
    for model_name in MODEL_ORDER:
        for fold in range(fold_count()):
            if not trainer.fold_training_done(storage_paths, model_name, source, fold):
                raise FileNotFoundError(
                    f"Treino ausente para {model_name}, dobra {fold}; "
                    "execute os estágios 09/10 antes da síntese."
                )
    xai_models: list[str] = []
    for model_name in MODEL_ORDER:
        metadata_file = xai_metadata_path(storage_paths, source, model_name)
        if not io.path_exists(metadata_file):
            continue
        maps_file = xai_maps_path(storage_paths, source, model_name)
        if not io.path_exists(maps_file):
            raise FileNotFoundError(
                f"Mapas de XAI ausentes para {model_name}: {maps_file}. "
                "Reexecute o estágio 13/14 correspondente."
            )
        metadata = _read_json(metadata_file)
        if not _xai_is_current(metadata, summaries[model_name], model_name):
            raise ValueError(
                f"Mapas de XAI de {model_name} desatualizados frente ao estágio 11; "
                "reexecute o estágio 13/14 correspondente."
            )
        xai_models.append(model_name)
    return source, summaries, comparison, xai_models


def _dataset_fingerprint(storage_paths: dict[str, Path]) -> dict[str, str]:
    """Fingerprint dos manifestos disponíveis das fontes habilitadas."""
    from src import io

    config = get_config()
    result: dict[str, str] = {}
    for source_name, source_cfg in config["ground_truth"]["sources"].items():
        if not source_cfg.get("enabled"):
            continue
        if io.path_exists(manifest_path(storage_paths, source_name)):
            result[source_name] = patch_fingerprint_hash(storage_paths, source_name)
    return result


def _training_fingerprint(
    storage_paths: dict[str, Path], source_name: str
) -> dict[str, dict[str, dict[str, str]]]:
    """Digests das métricas e do histórico de cada modelo/dobra."""
    result: dict[str, dict[str, dict[str, str]]] = {}
    for model_name in MODEL_ORDER:
        result[model_name] = {}
        for fold in range(fold_count()):
            result[model_name][str(fold)] = {
                "metrics_sha256": _digest(
                    trainer.metrics_path(storage_paths, model_name, source_name, fold)
                ),
                "history_sha256": _digest(
                    trainer.history_path(storage_paths, model_name, source_name, fold)
                ),
            }
    return result


def _xai_fingerprint(
    storage_paths: dict[str, Path], source_name: str, xai_models: list[str]
) -> dict[str, dict[str, str]]:
    """Fingerprint dos metadados e mapas de XAI incluídos na síntese."""
    result: dict[str, dict[str, str]] = {}
    for model_name in xai_models:
        metadata = _read_json(xai_metadata_path(storage_paths, source_name, model_name))
        result[model_name] = {
            "fingerprint": json.dumps(metadata.get("fingerprint"), sort_keys=True),
            "maps_sha256": _digest(xai_maps_path(storage_paths, source_name, model_name)),
        }
    return result


def synthesis_fingerprint(
    storage_paths: dict[str, Path],
    source_name: str,
    summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    xai_models: list[str],
) -> dict[str, Any]:
    """Fingerprint que valida a reutilização das tabelas e figuras da síntese."""
    return {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "source_name": source_name,
        "config_hash": trainer.config_hash(),
        "dataset": _dataset_fingerprint(storage_paths),
        "evaluation": {
            model_name: summaries[model_name]["fingerprint"] for model_name in MODEL_ORDER
        },
        "training": _training_fingerprint(storage_paths, source_name),
        "comparison": comparison["fingerprint"],
        "xai": _xai_fingerprint(storage_paths, source_name, xai_models),
    }


def build_dataset_table(storage_paths: dict[str, Path]) -> pd.DataFrame:
    """Resume os manifestos por fonte: patches, dobras e distribuição de café."""
    from src import io

    config = get_config()
    chosen = chosen_source()
    rows: list[dict[str, Any]] = []
    for source_name, source_cfg in config["ground_truth"]["sources"].items():
        if not source_cfg.get("enabled"):
            continue
        if not io.path_exists(manifest_path(storage_paths, source_name)):
            continue
        manifest = load_manifest(storage_paths, source_name)
        folds = int(manifest["fold"].nunique()) if "fold" in manifest.columns else 0
        valid = manifest["valid_fraction"] if "valid_fraction" in manifest.columns else None
        rows.append(
            {
                "source": source_name,
                "year": reference_year(source_name),
                "chosen": source_name == chosen,
                "n_patches": int(len(manifest)),
                "n_folds": folds,
                "coffee_ratio_mean": float(manifest["coffee_ratio"].mean()),
                "coffee_ratio_median": float(manifest["coffee_ratio"].median()),
                "valid_fraction_mean": float(valid.mean()) if valid is not None else np.nan,
                "valid_fraction_min": float(valid.min()) if valid is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_training_table(storage_paths: dict[str, Path], source_name: str) -> pd.DataFrame:
    """Resume as métricas e o melhor epoch de treino por modelo/dobra."""
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        for fold in range(fold_count()):
            payload = _read_json(trainer.metrics_path(storage_paths, model_name, source_name, fold))
            train_metrics = payload["train"]
            val_metrics = payload["val"]
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "epochs": int(payload["epochs"]),
                    "best_epoch": int(payload["best_epoch"]),
                    "train_loss": float(train_metrics["loss"]),
                    "train_iou": float(train_metrics["iou"]),
                    "train_f1": float(train_metrics["f1"]),
                    "train_precision": float(train_metrics["precision"]),
                    "train_recall": float(train_metrics["recall"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_iou": float(val_metrics["iou"]),
                    "val_f1": float(val_metrics["f1"]),
                    "val_precision": float(val_metrics["precision"]),
                    "val_recall": float(val_metrics["recall"]),
                }
            )
    return pd.DataFrame(rows)


def build_evaluation_per_fold_table(
    summaries: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Resume as métricas e contagens de avaliação por modelo/dobra."""
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        for item in summaries[model_name]["per_fold"]:
            counts = item["counts"]
            rows.append(
                {
                    "model": model_name,
                    "fold": int(item["fold"]),
                    "n_patches": int(item["n_patches"]),
                    "tp": int(counts["tp"]),
                    "fp": int(counts["fp"]),
                    "fn": int(counts["fn"]),
                    **{key: float(item["metrics"][key]) for key in evaluation.METRIC_KEYS},
                }
            )
    return pd.DataFrame(rows)


def build_evaluation_aggregate_table(
    summaries: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Resume o agregado micro e a média/desvio por dobra de cada modelo."""
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        summary = summaries[model_name]
        counts = summary["aggregate"]["counts"]
        metrics = summary["aggregate"]["metrics"]
        row: dict[str, Any] = {
            "model": model_name,
            "n_patches": int(summary["n_patches"]),
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
        }
        for key in evaluation.METRIC_KEYS:
            row[f"{key}_micro"] = float(metrics[key])
            row[f"{key}_mean"] = float(summary["fold_mean"][key])
            row[f"{key}_std"] = float(summary["fold_std"][key])
        rows.append(row)
    return pd.DataFrame(rows)


def build_comparative_paired_table(comparison: dict[str, Any]) -> pd.DataFrame:
    """Resume as estatísticas emparelhadas por métrica (SegFormer − U-Net)."""
    rows = [dict(values) for values in comparison["paired"].values()]
    return pd.DataFrame(rows)


def build_comparative_per_fold_table(comparison: dict[str, Any]) -> pd.DataFrame:
    """Resume o delta por dobra das quatro métricas (SegFormer − U-Net)."""
    rows = []
    for item in comparison["per_fold_delta"]:
        rows.append(
            {
                "fold": int(item["fold"]),
                "n_patches": int(item["n_patches"]),
                **{f"delta_{key}": float(item["delta"][key]) for key in evaluation.METRIC_KEYS},
            }
        )
    return pd.DataFrame(rows)


def _xai_explanation(metadata: dict[str, Any], model_name: str) -> str:
    """Descrição curta do alvo da explicação (camada ou etapa do rollout)."""
    if model_name == "unet":
        return str(metadata.get("target_layer", "encoders[-1]"))
    rollout = metadata.get("rollout", {})
    return f"stage_{rollout.get('stage_index', 'last')}"


def build_xai_table(
    storage_paths: dict[str, Path], source_name: str, xai_models: list[str]
) -> pd.DataFrame:
    """Calcula a relevância média dentro e fora da referência de café por mapa."""
    from src import io

    if not xai_models:
        return pd.DataFrame(columns=list(XAI_COLUMNS))
    stats, manifest = common.load_stats_and_manifest(storage_paths, source_name)
    cache: dict[str, common.PatchSample] = {}
    rows: list[dict[str, Any]] = []
    for model_name in xai_models:
        metadata = _read_json(xai_metadata_path(storage_paths, source_name, model_name))
        with np.load(
            io.ensure_local_copy(xai_maps_path(storage_paths, source_name, model_name))
        ) as data:
            key = "cams" if model_name == "unet" else "rollouts"
            heatmaps = {
                str(patch_id): data[key][index] for index, patch_id in enumerate(data["patch_ids"])
            }
        explanation = _xai_explanation(metadata, model_name)
        for item in metadata["patches"]:
            patch_id = str(item["patch_id"])
            if patch_id not in cache:
                cache[patch_id] = common.load_patch_sample(storage_paths, manifest, patch_id, stats)
            sample = cache[patch_id]
            if patch_id not in heatmaps:
                raise ValueError(f"Mapa de XAI ausente para o patch {patch_id} ({model_name}).")
            heatmap = heatmaps[patch_id]
            if heatmap.shape != sample.reference.shape:
                raise ValueError(
                    f"Mapa de XAI {heatmap.shape} incompatível com o patch "
                    f"{sample.reference.shape} ({patch_id})."
                )
            inside = float(heatmap[sample.reference].mean()) if sample.reference.any() else None
            outside_mask = sample.valid & ~sample.reference
            outside = float(heatmap[outside_mask].mean()) if outside_mask.any() else None
            difference = (
                float(inside - outside) if inside is not None and outside is not None else None
            )
            rows.append(
                {
                    "model": model_name,
                    "patch_id": patch_id,
                    "label": str(item["label"]),
                    "fold": int(item["fold"]),
                    "iou": float(item["iou"]),
                    "coffee_ratio": float(item["coffee_ratio"]),
                    "relevance_inside": inside,
                    "relevance_outside": outside,
                    "relevance_difference": difference,
                    "weights_sha256": str(item["weights_sha256"]),
                    "explanation": explanation,
                }
            )
    return pd.DataFrame(rows, columns=list(XAI_COLUMNS))


def _load_histories(
    storage_paths: dict[str, Path], source_name: str
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Carrega o histórico de treino de cada modelo/dobra para as curvas."""
    histories: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for model_name in MODEL_ORDER:
        histories[model_name] = {}
        for fold in range(fold_count()):
            payload = _read_json(trainer.history_path(storage_paths, model_name, source_name, fold))
            histories[model_name][fold] = payload["epochs"]
    return histories


def _persist_table(
    storage_paths: dict[str, Path], source_name: str, name: str, table: pd.DataFrame
) -> None:
    """Persiste a tabela em CSV e LaTeX no diretório canônico da síntese."""
    from src import io

    io.persist_bytes(
        table_path(storage_paths, source_name, name, "csv"),
        table.to_csv(index=False).encode("utf-8"),
    )
    latex = table.to_latex(
        index=False,
        float_format="%.4f",
        na_rep="--",
        escape=True,
    )
    io.persist_bytes(
        table_path(storage_paths, source_name, name, "tex"),
        latex.encode("utf-8"),
    )


def _current_artifacts(
    storage_paths: dict[str, Path],
    source_name: str,
    fingerprint: dict[str, Any],
    table_names: list[str],
    figure_names: list[str],
) -> bool:
    """Indica se metadata, resumo, tabelas e figuras existem e estão vigentes."""
    from src import io

    metadata_file = metadata_path(storage_paths, source_name)
    summary_file = summary_path(storage_paths, source_name)
    paths = [metadata_file, summary_file]
    paths += [table_path(storage_paths, source_name, name, "csv") for name in table_names]
    paths += [table_path(storage_paths, source_name, name, "tex") for name in table_names]
    paths += [figure_path(storage_paths, name, source_name) for name in figure_names]
    if not all(io.path_exists(path) for path in paths):
        return False
    try:
        stored = _read_json(metadata_file)
    except (json.JSONDecodeError, OSError):
        return False
    return stored.get("fingerprint") == fingerprint


def run_synthesis(
    storage_paths: dict[str, Path],
    source_name: str | None = None,
    force: bool = False,
) -> SynthesisResult:
    """Gera (ou reutiliza) as tabelas e figuras de síntese da fonte escolhida."""
    source, summaries, comparison, xai_models = validate_inputs(storage_paths, source_name)
    fingerprint = synthesis_fingerprint(storage_paths, source, summaries, comparison, xai_models)
    table_names = list(CORE_TABLES)
    figure_names = list(FIGURE_NAMES)
    if xai_models:
        table_names.append("xai")
        figure_names.append("xai_relevance")
    if not force and _current_artifacts(
        storage_paths, source, fingerprint, table_names, figure_names
    ):
        summary = _read_json(summary_path(storage_paths, source))
        tables_data = {
            name: pd.read_csv(table_path(storage_paths, source, name, "csv"))
            for name in table_names
        }
        print("Síntese já vigente (tabelas e figuras reutilizadas).")
        return SynthesisResult(
            source_name=source,
            tables={name: table_path(storage_paths, source, name, "csv") for name in table_names},
            figures={name: figure_path(storage_paths, name, source) for name in figure_names},
            tables_data=tables_data,
            summary=summary,
            summary_path=summary_path(storage_paths, source),
            metadata_path=metadata_path(storage_paths, source),
            skipped=True,
        )

    from src import io

    tables = {
        "dataset": build_dataset_table(storage_paths),
        "training": build_training_table(storage_paths, source),
        "evaluation_per_fold": build_evaluation_per_fold_table(summaries),
        "evaluation_aggregate": build_evaluation_aggregate_table(summaries),
        "comparative_paired": build_comparative_paired_table(comparison),
        "comparative_per_fold": build_comparative_per_fold_table(comparison),
    }
    xai_table = build_xai_table(storage_paths, source, xai_models)
    if not xai_table.empty:
        tables["xai"] = xai_table
    for name, table in tables.items():
        _persist_table(storage_paths, source, name, table)

    histories = _load_histories(storage_paths, source)
    figures = {
        "training_curves": figure_path(storage_paths, "training_curves", source),
        "evaluation_overview": figure_path(storage_paths, "evaluation_overview", source),
    }
    io.persist_bytes(
        figures["training_curves"],
        render_training_curves(histories),
    )
    io.persist_bytes(
        figures["evaluation_overview"],
        render_evaluation_overview(summaries, comparison),
    )
    if not xai_table.empty:
        figures["xai_relevance"] = figure_path(storage_paths, "xai_relevance", source)
        io.persist_bytes(figures["xai_relevance"], render_xai_relevance(xai_table))

    xai_summary: dict[str, Any] = {"included": not xai_table.empty, "models": xai_models}
    if not xai_table.empty:
        xai_summary["mean_inside"] = float(xai_table["relevance_inside"].mean())
        xai_summary["mean_outside"] = float(xai_table["relevance_outside"].mean())
        xai_summary["mean_difference"] = float(xai_table["relevance_difference"].mean())
    summary = {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "source_name": source,
        "region_code": get_config()["aoi"]["region_code"],
        "year": reference_year(source),
        "config_hash": trainer.config_hash(),
        "n_sources": int(len(tables["dataset"])),
        "n_patches_chosen": int(
            tables["dataset"].loc[tables["dataset"]["source"] == source, "n_patches"].iloc[0]
        ),
        "models": {
            model_name: {
                "aggregate": summaries[model_name]["aggregate"]["metrics"],
                "fold_mean": summaries[model_name]["fold_mean"],
                "fold_std": summaries[model_name]["fold_std"],
            }
            for model_name in MODEL_ORDER
        },
        "aggregate_delta": comparison["aggregate_delta"],
        "paired": comparison["paired"],
        "xai": xai_summary,
        "tables": {
            name: str(path)
            for name, path in (
                (name, table_path(storage_paths, source, name, "csv")) for name in tables
            )
        },
        "figures": {name: str(path) for name, path in figures.items()},
    }
    io.persist_json(summary_path(storage_paths, source), summary)
    io.persist_json(
        metadata_path(storage_paths, source),
        {
            "schema_version": SYNTHESIS_SCHEMA_VERSION,
            "source_name": source,
            "fingerprint": fingerprint,
            "tables": list(tables),
            "figures": list(figures),
            "xai_models": xai_models,
        },
    )
    print(f"Síntese concluída: {len(tables)} tabelas | {len(figures)} figuras")
    for name, path in figures.items():
        print(f"Figura {name}: {path}")
    return SynthesisResult(
        source_name=source,
        tables={name: table_path(storage_paths, source, name, "csv") for name in tables},
        figures=figures,
        tables_data=tables,
        summary=summary,
        summary_path=summary_path(storage_paths, source),
        metadata_path=metadata_path(storage_paths, source),
        skipped=False,
    )


def render_training_curves(histories: dict[str, dict[int, list[dict[str, Any]]]]) -> bytes:
    """Renderiza perda de treino/validação, IoU e F1 por época para os dois modelos."""
    from matplotlib.lines import Line2D

    from src.data.raster_utils import render_figure

    def _build(plt: Any) -> Any:
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        panels = (
            ("train_loss", "Perda de treino"),
            ("val_loss", "Perda de validação"),
            ("val_iou", "IoU de validação"),
            ("val_f1", "F1 de validação"),
        )
        for axis, (key, title) in zip(axes.ravel(), panels, strict=True):
            for model_name in MODEL_ORDER:
                for _fold, epochs in histories[model_name].items():
                    axis.plot(
                        [int(item["epoch"]) for item in epochs],
                        [float(item[key]) for item in epochs],
                        color=MODEL_COLORS[model_name],
                        alpha=0.45,
                        linewidth=1.2,
                    )
            axis.set_title(title)
            axis.set_xlabel("época")
            axis.set_ylabel(key)
            axis.grid(alpha=0.2)
        handles = [
            Line2D([0], [0], color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name])
            for model_name in MODEL_ORDER
        ]
        axes[0, 1].legend(handles=handles, loc="best")
        figure.suptitle("Curvas de treino por dobra (validação espacial)", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        return figure

    return render_figure(_build)


def render_evaluation_overview(
    summaries: dict[str, dict[str, Any]], comparison: dict[str, Any]
) -> bytes:
    """Renderiza métricas agregadas, por dobra, delta e intervalo de confiança."""
    from src.data.raster_utils import render_figure

    metrics = list(evaluation.METRIC_KEYS)

    def _build(plt: Any) -> Any:
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))

        positions = np.arange(len(metrics))
        width = 0.38
        for offset, model_name in zip((-width / 2, width / 2), MODEL_ORDER, strict=True):
            axes[0, 0].bar(
                positions + offset,
                [summaries[model_name]["aggregate"]["metrics"][key] for key in metrics],
                width=width,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )
        axes[0, 0].set_xticks(positions, metrics)
        axes[0, 0].set_ylim(0.0, 1.05)
        axes[0, 0].set_title("Métricas agregadas (micro)")
        axes[0, 0].legend()

        folds = [str(int(item["fold"])) for item in summaries["unet"]["per_fold"]]
        fold_positions = np.arange(len(folds))
        for offset, model_name in zip((-width / 2, width / 2), MODEL_ORDER, strict=True):
            axes[0, 1].bar(
                fold_positions + offset,
                [item["metrics"]["iou"] for item in summaries[model_name]["per_fold"]],
                width=width,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )
        axes[0, 1].set_xticks(fold_positions, folds)
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[0, 1].set_title("IoU por dobra")
        axes[0, 1].set_xlabel("dobra")
        axes[0, 1].legend()

        per_fold = comparison["per_fold_delta"]
        axes[1, 0].bar(
            [str(int(item["fold"])) for item in per_fold],
            [item["delta"]["iou"] for item in per_fold],
            color="#6a1b9a",
        )
        axes[1, 0].axhline(0.0, color="black", linewidth=1)
        axes[1, 0].set_title("Delta de IoU por dobra (SegFormer − U-Net)")
        axes[1, 0].set_xlabel("dobra")
        axes[1, 0].set_ylabel("delta de IoU")

        means = []
        errors_low = []
        errors_high = []
        for key in metrics:
            result = comparison["paired"][key]
            mean = result.get("mean_difference")
            low = result.get("ci95_low")
            high = result.get("ci95_high")
            means.append(0.0 if mean is None else float(mean))
            errors_low.append(
                0.0 if mean is None or low is None else max(0.0, float(mean) - float(low))
            )
            errors_high.append(
                0.0 if mean is None or high is None else max(0.0, float(high) - float(mean))
            )
        axes[1, 1].errorbar(
            positions,
            means,
            yerr=[errors_low, errors_high],
            fmt="o",
            color="#e53935",
            capsize=5,
        )
        axes[1, 1].axhline(0.0, color="black", linewidth=1)
        axes[1, 1].set_xticks(positions, metrics)
        axes[1, 1].set_title("Diferença pareada com IC 95% (SegFormer − U-Net)")
        axes[1, 1].set_ylabel("diferença média")

        figure.suptitle("Avaliação e comparação dos modelos", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        return figure

    return render_figure(_build)


def render_xai_relevance(table: pd.DataFrame) -> bytes:
    """Renderiza a relevância média dentro/fora do café e a diferença por patch."""
    from src.data.raster_utils import render_figure

    models = [model_name for model_name in MODEL_ORDER if model_name in set(table["model"])]

    def _build(plt: Any) -> Any:
        figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        positions = np.arange(len(models))
        width = 0.38
        for index, model_name in enumerate(models):
            subset = table[table["model"] == model_name]
            inside = subset["relevance_inside"].astype(float)
            outside = subset["relevance_outside"].astype(float)
            axes[0].bar(
                index - width / 2,
                float(inside.mean()),
                width=width,
                color="#2e7d32",
                label="Dentro do café" if index == 0 else None,
            )
            axes[0].bar(
                index + width / 2,
                float(outside.mean()),
                width=width,
                color="#f9a825",
                label="Fora do café" if index == 0 else None,
            )
            axes[0].scatter(
                np.full(len(subset), index - width / 2),
                inside,
                color="#1b5e20",
                s=18,
                zorder=3,
            )
            axes[0].scatter(
                np.full(len(subset), index + width / 2),
                outside,
                color="#ef6c00",
                s=18,
                zorder=3,
            )
        axes[0].set_xticks(positions, [MODEL_LABELS[name] for name in models])
        axes[0].set_ylim(0.0, 1.05)
        axes[0].set_ylabel("relevância normalizada")
        axes[0].set_title("Relevância média dentro/fora da referência de café")
        axes[0].legend()

        for index, model_name in enumerate(models):
            subset = table[table["model"] == model_name]
            jitter = np.linspace(-0.08, 0.08, len(subset))
            axes[1].scatter(
                np.full(len(subset), index) + jitter,
                subset["relevance_difference"].astype(float),
                color=MODEL_COLORS[model_name],
                s=40,
            )
        axes[1].axhline(0.0, color="black", linewidth=1)
        axes[1].set_xticks(positions, [MODEL_LABELS[name] for name in models])
        axes[1].set_ylabel("relevância dentro − fora")
        axes[1].set_title("Diferença de relevância por patch explicado")

        figure.suptitle("Síntese quantitativa dos mapas de XAI", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        return figure

    return render_figure(_build)
