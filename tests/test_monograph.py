"""Testes de src/monograph.py (elementos visuais da monografia — estágio 17)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from src import monograph
from src.comparative import ErrorMapSample
from src.config import get_config
from src.data.dataset import normalize_patch
from src.metrics import f1_score, iou, precision, recall
from src.xai.common import PatchSample

TEST_SOURCE = "emater"
PATCH_SIZE = 32
FOLD_COUNT = 2


def _metrics(counts: dict[str, int]) -> dict[str, float]:
    """Deriva as métricas de pixel a partir das contagens de confusão."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        "iou": iou(tp, fp, fn),
        "f1": f1_score(tp, fp, fn),
        "precision": precision(tp, fp),
        "recall": recall(tp, fn),
    }


def _summary(model_name: str) -> dict[str, Any]:
    """Resumo sintético do estágio 11 para um modelo."""
    per_fold: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        counts = {"tp": 90 + fold, "fp": 10, "fn": 12}
        per_fold.append(
            {
                "fold": fold,
                "n_patches": 2,
                "counts": counts,
                "metrics": _metrics(counts),
                "weights_sha256": f"{model_name}{fold}",
            }
        )
    total = {key: sum(item["counts"][key] for item in per_fold) for key in ("tp", "fp", "fn")}
    return {
        "model": model_name,
        "source_name": TEST_SOURCE,
        "threshold": 0.5,
        "n_folds": FOLD_COUNT,
        "n_patches": 2 * FOLD_COUNT,
        "per_fold": per_fold,
        "aggregate": {"counts": total, "metrics": _metrics(total)},
        "fold_mean": {
            key: float(np.mean([item["metrics"][key] for item in per_fold]))
            for key in ("iou", "f1", "precision", "recall")
        },
        "fold_std": {key: 0.01 for key in ("iou", "f1", "precision", "recall")},
        "fingerprint": {"config_hash": "cfg"},
    }


def _comparison() -> dict[str, Any]:
    """Comparação sintética do estágio 12."""
    paired = {}
    for index, metric in enumerate(("iou", "f1", "precision", "recall")):
        paired[metric] = {
            "metric": metric,
            "mean_difference": 0.02 - 0.005 * index,
            "ci95_low": 0.01,
            "ci95_high": 0.03,
            "p_value_holm": 0.002,
            "cohens_dz": 0.4,
            "cliffs_delta": 0.3,
        }
    return {
        "paired": paired,
        "fingerprint": {"comparison": "hash"},
        "per_fold_delta": [
            {"fold": fold, "n_patches": 2, "delta": {"iou": 0.02 if fold else -0.01}}
            for fold in range(FOLD_COUNT)
        ],
    }


def _patch_metrics() -> pd.DataFrame:
    """Métricas por patch emparelhadas (colunas usadas pelas figuras)."""
    rows = []
    for fold in range(FOLD_COUNT):
        for index in range(3):
            ratio = [0.1, 0.35, 0.7][index]
            rows.append(
                {
                    "patch_id": f"patch_{fold}_{index}",
                    "fold": fold,
                    "coffee_ratio": ratio,
                    "truth_positive": True,
                    "iou_unet": 0.5 + 0.1 * index,
                    "iou_segformer": 0.55 + 0.1 * index,
                }
            )
    return pd.DataFrame(rows)


def _sample(index: int) -> PatchSample:
    """Patch sintético (imagem de 4 bandas, máscara e pixels válidos)."""
    rng = np.random.default_rng(index)
    image = rng.uniform(0.0, 1.0, size=(4, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    mask[6:24, 8:26] = 1
    valid = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    valid[:2, :] = False
    stats = {
        "bands": ["B2", "B3", "B4", "B8"],
        "per_band": {band: {"mean": 0.0, "std": 1.0} for band in ("B2", "B3", "B4", "B8")},
    }
    return PatchSample(
        patch_id=f"patch_{index}",
        fold=index % FOLD_COUNT,
        image=image * valid,
        mask=mask,
        valid=valid,
        normalized=torch.from_numpy(normalize_patch(image, stats)),
        reference=(mask > 0) & valid,
    )


def _error_samples() -> list[ErrorMapSample]:
    """Amostras de erro para o painel qualitativo."""
    samples = []
    for index, label in enumerate(("Maior IoU", "Menor IoU")):
        sample = _sample(index)
        prediction = sample.reference.copy()
        prediction[:4, :4] = True
        other = sample.reference.copy()
        other[28:, 28:] = False
        samples.append(
            ErrorMapSample(
                patch_id=sample.patch_id,
                label=label,
                fold=sample.fold,
                image_rgb=np.dstack([sample.image[2], sample.image[1], sample.image[0]]) * 255,
                truth=sample.reference,
                valid=sample.valid,
                predictions={"unet": prediction & sample.valid, "segformer": other & sample.valid},
                iou={"unet": 0.6, "segformer": 0.7},
            )
        )
    return samples


def _xai_table() -> pd.DataFrame:
    """Tabela de relevância de XAI para os dois modelos."""
    rows = []
    for model_name in ("unet", "segformer"):
        rows.append(
            {
                "model": model_name,
                "patch_id": "patch_0",
                "label": "Maior IoU",
                "fold": 0,
                "iou": 0.6,
                "coffee_ratio": 0.5,
                "relevance_inside": 0.6,
                "relevance_outside": 0.3,
                "relevance_difference": 0.3,
                "weights_sha256": "hash",
                "explanation": "encoders[-1]",
            }
        )
    return pd.DataFrame(rows)


def _dataset_table() -> pd.DataFrame:
    """Resumo sintético do dataset por fonte (saída do estágio 15)."""
    return pd.DataFrame(
        [
            {
                "source": "emater",
                "year": 2018,
                "chosen": True,
                "n_patches": 6,
                "n_folds": FOLD_COUNT,
                "coffee_ratio_mean": 0.31,
                "coffee_ratio_median": 0.28,
                "valid_fraction_mean": 0.97,
                "valid_fraction_min": 0.4,
            }
        ]
    )


def _study_area() -> monograph.StudyAreaData:
    """Geometrias sintéticas para o mapa de localização."""
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    return monograph.StudyAreaData(
        mg_polygons=[square],
        aoi_polygons=[square * 25000],
        coffee_mask=np.ones((7, 7), dtype=np.uint8),
        coffee_extent=(-1000.0, 1000.0, -1000.0, 1000.0),
        region_name="Guaxupé",
        region_area_km2=1750.0,
        coffee_area_km2=310.0,
        n_regions=70,
    )


def _label_sources() -> monograph.LabelSourcesData:
    """Máscaras sintéticas das três fontes de referência."""
    rng = np.random.default_rng(0)
    masks = {name: rng.random((40, 40)) > 0.8 for name in ("mapbiomas", "alphaearth", "emater")}
    return monograph.LabelSourcesData(
        masks=masks,
        years={"mapbiomas": 2025, "alphaearth": 2024, "emater": 2018},
        coffee_share={"mapbiomas": 0.2, "alphaearth": 0.2, "emater": 0.2},
        kappa_vs_chosen={"mapbiomas": 0.4, "alphaearth": 0.35},
        chosen="emater",
    )


def _manifest() -> pd.DataFrame:
    """Manifesto sintético com dobras, linhas e colunas."""
    return pd.DataFrame(
        [
            {
                "patch_id": f"patch_{fold}_{index}",
                "fold": fold,
                "row": fold,
                "col": index,
                "coffee_ratio": 0.3,
                "valid_fraction": 1.0,
                "bbox": "0,0,1,1",
                "mask_source": TEST_SOURCE,
                "image_path": "image.npy",
                "mask_path": "mask.npy",
            }
            for fold in range(FOLD_COUNT)
            for index in range(3)
        ]
    )


def _panel() -> monograph.XaiPanel:
    """Painel de XAI sintético com o mapa de calor de cada modelo."""
    sample = _sample(0)
    heatmap = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    heatmap[6:24, 8:26] = 0.9
    return monograph.XaiPanel(
        patch_id=sample.patch_id,
        fold=0,
        image_rgb=np.dstack([sample.image[2], sample.image[1], sample.image[0]]) * 255,
        reference=sample.reference,
        heatmaps={"unet": heatmap, "segformer": heatmap * 0.8},
        iou={"unet": 0.6, "segformer": 0.7},
    )


def _data() -> monograph.MonographData:
    """Dados sintéticos completos para exercitar todas as renderizações."""
    summaries = {name: _summary(name) for name in monograph.MODEL_ORDER}
    return monograph.MonographData(
        source_name=TEST_SOURCE,
        config=get_config(),
        manifest=_manifest(),
        summaries=summaries,
        comparison=_comparison(),
        patch_metrics=_patch_metrics(),
        dataset_table=_dataset_table(),
        xai_table=_xai_table(),
        xai_models=list(monograph.MODEL_ORDER),
        study_area=_study_area(),
        label_sources=_label_sources(),
        showcase=_sample(0),
        error_samples=_error_samples(),
        xai_panel=_panel(),
    )


def test_catalog_covers_all_visual_elements() -> None:
    """O catálogo deve ter as 16 figuras e as 7 tabelas com legendas editoriais."""
    assert len(monograph.FIGURE_CATALOG) == 16
    assert len(monograph.TABLE_CATALOG) == 7
    assert set(monograph.FIGURE_RENDERERS) == set(monograph.FIGURE_CATALOG)
    for spec in monograph.FIGURE_CATALOG.values():
        assert spec.section in {"Metodologia", "Resultados"}
        assert spec.caption.startswith(spec.caption[0].upper())
        assert spec.caption.endswith(".")
    assert (
        sum(1 for spec in monograph.FIGURE_CATALOG.values() if spec.section == "Metodologia") == 8
    )


def test_number_formatting_uses_decimal_comma() -> None:
    """Números e porcentagens devem usar vírgula decimal."""
    assert monograph._number(0.9123) == "0,912"
    assert monograph._number(None) == "—"
    assert monograph._percent(0.314) == "31,4%"
    assert monograph._pvalue(0.0004) == "< 0,001"
    assert monograph._interval(0.01, 0.03) == "[0,010; 0,030]"
    assert monograph._signed_points(0.043) == "+4,3 p.p."


def test_interpret_difference_combines_magnitude_and_significance() -> None:
    """A interpretação deve unir magnitude e significância em linguagem comum."""
    assert monograph._interpret_difference(0.004, 0.2) == "desprezível (não significativa)"
    assert monograph._interpret_difference(0.02, 0.001) == "pequena (significativa)"
    assert monograph._interpret_difference(0.06, 0.01) == "relevante (significativa)"
    assert monograph._interpret_difference(None, None) == "—"


def test_pixel_area_is_derived_from_export_scale() -> None:
    """A área do pixel deve derivar da resolução de exportação (10 m → 0,01 ha)."""
    assert monograph.pixel_area_ha(get_config()) == pytest.approx(0.01)


def test_tables_translate_numbers_to_common_language() -> None:
    """As tabelas devem trazer cabeçalhos em pt-BR, hectares e interpretação."""
    data = _data()
    tables = monograph.build_tables(data)

    assert set(tables) == {
        "dataset",
        "protocol",
        "metric_glossary",
        "model_results",
        "paired_comparison",
        "area_confusion",
        "xai_relevance",
    }
    assert list(tables["dataset"]["Fonte"]) == ["Emater"]
    assert tables["dataset"].loc[0, "Café médio"] == "31,0%"
    assert "Épocas por dobra" in set(tables["protocol"]["Item"])
    assert "IoU" in set(tables["model_results"].columns)
    assert "±" in tables["model_results"].loc[0, "IoU"]
    assert "significativa" in tables["paired_comparison"].loc[0, "Interpretação"]
    assert tables["paired_comparison"].loc[0, "p (Holm)"] == "0,002"

    area = tables["area_confusion"]
    reference = sum(
        int(item["counts"]["tp"]) + int(item["counts"]["fn"])
        for item in data.summaries["unet"]["per_fold"]
    )
    assert float(area.loc[0, "Café de referência (ha)"].replace(",", ".")) == pytest.approx(
        reference * 0.01, abs=0.05
    )
    assert area.loc[0, "Erro relativo"].endswith("%")

    assert list(tables["xai_relevance"]["Modelo"]) == ["U-Net", "SegFormer"]


def test_dataset_table_marks_chosen_source() -> None:
    """A fonte escolhida deve ser marcada explicitamente na tabela."""
    table = monograph.build_dataset_table(_data())
    assert table.loc[0, "Fonte de treino"] == "sim"
    assert table.loc[0, "Ano"] == 2018


def test_available_figures_and_tables_respect_inputs() -> None:
    """Sem insumos opcionais, as figuras e tabelas dependentes devem ser omitidas."""
    data = _data()
    assert "xai_panel" in monograph.available_figures(data)
    data.xai_table = data.xai_table.iloc[0:0]
    assert "xai_relevance" not in monograph.available_tables(data)

    minimal = _data()
    minimal.study_area = None
    minimal.label_sources = None
    minimal.showcase = None
    minimal.error_samples = []
    minimal.xai_panel = None
    figures = monograph.available_figures(minimal)
    assert "study_area" not in figures
    assert "label_sources" not in figures
    assert "bands" not in figures
    assert "patch_anatomy" not in figures
    assert "qualitative_predictions" not in figures
    assert "xai_panel" not in figures
    assert "pipeline" in figures


@pytest.mark.parametrize("key", list(monograph.FIGURE_RENDERERS))
def test_every_figure_renders_png_and_pdf(key: str) -> None:
    """Todas as figuras devem renderizar PNG e PDF não vazios."""
    data = _data()
    context = monograph.RenderContext(width_cm=16.0, dpi=60, formats=("png", "pdf"))

    payloads = monograph.FIGURE_RENDERERS[key](data, context)

    assert payloads["png"].startswith(b"\x89PNG")
    assert payloads["pdf"].startswith(b"%PDF")
    assert len(payloads["png"]) > 1000


def test_captions_include_numbering_sections_and_labels() -> None:
    """As legendas devem numerar figuras/tabelas e gerar rótulos estáveis."""
    figures = list(monograph.FIGURE_CATALOG)
    tables = list(monograph.TABLE_CATALOG)

    latex = monograph.build_captions_latex(TEST_SOURCE, figures, tables)
    assert "\\label{fig:study_area}" in latex
    assert "\\label{tab:dataset}" in latex
    assert "\\includegraphics" in latex
    assert "\\input{artifacts/monograph/emater/tables/dataset.tex}" in latex

    markdown = monograph.build_captions_markdown(TEST_SOURCE, figures, tables)
    assert "**Figura 1 — study_area**" in markdown
    assert "**Tabela 1 — dataset**" in markdown


def _storage_paths(tmp_path: Path) -> dict[str, Path]:
    """Caminhos de armazenamento mínimos com a chave da monografia."""
    return {
        "data_processed": tmp_path / "data" / "processed",
        "data_interim": tmp_path / "data" / "interim",
        "data_processed_composites": tmp_path / "data" / "processed" / "composites",
        "data_processed_patches": tmp_path / "data" / "processed" / "patches",
        "models": tmp_path / "models",
        "artifacts_metrics": tmp_path / "artifacts" / "metrics",
        "artifacts_figures": tmp_path / "artifacts" / "figures",
        "artifacts_monograph": tmp_path / "artifacts" / "monograph",
        "artifacts_runs": tmp_path / "artifacts" / "runs",
    }


def test_run_monograph_persists_and_reuses_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O estágio deve gravar figuras/tabelas/legendas e reutilizá-las pelo fingerprint."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    data = _data()
    summaries = data.summaries
    comparison = data.comparison
    monkeypatch.setattr(
        "src.synthesis.validate_inputs",
        lambda storage, source: (TEST_SOURCE, summaries, comparison, ["unet", "segformer"]),
    )
    monkeypatch.setattr("src.monograph.build_monograph_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        "src.monograph.monograph_fingerprint",
        lambda *args, **kwargs: {"schema_version": 1, "hash": "fixed"},
    )

    result = monograph.run_monograph(paths, TEST_SOURCE)

    assert not result.skipped
    assert len(result.figures) == len(monograph.FIGURE_CATALOG)
    assert len(result.tables) == len(monograph.TABLE_CATALOG)
    for files in result.figures.values():
        assert files["png"].is_file()
        assert files["pdf"].is_file()
    for name in result.tables:
        assert monograph.table_path(paths, TEST_SOURCE, name, "csv").is_file()
        assert monograph.table_path(paths, TEST_SOURCE, name, "tex").is_file()
    assert result.captions_tex_path.is_file()
    assert result.captions_md_path.is_file()
    assert "Figura 1" in result.captions_md_path.read_text(encoding="utf-8")

    inventory = json.loads(result.inventory_path.read_text(encoding="utf-8"))
    assert inventory["source_name"] == TEST_SOURCE
    assert inventory["summary"]["n_patches"] == data.summaries["unet"]["n_patches"]

    mtime = result.inventory_path.stat().st_mtime_ns
    reused = monograph.run_monograph(paths, TEST_SOURCE)
    assert reused.skipped
    assert reused.inventory_path.stat().st_mtime_ns == mtime
    assert reused.tables_data["dataset"].equals(result.tables_data["dataset"])


def test_run_monograph_regenerates_when_fingerprint_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mudança de fingerprint deve forçar a regeneração das figuras e tabelas."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    paths = _storage_paths(tmp_path)
    data = _data()
    summaries = data.summaries
    comparison = data.comparison
    monkeypatch.setattr(
        "src.synthesis.validate_inputs",
        lambda storage, source: (TEST_SOURCE, summaries, comparison, ["unet", "segformer"]),
    )
    monkeypatch.setattr("src.monograph.build_monograph_data", lambda *args, **kwargs: data)
    fingerprints: list[dict[str, Any]] = [{"hash": "first"}]
    monkeypatch.setattr(
        "src.monograph.monograph_fingerprint", lambda *args, **kwargs: fingerprints[0]
    )

    monograph.run_monograph(paths, TEST_SOURCE)
    fingerprints[0] = {"hash": "second"}
    regenerated = monograph.run_monograph(paths, TEST_SOURCE)

    assert not regenerated.skipped
