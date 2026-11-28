"""Elementos visuais didáticos para a monografia (estágio 17).

Traduz os artefatos dos estágios 03–15 em figuras e tabelas de leitura imediata
para os capítulos de metodologia e de resultados. Cada figura comunica uma única
mensagem, em português, com valores anotados e unidades familiares ao leitor não
técnico (hectares, porcentagem, contagem de dobras); as tabelas reescrevem os
números da síntese técnica com cabeçalhos em português e leitura em linguagem
comum. As figuras são persistidas em PDF vetorial (para o LaTeX) e PNG de 300 dpi
(para editores de texto), junto com legendas prontas (`captions.tex`/`.md`).

O estágio não retreina, não avalia e não recalcula métricas: lê os artefatos
vigentes, reutiliza as seleções determinísticas dos estágios 12/13/14 (o mesmo
talhão aparece nas figuras qualitativas, nos mapas de erro e no XAI) e é
regenerado quando qualquer fingerprint de entrada muda.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import comparative, evaluation, figstyle, spatial_inference, synthesis, trainer
from src.config import get_config
from src.data.mask_utils import reference_year, source_mask_path
from src.data.patch_generation import load_manifest, patch_fingerprint_hash
from src.data.raster_utils import render_figure_formats, rgb_composite
from src.data.source_decision import chosen_source
from src.data.spatial_split import fold_count, split_fingerprint_hash
from src.xai import common

MONOGRAPH_SCHEMA_VERSION = 3
MODEL_ORDER = synthesis.MODEL_ORDER
MODEL_LABELS = synthesis.MODEL_LABELS
MODEL_COLORS = synthesis.MODEL_COLORS
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paleta segura para daltonismo (Okabe–Ito) usada nos mapas de acerto e erro.
COLOR_TRUE_POSITIVE = "#009e73"
COLOR_FALSE_POSITIVE = "#e69f00"
COLOR_FALSE_NEGATIVE = "#cc79a7"
COLOR_TRUE_NEGATIVE = "#d9d9d9"
COLOR_IGNORED = "#404040"
COLOR_REFERENCE = figstyle.COLOR_REFERENCE
COLOR_NEUTRAL = figstyle.COLOR_NEUTRAL
FOLD_COLORS = figstyle.FOLD_COLORS
METRIC_LABELS = figstyle.METRIC_LABELS
SOURCE_LABELS = {"mapbiomas": "MapBiomas", "alphaearth": "AlphaEarth", "emater": "Emater"}

# Parâmetros editoriais das figuras: compartilhados com os demais estágios em
# `src.figstyle` (ficam fora de config.yaml para não invalidar os estágios 09–15).
FIGURE_WIDTH_CM = figstyle.FIGURE_WIDTH_CM
FIGURE_DPI = figstyle.FIGURE_DPI
FIGURE_FORMATS = figstyle.FIGURE_FORMATS


# ---------------------------------------------------------------------------
# Catálogo de figuras e tabelas (a ordem define a numeração das legendas)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureSpec:
    """Identidade editorial de uma figura: capítulo e legenda pronta."""

    section: str
    caption: str


@dataclass(frozen=True)
class TableSpec:
    """Identidade editorial de uma tabela: capítulo e legenda pronta."""

    section: str
    caption: str


FIGURE_CATALOG: dict[str, FigureSpec] = {
    "study_area": FigureSpec(
        "Metodologia",
        "Localização da área de estudo: (a) Minas Gerais com as regiões geográficas "
        "imediatas e a Região de Guaxupé destacada; (b) a Região de Guaxupé com as glebas "
        "de café do mapeamento da Emater (2018).",
    ),
    "pipeline": FigureSpec(
        "Metodologia",
        "Fluxo metodológico do estudo. As mesmas imagens, máscaras, patches e dobras "
        "alimentam os dois modelos, que compartilham perda, otimizador, sementes e "
        "métricas — apenas a arquitetura difere.",
    ),
    "bands": FigureSpec(
        "Metodologia",
        "O mesmo talhão em composição natural (RGB) e em falsa-cor (infravermelho "
        "próximo): a vegetação sadia se destaca na falsa-cor, o que motiva o uso das "
        "quatro bandas do Sentinel-2.",
    ),
    "label_sources": FigureSpec(
        "Metodologia",
        "Máscaras de café das três fontes de referência sobre a mesma área, com a "
        "concordância (kappa) em relação à Emater, fonte escolhida para o treino.",
    ),
    "patch_anatomy": FigureSpec(
        "Metodologia",
        "Anatomia de um patch de treino: (a) imagem Sentinel-2; (b) o que o modelo deve "
        "aprender (café em verde); (c) pixels fora da área de estudo, ignorados na perda "
        "e nas métricas.",
    ),
    "spatial_folds": FigureSpec(
        "Metodologia",
        "Divisão espacial em cinco dobras: cada dobra de validação é uma região "
        "contígua, de modo que o modelo é sempre testado em áreas que não viu no treino.",
    ),
    "architectures": FigureSpec(
        "Metodologia",
        "Comparação conceitual das arquiteturas: a U-Net combina vizinhanças locais por "
        "convoluções; o SegFormer relaciona a cena inteira por atenção.",
    ),
    "loss_terms": FigureSpec(
        "Metodologia",
        "Os três termos da função de perda: Dice (sobreposição), Focal (pixels difíceis) "
        "e Boundary (bordas).",
    ),
    "qualitative_predictions": FigureSpec(
        "Resultados",
        "Exemplos qualitativos: imagem, referência e predições dos dois modelos, com o "
        "mapa de acertos e erros (verde: acerto; laranja: café inventado; roxo: café não "
        "detectado).",
    ),
    "metrics_overview": FigureSpec(
        "Resultados",
        "Desempenho agregado por métrica e diferença pareada entre os modelos "
        "(SegFormer − U-Net), com intervalo de confiança de 95%.",
    ),
    "inference_levels": FigureSpec(
        "Resultados",
        "Diferença de cada métrica (SegFormer − U-Net) com intervalo de confiança de 95% "
        "segundo a unidade de análise: recortes (descritivo, desconsidera a dependência "
        "espacial), blocos espaciais de recortes vizinhos e dobras de validação.",
    ),
    "threshold_sensitivity": FigureSpec(
        "Resultados",
        "Sensibilidade ao limiar de decisão: (a) curvas precisão × revocação, com a precisão "
        "média (AP) de cada modelo; (b) IoU agregado em função do limiar. O círculo marca o "
        "limiar de 0,5 usado nas métricas principais e a estrela, o limiar de maior IoU.",
    ),
    "patch_iou_distribution": FigureSpec(
        "Resultados",
        "Distribuição do IoU por patch de validação: cada ponto é um talhão e a linha "
        "horizontal marca a mediana.",
    ),
    "fold_consistency": FigureSpec(
        "Resultados",
        "IoU de cada modelo nas cinco dobras: a posição relativa dos modelos se mantém "
        "estável entre as dobras.",
    ),
    "area_hectares": FigureSpec(
        "Resultados",
        "Área de café na amostra de validação: referência, área mapeada por cada modelo "
        "e a decomposição do erro em área não detectada e área detectada a mais.",
    ),
    "stratum_performance": FigureSpec(
        "Resultados",
        "IoU médio por densidade de café no patch (barras de erro: desvio padrão): "
        "talhões com pouca café tendem a ser mais difíceis para os dois modelos.",
    ),
    "xai_panel": FigureSpec(
        "Resultados",
        "Mapas de explicabilidade do mesmo talhão (Grad-CAM da U-Net e Attention Rollout "
        "do SegFormer, ponderado pela probabilidade predita de café) e a relevância média "
        "dentro e fora da região de café, calculada apenas sobre pixels válidos.",
    ),
    "summary_card": FigureSpec(
        "Resultados",
        "Síntese dos principais resultados do estudo: IoU e F1 de cada modelo, diferença "
        "em pontos percentuais e número de dobras em que o SegFormer iguala ou supera a "
        "U-Net.",
    ),
}

TABLE_CATALOG: dict[str, TableSpec] = {
    "dataset": TableSpec(
        "Metodologia",
        "Composição do conjunto de dados por fonte de referência.",
    ),
    "protocol": TableSpec(
        "Metodologia",
        "Protocolo de treinamento compartilhado pelos dois modelos.",
    ),
    "metric_glossary": TableSpec(
        "Metodologia",
        "Glossário das métricas em linguagem comum.",
    ),
    "model_results": TableSpec(
        "Resultados",
        "Desempenho dos modelos na validação espacial (média e desvio entre dobras e agregado).",
    ),
    "paired_comparison": TableSpec(
        "Resultados",
        "Comparação pareada por patch entre SegFormer e U-Net, com interpretação do "
        "tamanho da diferença.",
    ),
    "threshold_sensitivity": TableSpec(
        "Resultados",
        "Precisão média e desempenho com limiar fixo (0,5), calibrado em outras dobras "
        "(leave-one-fold-out) e ótimo (limite superior otimista).",
    ),
    "area_confusion": TableSpec(
        "Resultados",
        "Área de café na amostra de validação em hectares e decomposição do erro.",
    ),
    "xai_relevance": TableSpec(
        "Resultados",
        "Relevância dos mapas de explicabilidade dentro e fora da região de café.",
    ),
}


# ---------------------------------------------------------------------------
# Estruturas de dados do estágio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyAreaData:
    """Geometrias e camada rasterizada de café para o mapa de localização."""

    mg_polygons: list[np.ndarray]
    aoi_polygons: list[np.ndarray]
    coffee_mask: np.ndarray
    coffee_extent: tuple[float, float, float, float]
    region_name: str
    region_area_km2: float
    coffee_area_km2: float
    n_regions: int


@dataclass(frozen=True)
class LabelSourcesData:
    """Máscaras das fontes (amostradas) e concordância com a fonte escolhida."""

    masks: dict[str, np.ndarray]
    years: dict[str, int]
    coffee_share: dict[str, float]
    kappa_vs_chosen: dict[str, float]
    chosen: str


@dataclass(frozen=True)
class XaiPanel:
    """Talhão explicado pelos dois modelos, com o mapa de cada um."""

    patch_id: str
    fold: int
    image_rgb: np.ndarray
    reference: np.ndarray
    heatmaps: dict[str, np.ndarray]
    iou: dict[str, float]


@dataclass
class MonographData:
    """Artefatos já carregados que alimentam todas as figuras e tabelas."""

    source_name: str
    config: dict[str, Any]
    manifest: pd.DataFrame
    summaries: dict[str, dict[str, Any]]
    comparison: dict[str, Any]
    patch_metrics: pd.DataFrame
    dataset_table: pd.DataFrame
    xai_table: pd.DataFrame
    xai_models: list[str]
    study_area: StudyAreaData | None = None
    label_sources: LabelSourcesData | None = None
    showcase: common.PatchSample | None = None
    error_samples: list[comparative.ErrorMapSample] = field(default_factory=list)
    xai_panel: XaiPanel | None = None


@dataclass(frozen=True)
class RenderContext:
    """Parâmetros de saída das figuras (largura, resolução e formatos)."""

    width_cm: float
    dpi: int
    formats: tuple[str, ...]


@dataclass(frozen=True)
class MonographResult:
    """Figuras, tabelas e legendas geradas (ou reutilizadas) para a fonte."""

    source_name: str
    figures: dict[str, dict[str, Path]]
    tables: dict[str, Path]
    tables_data: dict[str, pd.DataFrame]
    captions_tex_path: Path
    captions_md_path: Path
    inventory_path: Path
    summary: dict[str, Any]
    skipped: bool


# ---------------------------------------------------------------------------
# Caminhos canônicos
# ---------------------------------------------------------------------------


def monograph_dir(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Diretório canônico da monografia (chave dedicada ou derivada de artifacts/)."""
    root = storage_paths.get("artifacts_monograph")
    if root is None:
        root = storage_paths["artifacts_metrics"].parent / "monograph"
    return root / source_name


def figure_path(storage_paths: dict[str, Path], source_name: str, key: str, suffix: str) -> Path:
    """Caminho de uma figura da monografia no formato indicado."""
    return monograph_dir(storage_paths, source_name) / "figures" / f"{key}.{suffix}"


def table_path(storage_paths: dict[str, Path], source_name: str, key: str, suffix: str) -> Path:
    """Caminho de uma tabela da monografia no formato indicado."""
    return monograph_dir(storage_paths, source_name) / "tables" / f"{key}.{suffix}"


def captions_path(storage_paths: dict[str, Path], source_name: str, suffix: str) -> Path:
    """Caminho do arquivo de legendas (LaTeX ou Markdown)."""
    return monograph_dir(storage_paths, source_name) / f"captions.{suffix}"


def inventory_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho do inventário com fingerprint e metadados das figuras/tabelas."""
    return monograph_dir(storage_paths, source_name) / "inventory.json"


# ---------------------------------------------------------------------------
# Formatação em linguagem comum (pt-BR)
# ---------------------------------------------------------------------------


def _number(value: float | None, digits: int = 3) -> str:
    """Formata um número com vírgula decimal (ou traço quando ausente)."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.{digits}f}".replace(".", ",")


def _percent(value: float | None, digits: int = 1) -> str:
    """Formata uma proporção [0, 1] como porcentagem com vírgula decimal."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100.0 * float(value):.{digits}f}%".replace(".", ",")


def _signed_points(difference: float) -> str:
    """Formata uma diferença em pontos percentuais com sinal e vírgula decimal."""
    return f"{100.0 * float(difference):+.1f}".replace(".", ",") + " p.p."


def _pvalue(value: float | None) -> str:
    """Formata um p-valor (notação '< 0,001' quando muito pequeno)."""
    if value is None or not np.isfinite(value):
        return "—"
    if value < 0.001:
        return "< 0,001"
    return f"{float(value):.3f}".replace(".", ",")


def _interval(low: float | None, high: float | None) -> str:
    """Formata um intervalo de confiança como '[inferior; superior]'."""
    if low is None or high is None:
        return "—"
    return f"[{_number(low)}; {_number(high)}]"


def _interpret_difference(difference: float | None, pvalue: float | None) -> str:
    """Traduz magnitude e significância de uma diferença em linguagem comum.

    A magnitude usa pontos de métrica: abaixo de 0,01 é desprezível, até 0,03 é
    pequena, até 0,05 é moderada e acima disso é relevante. A significância usa o
    p-valor corrigido por Holm (0,05).
    """
    if difference is None or not np.isfinite(difference):
        return "—"
    magnitude = abs(float(difference))
    if magnitude < 0.01:
        label = "desprezível"
    elif magnitude < 0.03:
        label = "pequena"
    elif magnitude < 0.05:
        label = "moderada"
    else:
        label = "relevante"
    if pvalue is None or not np.isfinite(pvalue):
        return label
    return f"{label} ({'significativa' if pvalue < 0.05 else 'não significativa'})"


def pixel_area_ha(config: dict[str, Any]) -> float:
    """Área de um pixel em hectares, derivada da resolução de exportação."""
    scale = float(config["data"]["export"]["scale"])
    return scale**2 / 10_000.0


def _mean_std(values: list[float]) -> str:
    """Formata média ± desvio entre dobras com vírgula decimal."""
    array = np.asarray(values, dtype=float)
    return f"{_number(float(array.mean()))} ± {_number(float(array.std()))}"


# ---------------------------------------------------------------------------
# Carregamento de dados (I/O concentrado; as renderizações são puras)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """Lê um JSON do Drive canônico por meio do helper de I/O do projeto."""
    from src import io

    return json.loads(io.ensure_local_copy(path).read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    """Hash SHA-256 curto de um arquivo (detecta mudanças de entrada)."""
    from src import io

    return hashlib.sha256(io.ensure_local_copy(path).read_bytes()).hexdigest()[:12]


def _exterior_rings(geometries: Any, tolerance: float = 0.0) -> list[np.ndarray]:
    """Extrai os anéis externos das geometrias (simplificados para plotagem)."""
    rings: list[np.ndarray] = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        parts = list(geometry.geoms) if geometry.geom_type.startswith("Multi") else [geometry]
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            simplified = part.simplify(tolerance, preserve_topology=True) if tolerance else part
            rings.append(np.asarray(simplified.exterior.coords, dtype=float))
    return rings


def _load_coffee_layer(
    config: dict[str, Any], crs: Any, aoi_geometry: Any, max_dim: int = 900
) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
    """Rasteriza as glebas de café da Emater (milhares de polígonos) para o mapa.

    A camada vetorial tem dezenas de milhares de polígonos: rasterizá-la mantém a
    figura leve e o PDF pequeno, sem alterar a leitura visual das áreas de café.
    A máscara é recortada pela área de estudo, de modo que a área anotada na
    figura corresponde exatamente ao que é desenhado.
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    source_cfg = config["ground_truth"]["sources"].get("emater", {})
    data_dir = PROJECT_ROOT / str(source_cfg.get("data_dir", "data/external/emater"))
    empty = (np.zeros((1, 1), dtype=np.uint8), (0.0, 1.0, 0.0, 1.0), 0.0)
    if not data_dir.is_dir():
        return empty
    frames = [
        gpd.read_file(path)
        for path in sorted(data_dir.glob("*.json"))
        if path.name.startswith("GetGlebas")
    ]
    if not frames:
        return empty
    glebas = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    if glebas.crs is None:
        return empty
    glebas = glebas.to_crs(crs)
    geometries = [geometry for geometry in glebas.geometry if geometry is not None]
    if not geometries:
        return empty
    min_x, min_y, max_x, max_y = glebas.total_bounds
    width_span = max(max_x - min_x, 1e-6)
    height_span = max(max_y - min_y, 1e-6)
    width = max_dim
    height = max(1, int(round(max_dim * height_span / width_span)))
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    mask = rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    aoi_mask = rasterize(
        [(aoi_geometry, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    mask = np.where(aoi_mask > 0, mask, 0).astype(np.uint8)
    pixel_area_km2 = (width_span / width) * (height_span / height) / 1e6
    return mask, (min_x, max_x, min_y, max_y), float(mask.sum()) * pixel_area_km2


def load_study_area(config: dict[str, Any]) -> StudyAreaData | None:
    """Carrega a malha do IBGE e as glebas da Emater para o mapa de localização."""
    import geopandas as gpd

    mesh_path = PROJECT_ROOT / str(config["aoi"]["mesh_path"])
    if not mesh_path.is_file():
        return None
    mesh = gpd.read_file(mesh_path)
    if mesh.crs is None or "CD_RGI" not in mesh.columns:
        return None
    crs = config["data"]["crs"]
    mesh = mesh.to_crs(crs)
    region_code = str(config["aoi"]["region_code"])
    aoi = mesh[mesh["CD_RGI"].astype(str) == region_code]
    if aoi.empty:
        return None
    coffee_mask, coffee_extent, coffee_area_km2 = _load_coffee_layer(
        config, mesh.crs, aoi.geometry.iloc[0]
    )
    return StudyAreaData(
        mg_polygons=_exterior_rings(mesh.geometry, 200.0),
        aoi_polygons=_exterior_rings(aoi.geometry, 200.0),
        coffee_mask=coffee_mask,
        coffee_extent=coffee_extent,
        region_name=str(aoi.iloc[0]["NM_RGI"]),
        region_area_km2=float(aoi.iloc[0]["AREA_KM2"]),
        coffee_area_km2=coffee_area_km2,
        n_regions=int(len(mesh)),
    )


def load_label_sources(
    storage_paths: dict[str, Path], config: dict[str, Any]
) -> LabelSourcesData | None:
    """Carrega as máscaras das fontes habilitadas e a concordância entre elas."""
    from src import io
    from src.data import mask_comparison

    names = [
        name
        for name, source_cfg in config["ground_truth"]["sources"].items()
        if source_cfg.get("enabled")
    ]
    paths: dict[str, Path] = {}
    for name in names:
        path = source_mask_path(name, storage_paths)
        if io.path_exists(path):
            paths[name] = path
    if len(paths) < 2:
        return None
    mask_set = mask_comparison.load_masks(
        {name: io.ensure_local_copy(path) for name, path in paths.items()}
    )
    comparison = mask_comparison.compute_comparison(mask_set)
    chosen = chosen_source()
    kappa: dict[str, float] = {}
    for name in mask_set.masks:
        if name == chosen:
            continue
        key = f"{name}_vs_{chosen}"
        if key not in comparison["pairs"]:
            key = f"{chosen}_vs_{name}"
        value = comparison["pairs"][key]["metrics"]["kappa"]
        if value is not None:
            kappa[name] = float(value)
    stride = max(1, mask_set.shape[0] // 700)
    return LabelSourcesData(
        masks={name: mask[::stride, ::stride] for name, mask in mask_set.masks.items()},
        years={name: reference_year(name) for name in mask_set.masks},
        coffee_share={name: float(value) for name, value in comparison["coffee_share"].items()},
        kappa_vs_chosen=kappa,
        chosen=chosen,
    )


def _showcase_patch(
    storage_paths: dict[str, Path], manifest: pd.DataFrame, stats: dict[str, Any]
) -> common.PatchSample | None:
    """Seleciona deterministicamente o patch com maior proporção de café."""
    pool = manifest[manifest["coffee_ratio"] > 0]
    if pool.empty:
        pool = manifest
    ordered = pool.sort_values(["coffee_ratio", "patch_id"], ascending=[False, True])
    record = ordered.iloc[0]
    return common.load_patch_sample(storage_paths, manifest, str(record["patch_id"]), stats)


def load_xai_panel(
    storage_paths: dict[str, Path],
    source_name: str,
    xai_models: list[str],
    manifest: pd.DataFrame,
    stats: dict[str, Any],
) -> XaiPanel | None:
    """Carrega o primeiro talhão explicado e o mapa de cada modelo disponível."""
    from src import io

    if not xai_models:
        return None
    heatmaps: dict[str, np.ndarray] = {}
    iou: dict[str, float] = {}
    patch: dict[str, Any] | None = None
    for model_name in xai_models:
        metadata = _read_json(synthesis.xai_metadata_path(storage_paths, source_name, model_name))
        if not metadata.get("patches"):
            continue
        first = metadata["patches"][0]
        if patch is None:
            patch = first
        if str(first["patch_id"]) != str(patch["patch_id"]):
            continue
        iou[model_name] = float(first["iou"])
        maps_path = synthesis.xai_maps_path(storage_paths, source_name, model_name)
        key = "cams" if model_name == "unet" else "rollouts"
        with np.load(io.ensure_local_copy(maps_path)) as data:
            patch_ids = [str(item) for item in data["patch_ids"]]
            if str(patch["patch_id"]) not in patch_ids:
                continue
            heatmaps[model_name] = np.asarray(data[key][patch_ids.index(str(patch["patch_id"]))])
    if patch is None or not heatmaps:
        return None
    sample = common.load_patch_sample(storage_paths, manifest, str(patch["patch_id"]), stats)
    return XaiPanel(
        patch_id=str(patch["patch_id"]),
        fold=int(patch["fold"]),
        image_rgb=common.patch_rgb(sample),
        reference=sample.reference,
        heatmaps=heatmaps,
        iou=iou,
    )


def build_monograph_data(
    storage_paths: dict[str, Path],
    source_name: str | None = None,
    *,
    summaries: dict[str, dict[str, Any]] | None = None,
    comparison: dict[str, Any] | None = None,
    xai_models: list[str] | None = None,
) -> MonographData:
    """Valida as entradas e carrega tudo o que as figuras e tabelas consomem."""
    source = source_name or chosen_source()
    if summaries is None or comparison is None or xai_models is None:
        source, validated_summaries, validated_comparison, validated_xai = (
            synthesis.validate_inputs(storage_paths, source)
        )
        summaries = summaries or validated_summaries
        comparison = comparison or validated_comparison
        xai_models = xai_models if xai_models is not None else validated_xai
    config = get_config()
    manifest = load_manifest(storage_paths, source)
    stats = common.load_stats_and_manifest(storage_paths, source)[0]
    patch_metrics = comparative.load_aligned_patch_metrics(storage_paths, source, summaries)
    selected = comparative.select_error_map_patches(patch_metrics, max_patches=3)
    error_samples = (
        comparative.build_error_map_samples(storage_paths, source, selected) if selected else []
    )
    return MonographData(
        source_name=source,
        config=config,
        manifest=manifest,
        summaries=summaries,
        comparison=comparison,
        patch_metrics=patch_metrics,
        dataset_table=synthesis.build_dataset_table(storage_paths),
        xai_table=synthesis.build_xai_table(storage_paths, source, xai_models),
        xai_models=xai_models,
        study_area=load_study_area(config),
        label_sources=load_label_sources(storage_paths, config),
        showcase=_showcase_patch(storage_paths, manifest, stats),
        error_samples=error_samples,
        xai_panel=load_xai_panel(storage_paths, source, xai_models, manifest, stats),
    )


# ---------------------------------------------------------------------------
# Disponibilidade, fingerprint e reutilização
# ---------------------------------------------------------------------------


def available_figures(data: MonographData) -> list[str]:
    """Figuras que podem ser geradas com os artefatos carregados (ordem canônica)."""
    availability = {
        "threshold_sensitivity": data.comparison.get("threshold_sensitivity") is not None,
        "study_area": data.study_area is not None,
        "label_sources": data.label_sources is not None,
        "bands": data.showcase is not None,
        "patch_anatomy": data.showcase is not None,
        "qualitative_predictions": bool(data.error_samples),
        "xai_panel": data.xai_panel is not None,
    }
    return [key for key in FIGURE_CATALOG if availability.get(key, True)]


def available_tables(data: MonographData) -> list[str]:
    """Tabelas que podem ser geradas com os artefatos carregados (ordem canônica)."""
    availability = {
        "xai_relevance": not data.xai_table.empty,
        "threshold_sensitivity": data.comparison.get("threshold_sensitivity") is not None,
    }
    return [key for key in TABLE_CATALOG if availability.get(key, True)]


def monograph_fingerprint(
    storage_paths: dict[str, Path], data: MonographData, figures: list[str], tables: list[str]
) -> dict[str, Any]:
    """Fingerprint que valida a reutilização das figuras, tabelas e legendas."""
    xai_fingerprint: dict[str, dict[str, Any]] = {}
    for model_name in data.xai_models:
        metadata = _read_json(
            synthesis.xai_metadata_path(storage_paths, data.source_name, model_name)
        )
        xai_fingerprint[model_name] = {
            "fingerprint": metadata.get("fingerprint"),
            "maps_sha256": _digest(
                synthesis.xai_maps_path(storage_paths, data.source_name, model_name)
            ),
        }
    return {
        "schema_version": MONOGRAPH_SCHEMA_VERSION,
        "source_name": data.source_name,
        "config_hash": trainer.config_hash(),
        "manifest_fingerprint_hash": patch_fingerprint_hash(storage_paths, data.source_name),
        "split_fingerprint_hash": split_fingerprint_hash(storage_paths, data.source_name),
        "evaluation": {model: data.summaries[model]["fingerprint"] for model in MODEL_ORDER},
        "comparison": data.comparison["fingerprint"],
        "xai": xai_fingerprint,
        "figures": figures,
        "tables": tables,
    }


def _current_artifacts(
    storage_paths: dict[str, Path],
    source_name: str,
    fingerprint: dict[str, Any],
    figures: list[str],
    tables: list[str],
    formats: tuple[str, ...],
) -> bool:
    """Indica se inventário, legendas, figuras e tabelas existem e estão vigentes."""
    from src import io

    paths = [inventory_path(storage_paths, source_name)]
    paths += [captions_path(storage_paths, source_name, suffix) for suffix in ("tex", "md")]
    for key in figures:
        paths += [figure_path(storage_paths, source_name, key, fmt) for fmt in formats]
    for key in tables:
        paths += [table_path(storage_paths, source_name, key, suffix) for suffix in ("csv", "tex")]
    if not all(io.path_exists(path) for path in paths):
        return False
    try:
        stored = _read_json(inventory_path(storage_paths, source_name))
    except (json.JSONDecodeError, OSError):
        return False
    return stored.get("fingerprint") == fingerprint


# ---------------------------------------------------------------------------
# Tabelas didáticas
# ---------------------------------------------------------------------------


def build_dataset_table(data: MonographData) -> pd.DataFrame:
    """Reescreve o resumo do dataset com cabeçalhos e unidades em pt-BR."""
    table = data.dataset_table
    return pd.DataFrame(
        {
            "Fonte": [SOURCE_LABELS.get(str(name), str(name)) for name in table["source"]],
            "Ano": table["year"].astype(int),
            "Fonte de treino": ["sim" if bool(flag) else "não" for flag in table["chosen"]],
            "Patches": table["n_patches"].astype(int),
            "Dobras": table["n_folds"].astype(int),
            "Café médio": [_percent(value) for value in table["coffee_ratio_mean"]],
            "Pixels válidos (média)": [_percent(value) for value in table["valid_fraction_mean"]],
        }
    )


def build_protocol_table(config: dict[str, Any]) -> pd.DataFrame:
    """Resume o protocolo único de treino em uma tabela de leitura rápida."""
    training = config["training"]
    loss = config["loss"]
    model = config["model"]
    rows = [
        ("Épocas por dobra", str(int(training["epochs"]))),
        ("Tamanho do lote", str(int(training["batch_size"]))),
        ("Taxa de aprendizado", f"{float(training['lr']):.0e}".replace("e-0", "e-")),
        ("Weight decay", f"{float(training['weight_decay']):.0e}".replace("e-0", "e-")),
        ("Semente", str(int(config["reproducibility"]["seed"]))),
        ("Dobras espaciais", str(fold_count())),
        ("Patch", f"{int(config['data']['patch_size'])} × {int(config['data']['patch_size'])} px"),
        ("Resolução", f"{int(config['data']['export']['scale'])} m"),
        ("Bandas", ", ".join(config["data"]["bands"])),
        (
            "Função de perda",
            f"Dice {float(loss['dice_weight']):g} + Focal {float(loss['focal_weight']):g} "
            f"+ Boundary {float(loss['boundary_weight']):g}",
        ),
        ("Limiar de decisão", _number(evaluation.DEFAULT_THRESHOLD)),
        ("U-Net (canais)", ", ".join(str(int(c)) for c in model["unet_channels"])),
        ("Inicialização da U-Net", "pesos aleatórios (treino do zero)"),
        ("SegFormer", str(model["segformer_variant"])),
        ("Inicialização do SegFormer", "codificador pré-treinado; cabeça e entrada adaptadas"),
        ("Seleção da época", "maior IoU na própria dobra de validação (viés otimista)"),
        (
            "Unidade da inferência estatística",
            f"bloco espacial de {spatial_inference.SPATIAL_BLOCK_SIZE} × "
            f"{spatial_inference.SPATIAL_BLOCK_SIZE} recortes e dobra",
        ),
    ]
    return pd.DataFrame(rows, columns=["Item", "Valor"])


def build_metric_glossary_table() -> pd.DataFrame:
    """Explica as métricas em linguagem comum, para leitores não técnicos."""
    rows = [
        (
            "IoU",
            "Quanto da área de café foi encontrada, descontando o que foi marcado sem ser café",
            "0 a 1; quanto maior, melhor",
        ),
        (
            "F1",
            "Equilíbrio entre encontrar o café e não marcar café onde não há",
            "0 a 1; quanto maior, melhor",
        ),
        (
            "Precisão",
            "Do café apontado pelo modelo, quanto era café de verdade",
            "0 a 1; quanto maior, menos alarme falso",
        ),
        (
            "Recall",
            "Do café existente, quanto o modelo conseguiu encontrar",
            "0 a 1; quanto maior, menos café esquecido",
        ),
        (
            "Kappa (κ)",
            "Concordância entre dois mapas além do que se esperaria por acaso",
            "0 a 1; acima de 0,4 já é concordância moderada",
        ),
    ]
    return pd.DataFrame(rows, columns=["Métrica", "O que mede", "Como ler"])


def build_model_results_table(summaries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Resume média ± desvio entre dobras e o agregado de cada modelo."""
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        summary = summaries[model_name]
        row: dict[str, Any] = {"Modelo": MODEL_LABELS[model_name]}
        for metric in evaluation.METRIC_KEYS:
            values = [float(item["metrics"][metric]) for item in summary["per_fold"]]
            row[METRIC_LABELS[metric]] = _mean_std(values)
        row["IoU agregado"] = _number(summary["aggregate"]["metrics"]["iou"])
        row["Patches"] = int(summary["n_patches"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_comparison_table(comparison: dict[str, Any]) -> pd.DataFrame:
    """Traduz a comparação em diferença, IC por blocos e dobras, p corrigido e efeito.

    A inferência principal usa o bloco espacial como unidade (patches vizinhos não
    são independentes); o IC e o p por recorte permanecem como referência descritiva.
    """
    spatial = comparison["spatial_inference"]
    rows: list[dict[str, Any]] = []
    for metric in evaluation.METRIC_KEYS:
        patch = comparison["paired"][metric]
        block = spatial["block_level"][metric]
        fold = spatial["fold_level"][metric]
        difference = block.get("mean_difference")
        pvalue = block.get("p_value_holm")
        rows.append(
            {
                "Métrica": METRIC_LABELS[metric],
                "Diferença média (SegFormer − U-Net)": _number(difference),
                "IC 95% (blocos espaciais)": _interval(
                    block.get("ci95_low"), block.get("ci95_high")
                ),
                "p (blocos, Holm)": _pvalue(pvalue),
                "IC 95% (dobras, n = 5)": _interval(fold.get("ci95_low"), fold.get("ci95_high")),
                "IC 95% (recortes, descritivo)": _interval(
                    patch.get("ci95_low"), patch.get("ci95_high")
                ),
                "Tamanho de efeito (d_z)": _number(patch.get("cohens_dz"), 2),
                "Interpretação": _interpret_difference(difference, pvalue),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_table(comparison: dict[str, Any]) -> pd.DataFrame:
    """Compara os modelos sem depender do limiar fixo: AP e limiares fixo, calibrado e ótimo."""
    sensitivity = comparison["threshold_sensitivity"]
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        analysis = sensitivity["models"][model_name]
        rows.append(
            {
                "Modelo": MODEL_LABELS[model_name],
                "Precisão média (AP)": _number(analysis["average_precision"]),
                "IoU (limiar 0,5)": _number(analysis["fixed"]["metrics"]["iou"]),
                "IoU (limiar calibrado)": _number(analysis["calibrated"]["aggregate"]["iou"]),
                "IoU (limiar ótimo)": _number(analysis["oracle"]["metrics"]["iou"]),
                "Limiar ótimo": _number(analysis["oracle"]["threshold"], 2),
            }
        )
    return pd.DataFrame(rows)


def build_area_table(summaries: dict[str, dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    """Converte as contagens de pixel da validação em hectares por modelo."""
    area = pixel_area_ha(config)
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        counts = summaries[model_name]["aggregate"]["counts"]
        tp, fp, fn = int(counts["tp"]), int(counts["fp"]), int(counts["fn"])
        reference = tp + fn
        rows.append(
            {
                "Modelo": MODEL_LABELS[model_name],
                "Café de referência (ha)": _number(reference * area, 1),
                "Café mapeado (ha)": _number((tp + fp) * area, 1),
                "Não detectado (ha)": _number(fn * area, 1),
                "Detectado a mais (ha)": _number(fp * area, 1),
                "Erro relativo": _percent((fn + fp) / reference if reference else None),
            }
        )
    return pd.DataFrame(rows)


def build_xai_summary_table(xai_table: pd.DataFrame) -> pd.DataFrame:
    """Resume a relevância média dos mapas dentro e fora do café, por modelo."""
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        subset = xai_table[xai_table["model"] == model_name]
        if subset.empty:
            continue
        rows.append(
            {
                "Modelo": MODEL_LABELS[model_name],
                "Relevância dentro do café": _number(float(subset["relevance_inside"].mean())),
                "Relevância fora do café": _number(float(subset["relevance_outside"].mean())),
                "Diferença": _number(float(subset["relevance_difference"].mean())),
                "Talhões explicados": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def build_tables(data: MonographData) -> dict[str, pd.DataFrame]:
    """Constrói todas as tabelas disponíveis para a fonte carregada."""
    tables = {
        "dataset": build_dataset_table(data),
        "protocol": build_protocol_table(data.config),
        "metric_glossary": build_metric_glossary_table(),
        "model_results": build_model_results_table(data.summaries),
        "paired_comparison": build_paired_comparison_table(data.comparison),
        "area_confusion": build_area_table(data.summaries, data.config),
    }
    if data.comparison.get("threshold_sensitivity") is not None:
        tables["threshold_sensitivity"] = build_threshold_table(data.comparison)
    xai_summary = build_xai_summary_table(data.xai_table)
    if not xai_summary.empty:
        tables["xai_relevance"] = xai_summary
    return tables


# ---------------------------------------------------------------------------
# Estilo e renderização
# ---------------------------------------------------------------------------


publication_style = figstyle.publication_style


def _render(
    build_fn: Callable[[Any, Any], None], context: RenderContext, *, height_cm: float
) -> dict[str, bytes]:
    """Renderiza uma figura com a largura configurada e a altura informada."""

    def _build(plt: Any) -> Any:
        with publication_style():
            figure = plt.figure(
                figsize=(context.width_cm / 2.54, height_cm / 2.54), layout="constrained"
            )
            build_fn(plt, figure)
        return figure

    return render_figure_formats(_build, dpi=context.dpi, formats=context.formats)


def _row_titles(axes: Any, titles: list[str]) -> None:
    """Aplica títulos de coluna apenas na primeira linha de um painel de imagens."""
    for axis, title in zip(axes, titles, strict=True):
        axis.set_title(title, fontsize=9)


def _hide(axes: Any) -> None:
    """Remove eixos e marcas de um painel de imagem."""
    for axis in np.ravel(axes):
        axis.set_axis_off()


def render_study_area(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Mapa de localização em dois níveis (Minas Gerais e a Região de Guaxupé)."""
    area = data.study_area
    if area is None:
        raise ValueError("Geometrias da área de estudo indisponíveis.")
    from matplotlib.collections import PolyCollection

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(1, 2)
        axes[0].add_collection(
            PolyCollection(area.mg_polygons, facecolors=COLOR_NEUTRAL, edgecolors="#9e9e9e")
        )
        axes[0].add_collection(
            PolyCollection(area.aoi_polygons, facecolors="#fdd835", edgecolors="#212121")
        )
        axes[0].set_title(f"Minas Gerais\n{area.n_regions} regiões geográficas imediatas")
        from matplotlib.colors import ListedColormap

        axes[1].add_collection(
            PolyCollection(area.aoi_polygons, facecolors="#f7f7f7", edgecolors="none", zorder=1)
        )
        axes[1].imshow(
            np.ma.masked_where(area.coffee_mask == 0, area.coffee_mask),
            extent=area.coffee_extent,
            origin="upper",
            cmap=ListedColormap([COLOR_REFERENCE]),
            vmin=1,
            vmax=1,
            interpolation="nearest",
            zorder=2,
        )
        axes[1].add_collection(
            PolyCollection(area.aoi_polygons, facecolors="none", edgecolors="#212121", zorder=3)
        )
        axis_extent = np.asarray(area.aoi_polygons[0], dtype=float)
        axes[1].set_xlim(axis_extent[:, 0].min(), axis_extent[:, 0].max())
        axes[1].set_ylim(axis_extent[:, 1].min(), axis_extent[:, 1].max())
        axes[1].set_title(
            f"{area.region_name} — {area.region_area_km2:,.0f} km²\n"
            f"{area.coffee_area_km2:,.0f} km² de café mapeado (Emater, 2018)"
        )
        for axis in axes:
            axis.set_aspect("equal")
            axis.set_axis_off()
        # Barra de escala de 20 km no painel da região (leitura imediata da escala).
        (x_min, x_max), (y_min, y_max) = axes[1].get_xlim(), axes[1].get_ylim()
        bar = (x_max - x_min) * 0.18
        axes[1].plot(
            [x_min + 0.05 * (x_max - x_min), x_min + 0.05 * (x_max - x_min) + bar],
            [y_min + 0.06 * (y_max - y_min)] * 2,
            color="#212121",
            linewidth=2.5,
            solid_capstyle="butt",
        )
        axes[1].text(
            x_min + 0.05 * (x_max - x_min) + bar / 2,
            y_min + 0.09 * (y_max - y_min),
            f"{bar / 1000:,.0f} km".replace(",", "."),
            ha="center",
            va="bottom",
        )

    return _render(_build, context, height_cm=8.0)


def render_pipeline(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Diagrama vertical do fluxo metodológico, com o protocolo único destacado."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def _box(
        axis: Any,
        text: str,
        center_x: float,
        center_y: float,
        width: float = 6.4,
        height: float = 0.78,
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (center_x - width / 2, center_y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.015,rounding_size=0.08",
                facecolor="#eceff1",
                edgecolor="#455a64",
                linewidth=0.9,
                zorder=2,
            )
        )
        axis.text(center_x, center_y, text, ha="center", va="center", fontsize=8.5, zorder=3)

    def _arrow(axis: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                color="#455a64",
                linewidth=1.1,
                shrinkA=0,
                shrinkB=0,
                zorder=4,
            )
        )

    def _build(plt: Any, figure: Any) -> None:
        axis = figure.add_subplot()
        axis.set_xlim(0, 10)
        axis.set_ylim(0, 7.3)
        axis.set_axis_off()

        # Faixa do protocolo compartilhado: preenchimento sem contorno evita
        # que as setas de entrada/saída cruzem traços do retângulo tracejado.
        axis.add_patch(
            FancyBboxPatch(
                (1.7, 2.12),
                6.6,
                1.5,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor="#e3f2fd",
                edgecolor="none",
                zorder=0,
            )
        )
        axis.text(
            5.0,
            3.55,
            "Mesmo protocolo",
            ha="center",
            va="center",
            fontsize=7.0,
            color="#1565c0",
            zorder=3,
        )
        axis.text(
            5.0,
            3.42,
            "perda, otimizador, sementes e dobras",
            ha="center",
            va="center",
            fontsize=6.0,
            color="#1565c0",
            zorder=3,
        )

        _box(axis, "Sentinel-2 (bandas B2, B3, B4 e B8, 10 m)", 5.0, 6.55)
        _box(axis, "Máscaras de referência\n(MapBiomas, AlphaEarth e Emater)", 5.0, 5.45)
        _box(axis, "Patches de 512 × 512 px\ne divisão espacial em 5 dobras", 5.0, 4.35)
        _arrow(axis, (5.0, 6.16), (5.0, 5.84))
        _arrow(axis, (5.0, 5.06), (5.0, 4.74))

        _box(axis, "Treino U-Net\n(CNN)", 3.35, 2.9, width=2.7)
        _box(axis, "Treino SegFormer\n(Transformer)", 6.65, 2.9, width=2.7)
        # Entradas verticais, fora do bloco de texto do protocolo compartilhado.
        _arrow(axis, (3.35, 3.96), (3.35, 3.29))
        _arrow(axis, (6.65, 3.96), (6.65, 3.29))

        # Barramento de saída dos dois treinos em direção à avaliação.
        axis.plot([3.35, 6.65], [2.25, 2.25], color="#455a64", linewidth=1.1, zorder=2)
        axis.plot([3.35, 3.35], [2.51, 2.25], color="#455a64", linewidth=1.1, zorder=2)
        axis.plot([6.65, 6.65], [2.51, 2.25], color="#455a64", linewidth=1.1, zorder=2)
        _arrow(axis, (5.0, 2.25), (5.0, 1.99))

        _box(axis, "Avaliação por pixel\n(IoU, F1, precisão e recall)", 5.0, 1.6)
        _arrow(axis, (5.0, 1.21), (5.0, 0.89))
        _box(axis, "Explicabilidade (Grad-CAM e Attention Rollout)", 5.0, 0.5)

    return _render(_build, context, height_cm=12.0)


def render_bands(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Compara o mesmo talhão em composição natural e em falsa-cor."""
    sample = data.showcase
    if sample is None:
        raise ValueError("Nenhum patch disponível para a figura de bandas.")
    bands = list(data.config["data"]["bands"])

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(1, 2)
        axes[0].imshow(rgb_composite(sample.image, sample.valid, bands))
        axes[0].set_title("Composição natural\n(B4, B3, B2)")
        axes[1].imshow(
            rgb_composite(sample.image, sample.valid, bands, rgb_bands=("B8", "B4", "B3"))
        )
        axes[1].set_title("Falsa-cor\n(B8, B4, B3)")
        _hide(axes)
        figure.text(
            0.5,
            0.02,
            "Na falsa-cor, a vegetação sadia aparece mais clara: a banda B8 "
            "(infravermelho) separa o café do solo exposto e da área construída.",
            ha="center",
            fontsize=7.5,
            wrap=True,
        )
        figure.get_layout_engine().set(rect=(0, 0.07, 1, 1))

    return _render(_build, context, height_cm=8.2)


def render_label_sources(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Mostra as máscaras das três fontes e a concordância com a fonte escolhida."""
    sources = data.label_sources
    if sources is None:
        raise ValueError("Máscaras das fontes indisponíveis.")
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#f2f2f2", COLOR_REFERENCE])

    def _build(plt: Any, figure: Any) -> None:
        names = list(sources.masks)
        axes = figure.subplots(1, len(names), squeeze=False)[0]
        for axis, name in zip(axes, names, strict=True):
            axis.imshow(sources.masks[name].astype(np.uint8), cmap=cmap, vmin=0, vmax=1)
            title = f"{SOURCE_LABELS.get(name, name)} ({sources.years[name]})"
            if name == sources.chosen:
                title += "\nfonte escolhida"
                axis.set_title(title, fontweight="bold", color="#1b5e20")
            elif name in sources.kappa_vs_chosen:
                title += f"\nκ = {_number(sources.kappa_vs_chosen[name], 2)} vs Emater"
                axis.set_title(title)
            else:
                axis.set_title(title)
            axis.text(
                0.02,
                0.02,
                f"café em {_percent(sources.coffee_share[name])} da área",
                transform=axis.transAxes,
                fontsize=7.5,
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            axis.set_axis_off()

    return _render(_build, context, height_cm=7.0)


def render_patch_anatomy(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Mostra imagem, rótulo binário e pixels ignorados de um patch de treino."""
    sample = data.showcase
    if sample is None:
        raise ValueError("Nenhum patch disponível para a anatomia do patch.")
    from matplotlib.colors import ListedColormap

    patch_size = int(data.config["data"]["patch_size"])
    side_km = patch_size * float(data.config["data"]["export"]["scale"]) / 1000.0

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(1, 3)
        bands = list(data.config["data"]["bands"])
        axes[0].imshow(rgb_composite(sample.image, sample.valid, bands))
        axes[0].set_title("(a) Imagem Sentinel-2")
        axes[1].imshow(
            sample.reference.astype(np.uint8),
            cmap=ListedColormap(["#f2f2f2", COLOR_REFERENCE]),
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("(b) Referência (café em verde)")
        ignored = np.where(sample.valid, 0, 1).astype(np.uint8)
        axes[2].imshow(ignored, cmap=ListedColormap(["#ffffff", COLOR_IGNORED]), vmin=0, vmax=1)
        axes[2].set_title("(c) Fora da área")
        _hide(axes)
        figure.text(
            0.5,
            0.02,
            f"Cada patch cobre {patch_size} × {patch_size} px ≈ {side_km:.1f} km de lado; "
            "os pixels escuros em (c) não entram na perda nem nas métricas.",
            ha="center",
            fontsize=7.5,
            wrap=True,
        )

    return _render(_build, context, height_cm=7.2)


def _patch_rectangles_km(
    manifest: pd.DataFrame, patch_size: int, pixel_m: float
) -> list[tuple[int, float, float, float, float]]:
    """Retângulos (dobra, x, y, largura, altura) dos patches, em km.

    Usa a bbox geográfica do manifesto (EPSG:31983), garantindo posições reais
    no território. Quando a bbox não está disponível, reconstrói o quadrado a
    partir dos índices `row`/`col` do grid de patches.
    """
    side_km = patch_size * pixel_m / 1000.0
    has_bbox = "bbox" in manifest.columns and bool(manifest["bbox"].notna().all())
    rectangles: list[tuple[int, float, float, float, float]] = []
    for record in manifest.itertuples(index=False):
        fold = int(record.fold)
        if has_bbox:
            min_x, min_y, max_x, max_y = (float(value) for value in str(record.bbox).split(","))
            rectangles.append(
                (
                    fold,
                    min_x / 1000.0,
                    min_y / 1000.0,
                    (max_x - min_x) / 1000.0,
                    (max_y - min_y) / 1000.0,
                )
            )
        else:
            rectangles.append(
                (
                    fold,
                    float(record.col) * side_km,
                    float(record.row) * side_km,
                    side_km,
                    side_km,
                )
            )
    return rectangles


def render_spatial_folds(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Desenha os patches no território, coloridos pela dobra de validação."""
    from matplotlib.patches import Patch, Rectangle

    manifest = data.manifest
    patch_size = int(data.config["data"]["patch_size"])
    pixel_m = float(data.config["data"]["export"]["scale"])
    folds = sorted(int(fold) for fold in manifest["fold"].unique())
    rectangles = _patch_rectangles_km(manifest, patch_size, pixel_m)
    if not rectangles:
        raise ValueError("Manifesto sem patches para a figura da divisão espacial.")

    x_min = min(x for _, x, _, _, _ in rectangles)
    y_min = min(y for _, _, y, _, _ in rectangles)
    x_max = max(x + width for _, x, _, width, _ in rectangles)
    y_max = max(y + height for _, _, y, _, height in rectangles)
    # Margem explícita: o matplotlib 3.11 não amplia os limites ao adicionar patches.
    margin_x = max((x_max - x_min) * 0.02, 0.5)
    margin_y = max((y_max - y_min) * 0.02, 0.5)

    def _build(plt: Any, figure: Any) -> None:
        axis = figure.add_subplot()
        for fold, x, y, width, height in rectangles:
            axis.add_patch(
                Rectangle(
                    (x - x_min, y - y_min),
                    width,
                    height,
                    facecolor=FOLD_COLORS[fold % len(FOLD_COLORS)],
                    edgecolor="white",
                    linewidth=0.3,
                    alpha=0.9,
                )
            )
        axis.set_xlim(-margin_x, (x_max - x_min) + margin_x)
        axis.set_ylim(-margin_y, (y_max - y_min) + margin_y)
        axis.set_aspect("equal")
        axis.set_xlabel("distância (km, oeste → leste)")
        axis.set_ylabel("distância (km, sul → norte)")
        axis.set_title("Cada quadrado é um patch; cores diferentes são dobras diferentes")
        handles = [
            Patch(
                facecolor=FOLD_COLORS[fold % len(FOLD_COLORS)],
                label=f"dobra {fold} ({int((manifest['fold'] == fold).sum())} patches)",
            )
            for fold in folds
        ]
        axis.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            frameon=False,
        )
        axis.grid(alpha=0.2)

    return _render(_build, context, height_cm=10.5)


def render_architectures(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Esquema conceitual das duas arquiteturas, com a mesma linguagem visual."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def _block(
        axis: Any,
        x: float,
        y: float,
        width: float,
        height: float,
        text: str = "",
        fontsize: float = 7.5,
    ) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.015,rounding_size=0.05",
                facecolor="#eceff1",
                edgecolor="#455a64",
                linewidth=0.8,
                zorder=2,
            )
        )
        if text:
            axis.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3)

    def _arrow(axis: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                color="#455a64",
                linewidth=0.9,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(1, 2)
        for axis in axes:
            axis.set_xlim(0, 8.4)
            axis.set_ylim(0, 5.4)
            axis.set_axis_off()

        # U-Net: pilha do codificador desce, gargalo ao centro, decodificador sobe.
        unet = axes[0]
        unet.set_title("U-Net: convoluções (vizinhança local)", fontsize=10)
        encoder_x, decoder_x = 1.15, 7.05
        levels = (4.15, 3.45, 2.75, 2.05)
        for level in levels:
            _block(unet, encoder_x, level, 0.62, 0.42)
            _block(unet, decoder_x, level, 0.62, 0.42)
        _block(unet, 4.1, 1.35, 1.5, 0.5, "gargalo", fontsize=7.5)
        for upper, lower in zip(levels[:-1], levels[1:], strict=True):
            _arrow(unet, (encoder_x, upper - 0.21), (encoder_x, lower + 0.21))
        _arrow(unet, (encoder_x, levels[-1] - 0.21), (4.1 - 0.75, 1.35))
        _arrow(unet, (4.1 + 0.75, 1.35), (decoder_x, levels[-1] - 0.21))
        for lower, upper in zip(levels[::-1][:-1], levels[::-1][1:], strict=True):
            _arrow(unet, (decoder_x, lower + 0.21), (decoder_x, upper - 0.21))
        # Conexões de salto: setas longas apenas no espaço vazio entre as pilhas.
        for level in levels:
            _arrow(unet, (encoder_x + 0.31, level), (decoder_x - 0.31, level))
        unet.text(encoder_x, 0.9, "codificador", ha="center", fontsize=8)
        unet.text(decoder_x, 0.9, "decodificador", ha="center", fontsize=8)
        unet.text(
            4.1,
            0.35,
            "as setas horizontais são as conexões de salto",
            ha="center",
            fontsize=7.5,
        )

        # SegFormer: quatro estágios, cabeça MLP e logits com espaço próprio.
        segformer = axes[1]
        segformer.set_title("SegFormer: atenção (cena inteira)", fontsize=10)
        stages = (0.85, 1.75, 2.65, 3.55)
        for index, x in enumerate(stages, start=1):
            _block(segformer, x, 3.0, 0.62, 1.5, str(index), fontsize=8)
            if index > 1:
                _arrow(segformer, (stages[index - 2] + 0.31, 3.0), (x - 0.31, 3.0))
        _block(segformer, 4.8, 3.0, 1.25, 0.95, "cabeça\nMLP", fontsize=7.2)
        _arrow(segformer, (3.86, 3.0), (4.175, 3.0))
        _block(segformer, 7.0, 3.0, 1.8, 0.95, "logits\n1/4 → entrada", fontsize=6.4)
        _arrow(segformer, (5.425, 3.0), (6.1, 3.0))
        segformer.text(
            2.35,
            1.6,
            "codificador (resolução 1/4 → 1/32)",
            ha="center",
            fontsize=7.5,
        )
        segformer.text(
            4.2,
            0.75,
            "cada token enxerga a cena inteira desde a primeira etapa",
            ha="center",
            fontsize=7.5,
        )

    return _render(_build, context, height_cm=6.5)


def render_loss_terms(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Ilustra os três termos da perda com formas sintéticas (sem dados reais)."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    grid = np.mgrid[0:128, 0:128]
    rows, cols = grid[0], grid[1]
    reference = ((cols - 54) / 30) ** 2 + ((rows - 66) / 26) ** 2 <= 1
    prediction = ((cols - 76) / 28) ** 2 + ((rows - 60) / 26) ** 2 <= 1
    eroded = reference.copy()
    for shift_row in (-1, 0, 1):
        for shift_col in (-1, 0, 1):
            eroded &= np.roll(np.roll(reference, shift_row, axis=0), shift_col, axis=1)
    boundary = reference & ~eroded
    overlap = reference & prediction

    def _canvas() -> np.ndarray:
        canvas = np.ones((128, 128, 3), dtype=float)
        canvas[reference] = (0.78, 0.91, 0.79)
        canvas[prediction] = (0.99, 0.90, 0.75)
        canvas[overlap] = (0.0, 0.62, 0.45)
        return canvas

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(1, 3)
        axes[0].imshow(_canvas())
        axes[0].set_title("Dice: sobreposição")
        axes[0].legend(
            handles=[
                Patch(facecolor=(0.78, 0.91, 0.79), edgecolor="#616161", label="referência"),
                Patch(facecolor=(0.99, 0.90, 0.75), edgecolor="#616161", label="predição"),
                Patch(facecolor=(0.0, 0.62, 0.45), edgecolor="#616161", label="acerto"),
            ],
            loc="upper left",
            fontsize=6.5,
            framealpha=0.9,
        )

        axes[1].imshow(_canvas())
        axes[1].contour(reference.astype(float), levels=[0.5], colors=[COLOR_REFERENCE])
        hard_rows, hard_cols = 108, 20
        axes[1].scatter([hard_cols], [hard_rows], s=40, facecolor="none", edgecolor="#c62828")
        axes[1].annotate(
            "pixels difíceis\n(minoria)",
            xy=(hard_cols, hard_rows),
            xytext=(38, 112),
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "#c62828"},
        )
        axes[1].set_title("Focal: pixels difíceis")

        axes[2].imshow(np.ones((128, 128, 3)))
        axes[2].imshow(reference, cmap="Greys", alpha=0.25)
        axes[2].imshow(
            np.ma.masked_where(~boundary, boundary),
            cmap=ListedColormap(["#e65100"]),
            vmin=0,
            vmax=1,
        )
        axes[2].set_title("Boundary: bordas do talhão")

        _hide(axes)
        figure.text(
            0.5,
            0.02,
            "A perda soma os três termos: acerta a área, trata os pixels difíceis "
            "e alinha as bordas.",
            ha="center",
            fontsize=7.5,
            wrap=True,
        )

    return _render(_build, context, height_cm=4.8)


def render_qualitative_predictions(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Painel qualitativo com imagem, referência, predições e mapas de erro."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    samples = data.error_samples
    if not samples:
        raise ValueError("Nenhum patch disponível para o painel qualitativo.")
    error_cmap = ListedColormap(
        (
            COLOR_TRUE_NEGATIVE,
            COLOR_FALSE_POSITIVE,
            COLOR_FALSE_NEGATIVE,
            COLOR_TRUE_POSITIVE,
            "#ffffff",
        )
    )
    error_norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), error_cmap.N)

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(len(samples), 6, squeeze=False)
        _row_titles(
            axes[0][1:],
            ["Referência", "U-Net", "SegFormer", "Erro U-Net", "Erro SegFormer"],
        )
        for row, sample in enumerate(samples):
            axes[row, 0].set_title(f"{sample.label}\n(dobra {sample.fold})", fontsize=8, loc="left")
            axes[row, 0].imshow(sample.image_rgb)
            truth_code = np.where(sample.valid, sample.truth.astype(np.uint8), 2)
            axes[row, 1].imshow(
                truth_code,
                cmap=ListedColormap(["#f2f2f2", COLOR_REFERENCE, "#ffffff"]),
                vmin=0,
                vmax=2,
            )
            for column, model_name in enumerate(MODEL_ORDER, start=2):
                prediction = np.where(
                    sample.valid, sample.predictions[model_name].astype(np.uint8), 2
                )
                axes[row, column].imshow(
                    prediction,
                    cmap=ListedColormap(["#f2f2f2", COLOR_REFERENCE, "#ffffff"]),
                    vmin=0,
                    vmax=2,
                )
                axes[row, column].set_title(f"IoU {_number(sample.iou[model_name])}", fontsize=8)
                codes = comparative.error_code_map(
                    sample.predictions[model_name], sample.truth, sample.valid
                )
                axes[row, column + 2].imshow(codes, cmap=error_cmap, norm=error_norm)
                axes[row, column + 2].set_title(
                    f"IoU {_number(sample.iou[model_name])}", fontsize=8
                )
        _hide(axes)
        handles = [
            Patch(facecolor=COLOR_TRUE_POSITIVE, label="acerto"),
            Patch(facecolor=COLOR_FALSE_POSITIVE, label="café inventado (falso positivo)"),
            Patch(facecolor=COLOR_FALSE_NEGATIVE, label="café não detectado (falso negativo)"),
            Patch(facecolor=COLOR_TRUE_NEGATIVE, label="acerto de fundo"),
            Patch(facecolor="#ffffff", edgecolor="#9e9e9e", label="fora da área"),
        ]
        figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
        figure.get_layout_engine().set(rect=(0, 0.06, 1, 1))

    return _render(_build, context, height_cm=15.0)


def render_metrics_overview(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Barras das métricas agregadas e diferença pareada com intervalo de confiança."""
    metrics = list(evaluation.METRIC_KEYS)

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(1, 2, width_ratios=(1.15, 1.0))

        positions = np.arange(len(metrics))
        height = 0.36
        for offset, model_name in zip((-height / 2, height / 2), MODEL_ORDER, strict=True):
            values = [data.summaries[model_name]["aggregate"]["metrics"][key] for key in metrics]
            bars = axes[0].barh(
                positions + offset,
                values,
                height=height,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
            )
            axes[0].bar_label(
                bars, labels=[_number(value, 2) for value in values], padding=2, fontsize=7
            )
        axes[0].set_yticks(positions, [METRIC_LABELS[key] for key in metrics])
        axes[0].set_xlim(0, 1.08)
        axes[0].set_xlabel("valor (0 a 1; quanto maior, melhor)")
        axes[0].set_title("Desempenho agregado (micro) na validação espacial")
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)

        means = []
        lows = []
        highs = []
        for metric in metrics:
            result = data.comparison["spatial_inference"]["block_level"][metric]
            mean = result.get("mean_difference")
            means.append(0.0 if mean is None else float(mean))
            lows.append(
                0.0
                if mean is None or result.get("ci95_low") is None
                else max(0.0, float(mean) - float(result["ci95_low"]))
            )
            highs.append(
                0.0
                if mean is None or result.get("ci95_high") is None
                else max(0.0, float(result["ci95_high"]) - float(mean))
            )
        axes[1].errorbar(
            positions,
            means,
            yerr=[lows, highs],
            fmt="o",
            color="#c62828",
            capsize=4,
            markersize=6,
        )
        axes[1].axhline(0.0, color="#212121", linewidth=1)
        axes[1].set_xticks(positions, [METRIC_LABELS[key] for key in metrics])
        axes[1].set_ylabel("diferença média")
        axes[1].set_title("Diferença SegFormer − U-Net (IC 95%, blocos)")
        for position, mean in zip(positions, means, strict=True):
            axes[1].annotate(
                _number(mean),
                xy=(position, mean),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )

    return _render(_build, context, height_cm=6.8)


def render_inference_levels(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Floresta de IC 95% da diferença segundo a unidade de análise."""
    return comparative.render_inference_figure(data.comparison)


def render_threshold_sensitivity(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Curvas P×R e IoU×limiar, com o limiar fixo e o ótimo."""
    return comparative.render_threshold_figure(data.comparison)


def render_patch_iou_distribution(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Distribuição do IoU por patch, com mediana e dispersão de cada modelo."""
    frame = data.patch_metrics
    coffee = frame[frame["truth_positive"]] if "truth_positive" in frame else frame

    def _build(plt: Any, figure: Any) -> None:
        axis = figure.add_subplot()
        values = [coffee[f"iou_{model_name}"].to_numpy(dtype=float) for model_name in MODEL_ORDER]
        parts = axis.violinplot(values, showextrema=False, widths=0.7)
        for body, model_name in zip(parts["bodies"], MODEL_ORDER, strict=True):  # type: ignore[attr-defined]
            body.set_facecolor(MODEL_COLORS[model_name])
            body.set_edgecolor("#37474f")
            body.set_alpha(0.25)
        for index, model_name in enumerate(MODEL_ORDER):
            sample = values[index]
            jitter = (
                np.linspace(-0.12, 0.12, len(sample)) if len(sample) > 1 else np.zeros(len(sample))
            )
            axis.scatter(
                np.full(len(sample), index + 1) + jitter,
                sample,
                s=12,
                color=MODEL_COLORS[model_name],
                alpha=0.65,
            )
            median = float(np.median(sample))
            axis.hlines(median, index + 0.62, index + 1.30, color="#212121", linewidth=2)
            axis.text(
                index + 1.34,
                median + 0.04,
                f"mediana {_number(median)}",
                fontsize=7.5,
                ha="left",
                va="bottom",
            )
        axis.set_xticks([1, 2], [MODEL_LABELS[name] for name in MODEL_ORDER])
        axis.set_xlim(0.4, 2.9)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("IoU por patch")
        axis.set_title(f"IoU de cada talhão de validação ({len(coffee)} patches com café)")

    return _render(_build, context, height_cm=7.5)


def render_fold_consistency(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Mostra o IoU dos dois modelos lado a lado em cada dobra."""
    folds = [int(item["fold"]) for item in data.summaries["unet"]["per_fold"]]

    def _build(plt: Any, figure: Any) -> None:
        axis = figure.add_subplot()
        for index, fold in enumerate(folds):
            values = [
                float(data.summaries[model_name]["per_fold"][index]["metrics"]["iou"])
                for model_name in MODEL_ORDER
            ]
            axis.plot(
                [values[0], values[1]],
                [fold, fold],
                color="#b0bec5",
                linewidth=2,
                zorder=1,
            )
            for model_name, value in zip(MODEL_ORDER, values, strict=True):
                axis.scatter(
                    value,
                    fold,
                    s=60,
                    color=MODEL_COLORS[model_name],
                    zorder=3,
                    edgecolor="white",
                    linewidth=0.6,
                )
                axis.annotate(
                    _number(value),
                    xy=(value, fold),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
        axis.set_yticks(folds, [f"dobra {fold}" for fold in folds])
        axis.invert_yaxis()
        axis.set_ylim(len(folds) - 0.4, -0.8)
        axis.set_xlabel("IoU na dobra de validação")
        axis.set_xlim(
            min(float(data.summaries[model]["fold_mean"]["iou"]) - 0.15 for model in MODEL_ORDER),
            1.0,
        )
        axis.set_title("A ordem dos modelos se mantém nas 5 dobras")
        for model_name in MODEL_ORDER:
            axis.scatter(
                [], [], color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name], s=60
            )
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    return _render(_build, context, height_cm=6.5)


def render_area_hectares(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Traduz as contagens da validação em hectares e decompõe o erro."""
    area = pixel_area_ha(data.config)

    def _build(plt: Any, figure: Any) -> None:
        axes = figure.subplots(1, 2)
        counts_by_model = {
            model_name: data.summaries[model_name]["aggregate"]["counts"]
            for model_name in MODEL_ORDER
        }
        reference_area = (
            int(counts_by_model["unet"]["tp"]) + int(counts_by_model["unet"]["fn"])
        ) * area
        mapped = [
            (int(counts_by_model[name]["tp"]) + int(counts_by_model[name]["fp"])) * area
            for name in MODEL_ORDER
        ]
        labels = ["Referência", *(f"{MODEL_LABELS[name]}\n(mapeado)" for name in MODEL_ORDER)]
        values = [reference_area, *mapped]
        colors = ["#78909c", *(MODEL_COLORS[name] for name in MODEL_ORDER)]
        bars = axes[0].bar(labels, values, color=colors, width=0.6)
        axes[0].bar_label(
            bars,
            labels=[f"{value:,.1f} ha".replace(",", ".") for value in values],
            padding=3,
            fontsize=7.5,
        )
        axes[0].set_ylabel("área de café (ha)")
        axes[0].set_title("Quanto de café cada modelo mapeou")

        width = 0.36
        positions = np.arange(len(MODEL_ORDER))
        for offset, key, color, legend in (
            (-width / 2, "fn", COLOR_FALSE_NEGATIVE, "não detectado"),
            (width / 2, "fp", COLOR_FALSE_POSITIVE, "detectado a mais"),
        ):
            values = [int(counts_by_model[name][key]) * area for name in MODEL_ORDER]
            bars = axes[1].bar(positions + offset, values, width=width, color=color, label=legend)
            axes[1].bar_label(
                bars, labels=[f"{value:,.1f}".replace(",", ".") for value in values], fontsize=7
            )
        axes[1].set_xticks(positions, [MODEL_LABELS[name] for name in MODEL_ORDER])
        axes[1].set_ylabel("área (ha)")
        axes[1].set_title("Onde está o erro")
        axes[1].legend(loc="upper right")

        figure.text(
            0.5,
            0.02,
            f"Amostra de validação ({int(data.summaries['unet']['n_patches'])} patches); "
            f"1 pixel = {_number(area, 2)} ha.",
            ha="center",
            fontsize=7.5,
        )
        figure.get_layout_engine().set(rect=(0, 0.07, 1, 1))

    return _render(_build, context, height_cm=7.4)


def render_stratum_performance(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """IoU médio por faixa de densidade de café no patch (baixa, média e alta)."""
    frame = data.patch_metrics
    coffee = frame[frame["truth_positive"]] if "truth_positive" in frame else frame
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0001)]
    labels = ("baixa\n(< 25%)", "média\n(25–50%)", "alta\n(> 50%)")

    def _means(lower: float, upper: float, model_name: str) -> tuple[float, float, int]:
        subset = coffee[(coffee["coffee_ratio"] >= lower) & (coffee["coffee_ratio"] < upper)]
        values = subset[f"iou_{model_name}"].to_numpy(dtype=float)
        if values.size == 0:
            return 0.0, 0.0, 0
        return float(values.mean()), float(values.std()), int(values.size)

    def _build(plt: Any, figure: Any) -> None:
        axis = figure.add_subplot()
        width = 0.36
        positions = np.arange(len(bins))
        for offset, model_name in zip((-width / 2, width / 2), MODEL_ORDER, strict=True):
            stats = [_means(lower, upper, model_name) for lower, upper in bins]
            axis.bar(
                positions + offset,
                [item[0] for item in stats],
                width=width,
                color=MODEL_COLORS[model_name],
                label=MODEL_LABELS[model_name],
                yerr=[item[1] for item in stats],
                capsize=3,
                error_kw={"linewidth": 0.8},
            )
        counts = [item[2] for item in (_means(lower, upper, "unet") for lower, upper in bins)]
        axis.set_xticks(
            positions,
            [f"{label}\n{count} patches" for label, count in zip(labels, counts, strict=True)],
        )
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("IoU médio")
        axis.set_title("Dificuldade por quantidade de café no patch")
        axis.legend(loc="lower right")

    return _render(_build, context, height_cm=6.8)


def render_xai_panel(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Mapa de calor dos dois modelos no mesmo talhão e relevância dentro/fora."""
    panel = data.xai_panel
    if panel is None:
        raise ValueError("Mapas de explicabilidade indisponíveis.")

    def _build(plt: Any, figure: Any) -> None:
        grid = figure.add_gridspec(2, 4, height_ratios=(3.2, 1.0))
        axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
        axes[0].imshow(panel.image_rgb)
        axes[0].set_title("Imagem", fontsize=9)
        axes[1].imshow(
            np.where(panel.reference, 1, 0),
            cmap="Greens",
            vmin=0,
            vmax=1,
        )
        axes[1].set_title("Referência", fontsize=9)
        heatmap_artist = None
        for axis, model_name in zip(axes[2:], data.xai_models, strict=True):
            heatmap = panel.heatmaps[model_name]
            axis.imshow(panel.image_rgb)
            heatmap_artist = axis.imshow(heatmap, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
            axis.set_title(
                f"{MODEL_LABELS[model_name]}\nIoU {_number(panel.iou.get(model_name))}", fontsize=9
            )
        _hide(axes)
        for axis in axes[2 + len(data.xai_models) :]:
            axis.set_visible(False)
        if heatmap_artist is not None and data.xai_models:
            figure.colorbar(
                heatmap_artist,
                ax=list(axes[2 : 2 + len(data.xai_models)]),
                shrink=0.6,
                pad=0.02,
                label="relevância (0 a 1)",
            )

        bars_axis = figure.add_subplot(grid[1, :])
        positions = np.arange(len(data.xai_models))
        width = 0.34
        subset = data.xai_table[data.xai_table["patch_id"] == panel.patch_id]
        inside = [
            float(subset.loc[subset["model"] == name, "relevance_inside"].mean())
            for name in data.xai_models
        ]
        outside = [
            float(subset.loc[subset["model"] == name, "relevance_outside"].mean())
            for name in data.xai_models
        ]
        inside_bars = bars_axis.bar(
            positions - width / 2,
            inside,
            width=width,
            color=COLOR_REFERENCE,
            label="dentro do café",
        )
        outside_bars = bars_axis.bar(
            positions + width / 2, outside, width=width, color="#f9a825", label="fora do café"
        )
        bars_axis.bar_label(inside_bars, labels=[_number(value, 2) for value in inside], fontsize=7)
        bars_axis.bar_label(
            outside_bars, labels=[_number(value, 2) for value in outside], fontsize=7
        )
        bars_axis.set_xticks(positions, [MODEL_LABELS[name] for name in data.xai_models])
        bars_axis.set_ylim(0, 1.05)
        bars_axis.set_ylabel("relevância média")
        bars_axis.set_title(
            f"O modelo olha para o café? (talhão {panel.patch_id}, dobra {panel.fold})",
            fontsize=9,
        )
        bars_axis.legend(loc="upper right", ncol=2)

    return _render(_build, context, height_cm=11.0)


def render_summary_card(data: MonographData, context: RenderContext) -> dict[str, bytes]:
    """Cartão-síntese com os quatro números que fecham a defesa."""
    unet = data.summaries["unet"]["aggregate"]["metrics"]
    segformer = data.summaries["segformer"]["aggregate"]["metrics"]
    delta = float(segformer["iou"]) - float(unet["iou"])
    per_fold = data.comparison["per_fold_delta"]
    wins = sum(1 for item in per_fold if float(item["delta"]["iou"]) >= 0.0)

    def _build(plt: Any, figure: Any) -> None:
        axis = figure.add_subplot()
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_axis_off()
        axis.text(0.5, 0.92, "Síntese do estudo", ha="center", fontsize=16, fontweight="bold")
        axis.text(
            0.5,
            0.84,
            f"SegFormer vs U-Net · validação espacial em {len(per_fold)} dobras · "
            f"{int(data.summaries['unet']['n_patches'])} patches",
            ha="center",
            fontsize=9,
            color="#546e7a",
        )
        cards = (
            (0.27, 0.60, "IoU da U-Net", _number(unet["iou"]), MODEL_COLORS["unet"]),
            (0.73, 0.60, "IoU do SegFormer", _number(segformer["iou"]), MODEL_COLORS["segformer"]),
            (
                0.27,
                0.30,
                "Diferença (SegFormer − U-Net)",
                _signed_points(delta),
                "#c62828",
            ),
            (
                0.73,
                0.30,
                "Dobras em que o SegFormer iguala ou supera",
                f"{wins} de {len(per_fold)}",
                "#455a64",
            ),
        )
        for x, y, label, value, color in cards:
            axis.text(x, y + 0.06, label, ha="center", fontsize=9, color="#546e7a")
            axis.text(x, y - 0.03, value, ha="center", fontsize=22, fontweight="bold", color=color)
        axis.text(
            0.5,
            0.08,
            f"Fonte de referência: {SOURCE_LABELS.get(data.source_name, data.source_name)} "
            f"({reference_year(data.source_name)}) · quanto maior o IoU, melhor",
            ha="center",
            fontsize=8,
            color="#546e7a",
        )

    return _render(_build, context, height_cm=8.0)


FIGURE_RENDERERS: dict[str, Callable[[MonographData, RenderContext], dict[str, bytes]]] = {
    "study_area": render_study_area,
    "pipeline": render_pipeline,
    "bands": render_bands,
    "label_sources": render_label_sources,
    "patch_anatomy": render_patch_anatomy,
    "spatial_folds": render_spatial_folds,
    "architectures": render_architectures,
    "loss_terms": render_loss_terms,
    "qualitative_predictions": render_qualitative_predictions,
    "metrics_overview": render_metrics_overview,
    "inference_levels": render_inference_levels,
    "threshold_sensitivity": render_threshold_sensitivity,
    "patch_iou_distribution": render_patch_iou_distribution,
    "fold_consistency": render_fold_consistency,
    "area_hectares": render_area_hectares,
    "stratum_performance": render_stratum_performance,
    "xai_panel": render_xai_panel,
    "summary_card": render_summary_card,
}


# ---------------------------------------------------------------------------
# Legendas e persistência
# ---------------------------------------------------------------------------


def _figure_preamble(source_name: str) -> str:
    """Texto de abertura dos arquivos de legendas (fonte, ano e caminho)."""
    return (
        f"Legendas geradas pelo estágio 17 para a fonte {source_name} "
        f"({reference_year(source_name)})."
    )


def build_captions_markdown(
    source_name: str, figures: list[str], tables: list[str], figure_suffix: str = "png"
) -> str:
    """Monta o arquivo Markdown com legendas numeradas de figuras e tabelas."""
    lines = [f"# Legendas — {source_name}", "", _figure_preamble(source_name), ""]
    section = None
    for number, key in enumerate(figures, start=1):
        figure_spec = FIGURE_CATALOG[key]
        if figure_spec.section != section:
            section = figure_spec.section
            lines += [f"## {section}", ""]
        lines += [
            f"**Figura {number} — {key}**",
            "",
            f"![Figura {number}](figures/{key}.{figure_suffix})",
            "",
            f"*Figura {number}: {figure_spec.caption}*",
            "",
        ]
    section = None
    for number, key in enumerate(tables, start=1):
        table_spec = TABLE_CATALOG[key]
        if table_spec.section != section:
            section = table_spec.section
            lines += [f"## {section} (tabelas)", ""]
        lines += [f"**Tabela {number} — {key}**: {table_spec.caption}", ""]
    return "\n".join(lines).rstrip() + "\n"


def build_captions_latex(
    source_name: str, figures: list[str], tables: list[str], figure_suffix: str = "pdf"
) -> str:
    """Monta o arquivo LaTeX com ambientes figure/table e rótulos estáveis."""
    base = f"artifacts/monograph/{source_name}"
    lines = [
        f"% Legendas geradas pelo estágio 17 para a fonte {source_name} "
        f"({reference_year(source_name)}).",
        "% Ajuste o caminho das imagens com \\graphicspath se necessário.",
        "",
    ]
    section = None
    for key in figures:
        figure_spec = FIGURE_CATALOG[key]
        if figure_spec.section != section:
            section = figure_spec.section
            lines += [f"% --- {section} ---", ""]
        lines += [
            "\\begin{figure}[htbp]",
            "  \\centering",
            f"  \\includegraphics[width=\\textwidth]{{{base}/figures/{key}.{figure_suffix}}}",
            f"  \\caption{{{figure_spec.caption}}}",
            f"  \\label{{fig:{key}}}",
            "\\end{figure}",
            "",
        ]
    section = None
    for key in tables:
        table_spec = TABLE_CATALOG[key]
        if table_spec.section != section:
            section = table_spec.section
            lines += [f"% --- {section} (tabelas) ---", ""]
        lines += [
            "\\begin{table}[htbp]",
            "  \\centering",
            f"  \\caption{{{table_spec.caption}}}",
            f"  \\label{{tab:{key}}}",
            f"  \\input{{{base}/tables/{key}.tex}}",
            "\\end{table}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def _persist_table(
    storage_paths: dict[str, Path], source_name: str, key: str, table: pd.DataFrame
) -> Path:
    """Persiste a tabela didática em CSV (separador ';') e LaTeX prontos."""
    from src import io

    csv_path = table_path(storage_paths, source_name, key, "csv")
    io.persist_bytes(csv_path, table.to_csv(index=False, sep=";").encode("utf-8"))
    latex = table.to_latex(index=False, na_rep="--", escape=True)
    io.persist_bytes(table_path(storage_paths, source_name, key, "tex"), latex.encode("utf-8"))
    return csv_path


def _summary_payload(data: MonographData) -> dict[str, Any]:
    """Números-chave que resumem o estudo no inventário da monografia."""
    unet = data.summaries["unet"]["aggregate"]["metrics"]
    segformer = data.summaries["segformer"]["aggregate"]["metrics"]
    per_fold = data.comparison["per_fold_delta"]
    return {
        "n_patches": int(data.summaries["unet"]["n_patches"]),
        "n_folds": int(len(per_fold)),
        "unet_iou": float(unet["iou"]),
        "segformer_iou": float(segformer["iou"]),
        "iou_delta": float(segformer["iou"]) - float(unet["iou"]),
        "folds_segformer_at_least_unet": int(
            sum(1 for item in per_fold if float(item["delta"]["iou"]) >= 0.0)
        ),
        "area_reference_ha": float(
            (
                int(data.summaries["unet"]["aggregate"]["counts"]["tp"])
                + int(data.summaries["unet"]["aggregate"]["counts"]["fn"])
            )
            * pixel_area_ha(data.config)
        ),
    }


def run_monograph(
    storage_paths: dict[str, Path],
    source_name: str | None = None,
    force: bool = False,
) -> MonographResult:
    """Gera (ou reutiliza) as figuras, tabelas e legendas da monografia."""
    from src import io

    source, summaries, comparison, xai_models = synthesis.validate_inputs(
        storage_paths, source_name
    )
    data = build_monograph_data(
        storage_paths,
        source,
        summaries=summaries,
        comparison=comparison,
        xai_models=xai_models,
    )
    context = RenderContext(
        width_cm=FIGURE_WIDTH_CM,
        dpi=FIGURE_DPI,
        formats=FIGURE_FORMATS,
    )
    figures = available_figures(data)
    tables = available_tables(data)
    fingerprint = monograph_fingerprint(storage_paths, data, figures, tables)
    omitted = [key for key in FIGURE_CATALOG if key not in figures]
    if omitted:
        print(f"Figuras omitidas por falta de insumo: {', '.join(omitted)}")

    if not force and _current_artifacts(
        storage_paths, source, fingerprint, figures, tables, context.formats
    ):
        inventory = _read_json(inventory_path(storage_paths, source))
        print("Monografia já vigente (figuras, tabelas e legendas reutilizadas).")
        # Garante as legendas na sessão atual (no Kaggle o Drive é remoto).
        for suffix in ("tex", "md"):
            io.ensure_local_copy(captions_path(storage_paths, source, suffix))
        return MonographResult(
            source_name=source,
            figures={
                key: {fmt: figure_path(storage_paths, source, key, fmt) for fmt in context.formats}
                for key in figures
            },
            tables={key: table_path(storage_paths, source, key, "csv") for key in tables},
            tables_data={
                key: pd.read_csv(
                    io.ensure_local_copy(table_path(storage_paths, source, key, "csv")),
                    sep=";",
                )
                for key in tables
            },
            captions_tex_path=captions_path(storage_paths, source, "tex"),
            captions_md_path=captions_path(storage_paths, source, "md"),
            inventory_path=inventory_path(storage_paths, source),
            summary=inventory.get("summary", {}),
            skipped=True,
        )

    tables_data = build_tables(data)
    for key in tables:
        _persist_table(storage_paths, source, key, tables_data[key])

    rendered: dict[str, dict[str, Path]] = {}
    for key in figures:
        payloads = FIGURE_RENDERERS[key](data, context)
        rendered[key] = {}
        for figure_format, payload in payloads.items():
            path = figure_path(storage_paths, source, key, figure_format)
            io.persist_bytes(path, payload)
            rendered[key][figure_format] = path
        print(f"Figura {key}: {rendered[key][context.formats[0]]}")

    captions_tex = captions_path(storage_paths, source, "tex")
    captions_md = captions_path(storage_paths, source, "md")
    latex_suffix = "pdf" if "pdf" in context.formats else context.formats[0]
    markdown_suffix = "png" if "png" in context.formats else context.formats[0]
    io.persist_bytes(
        captions_tex,
        build_captions_latex(source, figures, tables, latex_suffix).encode("utf-8"),
    )
    io.persist_bytes(
        captions_md,
        build_captions_markdown(source, figures, tables, markdown_suffix).encode("utf-8"),
    )

    summary = _summary_payload(data)
    inventory = {
        "schema_version": MONOGRAPH_SCHEMA_VERSION,
        "source_name": source,
        "year": reference_year(source),
        "fingerprint": fingerprint,
        "figures": {
            key: {
                "section": FIGURE_CATALOG[key].section,
                "caption": FIGURE_CATALOG[key].caption,
                "files": {fmt: str(path) for fmt, path in rendered[key].items()},
            }
            for key in figures
        },
        "tables": {
            key: {
                "section": TABLE_CATALOG[key].section,
                "caption": TABLE_CATALOG[key].caption,
                "file": str(table_path(storage_paths, source, key, "csv")),
            }
            for key in tables
        },
        "captions": {"tex": str(captions_tex), "md": str(captions_md)},
        "summary": summary,
    }
    io.persist_json(inventory_path(storage_paths, source), inventory)
    print(f"Monografia concluída: {len(figures)} figuras | {len(tables)} tabelas")
    print(f"Legendas: {captions_tex}")
    return MonographResult(
        source_name=source,
        figures=rendered,
        tables={key: table_path(storage_paths, source, key, "csv") for key in tables},
        tables_data=tables_data,
        captions_tex_path=captions_tex,
        captions_md_path=captions_md,
        inventory_path=inventory_path(storage_paths, source),
        summary=summary,
        skipped=False,
    )
