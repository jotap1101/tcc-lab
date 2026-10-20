"""Testes de src/data/mask_utils.py (máscaras de referência por fonte de ground truth)."""

import pytest

from src.config import get_config
from src.data import mask_utils


def test_reference_year_derives_from_data_dates() -> None:
    """O ano de referência deve derivar das datas de início do mosaico."""
    config = get_config()
    expected = int(config["data"]["dates"]["start"][:4])
    assert mask_utils.reference_year() == expected


def test_reference_year_per_source_override() -> None:
    """Cada fonte habilitada deve declarar o seu ano próprio (mais recente disponível)."""
    assert mask_utils.reference_year("emater") == 2018
    assert mask_utils.reference_year("mapbiomas") == 2025
    assert mask_utils.reference_year("alphaearth") == 2024


def test_active_year_follows_chosen_source() -> None:
    """O ano ativo deve ser o da fonte escolhida em config.yaml."""
    chosen = get_config()["ground_truth"]["chosen_source"]
    assert mask_utils.active_year() == mask_utils.reference_year(chosen)


def test_required_data_years_derived_from_enabled_sources() -> None:
    """Os anos necessários devem ser os anos próprios das fontes habilitadas."""
    years = mask_utils.required_data_years()
    assert {2018, 2024, 2025} <= set(years)
    assert years == sorted(years)


def test_mask_file_name_stable() -> None:
    """O nome da máscara deve derivar da fonte, região e ano."""
    assert mask_utils.mask_file_name("mapbiomas", "310044", 2025) == "mask_mapbiomas_310044_2025"


def test_mask_file_name_embeds_threshold_for_alphaearth() -> None:
    """O nome da máscara da AlphaEarth deve embutir o limiar (idempotência ao recalibrar)."""
    threshold = get_config()["ground_truth"]["sources"]["alphaearth"]["probability_threshold"]
    suffix = f"_t{int(float(threshold) * 100):03d}"
    assert mask_utils.mask_file_name("alphaearth", "310044", 2024).endswith(suffix)


def test_ground_truth_sources_have_required_config() -> None:
    """Cada fonte habilitada deve expor coleção, ano, resolução nativa e classe/limiar."""
    sources = get_config()["ground_truth"]["sources"]

    mapbiomas = sources["mapbiomas"]
    assert mapbiomas["collection_id"]
    assert mapbiomas["coffee_class"] == 46
    assert mapbiomas["year"] == 2025
    assert mapbiomas["native_resolution_m"] == 30  # Landsat; reamostrada para 10 m

    alphaearth = sources["alphaearth"]
    assert alphaearth["collection_id"]
    assert 0 < alphaearth["probability_threshold"] <= 1
    assert alphaearth["year"] == 2024
    assert alphaearth["native_resolution_m"] == 10

    emater = sources["emater"]
    assert emater["native_resolution_m"] == 10  # rasterizada no grid de 10 m do AOI

    assert "s2dr" not in sources


def test_mask_builders_registered_per_source() -> None:
    """Deve existir um construtor GEE dedicado para cada fonte habilitada não-vetorial."""
    sources = get_config()["ground_truth"]["sources"]
    for name, cfg in sources.items():
        if cfg["enabled"] and cfg.get("kind") != "vector":
            assert name in mask_utils.MASK_BUILDERS


def test_vector_source_declares_year_and_data_dir() -> None:
    """Fontes vetoriais devem declarar ano próprio e diretório de dados."""
    emater = get_config()["ground_truth"]["sources"]["emater"]
    assert emater["kind"] == "vector"
    assert emater["year"] == 2018
    assert emater["data_dir"]


def test_build_mask_raises_for_unknown_source() -> None:
    """Uma fonte desconhecida deve levantar ValueError."""
    with pytest.raises(ValueError):
        mask_utils.build_mask("fonte_inexistente", None)


def test_ensure_source_mask_disabled_returns_none(monkeypatch) -> None:
    """Uma fonte desabilitada não deve exportar nada e retornar None."""
    config = get_config()
    config["ground_truth"]["sources"]["emater"]["enabled"] = False
    monkeypatch.setattr(mask_utils, "get_config", lambda: config)
    assert mask_utils.ensure_source_mask("emater", None, {}) is None
