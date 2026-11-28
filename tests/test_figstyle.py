"""Testes de src/figstyle.py (estilo e persistência das figuras de publicação)."""

from __future__ import annotations

from pathlib import Path

from src import figstyle


def test_render_publication_returns_png_and_pdf() -> None:
    """A renderização deve produzir PNG e PDF vetorial nas dimensões em centímetros."""

    def _build(plt, figure) -> None:
        axis = figure.subplots()
        axis.plot([0, 1], [0, 1])
        axis.set_title("Teste")

    rendered = figstyle.render_publication(_build, height_cm=6.0, width_cm=12.0, dpi=100)

    assert rendered["png"].startswith(b"\x89PNG")
    assert rendered["pdf"].startswith(b"%PDF")


def test_persist_figure_writes_every_format(tmp_path: Path, monkeypatch) -> None:
    """Todos os formatos devem ser gravados ao lado do PNG canônico."""
    monkeypatch.setattr("src.io.detect_platform", lambda: "local")
    target = tmp_path / "figs" / "exemplo.png"

    returned = figstyle.persist_figure(target, {"png": b"\x89PNGdados", "pdf": b"%PDFdados"})

    assert returned == target
    assert target.read_bytes() == b"\x89PNGdados"
    assert target.with_suffix(".pdf").read_bytes() == b"%PDFdados"
