"""Padrão tipográfico e de saída das figuras do projeto (monografia, artigo e slides).

Centraliza o estilo matplotlib, as dimensões em centímetros (largura da área de
texto ABNT), a resolução, os formatos de saída (PNG a 300 dpi e PDF vetorial) e a
paleta daltônica usada por todos os estágios que geram figuras. As dimensões são
definidas no tamanho final de impressão: as fontes (8–10 pt) não são reduzidas
quando a figura é inserida no texto, e o PDF preserva o texto como vetor.

Os parâmetros ficam neste módulo (e não em `config.yaml`) porque o hash do
`config.yaml` é o fingerprint dos artefatos de treino e avaliação: acrescentar
chaves de apresentação invalidaria os estágios 09–15.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from src.data.raster_utils import render_figure_formats

FIGURE_WIDTH_CM = 16.0  # largura da área de texto da página ABNT
FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "pdf")

MODEL_LABELS = {"unet": "U-Net", "segformer": "SegFormer"}
MODEL_COLORS = {"unet": "#1565c0", "segformer": "#6a1b9a"}
FOLD_COLORS = ("#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9")
METRIC_LABELS = {"iou": "IoU", "f1": "F1", "precision": "Precisão", "recall": "Revocação"}
COLOR_REFERENCE = "#2e7d32"
COLOR_INSIDE = "#2e7d32"
COLOR_OUTSIDE = "#e69f00"
COLOR_NEUTRAL = "#e0e0e0"
COLOR_ACCENT = "#c62828"


@contextlib.contextmanager
def publication_style() -> Iterator[None]:
    """Aplica o padrão tipográfico das figuras de publicação e o restaura ao final."""
    import matplotlib as mpl

    saved = dict(mpl.rcParams)
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.titleweight": "regular",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,  # fontes TrueType embutidas (texto editável no PDF)
            "ps.fonttype": 42,
        }
    )
    try:
        yield
    finally:
        mpl.rcParams.update(saved)


def render_publication(
    build_fn: Callable[[Any, Any], None],
    *,
    height_cm: float,
    width_cm: float = FIGURE_WIDTH_CM,
    dpi: int = FIGURE_DPI,
    formats: tuple[str, ...] = FIGURE_FORMATS,
) -> dict[str, bytes]:
    """Renderiza uma figura nas dimensões finais de impressão, em todos os formatos.

    ``build_fn`` recebe ``(plt, figure)`` e monta a figura (eixos, títulos e
    legendas); o layout `constrained` evita sobreposição de rótulos.
    """

    def _build(plt: Any) -> Any:
        with publication_style():
            figure = plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54), layout="constrained")
            build_fn(plt, figure)
        return figure

    return render_figure_formats(_build, dpi=dpi, formats=formats)


def panel_label(axis: Any, label: str) -> None:
    """Prefixa o título do painel com a letra da subfigura, alinhado à esquerda.

    Deve ser chamado depois de `set_title`; o título passa a ocupar a posição
    esquerda, o que evita a colisão da letra com títulos longos.
    """
    title = axis.get_title()
    axis.set_title("", loc="center")
    axis.set_title(f"{label} {title}".rstrip(), loc="left")


def wrap_label(text: str, width: int = 16) -> str:
    """Quebra um rótulo longo em linhas curtas (rótulos de linha de painéis de imagens)."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width)) or text


def figure_formats_paths(
    png_path: Path, formats: tuple[str, ...] = FIGURE_FORMATS
) -> dict[str, Path]:
    """Caminhos de cada formato de uma figura, derivados do caminho do PNG canônico."""
    return {fmt: png_path.with_suffix(f".{fmt}") for fmt in formats}


def persist_figure(png_path: Path, rendered: dict[str, bytes]) -> Path:
    """Persiste todos os formatos renderizados ao lado do PNG canônico e o retorna."""
    from src import io

    for fmt, payload in rendered.items():
        io.persist_bytes(png_path.with_suffix(f".{fmt}"), payload)
    return png_path
