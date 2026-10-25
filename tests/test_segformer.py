"""Testes de src/models/segformer.py (adaptação de entrada e saída binária)."""

from __future__ import annotations

import pytest
import torch

from src.models.segformer import PATCH_MULTIPLE, SegFormerBinary, build_segformer


def _model() -> SegFormerBinary:
    """Constrói o SegFormer com pesos aleatórios (sem download de pré-treino)."""
    return build_segformer(pretrained=False)


def test_build_segformer_adapts_first_projection_to_bands() -> None:
    """A projeção de entrada deve aceitar as quatro bandas do Sentinel-2."""
    model = _model()
    projection = model.model.segformer.stages[0].patch_embeddings.proj
    assert projection.in_channels == 4
    assert model.model.config.num_channels == 4
    assert model.in_channels == 4


def test_segformer_forward_preserves_input_resolution() -> None:
    """A saída deve ter 1 canal de logit na resolução espacial da entrada."""
    model = _model().eval()
    logits = model(torch.randn(1, 4, 64, 64))
    assert logits.shape == (1, 1, 64, 64)


def test_segformer_initializes_extra_channel_with_rgb_mean() -> None:
    """O quarto canal deve ser a média dos filtros RGB pré-treinados."""
    model = _model()
    projection = model.model.segformer.stages[0].patch_embeddings.proj
    assert torch.allclose(projection.weight[:, 3], projection.weight[:, :3].mean(dim=1))


def test_segformer_rejects_incompatible_input_channels() -> None:
    """Entrada com número de bandas diferente do configurado deve falhar."""
    model = _model()
    with pytest.raises(ValueError, match="Entrada esperada"):
        model(torch.randn(1, 3, 64, 64))


def test_segformer_rejects_non_multiple_of_patch_stride() -> None:
    """Resolução incompatível com o stride do MiT deve falhar com mensagem clara."""
    model = _model()
    with pytest.raises(ValueError, match=str(PATCH_MULTIPLE)):
        model(torch.randn(1, 4, 48, 48))


def test_segformer_backward_reaches_input_projection() -> None:
    """O gradiente deve fluir até a projeção de entrada adaptada."""
    model = _model()
    projection = model.model.segformer.stages[0].patch_embeddings.proj
    model(torch.randn(1, 4, 64, 64)).mean().backward()
    assert projection.weight.grad is not None
