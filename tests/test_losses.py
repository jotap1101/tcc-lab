"""Testes de src/losses.py (perda multivariada Dice + Focal + Boundary)."""

from __future__ import annotations

import torch

from src.losses import (
    SegmentationLoss,
    boundary_loss,
    dice_loss,
    focal_loss,
    signed_distance_map,
)


def _binary_logits(targets: torch.Tensor) -> torch.Tensor:
    """Logits perfeitamente alinhados à máscara binária (certo => +5, fundo => -5)."""
    return torch.where(targets > 0.5, torch.full_like(targets, 5.0), torch.full_like(targets, -5.0))


def test_dice_loss_perfect_prediction_is_zero() -> None:
    """Com previsão perfeita, a Dice Loss deve ser (aproximadamente) zero."""
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = _binary_logits(targets)
    assert float(dice_loss(logits, targets)) < 1e-2


def test_dice_loss_wrong_prediction_is_positive() -> None:
    """Com previsão invertida, a Dice Loss deve ser positiva e grande."""
    targets = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    logits = torch.full_like(targets, -5.0)
    assert float(dice_loss(logits, targets)) > 0.5


def test_dice_loss_rewards_correct_background_prediction() -> None:
    """Em cena sem café, prever fundo deve gerar perda menor que prever café."""
    targets = torch.zeros(1, 1, 8, 8)
    correct = dice_loss(torch.full_like(targets, -7.0), targets)
    wrong = dice_loss(torch.full_like(targets, 7.0), targets)
    assert float(correct) < float(wrong)


def test_focal_loss_perfect_prediction_is_small() -> None:
    """Previsão perfeita deve gerar perda focal próxima de zero."""
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    logits = _binary_logits(targets)
    assert float(focal_loss(logits, targets)) < 1e-2


def test_focal_loss_hard_examples_weighted_more() -> None:
    """Exemplos difíceis (pt baixo) devem contribuir mais que os fáceis."""
    targets = torch.tensor([[[[1.0, 1.0]]]])
    easy = focal_loss(torch.full_like(targets, 4.0), targets)
    hard = focal_loss(torch.full_like(targets, -4.0), targets)
    assert float(hard) > float(easy)


def test_signed_distance_map_negative_inside_positive_outside() -> None:
    """O mapa assinado deve ser negativo dentro e positivo fora do objeto."""
    targets = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]])
    distances = signed_distance_map(targets)
    assert distances[0, 0, 1, 1] < 0  # interior do objeto
    assert distances[0, 0, 2, 2] > 0  # região externa


def test_boundary_loss_empty_scene_is_zero() -> None:
    """Cena sem objeto (mapa de distância nulo) deve ter perda de fronteira zero."""
    targets = torch.zeros(1, 1, 8, 8)
    logits = torch.randn(1, 1, 8, 8)
    assert float(boundary_loss(logits, targets)) == 0.0


def test_boundary_loss_returns_scalar_with_gradient() -> None:
    """A perda de fronteira deve retornar um escalar diferenciável."""
    targets = torch.tensor([[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]])
    logits = torch.randn(1, 1, 3, 3, requires_grad=True)
    loss = boundary_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_losses_ignore_pixels_with_ignore_label() -> None:
    """Logits em pixels com rótulo -1 não devem afetar nenhum componente da perda."""
    targets = torch.tensor([[[[0.0, 1.0], [0.0, -1.0]]]])
    baseline = torch.zeros_like(targets)
    changed = baseline.clone()
    changed[0, 0, 1, 1] = 100.0  # pixel inválido com logit extremo

    assert torch.allclose(dice_loss(baseline, targets), dice_loss(changed, targets))
    assert torch.allclose(focal_loss(baseline, targets), focal_loss(changed, targets))
    assert torch.allclose(boundary_loss(baseline, targets), boundary_loss(changed, targets))


def test_losses_with_only_invalid_pixels_are_finite() -> None:
    """Cena integralmente inválida deve produzir perdas finitas e nulas na fronteira."""
    targets = torch.full((1, 1, 4, 4), -1.0)
    logits = torch.randn(1, 1, 4, 4)
    assert float(dice_loss(logits, targets)) == 0.0
    assert float(focal_loss(logits, targets)) == 0.0
    assert float(boundary_loss(logits, targets)) == 0.0


def test_segmentation_loss_returns_weighted_components() -> None:
    """A perda combinada deve retornar total e componentes com os pesos do config."""
    config = {"loss": {"dice_weight": 1.0, "focal_weight": 2.0, "boundary_weight": 0.0}}
    loss_fn = SegmentationLoss(config)
    targets = torch.tensor([[[[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]]])
    logits = torch.randn(1, 1, 2, 4, requires_grad=True)
    components = loss_fn(logits, targets)
    assert set(components) == {"total", "dice", "focal", "boundary"}
    expected = (components["dice"] + 2.0 * components["focal"]) / 3.0
    assert torch.allclose(components["total"], expected)


def test_segmentation_loss_requires_positive_weights() -> None:
    """A soma dos pesos deve ser positiva."""
    import pytest

    config = {"loss": {"dice_weight": 0.0, "focal_weight": 0.0, "boundary_weight": 0.0}}
    with pytest.raises(ValueError, match="positiva"):
        SegmentationLoss(config)
