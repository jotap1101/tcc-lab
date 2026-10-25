"""Funções de perda multivariada: Dice + Focal + Boundary (estágios 09/10).

A perda combinada segue o mesmo protocolo para U-Net (estágio 09) e SegFormer
(estágio 10), com pesos definidos em config.yaml (`loss.*`). A perda de fronteira
(baseada no mapa de distância assinada, Kervadec et al. 2019) é computada em CPU
via `scipy.ndimage.distance_transform_edt`, determinística, e o gradiente flui
apenas pelas probabilidades previstas — o mapa de distância é constante em
relação aos parâmetros do modelo. Pixels com rótulo -1 (fora do AOI) são
ignorados em todos os componentes.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

SMOOTH = 1.0  # suavização da Dice (evita divisão por zero em classes vazias)
FOCAL_ALPHA = 0.25  # ponderação de classes raras na Focal Loss
FOCAL_GAMMA = 2.0  # ênfase em exemplos difíceis da Focal Loss
PROB_EPS = 1e-6  # clamp das probabilidades para estabilidade numérica


def _valid_mask(targets: torch.Tensor) -> torch.Tensor:
    """Máscara 0/1 dos pixels com rótulo válido (valores >= 0; -1 é ignorado)."""
    return (targets >= 0).to(targets.dtype)


def _binary_targets(targets: torch.Tensor) -> torch.Tensor:
    """Converte o alvo em binário, tratando o rótulo de ignorados (-1) como 0."""
    return targets.clamp(min=0.0)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Dice Loss binária (soft, diferenciável) sobre a classe de interesse (café).

    Compara a probabilidade prevista (`sigmoid(logits)`) com a máscara binária
    alvo em nível de pixel; retorna 1 - Dice, com valor mínimo 0 para a
    previsão perfeita. Pixels com rótulo -1 não entram no cálculo.
    """
    valid = _valid_mask(targets)
    binary = _binary_targets(targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * binary * valid).sum()
    cardinality = (probs * valid).sum() + (binary * valid).sum()
    dice = (2.0 * intersection + SMOOTH) / (cardinality + SMOOTH)
    return 1.0 - dice


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    """Focal Loss binária, que concentra o aprendizado nos exemplos difíceis.

    Pondera a entropia cruzada por ``(1 - pt)**gamma``, onde ``pt`` é a
    probabilidade atribuída à classe correta; ``alpha`` reequilibra as classes.
    Pixels com rótulo -1 (fora do AOI) são ignorados.
    """
    valid = _valid_mask(targets)
    binary = _binary_targets(targets)
    probs = torch.sigmoid(logits).clamp(PROB_EPS, 1.0 - PROB_EPS)
    pt = probs * binary + (1.0 - probs) * (1.0 - binary)
    weights = (1.0 - pt) ** gamma
    alpha_t = alpha * binary + (1.0 - alpha) * (1.0 - binary)
    bce = nn.functional.binary_cross_entropy(probs, binary, reduction="none")
    denominator = valid.sum().clamp_min(1.0)
    return (alpha_t * weights * bce * valid).sum() / denominator


def signed_distance_map(targets: torch.Tensor) -> torch.Tensor:
    """Mapa de distância assinada em relação à fronteira do objeto (CPU).

    Pixels dentro do objeto recebem distância negativa até a borda; pixels fora
    recebem distância positiva. Cálculo determinístico via `distance_transform_edt`.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    masks = targets.detach().cpu().numpy() > 0.5
    distances = np.zeros_like(masks, dtype=np.float32)
    for index in range(masks.shape[0]):
        mask = masks[index, 0]
        # Cenas uniformes (objeto ausente ou preenchendo a cena) não têm fronteira.
        if mask.all() or not mask.any():
            continue
        inside = distance_transform_edt(mask)
        outside = distance_transform_edt(~mask)
        distances[index, 0] = np.where(mask, -inside, outside)
    return torch.from_numpy(distances)


def boundary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Boundary Loss baseada no mapa de distância assinada (Kervadec et al.).

    Penaliza previsões ponderadas pela distância à fronteira do objeto,
    normalizada pela soma dos valores absolutos do mapa — invariante à escala e
    nula quando o mapa é todo zero (objeto ausente ou preenchendo a cena).
    Pixels com rótulo -1 (fora do AOI) são ignorados.
    """
    valid = _valid_mask(targets)
    signed = signed_distance_map(_binary_targets(targets)).to(logits.device)
    probs = torch.sigmoid(logits)
    denominator = (signed.abs() * valid).sum()
    if denominator == 0:
        return torch.zeros((), device=logits.device)
    return (signed * probs * valid).sum() / denominator


class SegmentationLoss(nn.Module):
    """Perda multivariada Dice + Focal + Boundary com pesos de config.yaml.

    ``forward`` retorna um dicionário com a perda total (média ponderada
    normalizada pela soma dos pesos) e cada componente, para monitoramento.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        weights = config["loss"]
        self.dice_weight = float(weights["dice_weight"])
        self.focal_weight = float(weights["focal_weight"])
        self.boundary_weight = float(weights["boundary_weight"])
        self.normalizer = self.dice_weight + self.focal_weight + self.boundary_weight
        if self.normalizer <= 0:
            raise ValueError("A soma dos pesos da perda deve ser positiva.")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        dice = dice_loss(logits, targets)
        focal = focal_loss(logits, targets)
        if self.boundary_weight > 0:
            boundary = boundary_loss(logits, targets)
        else:
            boundary = torch.zeros((), device=logits.device)
        total = (
            self.dice_weight * dice + self.focal_weight * focal + self.boundary_weight * boundary
        ) / self.normalizer
        return {"total": total, "dice": dice, "focal": focal, "boundary": boundary}
