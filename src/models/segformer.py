"""SegFormer (ViT) para segmentação binária de lavouras cafeeiras (estágio 10).

Adapta o SegFormer do Hugging Face Transformers (`model.segformer_variant`,
padrão `nvidia/mit-b1`) à entrada de quatro bandas do Sentinel-2 (B2/B3/B4/B8) e
à saída de um canal de logit binário, compatível com o protocolo único de
`src.trainer`. A primeira convolução de projeção é expandida de 3 para 4 canais:
os pesos RGB pré-treinados são preservados e o quarto canal é inicializado com a
média dos três. O checkpoint `nvidia/mit-b1` é de backbone (ImageNet), então a
cabeça de decodificação é inicializada aleatoriamente e ajustada no treino. Os
logits saem em 1/4 da resolução e são reamostrados para a resolução da entrada.

O upsampling bilinear não possui backward determinístico no CUDA segundo o
PyTorch; por isso o estágio 10 habilita `warn_only=True` em
`torch.use_deterministic_algorithms` (ver `src.utils.setup_reproducibility`),
mantendo sementes, cuDNN e as demais operações determinísticas.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.config import get_config

PATCH_MULTIPLE = 32  # stride total das quatro etapas do MiT (4 x 2 x 2 x 2)


def _load_pretrained(variant: str) -> Any:
    """Carrega o SegFormer pré-treinado com uma cabeça binária de 1 logit."""
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    model_config = SegformerConfig.from_pretrained(variant)
    model_config.num_labels = 1
    model_config.id2label = {0: "coffee"}
    model_config.label2id = {"coffee": 0}
    # O checkpoint é de backbone; a cabeça de segmentação é inicializada do zero.
    return SegformerForSemanticSegmentation.from_pretrained(
        variant, config=model_config, ignore_mismatched_sizes=True
    )


def _build_random() -> Any:
    """Constrói um SegFormer com pesos aleatórios (uso em testes, sem rede)."""
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    model_config = SegformerConfig()
    model_config.num_labels = 1
    model_config.id2label = {0: "coffee"}
    model_config.label2id = {"coffee": 0}
    return SegformerForSemanticSegmentation(model_config)


def _adapt_input_channels(model: Any, in_channels: int) -> None:
    """Expande a projeção de entrada para ``in_channels`` bandas.

    Os canais existentes preservam os pesos pré-treinados; cada canal novo
    recebe a média dos filtros originais, mantendo a escala das ativações.
    """
    projection = model.segformer.stages[0].patch_embeddings.proj
    if projection.in_channels == in_channels:
        model.config.num_channels = in_channels
        return
    replacement = nn.Conv2d(
        in_channels,
        projection.out_channels,
        kernel_size=projection.kernel_size,
        stride=projection.stride,
        padding=projection.padding,
        dilation=projection.dilation,
        groups=projection.groups,
        bias=projection.bias is not None,
        padding_mode=projection.padding_mode,
    )
    with torch.no_grad():
        copied = min(in_channels, projection.in_channels)
        replacement.weight[:, :copied] = projection.weight[:, :copied]
        if in_channels > projection.in_channels:
            mean_filter = projection.weight.mean(dim=1, keepdim=True)
            replacement.weight[:, projection.in_channels :] = mean_filter.repeat(
                1, in_channels - projection.in_channels, 1, 1
            )
        replacement_bias = replacement.bias
        if projection.bias is not None and replacement_bias is not None:
            replacement_bias.copy_(projection.bias)
    model.segformer.stages[0].patch_embeddings.proj = replacement
    model.config.num_channels = in_channels


class SegFormerBinary(nn.Module):
    """SegFormer binário com logits na resolução espacial da entrada."""

    def __init__(self, model: Any, in_channels: int) -> None:
        super().__init__()
        self.model = model
        self.in_channels = in_channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Executa o SegFormer e reamostra os logits para a resolução de entrada."""
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(
                f"Entrada esperada (B, {self.in_channels}, H, W); recebido {tuple(image.shape)}."
            )
        height, width = image.shape[-2:]
        if height % PATCH_MULTIPLE or width % PATCH_MULTIPLE:
            raise ValueError(
                f"SegFormer exige H e W múltiplos de {PATCH_MULTIPLE}; recebido {height}x{width}."
            )
        outputs = self.model(pixel_values=image)
        return nn.functional.interpolate(
            outputs.logits, size=(height, width), mode="bilinear", align_corners=False
        )


def build_segformer(pretrained: bool = True, variant: str | None = None) -> SegFormerBinary:
    """Constrói o SegFormer a partir de config.yaml (uso no estágio 10).

    ``pretrained=False`` evita download de pesos (testes) e usa a configuração
    padrão do Transformers; ``variant`` sobrescreve `model.segformer_variant`.
    """
    config = get_config()
    in_channels = len(config["data"]["bands"])
    variant_name = variant or str(config["model"]["segformer_variant"])
    model = _load_pretrained(variant_name) if pretrained else _build_random()
    _adapt_input_channels(model, in_channels)
    return SegFormerBinary(model, in_channels)
