"""Dataset de patches e carregadores para os estágios 09/10 (treino).

Constroi os `Dataset`/`DataLoader` a partir do manifesto (estágio 06/07) e das
estatísticas de normalização (estágio 08). Cada patch de imagem é normalizado
com a média/desvio por banda e tem os NaN imputados com a média da banda antes
da padronização. Pixels inválidos (NaN em alguma banda, isto é, fora do AOI)
recebem rótulo -1 na máscara e são ignorados pela perda e pelas métricas. A
divisão por dobra é espacial (`fold` do manifesto): a dobra de validação fica de
fora do treino. A carga em lote é determinística (semente por dobra no shuffle)
e prepara cópias locais no Kaggle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentations import SegmentAugmentation
from src.data.eda import normalization_stats_path
from src.data.patch_generation import (
    load_manifest,
    manifest_has_folds,
    require_manifest,
    resolve_from_root,
)


def load_normalization_stats(storage_paths: dict[str, Path], source_name: str) -> dict[str, Any]:
    """Carrega as estatísticas de normalização da fonte do estágio 08 (falha se ausente)."""
    from src import io

    stats_path = normalization_stats_path(storage_paths, source_name)
    if not io.path_exists(stats_path):
        raise FileNotFoundError(
            f"Estatísticas de normalização da fonte {source_name} (estágio 08) "
            f"não encontradas: {stats_path}"
        )
    local_stats = io.ensure_local_copy(stats_path)
    return json.loads(local_stats.read_text(encoding="utf-8"))


def normalize_patch(image: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Normaliza um patch por banda: imputa NaN com a média e padroniza (z-score).

    Bandas sem estatística finita são zeradas (não há informação para
    normalizar). Retorna um array float32 na mesma forma (C, H, W).
    """
    image = image.astype(np.float32, copy=True)
    for index, band in enumerate(stats["bands"]):
        values = image[index]
        meta = stats["per_band"][band]
        mean = meta.get("mean")
        std = meta.get("std")
        if mean is None or std is None or std == 0.0:
            image[index] = 0.0
            continue
        finite = np.isfinite(values)
        imputed = np.where(finite, values, mean)
        image[index] = (imputed - mean) / std
    return image


def fold_indices(manifest: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Índices posicionais de treino (fold != dobra) e validação (fold == dobra).

    Divisão espacial determinística a partir da coluna `fold` do manifesto.
    """
    folds = manifest["fold"].astype(int).to_numpy()
    train_idx = np.flatnonzero(folds != fold)
    val_idx = np.flatnonzero(folds == fold)
    if val_idx.size == 0:
        raise ValueError(f"Dobra {fold} sem patches no manifesto; execute o estágio 07.")
    return train_idx, val_idx


def ensure_patches_local(storage_paths: dict[str, Path], manifest: pd.DataFrame) -> int:
    """Garante cópias locais de todos os patches (no-op no Colab/local).

    No Kaggle os patches são baixados uma única vez para o cache local antes do
    treino, evitando requisições à Drive API durante o carregamento por lote.
    """
    from src import io

    if io.detect_platform() != "kaggle":
        return 0
    paths = sorted(
        {
            resolve_from_root(str(value), storage_paths)
            for value in pd.concat([manifest["image_path"], manifest["mask_path"]])
        }
    )
    for path in paths:
        io.ensure_local_copy(path)
    return len(paths)


class PatchDataset(Dataset):
    """Dataset de patches (imagem normalizada + máscara binária) a partir do manifesto."""

    def __init__(
        self,
        records: pd.DataFrame,
        storage_paths: dict[str, Path],
        stats: dict[str, Any],
        transform: SegmentAugmentation | None = None,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.storage_paths = storage_paths
        self.stats = stats
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def _ensure_local(self, relative_path: str) -> Path:
        """Resolve um path relativo do manifesto para uma cópia local disponível."""
        from src import io

        return io.ensure_local_copy(resolve_from_root(relative_path, self.storage_paths))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records.iloc[index]
        image = np.load(self._ensure_local(str(record["image_path"])))
        mask = np.load(self._ensure_local(str(record["mask_path"])))
        # Pixels sem observação válida (NaN) são marcados com rótulo -1 e ignorados.
        valid = np.isfinite(image).all(axis=0)
        image = normalize_patch(image, self.stats)
        mask = np.where(valid, (mask > 0).astype(np.float32), -1.0).astype(np.float32)
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        return image_tensor, mask_tensor


def build_loaders(
    storage_paths: dict[str, Path],
    source_name: str,
    fold: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    transform: SegmentAugmentation | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Constroi os DataLoaders de treino e validação de uma dobra da fonte (determinístico).

    O shuffle do treino usa um gerador próprio fixado com a semente da dobra;
    a validação percorre os patches sem embaralhar.
    """
    require_manifest(storage_paths, source_name)
    manifest = load_manifest(storage_paths, source_name)
    if not manifest_has_folds(manifest):
        raise ValueError("Coluna fold ausente/incompleta; execute o estágio 07 antes.")
    stats = load_normalization_stats(storage_paths, source_name)
    train_idx, val_idx = fold_indices(manifest, fold)
    train_dataset = PatchDataset(
        manifest.iloc[train_idx], storage_paths, stats, transform=transform
    )
    val_dataset = PatchDataset(manifest.iloc[val_idx], storage_paths, stats)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
