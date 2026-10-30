"""Empacotamento e publicação do dataset e dos modelos no Hugging Face Hub (estágio 16).

Prepara uma versão de divulgação da fonte escolhida: o conjunto de patches
(imagens, máscaras, manifesto com caminhos reescritos e estatísticas de
normalização) em um repositório de dataset (CC BY 4.0) e os pesos por dobra da
U-Net e do SegFormer em repositórios de modelo (MIT), acompanhados de model
cards e de um `model_config.json` com métricas, hiperparâmetros e fingerprints.
A autenticação é feita no notebook com `HF_TOKEN`; o módulo nunca lê nem grava
segredos em disco. A publicação é idempotente: o release é reutilizado quando o
fingerprint (dataset, pesos e configuração) e os arquivos remotos coincidem.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src import evaluation, trainer
from src.config import get_config
from src.data.mask_utils import reference_year
from src.data.patch_generation import (
    load_manifest,
    manifest_path,
    patch_fingerprint_hash,
    resolve_from_root,
    verify_manifest,
)
from src.data.source_decision import chosen_source
from src.data.spatial_split import split_fingerprint_hash
from src.models.segformer import PATCH_MULTIPLE

RELEASE_SCHEMA_VERSION = 1
MODEL_ORDER = ("unet", "segformer")
DATASET_LICENSE = "cc-by-4.0"
MODEL_LICENSE = "mit"
MODEL_LABELS = {"unet": "U-Net", "segformer": "SegFormer"}


@dataclass(frozen=True)
class ReleaseResult:
    """Resultado da publicação (ou reaproveitamento) do release."""

    source_name: str
    repos: dict[str, str]
    commit_urls: dict[str, str]
    n_patches: int
    release_path: Path
    skipped: bool


def release_dir(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Diretório canônico dos metadados de release da fonte (derivado de artifacts/)."""
    root = storage_paths.get("artifacts_release")
    if root is None:
        root = storage_paths["artifacts_metrics"].parent / "release"
    return root / source_name


def release_metadata_path(storage_paths: dict[str, Path], source_name: str) -> Path:
    """Caminho do JSON de auditoria do release publicado."""
    return release_dir(storage_paths, source_name) / "release.json"


def resolve_repo_ids(
    username: str,
    source_name: str,
    dataset_repo: str | None = None,
    unet_repo: str | None = None,
    segformer_repo: str | None = None,
) -> dict[str, str]:
    """Resolve os IDs dos repositórios de dataset e de cada modelo."""
    prefix = f"{username}/tcc-coffee"
    return {
        "dataset": dataset_repo or f"{prefix}-{source_name}",
        "unet": unet_repo or f"{prefix}-unet-{source_name}",
        "segformer": segformer_repo or f"{prefix}-segformer-{source_name}",
    }


def repo_url(repo_id: str, repo_type: str) -> str:
    """URL pública de um repositório do Hugging Face Hub."""
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}"


def huggingface_token() -> str:
    """Lê o token do Hugging Face de variável de ambiente/segredo do runtime."""
    from src.utils import get_secret

    token = get_secret("HF_TOKEN")
    token = token.strip() if token else None
    if not token:
        raise RuntimeError(
            "Publicação no Hugging Face exige HF_TOKEN no ambiente ou nos Secrets "
            "da plataforma (com permissão de escrita). O token nunca é gravado no repositório."
        )
    return token


def authenticate_huggingface() -> tuple[str, str]:
    """Autentica no Hugging Face e retorna o token e o nome da conta."""
    from huggingface_hub import login, whoami

    token = huggingface_token()
    login(token=token, add_to_git_credential=False)
    user = whoami(token=token)
    username = user.get("name") or user.get("user") or user.get("email")
    if not username:
        raise RuntimeError("Não foi possível obter o nome da conta do Hugging Face.")
    return token, str(username)


def _read_json(path: Path) -> dict[str, Any]:
    """Lê um JSON local/canonizado pelo helper de I/O do projeto."""
    from src import io

    return json.loads(io.ensure_local_copy(path).read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    """Hash SHA-256 curto de um arquivo."""
    from src import io

    return hashlib.sha256(io.ensure_local_copy(path).read_bytes()).hexdigest()[:12]


def _fold_count() -> int:
    """Número de dobras da validação espacial configurada."""
    from src.data.spatial_split import fold_count

    return fold_count()


def release_fingerprint(
    storage_paths: dict[str, Path],
    source_name: str,
    repos: dict[str, str],
    private: bool,
) -> dict[str, Any]:
    """Fingerprint que valida a reutilização do release."""
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "source_name": source_name,
        "config_hash": trainer.config_hash(),
        "manifest_fingerprint_hash": patch_fingerprint_hash(storage_paths, source_name),
        "split_fingerprint_hash": split_fingerprint_hash(storage_paths, source_name),
        "weights_sha256": {
            model_name: {
                str(fold): _digest(
                    trainer.weights_path(storage_paths, model_name, source_name, fold)
                )
                for fold in range(_fold_count())
            }
            for model_name in MODEL_ORDER
        },
        "repos": dict(repos),
        "private": bool(private),
    }


def validate_inputs(storage_paths: dict[str, Path], source_name: str) -> dict[str, dict[str, Any]]:
    """Valida manifesto, avaliação e pesos antes do empacotamento."""
    from src import io

    if not io.path_exists(manifest_path(storage_paths, source_name)):
        raise FileNotFoundError(
            f"Manifesto da fonte {source_name} não encontrado; execute o estágio 06."
        )
    from src.data.eda import normalization_stats_is_current, normalization_stats_path

    stats_file = normalization_stats_path(storage_paths, source_name)
    if not io.path_exists(stats_file):
        raise FileNotFoundError(
            f"Estatísticas de normalização não encontradas: {stats_file}. "
            "Execute o estágio 08 antes do release."
        )
    if not normalization_stats_is_current(storage_paths, source_name):
        raise ValueError(
            "Estatísticas de normalização desatualizadas; reexecute o estágio 08 antes do release."
        )
    summaries: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_ORDER:
        if not evaluation.summary_is_current(storage_paths, model_name, source_name):
            raise ValueError(
                f"Resumo do estágio 11 desatualizado para {model_name}; "
                "reexecute a avaliação antes do release."
            )
        summaries[model_name] = evaluation.load_evaluation_summary(
            storage_paths, model_name, source_name
        )
        expected_weights = {
            int(item["fold"]): str(item["weights_sha256"])
            for item in summaries[model_name]["per_fold"]
        }
        for fold in range(_fold_count()):
            if not trainer.fold_training_done(storage_paths, model_name, source_name, fold):
                raise FileNotFoundError(
                    f"Pesos/artefatos ausentes para {model_name}, dobra {fold}; "
                    "execute os estágios 09/10 antes do release."
                )
            actual = _digest(trainer.weights_path(storage_paths, model_name, source_name, fold))
            if expected_weights.get(fold) != actual:
                raise ValueError(
                    f"Pesos de {model_name} (dobra {fold}) divergem da avaliação do estágio 11; "
                    "reexecute a avaliação antes do release."
                )
    return summaries


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Monta uma tabela Markdown sem dependência de tabulate."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _dataset_card(
    source_name: str,
    stats: dict[str, Any],
    manifest: pd.DataFrame,
) -> str:
    """Gera o dataset card (README.md) do repositório de dados."""
    config = get_config()
    folds = (
        manifest["fold"].value_counts().sort_index().to_dict() if "fold" in manifest.columns else {}
    )
    fold_lines = [["fold", "patches"]] + [
        [str(int(fold)), str(int(count))] for fold, count in folds.items()
    ]
    return f"""---
license: {DATASET_LICENSE}
task_categories:
- image-segmentation
tags:
- coffee
- sentinel-2
- semantic-segmentation
- precision-agriculture
pretty_name: TCC Coffee Patches ({source_name})
---

# TCC Coffee Patches — {source_name}

Conjunto de patches de café da Região Geográfica Imediata de Guaxupé (MG),
derivado de Sentinel-2 Level-2A (bandas B2, B3, B4 e B8, 10 m) e da máscara de
referência **{source_name}** ({reference_year(source_name)}).

## Estrutura

```
manifest.parquet                  # metadados e caminhos relativos dos patches
normalization_stats_{source_name}.json
images/<patch_id>.npy             # reflectância por banda (C, H, W)
masks/<patch_id>.npy              # máscara binária de café
README.md
```

`manifest.parquet` contém `patch_id`, `fold`, `bbox`, `coffee_ratio`,
`valid_fraction`, `image_path` e `mask_path` (caminhos relativos a este
repositório). A divisão espacial k-fold está na coluna `fold`; use a dobra de
validação indicada para reproduzir a avaliação.

## Estatísticas

- Região: `{config["aoi"]["region_code"]}`
- Ano da fonte: `{reference_year(source_name)}`
- Bandas: `{", ".join(config["data"]["bands"])}`
- Tamanho do patch: `{config["data"]["patch_size"]}x{config["data"]["patch_size"]}`
- Patches: `{stats["n_patches"]}`
- Patches com café: `{stats["n_with_coffee"]}`
- `coffee_ratio` médio: `{stats["coffee_ratio"]["mean"]:.4f}`
- Forma dos arrays: `{stats["patch_shape"]}`

{_markdown_table(["fold", "patches"], fold_lines[1:])}

## Uso

```python
import numpy as np
import pandas as pd

manifest = pd.read_parquet("manifest.parquet")
record = manifest.iloc[0]
image = np.load(record["image_path"])
mask = np.load(record["mask_path"])
```

Pixels fora do AOI são NaN na imagem; o rótulo de treino correspondente deve ser
ignorado (o pipeline usa `-1`). Consulte `normalization_stats_{source_name}.json`
para as estatísticas por banda.

## Licença

Dados e documentação: **CC BY 4.0**. Código do pipeline: MIT.
"""


def _model_card(
    model_name: str,
    source_name: str,
    summary: dict[str, Any],
    model_config: dict[str, Any],
) -> str:
    """Gera o model card (README.md) de um repositório de pesos."""
    fold_rows = [
        [
            int(item["fold"]),
            int(item["n_patches"]),
            f"{item['metrics']['iou']:.4f}",
            f"{item['metrics']['f1']:.4f}",
            f"{item['metrics']['precision']:.4f}",
            f"{item['metrics']['recall']:.4f}",
        ]
        for item in summary["per_fold"]
    ]
    aggregate = summary["aggregate"]["metrics"]
    fold_mean = summary["fold_mean"]
    fold_std = summary["fold_std"]
    module_name = "unet" if model_name == "unet" else "segformer"
    builder = f"build_{module_name}"
    metrics_table = _markdown_table(
        ["fold", "patches", "IoU", "F1", "Precisão", "Recall"],
        fold_rows,
    )
    return f"""---
license: {MODEL_LICENSE}
library_name: pytorch
tags:
- semantic-segmentation
- coffee
- sentinel-2
- {model_name}
---

# {MODEL_LABELS[model_name]} — café ({source_name})

Pesos do **{MODEL_LABELS[model_name]}** treinados para segmentação binária de
café na fonte de referência **{source_name}** ({reference_year(source_name)}),
com validação espacial k-fold. Cada `fold_i.pt` contém o state dict da dobra
`i` (treinada nas demais dobras).

## Configuração

```json
{json.dumps(model_config, ensure_ascii=False, indent=2)}
```

## Métricas de validação

{metrics_table}

Agregado micro: IoU `{aggregate["iou"]:.4f}`, F1 `{aggregate["f1"]:.4f}`,
Precisão `{aggregate["precision"]:.4f}`, Recall `{aggregate["recall"]:.4f}`.

Média por dobra ± desvio: IoU `{fold_mean["iou"]:.4f} ± {fold_std["iou"]:.4f}`,
F1 `{fold_mean["f1"]:.4f} ± {fold_std["f1"]:.4f}`.

## Uso

O pacote `src` e as instruções completas estão em
[github.com/jotap1101/tcc-lab](https://github.com/jotap1101/tcc-lab).

```python
import torch

from src.models.{module_name} import {builder}

model = {builder}()
state = torch.load("fold_0.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
```

Os arrays de entrada seguem `normalization_stats_{source_name}.json` e a ordem
de bandas do pipeline. Para o SegFormer, a primeira projeção aceita quatro
bandas e a saída é reamostrada para a resolução da entrada.

## Licença

Pesos e código: **MIT**.
"""


def _model_config(
    model_name: str,
    source_name: str,
    summary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Metadados do modelo publicados junto dos pesos."""
    architecture: dict[str, Any]
    if model_name == "unet":
        architecture = {
            "channels": list(config["model"]["unet_channels"]),
            "in_channels": len(config["data"]["bands"]),
        }
    else:
        architecture = {
            "variant": str(config["model"]["segformer_variant"]),
            "in_channels": len(config["data"]["bands"]),
            "patch_multiple": PATCH_MULTIPLE,
        }
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "model": model_name,
        "source_name": source_name,
        "year": reference_year(source_name),
        "region_code": str(config["aoi"]["region_code"]),
        "bands": list(config["data"]["bands"]),
        "patch_size": int(config["data"]["patch_size"]),
        "architecture": architecture,
        "threshold": float(summary["threshold"]),
        "per_fold": summary["per_fold"],
        "aggregate": summary["aggregate"],
        "fold_mean": summary["fold_mean"],
        "fold_std": summary["fold_std"],
        "fingerprint": summary["fingerprint"],
        "training": dict(config["training"]),
        "loss": dict(config["loss"]),
        "license": MODEL_LICENSE,
    }


def _materialize(source: Path, target: Path) -> None:
    """Cria o arquivo de staging (hardlink quando possível, cópia como fallback)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _stage_dataset(
    storage_paths: dict[str, Path],
    source_name: str,
    staging: Path,
) -> tuple[dict[str, Any], int]:
    """Monta o diretório do dataset com patches, manifesto e estatísticas."""
    from src import io

    manifest = load_manifest(storage_paths, source_name)
    stats = verify_manifest(storage_paths, source_name)
    release_manifest = manifest.copy()
    for row in release_manifest.itertuples(index=False):
        patch = str(row.patch_id)
        if Path(patch).name != patch:
            raise ValueError(f"Identificador de patch inválido para release: {patch}")
        image_source = io.ensure_local_copy(resolve_from_root(str(row.image_path), storage_paths))
        mask_source = io.ensure_local_copy(resolve_from_root(str(row.mask_path), storage_paths))
        _materialize(image_source, staging / "images" / f"{patch}.npy")
        _materialize(mask_source, staging / "masks" / f"{patch}.npy")
    release_manifest["image_path"] = [
        f"images/{patch}.npy" for patch in release_manifest["patch_id"].astype(str)
    ]
    release_manifest["mask_path"] = [
        f"masks/{patch}.npy" for patch in release_manifest["patch_id"].astype(str)
    ]
    release_manifest.to_parquet(staging / "manifest.parquet", index=False)
    stats_source = storage_paths["data_processed"] / f"normalization_stats_{source_name}.json"
    _materialize(io.ensure_local_copy(stats_source), staging / stats_source.name)
    (staging / "README.md").write_text(
        _dataset_card(source_name, stats, manifest), encoding="utf-8"
    )
    return stats, len(release_manifest)


def _stage_model(
    storage_paths: dict[str, Path],
    source_name: str,
    model_name: str,
    summary: dict[str, Any],
    staging: Path,
) -> dict[str, Any]:
    """Monta o diretório de um modelo com pesos, README e configuração."""
    from src import io

    for fold in range(_fold_count()):
        source = trainer.weights_path(storage_paths, model_name, source_name, fold)
        _materialize(io.ensure_local_copy(source), staging / f"fold_{fold}.pt")
    model_config = _model_config(model_name, source_name, summary, get_config())
    (staging / "model_config.json").write_text(
        json.dumps(model_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (staging / "README.md").write_text(
        _model_card(model_name, source_name, summary, model_config), encoding="utf-8"
    )
    return model_config


def _remote_is_current(api: Any, repo_id: str, repo_type: str, expected: set[str]) -> bool:
    """Indica se o repositório remoto existe e contém todos os arquivos esperados."""
    try:
        if not api.repo_exists(repo_id=repo_id, repo_type=repo_type):
            return False
        files = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
    except Exception:
        return False
    return expected.issubset(files)


def _release_is_current(
    storage_paths: dict[str, Path],
    source_name: str,
    fingerprint: dict[str, Any],
    repos: dict[str, str],
    api: Any,
) -> bool:
    """Indica se o release local e os repositórios remotos estão vigentes."""
    from src import io

    metadata_file = release_metadata_path(storage_paths, source_name)
    if not io.path_exists(metadata_file):
        return False
    try:
        stored = _read_json(metadata_file)
    except (json.JSONDecodeError, OSError):
        return False
    if stored.get("fingerprint") != fingerprint:
        return False
    manifest = load_manifest(storage_paths, source_name)
    expected_dataset = {"README.md", "manifest.parquet", f"normalization_stats_{source_name}.json"}
    for patch in manifest["patch_id"].astype(str):
        expected_dataset.add(f"images/{patch}.npy")
        expected_dataset.add(f"masks/{patch}.npy")
    expected_models = {
        model_name: {"README.md", "model_config.json"}
        | {f"fold_{fold}.pt" for fold in range(_fold_count())}
        for model_name in MODEL_ORDER
    }
    return _remote_is_current(api, repos["dataset"], "dataset", expected_dataset) and all(
        _remote_is_current(api, repos[model_name], "model", expected_models[model_name])
        for model_name in MODEL_ORDER
    )


def run_release(
    storage_paths: dict[str, Path],
    source_name: str | None = None,
    *,
    dataset_repo: str,
    unet_repo: str,
    segformer_repo: str,
    api: Any | None = None,
    token: str | None = None,
    private: bool = False,
    force: bool = False,
) -> ReleaseResult:
    """Empacota e publica o dataset e os modelos no Hugging Face Hub (idempotente)."""
    from src import io

    source = source_name or chosen_source()
    summaries = validate_inputs(storage_paths, source)
    repos = {
        "dataset": dataset_repo,
        "unet": unet_repo,
        "segformer": segformer_repo,
    }
    fingerprint = release_fingerprint(storage_paths, source, repos, private)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token or huggingface_token())
    if not force and _release_is_current(storage_paths, source, fingerprint, repos, api):
        metadata = _read_json(release_metadata_path(storage_paths, source))
        print("Release já vigente (dataset e modelos reutilizados).")
        return ReleaseResult(
            source_name=source,
            repos=repos,
            commit_urls=dict(metadata.get("commit_urls", {})),
            n_patches=int(metadata["n_patches"]),
            release_path=release_metadata_path(storage_paths, source),
            skipped=True,
        )

    with tempfile.TemporaryDirectory(prefix="tcc-release-") as temp:
        root = Path(temp)
        dataset_staging = root / "dataset"
        dataset_staging.mkdir()
        stats, n_patches = _stage_dataset(storage_paths, source, dataset_staging)
        model_stagings = {model_name: root / model_name for model_name in MODEL_ORDER}
        for model_name, staging in model_stagings.items():
            staging.mkdir()
            _stage_model(storage_paths, source, model_name, summaries[model_name], staging)

        api.create_repo(
            repo_id=repos["dataset"], repo_type="dataset", private=private, exist_ok=True
        )
        for model_name in MODEL_ORDER:
            api.create_repo(
                repo_id=repos[model_name],
                repo_type="model",
                private=private,
                exist_ok=True,
            )
        dataset_commit = api.upload_folder(
            repo_id=repos["dataset"],
            repo_type="dataset",
            folder_path=dataset_staging,
            commit_message=f"Release {source}: patches e manifesto",
        )
        model_commits = {
            model_name: api.upload_folder(
                repo_id=repos[model_name],
                repo_type="model",
                folder_path=model_stagings[model_name],
                commit_message=f"Release {source}: pesos {MODEL_LABELS[model_name]}",
            )
            for model_name in MODEL_ORDER
        }

    commit_urls = {
        "dataset": str(dataset_commit.commit_url),
        **{model_name: str(commit.commit_url) for model_name, commit in model_commits.items()},
    }
    metadata = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "source_name": source,
        "year": reference_year(source),
        "n_patches": n_patches,
        "repos": repos,
        "repo_urls": {
            "dataset": repo_url(repos["dataset"], "dataset"),
            **{model_name: repo_url(repos[model_name], "model") for model_name in MODEL_ORDER},
        },
        "commit_urls": commit_urls,
        "dataset_stats": stats,
        "models": {
            model_name: {
                "aggregate": summaries[model_name]["aggregate"],
                "fold_mean": summaries[model_name]["fold_mean"],
                "fold_std": summaries[model_name]["fold_std"],
            }
            for model_name in MODEL_ORDER
        },
        "fingerprint": fingerprint,
    }
    io.persist_json(release_metadata_path(storage_paths, source), metadata)
    print(f"Release publicado: {n_patches} patches | repos: {repos}")
    return ReleaseResult(
        source_name=source,
        repos=repos,
        commit_urls=commit_urls,
        n_patches=n_patches,
        release_path=release_metadata_path(storage_paths, source),
        skipped=False,
    )
