"""Utilidades de reprodutibilidade, autenticação e apoio aos notebooks."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_secret(name: str) -> str | None:
    """Lê um segredo de variável de ambiente ou dos Secrets do Kaggle."""
    value = os.getenv(name)
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def set_all_seeds(seed: int) -> None:
    """Fixa as sementes de python, numpy e torch (cpu e cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic_flags() -> None:
    """Habilita flags determinísticas do PyTorch (quando suportadas)."""
    # Exigido pelo PyTorch para operações cuBLAS determinísticas no CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def setup_reproducibility(config: dict[str, Any]) -> None:
    """Fixa sementes e flags determinísticas a partir da configuração."""
    seed = int(config["reproducibility"]["seed"])
    set_all_seeds(seed)
    set_deterministic_flags()
    print(f"Seed fixada: {seed}")


def log_environment(
    packages: tuple[str, ...] = (
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "earthengine-api",
    ),
) -> None:
    """Registra as versões dos pacotes principais do runtime."""
    import importlib.metadata

    for pkg in packages:
        try:
            print(f"{pkg}: {importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{pkg}: não instalado")


def check_dependencies(dependencies: dict[str, Path]) -> None:
    """Verifica a existência de dependências de estágios anteriores (falha se ausente)."""
    from src import io

    for name, path in dependencies.items():
        if not io.path_exists(path):
            raise FileNotFoundError(f"Dependência não encontrada: {path}")
        print(f"Disponível: {name}: {path}")


def print_summary(summary: dict[str, Any], stage: str) -> None:
    """Imprime o resumo da etapa e a mensagem de conclusão."""
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Estágio {stage} concluído.")


def authenticate_gee() -> None:
    """Autentica e inicializa o Earth Engine com a conta principal.

    O projeto Cloud do GEE vem de GEE_PROJECT (env) ou de config.yaml
    (gee.project). Se GEE_CREDENTIALS estiver definida (Secret no Kaggle),
    escreve a credencial em ~/.config/earthengine/credentials e inicializa de
    forma headless; caso contrário, executa o fluxo interativo ee.Authenticate().
    """
    import ee

    from src.config import get_config

    project = get_secret("GEE_PROJECT") or get_config().get("gee", {}).get("project")
    if not project:
        raise RuntimeError(
            "Earth Engine exige um projeto Cloud. Defina GEE_PROJECT (env) ou "
            "gee.project em src/config.yaml (veja o ID no Earth Engine Code Editor)."
        )

    credentials_env = get_secret("GEE_CREDENTIALS")
    if credentials_env:
        config_dir = Path.home() / ".config" / "earthengine"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "credentials").write_text(credentials_env, encoding="utf-8")
        ee.Initialize(project=project)
        return
    ee.Authenticate()
    ee.Initialize(project=project)
