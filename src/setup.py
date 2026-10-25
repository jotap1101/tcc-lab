"""Inicialização comum do runtime para todos os notebooks (prólogo padronizado).

Centraliza o bloco repetido entre os notebooks 00-08: detecção de plataforma,
resolução do armazenamento canônico (MyDrive/tcc/) e carregamento da
configuração, expondo um único ``initialize()`` que retorna um ``RuntimeContext``.
"""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import io
from src.config import get_config

# Bibliotecas com extensões compiladas: uma versão já carregada no processo não
# pode ser substituída sem reiniciar o kernel (a ABI fica inconsistente).
ABI_SENSITIVE_MODULES: dict[str, str] = {
    "numpy": "numpy",
    "scipy": "scipy",
    "torch": "torch",
    "torchvision": "torchvision",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "rasterio": "rasterio",
    "geopandas": "geopandas",
}


@dataclass(frozen=True)
class RuntimeContext:
    """Contexto de runtime resolvido para a execução de um notebook."""

    platform: str
    workspace: Path
    storage_paths: dict[str, Path]
    config: dict[str, Any]

    @property
    def seed(self) -> int:
        """Semente global de reprodutibilidade, definida em config.yaml."""
        return int(self.config["reproducibility"]["seed"])


def initialize(workspace: Path | None = None) -> RuntimeContext:
    """Resolve plataforma, armazenamento canônico e configuração (idempotente)."""
    platform = io.detect_platform()
    storage_paths = io.resolve_storage_paths()
    config = get_config()
    return RuntimeContext(
        platform=platform,
        workspace=Path.cwd() if workspace is None else workspace,
        storage_paths=storage_paths,
        config=config,
    )


def canonicalize_distribution_name(name: str) -> str:
    """Normaliza o nome de uma distribuição conforme o padrão do PyPI (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pinned_requirements(requirements_path: Path) -> dict[str, str]:
    """Lê pares ``distribuição==versão`` do arquivo de requisitos.

    Linhas vazias, comentários e restrições sem versão exata são ignorados.
    """
    pins: dict[str, str] = {}
    for raw_line in Path(requirements_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[canonicalize_distribution_name(name.strip())] = version.strip()
    return pins


def _installed_version(distribution: str) -> str | None:
    """Retorna a versão instalada no ambiente (ou None se ausente)."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _loaded_versions() -> dict[str, str]:
    """Mapeia distribuições sensíveis à ABI para a versão já carregada em memória."""
    versions: dict[str, str] = {}
    for distribution, module_name in ABI_SENSITIVE_MODULES.items():
        module = sys.modules.get(module_name)
        version = getattr(module, "__version__", None) if module is not None else None
        if version:
            versions[distribution] = str(version)
    return versions


def requirements_satisfied(pins: Mapping[str, str]) -> bool:
    """Indica se todas as distribuições pinadas estão instaladas nas versões exatas."""
    return all(_installed_version(name) == version for name, version in pins.items())


def detect_loaded_mismatches(
    pins: Mapping[str, str],
    loaded_versions: Mapping[str, str],
) -> dict[str, tuple[str, str]]:
    """Detecta extensões compiladas já carregadas em versão divergente do pin.

    Retorna ``{distribuição: (versão_carregada, versão_pinada)}``. Uma divergência
    exige reiniciar o kernel: os arquivos novos estão em disco, mas o processo
    continua usando o binário antigo (ex.: ``AttributeError`` em
    ``numpy._core._multiarray_umath``).
    """
    mismatches: dict[str, tuple[str, str]] = {}
    for distribution in ABI_SENSITIVE_MODULES:
        pinned = pins.get(canonicalize_distribution_name(distribution))
        loaded = loaded_versions.get(distribution)
        if pinned is not None and loaded is not None and loaded != pinned:
            mismatches[distribution] = (loaded, pinned)
    return mismatches


def _restart_kernel() -> None:
    """Solicita o reinício do kernel para recarregar as extensões compiladas."""
    try:
        from IPython import get_ipython
    except Exception:
        get_ipython = None  # type: ignore[assignment]

    ipython = get_ipython() if get_ipython is not None else None
    kernel = getattr(ipython, "kernel", None)
    if kernel is None:
        raise RuntimeError(
            "Reinicie a sessão manualmente para recarregar as dependências instaladas."
        )
    kernel.do_shutdown(restart=True)


def install_runtime_requirements(requirements_path: Path) -> None:
    """Garante as versões pinadas do runtime e mantém a ABI do kernel consistente.

    Instalar uma versão diferente de uma biblioteca com extensão compilada
    (numpy, scipy, torch, ...) no mesmo processo deixa os módulos já importados
    inconsistentes com os arquivos novos em disco. Nesse caso o kernel é
    reiniciado automaticamente para que a próxima execução importe o ambiente
    consistente; a segunda execução encontra tudo instalado e segue sem reinício.
    """
    requirements_path = Path(requirements_path)
    pins = parse_pinned_requirements(requirements_path)

    pending = {
        name: version for name, version in pins.items() if _installed_version(name) != version
    }
    if pending:
        print("Instalando dependências pinadas de:", requirements_path, flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(requirements_path),
            ],
            check=True,
        )

    mismatches = detect_loaded_mismatches(pins, _loaded_versions())
    if mismatches:
        details = ", ".join(
            f"{name} (carregada {loaded} -> pinada {pinned})"
            for name, (loaded, pinned) in mismatches.items()
        )
        print(
            "Kernel reiniciado: módulos já carregados divergem do ambiente instalado "
            f"[{details}]. Execute o notebook novamente a partir da primeira célula.",
            flush=True,
        )
        _restart_kernel()
        return

    if not pending:
        print("Dependências já estão nas versões pinadas:", requirements_path)
