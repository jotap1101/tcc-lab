"""Utilitários compartilhados pelos scripts locais que chamam o Kaggle CLI.

Mantém a localização do executável, a execução de subcomandos e a leitura do
usuário autenticado em um único lugar, sem dependências de terceiros.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_kaggle_cli() -> str:
    """Localiza o CLI do Kaggle no ambiente virtual atual ou no PATH."""
    candidates = [Path(sys.executable).with_name("kaggle")]
    found = shutil.which("kaggle")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(
        "Kaggle CLI não encontrado. Instale no ambiente virtual com "
        "`.venv/bin/pip install kaggle`."
    )


def run_cli(cli: str, args: list[str], echo: bool = True) -> tuple[int, str]:
    """Executa um subcomando do CLI e devolve (código de saída, saída combinada)."""
    result = subprocess.run([cli, *args], text=True, capture_output=True)
    output = f"{result.stdout}\n{result.stderr}".strip()
    if echo and output:
        print(output)
    return result.returncode, output


def authenticated_username(cli: str) -> str | None:
    """Lê o usuário autenticado a partir de `kaggle config view`."""
    _, output = run_cli(cli, ["config", "view"], echo=False)
    for line in output.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    return None
