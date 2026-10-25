"""Testes de src/setup.py (prólogo de runtime e dependências pinadas)."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import numpy as np

from src.setup import (
    canonicalize_distribution_name,
    detect_loaded_mismatches,
    parse_pinned_requirements,
    remove_incompatible_torchaudio,
    requirements_satisfied,
)


def test_canonicalize_distribution_name() -> None:
    """Nomes de distribuição devem ser normalizados conforme o padrão do PyPI."""
    assert canonicalize_distribution_name("Scikit_Learn") == "scikit-learn"
    assert canonicalize_distribution_name("huggingface.hub") == "huggingface-hub"


def test_parse_pinned_requirements_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    """O parser deve extrair apenas pares distribuição==versão."""
    path = tmp_path / "requirements-runtime.txt"
    path.write_text(
        "# comentário\ntorch==2.14.0\n\nscikit-learn==1.9.1  # comentário na linha\n",
        encoding="utf-8",
    )
    assert parse_pinned_requirements(path) == {"torch": "2.14.0", "scikit-learn": "1.9.1"}


def test_requirements_satisfied_compares_installed_versions() -> None:
    """A verificação deve aceitar a versão instalada e rejeitar versões divergentes."""
    assert requirements_satisfied({"numpy": np.__version__})
    assert not requirements_satisfied({"numpy": "0.0.0"})


def test_detect_loaded_mismatches_flags_stale_compiled_extension() -> None:
    """Módulos carregados em versão diferente do pin devem ser sinalizados."""
    pins = {"numpy": "2.5.3", "torch": "2.14.0"}
    loaded = {"numpy": "2.2.0"}
    assert detect_loaded_mismatches(pins, loaded) == {"numpy": ("2.2.0", "2.5.3")}


def test_detect_loaded_mismatches_ignores_unloaded_modules() -> None:
    """Bibliotecas ainda não importadas não exigem reinício do kernel."""
    assert detect_loaded_mismatches({"numpy": "2.5.3"}, {}) == {}


def test_remove_incompatible_torchaudio_noop_when_absent(monkeypatch) -> None:
    """Sem torchaudio instalado, o helper não deve remover nada."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name, *args: None if name == "torchaudio" else real_find_spec(name, *args),
    )
    assert remove_incompatible_torchaudio() is False


def test_remove_incompatible_torchaudio_removes_when_probe_fails(monkeypatch) -> None:
    """Um torchaudio que não importa deve ser removido do ambiente."""
    real_find_spec = importlib.util.find_spec
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        failed_probe = command[2] == "import torchaudio"
        return subprocess.CompletedProcess(command, 1 if failed_probe else 0, "", "")

    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name, *args: object() if name == "torchaudio" else real_find_spec(name, *args),
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    assert remove_incompatible_torchaudio() is True
    assert any("uninstall" in command for command in calls)
