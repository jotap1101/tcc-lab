"""Testes de src/utils.py (sementes fixas, flags e apoio aos notebooks)."""

import os

import pytest

from src.config import get_config
from src.utils import (
    check_dependencies,
    log_environment,
    print_summary,
    set_all_seeds,
    set_deterministic_flags,
    setup_reproducibility,
)


def test_set_all_seeds_and_flags_run() -> None:
    """Sementes e flags determinísticas devem executar sem erro."""
    set_all_seeds(42)
    set_deterministic_flags()


def test_set_deterministic_flags_sets_cublas_workspace(monkeypatch) -> None:
    """O CUDA determinístico exige CUBLAS_WORKSPACE_CONFIG definido."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    set_deterministic_flags()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] in {":4096:8", ":16:8"}


def test_setup_reproducibility_prints_seed(capsys) -> None:
    """setup_reproducibility deve fixar sementes e imprimir a semente usada."""
    setup_reproducibility(get_config())
    captured = capsys.readouterr()
    assert "Seed fixada: 42" in captured.out


def test_log_environment_prints_versions(capsys) -> None:
    """log_environment deve registrar versões dos pacotes sem falhar."""
    log_environment(packages=("numpy",))
    assert "numpy:" in capsys.readouterr().out


def test_log_environment_handles_missing_package(capsys) -> None:
    """Pacote ausente deve ser reportado como 'não instalado'."""
    log_environment(packages=("pacote_inexistente_tcc",))
    assert "não instalado" in capsys.readouterr().out


def test_check_dependencies_raises_on_missing_path(tmp_path) -> None:
    """check_dependencies deve falhar quando uma dependência está ausente."""
    with pytest.raises(FileNotFoundError):
        check_dependencies({"ausente": tmp_path / "nao-existe.tif"})


def test_check_dependencies_ok_when_present(tmp_path, capsys) -> None:
    """check_dependencies deve imprimir 'Disponível' quando o caminho existe."""
    existing = tmp_path / "existe.tif"
    existing.write_bytes(b"x")
    check_dependencies({"entrada": existing})
    assert "Disponível: entrada:" in capsys.readouterr().out


def test_print_summary_prints_items_and_conclusion(capsys) -> None:
    """print_summary deve exibir os itens e a mensagem de conclusão do estágio."""
    print_summary({"chave": "valor"}, "99")
    captured = capsys.readouterr()
    assert "chave: valor" in captured.out
    assert "Estágio 99 concluído." in captured.out
