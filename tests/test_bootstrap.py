"""Testes de src/bootstrap.py (entrega do código-fonte ao runtime)."""

import pathlib
import shutil
import tarfile

import pytest

from src import bootstrap


def _make_tarball(tmp_path: pathlib.Path, repo_name: str) -> pathlib.Path:
    """Cria um tarball com a mesma estrutura de um arquivo do GitHub."""
    repo_dir = tmp_path / "source" / repo_name
    (repo_dir / "src").mkdir(parents=True)
    (repo_dir / "src" / "__init__.py").write_text("")
    (repo_dir / "requirements-runtime.txt").write_text("numpy\n")
    (repo_dir / "data" / "external").mkdir(parents=True)
    (repo_dir / "data" / "external" / "reference.txt").write_text("reference")

    archive = tmp_path / "repo.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(repo_dir, arcname=repo_name)
    return archive


def _patch_download(monkeypatch: pytest.MonkeyPatch, archive: pathlib.Path) -> None:
    """Substitui o download pela cópia de um tarball local."""

    def _urlretrieve(url: str, filename: str) -> None:
        shutil.copy(archive, filename)

    monkeypatch.setattr(bootstrap.urllib.request, "urlretrieve", _urlretrieve)


def test_download_and_extract_resolves_repo_dir_by_content(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raiz extraída é descoberta pelo conteúdo, sem depender do nome do repo."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _patch_download(monkeypatch, _make_tarball(tmp_path, "tcc-lab-main"))

    bootstrap._download_and_extract(workspace)

    assert (workspace / "src" / "__init__.py").is_file()
    assert (workspace / "requirements-runtime.txt").is_file()
    assert (workspace / "data" / "external" / "reference.txt").is_file()


def test_download_and_extract_preserves_workspace_on_invalid_tarball(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Um tarball incompleto não deve apagar os arquivos já presentes no workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    requirements = workspace / "requirements-runtime.txt"
    requirements.write_text("keep")

    repo_dir = tmp_path / "source" / "tcc-lab-main"
    (repo_dir / "src").mkdir(parents=True)
    archive = tmp_path / "repo.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(repo_dir, arcname="tcc-lab-main")
    _patch_download(monkeypatch, archive)

    with pytest.raises(FileNotFoundError):
        bootstrap._download_and_extract(workspace)

    assert requirements.read_text() == "keep"


def test_resolve_repo_dir_rejects_ambiguous_layout(tmp_path: pathlib.Path) -> None:
    """Mais de um diretório extraído indica estrutura inesperada do tarball."""
    (tmp_path / "repo-a").mkdir()
    (tmp_path / "repo-b").mkdir()

    with pytest.raises(RuntimeError):
        bootstrap._resolve_repo_dir(tmp_path)
