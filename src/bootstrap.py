"""Bootstrap do workspace, executado ANTES do `import src` (somente stdlib).

A primeira célula do notebook 00 baixa e executa este módulo: ele obtém o
repositório público, extrai `src/`, `data/external/` e `requirements-runtime.txt`
para o workspace e adiciona o workspace ao `sys.path`. Sem dependências externas,
o que resolve o ciclo "precisa de src para baixar src".
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tarfile
import urllib.request

REPO_URL = "https://github.com/jotap1101/tcc-lab"
WORKSPACE_COLAB = pathlib.Path("/content")
WORKSPACE_KAGGLE = pathlib.Path("/kaggle/working")
DRIVE_MOUNT_POINT = pathlib.Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT_POINT / "MyDrive"


def detect_platform() -> str:
    """Identifica a plataforma de execução: 'colab', 'kaggle' ou 'local'.

    A variável do Kaggle é verificada ANTES do google.colab, pois o pacote
    google-colab também está instalado em kernels do Kaggle.
    """
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401

        return "colab"
    except Exception:
        return "local"


def workspace_root() -> pathlib.Path:
    """Retorna a raiz do workspace de acordo com a plataforma."""
    platform = detect_platform()
    if platform == "colab":
        return WORKSPACE_COLAB
    if platform == "kaggle":
        return WORKSPACE_KAGGLE
    return pathlib.Path.cwd()


def _resolve_repo_dir(extract_dir: pathlib.Path) -> pathlib.Path:
    """Retorna o diretório raiz extraído do tarball, independente do nome do repo.

    O GitHub nomeia a raiz do arquivo como `<repositorio>-main`; derivar o
    caminho do conteúdo extraído evita quebrar o bootstrap quando o repositório
    é renomeado.
    """
    extracted = sorted(path for path in extract_dir.iterdir() if path.is_dir())
    if len(extracted) != 1:
        raise RuntimeError(
            "Estrutura inesperada no tarball do repositório: "
            f"{len(extracted)} diretórios encontrados em {extract_dir}"
        )
    return extracted[0]


def _validate_repo_dir(repo_dir: pathlib.Path) -> None:
    """Confere os itens obrigatórios antes de substituir o workspace."""
    required = ("src", "requirements-runtime.txt")
    missing = [rel for rel in required if not (repo_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Itens obrigatórios ausentes no tarball do repositório: {missing}")


def _download_and_extract(workspace: pathlib.Path) -> None:
    """Baixa o tarball do repo e extrai src/, data/external/ e requirements-runtime.txt."""
    archive = workspace / "_repo_bootstrap.tar.gz"
    extract_dir = workspace / "_repo_bootstrap"
    try:
        urllib.request.urlretrieve(f"{REPO_URL}/archive/refs/heads/main.tar.gz", archive)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(extract_dir)
        repo_dir = _resolve_repo_dir(extract_dir)
        _validate_repo_dir(repo_dir)
        for rel in ("src", "data", "requirements-runtime.txt"):
            target = workspace / rel
            if target.is_dir():
                shutil.rmtree(target)
            elif target.is_file():
                target.unlink(missing_ok=True)
        for rel in ("src", "requirements-runtime.txt"):
            source = repo_dir / rel
            if source.is_dir():
                shutil.copytree(source, workspace / rel)
            elif source.is_file():
                shutil.copy(source, workspace / rel)
        external = repo_dir / "data" / "external"
        if external.is_dir():
            shutil.copytree(external, workspace / "data" / "external")
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)


def _seed_drive_mirror(workspace: pathlib.Path) -> None:
    """Cria/atualiza o espelho MyDrive/tcc/repo (idempotente) no Colab/local."""
    if detect_platform() == "kaggle":
        return
    from google.colab import drive  # noqa: PLC0415

    if not DRIVE_ROOT.exists():
        drive.mount(str(DRIVE_MOUNT_POINT))
    mirror = DRIVE_ROOT / "tcc" / "repo"

    def _refresh(source: pathlib.Path, target: pathlib.Path) -> None:
        # Preserva o espelho existente quando a origem não foi entregue ao workspace.
        if not source.exists():
            return
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)

    _refresh(workspace / "src", mirror / "src")
    _refresh(workspace / "data" / "external", mirror / "data" / "external")


def _purge_src_modules() -> None:
    """Remove os módulos src.* em cache para forçar a versão recém-entregue.

    Reexecuções no mesmo kernel podem manter módulos antigos importados; a
    remoção de `sys.modules` garante que a próxima importação use o código
    recém-sincronizado pelo bootstrap.
    """
    for module in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[module]


def bootstrap_workspace(workspace: pathlib.Path | None = None) -> pathlib.Path:
    """Executa o bootstrap completo e retorna o caminho do workspace."""
    if workspace is None:
        workspace = workspace_root()
    _download_and_extract(workspace)
    _purge_src_modules()
    _seed_drive_mirror(workspace)
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    return workspace
