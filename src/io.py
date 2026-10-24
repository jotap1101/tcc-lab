"""Abstração de plataforma e de armazenamento no Google Drive (raiz tcc/).

Contrato único usado pelos notebooks: detect_platform(), mount_drive(),
ensure_storage_root() e resolve_storage_paths(). A diferença entre Colab
(montagem nativa) e Kaggle (Drive API + cache local) fica isolada aqui.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from src import bootstrap
from src.config import get_config

COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")
KAGGLE_DRIVE_CACHE = Path("/kaggle/working/drive")


def detect_platform() -> str:
    """Identifica a plataforma de execução: 'colab', 'kaggle' ou 'local'.

    A implementação é única em `src.bootstrap` (executada antes do import de
    `src`); esta função delega para ela, mantendo o ponto de acesso usado pelos
    notebooks e pelos módulos de `src`.
    """
    return bootstrap.detect_platform()


class DriveClient:
    """Cliente da Google Drive API — usado no Kaggle, onde não há mount nativo.

    A credencial é um JSON de OAuth (client_id, client_secret, refresh_token)
    autenticado com a conta principal, lido de GDRIVE_TOKEN ou GDRIVE_TOKEN_FILE.
    """

    def __init__(self) -> None:
        self._service: Any = None

    def _build_service(self) -> Any:
        """Constrói (lazy) o serviço da Drive API com a credencial OAuth do ambiente."""
        if self._service is None:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            from src.utils import get_secret

            token = get_secret("GDRIVE_TOKEN")
            token_file = get_secret("GDRIVE_TOKEN_FILE")
            if token is None and token_file:
                token = Path(token_file).read_text(encoding="utf-8")
            if not token:
                raise RuntimeError(
                    "Acesso ao Drive no Kaggle exige a credencial OAuth da conta principal "
                    "em GDRIVE_TOKEN (ou GDRIVE_TOKEN_FILE). Se o segredo já existe no "
                    "painel Add-ons > Secrets, confirme que ele está HABILITADO para este "
                    "notebook (e reinicie o kernel); ao reimportar o notebook, os secrets "
                    "vinculados ao kernel anterior não acompanham. Alternativa: persista o "
                    "token em MyDrive/tcc/secrets/ na primeira autenticação."
                )
            if token_file or token.lstrip().startswith("{"):
                creds = Credentials.from_authorized_user_info(json.loads(token))
            else:
                creds = Credentials(token=token)
            self._service = build("drive", "v3", credentials=creds)
        return self._service

    def _folder_id(self, name: str, parent: str) -> str | None:
        """Retorna o ID de uma pasta filha, ou None se ela não existir."""
        res = (
            self._build_service()
            .files()
            .list(
                q=(
                    f"name='{name}' and '{parent}' in parents and mimeType="
                    "'application/vnd.google-apps.folder' and trashed=false"
                ),
                fields="files(id)",
            )
            .execute()
        )
        files = res.get("files", [])
        return files[0]["id"] if files else None

    def ensure_folder(self, remote_path: str) -> str:
        """Cria recursivamente uma pasta (separada por '/') no Drive, se ausente.

        Retorna o ID da pasta folha (ou 'root' quando o caminho é vazio), de modo
        que o chamador possa reutilizá-lo como pai de um upload.
        """
        service = self._build_service()
        parent = "root"
        for part in remote_path.split("/"):
            if not part:
                continue
            folder_id = self._folder_id(part, parent)
            if folder_id is not None:
                parent = folder_id
                continue
            metadata = {
                "name": part,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent],
            }
            created = service.files().create(body=metadata, fields="id").execute()
            parent = created["id"]
        return parent

    def upload(self, local_path: Path, remote_path: str) -> None:
        """Faz upload de um arquivo local para um caminho remoto no Drive.

        Cria recursivamente as pastas pai ausentes (ex.: models/<modelo>/<fonte>/
        no treino) e substitui o conteúdo quando o arquivo já existe, evitando
        duplicatas homônimas em reexecuções. Usa upload resumível em blocos
        (MediaFileUpload) para arquivos grandes, como o composite normalizado do
        estágio 05 — mais robusto que um upload simples em uma única requisição.
        """
        from googleapiclient.http import DEFAULT_CHUNK_SIZE, MediaFileUpload

        service = self._build_service()
        parts = [part for part in remote_path.split("/") if part]
        if not parts:
            raise ValueError("Caminho remoto vazio para upload no Drive.")
        parent = self.ensure_folder("/".join(parts[:-1]))
        media = MediaFileUpload(str(local_path), resumable=True, chunksize=DEFAULT_CHUNK_SIZE)
        existing = self._file_id(remote_path)
        if existing is not None:
            service.files().update(fileId=existing, media_body=media).execute()
            return
        metadata = {"name": parts[-1], "parents": [parent]}
        service.files().create(body=metadata, media_body=media, fields="id").execute()

    def download(self, remote_path: str, local_path: Path) -> None:
        """Faz download de um arquivo remoto do Drive para um caminho local."""
        service = self._build_service()
        file_id = self._file_id(remote_path)
        if file_id is None:
            raise FileNotFoundError(f"Arquivo não encontrado no Drive: {remote_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        media = service.files().get_media(fileId=file_id).execute()
        local_path.write_bytes(media)

    def _file_id(self, remote_path: str) -> str | None:
        """Retorna o ID de um arquivo no caminho remoto, ou None se não existir."""
        parts = [p for p in remote_path.split("/") if p]
        if not parts:
            return None
        parent = "root"
        for part in parts[:-1]:
            folder_id = self._folder_id(part, parent)
            if folder_id is None:
                return None
            parent = folder_id
        res = (
            self._build_service()
            .files()
            .list(
                q=f"name='{parts[-1]}' and '{parent}' in parents and trashed=false",
                fields="files(id)",
            )
            .execute()
        )
        files = res.get("files", [])
        return files[0]["id"] if files else None

    def exists(self, remote_path: str) -> bool:
        """Indica se um arquivo remoto existe no caminho exato (pasta + nome)."""
        return self._file_id(remote_path) is not None

    def size(self, remote_path: str) -> int | None:
        """Retorna o tamanho em bytes de um arquivo remoto, ou None se ausente."""
        file_id = self._file_id(remote_path)
        if file_id is None:
            return None
        res = self._build_service().files().get(fileId=file_id, fields="size").execute()
        size = res.get("size")
        return int(size) if size is not None else None

    def move(self, remote_src: str, remote_dst: str) -> None:
        """Move um arquivo remoto para outro caminho no Drive (muda a pasta pai)."""
        service = self._build_service()
        src_parts = [p for p in remote_src.split("/") if p]
        src_parent = "root"
        for part in src_parts[:-1]:
            src_parent = self._folder_id(part, src_parent) or src_parent
        file_id = self._file_id(remote_src)
        if file_id is None:
            raise FileNotFoundError(f"Arquivo não encontrado no Drive: {remote_src}")
        dst_parts = [p for p in remote_dst.split("/") if p]
        if dst_parts[:-1]:
            self.ensure_folder("/".join(dst_parts[:-1]))
        dst_parent = "root"
        for part in dst_parts[:-1]:
            dst_parent = self._folder_id(part, dst_parent) or dst_parent
        service.files().update(
            fileId=file_id,
            addParents=dst_parent,
            removeParents=src_parent,
            fields="id",
        ).execute()

    def upload_tree(self, local_dir: Path, remote_prefix: str) -> None:
        """Sobe recursivamente um diretório local para um prefixo remoto no Drive."""
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            parent = "/".join(rel.split("/")[:-1])
            if parent:
                self.ensure_folder(f"{remote_prefix}/{parent}")
            self.upload(path, f"{remote_prefix}/{rel}")


def get_drive_client() -> DriveClient:
    """Retorna o cliente do Drive (Kaggle) para operações remotas."""
    return DriveClient()


def path_exists(path: Path) -> bool:
    """Indica se um caminho existe no Drive canônico (remoto no Kaggle, local no Colab)."""
    if detect_platform() == "kaggle":
        return get_drive_client().exists(str(path.relative_to(mount_drive())))
    return path.exists()


def ensure_local_copy(path: Path) -> Path:
    """Garante uma cópia local de um arquivo do Drive canônico para leitura.

    No Colab/local o caminho já é local (Drive montado ou raiz local); no Kaggle,
    se o arquivo ainda não está no cache local, faz o download da cópia remota
    canônica. Idempotente: reexecuções reutilizam o arquivo já baixado na sessão.
    """
    if detect_platform() != "kaggle":
        return path
    if not path.is_file():
        get_drive_client().download(str(path.relative_to(mount_drive())), path)
    return path


def persist_bytes(path: Path, data: bytes) -> None:
    """Persiste bytes em um caminho do Drive (upload no Kaggle; escrita direta no Colab)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if detect_platform() == "kaggle":
        get_drive_client().upload(path, str(path.relative_to(mount_drive())))


def persist_file(local_path: Path, target_path: Path) -> None:
    """Persiste um arquivo local em um caminho do Drive canônico.

    No Colab/local o caminho canônico já é o real (Drive montado ou raiz local),
    então basta copiar o arquivo quando os caminhos diferem. No Kaggle o arquivo
    local é enviado via Drive API com upload resumível (streaming), adequado para
    arquivos grandes como o composite normalizado.
    """
    if detect_platform() == "kaggle":
        get_drive_client().upload(local_path, str(target_path.relative_to(mount_drive())))
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.resolve() != target_path.resolve():
        shutil.copy2(str(local_path), str(target_path))


def relocate_exported_file(
    file_name: str,
    staging_folder: str,
    target_path: Path,
    wait_timeout_seconds: int = 600,
    poll_interval: int = 10,
) -> None:
    """Move um artefato exportado pelo GEE para o caminho canônico.

    O GEE exporta para uma única pasta nomeada na raiz do Drive (sem aceitar
    subcaminhos); após a conclusão da tarefa, o arquivo é realocado para o
    caminho aninhado canônico resolvido do config.yaml. No Colab, a conclusão
    da task (lado servidor) pode preceder a propagação do FUSE mount, então o
    arquivo é aguardado (polling) antes de movê-lo.
    """
    if detect_platform() == "kaggle":
        remote_src = f"{staging_folder}/{file_name}"
        remote_dst = str(target_path.relative_to(mount_drive()))
        get_drive_client().move(remote_src, remote_dst)
        return
    source = mount_drive() / staging_folder / file_name
    deadline = time.monotonic() + wait_timeout_seconds
    while not source.is_file() and time.monotonic() < deadline:
        print(f"  Aguardando arquivo exportado no Drive: {source.name} ...")
        time.sleep(poll_interval)
    if not source.is_file():
        raise FileNotFoundError(f"Arquivo exportado não encontrado no Drive: {source}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target_path))


def mount_drive() -> Path:
    """Monta/acessa o Google Drive e retorna a raiz MyDrive.

    Colab: montagem nativa em /content/drive.
    Kaggle: cache local em /kaggle/working/drive (I/O remoto via Drive API).
    Local: usa a variável DRIVE_ROOT, com fallback em ~/MyDrive (dev only).
    """
    platform = detect_platform()
    if platform == "colab":
        from google.colab import drive

        drive.mount(str(COLAB_DRIVE_ROOT.parent))
        return COLAB_DRIVE_ROOT
    if platform == "kaggle":
        KAGGLE_DRIVE_CACHE.mkdir(parents=True, exist_ok=True)
        return KAGGLE_DRIVE_CACHE
    return Path(os.getenv("DRIVE_ROOT", str(Path.home() / "MyDrive")))


def ensure_storage_root() -> Path:
    """Garante que a raiz tcc/ e todas as subpastas existam (idempotente).

    No Colab/local cria diretórios reais; no Kaggle cria o cache local e as
    pastas remotas via Drive API (best-effort, validado no notebook 00).
    """
    config = get_config()
    root_name = config["storage"]["drive_root"]
    drive_root = mount_drive()
    root = drive_root / root_name
    subfolders = config["storage"]["subfolders"].values()

    if detect_platform() == "kaggle":
        client = get_drive_client()
        client.ensure_folder(root_name)
        for sub in subfolders:
            client.ensure_folder(f"{root_name}/{sub}")
    for sub in subfolders:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def resolve_storage_paths() -> dict[str, Path]:
    """Resolve os caminhos de armazenamento a partir do config.yaml."""
    config = get_config()
    root = ensure_storage_root()
    return {key: root / sub for key, sub in config["storage"]["subfolders"].items()}


def _mirror_repo_root() -> Path:
    """Raiz local do espelho do repositório (MyDrive/tcc/repo) no Colab/local."""
    config = get_config()
    return mount_drive() / config["storage"]["drive_root"] / "repo"


def _mirror_available() -> bool:
    """Indica se o espelho MyDrive/tcc/repo/src existe localmente (Colab/local).

    No Kaggle o espelho é remoto (Drive API) e a leitura de uma árvore inteira
    por API não é usada: lá o notebook sempre obtém o código via bootstrap.
    """
    if detect_platform() == "kaggle":
        return False
    return (_mirror_repo_root() / "src").is_dir()


def _copy_mirror_to_workspace(workspace: Path) -> None:
    """Copia src/ (e data/external/, se houver) do espelho para o workspace."""
    mirror = _mirror_repo_root()
    shutil.copytree(mirror / "src", workspace / "src")
    external = mirror / "data" / "external"
    if external.is_dir():
        shutil.copytree(external, workspace / "data" / "external")


def sync_repo_to_workspace(workspace: Path) -> Path:
    """Entrega o código (src/) e os dados externos (data/external/) ao runtime.

    Se o espelho MyDrive/tcc/repo existir, copia dele; se não existir (primeiro
    run), usa src/bootstrap (stdlib) que baixa o repositório público e cria o
    espelho no Drive — os próprios notebooks geram a estrutura dentro de tcc/.
    """
    for rel in ("src", "data"):
        path = workspace / rel
        if path.exists():
            shutil.rmtree(path)
    if _mirror_available():
        _copy_mirror_to_workspace(workspace)
    else:
        bootstrap.bootstrap_workspace(workspace)
    # Remove módulos src.* em cache para forçar a versão recém-copiada/baixada.
    bootstrap._purge_src_modules()
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    return workspace / "src"
