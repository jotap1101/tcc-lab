"""Testes de src/io.py (abstração de plataforma e de armazenamento tcc/)."""

from pathlib import Path

from src import io
from src.config import get_config


class _FakeRequest:
    """Requisição falsa da Drive API que devolve um payload fixo."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def execute(self) -> dict:
        return self._payload


class _FakeFiles:
    """Recurso `files()` falso, registrando criações e atualizações."""

    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[str] = []

    def create(self, body: dict, media_body: object, fields: str) -> _FakeRequest:
        self.created.append(body)
        return _FakeRequest({"id": "created-id"})

    def update(self, fileId: str, media_body: object) -> _FakeRequest:
        self.updated.append(fileId)
        return _FakeRequest({"id": fileId})


class _FakeService:
    """Serviço falso da Drive API expondo apenas o recurso `files()`."""

    def __init__(self, files: _FakeFiles) -> None:
        self._files = files

    def files(self) -> _FakeFiles:
        return self._files


def test_upload_creates_parent_folders_and_overwrites(tmp_path: Path, monkeypatch) -> None:
    """O upload deve criar pastas pai aninhadas e atualizar arquivos existentes."""
    client = io.DriveClient()
    local = tmp_path / "fold_0.pt"
    local.write_bytes(b"weights")
    files = _FakeFiles()
    ensured: list[str] = []

    def _ensure_folder(path: str) -> str:
        ensured.append(path)
        return "parent-id"

    monkeypatch.setattr(client, "_build_service", lambda: _FakeService(files))
    monkeypatch.setattr(client, "ensure_folder", _ensure_folder)
    monkeypatch.setattr(client, "_file_id", lambda path: None)

    client.upload(local, "tcc/models/unet/emater/fold_0.pt")

    assert ensured == ["tcc/models/unet/emater"]
    assert [body["name"] for body in files.created] == ["fold_0.pt"]
    assert files.created[0]["parents"] == ["parent-id"]
    assert files.updated == []

    # Reexecução: o arquivo remoto existente deve ser atualizado, sem duplicata.
    monkeypatch.setattr(client, "_file_id", lambda path: "existing-id")
    client.upload(local, "tcc/models/unet/emater/fold_0.pt")

    assert files.updated == ["existing-id"]


def test_detect_platform_returns_known_value() -> None:
    """A detecção de plataforma deve retornar colab, kaggle ou local."""
    assert io.detect_platform() in {"colab", "kaggle", "local"}


def test_ensure_storage_root_idempotent_local(tmp_path, monkeypatch) -> None:
    """No modo local, a raiz tcc/ e todas as subpastas são criadas de forma idempotente."""
    monkeypatch.setattr(io, "detect_platform", lambda: "local")
    monkeypatch.setattr(io, "mount_drive", lambda: tmp_path)

    root = io.ensure_storage_root()
    assert root == tmp_path / "tcc"
    assert root.is_dir()

    root_again = io.ensure_storage_root()
    assert root_again == root

    for sub in get_config()["storage"]["subfolders"].values():
        assert (root / sub).is_dir()


def test_resolve_storage_paths_local(tmp_path, monkeypatch) -> None:
    """Os caminhos resolvidos devem apontar para dentro da raiz tcc/ no Drive."""
    monkeypatch.setattr(io, "detect_platform", lambda: "local")
    monkeypatch.setattr(io, "mount_drive", lambda: tmp_path)

    paths = io.resolve_storage_paths()
    assert paths["data_raw"] == tmp_path / "tcc" / "data" / "raw"
    assert paths["models"] == tmp_path / "tcc" / "models"


def test_relocate_exported_file_local(tmp_path, monkeypatch) -> None:
    """No modo local, o artefato da pasta de staging é realocado ao caminho canônico."""
    monkeypatch.setattr(io, "detect_platform", lambda: "local")
    monkeypatch.setattr(io, "mount_drive", lambda: tmp_path)

    file_name = "sentinel2_310044_2023-01-01_2023-12-31.tif"
    staging = tmp_path / "tcc" / file_name
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_bytes(b"mosaic")

    target = tmp_path / "tcc" / "data" / "raw" / "sentinel2" / file_name
    io.relocate_exported_file(file_name, "tcc", target)

    assert not staging.exists()
    assert target.read_bytes() == b"mosaic"


def test_path_exists_local(tmp_path, monkeypatch) -> None:
    """No modo local, path_exists reflete a existência do arquivo no sistema."""
    monkeypatch.setattr(io, "detect_platform", lambda: "local")
    target = tmp_path / "tcc" / "data" / "interim" / "mapbiomas" / "mask.tif"
    target.parent.mkdir(parents=True)

    assert not io.path_exists(target)
    target.write_bytes(b"mask")
    assert io.path_exists(target)


def test_persist_bytes_local(tmp_path, monkeypatch) -> None:
    """No modo local, persist_bytes escreve os bytes no caminho informado."""
    monkeypatch.setattr(io, "detect_platform", lambda: "local")
    target = tmp_path / "tcc" / "artifacts" / "figures" / "preview.png"
    io.persist_bytes(target, b"\x89PNG")
    assert target.read_bytes() == b"\x89PNG"


def test_persist_file_local(tmp_path, monkeypatch) -> None:
    """No modo local, persist_file copia o arquivo para o caminho canônico."""
    monkeypatch.setattr(io, "detect_platform", lambda: "local")
    source = tmp_path / "composite.tif"
    source.write_bytes(b"composite")
    target = tmp_path / "tcc" / "data" / "processed" / "composites" / "composite.tif"
    io.persist_file(source, target)
    assert target.read_bytes() == b"composite"


def test_persist_file_local_same_path_noop(tmp_path, monkeypatch) -> None:
    """No modo local, persist_file no próprio caminho é um no-op (sem duplicação)."""
    monkeypatch.setattr(io, "detect_platform", lambda: "local")
    path = tmp_path / "composite.tif"
    path.write_bytes(b"composite")
    io.persist_file(path, path)
    assert path.read_bytes() == b"composite"


def test_ensure_local_copy_returns_path_on_local(tmp_path, monkeypatch) -> None:
    """No modo local, ensure_local_copy deve retornar o próprio caminho."""
    monkeypatch.setattr(io, "detect_platform", lambda: "local")
    path = tmp_path / "mask.tif"
    assert io.ensure_local_copy(path) == path
