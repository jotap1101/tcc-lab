"""Publica/atualiza o dataset privado do Kaggle com os tokens locais.

Executar na máquina após gerar ou rotacionar os tokens em `secrets/`. O dataset
privado declarado em `src/config.yaml` (`secrets.kaggle_dataset`) é anexado aos
notebooks como input e lido pelo fallback de `get_secret` em
`/kaggle/input/<slug>/`, eliminando o anexo manual de Secrets por notebook.
Nenhum token é versionado: este script apenas os envia ao Kaggle.

Autenticação: `~/.kaggle/kaggle.json` ou `KAGGLE_API_TOKEN`.

Uso:
    python scripts/upload_kaggle_secrets.py
    python scripts/upload_kaggle_secrets.py --message "rotate drive token"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _kaggle_cli import authenticated_username, find_kaggle_cli, run_cli

REPO_ROOT = Path(__file__).resolve().parent.parent
SECRETS_DIR = REPO_ROOT / "secrets"
DATASET_FILES = ("drive_token.json", "earthengine_credentials.json")
OPTIONAL_DATASET_FILES = ("hf_token.txt",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=None, help="owner/slug (padrão: config.yaml)")
    parser.add_argument(
        "--message", default="Atualiza tokens do projeto", help="mensagem da versão"
    )
    parser.add_argument("--create", action="store_true", help="força a criação do dataset")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from src.config import get_config

    dataset_id = args.dataset_id or get_config()["secrets"]["kaggle_dataset"]
    missing = [name for name in DATASET_FILES if not (SECRETS_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Tokens ausentes em {SECRETS_DIR}: {missing}")

    cli = find_kaggle_cli()
    username = authenticated_username(cli)
    owner = dataset_id.split("/")[0]
    if username and owner != username:
        raise SystemExit(
            f"O dataset '{dataset_id}' pertence a '{owner}', mas a conta autenticada "
            f"é '{username}'. Use --dataset-id {username}/tcc-secrets ou ajuste "
            "secrets.kaggle_dataset em src/config.yaml."
        )

    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging)
        for name in (*DATASET_FILES, *OPTIONAL_DATASET_FILES):
            source = SECRETS_DIR / name
            if source.is_file():
                shutil.copy2(source, staging_path / name)
            elif name in DATASET_FILES:
                raise FileNotFoundError(f"Token ausente em {SECRETS_DIR}: {name}")
        metadata = {
            "id": dataset_id,
            "title": dataset_id.split("/")[-1].replace("-", " "),
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging_path / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        exists = (
            False
            if args.create
            else run_cli(cli, ["datasets", "files", dataset_id], echo=False)[0] == 0
        )
        if exists:
            code, output = run_cli(
                cli, ["datasets", "version", "-p", str(staging_path), "-m", args.message]
            )
        else:
            code, output = run_cli(cli, ["datasets", "create", "-p", str(staging_path)])

    if code != 0 or "error" in output.lower():
        raise SystemExit(
            "Falha ao publicar o dataset. Verifique a autenticação do Kaggle "
            "(kaggle auth login) e o slug/dono do dataset."
        )
    print(f"Dataset publicado: {dataset_id}")
    print("Anexe-o uma vez em Input > Add Input (ou use dataset_sources no kernel-metadata).")


if __name__ == "__main__":
    main()
