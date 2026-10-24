"""Publica/atualiza os notebooks do repositório no Kaggle via API.

Para cada notebook, gera um `kernel-metadata.json` com o dataset privado de
tokens (`secrets.kaggle_dataset`) em `dataset_sources` e publica com
`kaggle kernels push`. Assim o input do dataset acompanha sempre o notebook,
mesmo quando o código é mantido no GitHub, sem anexo manual na UI.

Por padrão publica com `--no-run` (equivalente ao Quick Save). Use `--run`
para também executar o notebook.

Uso:
    python scripts/push_kaggle_notebooks.py                  # todos, sem executar
    python scripts/push_kaggle_notebooks.py 00 01            # estágios específicos
    python scripts/push_kaggle_notebooks.py 09 --run         # publica e executa
    python scripts/push_kaggle_notebooks.py --dry-run        # só mostra o metadata
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _kaggle_cli import authenticated_username, find_kaggle_cli, run_cli

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
GPU_STAGES = {"09", "10"}  # estágios de treinamento (U-Net e SegFormer)


def _select_notebooks(stages: list[str]) -> list[Path]:
    """Seleciona notebooks por estágio (prefixo numérico) ou todos por padrão."""
    available = {
        path.name.split("_")[0]: path for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    }
    if not stages:
        return [available[key] for key in sorted(available)]
    selected: list[Path] = []
    for stage in stages:
        key = Path(stage).name.split("_")[0]
        if key not in available:
            raise SystemExit(
                f"Notebook do estágio '{stage}' não encontrado em {NOTEBOOKS_DIR}."
            )
        selected.append(available[key])
    return selected


def _kernel_metadata(notebook: Path, owner: str, dataset_id: str) -> dict[str, object]:
    """Monta o kernel-metadata.json do notebook para publicação."""
    stage = notebook.name.split("_")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", notebook.stem.lower()).strip("-")
    return {
        "id": f"{owner}/{slug}",
        "title": notebook.stem.replace("_", " "),
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": stage in GPU_STAGES,
        "enable_internet": True,
        "dataset_sources": [dataset_id],
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", help="estágios a publicar (ex.: 00 09)")
    parser.add_argument("--run", action="store_true", help="executa após publicar")
    parser.add_argument("--dry-run", action="store_true", help="apenas exibe o metadata")
    parser.add_argument("--dataset-id", default=None, help="owner/slug (padrão: config.yaml)")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from src.config import get_config

    dataset_id = args.dataset_id or get_config()["secrets"]["kaggle_dataset"]
    cli = find_kaggle_cli()
    owner = authenticated_username(cli)
    if not owner:
        raise SystemExit("Usuário do Kaggle não identificado; rode `kaggle auth login`.")
    if dataset_id.split("/")[0] != owner:
        raise SystemExit(
            f"O dataset '{dataset_id}' não pertence ao usuário autenticado '{owner}'."
        )

    for notebook in _select_notebooks(args.stages):
        metadata = _kernel_metadata(notebook, owner, dataset_id)
        if args.dry_run:
            print(f"# {notebook.name}")
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
            continue
        with tempfile.TemporaryDirectory() as staging:
            staging_path = Path(staging)
            shutil.copy2(notebook, staging_path / notebook.name)
            (staging_path / "kernel-metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            command = ["kernels", "push", "-p", str(staging_path)]
            if not args.run:
                command.append("--no-run")
            code, output = run_cli(cli, command)
        if code != 0 or "error" in output.lower():
            raise SystemExit(f"Falha ao publicar {notebook.name}.")
        print(f"Publicado: https://www.kaggle.com/code/{metadata['id']}")


if __name__ == "__main__":
    main()
