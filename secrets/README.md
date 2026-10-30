# Tokens e credenciais (Kaggle)

Este diretório **não contém segredos** — contém apenas modelos/documentação. Os
tokens reais NUNCA são versionados (repo público; regra do `AGENTS.md`).

## Por que existem tokens?

O **Colab** autentica com a conta logada de forma nativa (Drive e Earth Engine).
O **Kaggle** não monta o Drive e roda em modo headless — por isso precisa de tokens obtidos uma única vez com a **conta principal** (Drive, Earth Engine) e de um token de escrita do Hugging Face para o release:

| Variável            | Para quê                                  | Como obter                                   | Onde guardar                          |
| ------------------- | ----------------------------------------- | -------------------------------------------- | ------------------------------------- |
| `GDRIVE_TOKEN`      | Acesso ao `MyDrive/tcc/` via Drive API    | `scripts/generate_drive_token.py`            | dataset privado `tcc-secrets`          |
| `GDRIVE_TOKEN_FILE` | Alternativa: caminho de um arquivo token  | idem (envie o arquivo como dataset)          | dataset privado `tcc-secrets`          |
| `GEE_CREDENTIALS`   | Autenticação headless do Earth Engine     | `scripts/generate_gee_credentials.py`        | dataset privado `tcc-secrets`          |
| `GEE_PROJECT`       | Projeto Cloud do GEE (`ee.Initialize`)    | ID visto no Earth Engine Code Editor         | `src/config.yaml` (`gee.project`)      |
| `HF_TOKEN`          | Publicação do dataset/modelos no HF Hub   | `huggingface.co/settings/tokens` (escrita)   | dataset privado (`hf_token.txt`), Kaggle Secrets ou env var |

O `HF_TOKEN` é lido pelo estágio 16 (`src.release.huggingface_token`) e nunca é impresso. O dataset privado é declarado em `src/config.yaml` (`secrets.kaggle_dataset`).
`src/utils.get_secret` resolve os segredos na ordem **env var → Kaggle Secret →
dataset** (`/kaggle/input/<slug>/`), então anexar o dataset uma vez dispensa
configurar Secrets em cada notebook novo.

## Como usar

1. Gere os tokens na sua máquina (ou num Colab) com os scripts em `scripts/`.
2. **Cópia local (opcional, git-ignored):** salve os tokens aqui em `secrets/`
   (`drive_token.json`, `earthengine_credentials.json`) — este diretório tem um
   `.gitignore` próprio e esses arquivos **não** são versionados.
3. **Kaggle (recomendado):** publique os tokens no dataset privado com
   `python scripts/upload_kaggle_secrets.py` e anexe o dataset aos notebooks
   (**Input → Add Input** ou `dataset_sources` no `kernel-metadata.json`).
   Alternativa: Secrets do Kaggle (Settings → Secrets) com `GDRIVE_TOKEN` e
   `GEE_CREDENTIALS`.
4. Os notebooks leem os segredos via `src.utils.get_secret`; em Colab, com
   fallback para arquivos em `MyDrive/tcc/secrets/` (ex.: `drive_token.json`,
   `earthengine_credentials.json`).

## Regras

- Autorize sempre com a **mesma conta pessoal** — caso contrário a estrutura
  `tcc/` aparecerá em outro Drive/conta.
- Nunca commitar, logar ou expor os tokens. Use variáveis de ambiente, Secrets
  do Kaggle ou o dataset privado — nunca arquivos versionados.
- O `GDRIVE_TOKEN` também é usado pelo `sync_repo_to_workspace()` e
  `ensure_storage_root()` no Kaggle (ver `src/io.py`).