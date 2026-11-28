# Guia de reprodutibilidade

Este guia explica como reproduzir, auditar e estender os resultados do estudo
comparativo **U-Net × SegFormer** na segmentação de lavouras cafeeiras (Região
Geográfica Imediata de Guaxupé – MG). As decisões metodológicas e as limitações
conhecidas estão em [`DECISIONS.md`](DECISIONS.md); o roteiro por estágio, em
[`../PLAN.md`](../PLAN.md); as regras de código, em [`../AGENTS.md`](../AGENTS.md).

## 1. Três níveis de reprodução

| Nível | Objetivo | Precisa de | O que fazer |
| --- | --- | --- | --- |
| **A. Qualidade do código** | Conferir que o pacote `src/` e os testes passam | Python ≥ 3.10 e `uv` | [Seção 3](#3-ambiente-local-nível-a) |
| **B. Pipeline completo** | Refazer dados, treino, avaliação, XAI e figuras | Google Drive, conta do Earth Engine e GPU (Colab ou Kaggle) | [Seções 4–6](#4-preparar-um-fork-nível-b) |
| **C. Reuso dos artefatos publicados** | Reavaliar ou inspecionar dataset e pesos sem retreinar | Conta no Hugging Face | Baixar dataset e pesos do estágio `16` e reaproveitar os estágios `11`–`17` |

> Os resultados são reproduzidos **no protocolo**, não bit a bit entre
> hardwares: mesmas sementes, versões, configuração e processamento (ver
> [Seção 8](#8-determinismo-o-que-esperar)).

## 2. Visão do fluxo

```text
IBGE (AOI) + Sentinel-2 (GEE) ─┐
MapBiomas / AlphaEarth / Emater ┴─► 01–02 mosaicos e máscaras por fonte
                                    03–04 comparação e DECISÃO MANUAL da fonte
                                    05–08 composite, patches 512×512, k-fold espacial, EDA
                                    09–10 treino U-Net / SegFormer (5 dobras)
                                    11    avaliação por pixel + histogramas de probabilidade
                                    12    comparação (blocos/dobras) + sensibilidade ao limiar
                                    13–14 Grad-CAM / Attention Rollout
                                    15    tabelas e figuras de síntese
                                    16    publicação no Hugging Face Hub
                                    17    figuras e tabelas didáticas (monografia, artigo, slides)
```

Todo o armazenamento canônico fica em `MyDrive/tcc/` (árvore em
[`../README.md`](../README.md#armazenamento-canônico-google-drive)).

## 3. Ambiente local (nível A)

```bash
git clone --recurse-submodules https://github.com/jotap1101/tcc-lab.git
cd tcc-lab
uv sync --locked                       # dependências do projeto + dev (uv.lock)
uv run ruff check src tests            # lint
uv run ruff format --check src tests   # formatação
uv run mypy src tests                  # tipos
uv run pytest tests                    # testes (não exigem GPU, rede nem Drive)
python .opencode/tools/scripts/check_notebook.py   # convenções dos notebooks
```

Os testes usam dados sintéticos e modelos pequenos; nenhuma credencial é
necessária. O CI (`.github/workflows/ci.yml`) executa exatamente esses gates.

## 4. Preparar um fork (nível B)

Se você não for o autor, ajuste antes de executar:

1. **Faça fork** do repositório. Os notebooks baixam `src/` do **repositório
   público no GitHub** (`main`); portanto, aponte `REPO_URL`, `COMMIT_API_URL` e
   `COMMIT_FEED_URL` em `src/bootstrap.py` para o seu fork.
2. **Edite `src/config.yaml`:**
   - `gee.project`: ID do **seu** projeto Cloud habilitado para o Earth Engine;
   - `secrets.kaggle_dataset`: slug do **seu** dataset privado de segredos;
   - `storage.drive_root`: nome da pasta na raiz do Drive (padrão `tcc`);
   - `ground_truth.chosen_source`: fonte de treino (ver [`DECISIONS.md`](DECISIONS.md)).
3. **Publicação (opcional):** os repositórios do Hugging Face derivam de
   `<usuário>/tcc-coffee-...`; sobrescreva com `HF_DATASET_REPO`,
   `HF_UNET_REPO` e `HF_SEGFORMER_REPO`.
4. **Confirme que `src/` está publicado:** qualquer mudança local só chega aos
   notebooks depois de `git push` para a `main` do repositório que o bootstrap
   baixa.

> **Atenção:** qualquer edição em `src/config.yaml` muda o `config_hash`, que é
> parte do fingerprint de treino e avaliação. Trocar `chosen_source`, a perda,
> as épocas etc. **invalida e refaz o treino** das dobras afetadas. Parâmetros
> puramente editoriais (largura/dpi das figuras) ficam em `src/figstyle.py`
> justamente para não invalidar o treino.

## 5. Credenciais

- **Colab:** autenticação nativa com a conta logada (Drive e Earth Engine).
- **Kaggle (obrigatório para o treino):** o Kaggle não monta o Drive; gere os
  tokens uma única vez com `scripts/generate_drive_token.py` e
  `scripts/generate_gee_credentials.py` e publique-os em um dataset **privado**
  com `scripts/upload_kaggle_secrets.py`.
- **Hugging Face:** `HF_TOKEN` com permissão de escrita, apenas para o estágio `16`.
- Use sempre a **mesma conta pessoal** em todos os serviços; caso contrário, a
  árvore `tcc/` aparecerá em outro Drive.
- Nenhum segredo é versionado. Detalhes em [`../secrets/README.md`](../secrets/README.md).

## 6. Executar os notebooks

Execute em **ordem numérica**. Cada notebook é idempotente: reexecutá-lo
reutiliza o que está vigente e regenera apenas o que mudou.

| Estágio | Notebook | GPU | Decisão humana | Saída principal (sob `MyDrive/tcc/`) |
| --- | --- | --- | --- | --- |
| 00 | `00_setup_environment` | — | autenticar GEE | ambiente, `src/`, árvore `tcc/` |
| 01 | `01_gee_sentinel2_acquisition` | — | — | mosaicos Sentinel-2 por ano (`data/raw/`) |
| 02 | `02_gee_reference_masks` | — | — | máscaras por fonte (`data/interim/<fonte>/`) |
| 03 | `03_mask_sources_comparison` | — | — | diagnóstico entre fontes + figura |
| 04 | `04_source_decision` | — | **escolher a fonte** | `source_decision_<região>.json` |
| 05 | `05_preprocessing` | — | — | composites normalizados |
| 06 | `06_patch_generation` | — | — | patches + `manifest_<fonte>.parquet` |
| 07 | `07_spatial_kfold_split` | — | — | coluna `fold` + `split_<fonte>.meta.json` |
| 08 | `08_dataset_eda` | — | — | `normalization_stats_<fonte>.json` + figura |
| 09 | `09_train_unet` | **sim** | — | `models/unet/<fonte>/fold_i.pt` + métricas/histórico |
| 10 | `10_train_segformer` | **sim** | — | `models/segformer/<fonte>/fold_i.pt` + métricas/histórico |
| 11 | `11_evaluation` | **sim** | — | métricas por patch/dobra, predições e histogramas de probabilidade |
| 12 | `12_comparative_analysis` | — | — | `comparison.json` + figuras (comparação, inferência, limiar, erros) |
| 13 | `13_xai_gradcam_unet` | opcional | — | mapas Grad-CAM + figura |
| 14 | `14_xai_attention_rollout_segformer` | opcional | — | mapas Attention Rollout + figura |
| 15 | `15_results_synthesis` | — | — | tabelas CSV/LaTeX + figuras de síntese |
| 16 | `16_export_release` | — | — | dataset e pesos no Hugging Face Hub |
| 17 | `17_monograph_figures` | — | — | figuras (PDF/PNG), tabelas e legendas para publicação |

Dicas de execução:

- **Kaggle:** `python scripts/push_kaggle_notebooks.py [NN ...] [--run|--dry-run]`
  envia notebooks pela CLI do Kaggle (uso local; não é chamado pelos notebooks).
- **Estágio 04 é manual por desenho:** o sistema nunca elege a fonte. Edite
  `ground_truth.chosen_source`, faça commit/push e reexecute `04` e seguintes.
- **Primeira célula de cada notebook:** baixa o `src/` e instala as versões de
  `requirements-runtime.txt`. Se aparecer reinício de kernel, reexecute a
  partir da primeira célula (é esperado quando um módulo compilado diverge).
- **Mantenha os notebooks sem saídas** ao commitar (o CI falha caso contrário).

## 7. O que reexecutar quando algo muda

Cada estágio guarda um *fingerprint* das entradas; reexecutar basta para
regenerar o que ficou desatualizado. Use esta matriz para saber **onde começar**:

| O que mudou | Reexecutar a partir de |
| --- | --- |
| Fonte de ground truth, `data.*`, filtro de validade, patches | `06` (e todos os seguintes) |
| Divisão espacial / `splits.fold_count` | `07` (e seguintes) |
| `training.*`, `loss.*`, `model.*`, `augmentation.*`, qualquer chave de `config.yaml` | `09`–`10` (retreina), depois `11`–`17` |
| Código de avaliação (`src/evaluation.py`, versão do esquema) | `11`–`17` (sem retreino) |
| Estatística da comparação (`src/comparative.py`, `src/spatial_inference.py`, `src/thresholds.py`) | `12`, `15`, `17` |
| XAI (`src/xai/*`) | `13`–`14`, depois `15`, `17` |
| Só estilo/legendas das figuras (`src/figstyle.py`, `src/monograph.py`) | o estágio que gera a figura (`03`, `07`, `08`, `11`–`15`) e `17` |
| Publicação (`src/release.py`) | `16` |

Baselines históricos usam o sufixo `_v1` (`models/unet/emater_v1`,
`data/processed/archive_v1`, `artifacts/figures/archive_v1`) e **nunca** são
sobrescritos.

## 8. Determinismo: o que esperar

- Sementes fixadas (`python`/`numpy`/`torch`/`cuda`, `reproducibility.seed = 42`),
  `cudnn.deterministic = True`, `benchmark = False`,
  `use_deterministic_algorithms(True)` quando suportado e `num_workers = 0`.
- Os estágios `10` e `11` (SegFormer) usam `warn_only=True`, porque o *upsampling*
  bilinear não tem *backward* determinístico no CUDA; a configuração efetiva fica
  em `artifacts/runs/<modelo>/<fonte>/run.meta.json` (versões, sementes,
  `config_hash`, commit e fingerprint do manifesto).
- **Não se espera igualdade bit a bit entre GPUs diferentes**; espera-se a mesma
  conclusão e métricas muito próximas. Compare resultados pelos fingerprints e
  intervalos de confiança, não por igualdade exata.
- A análise estatística é determinística dado o mesmo conjunto de artefatos
  (sementes derivadas de `reproducibility.seed`).

## 9. Onde está cada resultado

| Pergunta | Arquivo |
| --- | --- |
| Métricas agregadas por modelo | `artifacts/metrics/<modelo>/<fonte>/evaluation/summary.json` |
| Diferença SegFormer − U-Net, testes por patch, blocos e dobras, limiar | `artifacts/metrics/comparative/<fonte>/comparison.json` |
| Tabelas para o texto (CSV e LaTeX) | `artifacts/synthesis/<fonte>/tables/` e `artifacts/monograph/<fonte>/tables/` |
| Figuras (PNG 300 dpi + PDF vetorial) | `artifacts/figures/` e `artifacts/monograph/<fonte>/figures/` |
| Legendas prontas | `artifacts/monograph/<fonte>/captions.{tex,md}` |
| Proveniência de uma execução | `run.meta.json`, `release.json`, `synthesis_metadata.json`, `inventory.json` |

Chaves principais de `comparison.json`: `paired` (por patch, **descritivo**),
`spatial_inference.block_level` e `.fold_level` (**inferência principal**),
`threshold_sensitivity` (AP, limiar fixo/calibrado/ótimo) e `fingerprint`.

## 10. Padrão das figuras

Todas as figuras usam `src/figstyle.py`: largura de 16 cm (área de texto ABNT),
fontes de 8–10 pt, paleta segura para daltonismo, **PNG a 300 dpi e PDF
vetorial** lado a lado. Para uma nova figura:

1. Monte-a com `figstyle.render_publication(build_fn, height_cm=...)`;
2. Persista com `figstyle.persist_figure(caminho_png, renderizado)`;
3. Não acrescente parâmetros de apresentação a `config.yaml`.

## 11. Solução de problemas

| Sintoma | Causa provável | O que fazer |
| --- | --- | --- |
| Notebook reinicia o kernel logo no início | Módulo compilado já carregado difere do pin | Reexecutar desde a primeira célula |
| `ImportError` ao importar `transformers` no Kaggle | `torchaudio` compilado para outra versão de CUDA | `remove_incompatible_torchaudio()` (já chamado nos notebooks `10`, `11`, `14`) |
| “Resumo do estágio 11 desatualizado ou incompleto” | Esquema/fingerprint de avaliação mudou | Reexecutar `11` para os dois modelos |
| Mudança em `src/` não aparece no notebook | `src/` é baixado da `main` pública | `git push` antes de executar |
| `tcc/` aparece em outro Drive | Contas diferentes entre Colab/Kaggle/tokens | Reautorizar com a mesma conta |
| Treino recomeça sem motivo aparente | `config.yaml` (hash) ou manifesto/split mudaram | Conferir `git diff src/config.yaml` |
| Erro de cota/latência do Drive | Escrita pesada (patches, checkpoints) | Reexecutar; os estágios retomam do que já existe |
| CI de notebooks falha | Saídas de célula commitadas | Limpar as saídas (ex.: `jupyter nbconvert --clear-output --inplace`) |
