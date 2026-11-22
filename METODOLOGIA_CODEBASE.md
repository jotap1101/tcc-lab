# METODOLOGIA_CODEBASE.md — Auditoria Factual da Implementação

> **Finalidade.** Este arquivo é a fonte factual de implementação ("ground truth" de código) para a redação do Capítulo 3 (Metodologia) da monografia. Ele descreve **apenas** o que está efetivamente implementado no repositório `tcc-lab`, com parâmetros, fórmulas, caminhos e decisões extraídos do código, da configuração e do histórico Git.
>
> **Commit auditado.** `bd612ac4a709e1e171fd06dd62aff1d458527830` (`bd612ac`, 2026-11-10 20:35:55 -0300, merge do PR #5). O repositório tem **115 commits** (2026-08-12 → 2026-11-10).
>
> **Estado da árvore de trabalho no momento da auditoria.** `git status` registra: `D METODOLOGIA.md`, `D RESULTADOS.md` (ambos existem no commit `4564e35` e foram removidos da árvore local) e dois arquivos modificados em `.opencode/plugins/`. Nada disso foi alterado por esta auditoria.
>
> **Método.** Leitura integral de `src/` (33 arquivos `.py`, 8.796 linhas), `src/config.yaml`, `pyproject.toml`, `requirements-runtime.txt`, `.github/workflows/ci.yml`, dos 17 notebooks de `notebooks/`, dos scripts de `scripts/`, dos dados vetoriais versionados em `data/external/` e do histórico completo de `git log`. Os quantitativos de modelos foram **recalculados** por instanciação local das arquiteturas; os quantitativos de área foram recalculados a partir do shapefile do IBGE e dos GeoJSONs da Emater.
>
> **Legenda de proveniência.**
>
> - `[código]` — verificado diretamente no código-fonte.
> - `[config]` — verificado em `src/config.yaml`.
> - `[git]` — verificado no histórico de commits.
> - `[calculado]` — recalculado nesta auditoria a partir dos dados/código do repositório.
> - `[doc 4564e35]` — valor documentado em `METODOLOGIA.md`/`RESULTADOS.md` no commit `4564e35`, derivado de artefatos de execução que **não estão versionados** no repositório (ficam no Drive/Hugging Face Hub). Não é reproduzível apenas com o repositório.
> - `[NÃO ENCONTRADO NO REPOSITÓRIO]` — o item solicitado não existe ou não está registrado no repositório.

---

## 1. Visão Geral do Pipeline e Evolução via Histórico Git

### 1.1 Inventário do repositório (commit auditado)

| Elemento | Conteúdo |
| --- | --- |
| `src/` | 33 arquivos `.py` (8.796 linhas): pacote compartilhado, único alvo de `ruff`/`mypy`/`pytest` `[código]` |
| `notebooks/` | 17 estágios `00`–`16`, executados em ordem numérica `[código]` |
| `tests/` | 26 arquivos de teste, **239 funções `test_`** `[calculado]` |
| `scripts/` | 6 utilitários locais (tokens OAuth do Drive/GEE, publicação de notebooks/secrets no Kaggle) `[código]` |
| `data/external/` | malha IBGE 2025 (shapefile) e 9 GeoJSONs da Emater — **versionados** `[código]` |
| `src/config.yaml` | fonte única de verdade de parâmetros `[config]` |
| `requirements-runtime.txt` | versões pinadas instaladas por notebook `[código]` |
| `uv.lock` | lockfile commitado (959.667 bytes) `[calculado]` |
| `.github/workflows/ci.yml` | 2 jobs: qualidade (`ruff`, `mypy`, `pytest`) e convenções de notebook `[código]` |
| `tcc-docs/` | submódulo Git somente leitura (material teórico) `[código]` |
| Artefatos pesados | **não versionados**: `data/{raw,interim,processed}`, `models/`, `artifacts/` são git-ignored e espelham `MyDrive/tcc/` `[código]` |

### 1.2 Sequência exata das etapas do pipeline

Cada linha foi confirmada pelo notebook correspondente e pelo módulo de `src/` que ele importa.

| Estágio | Notebook | Ponto de entrada em `src/` | Entrada → saída principal |
| --- | --- | --- | --- |
| 00 Setup | `00_setup_environment.ipynb` | `bootstrap.bootstrap_workspace`, `setup.initialize`, `setup.install_runtime_requirements`, `utils.authenticate_gee`, `utils.setup_reproducibility` | — → workspace com `src/`+`data/external/`, raiz `MyDrive/tcc/` resolvida, GEE autenticado, sementes fixadas |
| 01 Aquisição | `01_gee_sentinel2_acquisition.ipynb` | `data.gee_client.{load_aoi_geometry, build_s2_collection, build_annual_mosaic, ensure_mosaic_export, save_mosaic_preview}` | malha IBGE → AOI → mosaico GeoTIFF mediano por ano |
| 02 Máscaras | `02_gee_reference_masks.ipynb` | `data.mask_utils.ensure_source_mask`, `data.emater_masks.ensure_emater_mask` | fontes → máscara binária 10 m por fonte |
| 03 Comparação | `03_mask_sources_comparison.ipynb` | `data.mask_comparison.{load_masks, compute_comparison, save_report, save_figure}` | máscaras → relatório JSON + figura multi-painel |
| 04 Decisão | `04_source_decision.ipynb` | `data.source_decision.{chosen_source, save_source_decision_report}` | relatório → registro da decisão **manual** |
| 05 Pré-processamento | `05_preprocessing.ipynb` | `data.preprocessing.{build_preprocessed_composite, verify_composite, save_composite_preview}` | mosaicos → composite alinhado e normalizado |
| 06 Patches | `06_patch_generation.ipynb` | `data.patch_generation.{generate_patches, verify_manifest, save_patch_montage}` | composite + máscara → patches `.npy` + manifesto Parquet |
| 07 Split | `07_spatial_kfold_split.ipynb` | `data.spatial_split.{assign_spatial_folds, verify_split, save_split_figure}` | manifesto → coluna `fold` + metadata do split |
| 08 EDA | `08_dataset_eda.ipynb` | `data.eda.{compute_normalization_stats, run_sanity_checks, save_eda_figure}` | manifesto → estatísticas de normalização + sanidade |
| 09 U-Net | `09_train_unet.ipynb` | `models.unet.build_unet`, `trainer.train_fold`, `trainer.save_run_metadata` | patches da fonte escolhida → pesos/métricas/histórico |
| 10 SegFormer | `10_train_segformer.ipynb` | `models.segformer.build_segformer`, `trainer.train_fold` | idem, com `setup_reproducibility(..., warn_only=True)` |
| 11 Avaliação | `11_evaluation.ipynb` | `evaluation.{evaluate_model, summarize_model, save_evaluation_figure}` | pesos → predições + métricas por pixel |
| 12 Comparação | `12_comparative_analysis.ipynb` | `comparative.{analyze_comparison, save_comparison_figure, select_error_map_patches, save_error_maps_figure}` | métricas por patch → estatística + mapas de erro |
| 13 Grad-CAM | `13_xai_gradcam_unet.ipynb` | `xai.gradcam.run_gradcam` | pesos U-Net → mapas Grad-CAM |
| 14 Rollout | `14_xai_attention_rollout_segformer.ipynb` | `xai.attention_rollout.run_attention_rollout` | pesos SegFormer → mapas Attention Rollout |
| 15 Síntese | `15_results_synthesis.ipynb` | `synthesis.run_synthesis` | artefatos 06–14 → tabelas CSV/LaTeX + figuras |
| 16 Release | `16_export_release.ipynb` | `release.{authenticate_huggingface, resolve_repo_ids, run_release}` | patches + pesos → Hugging Face Hub |

Fluxo de dependências (verificado nos `check_dependencies` dos notebooks) `[código]`:

```text
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 ─┬→ 11 → 12 ─┬→ 15 → 16
                                                  └→ 10 ─┘         │
                                          13 (depende de 09+11) ──┤
                                          14 (depende de 10+11) ──┘
```

Detalhe relevante: `15` e `16` exigem o resumo do estágio `11` para **os dois** modelos e a comparação do estágio `12`; os estágios `13`/`14` exigem `summary_is_current(...)` do estágio `11` e os artefatos de treino da dobra `[código]`.

### 1.3 Cronologia extraída do `git log`

O histórico tem 115 commits distribuídos em ondas sucessivas de trabalho. Datas e hashes conferidos em `git log --date=short`.

| Período | Commits-chave | Marco técnico |
| --- | --- | --- |
| 2026-08-12 | `c6b1896` (Initial commit), `db91c3d`, `30db13d`, `ea833f7`, `240abc6`, `5ebff93`, `839831f` | Constituição do repositório: submódulo de material teórico, `AGENTS.md`, `PLAN.md` inicial, `README.md`, `.gitignore` |
| 2026-09-12 → 2026-09-17 | `6c3420d`, `3d6d87f`, `9465889`, `ef9faba`, `5392b38`, `32099d7`, `1efd47d`, `1d090dc`, `c72a2aa`, `0249e73`, `4512291`, `4d886ea`, `6006b42`, `0abc4c9`, `3a9a61c`, `01a2103`, `a183cd4`, `3238db4`, `4aea642`, `787c051`, `eef7cd8`, `3745793` | Base de infraestrutura: configuração YAML + loader, `io`/`bootstrap`, CI (ruff/mypy/pytest), malha IBGE versionada, scripts de tokens OAuth, notebook `00`, autenticação GEE com projeto Cloud, detecção de plataforma priorizando Kaggle, `get_config` sem cache, `get_secret` unificado |
| 2026-09-18 → 2026-09-19 | `8d9b051`, `71658f4`, `c34c744`, `356c1f1`, `ab7789b`, `4ddd7de`, `61f5a64`, `c2a60e3`, `ea4e4eb`, `9cbd7a0` | Aquisição Sentinel-2 (notebook `01`), exportação GEE + realocação de arquivo, máscaras de referência (notebook `02`), comparação de máscaras, primeira "finalização da máscara binária" |
| 2026-09-21 → 2026-09-23 | `85442d9`, `38ee869`, `85d6b12`, `eb0da71`, `2001aef`, `cd2ba16`, `941a051`, `06ca6f2`, `5d89313`, `b9a4ed8`, `63256f2`, `5e1960b` | `05`–`08` e U-Net: composite, geração de patches + manifesto, k-means k-fold, EDA/normalização, abstração de armazenamento, modelo U-Net |
| 2026-10-03 → 2026-10-04 | `3b94030`, `08b01d5`, `42babe7`, `52888b2`, `672c0bd`, `332effd`, `29b3300`, `ee92e8f`, `375a55f`, `0889056`, `21cd01a`, `6c42cab`, `5025bc6`, `4931d30`, `dfefc42`, `6cd9778`, `39433eb` | Terceira fonte de ground truth (Emater vetorial); remoção do módulo `mask_finalization`; comparação sem fonte de referência fixa; reamostragem `nearest` documentada; `chosen_source = emater` |
| 2026-10-19 → 2026-10-24 | `f9951b2`, `553041b`, `30ca166`, `9ae273e`, `0efb7a6`, `7fd3fe8`, `288d217`, `b641d7e`, `ba3b1c6`, `7cce179`, `d5f8038`, `f08826e`, `419f590`, `69240b2` | Consolidação do bootstrap/Drive/Kaggle/segredos; `config_hash` e `commit_sha` no metadata de execução; instalação segura de dependências com reinício de kernel por ABI |
| 2026-10-25 → 2026-10-27 | `07f99c5`, `d2c7c34`, `97605da`, `97e806e`, `0117b15`, `8ad0e3b`, `bdefc23`, `b58d53a` | Filtro por `min_valid_fraction` e rótulo `-1`; versionamento de `bbox`/split; correção da captura do LR; treino do SegFormer e `warn_only`; remoção de `torchaudio` incompatível; avaliação por pixel; testes da comparação |
| 2026-10-29 → 2026-11-10 | `3ea7e0f`, `3533d34`, `9d4c331`, `4123f4d`, `82e594a`, `3f15dc0`, `24a5ac9`, `760acae`, `4e77998`, `dfb9796`, `4564e35`, `bd612ac` | Attention Rollout, Grad-CAM, publicação no Hugging Face Hub, testes da síntese, documentação metodológica (`METODOLOGIA.md`/`RESULTADOS.md`), citação BibTeX nos cards, bumps de dependências |

### 1.4 Decisões técnicas e justificativas registradas nos commits

| Decisão | Evidência | Justificativa registrada/inferida |
| --- | --- | --- |
| k-means em numpy puro (sem `scikit-learn`) | `941a051` (`refactor(spatial_split)`) | Evitar dependência do scikit-learn, cuja importação de `numpy.testing` é sensível à versão do numpy (docstring de `spatial_split.py`) |
| `bbox` calculada da transformação afim, sem `rasterio.windows` | `2001aef`, `38ee869`, `85d6b12` | Compatibilidade entre versões do rasterio; o valor anterior usava offset em pixels, gerando bboxes sobrepostas e split não espacial |
| Versionamento de esquema (`BBOX_SCHEMA_VERSION=1`, `SPLIT_SCHEMA_VERSION=2`) | `97605da` | Invalidar manifesto/split antigos **sem** reenviar patches já persistidos |
| Reamostragem `nearest` na exportação GEE | `29b3300`, `ee92e8f`, `c2a60e3` | Preservar valores categóricos (0/1) ao passar 30 m → 10 m; GEE só aceita `bilinear`/`bicubic` em `resample()`, então `nearest` é o comportamento nativo |
| Realocação de arquivo exportado (GEE → pasta aninhada) | `ab7789b`, `375a55f` | O GEE aceita apenas **um** nome de pasta na raiz do Drive; o arquivo é exportado para pasta plana e movido para o caminho canônico, com espera da propagação do mount FUSE no Colab |
| Filtro `min_valid_fraction` + rótulo `-1` | `07f99c5` | Descartar patches fora do AOI e excluir pixels inválidos de perda e métricas |
| `warn_only=True` no SegFormer | `0117b15` | O upsampling bilinear não tem backward determinístico no CUDA no PyTorch |
| Remoção de `torchaudio` incompatível | `8ad0e3b` | `transformers` importa `torchaudio`; no Kaggle ele pode ter sido compilado para outra CUDA |
| Reinício automático do kernel por ABI | `69240b2` | Evitar inconsistência entre extensões compiladas já carregadas e as versões pinadas |
| `chosen_source = "emater"` | `39433eb` | Decisão manual do pesquisador; o sistema não elege fonte |
| Repositório de docs renomeado para `tcc-docs` | `f9951b2` | Submódulo somente leitura de material teórico |

### 1.5 Abordagens descartadas / refatoradas

1. **Módulo `src/data/mask_finalization.py` (estágio 04) foi removido.** Ele copiava a máscara da fonte escolhida para `data/processed/ground_truth/mask_final_<região>_<ano>.tif` e persistia relatório/figura de finalização (`git show 5d89313:src/data/mask_finalization.py`). No commit `332effd` (2026-10-04) o módulo e o teste `tests/test_mask_finalization.py` foram **excluídos** e substituídos por `src/data/source_decision.py`, que apenas registra a decisão manual. Consequência para a metodologia: **não existe "máscara final" consolidada**; o estágio `06` consome diretamente `data/interim/<fonte>/mask_<fonte>_<região>_<ano>.tif` da fonte escolhida.
2. **Comparação de máscaras com referência fixa** foi substituída por comparação de **todos os pares** de fontes (`dfefc42`), com métricas direcionais relativas à primeira fonte do par.
3. **Split inicialmente com `KMeans` do scikit-learn** foi reimplementado em numpy puro (`cd2ba16` → `941a051`).
4. **`bbox` via helpers de janela do rasterio** foi substituída por aritmética direta da transformação afim (`2001aef`).
5. **Duplicação de prólogo entre notebooks** foi centralizada em `src/setup.py` (`initialize`) e a instalação de dependências em `install_runtime_requirements` (`69240b2`).

### 1.6 Limitação de evidência experimental nos notebooks

Os 17 notebooks estão versionados **com todas as saídas limpas** (`outputs: []` em todas as células de código) `[calculado]`. O CI tem um job dedicado que falha se houver outputs (`notebook-conventions`, `.opencode/tools/scripts/check_notebook.py`) `[código]`. Portanto:

- não há logs de execução, métricas impressas, contagens de patches ou gráficos versionados nos notebooks;
- os únicos números de execução presentes no repositório são os documentados em `METODOLOGIA.md`/`RESULTADOS.md` (commit `4564e35`), que declaram proveniência em artefatos do Drive/Hugging Face Hub;
- qualquer valor numérico de execução não coberto por esses documentos é `[NÃO ENCONTRADO NO REPOSITÓRIO]`.

### 1.7 Baseline histórico

`PLAN.md`, `README.md` e `AGENTS.md` afirmam que o primeiro treino da U-Net foi preservado como `emater_v1` (pesos/métricas/histórico), com metadados em `data/processed/archive_v1/`, figuras em `artifacts/figures/archive_v1/` e um `BASELINE.json` registrando a limitação de `bbox`/split `[código: documentação]`. Esses arquivos **não estão versionados** e não existem na árvore local — a existência deles é `[NÃO ENCONTRADO NO REPOSITÓRIO]` (verificável apenas no Drive).

---

## 2. Ambiente Computacional, Arquitetura de Software e Reprodutibilidade

### 2.1 Estrutura de diretórios do repositório

```text
tcc-lab/
├── AGENTS.md                     # regras de código, formatação e infraestrutura
├── PLAN.md                       # roteiro por estágio (17 notebooks)
├── README.md
├── pyproject.toml                # uv + ruff + mypy + pytest
├── uv.lock                       # lockfile commitado
├── requirements-runtime.txt      # pins de runtime instalados por notebook
├── src/                          # pacote compartilhado (lógica reutilizável)
│   ├── config.py  config.yaml    # fonte única de configuração
│   ├── bootstrap.py              # entrega de src/ + data/external/ ao runtime
│   ├── setup.py  io.py  utils.py # prólogo, Drive/plataforma, segredos/sementes
│   ├── data/                     # aquisição, máscaras, patches, split, dataset
│   ├── models/{unet,segformer}.py
│   ├── losses.py  metrics.py
│   ├── trainer.py                # protocolo único de treino
│   ├── evaluation.py  comparative.py
│   ├── xai/{common,gradcam,attention_rollout}.py
│   └── synthesis.py  release.py
├── notebooks/                    # 17 estágios 00–16 (outputs limpos)
├── tests/                        # 26 arquivos, 239 testes de src/
├── scripts/                      # utilitários locais (tokens OAuth, Kaggle CLI)
├── secrets/                      # templates versionados; tokens reais git-ignored
├── data/external/                # VERSIONADO: malha IBGE + GeoJSONs Emater
├── data/{raw,interim,processed}/ # espelho git-ignored de MyDrive/tcc/data
├── models/  artifacts/           # espelhos git-ignored de MyDrive/tcc/
├── tcc-docs/                     # submódulo Git somente leitura
├── .github/workflows/ci.yml
└── .opencode/                    # validador de convenções de notebook e skills
```

### 2.2 Responsabilidade de cada módulo de `src/`

| Módulo | Responsabilidade (docstring + uso real) |
| --- | --- |
| `config.py` | Carrega `config.yaml` **sem cache** e valida a chave de topo `storage`; expõe `CONFIG_PATH` |
| `bootstrap.py` | Somente stdlib: baixa o tarball de `main` de `github.com/jotap1101/tcc-lab`, extrai `src/`, `data/external/`, `requirements-runtime.txt`, grava `src/.tcc_commit` (API do GitHub com fallback Atom), limpa `sys.modules` e cria o espelho `MyDrive/tcc/repo/` no Colab |
| `io.py` | `detect_platform` (delega ao bootstrap), `DriveClient` (Drive API v3), `mount_drive`, `ensure_storage_root`, `resolve_storage_paths`, `path_exists`, `ensure_local_copy`, `persist_bytes/json/file`, `relocate_exported_file`, `sync_repo_to_workspace` |
| `setup.py` | `RuntimeContext`/`initialize`; `install_runtime_requirements` (pip pinado + reinício de kernel quando ABI diverge); `remove_incompatible_torchaudio` |
| `utils.py` | `get_secret` (env → Kaggle Secrets → dataset privado), `set_all_seeds`, `set_deterministic_flags`, `setup_reproducibility`, `log_environment`, `check_dependencies`, `print_summary`, `authenticate_gee` |
| `data/gee_client.py` | AOI IBGE, coleção Sentinel-2, máscara QA60, mosaico mediano, exportação GeoTIFF, espera de task, miniatura RGB |
| `data/mask_utils.py` | Ano por fonte, nome/caminho canônico de máscara, construtores GEE (MapBiomas/AlphaEarth), exportação e miniatura |
| `data/emater_masks.py` | Carga/concatenação dos GeoJSONs, reprojeção, `buffer(0)`, grade de 10 m, rasterização e GeoTIFF `uint8` |
| `data/mask_comparison.py` | Grid comum, áreas, matriz de confusão 2×2, acordo global/IoU/F1/precisão/recall/kappa, figura |
| `data/source_decision.py` | Lê/valida `chosen_source`, carrega o relatório do estágio 03, persiste o relatório de auditoria |
| `data/preprocessing.py` | Alinhamento ao grid da máscara (bilinear, NaN), `rho = clip(DN·1e-4, 0, 1)`, GeoTIFF, verificação e figura |
| `data/patch_generation.py` | Corte 512×512 sem sobreposição, filtros, `.npy`, manifesto Parquet, fingerprint, montagem de amostras |
| `data/spatial_split.py` | k-means++ determinístico em numpy sobre centroides de bbox, rótulos, metadata do split, figura |
| `data/eda.py` | Estatísticas por banda (média, desvio, fração NaN) sobre todos os patches, sanidade, figura |
| `data/dataset.py` | `PatchDataset`, imputação+padronização, rótulo `-1`, `fold_indices`, `build_loaders`, cópias locais no Kaggle |
| `data/augmentations.py` | Rot90/hflip/vflip no par (imagem, máscara) com `torch.Generator` próprio |
| `data/raster_utils.py` | `same_grid`, `percentile_stretch`, `rgb_composite`, `render_figure` (backend Agg → PNG) |
| `losses.py` | `dice_loss`, `focal_loss`, `signed_distance_map`, `boundary_loss`, `SegmentationLoss` |
| `metrics.py` | TP/FP/FN acumulados, IoU, F1, precisão, recall com fallback 1.0 |
| `trainer.py` | `train_fold` (protocolo único), persistência, fingerprints, `config_hash`, `run_metadata` |
| `models/unet.py` | `DoubleConv`, `UNet`, `build_unet` |
| `models/segformer.py` | Carga `SegformerForSemanticSegmentation`, `num_labels=1`, adaptação 3→4 canais, `SegFormerBinary` |
| `evaluation.py` | Inferência por dobra, predições `.npz`, métricas por patch `.parquet`, resumo micro/macro |
| `comparative.py` | Emparelhamento por `patch_id`, bootstrap, t pareado, Wilcoxon, Holm, `d_z`, delta de Cliff, figura e mapas de erro |
| `xai/common.py` | Seleção determinística de patches, carga normalizada, RGB, fingerprints de XAI |
| `xai/gradcam.py` | Hook em `encoders[-1]`, score na região de café, mapa normalizado, figura |
| `xai/attention_rollout.py` | Extração `eager`/`output_attentions`, rollout da última etapa, mapa normalizado, figura |
| `synthesis.py` | 6 tabelas + `xai`, 3 figuras, fingerprint das entradas |
| `release.py` | Empacotamento e publicação no Hugging Face Hub, cards, `model_config.json`, `release.json` |

### 2.3 Notebooks

| Notebook | Células | Notebook | Células |
| --- | ---: | --- | ---: |
| `00_setup_environment` | 13 | `09_train_unet` | 19 |
| `01_gee_sentinel2_acquisition` | 25 | `10_train_segformer` | 19 |
| `02_gee_reference_masks` | 29 | `11_evaluation` | 19 |
| `03_mask_sources_comparison` | 23 | `12_comparative_analysis` | 27 |
| `04_source_decision` | 19 | `13_xai_gradcam_unet` | 17 |
| `05_preprocessing` | 21 | `14_xai_attention_rollout_segformer` | 17 |
| `06_patch_generation` | 19 | `15_results_synthesis` | 19 |
| `07_spatial_kfold_split` | 19 | `16_export_release` | 19 |
| `08_dataset_eda` | 21 | | |

Todos têm `kernelspec.name = python3` e `language_info.version = 3.10` no metadata (exceto `04`, sem `kernelspec`) `[calculado]`.

### 2.4 Versões de Python e dependências

| Item | Valor | Proveniência |
| --- | --- | --- |
| `requires-python` | `>=3.10` | `pyproject.toml` `[código]` |
| `ruff.target-version` | `py310` | `pyproject.toml` `[código]` |
| `mypy.python_version` | `3.12` | `pyproject.toml` `[código]` |
| Python do kernel nos notebooks | 3.10 | metadata dos `.ipynb` `[calculado]` |
| Python do `.venv` local de desenvolvimento | 3.14.4 | `python -V` no `.venv` `[calculado]` |
| Python usado no Colab/Kaggle | não fixado pelo repositório | `[NÃO ENCONTRADO NO REPOSITÓRIO]` (só `>=3.10` é exigido) |

Versões de runtime pinadas em `requirements-runtime.txt`, instaladas pela célula de dependências de **todos** os notebooks via `src.setup.install_runtime_requirements()` `[código]`:

| Pacote | Versão | Pacote | Versão |
| --- | ---: | --- | ---: |
| `torch` | 2.14.0 | `numpy` | 2.5.3 |
| `torchvision` | 0.29.0 | `pandas` | 3.0.6 |
| `transformers` | 5.17.0 | `pyarrow` | 25.0.1 |
| `earthengine-api` | 1.7.43 | `scipy` | 1.18.1 |
| `geopandas` | 1.1.4 | `scikit-learn` | 1.9.1 |
| `pyogrio` | 0.13.0 | `matplotlib` | 3.11.2 |
| `huggingface-hub` | 1.32.0 | `seaborn` | 0.13.2 |
| `rasterio` | 1.5.1 | `pyyaml` | 6.0.3 |
| `google-api-python-client` | 2.200.0 | `google-auth-oauthlib` | 1.4.1 |

O comentário do arquivo declara que os pins correspondem às versões mais recentes no PyPI em **setembro de 2026**, suportando Python 3.10+ `[código]`. Dependências de desenvolvimento em `pyproject.toml`: `ruff`, `mypy`, `pytest`, `kaggle>=1.7.4.5` `[código]`.

### 2.5 Sementes e determinismo

Semente global: `reproducibility.seed = 42` `[config]`.

```python
# trecho de src/utils.py
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=warn_only)
```

| Escopo | Semente / regra | Proveniência |
| --- | --- | --- |
| Global (notebooks) | `42` | `utils.setup_reproducibility` `[código]` |
| Por dobra (pesos e `DataLoader`) | `42 + fold` | `trainer.train_fold`; notebooks `09`/`10` fixam `set_all_seeds(ctx.seed + fold)` antes de construir o modelo `[código]` |
| Aumentações | `42 + fold + época` | `augmentation.set_seed(fold_seed + epoch)` `[código]` |
| Split espacial (k-means++) | `42` | `spatial_split.split_seed` `[config]` |
| Bootstrap estatístico | `42 + índice da métrica` (IoU 42, F1 43, Precisão 44, Recall 45) | `comparative.paired_comparison` `[código]` |
| Amostragem da EDA/sanidade | `random_state=0` | `eda._deterministic_sample` `[código]` |

Regras de determinismo adicionais: `training.num_workers = 0` (determinismo bit-a-bit da carga e das aumentações) `[config]`; estágios `10` e `11` usam `warn_only=True`; a configuração efetiva é registrada em `run.meta.json` nos campos `deterministic_algorithms.enabled` e `deterministic_algorithms.warn_only` `[código]`. O determinismo é de **protocolo**, não bit-a-bit entre hardwares distintos (documentado em `AGENTS.md`/`PLAN.md`).

### 2.6 Infraestrutura de execução

| Item | Valor | Proveniência |
| --- | --- | --- |
| Plataformas-alvo | Google Colab e Kaggle (Kaggle obrigatório); `local` apenas desenvolvimento | `AGENTS.md`, `bootstrap.detect_platform` `[código]` |
| Detecção de plataforma | `KAGGLE_KERNEL_RUN_TYPE` tem prioridade sobre `google.colab` | `bootstrap.py` `[código]` |
| Armazenamento canônico | `MyDrive/tcc/` (`storage.drive_root = "tcc"`) | `config.yaml` `[config]` |
| Colab | montagem nativa em `/content/drive/MyDrive` | `io.mount_drive` `[código]` |
| Kaggle | Drive API v3 + cache local em `/kaggle/working/drive`; upload resumível (`MediaFileUpload`) | `io.DriveClient` `[código]` |
| Local | `DRIVE_ROOT` ou `~/MyDrive` | `io.mount_drive` `[código]` |
| GPU | obrigatória para treino/avaliação: os notebooks `09`, `10`, `11` lançam `RuntimeError` se `torch.cuda.is_available()` for falso; mensagem cita "T4/P100" | notebooks `[código]` |
| GPU efetivamente usada nas execuções | `[NÃO ENCONTRADO NO REPOSITÓRIO]` (o `run.meta.json` registra plataforma, não o modelo da GPU) | — |
| Precisão numérica | `float32`, **sem AMP/FP16** (`grep` por `autocast`/`GradScaler` não retorna nada) | `[código]` |
| GEE | projeto Cloud `articulate-case-508513-m4`; OAuth interativo ou credencial headless em `GEE_CREDENTIALS` | `config.yaml`, `utils.authenticate_gee` `[config]` |
| Segredos | `GDRIVE_TOKEN`/`GDRIVE_TOKEN_FILE`, `GEE_CREDENTIALS`, `GEE_PROJECT`, `HF_TOKEN`; fallback no dataset privado `jotap1101kaggle/tcc-secrets` | `utils.get_secret` `[código]` |

### 2.7 Idempotência e fingerprints

| Fingerprint | Conteúdo | Implementação |
| --- | --- | --- |
| Manifesto | `schema_version=2`, `region_code`, `source_name`, `year`, bandas, `patch_size`, `coffee_min_ratio`, `min_valid_fraction`, `mask_source`, `tile_id`, nome/tamanho do composite e da máscara → SHA-256 truncado em 12 hex | `patch_generation.manifest_fingerprint` / `patch_fingerprint_hash` |
| Split | `schema_version=2`, `source_name`, fingerprint do manifesto, `bbox_schema_version=1`, `fold_count`, `seed`; hash dos pares `patch_id:fold` | `spatial_split.split_meta_payload` / `split_fingerprint_hash` |
| Configuração | SHA-256 dos bytes de `config.yaml`, 12 hex | `trainer.config_hash` |
| Pesos | SHA-256 do arquivo `fold_<i>.pt`, 12 hex | `evaluation._weights_digest` |
| Avaliação | `schema_version=1`, `threshold`, fingerprint do manifesto, do split, hash da configuração e hash dos pesos | `evaluation.evaluation_fingerprint` |
| Comparação | fingerprints e hashes de pesos do estágio 11 dos dois modelos | `comparative.comparison_fingerprint` |
| XAI | fingerprint do estágio 11 + hashes de pesos das dobras usadas + lista de patches + escopo do método | `gradcam.gradcam_fingerprint`, `attention_rollout.attention_rollout_fingerprint` |
| Síntese | manifestos, resumos do estágio 11, digests de métricas/históricos, comparação e metadados/mapas de XAI | `synthesis.synthesis_fingerprint` |
| Release | fingerprints de dataset, pesos e configuração + presença dos arquivos remotos esperados | `release.release_fingerprint` |

Regra de reuso: artefatos são reaproveitados **apenas** quando todos os componentes do fingerprint coincidem; caso contrário, são regenerados no lugar (nunca há sobrescrita silenciosa de pesos de outro dataset). `force=True` está disponível em `train_fold`, `evaluate_fold`, `run_gradcam`, `run_attention_rollout`, `analyze_comparison` e `run_release` `[código]`.

### 2.8 Qualidade, testes e CI

| Gate | Comando | Escopo |
| --- | --- | --- |
| Lint | `uv run ruff check src tests` | `src/`, `tests/` |
| Formatação | `uv run ruff format --check src tests` | `src/`, `tests/` |
| Tipagem | `uv run mypy src tests` | `src/`, `tests/` |
| Testes | `uv run pytest tests` | `src/` |
| Convenções de notebook | `python .opencode/tools/scripts/check_notebook.py` | `notebooks/*.ipynb` |

O validador de notebooks verifica: nome `^\d{2}_[a-z0-9]+(?:_[a-z0-9]+)*\.ipynb$`; **markdown imediatamente acima de cada célula de código**; ausência de células de código vazias; **ausência de outputs versionados** `[código]`. Não há cobertura de testes para notebooks (por decisão de arquitetura).

---

## 3. Área de Estudo, Aquisição Orbital e Curadoria de Máscaras (_Ground Truth_)

### 3.1 Arquivos vetoriais e filtro espacial

| Item | Valor | Proveniência |
| --- | --- | --- |
| Malha | `data/external/ibge/mg_rg_immediatas_2025/mg_rg_immediatas_2025.shp` (Malha das Regiões Geográficas Imediatas, IBGE 2025) | `config.yaml` `[config]` |
| CRS de origem | `EPSG:4674` (SIRGAS 2000 geográfico) | shapefile `.prj` `[calculado]` |
| CRS para o Earth Engine | `EPSG:4326` (reprojeção em `load_aoi_gdf`) | `gee_client.py` `[código]` |
| CRS de processamento raster | `EPSG:31983` (SIRGAS 2000 / UTM 23S) | `config.yaml` `[config]` |
| Filtro | `CD_RGI == "310044"`, exigindo **exatamente 1** feição (erro caso contrário) | `gee_client.load_aoi_gdf` `[código]` |
| Feições na malha | 70 RGI de MG | `[calculado]` |
| Atributos relevantes | `CD_RGI`, `NM_RGI`, `CD_RGINT`, `NM_RGINT`, `CD_UF`, `NM_UF`, `SIGLA_UF`, `CD_REGIAO`, `NM_REGIAO`, `SIGLA_RG`, `AREA_KM2` | `[calculado]` |
| AOI selecionado | `NM_RGI = "Guaxupé"`, `CD_RGI = "310044"`, `CD_RGINT = "3108"` | `[calculado]` |
| Área oficial do atributo | `AREA_KM2 = 2814.869` km² | `[calculado]` (campo do shapefile) |
| Área recalculada em `EPSG:31983` | ≈ 2814,42 km² | `[calculado]` (diferença de ~0,02% em relação ao atributo, por projeção) |
| Bounding box em `EPSG:31983` | x ∈ [290.982,44; 375.422,06] m; y ∈ [7.611.885,91; 7.681.408,66] m | `[calculado]` |
| Bounding box em `EPSG:4326` | (-47,016247; -21,589315) a (-46,201407; -20,962017) | `[calculado]` |

Não há filtro adicional por município, declividade, altitude ou máscara de bioma: o AOI é o polígono integral da RGI `[código]`.

### 3.2 Consulta no Google Earth Engine — parâmetros exatos

Construção da coleção (`gee_client.build_s2_collection`) `[código]`:

```python
ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(aoi)
  .filterDate(start, end)                              # YYYY-01-01 a YYYY-12-31
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
  .map(_mask_s2_clouds)                                # QA60 bits 10 e 11
  .select(["B2", "B3", "B4", "B8"])
```

| Parâmetro | Valor | Proveniência |
| --- | --- | --- |
| Coleção | `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Nível-2A harmonizado) | `config.yaml` `[config]` |
| Bandas | `B2`, `B3`, `B4`, `B8` (azul, verde, vermelho, NIR), 10 m nativas | `config.yaml` `[config]` |
| Intervalo de datas | ano civil completo de cada fonte: `2018`, `2024`, `2025` (`YYYY-01-01`–`YYYY-12-31`); fallback configurado `2023-01-01`–`2023-12-31` | `gee_client.year_dates`, `config.yaml` `[código]` |
| Estação | **não há restrição sazonal** (nem "estação seca") | `[código]` |
| Filtro de nuvem por cena | `CLOUDY_PIXEL_PERCENTAGE < 20` | `config.yaml` `[config]` |
| Máscara de nuvem por pixel | `QA60` com `bitwiseAnd((1<<10) \| (1<<11)).eq(0)` (bits 10 = nuvem opaca, 11 = cirros) → `updateMask` | `gee_client.py` `[código]` |
| Composição temporal | `collection.median()` por pixel | `gee_client.build_annual_mosaic` `[código]` |
| Recorte | `.clip(aoi)` | `[código]` |
| Escala de exportação | 10 m | `config.yaml` `[config]` |
| `maxPixels` | `100000000` (1e8; acima dos ~58,7 M pixels do AOI) | `config.yaml` `[config]` |
| CRS de exportação | `EPSG:31983` | `config.yaml` `[config]` |
| Reamostragem | `nearest` (padrão; `Image.resample()` só é chamado para `bilinear`/`bicubic`) | `gee_client.export_image_to_drive` `[código]` |
| Formato | GeoTIFF via `ee.batch.Export.image.toDrive` | `[código]` |
| Espera da task | polling de 30 s, timeout de 7200 s; estados `READY/PENDING/RUNNING`; `SUCCEEDED` normalizado para `COMPLETED` | `gee_client.wait_for_task` `[código]` |
| Nome canônico | `sentinel2_<CD_RGI>_<YYYY-01-01>_<YYYY-12-31>.tif` em `data/raw/sentinel2/` | `gee_client.mosaic_file_name` `[código]` |

Detalhe de infraestrutura: o GEE exporta para **uma única pasta** na raiz do Drive (o parâmetro `folder` não aceita `/`; há validação explícita), e `io.relocate_exported_file` move o arquivo para o caminho canônico — no Colab com polling de até 600 s aguardando a propagação do mount `[código]`.

### 3.3 Fontes de _ground truth_ e regras de binarização

| Fonte | Ano | Natureza | Regra exata | Resolução nativa | Origem | Proveniência |
| --- | ---: | --- | --- | ---: | --- | --- |
| **MapBiomas** | 2025 | raster categórico GEE | `classification == 46` (classe 3.2.2.1 – Café), coleção filtrada por `collection_id == 11` e `year == 2025` | 30 m (reamostrada para 10 m) | `projects/mapbiomas-public/assets/brazil/lulc/v1` | `config.yaml`, `mask_utils.build_mapbiomas_mask` `[código]` |
| **AlphaEarth (FDaP)** | 2024 | raster probabilístico GEE | `probability >= 0.28`, mosaico do ano | 10 m | `projects/forestdatapartnership/assets/coffee/model_2025b` | `config.yaml`, `mask_utils.build_alphaearth_mask` `[código]` |
| **Emater-MG** | 2018 | vetor local (GeoJSON) | rasterização das glebas (`fill=0`, `all_touched=False`) recortada pelo AOI | 10 m | Geoportal do Café, `data/external/emater/` | `config.yaml`, `emater_masks.py` `[código]` |
| Todas | — | — | `enabled: true` para as três; fonte de treino escolhida manualmente: `emater` | — | — | `config.yaml` `[config]` |

O limiar da AlphaEarth entra no nome do arquivo (`_t028`, via `mask_file_name`), de modo que recalibrar o limiar produz um novo GeoTIFF em vez de reutilizar a máscara antiga `[código]`. Os limiares testados no notebook `02` para calibrar a AlphaEarth são `[0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50]` (célula de estimativa de área por limiar) `[código]`.

Caminho canônico da máscara `[código]`:

```text
data/interim/<fonte>/mask_<fonte>_310044_<ano>[_t<limiar>].tif
```

### 3.4 Rasterização e alinhamento da máscara vetorial da Emater

Etapas exatas de `emater_masks.ensure_emater_mask` `[código]`:

1. `_load_geojson_paths`: união de `*.geojson` e `*.json` no diretório, ordenados; erro se vazio.
2. `load_emater_features`: `geopandas.read_file` por arquivo, adiciona `municipio` (do nome do arquivo) e concatena com `crs="EPSG:4326"`.
3. Reprojeção para `EPSG:31983` (`to_crs`).
4. Reparo de geometrias inválidas com `geometry.buffer(0)`.
5. `_aoi_grid`: grade de 10 m alinhada à escala, com origem e extensão arredondadas:
   `x0 = floor(minx/scale)*scale`, `y0 = floor(miny/scale)*scale`, `x1 = ceil(maxx/scale)*scale`, `y1 = ceil(maxy/scale)*scale`; `transform = from_origin(x0, y1, scale, scale)`; `height = round((y1-y0)/scale)`, `width = round((x1-x0)/scale)`.
6. `rasterize_emater_mask`: rasteriza as glebas com valor 1, multiplica pela rasterização do polígono do AOI (clip booleano), dtype `uint8`.
7. `_write_mask_geotiff`: GTiff `uint8`, 1 banda, `crs=EPSG:31983`, `compress="deflate"`.

Dados dos insumos vetoriais, verificados nos arquivos `[calculado]`:

| Item | Valor |
| --- | --- |
| Arquivos GeoJSON | 9 (`Arceburgo`, `Cabo Verde`, `Guaranésia`, `Guaxupé`, `Juruaia`, `Monte Belo`, `Muzambinho`, `Nova Resende`, `São Pedro da União`) |
| Polígonos no conjunto | **34.985** |
| Área total sem clip ao AOI (reprojetada em `EPSG:31983`) | ≈ 604,52 km² |
| Área de café após clip/rasterização, reportada nos artefatos | 603,2446 km² `[doc 4564e35]` |
| Atributos dos GeoJSON | `id`, `area`, `altitude`, `variedade`, `espacamento`, `anoplantio`, `corfundo`, `opfundo`, `opborda`, `zindex`, `layer`, `visible`, `sel`, `geometry`; `variedade = "Café"` |
| Ano declarado | 2018 (os arquivos não têm atributo de ano; o ano é declarado em `config.yaml`) |

**Não há pós-processamento geométrico** (abertura/fechamento morfológico, remoção de componentes menores, suavização) em nenhuma fonte `[código]`. As únicas operações geométricas são: `buffer(0)` para reparo de topologia e recorte ao AOI. O módulo que materializava uma "máscara final" foi removido (Seção 1.5).

### 3.5 Comparação e finalização das máscaras

O estágio `03` carrega todas as máscaras em um grid comum (a **primeira fonte** define o grid; as demais são reprojetadas com `Resampling.nearest`) e calcula, para cada fonte, área (`pixels × pixel_size² / 1e6`) e fração de café na grade. Para cada par (`itertools.combinations`), monta a matriz 2×2 `both_coffee`, `only_reference`, `only_other`, `both_non_coffee` e deriva `[código]`:

$$
\text{Acordo} = \frac{both + neither}{total},\qquad
IoU = \frac{both}{both + only_{ref} + only_{other}},\qquad
F1 = \frac{2\,both}{2\,both + only_{ref} + only_{other}}
$$

$$
\text{Precisão} = \frac{both}{both + only_{ref}},\qquad
\text{Revocação} = \frac{both}{both + only_{other}},\qquad
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

com `p_e` calculado das marginais da tabela 2×2 (`mask_comparison._cohen_kappa`) `[código]`. Não há eleição automática de fonte: o estágio `04` apenas lê `ground_truth.chosen_source` (validando que a fonte está habilitada) e grava `artifacts/metrics/ground_truth/source_decision_310044.json` com `decision: "manual"` e a base métrica dos pares que envolvem a fonte escolhida `[código]`.

Artefatos `[código]`:

```text
artifacts/metrics/ground_truth/mask_sources_comparison_310044.json
artifacts/metrics/ground_truth/source_decision_310044.json
artifacts/figures/mask_sources_comparison_310044.png
```

Valores efetivos documentados `[doc 4564e35]` (proveniência declarada: `artifacts/metrics/ground_truth/mask_sources_comparison_310044.json`; **não verificáveis no repositório**):

| Fonte | Área de café (km²) | Fração da grade |
| --- | ---: | ---: |
| MapBiomas 2025 | 471,9503 | 0,0804 |
| AlphaEarth 2024 | 542,2271 | 0,0923 |
| Emater 2018 | 603,2446 | 0,1027 |

Grade de referência declarada: `6953 × 8445` pixels, `EPSG:31983`, 10 m, `58.718.085` pixels (`5.871,8085 km²`). Concordância: MapBiomas×AlphaEarth IoU 0,4358 / κ 0,5701; MapBiomas×Emater IoU 0,5677 / κ 0,6969; AlphaEarth×Emater IoU 0,4641 / κ 0,5945 `[doc 4564e35]`.

Observação de auditoria: a grade `6953 × 8445` é compatível com o cálculo de `_aoi_grid` a partir dos limites do AOI (84.440 m × 69.523 m a 10 m) `[calculado]`. O `METODOLOGIA.md` (Seção 5.1) afirma que o composite da Emater tem `8192 × 6656` pixels; como `preprocessing._composite_profile` **sempre** grava o composite com `height/width/transform/crs` da máscara de referência, o composite tem a dimensão da máscara. Essa divergência só pode ser resolvida inspecionando os GeoTIFFs no Drive — registrar como pendência de verificação documental.

---

## 4. Pré-processamento, Geração de _Patches_ e Validação Cruzada Espacial

### 4.1 Normalização radiométrica

O composite é construído por ano e normalizado em **duas etapas distintas** `[código]`:

1. **Conversão radiométrica (reflectância de superfície)** em `preprocessing.normalize_reflectance`:

$$
\rho_{b,i} = \mathrm{clip}\!\left(DN_{b,i} \times 10^{-4},\; 0,\; 1\right)
$$

com `REFLECTANCE_SCALE = 1e-4` e dtype `float32`. `NaN` é preservado (propaga pelo `clip` e pela multiplicação) e marca pixels sem observação válida.

2. **Padronização estatística (z-score por banda)** em `dataset.normalize_patch`, aplicada **no carregamento de cada patch** (não é gravada no GeoTIFF):

$$
\tilde{x}_{b,i} = \frac{x_{b,i} - \mu_b}{\sigma_b},\qquad
\mu_b = \frac{\sum_i x_{b,i}}{N_b},\qquad
\sigma_b = \sqrt{\frac{\sum_i x_{b,i}^2}{N_b} - \mu_b^2}
$$

Regras adicionais implementadas `[código]`:

- antes da padronização, `NaN` é **imputado com a média da banda** (`np.where(finite, values, mean)`);
- banda com `mean`/`std` ausente (`None`, banda toda `NaN`) ou `std == 0.0` é **zerada**;
- após a normalização, os pixels inválidos têm valor 0 na imagem, mas recebem rótulo `-1` na máscara e são excluídos da perda e das métricas;
- as estatísticas são calculadas sobre **todos os patches do manifesto da fonte** (treino + validação), não por dobra.

Estatísticas por banda documentadas para a fonte Emater `[doc 4564e35]`:

| Banda | Média | Desvio padrão | Fração de `NaN` |
| --- | ---: | ---: | ---: |
| B2 | 0,039683 | 0,015657 | 15,26% |
| B3 | 0,069281 | 0,023501 | 15,26% |
| B4 | 0,061667 | 0,034175 | 15,26% |
| B8 | 0,321997 | 0,061097 | 15,26% |

### 4.2 Alinhamento do mosaico ao grid da máscara

`preprocessing.load_aligned_mosaic` `[código]`:

- se `same_grid(mosaico, máscara)` (mesma altura, largura, `transform` e `crs`), lê os valores diretamente e marca `aligned=True`;
- caso contrário, reprojeta **banda a banda** com `rasterio.warp.reproject`, `resampling=Resampling.bilinear`, `src_nodata=None`, `dst_nodata=np.nan`, para o grid exato da máscara (`dst_transform`/`dst_crs`), marcando `aligned=False`;
- o perfil de escrita é sempre o da máscara: `GTiff`, `float32`, `nodata=NaN`, `compress="deflate"`, `predictor=3`, `tiled=True`, blocos `256×256`.

### 4.3 Geração de _patches_

`patch_generation.generate_patches` `[código]`:

| Item | Valor |
| --- | --- |
| Tamanho do patch | `512 × 512` px (`data.patch_size = 512`) ⇒ 5.120 m × 5.120 m = **26,2144 km²** |
| Sobreposição (_stride_) | **0** — janelas `Window(col*512, row*512, 512, 512)` |
| Bordas parciais | descartadas: `rows = height // 512`, `cols = width // 512` (`grid_dims`) |
| Validação de grid | `_check_same_grid` exige composite e máscara no mesmo grid (erro se não) |
| Pixels válidos | `valid = np.isfinite(image_win).all(axis=0)` (finito em **todas** as bandas) |
| `valid_fraction` | média de `valid` no patch |
| Descarte por invalidez | `valid_fraction <= 0` **ou** `valid_fraction < data.min_valid_fraction` (`0.10`) |
| `coffee_ratio` | `mean(mask_win[valid] > 0)` — medido **apenas na área válida** do patch |
| Descarte por café | `coffee_ratio < data.coffee_min_ratio` (`0.0`, portanto não descarta nenhum patch) |
| Formato persistido | `.npy` (`np.save`): imagem `(4, 512, 512)`; máscara `(512, 512)` |
| `patch_id` | `<tile_id>_r<row:04d>_c<col:04d>`, com `tile_id = composite_310044_<ano>` |
| `bbox` | `minx,miny,maxx,maxy` em `EPSG:31983`, calculada da transformação afim: `left = c + a·(col·S) + b·(row·S)` etc. |
| Manifesto | `data/processed/manifest_<fonte>.parquet` |
| Metadata | `data/processed/manifest_<fonte>.meta.json` com `fingerprint_hash`, `bbox_schema_version`, `grid.rows/cols`, `n_patches`, `discarded_invalid`, `min_valid_fraction` |
| Schema | `MANIFEST_SCHEMA_VERSION = 2`, `BBOX_SCHEMA_VERSION = 1` |
| Versionamento dos arquivos | `data/processed/patches/<fonte>/{images,masks}/<fingerprint_hash>/` (hash de 12 hex) |

Colunas do manifesto (verificadas no código e no `PLAN.md`): `patch_id`, `tile_id`, `fold` (`None` até o estágio 07), `row`, `col`, `bbox`, `coffee_ratio`, `valid_fraction`, `mask_source`, `image_path`, `mask_path` `[código]`.

### 4.4 Quantitativos reais do dataset

Como os notebooks estão sem outputs (Seção 1.6), os números abaixo são os **documentados** em `RESULTADOS.md`/`METODOLOGIA.md` (commit `4564e35`), derivados de `manifest_emater.parquet` e das tabelas de síntese (artefatos no Drive/Hugging Face Hub) `[doc 4564e35]`:

| Estatística | Valor |
| --- | ---: |
| Grade de patches do composite (Emater) | 13 linhas × 16 colunas = **208 patches candidatos** (derivado de `height//512`, `width//512`) |
| Patches retidos no manifesto | **125** |
| Patches não retidos | **83** (compatível com o campo `discarded_invalid` do metadata; 208 − 125) |
| Patches com café (`coffee_ratio > 0`) | **124** |
| Patches sem café | **1** |
| Pixels totais (125 × 512²) | 32.768.000 |
| Pixels válidos | 27.767.193 (fração global 0,8474) |
| Pixels de café | 5.962.685 (≈ 596,27 km²) |
| `coffee_ratio` mín / médio / mediano / máx | 0,0000 / 0,2150 / 0,1986 / 0,5954 |
| `valid_fraction` mín / médio / mediano / máx | 0,1330 / 0,8474 / 1,0000 / 1,0000 |

Distribuição por dobra `[doc 4564e35]`:

| Dobra | Patches | Pixels válidos | `coffee_ratio` médio | Centroide aproximado (x, y) em m |
| ---: | ---: | ---: | ---: | --- |
| 0 | 18 | 3.792.117 | 0,1032 | (304.918, 7.641.019) |
| 1 | 21 | 4.655.682 | 0,1985 | (329.624, 7.661.296) |
| 2 | 25 | 5.605.627 | 0,2167 | (331.838, 7.639.733) |
| 3 | 30 | 6.308.562 | 0,2918 | (355.492, 7.661.613) |
| 4 | 31 | 7.405.205 | 0,2156 | (357.292, 7.630.623) |

Nota: o número de patches por dobra **não é balanceado** (18–31), o que é consequência do agrupamento espacial e deve ser declarado como característica do delineamento.

### 4.5 Particionamento espacial _k-fold_

Implementação em `spatial_split.py` `[código]`:

| Item | Valor |
| --- | --- |
| `k` | `splits.fold_count = 5` |
| Semente | `reproducibility.seed = 42` (`np.random.default_rng(seed)`) |
| Entrada do agrupamento | centroide de cada `bbox`: `x_c = (minx+maxx)/2`, `y_c = (miny+maxy)/2`, em metros (`EPSG:31983`) |
| Algoritmo | k-means determinístico em **numpy puro** (sem scikit-learn) |
| Inicialização | k-means++: 1º centro uniforme; seguintes com probabilidade ∝ distância² ao centro mais próximo; fallback uniforme se a soma das distâncias for 0/não finita |
| Iterações | até **100**; converge quando rótulos e centros não mudam (`array_equal` + `allclose`) |
| Reordenação dos rótulos | `np.lexsort((centers[:,1], centers[:,0]))` ⇒ dobras numeradas por x e depois y, estáveis e independentes da ordem do manifesto |
| Validação | erro se `n_patches < k` |
| Persistência | coluna `fold` regravada no manifesto + `data/processed/split_<fonte>.meta.json` |
| Schema | `SPLIT_SCHEMA_VERSION = 2`; invalida quando mudam fingerprint do manifesto, schema do split, `bbox_schema_version`, `fold_count` ou `seed` |
| Uso no treino | `fold_indices`: treino = `fold != f`, validação = `fold == f` (uma dobra inteira de fora) |

O split é recalculado automaticamente quando qualquer componente do metadata muda; a divisão espacial por blocos contíguos é o mecanismo adotado para reduzir vazamento por autocorrelação espacial entre treino e validação `[código]`.

### 4.6 Dataset e carregadores

`data/dataset.py` `[código]`:

- `PatchDataset.__getitem__`: carrega imagem e máscara `.npy`; calcula `valid = np.isfinite(image).all(axis=0)`; normaliza a imagem (imputação + z-score); máscara vira `np.where(valid, (mask > 0).astype(float32), -1.0)` e recebe `unsqueeze(0)` ⇒ tensor `(1, 512, 512)`; imagem ⇒ `(4, 512, 512)`;
- `build_loaders`: `train_loader` com `shuffle=True` e `generator=torch.Generator().manual_seed(seed)`; `val_loader` com `shuffle=False`; `batch_size` e `num_workers` de `config.yaml`; valida `manifest_has_folds`;
- `ensure_patches_local`: no Kaggle baixa uma única vez todos os patches do manifesto para o cache local; no Colab/local é no-op.

### 4.7 Aumentações de dados

`data/augmentations.py` `[código]` — apenas geométricas, aplicadas ao par (imagem, máscara) com as **mesmas** decisões, somente no treino:

| Transformação | Chamada | Probabilidade (`config.yaml`) |
| --- | --- | ---: |
| Rotação de 90° | `torch.rot90(k=1, dims=(1,2))` | `augmentation.rot90_prob = 0.25` |
| Flip horizontal | `torch.flip(dims=[2])` | `augmentation.hflip_prob = 0.50` |
| Flip vertical | `torch.flip(dims=[1])` | `augmentation.vflip_prob = 0.50` |

Ordem de aplicação: rotação → flip horizontal → flip vertical. As decisões usam `torch.rand(1, generator=self.generator)` com um `torch.Generator` próprio, inicializado com 0 e **reseedado a cada época** com `seed + fold + época` pelo `trainer` `[código]`. `augmentation.enabled = true` `[config]`. Não há aumentação fotométrica, ruído, corte, escala ou mixup `[código]`.

### 4.8 EDA e verificações de sanidade

`data/eda.py` `[código]`:

- percorre **todos** os patches do manifesto acumulando soma, soma dos quadrados, contagem de finitos e de `NaN` por banda; deriva `mean`, `std`, `nan_fraction`;
- persiste `data/processed/normalization_stats_<fonte>.json` com fingerprint (`EDA_SCHEMA_VERSION = 1`, bandas, fingerprint do manifesto);
- `run_sanity_checks` valida amostra determinística de **10 patches** (`random_state=0`): forma esperada `(4, 512, 512)`, dtype, máscara binária, intervalo de valores, fração de `NaN`, distribuição de café, presença/balanceamento de `fold` e unicidade de `patch_id`;
- figura `eda_summary_<região>_<ano>_<fonte>.png` com distribuição de café, patches por dobra, média ± desvio por banda e histogramas de reflectância.

---

## 5. Arquiteturas dos Modelos, Função de Perda e Protocolo de Treinamento

### 5.1 U-Net (implementação própria)

`models/unet.py` `[código]` — **não usa** `segmentation-models-pytorch` nem pesos pré-treinados: a rede é construída e treinada **do zero**.

| Componente | Definição |
| --- | --- |
| Bloco `DoubleConv` | `Conv2d(3×3, padding=1, bias=False)` → `BatchNorm2d` → `ReLU(inplace=True)` → `Conv2d(3×3, padding=1, bias=False)` → `BatchNorm2d` → `ReLU` |
| Codificador | 4 níveis com canais `model.unet_channels = [32, 64, 128, 256]`; cada nível seguido de `MaxPool2d(2, 2)` |
| Bottleneck | `DoubleConv(256 → 512)` |
| Decodificador | 4 níveis: `ConvTranspose2d(kernel=2, stride=2)` → concatenação com o skip → `DoubleConv(2C → C)` |
| Saída | `Conv2d(1×1)` para **1** canal de logit binário |
| Entrada | `len(data.bands) = 4` canais (B2, B3, B4, B8) |
| Parâmetros | **7.763.329** — recalculado nesta auditoria por instanciação (`sum(p.numel())`) `[calculado]`; coincide com `[doc 4564e35]` |
| Resoluções (512×512) | 512 → 256 → 128 → 64 (saída de `encoders[-1]`: 64×64, 256 canais); bottleneck 32×32 |
| Inicialização | padrão do PyTorch (sem transfer learning) |

### 5.2 SegFormer (Hugging Face Transformers)

`models/segformer.py` `[código]`:

| Item | Valor |
| --- | --- |
| Biblioteca | `transformers` (`SegformerForSemanticSegmentation`) |
| Variante | `model.segformer_variant = "nvidia/mit-b1"` (fixa na configuração; `build_segformer(variant=...)` permite sobrescrever) |
| Configuração da cabeça | `num_labels = 1`; `id2label = {0: "coffee"}`; `label2id = {"coffee": 0}` |
| Carga | `from_pretrained(variant, config=..., ignore_mismatched_sizes=True)`; checkpoint de **backbone** (ImageNet) ⇒ cabeça de decodificação inicializada aleatoriamente |
| Adaptação de entrada | `_adapt_input_channels`: substitui `stages[0].patch_embeddings.proj` por `Conv2d(4, out_channels, ...)`; copia os 3 filtros RGB, inicializa o 4º canal com `projection.weight.mean(dim=1, keepdim=True)` (média dos três), copia o viés; `model.config.num_channels = 4` |
| Decodificador | **All-MLP** padrão do `SegformerForSemanticSegmentation` (não é reimplementado) |
| Saída | logits em **1/4** da resolução; `SegFormerBinary.forward` reamostra com `F.interpolate(mode="bilinear", align_corners=False)` |
| Restrição de forma | `PATCH_MULTIPLE = 32` (stride total 4·2·2·2); erro se `H % 32 != 0` ou `W % 32 != 0`; 512 é múltiplo de 32 |
| Parâmetros | **13.680.641** — recalculado nesta auditoria `[calculado]`; coincide com `[doc 4564e35]` |
| Configuração MiT-B1 efetiva (lida do checkpoint) | `depths = [2, 2, 2, 2]`; `hidden_sizes = [64, 128, 320, 512]`; `num_attention_heads = [1, 2, 5, 8]`; `sr_ratios = [8, 4, 2, 1]`; `num_channels = 4` `[calculado]` |
| Última etapa (entrada 512) | 16×16 = 256 tokens, 2 camadas, 8 cabeças, atenção quadrada (`sr_ratio = 1`) |

**Divergência documental identificada.** O `METODOLOGIA.md` (Seção 4.15) lista 4 cabeças na etapa 3 do MiT-B1; a configuração efetiva do checkpoint `nvidia/mit-b1` é `[1, 2, 5, 8]` `[calculado]`. Para o Attention Rollout isso não altera nada (a etapa usada é a última, com 8 cabeças), mas o valor deve ser corrigido no texto.

### 5.3 Protocolo de comparação justa e assimetrias

Compartilhado pelos dois modelos `[código]`:

- mesmo conjunto de patches da fonte escolhida (`chosen_source`);
- mesmo split espacial (coluna `fold`);
- mesmas estatísticas de normalização;
- mesma perda (Dice + Focal + Boundary), mesmo otimizador (Adam), mesmo scheduler (`CosineAnnealingLR`), mesmas sementes por dobra, mesmas métricas e mesmo limiar (0,5);
- mesmas aumentações (`SegmentAugmentation`).

Assimetrias **não controladas** (devem ser declaradas como limitação) `[código]`:

1. U-Net treinada do zero; SegFormer com backbone pré-treinado em ImageNet;
2. melhor checkpoint escolhido e avaliado na **mesma** dobra de validação (viés otimista);
3. normalização calculada sobre todos os patches, incluindo os de validação (transdução leve, sem uso de rótulos).

### 5.4 Função de perda

`losses.py` `[código]`. Pesos em `config.yaml`: `dice_weight = 1.0`, `focal_weight = 1.0`, `boundary_weight = 1.0` ⇒ normalizador `3.0`.

$$
\mathcal{L} = \frac{w_D\mathcal{L}_{Dice} + w_F\mathcal{L}_{Focal} + w_B\mathcal{L}_{Boundary}}{w_D + w_F + w_B}
$$

Máscara de validade e alvo binário: `m_i = 1[t_i >= 0]`, `y_i = max(0, t_i)` (pixels `-1` viram 0 **e** são excluídos por `m_i`).

**Dice Loss** (suave, diferenciável, classe café):

$$
\mathcal{L}_{Dice} = 1 - \frac{2\sum_i p_i y_i m_i + \epsilon}{\sum_i p_i m_i + \sum_i y_i m_i + \epsilon},\qquad \epsilon = 1.0
$$

com `p_i = sigmoid(z_i)`. O cálculo é em nível de batch (soma sobre todos os pixels), não por imagem.

**Focal Loss**:

$$
\mathcal{L}_{Focal} = \frac{1}{\sum_i m_i}\sum_i m_i\,\alpha_{t,i}\,(1-p_{t,i})^{\gamma}\,\mathrm{BCE}(p_i, y_i)
$$

$$
p_{t,i} = p_i y_i + (1-p_i)(1-y_i),\qquad
\alpha_{t,i} = \alpha y_i + (1-\alpha)(1-y_i)
$$

Constantes exatas: `FOCAL_ALPHA = 0.25`, `FOCAL_GAMMA = 2.0`, `PROB_EPS = 1e-6` (clamp de `p` em `[1e-6, 1-1e-6]`), `denominator = max(Σm_i, 1)`.

**Boundary Loss** (Kervadec et al., mapa de distância assinada):

$$
\mathcal{L}_{Boundary} = \frac{\sum_i m_i D_i p_i}{\sum_i m_i |D_i|}
$$

- `D_i` é calculado em CPU com `scipy.ndimage.distance_transform_edt`: negativo dentro do objeto, positivo fora; cenas uniformes (objeto ausente ou preenchendo a cena) recebem mapa zero;
- se `Σ m_i|D_i| == 0`, a perda é `0`;
- o mapa `D` é constante em relação aos parâmetros; o gradiente flui apenas por `p = sigmoid(z)`;
- `forward` retorna `{"total", "dice", "focal", "boundary"}`; a validação usa os mesmos três termos.

### 5.5 Hiperparâmetros de treinamento

| Parâmetro | Valor | Proveniência |
| --- | --- | --- |
| Otimizador | `torch.optim.Adam` | `trainer.py` `[código]` |
| Learning rate inicial | `1.0e-4` | `config.yaml` `[config]` |
| Weight decay | `1.0e-4` | `config.yaml` `[config]` |
| Scheduler | `CosineAnnealingLR(T_max=epochs)` | `trainer.py` `[código]` |
| Épocas | `100` | `config.yaml` `[config]` |
| Batch size | `16` | `config.yaml` `[config]` |
| `num_workers` | `0` | `config.yaml` `[config]` |
| Precisão mista (AMP/FP16) | **não usada** (float32; sem `autocast`/`GradScaler`) | `[código]` |
| Gradient clipping | **não usado** (`clip_grad_*` ausente) | `[código]` |
| Early stopping | **não implementado** | `[código]` |
| Critério de melhor modelo | maior `val iou` na época; guarda uma cópia do `state_dict` (no dispositivo corrente) | `trainer.train_fold` `[código]` |
| Pesos salvos | estado do melhor epoch | `[código]` |
| Semente por dobra | `42 + fold` | `[código]` |
| Semente das aumentações | `42 + fold + época` | `[código]` |

### 5.6 Laço de treino por dobra

`trainer.train_fold` `[código]`, para `fold ∈ {0..4}`:

1. resolve a fonte (`source_name or chosen_source()`), lê os hiperparâmetros do config e calcula `fold_seed = seed + fold`;
2. calcula os fingerprints (`manifest_fingerprint_hash`, `split_fingerprint_hash`, `config_hash`) e **reutiliza** a dobra se os três coincidirem com o JSON de métricas salvo e se pesos/métricas/histórico existirem;
3. `set_all_seeds(fold_seed)`; constrói `SegmentAugmentation`, `build_loaders` e move o modelo para o dispositivo;
4. `SegmentationLoss(config)`, `Adam(lr, weight_decay)`, `CosineAnnealingLR(T_max=epochs)`;
5. para cada época: `augmentation.set_seed(fold_seed + epoch)`; captura `epoch_lr = optimizer.param_groups[0]["lr"]` **antes** do passo do scheduler; executa treino; `scheduler.step()`; executa validação; registra histórico; se `val_stats["iou"] > best_iou`, guarda `state_dict`, época e métricas;
6. restaura o melhor estado; persiste pesos (`fold_i.pt`), métricas (`fold_i.json`) e histórico (`fold_i.json`).

Métricas de treino/validação por época são derivadas de contagens acumuladas (`src/metrics.py`) e a perda reportada é a **média das perdas por lote** da época (`sums[key]/batches`). O histórico registra `epoch`, `lr` e os pares `train_*`/`val_*` para `loss`, `dice`, `focal`, `boundary`, `iou`, `f1`, `precision`, `recall` `[código]`.

### 5.7 Artefatos e metadata de execução

| Artefato | Caminho |
| --- | --- |
| Pesos por dobra | `models/<modelo>/<fonte>/fold_<i>.pt` |
| Métricas por dobra | `artifacts/metrics/<modelo>/<fonte>/fold_<i>.json` |
| Histórico por época | `artifacts/runs/<modelo>/<fonte>/fold_<i>.json` |
| Metadata da execução | `artifacts/runs/<modelo>/<fonte>/run.meta.json` |

O JSON de métricas registra `model`, `fold`, `epochs`, `seed`, `best_epoch`, os três fingerprints e as métricas de treino/validação do melhor epoch `[código]`. O `run.meta.json` registra `model`, `source_name`, `year`, `fold_count`, `platform`, `config_hash`, `commit_sha`, `seed`, `epochs`, `batch_size`, `lr`, `weight_decay`, `num_workers`, `model_config`, o estado do determinismo (`enabled`/`warn_only`), os fingerprints e as versões de `torch`, `torchvision`, `transformers`, `numpy`, `pandas` `[código]`.

---

## 6. Protocolo de Avaliação Quantitativa e Métodos de Explicabilidade (XAI)

### 6.1 Métricas de pixel efetivamente implementadas

`metrics.py` e `evaluation.py` `[código]`. A classe de interesse é **café** (positivo); `TP/FP/FN` são acumulados por lote e as métricas derivadas no fim (não há média de métricas por lote).

$$
IoU = \frac{TP}{TP+FP+FN},\qquad
F1 = \frac{2TP}{2TP+FP+FN},\qquad
Precisão = \frac{TP}{TP+FP},\qquad
Revocação = \frac{TP}{TP+FN}
$$

| Item | Valor |
| --- | --- |
| Limiar de binarização | **fixo em 0,5**: `ŷ = 1[sigmoid(z) ≥ 0.5]` (`DEFAULT_THRESHOLD = 0.5`) |
| Pixels ignorados | `targets >= 0` define validade; `-1` é excluído de TP/FP/FN |
| Denominador zero | métrica = **1,0** (classe vazia considerada perfeita), via `_safe_ratio(..., fallback=1.0)` |
| Agregação por patch | TP/FP/FN e as quatro métricas por patch, persistidos em Parquet |
| Agregação por dobra | soma de TP/FP/FN da dobra → métricas (micro dentro da dobra) |
| Agregado do modelo | soma de TP/FP/FN de todas as dobras → métricas **micro** |
| Agregação macro | média e desvio padrão **populacional** (`np.std`, `ddof=0`) das métricas das 5 dobras |
| Artefatos | `evaluation/fold_<i>.json`, `patch_metrics_fold_<i>.parquet`, `predictions_fold_<i>.npz` (`patch_ids`, `predictions` `uint8`), `summary.json` |
| Figura | métricas por dobra; histograma de IoU por patch com café; IoU × `coffee_ratio`; IoU × `valid_fraction` |

**Métricas solicitadas que NÃO existem no código:** `BIoU` (boundary IoU), acurácia global, F1 por classe não-café, matriz de confusão multiclasse, FLOPs, FPS e tempo por época `[NÃO ENCONTRADO NO REPOSITÓRIO]`. Não há medição de tempo em `trainer.py`/`evaluation.py` (nenhuma chamada a `time.perf_counter`/`time.time`) e nenhum contador de FLOPs. `RESULTADOS.md` (Seção 5) registra explicitamente "DADO NÃO ENCONTRADO" para GFLOPs e FPS.

### 6.2 Comparação estatística emparelhada (estágio 12)

`comparative.py` `[código]`:

| Item | Implementação exata |
| --- | --- |
| Emparelhamento | merge por `patch_id` (`validate="one_to_one"`), com validação de: conjuntos de patches idênticos, mesma dobra, mesmo `coffee_ratio` e mesma contagem de positivos reais (`TP+FN`) |
| Direção | `DIRECTION = "segformer_minus_unet"`; `d_i = m_i^SegFormer − m_i^U-Net` |
| Bootstrap | `BOOTSTRAP_ITERATIONS = 2000`, IC percentil de 95% (`2,5%` e `97,5%`), RNG `default_rng(seed)` com `seed = 42 + índice da métrica`; reamostragem em blocos de 256 |
| Teste t | `scipy.stats.ttest_rel(seg, base)` (bicaudal), apenas se `n ≥ 2` e as diferenças não forem todas nulas |
| Wilcoxon | `scipy.stats.wilcoxon(differences, zero_method="wilcox")` (teste **principal**), bicaudal |
| Correção múltipla | Holm–Bonferroni sobre os p-valores de Wilcoxon das 4 métricas: ordenação, `(m−k+1)·p`, máximo acumulado, truncamento em 1,0 |
| `d_z` de Cohen | `mean(d) / std(d, ddof=1)`; `None` se `std ≤ 1e-12` |
| Delta de Cliff | `[#(S>U) − #(S<U)] / (n_S·n_U)` sobre **todas** as comparações entre patches dos dois modelos (não apenas pares posicionais) |
| Vitórias/empates | `segformer_better` se `d > 1e-12`; `unet_better` se `d < −1e-12`; demais empates |
| Deltas | agregado micro (diferença das métricas micro) e por dobra |
| Mapas de erro | 4 patches: maior ΔIoU, menor ΔIoU, maior IoU médio, menor IoU médio (desempate por `patch_id`); códigos `TN=0`, `FP=1`, `FN=2`, `TP=3`, `ignorado=4` |
| Artefatos | `artifacts/metrics/comparative/<fonte>/comparison.json`, figura comparativa e figura de mapas de erro |

Valores efetivos documentados `[doc 4564e35]` (125 pares): ΔIoU médio 0,0666 (IC95 [0,0553; 0,0778]); `p` Wilcoxon 2,84e-19 (Holm 1,14e-18); `d_z` 1,0528; delta de Cliff 0,3620; 119 patches com SegFormer melhor, 1 empate, 5 com U-Net melhor. Esses números pertencem ao Capítulo 4 (Resultados), não a parâmetros metodológicos.

### 6.3 Grad-CAM (U-Net, estágio 13)

`xai/gradcam.py` `[código]`:

| Item | Valor |
| --- | --- |
| Camada-alvo | `encoders[-1]` (`default_target_layer`) — codificador mais profundo; 256 canais, 64×64 para entrada 512×512 |
| Captura da ativação | `layer.register_forward_hook` (removido em `finally`) |
| Pontuação | soma de `logits[0,0][mask]` na **região de referência de café válida**; se a máscara for vazia, `logits.sum()` |
| Gradiente | `torch.autograd.grad(score, activation)[0]`, com `torch.enable_grad()` |
| Pesos por canal | `gradients.mean(dim=(2,3), keepdim=True)` |
| Mapa | `ReLU(Σ_k α_k A^k)` |
| Upsampling | `F.interpolate(..., mode="bilinear", align_corners=False)` para `(H,W)` da entrada |
| Normalização | min-max para `[0,1]`; mapa de zeros se `max ≤ min` |
| Modelo usado | pesos da dobra à qual o patch pertence (patch não visto no treino daquela dobra) |
| Seleção de patches | 4 patches com café: maior IoU, menor IoU, maior `coffee_ratio` e IoU mais próximo da mediana; desempate por `patch_id` |
| Dispositivo | CUDA se disponível; fallback CPU |
| Artefatos | `artifacts/xai/gradcam/unet/<fonte>/gradcam_maps.npz` (`patch_ids`, `folds`, `cams`, `iou`, `coffee_ratio`), `gradcam_metadata.json` e figura |

Constantes de escopo: `GRADCAM_SCHEMA_VERSION = 1`, `TARGET_LAYER_NAME = "encoders[-1]"`, `SCORE_REGION = "reference_coffee"` `[código]`.

### 6.4 Attention Rollout (SegFormer, estágio 14)

`xai/attention_rollout.py` `[código]`:

| Item | Valor |
| --- | --- |
| Extração | `backbone(pixel_values=image, output_attentions=True)` dentro de `torch.no_grad()` |
| Implementação de atenção | forçada para **`eager`** via `set_attn_implementation("eager")` (a `sdpa` não expõe atenções); valida a implementação efetiva |
| Escopo | `ROLLOUT_SCOPE = "last_stage"`: as **últimas `depths[-1] = 2`** atenções (etapa 4, `sr_ratio = 1`, atenção quadrada) |
| Validação | erro se a atenção não for `(cabeças, N, N)` ou se a última etapa não for quadrada |
| Média de cabeças | `A_hat = attention.mean(dim=0)` |
| Resíduo | `A_tilde = A_hat + I` (`RESIDUAL = True`) |
| Normalização de linhas | `A_bar = A_tilde / clamp(Σ_j A_tilde_ij, min=1e-12)` |
| Composição | `rollout = A_bar_l @ rollout`, na ordem de execução das camadas |
| Relevância por token | sem token CLS: `r = rollout.mean(dim=0)` (média das linhas), normalizada para soma 1 (`clamp_min=1e-12`) |
| Grade | `side = round(sqrt(N))`; erro se `side² ≠ N`; para MiT-B1/512 ⇒ 16×16 = 256 tokens |
| Upsampling | `F.interpolate(..., mode="bilinear", align_corners=False)` |
| Normalização final | min-max em `[0,1]` |
| Modelo usado | pesos da dobra do patch (sem vazamento) |
| Seleção de patches | mesma função determinística do Grad-CAM (`select_representative_patches`) |
| Artefatos | `artifacts/xai/attention_rollout/segformer/<fonte>/attention_rollout_maps.npz` (`patch_ids`, `folds`, `rollouts`, `iou`, `coffee_ratio`), metadata com `rollout_scope`, `stage_index`, `stage_layers`, `total_layers`, `stage_grid`, `num_heads`, `head_reduction="mean"`, `token_reduction="mean"`, `residual`, `normalization="min_max"`, `attention_implementation="eager"`, `input_shape`, `upsample_mode="bilinear"`; e figura |

### 6.5 Quantificação da relevância de XAI (estágio 15)

`synthesis.build_xai_table` `[código]`: para cada patch explicado, calcula a média do mapa normalizado dentro da referência de café (`relevance_inside`), fora da referência mas dentro dos pixels válidos (`relevance_outside`) e a diferença (`inside − outside`). A tabela `xai.csv`/`xai.tex` registra `model`, `patch_id`, `label`, `fold`, `iou`, `coffee_ratio`, as três relevâncias, `weights_sha256` e uma explicação textual; o resumo agrega `mean_inside`, `mean_outside` e `mean_difference` `[código]`.

Valores documentados `[doc 4564e35]`: U-Net (Grad-CAM) relevância dentro 0,2431 / fora 0,0629 / diferença 0,1802; SegFormer (Rollout) dentro 0,0055 / fora 0,0075 / diferença −0,0021. A escala menor do rollout decorre da média sobre 256 tokens (≈ 1/256), portanto a comparação válida é dentro × fora **no mesmo modelo**, não entre modelos.

### 6.6 Síntese e disseminação (estágios 15/16)

`synthesis.py` `[código]` produz 6 tabelas obrigatórias (`dataset`, `training`, `evaluation_per_fold`, `evaluation_aggregate`, `comparative_paired`, `comparative_per_fold`), mais `xai` quando há mapas; cada tabela é gravada em CSV e LaTeX (`float_format` de 4 casas, `na_rep="--"`); figuras `synthesis_training_curves_*`, `synthesis_evaluation_overview_*` e `synthesis_xai_relevance_*`; a síntese **não retreina nem reavalia** `[código]`.

`release.py` `[código]`: dataset em `cc-by-4.0`, modelos em `mit`; IDs padrão `<usuário>/tcc-coffee-<fonte>`, `<usuário>/tcc-coffee-unet-<fonte>`, `<usuário>/tcc-coffee-segformer-<fonte>`, sobrescrevíveis por `HF_DATASET_REPO`, `HF_UNET_REPO`, `HF_SEGFORMER_REPO`; empacota `images/<patch_id>.npy`, `masks/<patch_id>.npy`, `manifest.parquet` (caminhos relativos), `normalization_stats_<fonte>.json`, `README.md`, `fold_i.pt`, `model_config.json`; publicação idempotente com `upload_folder` e auditoria em `artifacts/release/<fonte>/release.json`.

---

## 7. Divergências, lacunas e itens não encontrados (auditoria)

| # | Item | Situação |
| --- | --- | --- |
| 1 | Saídas/logs/gráficos dos notebooks | **Não existem no repositório**: todos os notebooks estão com outputs limpos e o CI proíbe outputs. Qualquer número de execução não documentado é `[NÃO ENCONTRADO NO REPOSITÓRIO]` |
| 2 | `METODOLOGIA.md` / `RESULTADOS.md` | Existem no commit `4564e35`, mas estão **deletados na árvore de trabalho** (não commitado). Os valores `[doc 4564e35]` deste arquivo vêm dessa versão |
| 3 | Cabeças de atenção da etapa 3 do MiT-B1 | `METODOLOGIA.md` diz 4; a configuração real é 5 (`[1,2,5,8]`) `[calculado]`. Corrigir no texto |
| 4 | Dimensões do composite da Emater | `METODOLOGIA.md` diz 8192×6656; o código grava o composite no grid da máscara (6953×8445, conforme `RESULTADOS.md`). Verificar GeoTIFF no Drive |
| 5 | "Máscara final" consolidada | O módulo `mask_finalization` foi **removido** em `332effd`; não há máscara final única. O treino usa a máscara da fonte escolhida |
| 6 | Baseline `emater_v1` / `archive_v1` / `BASELINE.json` | Documentado, mas **não versionado**; existência só verificável no Drive `[NÃO ENCONTRADO NO REPOSITÓRIO]` |
| 7 | FLOPs, FPS, tempo por época, uso de memória | **Não implementados/medidos** em nenhum estágio `[NÃO ENCONTRADO NO REPOSITÓRIO]` |
| 8 | BIoU e acurácia | **Não implementados** `[NÃO ENCONTRADO NO REPOSITÓRIO]` |
| 9 | GPU/modelo de hardware usado nas execuções | Os notebooks exigem CUDA e citam "T4/P100", mas **não registram** o modelo efetivo; `run.meta.json` grava `platform`, não a GPU |
| 10 | Python exato do Colab/Kaggle | Não fixado; apenas `requires-python >= 3.10` |
| 11 | Número de cenas Sentinel-2 por ano | Impresso pelo notebook `01` (`collections[year].size()`), mas **não persistido** e sem output versionado |
| 12 | Filtro por classe no estágio 06 | `coffee_min_ratio = 0.0` ⇒ **não há filtro por classe**; não existe limiar de 5% nem de representatividade mínima de café no código |
| 13 | Estação seca / restrição sazonal | **Não existe** no código; usa-se ano civil completo com filtro de nuvem |
| 14 | Super-resolução (S2DR3/S2DR4) | **Não implementada** |
| 15 | AdamW | **Não usado**; o otimizador é `Adam` |
| 16 | Acurácia por época / métricas de treino além de IoU/F1/P/R | Só IoU, F1, precisão e recall (mais as três perdas) |
| 17 | Cálculo de normalização por dobra | **Não existe**; é global por fonte |
| 18 | Conjunto de teste separado | **Não existe**; a validação cruzada espacial é o único mecanismo de generalização |

---

## 8. Proveniência das evidências (comandos e arquivos consultados)

| Evidência | Como foi obtida |
| --- | --- |
| Cronologia e justificativas | `git log --date=short --pretty=format:'%h\|%ad\|%s'`, `git log -1 --format='%B' <hash>`, `git show --stat <hash>` |
| Remoção do `mask_finalization` | `git log --diff-filter=D -- src/data/mask_finalization.py`; `git show 5d89313:src/data/mask_finalization.py` |
| Parâmetros U-Net | instanciação `build_unet()` no `.venv` (torch 2.14.0+cu130) e `sum(p.numel())` = 7.763.329 |
| Parâmetros e configuração MiT-B1 | instanciação `build_segformer()` (transformers 5.x) = 13.680.641; `config.depths/hidden_sizes/num_attention_heads/sr_ratios` |
| Malha IBGE | `geopandas.read_file(...)` → CRS, colunas, 70 feições, `CD_RGI=310044`, `AREA_KM2=2814.869`, bounds em `EPSG:31983` |
| GeoJSONs Emater | `geopandas.read_file` nos 9 arquivos → 34.985 polígonos, atributos, área ≈ 604,52 km² em `EPSG:31983` |
| Ausência de AMP/clipping/tempo/FLOPs | `grep -rn "autocast\|GradScaler\|clip_grad\|time.perf_counter\|flops\|fps\|accuracy\|biou" src/` (sem resultados) |
| Notebooks sem outputs | parsing JSON dos 17 `.ipynb` (`any(cell.get("outputs"))` = falso) |
| Regras de CI | `.github/workflows/ci.yml` e `.opencode/tools/scripts/check_notebook.py` |
| Configuração e pins | `src/config.yaml`, `pyproject.toml`, `requirements-runtime.txt` |
| Números de execução | `git show HEAD:METODOLOGIA.md`, `git show HEAD:RESULTADOS.md` (commit `4564e35`) |

### 8.1 Arquivos que devem ser citados como fonte primária no texto da monografia

| Tema | Arquivo |
| --- | --- |
| Parâmetros de todas as etapas | `src/config.yaml` |
| Aquisição GEE | `src/data/gee_client.py` |
| Máscaras (GEE e vetorial) | `src/data/mask_utils.py`, `src/data/emater_masks.py` |
| Comparação e decisão | `src/data/mask_comparison.py`, `src/data/source_decision.py` |
| Pré-processamento | `src/data/preprocessing.py` |
| Patches e manifesto | `src/data/patch_generation.py` |
| Split espacial | `src/data/spatial_split.py` |
| Dataset, normalização e aumentações | `src/data/dataset.py`, `src/data/augmentations.py`, `src/data/eda.py` |
| Modelos | `src/models/unet.py`, `src/models/segformer.py` |
| Perda | `src/losses.py` |
| Protocolo de treino | `src/trainer.py` |
| Métricas e avaliação | `src/metrics.py`, `src/evaluation.py` |
| Comparação estatística | `src/comparative.py` |
| XAI | `src/xai/common.py`, `src/xai/gradcam.py`, `src/xai/attention_rollout.py` |
| Síntese e publicação | `src/synthesis.py`, `src/release.py` |
| Reprodutibilidade e plataforma | `src/utils.py`, `src/io.py`, `src/bootstrap.py`, `src/setup.py`, `requirements-runtime.txt`, `uv.lock`, `.github/workflows/ci.yml` |

---

### Resumo executivo para o Capítulo 3

- **Tarefa:** segmentação semântica binária (café vs. não café) em patches Sentinel-2 L2A de 512×512 px, 4 bandas a 10 m, RGI de Guaxupé (`310044`), `EPSG:31983`.
- **Dados:** mosaico mediano anual (`COPERNICUS/S2_SR_HARMONIZED`, `CLOUDY_PIXEL_PERCENTAGE < 20`, QA60 bits 10–11); três fontes de ground truth com ano próprio (MapBiomas 2025, AlphaEarth 2024, Emater 2018) e decisão **manual** por Emater 2018.
- **Dataset:** 208 patches candidatos → 125 retidos (`valid_fraction ≥ 0.10`), 124 com café; z-score por banda; rótulo `-1` fora do AOI; `k = 5` dobras por k-means++ determinístico (seed 42) sobre centroides de bbox.
- **Modelos:** U-Net própria (7.763.329 parâmetros, do zero) e SegFormer `nvidia/mit-b1` (13.680.641 parâmetros, backbone ImageNet, projeção 3→4 bandas, logits 1/4 reamostrados).
- **Treino:** Adam (lr 1e-4, wd 1e-4), `CosineAnnealingLR(T_max=100)`, 100 épocas, batch 16, sem AMP/clipping/early stopping; semente `42 + fold`; melhor epoch por IoU de validação.
- **Perda:** `(Dice + Focal + Boundary)/3`, com Dice suavizada (ε = 1,0), Focal (α = 0,25; γ = 2,0; clamp 1e-6) e Boundary por mapa de distância assinada.
- **Avaliação:** IoU/F1/Precisão/Recall por pixel, limiar 0,5, fallback 1,0 em denominador nulo; agregação por patch, por dobra (micro), micro global e macro (média ± desvio populacional).
- **Comparação:** diferença SegFormer − U-Net pareada por patch, bootstrap 2.000 (IC95), t pareado, Wilcoxon + Holm, `d_z`, delta de Cliff e mapas TP/FP/FN/ignorado.
- **XAI:** Grad-CAM em `encoders[-1]` com score na região de café; Attention Rollout na última etapa do MiT (16×16 tokens, média de cabeças, resíduo, normalização de linhas, sem CLS), ambos normalizados em [0,1] e quantificados dentro/fora da referência.

