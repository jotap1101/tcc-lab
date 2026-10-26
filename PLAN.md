# PLAN.md — Roteiro de Implementação

Roteiro definitivo de execução para o estudo comparativo **U-Net (CNN) vs SegFormer (ViT)** em **segmentação semântica de lavouras cafeeiras** (Guaxupé – MG, Brasil) usando imagens Sentinel-2 Nível-2A.

> `AGENTS.md` é a fonte única de verdade para regras de codificação, formatação, idioma e infraestrutura. Todo artefato descrito aqui deve estar em conformidade com ele.

## 1. Propósito

Guiar a implementação sequencial e modular de todo o pipeline — da aquisição no Google Earth Engine à análise de explicabilidade — preservando transparência metodológica e reprodutibilidade computacional.

## 2. Decisões de arquitetura (MLOps)

- **Layout híbrido**: notebooks Jupyter enxutos (`.ipynb`) atuam como camada de orquestração/visualização; toda lógica reutilizável vive no pacote compartilhado **`src/`** (Python puro), único alvo de `ruff`, `mypy` e `pytest`.
- **Fonte única de verdade**: `src/config.yaml` (carregado por `config.py`) centraliza caminhos, bandas, tamanho de patch, hiperparâmetros e sementes. Nenhum caminho hardcoded nos notebooks.
- **Dados orientados por manifesto**: cada fonte de ground truth tem um manifesto próprio `manifest_<fonte>.parquet` que registra cada patch (`patch_id`, `tile_id`, `fold`, bbox, `coffee_ratio`, `mask_source`, caminhos). É a espinha dorsal da reprodutibilidade.
- **Protocolo de treinamento idêntico** entre modelos: mesmo split, perda, otimizador, scheduler, métricas e semente — apenas a arquitetura difere. Isso garante uma comparação cientificamente justa.
- **Notebooks agnósticos de plataforma**: todo notebook executa **corretamente** no **Google Colab e no Kaggle**, com o mesmo protocolo (mesmo código, sementes, config e processamento). Toda detecção de ambiente, montagem do Drive e resolução de caminhos é abstraída em `src/` — notebooks nunca usam caminhos ou APIs específicos de plataforma.
- **Abstração de armazenamento (`src/io.py`)**: o contrato único para detecção de plataforma, montagem do Drive e resolução de caminhos. Expõe no mínimo `detect_platform()`, `mount_drive()` (Colab nativo; Kaggle via Drive API com cache local), `ensure_storage_root()` (idempotente — cria `MyDrive/tcc/` e toda subpasta necessária recursivamente quando ausente, no-op quando presente), `resolve_storage_paths()` (construído a partir do `config.yaml`) e `sync_repo_to_workspace()` (copia o espelho `MyDrive/tcc/repo/` para o workspace quando ele já existe). Notebooks consomem apenas este módulo.
- **Entrega do `src/` ao runtime**: a primeira célula de todo notebook baixa e executa `src/bootstrap.py` (somente stdlib) do repositório público — ele extrai `src/`, `data/external/` e `requirements-runtime.txt` para o workspace, adiciona o workspace ao `sys.path`, grava o SHA do commit de `main` em `src/.tcc_commit` (API do GitHub, com fallback Atom) e, no Colab, cria o espelho `MyDrive/tcc/repo/` (idempotente). O `sync_repo_to_workspace()` de `src/io.py` copia do espelho quando ele já existe. Nenhum upload manual ou configuração de clonagem.
- **Pipeline idempotente e re-executável**: reexecutar um notebook é seguro — ele abre/cria a árvore `tcc/`. Artefatos derivados (manifesto, split, estatísticas de normalização, patches, pesos, métricas, histórico e figuras) são reutilizados apenas enquanto o fingerprint das entradas e o hash da configuração coincidem; quando mudam, são regenerados no lugar. Para preservar um baseline histórico, as pastas afetadas são arquivadas com sufixo `_v1` (ex.: `models/unet/emater_v1`, `data/processed/archive_v1`, `artifacts/figures/archive_v1`), que o pipeline nunca sobrescreve.
- **Filtro de validação do dataset**: patches com fração de pixels válidos abaixo de `data.min_valid_fraction` são descartados (área fora do AOI); pixels inválidos remanescentes recebem o rótulo de ignorado `-1` e são excluídos da perda e das métricas. O manifesto registra `valid_fraction` por patch e tem schema versionado; o metadata registra `bbox_schema_version` para invalidar manifestos gerados com cálculo de `bbox` antigo sem reenviar os patches.
- **Dependências específicas de plataforma**: bibliotecas atreladas a um host (ex.: `google.colab` para montagem do Drive, secrets do Kaggle) são fornecidas pela plataforma e usadas apenas dentro de `src/` — elas **não** entram em `pyproject.toml`/`uv.lock`, mantendo o repositório agnóstico de ambiente.
- **Armazenamento canônico no Google Drive**: a pasta `tcc/` na **raiz do Google Drive** (`MyDrive/tcc/`) é a raiz única de armazenamento para todo dado, modelo, métrica, figura e log. Notebooks a acessam se já existir, ou a criam (e qualquer subpasta necessária) caso contrário — os próprios notebooks geram a estrutura de pastas.
- **Ground truth por fonte**: cada fonte de ground truth (MapBiomas, AlphaEarth, Emater e qualquer fonte futura) declara o seu **ano mais recente** (`year` em `src/config.yaml`) e é integrada em **célula de notebook dedicada**, armazenada em **subpasta própria** sob `MyDrive/tcc/data/interim/<fonte>/`, com **conjunto próprio de patches e manifesto** (`patches/<fonte>/`, `manifest_<fonte>.parquet`). Adicionar uma fonte significa adicionar uma célula — nunca modificar a lógica de outra fonte. **Sem eleição automática**: o estágio 03 compara todas as fontes e o estágio 04 registra a decisão **manual** do pesquisador (`ground_truth.chosen_source`), que apenas seleciona o conjunto de patches do treino (09/10).
- **CI (GitHub Actions)** executa `ruff` + `mypy` + `pytest` no push; treinos pesados de GPU rodam no Colab/Kaggle com artefatos persistidos na raiz `tcc/` do Drive.
- **Autenticação dentro dos notebooks**: toda etapa de autenticação (GEE, Hugging Face, Kaggle) é executada nas próprias células de código dos notebooks com as bibliotecas necessárias — ex.: GEE via OAuth com a conta principal (`ee.Authenticate()`, token persistido na raiz `tcc/` do Drive). Credenciais nunca são commitadas; não há etapa externa/manual de autenticação.
- **Princípios de qualidade de código**: DRY, Responsabilidade Única, KISS, YAGNI e Separação de Conceitos se aplicam em todo o projeto — notebooks orquestram/visualizam e nunca reimplementam ou duplicam lógica que vive em `src/`; o código de notebook atende ao mesmo padrão de qualidade do pacote compartilhado.
- **Publicação**: dataset (CC BY 4.0) e pesos de modelo publicados no Hugging Face Hub.

## 3. Layout de diretórios

```bash
tcc/
├── AGENTS.md                      # fonte de verdade (regras)
├── PLAN.md                        # este roteiro
├── pyproject.toml / uv.lock       # uv + ruff + mypy + pytest
├── src/                           # pacote compartilhado (lógica reutilizável)
│   ├── config.py + config.yaml
│   ├── bootstrap.py  setup.py  io.py  utils.py
│   ├── data/{augmentations,dataset,eda,gee_client,mask_comparison,mask_utils,patch_generation,preprocessing,raster_utils,source_decision,spatial_split}.py
│   ├── losses.py
│   ├── metrics.py
│   ├── evaluation.py
│   ├── trainer.py
│   ├── models/{unet,segformer}.py
│   └── xai/{gradcam,attention_rollout}.py
├── tests/                         # pytest somente sobre src/
├── notebooks/                     # 17 arquivos .ipynb sequenciais
├── scripts/                       # geradores de tokens OAuth (Drive, GEE) — uso local
├── secrets/                       # apenas templates versionados (README.md, .env.example)
├── data/{raw,interim,processed}/            # espelho git-ignored de MyDrive/tcc/data
│   └── external/                            # dados de referência VERSIONADOS (ex.: malha IBGE)
├── models/                        # espelho git-ignored de MyDrive/tcc/models
├── artifacts/                     # espelho git-ignored de MyDrive/tcc/artifacts
└── .github/workflows/ci.yml       # ruff + mypy + pytest
```

**Layout de armazenamento (canônico — raiz `tcc/` do Google Drive).** Os notebooks criam/acessam esta árvore sob demanda; nenhuma pasta pré-criada pode ser assumida:

```bash
MyDrive/tcc/                       # raiz canônica de todo o armazenamento do projeto (criada/acessada pelos notebooks)
├── data/{raw,interim,processed,external}/   # dados espectrais, máscaras, patches
│   └── interim/{mapbiomas,alphaearth,emater,...}/   # ground truth por fonte (ano próprio)
├── models/{unet,segformer}/       # pesos (espelhados localmente, git-ignored; baselines: *_v1/)
├── artifacts/{metrics,figures,runs}/       # métricas, figuras, logs (baselines: *_v1/ e archive_v1/)
├── repo/src/                      # espelho do código-fonte (entrega do src/ ao runtime)
├── secrets/                       # tokens OAuth persistidos (GEE, Drive) — nunca versionar
└── ...                            # qualquer subpasta mais profunda exigida pelo pipeline
```

## 4. Fluxo de dados

```bash
Malha vetorial IBGE (Região Geográfica Imediata 310044) → polígono AOI
GEE Sentinel-2 L2A (B2/B3/B4/B8) filtrado pelo AOI
  → mosaicos sem nuvens por ano (QA60)                  [01]
  → máscaras de referência por fonte (ano próprio)      [02]
  → comparação de máscaras por fonte + decisão manual   [03, 04]
  → composites alinhados e normalizados por ano         [05]
  → patches 512x512 filtrados por pixels válidos + manifesto por fonte  [06]
  → atribuição espacial k-fold por fonte                [07]
  → EDA / estatísticas de normalização por fonte        [08]
  → treino U-Net / SegFormer (k=5, fonte escolhida)     [09, 10]
  → predições + métricas por pixel (IoU/F1/P/R)        [11]
  → comparação estatística                         [12]
  → Grad-CAM + Attention Rollout                   [13, 14]
  → figuras para a monografia                         [15]
  → empacotamento + publicação                              [16]
```

## 5. Fases e notebooks

Cada notebook é um estágio isolado com uma única responsabilidade e entradas/saídas declaradas. A ordem de execução é numérica.

| Notebook                                   | Fase                 | Entrada → Saída                                                                                         |
| ------------------------------------------ | -------------------- | ------------------------------------------------------------------------------------------------------- |
| `00_setup_environment.ipynb`               | 0. Setup             | — → ambiente pronto, `src/` entregue, raiz `tcc/` do Drive resolvida, config carregada, GEE autenticado |
| `01_gee_sentinel2_acquisition.ipynb`       | 1. Aquisição         | malha IBGE (310044) → polígono AOI → mosaicos GeoTIFF Sentinel-2 L2A por ano das fontes |
| `02_gee_reference_masks.ipynb`             | 1. Aquisição         | cada fonte (MapBiomas 2025/AlphaEarth 2024/Emater 2018) → máscaras binárias de 10 m por fonte |
| `03_mask_sources_comparison.ipynb`         | 2. Ground Truth      | máscaras por fonte → diagnóstico comparativo (todas as fontes e pares) |
| `04_source_decision.ipynb`                 | 2. Ground Truth      | relatório comparativo → registro da decisão manual da fonte de treino |
| `05_preprocessing.ipynb`                   | 3. Pré-processamento | mosaicos por ano → composites sem nuvens normalizados (alinhados à máscara de referência do ano) |
| `06_patch_generation.ipynb`                | 3. Dataset           | composites+máscaras por fonte → patches 512x512 filtrados por pixels válidos + manifesto por fonte |
| `07_spatial_kfold_split.ipynb`             | 3. Dataset           | manifesto por fonte → manifesto com `fold` por fonte |
| `08_dataset_eda.ipynb`                     | 3. Dataset           | manifesto por fonte → estatísticas de normalização + verificações de sanidade por fonte |
| `09_train_unet.ipynb`                      | 4. Treinamento       | dataset filtrado da fonte escolhida → pesos do U-Net + métricas/histórico por dobra (retreino por fingerprint) |
| `10_train_segformer.ipynb`                 | 4. Treinamento       | dataset filtrado da fonte escolhida → pesos do SegFormer + métricas/histórico por dobra (protocolo único, retreino por fingerprint) |
| `11_evaluation.ipynb`                      | 5. Avaliação         | pesos por dobra → predições persistidas + IoU/F1/P/R em nível de pixel (por patch, por dobra e agregado) |
| `12_comparative_analysis.ipynb`            | 5. Avaliação         | métricas → comparação estatística + mapas de erro                                                       |
| `13_xai_gradcam_unet.ipynb`                | 6. XAI               | U-Net → mapas de calor Grad-CAM                                                                         |
| `14_xai_attention_rollout_segformer.ipynb` | 6. XAI               | SegFormer → mapas Attention Rollout                                                                     |
| `15_results_synthesis.ipynb`               | 7. Síntese           | tudo → figuras/tabelas para a monografia                                                                |
| `16_export_release.ipynb`                  | 7. Disseminação      | patches+pesos → dataset + modelos no HF Hub                                                             |

## 6. Esquema de config e artefatos

- **`config.yaml`**: `aoi` (código da região `310044`, `vector_source` `ibge_mesh`, `mesh_path` apontando para `data/external/ibge/mg_rg_immediatas_2025/`), `data` (`collection`, `dates` — fallback para fontes sem `year`, `bands`, `patch_size`, `coffee_min_ratio`, `min_valid_fraction` — fração mínima de pixels válidos por patch, descartando área fora do AOI, `cloud_threshold`, `crs`, `export`), `gee` (`project` — ID do projeto Cloud para `ee.Initialize`), `ground_truth` (`chosen_source` — decisão manual — + bloco `sources` com `year` próprio por fonte, `collection_id`/`collection_filter`, classe/limiar e `enabled`), `splits.fold_count`, `reproducibility.seed`, `model` (`unet_channels`, `segformer_variant`), pesos de `loss`, `augmentation` (aumentações geométricas determinísticas reseedadas por época: `hflip_prob`, `vflip_prob`, `rot90_prob`), `training` (`lr`, `epochs`, `batch_size`, `weight_decay`, `num_workers`). O bloco `storage` mapeia a raiz `tcc/` do Drive e toda subpasta (`data`, `models`, `artifacts`, `repo`, `secrets`, ...) — todos os caminhos são resolvidos daqui, nunca hardcoded.
- **Colunas do `manifest_<fonte>.parquet`**: `patch_id`, `tile_id`, `fold`, `row`, `col`, `bbox`, `coffee_ratio`, `valid_fraction`, `mask_source`, `image_path`, `mask_path`. `coffee_ratio` é medido apenas sobre os pixels válidos do patch; `valid_fraction` registra a fração dentro do AOI.
- **Artefatos** (todos sob a raiz `tcc/` do Drive, caminhos resolvidos de `config.yaml`): `MyDrive/tcc/artifacts/metrics/{model}/{fonte}/fold_{i}.json` (com `manifest_fingerprint_hash`, `split_fingerprint_hash` e `config_hash`), `MyDrive/tcc/artifacts/metrics/{model}/{fonte}/evaluation/{fold_{i}.json,summary.json,patch_metrics_fold_{i}.parquet,predictions_fold_{i}.npz}` (métricas de avaliação por dobra, resumo agregado, métricas por patch e predições binárias — estágio 11), `MyDrive/tcc/artifacts/runs/{model}/{fonte}/fold_{i}.json` + `run.meta.json`, `MyDrive/tcc/artifacts/figures/`, `MyDrive/tcc/models/{model}/{fonte}/fold_{i}.pt`, `MyDrive/tcc/data/processed/manifest_<fonte>.parquet` e `MyDrive/tcc/data/processed/normalization_stats_<fonte>.json` (estatísticas de normalização por banda — estágio 08). Baselines históricos ficam em `models/{model}/{fonte}_v1/`, `artifacts/metrics|runs/{model}/{fonte}_v1/`, `data/processed/archive_v1/` e `artifacts/figures/archive_v1/`.

## 7. Rastreador de progresso

### Fase 0 — Setup

- [x] `00_setup_environment.ipynb` — detecção de plataforma, montagem do Drive + resolução da raiz `tcc/`, entrega do `src/` (a primeira célula baixa e executa `src/bootstrap.py`: extrai `src/`, `data/external/` e `requirements-runtime.txt`, adiciona o workspace ao `sys.path`, grava `src/.tcc_commit` e cria o espelho `MyDrive/tcc/repo/` no Colab), instalação de dependências (`requirements-runtime.txt`), autenticação do GEE (executada nas células dos notebooks), sementes + flags determinísticas, carregamento da config, auto-verificação do ambiente
- [x] `src/io.py` (detecção de plataforma, montagem do Drive, garantia/resolução da raiz `tcc/`, entrega do `src/`) + testes unitários
- [x] `pyproject.toml` (uv, ruff, mypy, pytest)
- [x] `requirements-runtime.txt` (versões pinadas instaladas pela célula de dependências de cada notebook via `src.setup.install_runtime_requirements()`, com reinício automático do kernel quando uma extensão compilada já carregada diverge)
- [x] `uv.lock` (gerado via `uv lock`)
- [x] `.github/workflows/ci.yml` (ruff + mypy + pytest + validação de notebooks)

### Fase 1 — Aquisição

- [x] `01_gee_sentinel2_acquisition.ipynb` — importação da malha IBGE (polígono AOI) + aquisição Sentinel-2
- [x] `02_gee_reference_masks.ipynb`

### Fase 2 — Ground Truth

- [x] `03_mask_sources_comparison.ipynb` — comparação de todas as fontes (áreas, concordância entre todos os pares, figura multi-painel)
- [x] `04_source_decision.ipynb` — registro da decisão manual da fonte de treino (sem eleição automática)

### Fase 3 — Pré-processamento e Dataset

- [x] `05_preprocessing.ipynb`
- [x] `06_patch_generation.ipynb` — patches 512x512 por fonte + manifesto `manifest_<fonte>.parquet`; descarta patches abaixo de `data.min_valid_fraction` de pixels válidos, registra `valid_fraction` e versiona o cálculo de `bbox` (`bbox_schema_version`)
- [x] `07_spatial_kfold_split.ipynb` — divisão espacial k-fold (k-means determinístico em numpy puro sobre centroides das bboxes corrigidas) gravada na coluna `fold` do manifesto por fonte, com `split_<fonte>.meta.json` para idempotência; recalculada automaticamente quando o fingerprint do manifesto, o schema do split ou o schema de `bbox` mudam
- [x] `08_dataset_eda.ipynb` — estatísticas de normalização por banda (média, desvio, fração de NaN) persistidas em `normalization_stats_<fonte>.json` com fingerprint, verificações de sanidade e figura de resumo por fonte (regenerada a cada execução)

### Fase 4 — Treinamento

- [x] `09_train_unet.ipynb` — treino do U-Net na fonte escolhida com o protocolo único (`src/trainer.train_fold`): perda multivariada, Adam + cosine annealing, dobras 0..4, pesos `models/unet/<fonte>/fold_i.pt`, métricas e histórico por dobra + metadata de execução. Métricas guardam `manifest_fingerprint_hash`/`config_hash`; dobras com artefatos desatualizados são retreinadas automaticamente. O primeiro treino (dataset sem filtro) está preservado em `emater_v1`/`archive_v1`.
- [x] `10_train_segformer.ipynb` — treino do SegFormer (`model.segformer_variant`, padrão `nvidia/mit-b1`) na fonte escolhida com o mesmo `src/trainer.train_fold` do estágio 09: mesmas dobras, perda, otimizador, scheduler, sementes e métricas. A primeira projeção é adaptada de 3 para 4 bandas (pesos RGB preservados; quarto canal = média dos três), a cabeça de segmentação é inicializada do zero (checkpoint de backbone ImageNet) e os logits de 1/4 de resolução são reamostrados bilinearmente para a entrada. Pesos em `models/segformer/<fonte>/fold_i.pt`, métricas/histórico e `run.meta.json` sob `artifacts/{metrics,runs}/segformer/<fonte>/`.

### Fase 5 — Avaliação

- [x] `11_evaluation.ipynb` — inferência dos pesos por dobra dos estágios 09/10 sobre a validação espacial, com IoU/F1/Precisão/Recall em nível de pixel (por patch, por dobra e agregado micro), ignorando `-1`. Persiste `evaluation/{fold_{i}.json,summary.json,patch_metrics_fold_{i}.parquet,predictions_fold_{i}.npz}` por modelo/fonte e figura de resumo; reutiliza artefatos apenas quando manifesto, split, config e hash dos pesos coincidem.
- [ ] `12_comparative_analysis.ipynb`

### Fase 6 — XAI

- [ ] `13_xai_gradcam_unet.ipynb`
- [ ] `14_xai_attention_rollout_segformer.ipynb`

### Fase 7 — Síntese e Disseminação

- [ ] `15_results_synthesis.ipynb`
- [ ] `16_export_release.ipynb`

### Pacote compartilhado (`src/`) — construído junto das fases

- [x] `config.py` + `config.yaml`
- [x] `data/dataset.py` (dataset de patches, normalização, divisão por dobra e carregadores — estágio 09; pixels inválidos recebem rótulo `-1`), `data/augmentations.py` (aumentações geométricas determinísticas por época)
- [x] `data/mask_utils.py` (uma função por fonte de ground truth, com `year` próprio por fonte), `data/gee_client.py`
- [x] `data/mask_comparison.py` (diagnóstico comparativo de todas as fontes — estágio 03)
- [x] `data/source_decision.py` (registro da decisão manual da fonte — estágio 04)
- [x] `data/preprocessing.py` (alinhamento e normalização do composite por ano — estágio 05)
- [x] `data/patch_generation.py` (patches 512x512 por fonte + manifesto Parquet — estágio 06)
- [x] `data/spatial_split.py` (divisão espacial k-fold do manifesto por fonte — estágio 07)
- [x] `data/eda.py` (estatísticas de normalização + sanidade do dataset por fonte — estágio 08)
- [x] `data/emater_masks.py` (máscara vetorial da Emater — estágio 02)
- [x] `losses.py` (Dice + Focal + Boundary, com pesos de config.yaml; ignora pixels com rótulo `-1`)
- [x] `metrics.py` (IoU, F1, Precision, Recall com acumulador por época; ignora pixels com rótulo `-1`)
- [x] `evaluation.py` (inferência dos pesos por dobra, métricas por patch/dobra/agregadas, predições persistidas e figura — estágio 11)
- [x] `trainer.py` (protocolo de treino único, parametrizado pelo modelo — sem duplicação entre notebooks 09/10)
- [x] `models/unet.py`
- [x] `models/segformer.py` (SegFormer binário com adaptação 3→4 bandas e logits na resolução da entrada)
- [ ] `xai/gradcam.py`, `xai/attention_rollout.py`
- [x] `io.py` (detecção de plataforma + montagem do Drive + garantia/resolução da raiz `tcc/` + entrega do `src/` via espelho), `bootstrap.py` (entrega do `src/` ao runtime antes do import + gravação de `src/.tcc_commit`), `setup.py` (prólogo de runtime + `install_runtime_requirements` com reinício automático em divergência de ABI), `utils.py`
- [x] `tests/` (config, io, bootstrap, setup, utils, data/*, losses, metrics, evaluation, trainer e modelos — pytest)

## 8. Checklist de reprodutibilidade

- [x] Sementes fixadas (`python`/`numpy`/`torch`/`cuda`)
- [x] `uv.lock` commitado; dump do ambiente registrado por execução
- [x] Versões das bibliotecas de runtime pinadas na célula de dependências de cada notebook (`requirements-runtime.txt`) e registradas por execução (os ambientes padrão do Colab/Kaggle diferem)
- [x] Flags determinísticas do Torch ativadas (`cudnn.deterministic`, `benchmark=False`, `use_deterministic_algorithms`) e não-determinismo residual documentado
- [x] Notebooks commitados com outputs limpos (sem outputs de célula salvos)
- [x] Convenções estruturais dos notebooks validadas no CI (nomenclatura, markdown acima do código, outputs vazios)
- [x] Operações de armazenamento idempotentes: artefatos derivados são reutilizados somente com fingerprint/config hash coincidentes e regenerados no lugar quando mudam; baselines históricos são arquivados com sufixo `_v1`
- [x] Geração de patches determinística e sem sobreposição, com filtro por pixels válidos
- [x] Todo notebook resolve caminhos de entrada a partir de `src/config.py`
- [x] Notebooks agnósticos de plataforma: executam corretamente no Colab e no Kaggle (mesmo protocolo), sem caminhos/APIs hardcoded de plataforma
- [x] Todos os artefatos armazenados sob a raiz `tcc/` do Drive; notebooks criam a raiz e subpastas sob demanda
- [x] Split/perda/otimizador/métricas idênticos entre os dois modelos

## 9. Decisões consolidadas e riscos em aberto

### Decisões consolidadas

- **Fonte de ground truth:** Emater 2018, escolhida manualmente no estágio `04` (decisão registrada em `ground_truth.chosen_source`).
- **Filtro de dataset:** `data.min_valid_fraction = 0.10`; patches abaixo desse limiar são descartados e pixels inválidos remanescentes usam o rótulo `-1`, ignorado por perda e métricas.
- **Reuso de artefatos:** por fingerprint do manifesto + fingerprint do split + hash da configuração; dobras desatualizadas são retreinadas automaticamente. A avaliação (`11`) acrescenta o hash dos pesos da dobra ao fingerprint e reavalia quando qualquer um deles muda.
- **Avaliação (estágio 11):** inferência dos pesos de cada dobra na validação espacial, limiar fixo de 0,5 (mesmo de `src/metrics.py`), pixels `-1` ignorados. Reporta métricas por patch, por dobra e o agregado *micro* (TP/FP/FN somados), além da média/desvio das métricas por dobra. Predições binárias e métricas por patch ficam persistidas para o estágio 12 (comparação estatística e mapas de erro).
- **Correção de bbox/split:** o cálculo de `bbox` usava offset em pixels (em vez de `row/col * patch_size`), fazendo as bboxes se sobreporem e o split não ser espacialmente válido. Corrigido em `BBOX_SCHEMA_VERSION=1`/`SPLIT_SCHEMA_VERSION=2`; manifestos antigos são regenerados sem reenviar os patches.
- **Baseline:** o primeiro treino do U-Net (dataset sem filtro e antes da correção de bbox/split) foi preservado como `emater_v1`; os metadados do dataset antigo estão em `data/processed/archive_v1/`, as figuras em `artifacts/figures/archive_v1/` e o `BASELINE.json` registra a limitação.
- **SegFormer (estágio 10):** `nvidia/mit-b1` via Hugging Face Transformers, com o mesmo `src/trainer.py` do U-Net; a primeira projeção é expandida de 3 para 4 bandas preservando os pesos RGB e inicializando o quarto canal com a média dos três, a cabeça de segmentação é inicializada do zero (o checkpoint é de backbone ImageNet) e os logits em 1/4 de resolução são reamostrados bilinearmente para a entrada.
- **Determinismo do SegFormer:** o upsampling bilinear não tem backward determinístico no CUDA no PyTorch; os estágios 10 e 11 usam `torch.use_deterministic_algorithms(True, warn_only=True)`, mantendo seeds/cuDNN e registrando a configuração efetiva do treino em `run.meta.json` (`deterministic_algorithms.warn_only = true`).
- **Dependências:** instalação centralizada em `src.setup.install_runtime_requirements()`, com reinício automático do kernel em divergência de ABI. Notebooks que usam Transformers (`10`, `11`) também chamam `src.setup.remove_incompatible_torchaudio()`: o ambiente do Kaggle pode trazer um `torchaudio` compilado para outra versão de CUDA, que impede o import do Transformers; como o projeto não usa áudio, o pacote é removido apenas quando não importa.
- **Rastreabilidade:** `commit_sha` vem do git local ou de `src/.tcc_commit`, gravado pelo bootstrap; `run.meta.json` registra config hash, fingerprint do manifesto, configuração do modelo (`model_config`) e estado do determinismo, além do ambiente.

### Riscos em aberto

- **Fonte de ground truth** é o principal risco (MapBiomas 2025 vs AlphaEarth 2024 vs Emater 2018 — anos distintos) — cada fonte tem sua própria célula de integração, subpasta no Drive e conjunto de patches; a escolha da fonte de treino é uma decisão **manual** do pesquisador registrada na Fase 2 (estágio 04), antes de qualquer treino.
- **CRS/georreferenciamento** dos rótulos vs a grade do Sentinel (zona UTM para MG).
- **OAuth do GEE no Kaggle** (fluxo headless do `ee.Authenticate()` + persistência do token entre sessões) — validado no notebook `00`.
- **Montagem do Drive no Kaggle** (sem montagem nativa — Drive API via token OAuth `GDRIVE_TOKEN`/`GDRIVE_TOKEN_FILE` + cache local) — deve ser validada no notebook `00` antes de qualquer escrita de armazenamento; a abstração do `src/` isola isso do código dos notebooks.
- **Provisionamento de credenciais por plataforma** (Colab `userdata`/arquivo no Drive vs secrets do Kaggle vs variáveis de ambiente) — `io.py` e os notebooks leem credenciais uniformemente de variáveis de ambiente; validado no notebook `00`.
- **Latência e cota de I/O do Drive** em etapas de escrita pesada (geração de patches, checkpoints de treino) — mitigado por um cache de trabalho local no workspace da plataforma, com a raiz `tcc/` do Drive como alvo autoritativo.
- **Variante MiT** (b0–b2) balanceada contra a VRAM de T4/P100.
