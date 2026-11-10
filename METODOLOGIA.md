# Contexto Metodológico Completo — Segmentação Semântica de Lavouras Cafeeiras

> **Finalidade.** Este documento reúne, de forma detalhada e fiel à implementação, toda a metodologia executada no repositório `tcc-lab`. Ele serve de contexto para a redação do capítulo de Metodologia da monografia. Não é o capítulo final: é a fonte técnica de consulta, com parâmetros, decisões, equações, caminhos de artefatos, protocolos e limitações.
>
> **Escopo.** Descreve o que foi efetivamente implementado, não apenas o que foi planejado. Quando o texto atual da monografia divergir da implementação, a implementação e os valores de `src/config.yaml` prevalecem. A Seção 18 lista as divergências mais importantes que devem ser corrigidas.
>
> **Identificadores de referência da execução versionada.**
>
> - Commit do repositório: `0169d8dd878d31ba1f806c9cbdd44bcd42fdbae7`.
> - Hash da configuração (`config.yaml`): `9330341efa14`.
> - Fingerprint do manifesto Emater: `adb0b7b0229e`.
> - Fingerprint do split espacial Emater: `ece582ed5e14`.
> - Fonte de ground truth escolhida manualmente: `emater` (ano 2018).
> - Dataset e pesos publicados: `jotap1101/tcc-coffee-emater`, `jotap1101/tcc-coffee-unet-emater` e `jotap1101/tcc-coffee-segformer-emater` no Hugging Face Hub.

---

## 0. Resumo do pipeline em oito passos

1. **Área de estudo:** polígono da RGI de Guaxupé (`310044`) extraído da malha do IBGE.
2. **Aquisição:** mosaico anual mediano do Sentinel-2 Nível-2A (B2, B3, B4, B8, 10 m), filtrado por nuvens e recortado ao AOI.
3. **Ground truth:** máscaras binárias de MapBiomas 2025, AlphaEarth 2024 e Emater 2018; comparação entre fontes e decisão manual por Emater.
4. **Dataset:** composite normalizado, patches de 512×512 sem sobreposição, filtro de pixels válidos (`valid_fraction >= 0.10`), manifesto por fonte.
5. **Validação:** divisão espacial k-fold (k = 5) por k-means++ determinístico sobre centroides das bboxes.
6. **Modelos:** U-Net e SegFormer `nvidia/mit-b1`, treinados com o mesmo protocolo (perda Dice + Focal + Boundary, Adam, cosine annealing, 100 épocas, semente por dobra).
7. **Avaliação:** IoU, F1, Precisão e Revocação por pixel; agregação micro/macro; comparação emparelhada com bootstrap, testes t/Wilcoxon, Holm e tamanhos de efeito; Grad-CAM e Attention Rollout.
8. **Síntese e disseminação:** tabelas CSV/LaTeX, figuras e publicação do dataset (CC BY 4.0) e dos pesos (MIT) no Hugging Face Hub.

---

## 1. Delineamento da pesquisa

A pesquisa é **aplicada**, de abordagem **quantitativo-analítica**, com delineamento **comparativo** entre duas famílias de arquiteturas de aprendizado profundo para **segmentação semântica binária** de lavouras cafeeiras:

- **classe de interesse:** café;
- **classe complementar:** não café (solo, vegetação não cafeeira, água, área urbana, sombra, estradas, carreadores etc.);
- **unidade de análise:** patch de imagem multiespectral Sentinel-2 de 512 × 512 pixels;
- **rótulo:** máscara binária por pixel (1 = café; 0 = não café; -1 = pixel inválido/ignorado);
- **recorte territorial:** Região Geográfica Imediata (RGI) de Guaxupé, Minas Gerais, Brasil, código IBGE `310044`;
- **modelos comparados:** U-Net (CNN) e SegFormer (ViT);
- **validação:** validação cruzada espacial em k = 5 dobras;
- **métricas principais:** IoU, F1-Score, Precisão e Revocação (Recall), em nível de pixel;
- **explicabilidade:** Grad-CAM para a U-Net e Attention Rollout para o SegFormer;
- **produtos:** dataset de patches, pesos por dobra, mapas de XAI, tabelas e figuras para a monografia, com publicação no Hugging Face Hub.

O objetivo comparativo central é verificar como a transição de um **viés convolucional local** (U-Net) para a **autoatenção global** (SegFormer) afeta a precisão geométrica das máscaras, o comportamento sob relevo montanhoso e a interpretabilidade das decisões.

---

## 2. Área de estudo e dados

### 2.1 Área de estudo (AOI)

A área de estudo é delimitada a partir da **malha vetorial das Regiões Geográficas Imediatas de Minas Gerais**, versão 2025, do IBGE, versionada em:

```text
data/external/ibge/mg_rg_immediatas_2025/mg_rg_immediatas_2025.shp
```

O filtro é feito por `CD_RGI == "310044"` (RGI de Guaxupé). A malha é reprojetada para `EPSG:4326` para virar geometria do Earth Engine; o processamento raster usa `EPSG:31983` (SIRGAS 2000 / UTM 23S).

| Item | Valor |
| --- | --- |
| Código IBGE da RGI | `310044` |
| Nome | Guaxupé |
| Área no atributo `AREA_KM2` | 2.814,869 km² |
| Área geométrica em EPSG:31983 | ≈ 2.814,42 km² |
| Extensão da caixa envolvente em EPSG:31983 | x: 290.980–372.900 m; y: 7.614.850–7.681.410 m |
| Dimensão da caixa envolvente | ≈ 81,92 km × 66,56 km |

A área do AOI é lida em tempo de execução do atributo `AREA_KM2`. A documentação da fonte Emater menciona “≈10% do AOI” para os ~603 km² de café. Sobre o polígono do AOI (2.814,87 km²), a fração é de aproximadamente **21,2%**; o valor de ~10% corresponde aproximadamente à razão entre 603 km² e a área da caixa envolvente (~5.452 km²). Esse ponto deve ser tratado com clareza no texto.

### 2.2 Imagens Sentinel-2

| Parâmetro | Valor |
| --- | --- |
| Coleção GEE | `COPERNICUS/S2_SR_HARMONIZED` |
| Produto | Sentinel-2 Nível-2A (reflectância de superfície harmonizada) |
| Bandas | `B2` (azul), `B3` (verde), `B4` (vermelho), `B8` (infravermelho próximo) |
| Resolução espacial | 10 m (bandas nativas) |
| Filtro de cena | `CLOUDY_PIXEL_PERCENTAGE < 20` |
| Máscara de nuvens | `QA60`, bits 10 e 11 (nuvens opacas e cirros) |
| Período | Ano civil completo de cada fonte (`YYYY-01-01` a `YYYY-12-31`) |
| Composição | Mediana temporal de todas as cenas válidas do período |
| Recorte | Clip ao polígono do AOI |
| CRS de exportação | `EPSG:31983` (SIRGAS 2000 / UTM 23S) |
| Escala de exportação | 10 m |
| `maxPixels` | 100.000.000 |
| Formato | GeoTIFF |

O pipeline **não** restringe a aquisição aos meses da estação seca. Usa o ano civil completo e controla a contaminação por nuvens por dois mecanismos: filtro de percentual de nuvens da cena e máscara `QA60`. O mosaico é a **mediana** por pixel das cenas filtradas, o que reduz outliers e lacunas.

O nome canônico do mosaico é:

```text
data/raw/sentinel2/sentinel2_310044_<YYYY-01-01>_<YYYY-12-31>.tif
```

Para as fontes habilitadas, os anos de mosaico necessários são `2018` (Emater), `2024` (AlphaEarth) e `2025` (MapBiomas). O ano de cada fonte vem de `ground_truth.sources.<fonte>.year` em `src/config.yaml`.

### 2.3 Fontes de ground truth

O trabalho não adota uma única fonte de referência de forma automática. Três fontes são integradas, cada uma com **ano próprio**, **máscara binária própria**, **subpasta própria** e **conjunto de patches próprio**:

| Fonte | Ano | Natureza | Regra de binarização | Resolução nativa | Coleção/origem |
| --- | ---: | --- | --- | ---: | --- |
| **MapBiomas** | 2025 | Raster categórico GEE | `classification == 46` (classe 3.2.2.1 — Café) | 30 m (reamostrada para 10 m por vizinho mais próximo) | `projects/mapbiomas-public/assets/brazil/lulc/v1`, `collection_id = 11` |
| **AlphaEarth** | 2024 | Raster probabilístico GEE | `probability >= 0.28` | 10 m | `projects/forestdatapartnership/assets/coffee/model_2025b` |
| **Emater** | 2018 | Vetor local (GeoJSON por município) | Rasterização das glebas de café (0/1) | 10 m | Geoportal do Café / Emater-MG, `data/external/emater/` |

Detalhes:

- **MapBiomas:** a coleção consolidada é filtrada por `collection_id` (11) e `year` (2025). A imagem `classification` é comparada à classe 46 e recortada ao AOI. A exportação usa vizinho mais próximo para preservar valores categóricos ao passar de 30 m para 10 m.
- **AlphaEarth:** o modelo 2025b do Forest Data Partnership (embeddings AlphaEarth Foundations) fornece a banda `probability`; a máscara é `probability >= 0.28`. O limiar é calibrável e fica embutido no nome do arquivo (`_t028`), de modo que recalibrar o limiar gera um novo GeoTIFF em vez de reutilizar silenciosamente a máscara antiga.
- **Emater:** são 9 arquivos GeoJSON, um por município da RGI de Guaxupé (Arceburgo, Cabo Verde, Guaranésia, Guaxupé, Juruaia, Monte Belo, Muzambinho, Nova Resende e São Pedro da União). O conjunto tem **34.985 polígonos** e aproximadamente **603 km²** de café. O mapeamento foi iniciado em 2016 e concluído em 2018, com validação em campo pelos extensionistas da Emater-MG. É um snapshot estático de ~2018, sem atributo de ano nos arquivos; por isso, `year: 2018` é declarado no `config.yaml`.

Tratamento da fonte vetorial Emater:

1. concatenação dos GeoJSONs dos municípios;
2. reprojeção para `EPSG:31983`;
3. reparo de geometrias inválidas com `buffer(0)`;
4. clip ao polígono do AOI;
5. rasterização no grid de 10 m do AOI, com `all_touched=False`;
6. persistência em GeoTIFF `uint8` com compressão `deflate`.

A máscara binária de cada fonte é armazenada em:

```text
data/interim/<fonte>/mask_<fonte>_310044_<ano>[_t<limiar>].tif
```

### 2.4 Comparação entre fontes e decisão manual

O estágio `03` compara **todas as fontes habilitadas entre si**, sem eleger uma referência. Para cada fonte, calcula área de café em km², parcela do AOI e pixels de café. Para cada par de fontes, sobre um grid comum de 10 m (a primeira fonte define o grid; as demais são reamostradas por vizinho mais próximo quando necessário), calcula a matriz de confusão:

- `both_coffee`: café em ambas;
- `only_reference`: café apenas na primeira fonte do par;
- `only_other`: café apenas na segunda;
- `both_non_coffee`: não café em ambas.

E as métricas de concordância:

$$
\text{Acordo global} = \frac{both + neither}{total}
$$

$$
IoU = \frac{both}{both + only_{ref} + only_{other}}
$$

$$
F1 = \frac{2 \cdot both}{2 \cdot both + only_{ref} + only_{other}}
$$

$$
Precisão = \frac{both}{both + only_{ref}}
$$

$$
Revocação = \frac{both}{both + only_{other}}
$$

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

em que \(p_o\) é o acordo observado e \(p_e\) é o acordo esperado por acaso, calculado a partir das marginais da tabela 2×2. Precisão e Revocação são direcionais: o ponto de vista é a primeira fonte do par, conforme a ordem em `config.yaml`.

O relatório é persistido em:

```text
artifacts/metrics/ground_truth/mask_sources_comparison_310044.json
```

e a figura multi-painel em:

```text
artifacts/figures/mask_sources_comparison_310044.png
```

O estágio `04` **não elege** fonte alguma. Ele apenas lê e valida a decisão manual do pesquisador em `ground_truth.chosen_source` e persiste um relatório de auditoria com a base métrica da decisão:

```text
artifacts/metrics/ground_truth/source_decision_310044.json
```

A escolha registrada no repositório é `emater`. Ela determina qual conjunto de patches é usado nos estágios `09` e `10`; todas as fontes, porém, têm máscara e patches próprios. A decisão é **manual e auditável**, não automática.

---

## 3. Arquitetura computacional e reprodutibilidade

### 3.1 Layout híbrido: notebooks + pacote `src/`

O projeto usa um **layout híbrido**:

- **notebooks Jupyter** em `notebooks/` são a camada de orquestração e visualização; executam em ordem numérica (`00` a `16`);
- **pacote Python** `src/` contém toda a lógica reutilizável (configuração, I/O, dados, perdas, métricas, avaliação, comparação, treinamento, modelos, XAI, síntese e publicação);
- **testes** em `tests/` cobrem `src/`;
- **scripts** em `scripts/` são utilitários locais de geração de tokens OAuth (Drive e GEE), não executados pelos notebooks.

Os notebooks não reimplementam lógica do pacote. Eles chamam funções de `src/` e resolvem todos os caminhos a partir de `src/config.py` / `src/config.yaml`. `ruff`, `mypy` e `pytest` têm como alvo apenas `src/` e `tests/`; notebooks são validados estruturalmente no CI.

### 3.2 Fonte única de configuração

O arquivo `src/config.yaml` é a fonte única de verdade. Ele concentra:

- AOI (código, malha, caminho);
- dados (coleção, bandas, datas, limiar de nuvem, tamanho de patch, `coffee_min_ratio`, `min_valid_fraction`, CRS, escala e `maxPixels`);
- fontes de ground truth (ano, coleção, classe/limiar, resolução nativa);
- projeto GEE;
- armazenamento (raiz `tcc/` e subpastas);
- número de dobras;
- pesos da perda;
- arquitetura dos modelos;
- aumentações;
- treinamento (LR, épocas, batch, weight decay, workers);
- semente global.

O `config.py` carrega o YAML sem cache, para refletir alterações do arquivo, e valida a presença das chaves obrigatórias de topo. O hash SHA-256 curto do `config.yaml` é usado como fingerprint de configuração em todas as etapas.

### 3.3 Execução agnóstica de plataforma

O mesmo notebook deve executar corretamente em **Google Colab** e **Kaggle**, com o mesmo protocolo (código, sementes, configuração e processamento). A execução em nuvem é obrigatória; o treinamento exige GPU.

A primeira célula de todo notebook baixa e executa `src/bootstrap.py` (somente stdlib) a partir do repositório público:

```text
https://raw.githubusercontent.com/jotap1101/tcc-lab/main/src/bootstrap.py
```

O bootstrap:

1. baixa o tarball de `main` do GitHub;
2. extrai `src/`, `data/external/` e `requirements-runtime.txt` para o workspace;
3. adiciona o workspace ao `sys.path`;
4. grava o SHA do commit de `main` em `src/.tcc_commit` (API do GitHub, com fallback Atom);
5. no Colab, cria/atualiza o espelho `MyDrive/tcc/repo/` de forma idempotente;
6. remove módulos `src.*` em cache para forçar a versão recém-baixada.

Toda detecção de plataforma, montagem do Drive e resolução de caminhos fica em `src/io.py`:

- **Colab:** montagem nativa do Drive em `/content/drive`;
- **Kaggle:** acesso via Google Drive API com token OAuth (`GDRIVE_TOKEN` ou `GDRIVE_TOKEN_FILE`) e cache local em `/kaggle/working/drive`;
- **local:** `DRIVE_ROOT` ou `~/MyDrive` (apenas desenvolvimento).

A raiz canônica de armazenamento é a pasta `tcc/` na raiz do Google Drive (`MyDrive/tcc/`). Os notebooks a criam e acessam sob demanda, incluindo todas as subpastas. Nada é persistido fora dela. Os caminhos são resolvidos por `resolve_storage_paths()` a partir do `config.yaml`.

### 3.4 Dependências pinadas

As versões de runtime são instaladas pela célula de dependências de cada notebook via `src.setup.install_runtime_requirements()`, a partir de `requirements-runtime.txt`:

| Pacote | Versão |
| --- | --- |
| `torch` | 2.14.0 |
| `torchvision` | 0.29.0 |
| `transformers` | 5.17.0 |
| `earthengine-api` | 1.7.43 |
| `geopandas` | 1.1.4 |
| `pyogrio` | 0.13.0 |
| `huggingface-hub` | 1.32.0 |
| `numpy` | 2.5.3 |
| `pandas` | 3.0.6 |
| `pyarrow` | 25.0.1 |
| `scipy` | 1.18.1 |
| `scikit-learn` | 1.9.1 |
| `matplotlib` | 3.11.2 |
| `seaborn` | 0.13.2 |
| `rasterio` | 1.5.1 |
| `pyyaml` | 6.0.3 |
| `google-api-python-client` | 2.200.0 |
| `google-auth-oauthlib` | 1.4.1 |

O `uv.lock` é commitado. A instalação é segura: se uma extensão compilada já carregada no kernel (`numpy`, `scipy`, `torch`, `torchvision`, `pandas`, `pyarrow`, `matplotlib`, `scikit-learn`, `rasterio`, `geopandas`) divergir da versão pinada, o kernel é reiniciado automaticamente para evitar inconsistência de ABI. Notebooks que carregam Transformers removem um `torchaudio` eventualmente incompatível (compilado para outra versão de CUDA), que impediria o import; o projeto não usa áudio.

### 3.5 Sementes e determinismo

A semente global é `42`. Em cada etapa:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

As flags determinísticas do PyTorch são:

```python
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

Nos estágios `10` (treino do SegFormer) e `11` (avaliação), usa-se `torch.use_deterministic_algorithms(True, warn_only=True)`, porque o upsampling bilinear do SegFormer não possui backward determinístico no CUDA no PyTorch. A checagem permanece ativa, mas troca erro por aviso apenas nessas operações. Sementes, cuDNN, ordem dos lotes e as demais operações permanecem determinísticos. A configuração efetiva é registrada em `run.meta.json` (`deterministic_algorithms.enabled` e `warn_only`).

O `num_workers` do `DataLoader` é `0`, o que garante determinismo bit-a-bit das aumentações e da ordem de carga. Isso reduz o paralelismo de I/O, mas privilegia a reprodutibilidade.

O determinismo é de **protocolo** e de execução na mesma plataforma; não se exige saída bit-a-bit entre hardwares diferentes.

### 3.6 Idempotência, fingerprints e versionamento

As operações de armazenamento são idempotentes e re-executáveis. Cada artefato derivado é reutilizado apenas quando seus fingerprints coincidem com as entradas atuais; caso contrário, é regenerado no lugar. Os principais fingerprints são:

| Fingerprint | Conteúdo |
| --- | --- |
| **Manifesto** | `schema_version`, `region_code`, `source_name`, `year`, bandas, `patch_size`, `coffee_min_ratio`, `min_valid_fraction`, `mask_source`, `tile_id`, nome e tamanho do composite e da máscara; SHA-256 curto (12 hex) do JSON canônico |
| **Split** | `schema_version`, `source_name`, fingerprint do manifesto, `bbox_schema_version`, `fold_count`, `seed`; hash estável de todos os pares `patch_id:fold` |
| **Configuração** | SHA-256 curto (12 hex) dos bytes de `src/config.yaml` |
| **Pesos** | SHA-256 curto (12 hex) do arquivo de pesos da dobra |
| **Avaliação** | `schema_version`, `threshold`, fingerprint do manifesto, fingerprint do split, hash da configuração e hash dos pesos |
| **Comparação** | `schema_version`, fonte, threshold, fingerprint da avaliação, fingerprints e hashes dos pesos dos dois modelos |
| **XAI** | `schema_version`, modelo, camada/escopo, fingerprint da avaliação, fingerprints do manifesto/split/config, threshold, hashes dos pesos das dobras usadas e patches explicados |
| **Síntese** | `schema_version`, fonte, hash da configuração, fingerprints dos manifestos, fingerprints das avaliações, digests das métricas/históricos de treino, fingerprint da comparação e fingerprints/mapas de XAI |
| **Release** | `schema_version`, fonte, hash da configuração, fingerprints do manifesto/split, hashes dos pesos, repositórios e visibilidade |

O manifesto tem `MANIFEST_SCHEMA_VERSION = 2` e `BBOX_SCHEMA_VERSION = 1`; o split tem `SPLIT_SCHEMA_VERSION = 2`. A versão do schema de `bbox` invalida manifestos gerados com o cálculo antigo, sem reenviar os patches.

Baselines históricos são preservados com sufixo `_v1` e nunca sobrescritos. O primeiro treino da U-Net (dataset sem filtro e antes da correção de `bbox`/split) está preservado como `emater_v1`, com metadados em `data/processed/archive_v1/` e figuras em `artifacts/figures/archive_v1/`; o `BASELINE.json` registra a limitação conhecida. O experimento final usa o dataset corrigido e filtrado.

### 3.7 Qualidade e CI

O CI (GitHub Actions) executa:

- `ruff check src tests`;
- `ruff format --check src tests`;
- `mypy src tests`;
- `pytest tests`;
- validação estrutural dos notebooks: nome `NN_verb_snake_case.ipynb`, markdown imediatamente acima de cada célula de código, células de código não vazias e ausência de outputs versionados.

Os notebooks são commitados com **outputs limpos**.

---

## 4. Pipeline completo, estágio por estágio

| Notebook | Fase | Entrada principal → saída principal |
| --- | --- | --- |
| `00_setup_environment.ipynb` | Setup | — → ambiente pronto, `src/` entregue, raiz `tcc/` resolvida, GEE autenticado, sementes fixadas |
| `01_gee_sentinel2_acquisition.ipynb` | Aquisição | malha IBGE → AOI → mosaicos Sentinel-2 por ano |
| `02_gee_reference_masks.ipynb` | Aquisição | fontes de ground truth → máscaras binárias de 10 m por fonte |
| `03_mask_sources_comparison.ipynb` | Ground truth | máscaras → diagnóstico comparativo de todas as fontes e pares |
| `04_source_decision.ipynb` | Ground truth | relatório comparativo → registro da decisão manual |
| `05_preprocessing.ipynb` | Pré-processamento | mosaicos por ano → composites alinhados e normalizados |
| `06_patch_generation.ipynb` | Dataset | composites + máscaras → patches 512×512 + manifesto por fonte |
| `07_spatial_kfold_split.ipynb` | Dataset | manifesto → coluna `fold` + metadata do split |
| `08_dataset_eda.ipynb` | Dataset | manifesto → estatísticas de normalização + EDA/sanidade |
| `09_train_unet.ipynb` | Treinamento | dataset da fonte escolhida → pesos, métricas e histórico da U-Net |
| `10_train_segformer.ipynb` | Treinamento | dataset da fonte escolhida → pesos, métricas e histórico do SegFormer |
| `11_evaluation.ipynb` | Avaliação | pesos por dobra → predições + IoU/F1/Precisão/Recall por pixel |
| `12_comparative_analysis.ipynb` | Avaliação | métricas por patch → comparação estatística + mapas de erro |
| `13_xai_gradcam_unet.ipynb` | XAI | U-Net → mapas Grad-CAM |
| `14_xai_attention_rollout_segformer.ipynb` | XAI | SegFormer → mapas Attention Rollout |
| `15_results_synthesis.ipynb` | Síntese | artefatos 06–14 → tabelas CSV/LaTeX + figuras |
| `16_export_release.ipynb` | Disseminação | patches + pesos → dataset e modelos no Hugging Face Hub |

### 4.1 Estágio 00 — Setup

- Bootstrap do workspace e entrega de `src/`, `data/external/` e `requirements-runtime.txt`.
- Instalação das versões pinadas.
- Detecção de plataforma e resolução da raiz `MyDrive/tcc/` com todas as subpastas.
- Autenticação do Earth Engine com a conta principal. No Colab, fluxo interativo `ee.Authenticate()`; no Kaggle, credencial headless em `GEE_CREDENTIALS` e `ee.Initialize(project=...)`.
- Fixação de sementes e flags determinísticas.
- Registro das versões do ambiente.
- Validação da malha IBGE e da configuração.

### 4.2 Estágio 01 — Aquisição Sentinel-2

- Carrega a malha IBGE, filtra `CD_RGI == 310044`, reprojeta para `EPSG:4326` e converte o polígono em geometria GEE.
- Lista os anos requeridos a partir das fontes habilitadas.
- Para cada ano, constrói a coleção Sentinel-2 filtrada por AOI, datas e `CLOUDY_PIXEL_PERCENTAGE < 20`, aplica a máscara `QA60` e seleciona as quatro bandas.
- Compõe o mosaico mediano e recorta ao AOI.
- Exporta GeoTIFF para uma pasta única na raiz do Drive (limitação do GEE) e depois realoca para o caminho canônico.
- Gera miniatura RGB (B4/B3/B2) do mosaico.
- Reexecuções reutilizam o GeoTIFF existente.

### 4.3 Estágio 02 — Máscaras de referência

- Para cada fonte habilitada, garante a máscara binária no caminho canônico.
- MapBiomas e AlphaEarth são construídos no GEE; Emater é rasterizada localmente.
- A exportação usa o CRS e a escala do config; a reamostragem padrão é vizinho mais próximo, preservando valores categóricos.
- O ano de referência de cada fonte vem de `ground_truth.sources.<fonte>.year`.
- Gera miniatura de cada máscara.
- Reexecuções reutilizam o GeoTIFF existente.

### 4.4 Estágio 03 — Comparação das fontes

- Exige pelo menos duas fontes habilitadas.
- Carrega todas as máscaras no grid comum de 10 m.
- Calcula área, parcela do AOI, matriz de confusão e métricas de concordância para todos os pares.
- Persiste o relatório JSON e a figura multi-painel.
- Não elege fonte.

### 4.5 Estágio 04 — Decisão manual

- Lê `ground_truth.chosen_source` e valida que a fonte está habilitada.
- Carrega o relatório do estágio `03`.
- Exibe a base métrica da decisão (área, parcela do AOI, concordância dos pares que envolvem a fonte escolhida).
- Persiste o relatório de auditoria da decisão.
- Não há seleção automática.

### 4.6 Estágio 05 — Pré-processamento

Para cada ano requerido:

- carrega o mosaico do estágio `01` e a máscara de referência do ano (a primeira fonte habilitada com aquele ano);
- se os grids coincidem (mesmo CRS, transform e dimensões), lê diretamente;
- caso contrário, reamostra cada banda do mosaico para o grid exato da máscara com interpolação **bilinear** e `nodata = NaN`;
- converte o valor digital (DN) em reflectância:

$$
\rho = \text{clip}(DN \times 10^{-4},\,0,\,1)
$$

- preserva `NaN` para pixels sem observação válida;
- persiste o composite em GeoTIFF `float32`, com compressão `deflate`, `predictor=3`, blocos 256×256.

O composite é armazenado em:

```text
data/processed/composites/composite_310044_<ano>.tif
```

A ordem das bandas segue `data.bands`: B2, B3, B4, B8.

### 4.7 Estágio 06 — Geração de patches e manifesto

Para cada fonte habilitada:

- abre o composite do ano da fonte e a máscara binária da própria fonte;
- exige que ambos estejam no mesmo grid;
- divide o raster em patches de `512 × 512` pixels, **sem sobreposição**;
- descarta as bordas parciais (`height // 512`, `width // 512`);
- para cada patch, calcula:
  - **pixels válidos:** pixels finitos em todas as bandas;
  - **`valid_fraction`:** fração de pixels válidos no patch;
  - **`coffee_ratio`:** proporção de café **apenas entre os pixels válidos**;
- descarta patches com `valid_fraction <= 0` ou `valid_fraction < 0.10`;
- aplica o filtro de café `coffee_ratio >= coffee_min_ratio`. No config atual, `coffee_min_ratio = 0.0`, portanto **não há filtro adicional por café**: todos os patches com fração válida suficiente são mantidos, inclusive patches sem café.

Cada patch é salvo como `.npy`:

```text
data/processed/patches/<fonte>/images/<fingerprint>/<patch_id>.npy
data/processed/patches/<fonte>/masks/<fingerprint>/<patch_id>.npy
```

O `patch_id` é determinístico:

```text
< tile_id >_r< row:04d >_c< col:04d >
```

com `tile_id = composite_310044_<ano>`.

A `bbox` de cada patch é calculada diretamente da transformação afim do raster, usando `row` e `col` do grid de patches e `patch_size`, no CRS do tile (`EPSG:31983`), no formato `minx,miny,maxx,maxy` em metros.

O manifesto Parquet é:

```text
data/processed/manifest_<fonte>.parquet
```

com as colunas:

| Coluna | Descrição |
| --- | --- |
| `patch_id` | identificador único e determinístico |
| `tile_id` | identificador do composite/tile |
| `fold` | dobra espacial (`None` até o estágio 07) |
| `row`, `col` | índices do patch no grid |
| `bbox` | `minx,miny,maxx,maxy` em EPSG:31983 |
| `coffee_ratio` | proporção de café entre os pixels válidos |
| `valid_fraction` | fração de pixels válidos |
| `mask_source` | nome da fonte de ground truth |
| `image_path` | caminho relativo à raiz `tcc/` |
| `mask_path` | caminho relativo à raiz `tcc/` |

O metadata do manifesto (`manifest_<fonte>.meta.json`) registra o fingerprint, a versão do schema de `bbox`, as dimensões do grid, o número de patches, os patches descartados por pixels inválidos e o `min_valid_fraction`.

### 4.8 Estágio 07 — Divisão espacial k-fold

A divisão é feita **por proximidade geográfica**, não aleatoriamente:

1. calcula o centroide de cada `bbox` em coordenadas métricas (`EPSG:31983`):

$$
x_c = \frac{minx + maxx}{2}, \qquad y_c = \frac{miny + maxy}{2}
$$

2. agrupa os centroides com **k-means determinístico implementado em numpy puro**:
   - inicialização **k-means++** com `numpy.random.default_rng(42)`;
   - primeiro centro sorteado uniformemente;
   - centros seguintes sorteados com probabilidade proporcional à distância ao quadrado até o centro mais próximo;
   - até 100 iterações;
   - convergência quando os rótulos não mudam e os centros estabilizam;
3. reordena os rótulos dos grupos pela posição dos centros (`lexsort` por x e depois y), tornando o número da dobra estável e independente da ordem dos patches no manifesto;
4. grava a coluna `fold` no manifesto;
5. persiste `split_<fonte>.meta.json` com fingerprint do manifesto, `bbox_schema_version`, `fold_count` e `seed`.

O número de dobras é `splits.fold_count = 5` e a semente é `reproducibility.seed = 42`. Cada patch pertence a exatamente uma dobra. Em cada treino, uma dobra é validação e as outras quatro são treino. O agrupamento por centroides produz blocos espaciais contíguos, reduzindo o vazamento por autocorrelação espacial entre treino e validação.

O split é recalculado automaticamente quando o fingerprint do manifesto, o schema do split, o schema de `bbox`, o número de dobras ou a semente mudam.

### 4.9 Estágio 08 — EDA e estatísticas de normalização

Para cada fonte:

- percorre todos os patches do manifesto;
- para cada banda, acumula soma, soma dos quadrados, contagem de pixels finitos e contagem de `NaN`;
- calcula média, desvio padrão e fração de `NaN` por banda:

$$
\mu_b = \frac{\sum_i x_{b,i}}{N_b}, \qquad
\sigma_b = \sqrt{\frac{\sum_i x_{b,i}^2}{N_b} - \mu_b^2}
$$

em que \(N_b\) é o número de pixels finitos da banda \(b\);

- persiste `data/processed/normalization_stats_<fonte>.json` com fingerprint;
- executa verificações de sanidade: forma esperada `(4, 512, 512)`, dtype, máscara binária, distribuição de café, presença e balanceamento da coluna `fold`, unicidade de `patch_id`, amostra determinística de 10 patches;
- gera figura de resumo com distribuição de café, patches por dobra, média ± desvio por banda e histogramas de reflectância.

As estatísticas são calculadas sobre **todos os patches da fonte** (treino e validação), não por dobra. Isso deve ser declarado nas limitações: há uma leve transdução na normalização, embora os rótulos não sejam usados no cálculo.

### 4.10 Estágio 09 — Treino da U-Net

- Constrói a U-Net a partir de `model.unet_channels`.
- Exige GPU.
- Para cada dobra `0..4`:
  - fixa `seed + fold` antes de inicializar os pesos;
  - constrói um novo modelo;
  - chama `train_fold(unet, "unet", ...)`.
- O protocolo completo de treino está na Seção 8.
- Persiste pesos, métricas e histórico por dobra.
- Persiste `run.meta.json` com configuração, fonte, fingerprints, commit, ambiente e estado de determinismo.

### 4.11 Estágio 10 — Treino do SegFormer

- Mesmo protocolo do estágio `09`, com `build_segformer()`.
- Variante fixa `nvidia/mit-b1`.
- A primeira projeção é adaptada de 3 para 4 bandas.
- A cabeça de segmentação é inicializada do zero (o checkpoint é de backbone ImageNet).
- Os logits saem em 1/4 da resolução e são reamostrados bilinearmente para a resolução da entrada.
- Usa `setup_reproducibility(config, warn_only=True)` e remove `torchaudio` incompatível.
- Persiste pesos, métricas, histórico e `run.meta.json`.

### 4.12 Estágio 11 — Avaliação por pixel

Para cada modelo e cada dobra:

- carrega os pesos da dobra;
- reconstrói o modelo com `build_*()` e `load_state_dict(weights_only=True)`;
- executa inferência sobre os patches da dobra de validação;
- aplica `sigmoid` e limiar fixo `0.5`;
- ignora pixels com rótulo `-1`;
- calcula TP, FP, FN por patch e por dobra;
- persiste métricas por dobra, métricas por patch e predições binárias;
- consolida o resumo do modelo/fonte com agregado micro e média/desvio macro por dobra;
- gera figura de avaliação.

O estágio **não retreina**. Ele apenas avalia os pesos existentes.

### 4.13 Estágio 12 — Comparação estatística

- Emparelha as métricas por patch dos dois modelos pelo `patch_id`.
- Valida que o conjunto de patches, a dobra, o `coffee_ratio` e a contagem de positivos reais (`TP + FN`) coincidem.
- Calcula a diferença sempre como **SegFormer − U-Net**.
- Para cada métrica, calcula IC bootstrap, teste t pareado, Wilcoxon, Holm, \(d_z\) de Cohen, delta de Cliff e contagem de vitórias/empates/derrotas.
- Calcula diferenças por dobra e diferenças agregadas micro.
- Gera figura comparativa e mapas de erro TP/FP/FN/ignorado de patches representativos.
- Persiste `artifacts/metrics/comparative/<fonte>/comparison.json`.

### 4.14 Estágio 13 — Grad-CAM da U-Net

- Usa os pesos da dobra de cada patch (modelo que não viu aquele patch no treino).
- Camada-alvo: `encoders[-1]` (codificador mais profundo da U-Net, 256 canais, resolução 1/8 da entrada para patches 512×512).
- Pontuação: soma dos logits da classe café na região de referência do patch (pixels de café válidos).
- Calcula gradientes da pontuação em relação à ativação da camada-alvo.
- Agrega os gradientes por média espacial, pondera as ativações, aplica ReLU, reamostra bilinearmente para 512×512 e normaliza em [0,1].
- Seleciona deterministicamente 4 patches representativos: maior IoU, menor IoU, maior `coffee_ratio` e IoU mais próximo da mediana; empates por `patch_id`.
- Persiste mapas, metadata com fingerprint e figura.
- Reutiliza apenas quando o resumo do estágio `11`, os pesos das dobras e a configuração coincidem.
- Usa GPU quando disponível, com fallback para CPU.

### 4.15 Estágio 14 — Attention Rollout do SegFormer

- Usa os pesos da dobra de cada patch (modelo que não viu aquele patch no treino).
- Extrai atenções com `output_attentions=True` em implementação `eager` (a `sdpa` não expõe atenções).
- Restringe o rollout à **última etapa do codificador hierárquico**, a única com atenção quadrada (`sr_ratio = 1`).
- Agrega as cabeças pela média, soma a identidade (conexões residuais), normaliza as linhas e compõe as camadas.
- Como o SegFormer não possui token CLS, a relevância de cada token é a média das linhas do rollout.
- Reamostra o mapa para 512×512 e normaliza em [0,1].
- Seleciona os mesmos 4 patches representativos do estágio `13`.
- Persiste mapas, metadata com fingerprint e figura.
- Reutiliza apenas quando o resumo do estágio `11`, os pesos das dobras e a configuração coincidem.
- Usa GPU quando disponível, com fallback para CPU.

Para a variante `nvidia/mit-b1` e patches 512×512, a última etapa tem grade espacial 16×16 (256 tokens), 2 camadas e 8 cabeças de atenção. A arquitetura MiT-B1 tem:

| Etapa | Dimensão | Profundidade | Cabeças | `sr_ratio` | Stride acumulado |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 2 | 1 | 8 | 4 |
| 2 | 128 | 2 | 2 | 4 | 8 |
| 3 | 320 | 2 | 4 | 2 | 16 |
| 4 | 512 | 2 | 8 | 1 | 32 |

### 4.16 Estágio 15 — Síntese

- Lê apenas artefatos vigentes; não retreina nem reavalia.
- Gera tabelas CSV e LaTeX:
  - `dataset`: resumo do dataset por fonte (patches, dobras, café, fração válida);
  - `training`: métricas e melhor época por modelo/dobra;
  - `evaluation_per_fold`: métricas e contagens por modelo/dobra;
  - `evaluation_aggregate`: agregado micro e média/desvio por dobra;
  - `comparative_paired`: estatísticas emparelhadas por métrica;
  - `comparative_per_fold`: delta por dobra;
  - `xai`: relevância dentro/fora da referência de café por patch explicado.
- Gera figuras:
  - curvas de treino (perda de treino, perda de validação, IoU de validação, F1 de validação, todas as dobras);
  - visão geral da avaliação/comparação (métricas agregadas, IoU por dobra, delta de IoU por dobra, diferença pareada com IC 95%);
  - relevância quantitativa de XAI (média dentro/fora e diferença por patch).
- Persiste em `artifacts/synthesis/<fonte>/` e `artifacts/figures/`.
- O fingerprint cobre manifestos, resumos do estágio `11`, digests de métricas/históricos de treino, comparação do estágio `12` e metadados/mapas de XAI. Qualquer mudança regenera a síntese.

A relevância de XAI é quantificada, para cada patch explicado, como:

$$
\text{relevância dentro} = \frac{1}{|\Omega_{café}|}\sum_{i \in \Omega_{café}} h_i
$$

$$
\text{relevância fora} = \frac{1}{|\Omega_{válido \setminus café}|}\sum_{i \in \Omega_{válido \setminus café}} h_i
$$

$$
\text{diferença} = \text{relevância dentro} - \text{relevância fora}
$$

em que \(h_i\) é o valor normalizado do mapa de XAI no pixel \(i\).

### 4.17 Estágio 16 — Publicação

- Valida manifesto, estatísticas de normalização, treinos e avaliações.
- Autentica no Hugging Face com `HF_TOKEN` (lido de env/Secrets; nunca impresso nem versionado).
- Cria três repositórios:
  - dataset: `<usuario>/tcc-coffee-<fonte>` (CC BY 4.0);
  - U-Net: `<usuario>/tcc-coffee-unet-<fonte>` (MIT);
  - SegFormer: `<usuario>/tcc-coffee-segformer-<fonte>` (MIT).
- Os IDs podem ser sobrescritos por `HF_DATASET_REPO`, `HF_UNET_REPO` e `HF_SEGFORMER_REPO`.
- Empacota o dataset com `images/<patch_id>.npy`, `masks/<patch_id>.npy`, `manifest.parquet` com caminhos relativos, `normalization_stats_<fonte>.json` e `README.md`.
- Empacota os modelos com `fold_i.pt`, `model_config.json` e `README.md`.
- A publicação é idempotente: reutiliza quando o fingerprint de dataset/pesos/configuração e os arquivos remotos esperados coincidem.
- Persiste a auditoria em `artifacts/release/<fonte>/release.json`, com repositórios, URLs de commit e métricas.

---

## 5. Formalização do dataset

### 5.1 Grid e patches

O composite de cada ano está no grid da máscara de referência, em `EPSG:31983`, com pixel de 10 m. Para a fonte Emater (2018), o grid efetivo é:

- largura: 8.192 pixels (81,92 km);
- altura: 6.656 pixels (66,56 km);
- grid de patches: 16 colunas × 13 linhas = 208 patches candidatos;
- patches de 512×512 pixels = 26,2144 km² cada;
- patches retidos após o filtro de `valid_fraction >= 0.10`: **125**;
- patches com café: **124**;
- patches sem café: **1**.

Área total dos patches (incluindo pixels inválidos): 125 × 26,2144 = **3.276,8 km²**.
Área válida (dentro do AOI): **≈ 2.776,7 km²** (98,6% da área do polígono do AOI).
Pixels de café no dataset: **5.962.685**, equivalentes a **596,27 km²**.

A diferença entre os ~603 km² da fonte vetorial e os 596,27 km² do dataset rasterizado decorre da rasterização, do recorte ao AOI e do descarte de patches com pouca área válida.

### 5.2 Fração válida e proporção de café

Para um patch \(p\) com pixels \((i,j)\):

$$
v_p = \frac{1}{512^2}\sum_{i,j}\prod_{b=1}^{4}\mathbb{1}\left[x_{b,i,j}\text{ é finito}\right]
$$

$$
c_p = \frac{1}{|\Omega_p|}\sum_{(i,j)\in\Omega_p}\mathbb{1}\left[y_{i,j} > 0\right]
$$

em que \(\Omega_p\) é o conjunto de pixels válidos do patch e \(y_{i,j}\) é o rótulo binário da máscara.

O patch é descartado se \(v_p < 0.10\). O filtro de café é \(c_p \ge 0.0\), ou seja, não descarta nenhum patch por ausência de café.

### 5.3 Distribuição efetiva do dataset Emater

| Estatística | Valor |
| --- | ---: |
| Patches | 125 |
| Patches com café | 124 |
| `coffee_ratio` mínimo | 0,0000 |
| `coffee_ratio` médio | 0,2150 |
| `coffee_ratio` mediano | 0,1986 |
| `coffee_ratio` máximo | 0,5954 |
| `valid_fraction` mínimo | 0,1330 |
| `valid_fraction` médio | 0,8474 |
| `valid_fraction` mediano | 1,0000 |
| `valid_fraction` máximo | 1,0000 |

Distribuição por dobra:

| Dobra | Patches | `coffee_ratio` médio | `valid_fraction` médio | Centroide aproximado (x, y) em m |
| ---: | ---: | ---: | ---: | --- |
| 0 | 18 | 0,1032 | 0,8037 | (304.918, 7.641.019) |
| 1 | 21 | 0,1985 | 0,8457 | (329.624, 7.661.296) |
| 2 | 25 | 0,2167 | 0,8554 | (331.838, 7.639.733) |
| 3 | 30 | 0,2918 | 0,8022 | (355.492, 7.661.613) |
| 4 | 31 | 0,2156 | 0,9112 | (357.292, 7.630.623) |

As dobras não são perfeitamente balanceadas em número de patches nem em proporção de café. Isso é esperado em uma divisão espacial por blocos contíguos e deve ser discutido como característica do delineamento, não como falha.

### 5.4 Normalização por banda

As estatísticas efetivas do dataset Emater são:

| Banda | Média | Desvio padrão | Fração de `NaN` |
| --- | ---: | ---: | ---: |
| B2 | 0,039683 | 0,015657 | 15,26% |
| B3 | 0,069281 | 0,023501 | 15,26% |
| B4 | 0,061667 | 0,034175 | 15,26% |
| B8 | 0,321997 | 0,061097 | 15,26% |

A normalização é z-score por banda:

$$
\tilde{x}_{b,i} = \frac{x_{b,i} - \mu_b}{\sigma_b}
$$

Antes da padronização, pixels `NaN` são imputados com a média da banda. Bandas sem estatística finita ou com desvio zero são zeradas. Após a normalização, pixels inválidos ficam com valor 0 na imagem, mas recebem rótulo `-1` na máscara e são excluídos da perda e das métricas.

### 5.5 Aumentações

As aumentações são geométricas, aplicadas ao par (imagem, máscara) com as mesmas decisões:

| Transformação | Probabilidade |
| --- | ---: |
| Rotação de 90° | 0,25 |
| Flip horizontal | 0,50 |
| Flip vertical | 0,50 |

A ordem é: rotação de 90°, flip horizontal, flip vertical. O gerador é um `torch.Generator` próprio, reseedado a cada época com `seed + fold + época`. Com `num_workers = 0`, o treino é bit-reproduzível.

As aumentações são aplicadas apenas ao conjunto de treino; a validação não é aumentada.

---

## 6. Modelos

### 6.1 U-Net (CNN)

A U-Net implementada é uma encoder-decoder com conexões de salto (skip connections), batch normalization e ReLU. A entrada tem 4 canais (B2, B3, B4, B8) e a saída tem 1 canal de logit binário.

Bloco `DoubleConv`:

- `Conv2d(3×3, padding=1, bias=False)`;
- `BatchNorm2d`;
- `ReLU`;
- `Conv2d(3×3, padding=1, bias=False)`;
- `BatchNorm2d`;
- `ReLU`.

Codificador:

| Nível | Canais de saída | Resolução para entrada 512×512 |
| --- | ---: | ---: |
| Encoder 1 | 32 | 512×512 |
| Encoder 2 | 64 | 256×256 |
| Encoder 3 | 128 | 128×128 |
| Encoder 4 | 256 | 64×64 |
| Bottleneck | 512 | 32×32 |

Cada nível do codificador é seguido por `MaxPool2d(2, 2)`. O bottleneck dobra o último canal (256 → 512).

Decodificador:

- `ConvTranspose2d(kernel=2, stride=2)` para dobrar a resolução;
- concatenação com o skip correspondente;
- `DoubleConv` para fundir os canais;
- repetição para os níveis 256, 128, 64 e 32;
- `Conv2d(1×1)` final para 1 logit.

Número de parâmetros: **7.763.329**.

A U-Net é treinada **do zero**, sem pesos pré-treinados. Isso deve ser declarado, pois o SegFormer usa backbone ImageNet e a comparação não é pareada em pré-treinamento.

### 6.2 SegFormer (ViT)

O SegFormer é carregado via Hugging Face Transformers:

```python
SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b1",
    config=config_com_num_labels_1,
    ignore_mismatched_sizes=True,
)
```

A configuração é ajustada para `num_labels = 1`, `id2label = {0: "coffee"}` e `label2id = {"coffee": 0}`.

A primeira projeção de entrada (`model.segformer.stages[0].patch_embeddings.proj`) é adaptada de 3 para 4 canais:

- os pesos RGB pré-treinados são preservados;
- o quarto canal é inicializado com a **média dos três filtros RGB**;
- o viés é copiado;
- `model.config.num_channels = 4`.

O checkpoint `nvidia/mit-b1` é de backbone (ImageNet); a cabeça de decodificação é inicializada aleatoriamente e ajustada no treino.

A saída do `SegformerForSemanticSegmentation` tem logits em **1/4 da resolução** da entrada. O wrapper `SegFormerBinary` reamostra esses logits para a resolução da entrada com interpolação bilinear (`align_corners=False`). A entrada deve ter altura e largura múltiplas de 32 (stride total do MiT); 512 é múltiplo de 32.

Arquitetura MiT-B1:

| Etapa | Dimensão | Profundidade | Cabeças | `sr_ratio` | Stride acumulado | Resolução para 512×512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 2 | 1 | 8 | 4 | 128×128 |
| 2 | 128 | 2 | 2 | 4 | 8 | 64×64 |
| 3 | 320 | 2 | 4 | 2 | 16 | 32×32 |
| 4 | 512 | 2 | 8 | 1 | 32 | 16×16 |

Número de parâmetros com entrada de 4 canais e 1 rótulo: **13.680.641**.

### 6.3 Protocolo justo de comparação

Os dois modelos usam:

- o mesmo conjunto de patches da fonte escolhida;
- a mesma divisão espacial k-fold;
- as mesmas estatísticas de normalização;
- a mesma função de perda;
- o mesmo otimizador;
- o mesmo scheduler;
- as mesmas sementes por dobra;
- as mesmas métricas e o mesmo limiar de decisão;
- as mesmas aumentações.

A única diferença deliberada é a arquitetura. Há, porém, uma diferença não controlada: a U-Net é treinada do zero e o SegFormer usa backbone pré-treinado em ImageNet. Essa assimetria deve ser explicitada nas limitações.

---

## 7. Função de perda

A perda é multivariada e combina três termos: Dice, Focal e Boundary. Os pesos são definidos em `config.yaml`:

```yaml
loss:
  dice_weight: 1.0
  focal_weight: 1.0
  boundary_weight: 1.0
```

A perda total é a média ponderada normalizada pela soma dos pesos:

$$
\mathcal{L} = \frac{w_D \mathcal{L}_{Dice} + w_F \mathcal{L}_{Focal} + w_B \mathcal{L}_{Boundary}}{w_D + w_F + w_B}
$$

Com os pesos atuais, \(\mathcal{L} = (\mathcal{L}_{Dice} + \mathcal{L}_{Focal} + \mathcal{L}_{Boundary})/3\).

Em todas as componentes, pixels com rótulo \(-1\) são ignorados por uma máscara \(m_i = \mathbb{1}[t_i \ge 0]\). O alvo binário é \(y_i = \max(0, t_i)\).

### 7.1 Dice Loss

$$
\mathcal{L}_{Dice} = 1 - \frac{2\sum_i p_i y_i m_i + \epsilon}{\sum_i p_i m_i + \sum_i y_i m_i + \epsilon}
$$

com \(p_i = \sigma(z_i)\), \(\epsilon = 1\) (suavização para evitar divisão por zero) e \(z_i\) o logit previsto. A Dice mede a sobreposição entre a probabilidade prevista e a máscara binária, diretamente na classe de interesse.

### 7.2 Focal Loss

$$
\mathcal{L}_{Focal} = \frac{1}{\sum_i m_i}\sum_i m_i \, \alpha_{t,i} \, (1 - p_{t,i})^\gamma \, \mathrm{BCE}(p_i, y_i)
$$

com:

$$
p_{t,i} = p_i y_i + (1 - p_i)(1 - y_i)
$$

$$
\alpha_{t,i} = \alpha y_i + (1 - \alpha)(1 - y_i)
$$

$$
\alpha = 0.25, \qquad \gamma = 2.0
$$

A Focal Loss reduz o peso de exemplos fáceis e concentra o aprendizado em pixels difíceis, o que é útil no desbalanceamento entre café e não café. As probabilidades são limitadas a \([10^{-6}, 1 - 10^{-6}]\) por estabilidade numérica.

### 7.3 Boundary Loss

A Boundary Loss é baseada no mapa de distância assinada da fronteira do objeto, conforme Kervadec et al. (2019). O mapa \(D_i\) é calculado em CPU com `scipy.ndimage.distance_transform_edt`:

- pixels dentro do objeto recebem distância negativa até a borda;
- pixels fora recebem distância positiva até o objeto;
- cenas uniformes (objeto ausente ou objeto preenchendo toda a cena) recebem mapa zero.

$$
\mathcal{L}_{Boundary} = \frac{\sum_i m_i \, D_i \, p_i}{\sum_i m_i \, |D_i|}
$$

Se o denominador for zero, a perda é zero. O mapa de distância é constante em relação aos parâmetros do modelo; o gradiente flui apenas pelas probabilidades previstas. A Boundary Loss penaliza com mais força erros próximos à fronteira, o que é relevante para a delimitação de talhões e carreadores.

### 7.4 Observações de implementação

- A Dice é calculada em nível de batch (soma sobre todos os pixels do batch), não por imagem.
- A Focal é normalizada pelo número de pixels válidos do batch.
- A Boundary é calculada por amostra e depois agregada pela soma dos pixels válidos.
- Não há peso de classe explícito além do `alpha` da Focal.
- A perda de validação usa os mesmos três termos, inclusive a Boundary.

---

## 8. Protocolo de treinamento

### 8.1 Hiperparâmetros

| Parâmetro | Valor |
| --- | --- |
| Otimizador | `torch.optim.Adam` |
| Learning rate inicial | \(1 \times 10^{-4}\) |
| Weight decay | \(1 \times 10^{-4}\) |
| Scheduler | `CosineAnnealingLR`, `T_max = epochs` |
| Épocas | 100 |
| Batch size | 16 |
| `num_workers` | 0 |
| Semente global | 42 |
| Semente por dobra | `42 + fold` |
| Semente das aumentações | `42 + fold + época` |
| Precisão | `float32` (sem AMP) |
| Gradiente | retropropagação padrão, sem clipping |
| Early stopping | não há |
| Critério de melhor modelo | maior IoU de validação na época |
| Pesos salvos | estado do melhor epoch |

### 8.2 Laço por dobra

Para cada dobra \(f \in \{0,1,2,3,4\}\):

1. fixa a semente `42 + f`;
2. constrói um modelo novo;
3. carrega os `DataLoader`s de treino (dobras \(\neq f\)) e validação (dobra \(f\));
4. para cada época \(e = 1 \ldots 100\):
   - reseeda as aumentações com `42 + f + e`;
   - executa uma época de treino;
   - atualiza o scheduler;
   - executa uma época de validação;
   - registra loss e métricas no histórico;
   - se o IoU de validação for maior que o melhor até então, guarda o estado do modelo, a época e as métricas;
5. restaura o melhor estado;
6. persiste pesos, métricas e histórico.

O scheduler é `CosineAnnealingLR` com `T_max = 100`. O learning rate da época é capturado **antes** do passo do scheduler, de modo que o histórico registre o LR efetivamente usado.

### 8.3 Artefatos por dobra

| Artefato | Caminho |
| --- | --- |
| Pesos | `models/<modelo>/<fonte>/fold_<i>.pt` |
| Métricas de treino/validação | `artifacts/metrics/<modelo>/<fonte>/fold_<i>.json` |
| Histórico por época | `artifacts/runs/<modelo>/<fonte>/fold_<i>.json` |
| Metadata da execução | `artifacts/runs/<modelo>/<fonte>/run.meta.json` |

O JSON de métricas registra:

- `model`, `fold`, `epochs`, `seed`, `best_epoch`;
- `manifest_fingerprint_hash`, `split_fingerprint_hash`, `config_hash`;
- métricas de treino e validação no melhor epoch (`loss`, `dice`, `focal`, `boundary`, `iou`, `f1`, `precision`, `recall`).

O histórico registra, por época:

- `epoch`, `lr`;
- `train_loss`, `train_dice`, `train_focal`, `train_boundary`, `train_iou`, `train_f1`, `train_precision`, `train_recall`;
- `val_loss`, `val_dice`, `val_focal`, `val_boundary`, `val_iou`, `val_f1`, `val_precision`, `val_recall`.

O `run.meta.json` registra:

- `model`, `source_name`, `year`, `fold_count`;
- `platform`, `config_hash`, `commit_sha`;
- `seed`, `epochs`, `batch_size`, `lr`, `weight_decay`, `num_workers`;
- `model_config`;
- `deterministic_algorithms.enabled` e `warn_only`;
- `manifest_fingerprint_hash`, `split_fingerprint_hash`;
- versões de `torch`, `torchvision`, `transformers`, `numpy`, `pandas`.

### 8.4 Reuso e retreinamento

Uma dobra é reutilizada apenas se pesos, métricas e histórico existirem e se as métricas salvas tiverem:

- o mesmo `manifest_fingerprint_hash`;
- o mesmo `split_fingerprint_hash`;
- o mesmo `config_hash`.

Se qualquer um mudar, a dobra é retreinada automaticamente. Isso garante que pesos antigos não sejam usados com um dataset, split ou configuração diferentes.

### 8.5 Pseudoalgoritmo

```text
para cada modelo em {U-Net, SegFormer}:
    para cada fold em 0..4:
        seed = 42 + fold
        fixar(seed)
        modelo = construir_modelo()
        treino, validação = loaders(fold)
        melhor_iou = -inf
        para época em 1..100:
            reseed_augmentations(seed + época)
            treinar_uma_época(modelo, treino)
            scheduler.step()
            métricas_val = validar(modelo, validação)
            se métricas_val.iou > melhor_iou:
                melhor_iou = métricas_val.iou
                salvar_estado(modelo)
        restaurar_melhor_estado()
        persistir(pesos, métricas, histórico)
```

---

## 9. Avaliação

### 9.1 Métricas

Para a classe café, a partir das contagens de confusão:

$$
IoU = \frac{TP}{TP + FP + FN}
$$

$$
F1 = \frac{2TP}{2TP + FP + FN}
$$

$$
Precisão = \frac{TP}{TP + FP}
$$

$$
Revocação = \frac{TP}{TP + FN}
$$

Pixels com rótulo \(-1\) são excluídos de \(TP\), \(FP\) e \(FN\). Quando o denominador é zero, a métrica é definida como \(1.0\) (classe vazia considerada perfeita), evitando divisão por zero. Essa convenção deve ser declarada, pois afeta patches sem café.

O limiar de decisão é fixo em \(0.5\): \(\hat{y}_i = \mathbb{1}[\sigma(z_i) \ge 0.5]\).

### 9.2 Níveis de agregação

- **Por patch:** TP, FP, FN, IoU, F1, Precisão e Revocação para cada patch.
- **Por dobra:** soma de TP/FP/FN de todos os patches da dobra e cálculo das métricas (agregado micro dentro da dobra).
- **Agregado micro do modelo:** soma de TP/FP/FN de todas as dobras e cálculo das métricas. É o número principal de desempenho global.
- **Macro por dobra:** média e desvio padrão das métricas das cinco dobras. O desvio usa a população (`np.std`, `ddof=0`).

### 9.3 Artefatos

Para cada modelo/fonte/dobra:

```text
artifacts/metrics/<modelo>/<fonte>/evaluation/fold_<i>.json
artifacts/metrics/<modelo>/<fonte>/evaluation/patch_metrics_fold_<i>.parquet
artifacts/metrics/<modelo>/<fonte>/evaluation/predictions_fold_<i>.npz
```

E o resumo consolidado:

```text
artifacts/metrics/<modelo>/<fonte>/evaluation/summary.json
```

O `.npz` de predições contém `patch_ids` e `predictions` binárias (`uint8`). As métricas por patch contêm `patch_id`, `fold`, `coffee_ratio`, `valid_fraction`, `tp`, `fp`, `fn`, `iou`, `f1`, `precision`, `recall`.

A figura de avaliação contém:

- métricas por dobra;
- histograma de IoU por patch (patches com café);
- IoU vs `coffee_ratio`;
- IoU vs `valid_fraction`.

### 9.4 Reuso

A avaliação de uma dobra é reutilizada apenas se o fingerprint incluir:

- `schema_version`;
- `threshold`;
- fingerprint do manifesto;
- fingerprint do split;
- hash da configuração;
- hash SHA-256 dos pesos.

Se os pesos mudarem, a dobra é reavaliada.

---

## 10. Comparação estatística entre U-Net e SegFormer

### 10.1 Emparelhamento

As métricas por patch dos dois modelos são emparelhadas por `patch_id`. Antes do cálculo, o pipeline valida:

- conjuntos de patches idênticos;
- mesma dobra para cada patch;
- mesmo `coffee_ratio`;
- mesma contagem de positivos reais (`TP + FN`), ou seja, mesma máscara de referência.

A diferença é sempre:

$$
d_i = m_i^{SegFormer} - m_i^{U-Net}
$$

para \(m \in \{IoU, F1, Precisão, Revocação\}\). Valores positivos favorecem o SegFormer; negativos favorecem a U-Net.

### 10.2 Intervalo de confiança bootstrap

Para cada métrica, o pipeline calcula a média das diferenças e um IC de 95% por bootstrap percentil:

- 2.000 reamostragens com reposição;
- semente `42 + índice da métrica` (IoU = 42, F1 = 43, Precisão = 44, Revocação = 45);
- intervalos nos percentis 2,5% e 97,5% das médias reamostradas.

### 10.3 Teste t pareado

$$
t = \frac{\bar{d}}{s_d / \sqrt{n}}
$$

com \(n\) pares e \(s_d\) o desvio padrão das diferenças. O p-valor é bicaudal. O teste pressupõe, de forma aproximada, normalidade das diferenças; com \(n = 125\) patches, o Teorema Central do Limite ajuda, mas a suposição deve ser discutida.

### 10.4 Wilcoxon dos postos sinalizados

O teste de Wilcoxon é o teste **principal** para a comparação, por não pressupor normalidade. É aplicado às diferenças, com `zero_method="wilcox"` (descarta diferenças nulas). O p-valor é bicaudal.

### 10.5 Correção de Holm–Bonferroni

Como são testadas quatro métricas, os p-valores de Wilcoxon são corrigidos por Holm–Bonferroni:

1. ordenam-se os p-valores;
2. para o rank \(k\) (1-indexado), calcula-se \((m - k + 1)p_{(k)}\), com \(m = 4\);
3. aplica-se o máximo acumulado para garantir monotonicidade;
4. limita-se a 1,0.

A correção controla a taxa de erro família para as quatro métricas.

### 10.6 Tamanhos de efeito

**\(d_z\) de Cohen (amostras pareadas):**

$$
d_z = \frac{\bar{d}}{s_d}
$$

com \(s_d\) calculado com \(n-1\) no denominador. Se \(s_d \le 10^{-12}\), o valor é indefinido (`None`).

**Delta de Cliff:**

$$
\delta = P(S > U) - P(S < U)
$$

O delta de Cliff é calculado sobre **todas as comparações entre patches dos dois modelos**, não apenas os pares posicionais. Ele é uma medida ordinal de dominância. A diferença emparelhada por `patch_id` é avaliada pelos testes t e Wilcoxon.

Interpretação convencional (a ser usada com cautela no texto):

- \(|d_z| < 0,2\): efeito desprezível; \(0,2\)–\(0,5\): pequeno; \(0,5\)–\(0,8\): médio; \(>0,8\): grande.
- \(|\delta| < 0,147\): desprezível; \(0,147\)–\(0,33\): pequeno; \(0,33\)–\(0,474\): médio; \(>0,474\): grande.

### 10.7 Vitórias, empates e derrotas

Para cada métrica, o pipeline conta:

- `segformer_better`: \(d_i > 10^{-12}\);
- `unet_better`: \(d_i < -10^{-12}\);
- `ties`: os demais.

### 10.8 Diferenças por dobra e agregado

- **Delta agregado micro:** diferença das métricas micro (TP/FP/FN somados entre dobras).
- **Delta por dobra:** diferença das métricas de cada dobra, ordenada por dobra.

Esses números permitem verificar se o sinal da comparação é consistente entre regiões do AOI, e não apenas um efeito do agregado.

### 10.9 Mapas de erro

Os mapas de erro usam quatro patches representativos, escolhidos deterministicamente entre patches com café:

1. maior diferença de IoU (SegFormer − U-Net);
2. menor diferença de IoU;
3. maior IoU médio entre os modelos;
4. menor IoU médio entre os modelos.

Empates são desfeitos por `patch_id`. Cada linha da figura mostra:

- RGB (B4/B3/B2);
- máscara de referência;
- mapa de erro da U-Net;
- mapa de erro do SegFormer.

Códigos de cor:

| Código | Significado | Cor |
| ---: | --- | --- |
| 3 | Verdadeiro positivo (TP) | verde |
| 1 | Falso positivo (FP) | vermelho |
| 2 | Falso negativo (FN) | azul |
| 0 | Verdadeiro negativo (TN) | cinza |
| 4 | Ignorado (fora do AOI) | branco |

### 10.10 Limitações estatísticas

Os testes tratam os patches como unidades independentes. Como os patches têm autocorrelação espacial, essa suposição é apenas aproximada; a divisão espacial reduz o vazamento entre treino e validação, mas não elimina a dependência entre patches da mesma região. Os p-valores devem ser interpretados como evidência, não como prova definitiva. O IC bootstrap e os tamanhos de efeito ajudam a contextualizar a significância prática.

---

## 11. Explicabilidade (XAI)

### 11.1 Grad-CAM da U-Net

**Objetivo:** identificar as regiões da imagem que mais influenciam a predição da classe café.

**Camada-alvo:** `encoders[-1]`, o bloco do codificador mais profundo da U-Net. Para entrada 512×512, essa camada tem 256 canais e resolução 64×64 (1/8 da entrada). A escolha é feita no codificador, não no decodificador.

**Pontuação:** soma dos logits da classe café nos pixels da região de referência do patch (pixels de café válidos). Se a máscara de referência estiver vazia, usa-se a soma de todos os logits.

**Cálculo:**

1. registra-se um forward hook na camada-alvo para capturar a ativação \(A^k\);
2. executa-se o forward com gradiente habilitado;
3. calcula-se o gradiente da pontuação \(y^c\) em relação a \(A^k\);
4. os pesos dos canais são a média espacial dos gradientes:

$$
\alpha_k^c = \frac{1}{HW}\sum_{i,j}\frac{\partial y^c}{\partial A_{ij}^k}
$$

5. o mapa é:

$$
L^c = \mathrm{ReLU}\left(\sum_k \alpha_k^c A^k\right)
$$

6. o mapa é reamostrado bilinearmente para 512×512;
7. normaliza-se em [0,1] por min-max.

**Seleção de patches:** 4 patches com café, escolhidos deterministicamente por:

- maior IoU;
- menor IoU;
- maior `coffee_ratio`;
- IoU mais próximo da mediana.

Empates por `patch_id`.

**Modelo usado:** o modelo da dobra à qual o patch pertence. Como o patch está na dobra de validação desse modelo, ele não foi visto no treino. Isso evita vazamento entre treino e explicação.

**Artefatos:**

```text
artifacts/xai/gradcam/unet/<fonte>/gradcam_maps.npz
artifacts/xai/gradcam/unet/<fonte>/gradcam_metadata.json
artifacts/figures/gradcam_unet_310044_2018_emater.png
```

O `.npz` contém `patch_ids`, `folds`, `cams`, `iou` e `coffee_ratio`. O metadata contém modelo, fonte, dispositivo, camada-alvo, região de pontuação, patches e fingerprint.

### 11.2 Attention Rollout do SegFormer

**Objetivo:** propagar as matrizes de atenção do codificador para estimar a relevância de cada token espacial na predição.

**Escopo:** apenas a **última etapa do codificador hierárquico** (`last_stage`), a única com atenção quadrada (`sr_ratio = 1`). Para MiT-B1 e entrada 512×512, a última etapa tem grade 16×16 (256 tokens), 2 camadas e 8 cabeças.

**Extração:** o backbone é executado com `output_attentions=True` em implementação `eager`, pois a `sdpa` não expõe as atenções. Não há backward nem atualização de pesos.

**Cálculo por camada:**

1. média das cabeças:

$$
\hat{A}^{(l)} = \frac{1}{H}\sum_{h=1}^{H} A_h^{(l)}
$$

2. adição da identidade (conexões residuais):

$$
\tilde{A}^{(l)} = \hat{A}^{(l)} + I
$$

3. normalização das linhas:

$$
\bar{A}^{(l)}_{ij} = \frac{\tilde{A}^{(l)}_{ij}}{\sum_j \tilde{A}^{(l)}_{ij}}
$$

4. composição das camadas, na ordem de execução:

$$
R^{(L)} = \bar{A}^{(L)} \, \bar{A}^{(L-1)} \cdots \bar{A}^{(1)}
$$

5. como o SegFormer não tem token CLS, a relevância de cada token é a média das linhas:

$$
r_j = \frac{1}{N}\sum_i R_{ij}
$$

6. normaliza-se \(r\) para soma 1;
7. reorganiza-se o vetor de \(N\) tokens em uma grade \(\sqrt{N} \times \sqrt{N}\);
8. reamostra-se bilinearmente para 512×512;
9. normaliza-se em [0,1] por min-max.

**Seleção de patches:** os mesmos critérios determinísticos do Grad-CAM.

**Modelo usado:** o modelo da dobra à qual o patch pertence, sem vazamento de treino.

**Artefatos:**

```text
artifacts/xai/attention_rollout/segformer/<fonte>/attention_rollout_maps.npz
artifacts/xai/attention_rollout/segformer/<fonte>/attention_rollout_metadata.json
artifacts/figures/attention_rollout_segformer_310044_2018_emater.png
```

O metadata registra `rollout_scope`, `stage_index`, `stage_layers`, `total_layers`, `stage_grid`, `num_heads`, `head_reduction`, `token_reduction`, `residual`, `normalization`, `attention_implementation`, `input_shape` e `upsample_mode`.

### 11.3 Quantificação da relevância

Para cada patch explicado, a síntese calcula a média do mapa normalizado:

- dentro da região de referência de café;
- fora da referência, mas dentro dos pixels válidos;
- diferença entre as duas.

Essa quantificação permite comparar, de forma agregada, se os mapas concentram relevância na classe de interesse. É uma análise descritiva; não substitui uma validação humana ou uma métrica de fidelidade causal.

---

## 12. Síntese e disseminação

### 12.1 Tabelas da síntese

| Tabela | Conteúdo |
| --- | --- |
| `dataset` | por fonte: ano, se é a escolhida, patches, dobras, média/mediana de café, média/mínimo da fração válida |
| `training` | por modelo/dobra: épocas, melhor época, loss e métricas de treino/validação |
| `evaluation_per_fold` | por modelo/dobra: patches, TP/FP/FN, IoU/F1/Precisão/Revocação |
| `evaluation_aggregate` | por modelo: agregado micro, média e desvio por dobra |
| `comparative_paired` | por métrica: pares, diferença média/mediana, IC95, p-valores, \(d_z\), delta de Cliff, vitórias/empates/derrotas |
| `comparative_per_fold` | por dobra: delta das quatro métricas (SegFormer − U-Net) |
| `xai` | por modelo/patch: IoU, café, relevância dentro/fora, diferença, hash dos pesos, explicação |

As tabelas são persistidas em CSV e LaTeX (float com 4 casas, `na_rep="--"`).

### 12.2 Figuras da síntese

| Figura | Conteúdo |
| --- | --- |
| `synthesis_training_curves_310044_2018_emater.png` | perda de treino, perda de validação, IoU de validação e F1 de validação por época, todas as dobras, cores por modelo |
| `synthesis_evaluation_overview_310044_2018_emater.png` | métricas agregadas micro, IoU por dobra, delta de IoU por dobra e diferença pareada com IC95 |
| `synthesis_xai_relevance_310044_2018_emater.png` | relevância média dentro/fora da referência de café e diferença por patch explicado |

### 12.3 Publicação

O release publica:

- **dataset** (CC BY 4.0): patches, máscaras, manifesto com caminhos relativos, estatísticas de normalização e dataset card;
- **U-Net** (MIT): pesos por dobra, `model_config.json` e model card;
- **SegFormer** (MIT): pesos por dobra, `model_config.json` e model card.

O `model_config.json` registra arquitetura, bandas, tamanho de patch, threshold, métricas por dobra, agregado, média/desvio, fingerprint, hiperparâmetros de treino, pesos da perda e licença.

O `release.json` registra repositórios, URLs, commits, estatísticas do dataset, métricas dos modelos e fingerprint.

---

## 13. Receita experimental consolidada

| Item | Valor |
| --- | --- |
| Tarefa | Segmentação semântica binária (café vs. não café) |
| Região | RGI de Guaxupé, MG, IBGE `310044` |
| Área do AOI | 2.814,87 km² |
| Sensor | Sentinel-2 Nível-2A harmonizado |
| Bandas | B2, B3, B4, B8 |
| Resolução | 10 m |
| Coleção | `COPERNICUS/S2_SR_HARMONIZED` |
| Filtro de nuvem | `CLOUDY_PIXEL_PERCENTAGE < 20` |
| Máscara de nuvem | QA60 bits 10 e 11 |
| Composição | Mediana anual |
| CRS | EPSG:31983 |
| Patch | 512×512 px (26,2144 km²) |
| Sobreposição | 0% |
| Filtro de pixels válidos | `valid_fraction >= 0.10` |
| Filtro de café | `coffee_ratio >= 0.0` |
| Fontes de ground truth | MapBiomas 2025, AlphaEarth 2024, Emater 2018 |
| Fonte escolhida | Emater 2018 |
| Patches (Emater) | 125 (124 com café) |
| Dobras | 5 |
| Split | k-means++ determinístico sobre centroides das bboxes, seed 42 |
| Normalização | z-score por banda, média/desvio globais |
| Aumentações | rot90 0,25; hflip 0,50; vflip 0,50 |
| Modelo A | U-Net, canais [32, 64, 128, 256], 7.763.329 parâmetros, treinada do zero |
| Modelo B | SegFormer `nvidia/mit-b1`, 4 canais, 13.680.641 parâmetros, backbone ImageNet |
| Perda | Dice + Focal + Boundary, pesos 1:1:1 |
| Focal | \(\alpha=0.25\), \(\gamma=2.0\) |
| Otimizador | Adam |
| LR | \(1 \times 10^{-4}\) |
| Weight decay | \(1 \times 10^{-4}\) |
| Scheduler | CosineAnnealingLR, `T_max=100` |
| Épocas | 100 |
| Batch | 16 |
| Workers | 0 |
| Semente | 42 |
| Semente por dobra | 42 + fold |
| Critério de seleção | maior IoU de validação |
| Threshold de avaliação | 0,5 |
| Métricas | IoU, F1, Precisão, Revocação |
| Agregação | por patch, por dobra, micro e macro |
| Comparação | diferença SegFormer − U-Net, bootstrap 2.000, t pareado, Wilcoxon, Holm, \(d_z\), delta de Cliff |
| XAI U-Net | Grad-CAM em `encoders[-1]`, score na região de café |
| XAI SegFormer | Attention Rollout na última etapa, média de cabeças, resíduo, normalização de linhas, sem CLS |
| Publicação | Hugging Face Hub, dataset CC BY 4.0, modelos MIT |

---

## 14. Estrutura de artefatos

A árvore canônica é criada pelos notebooks sob `MyDrive/tcc/`:

```text
MyDrive/tcc/
├── data/
│   ├── raw/sentinel2/                 # mosaicos Sentinel-2 por ano
│   ├── interim/<fonte>/               # máscaras binárias por fonte
│   ├── processed/
│   │   ├── composites/                # composites normalizados por ano
│   │   ├── patches/<fonte>/{images,masks}/<fingerprint>/
│   │   ├── manifest_<fonte>.parquet
│   │   ├── manifest_<fonte>.meta.json
│   │   ├── split_<fonte>.meta.json
│   │   └── normalization_stats_<fonte>.json
│   └── external/                      # malha IBGE e GeoJSONs Emater (versionados)
├── models/<modelo>/<fonte>/fold_<i>.pt
├── artifacts/
│   ├── metrics/
│   │   ├── <modelo>/<fonte>/fold_<i>.json
│   │   ├── <modelo>/<fonte>/evaluation/...
│   │   ├── comparative/<fonte>/comparison.json
│   │   └── ground_truth/...
│   ├── runs/<modelo>/<fonte>/fold_<i>.json
│   ├── runs/<modelo>/<fonte>/run.meta.json
│   ├── xai/gradcam/unet/<fonte>/...
│   ├── xai/attention_rollout/segformer/<fonte>/...
│   ├── synthesis/<fonte>/tables/...
│   ├── release/<fonte>/release.json
│   └── figures/...
├── repo/                              # espelho do src/ (Colab)
└── secrets/                           # tokens OAuth (nunca versionar)
```

Os caminhos locais `data/`, `models/` e `artifacts/` são espelhos git-ignored da árvore do Drive. A exceção é `data/external/`, versionado no repositório.

---

## 15. Decisões, limitações e ameaças à validade

Esta seção é essencial para a honestidade metodológica do capítulo. Ela não deve ser omitida.

### 15.1 Fonte de ground truth

- A escolha da fonte é manual e recaiu sobre **Emater 2018**. As fontes têm anos distintos (2018, 2024, 2025), o que impede uma comparação temporal direta entre elas.
- Emater é um snapshot de 2018, com validação de campo, mas pode não refletir o parque cafeeiro atual. O mosaico Sentinel-2 usado para a fonte também é de 2018, então imagem e rótulo são contemporâneos; ainda assim, o modelo treinado com 2018 pode ter limitações de generalização temporal.
- MapBiomas e AlphaEarth têm resoluções nativas e critérios de classe diferentes; a concordância entre fontes é medida, mas não há verdade de campo independente para arbitrar discordâncias.
- A decisão manual é auditável, mas não é isenta de subjetividade. O relatório do estágio `04` registra a base métrica usada.

### 15.2 Assimetria de pré-treinamento

- A U-Net é treinada **do zero**.
- O SegFormer usa backbone **pré-treinado em ImageNet** e cabeça de segmentação inicializada do zero.
- Isso confere uma vantagem potencial ao SegFormer e é um confundidor na comparação. O capítulo deve declarar que a comparação mede o desempenho das arquiteturas **como implementadas**, não um confronto com pré-treinamento controlado.

### 15.3 Seleção de modelo e viés otimista

- O melhor checkpoint de cada dobra é escolhido pelo **IoU de validação**.
- O estágio `11` avalia esse mesmo checkpoint na **mesma dobra de validação** usada para a seleção.
- Isso pode produzir uma estimativa otimista do desempenho: a dobra de validação participa da escolha do modelo.
- Não há conjunto de teste separado. A validação cruzada espacial é o único mecanismo de generalização.
- Uma alternativa mais rigorosa seria uma validação interna dentro de cada dobra de treino para seleção, reservando a dobra espacial apenas para teste. O pipeline atual não faz isso.

### 15.4 Normalização global

- As estatísticas de média/desvio são calculadas sobre **todos os patches da fonte**, incluindo os de validação.
- Isso é uma forma leve de transdução: os valores de normalização usam informação do conjunto de validação, embora não usem rótulos.
- O efeito tende a ser pequeno, mas deve ser declarado.

### 15.5 Dependência espacial e testes estatísticos

- A divisão espacial reduz o vazamento, mas os patches de uma mesma dobra continuam espacialmente correlacionados.
- Os testes t e Wilcoxon tratam os patches como independentes; a autocorrelação espacial pode inflar a significância.
- Os tamanhos de efeito e o IC bootstrap ajudam a interpretar a magnitude, mas não corrigem a dependência espacial.
- Não há modelagem multinível ou teste com correção para dependência espacial.

### 15.6 Dataset pequeno e desbalanceado

- O dataset Emater tem 125 patches, dos quais 124 têm café. A proporção média de café é 21,5%, mas varia de 0% a 59,5%.
- Um patch não tem café. Para esse patch, as métricas de classe vazia usam fallback 1,0; isso pode inflar levemente as médias por patch.
- O número de patches por dobra varia de 18 a 31, o que afeta a variância das métricas por dobra.
- O dataset é pequeno para os padrões de deep learning; os resultados devem ser interpretados com cautela.

### 15.7 Limiar fixo e calibração

- O limiar de decisão é fixo em 0,5, tanto no treino quanto na avaliação.
- Não há calibração de probabilidades nem busca de limiar ótimo no conjunto de validação.
- Mudanças de limiar alterariam Precisão e Revocação; o IoU também depende do limiar.

### 15.8 Boundary Loss e CPU

- A Boundary Loss é calculada em CPU com `scipy.ndimage.distance_transform_edt`, o que é determinístico, mas adiciona custo.
- O mapa de distância é constante em relação aos parâmetros; o gradiente flui apenas pelas probabilidades.
- Cenas sem objeto ou completamente preenchidas recebem mapa zero e Boundary Loss zero.

### 15.9 Determinismo do SegFormer

- O upsampling bilinear do SegFormer não tem backward determinístico no CUDA no PyTorch.
- Os estágios `10` e `11` usam `warn_only=True`; a configuração efetiva é registrada em `run.meta.json`.
- Isso significa que o treino do SegFormer não é bit-a-bit determinístico no CUDA, embora o protocolo, as sementes e as demais operações sejam determinísticos.

### 15.10 Correção de bbox/split e baseline

- O primeiro treino da U-Net usou um cálculo de `bbox` com offset em pixels, em vez de `row/col * patch_size`, o que fazia as bboxes se sobreporem e invalidava o split espacial.
- O problema foi corrigido com `BBOX_SCHEMA_VERSION = 1` e `SPLIT_SCHEMA_VERSION = 2`.
- O primeiro treino foi preservado como `emater_v1`, com metadados em `data/processed/archive_v1/`, figuras em `artifacts/figures/archive_v1/` e `BASELINE.json` documentando a limitação.
- O experimento final usa o dataset filtrado e o split corrigido. O baseline não deve ser usado para a comparação principal.

### 15.11 Generalização e escopo

- O estudo cobre uma única RGI (Guaxupé), com relevo montanhoso e café em pequenas propriedades. Os resultados não são automaticamente generalizáveis para outras regiões.
- O modelo é binário (café vs. não café); não distingue variedades, idade, densidade ou condição fitossanitária.
- A resolução de 10 m impõe limitações a talhões estreitos, bordas e carreadores.
- A máscara de referência tem incerteza própria; erros de rótulo afetam o teto de desempenho.

---

## 16. O que deve ser preenchido com os artefatos

O repositório não versiona os artefatos pesados (ficam no Drive e no Hugging Face Hub). Para o capítulo, os seguintes números devem ser lidos dos artefatos, não inventados:

| Informação | Onde obter |
| --- | --- |
| Número de patches por fonte | `manifest_<fonte>.parquet` / tabela `dataset` da síntese |
| Distribuição de café e fração válida | `manifest_<fonte>.parquet` / tabela `dataset` |
| Distribuição de patches por dobra | `manifest_<fonte>.parquet` / `split_<fonte>.meta.json` |
| Estatísticas de normalização | `normalization_stats_<fonte>.json` |
| Melhor época por dobra | `artifacts/metrics/<modelo>/<fonte>/fold_<i>.json` / tabela `training` |
| Métricas por dobra | `artifacts/metrics/<modelo>/<fonte>/evaluation/fold_<i>.json` / tabela `evaluation_per_fold` |
| Métricas agregadas | `artifacts/metrics/<modelo>/<fonte>/evaluation/summary.json` / tabela `evaluation_aggregate` |
| Comparação estatística | `artifacts/metrics/comparative/<fonte>/comparison.json` / tabela `comparative_paired` |
| Relevância de XAI | `artifacts/synthesis/<fonte>/tables/xai.csv` |
| Figuras | `artifacts/figures/` |
| Commit, ambiente e fingerprints | `artifacts/runs/<modelo>/<fonte>/run.meta.json` |
| Release e links | `artifacts/release/<fonte>/release.json` |

Para a fonte Emater, os valores efetivos do release estão no Apêndice A.

---

## 17. Mapa de arquivos do repositório

| Arquivo | Responsabilidade |
| --- | --- |
| `src/config.yaml` | fonte única de parâmetros |
| `src/config.py` | carregamento e validação da configuração |
| `src/bootstrap.py` | entrega do `src/`, `data/external/` e `requirements-runtime.txt` ao runtime |
| `src/io.py` | plataforma, Drive, armazenamento, cópias locais e persistência |
| `src/setup.py` | prólogo de runtime, instalação pinada e reinício de kernel |
| `src/utils.py` | sementes, flags determinísticas, segredos, autenticação GEE, logging |
| `src/data/gee_client.py` | AOI, coleção Sentinel-2, mosaico e exportação GEE |
| `src/data/mask_utils.py` | máscaras MapBiomas/AlphaEarth, caminhos e exportação |
| `src/data/emater_masks.py` | rasterização da máscara vetorial Emater |
| `src/data/mask_comparison.py` | comparação entre fontes (estágio 03) |
| `src/data/source_decision.py` | registro da decisão manual (estágio 04) |
| `src/data/preprocessing.py` | composite alinhado e normalizado (estágio 05) |
| `src/data/patch_generation.py` | patches e manifesto (estágio 06) |
| `src/data/spatial_split.py` | k-fold espacial (estágio 07) |
| `src/data/eda.py` | estatísticas de normalização e sanidade (estágio 08) |
| `src/data/dataset.py` | `PatchDataset`, normalização e `DataLoader` |
| `src/data/augmentations.py` | aumentações geométricas determinísticas |
| `src/data/raster_utils.py` | grid, percentis e renderização |
| `src/losses.py` | Dice + Focal + Boundary |
| `src/metrics.py` | IoU, F1, Precisão, Revocação |
| `src/trainer.py` | protocolo único de treino |
| `src/models/unet.py` | U-Net |
| `src/models/segformer.py` | SegFormer binário com adaptação 3→4 bandas |
| `src/evaluation.py` | avaliação por pixel (estágio 11) |
| `src/comparative.py` | comparação estatística (estágio 12) |
| `src/xai/common.py` | seleção de patches e utilitários de XAI |
| `src/xai/gradcam.py` | Grad-CAM da U-Net (estágio 13) |
| `src/xai/attention_rollout.py` | Attention Rollout do SegFormer (estágio 14) |
| `src/synthesis.py` | tabelas e figuras da síntese (estágio 15) |
| `src/release.py` | publicação no Hugging Face Hub (estágio 16) |
| `tests/` | testes unitários de `src/` |
| `.github/workflows/ci.yml` | ruff, mypy, pytest e convenções de notebook |
| `requirements-runtime.txt` | versões pinadas de runtime |
| `uv.lock` | lockfile do `uv` |
| `data/external/ibge/` | malha IBGE versionada |
| `data/external/emater/` | GeoJSONs da Emater versionados |
| `tcc-docs/` | submódulo de material teórico (somente leitura) |

---

## 18. Correções em relação ao texto atual da metodologia

O texto atual da monografia (v1) contém informações desatualizadas ou incompatíveis com a implementação. As correções abaixo são obrigatórias para que o capítulo descreva o trabalho real.

| Ponto no texto atual | O que o texto diz | O que a implementação faz |
| --- | --- | --- |
| Código IBGE | `310027` | `310044` (RGI de Guaxupé) |
| Malha | “malha vetorial municipal” | Malha das **Regiões Geográficas Imediatas** do IBGE (2025) |
| Período | “meses da estação seca” | Ano civil completo, com filtro de nuvens e mediana |
| Tamanho do patch | em branco | 512 × 512 pixels (26,2144 km²) |
| Limiar de exclusão | 5% de representatividade da classe alvo | `min_valid_fraction = 0.10` (fração de pixels válidos); `coffee_min_ratio = 0.0` |
| Fontes de ground truth | MapBiomas inicial + AlphaEarth/S2DR3 como alternativas | MapBiomas 2025, AlphaEarth 2024 e Emater 2018; escolha manual = Emater 2018 |
| Super-resolução | S2DR3/S2DR4 | Não implementada; remover ou tratar como trabalho futuro |
| Otimizador | AdamW | Adam |
| Scheduler | “taxa de decaimento periódico” | `CosineAnnealingLR`, `T_max = 100` |
| Variante SegFormer | MiT-b0 até b2 | Fixa `nvidia/mit-b1` |
| U-Net | Transfer learning | Treinada do zero; sem pesos pré-treinados |
| Grad-CAM | “filtros ao longo do decodificador” | Camada-alvo `encoders[-1]`, no codificador mais profundo |
| k-fold | “(com )” em branco | k = 5 |
| Função objetivo | “modelada de forma simplificada por:” sem equação | Dice + Focal + Boundary com pesos 1:1:1 (Seção 7) |
| Plataforma | “Kaggle Kernels” | Código agnóstico Colab/Kaggle; treinamento em GPU; Kaggle como plataforma obrigatória do projeto |
| Attention Rollout | descrição genérica | Última etapa do codificador, média de cabeças, resíduo, normalização de linhas, sem CLS, grade 16×16 |
| Métricas | apenas citadas | IoU, F1, Precisão e Revocação por pixel, com agregado micro e macro por dobra |
| Comparação estatística | não descrita | Wilcoxon principal + Holm, t pareado, bootstrap, \(d_z\), delta de Cliff, mapas de erro |

Além disso, o texto atual não menciona:

- a existência de três fontes e o protocolo de comparação/decisão manual;
- a validação espacial por k-means sobre centroides de bboxes;
- o filtro de pixels válidos e o rótulo `-1`;
- a normalização z-score por banda;
- as aumentações geométricas;
- o protocolo de reuso por fingerprints e o baseline `emater_v1`;
- a publicação no Hugging Face Hub;
- a assimetria de pré-treinamento entre U-Net e SegFormer;
- o viés de seleção do melhor epoch na mesma dobra de validação.

---

## 19. Sugestão de estrutura para o capítulo 3

Uma organização possível, alinhada ao que foi implementado:

- **3.1 Delineamento e área de estudo** — pesquisa aplicada, comparativa, RGI de Guaxupé, malha IBGE, AOI.
- **3.2 Aquisição e fontes de ground truth** — Sentinel-2 L2A, bandas, filtro de nuvens, mosaico mediano; MapBiomas, AlphaEarth e Emater; comparação e decisão manual.
- **3.3 Pré-processamento e construção do dataset** — composite, alinhamento, reflectância, patches 512×512, filtro de pixels válidos, manifesto.
- **3.4 Divisão espacial e normalização** — k-means++ sobre centroides, k=5, estatísticas por banda, aumentações.
- **3.5 Arquiteturas** — U-Net e SegFormer MiT-B1, adaptação 3→4 bandas, parâmetros, protocolo justo.
- **3.6 Função de perda e treinamento** — Dice + Focal + Boundary, Adam, cosine annealing, 100 épocas, semente por dobra, seleção do melhor epoch.
- **3.7 Avaliação e comparação estatística** — métricas por pixel, agregação micro/macro, bootstrap, t pareado, Wilcoxon + Holm, tamanhos de efeito, mapas de erro.
- **3.8 Explicabilidade** — Grad-CAM e Attention Rollout, seleção de patches, quantificação dentro/fora.
- **3.9 Reprodutibilidade e infraestrutura** — configuração única, fingerprints, ambiente, bootstrap, Colab/Kaggle, CI.
- **3.10 Síntese e disseminação** — tabelas, figuras, dataset e modelos publicados.
- **3.11 Limitações e ameaças à validade** — assimetria de pré-treinamento, viés de seleção, normalização global, dependência espacial, dataset pequeno, fonte 2018.

---

## 20. Referências metodológicas canônicas

As referências abaixo são as mais diretamente associadas aos métodos implementados. Elas devem ser confrontadas com a revisão de literatura já escrita e formatadas conforme a norma adotada no trabalho.

- Ronneberger, O.; Fischer, P.; Brox, T. **U-Net: Convolutional Networks for Biomedical Image Segmentation.** MICCAI, 2015.
- Xie, E. et al. **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.** NeurIPS, 2021.
- Milletari, F.; Navab, N.; Ahmadi, S.-A. **V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.** 3DV, 2016. (Dice Loss)
- Lin, T.-Y. et al. **Focal Loss for Dense Object Detection.** ICCV, 2017.
- Kervadec, H. et al. **Boundary Loss for Highly Unbalanced Segmentation.** Medical Image Analysis, 2019.
- Selvaraju, R. R. et al. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.** ICCV, 2017.
- Abnar, S.; Zuidema, W. **Quantifying Attention Flow in Transformers.** ACL, 2020. (Attention Rollout)
- Efron, B.; Tibshirani, R. J. **An Introduction to the Bootstrap.** Chapman & Hall, 1993.
- Wilcoxon, F. **Individual Comparisons by Ranking Methods.** Biometrics Bulletin, 1945.
- Holm, S. **A Simple Sequentially Rejective Multiple Test Procedure.** Scandinavian Journal of Statistics, 1979.
- Cohen, J. **Statistical Power Analysis for the Behavioral Sciences.** 2. ed. Lawrence Erlbaum, 1988.
- Cliff, N. **Dominance Statistics: Ordinal Analyses to Answer Ordinal Questions.** Psychological Bulletin, 1993.
- IBGE. **Malha das Regiões Geográficas Imediatas, 2025.**
- MapBiomas. **Coleção 11 do Mapeamento Anual da Cobertura e Uso da Terra no Brasil.**
- Forest Data Partnership. **AlphaEarth Foundations / Coffee Model 2025b.**
- Emater-MG. **Geoportal do Café — mapeamento do parque cafeeiro de Minas Gerais (2016–2018).**

---

## Apêndice A — Dados efetivos do release Emater

Estes valores vêm dos artefatos publicados no Hugging Face Hub e podem ser usados para conferência. Eles são **resultados** e devem ser usados no capítulo de Resultados, não como parâmetros metodológicos.

### A.1 Dataset

- Repositório: `jotap1101/tcc-coffee-emater`.
- 125 patches; 124 com café; 1 sem café.
- `coffee_ratio` médio: 0,2150.
- `valid_fraction` médio: 0,8474.
- Fold 0: 18 patches; fold 1: 21; fold 2: 25; fold 3: 30; fold 4: 31.
- Estatísticas de normalização:
  - B2: média 0,039683; desvio 0,015657.
  - B3: média 0,069281; desvio 0,023501.
  - B4: média 0,061667; desvio 0,034175.
  - B8: média 0,321997; desvio 0,061097.
  - Fração de `NaN`: 15,26% em todas as bandas.

### A.2 Métricas agregadas publicadas

| Modelo | IoU micro | F1 micro | Precisão micro | Revocação micro | IoU macro (média ± desvio) |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net | 0,6173 | 0,7633 | 0,6936 | 0,8487 | 0,6016 ± 0,0682 |
| SegFormer | 0,6878 | 0,8150 | 0,8020 | 0,8285 | 0,6770 ± 0,0308 |

Diferenças agregadas (SegFormer − U-Net):

| Métrica | Delta |
| --- | ---: |
| IoU | +0,0706 |
| F1 | +0,0517 |
| Precisão | +0,1084 |
| Revocação | −0,0201 |

### A.3 Métricas por dobra publicadas

**U-Net:**

| Dobra | Patches | IoU | F1 | Precisão | Revocação |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 18 | 0,4783 | 0,6471 | 0,5317 | 0,8263 |
| 1 | 21 | 0,6222 | 0,7671 | 0,7341 | 0,8033 |
| 2 | 25 | 0,6282 | 0,7716 | 0,7781 | 0,7653 |
| 3 | 30 | 0,5949 | 0,7460 | 0,6198 | 0,9368 |
| 4 | 31 | 0,6844 | 0,8126 | 0,7810 | 0,8469 |

**SegFormer:**

| Dobra | Patches | IoU | F1 | Precisão | Revocação |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 18 | 0,6362 | 0,7777 | 0,7753 | 0,7801 |
| 1 | 21 | 0,6646 | 0,7985 | 0,7750 | 0,8235 |
| 2 | 25 | 0,6645 | 0,7984 | 0,8079 | 0,7892 |
| 3 | 30 | 0,6920 | 0,8180 | 0,7991 | 0,8378 |
| 4 | 31 | 0,7275 | 0,8422 | 0,8244 | 0,8609 |

### A.4 Contagens de confusão agregadas

| Modelo | TP | FP | FN |
| --- | ---: | ---: | ---: |
| U-Net | 5.060.378 | 2.235.495 | 902.307 |
| SegFormer | 4.940.265 | 1.219.662 | 1.022.420 |

Esses números mostram que o SegFormer reduziu falsos positivos, mas aumentou ligeiramente falsos negativos, resultando em maior precisão e menor revocação no agregado.

---

## Apêndice B — Esquema do manifesto

O manifesto Parquet de cada fonte tem o seguinte esquema lógico:

| Campo | Tipo | Descrição |
| --- | --- | --- |
| `patch_id` | string | `<tile_id>_r<row:04d>_c<col:04d>` |
| `tile_id` | string | `composite_310044_<ano>` |
| `fold` | int | dobra espacial 0..4 |
| `row` | int | linha do patch no grid |
| `col` | int | coluna do patch no grid |
| `bbox` | string | `minx,miny,maxx,maxy` em EPSG:31983 |
| `coffee_ratio` | float | proporção de café entre pixels válidos |
| `valid_fraction` | float | fração de pixels válidos |
| `mask_source` | string | `mapbiomas`, `alphaearth` ou `emater` |
| `image_path` | string | caminho relativo à raiz `tcc/` |
| `mask_path` | string | caminho relativo à raiz `tcc/` |

O manifesto é acompanhado por `manifest_<fonte>.meta.json`, que registra:

- `fingerprint_hash`;
- `bbox_schema_version`;
- `grid.rows` e `grid.cols`;
- `n_patches`;
- `discarded_invalid`;
- `min_valid_fraction`.

---

## Apêndice C — Checklist para a redação do capítulo

- [ ] Corrigir o código IBGE para `310044`.
- [ ] Trocar “malha municipal” por “malha das Regiões Geográficas Imediatas”.
- [ ] Remover a restrição a “meses da estação seca” e descrever o ano civil com filtro de nuvens.
- [ ] Preencher o tamanho do patch: 512×512 pixels (26,2144 km²).
- [ ] Substituir o limiar de 5% por `valid_fraction >= 0.10` e explicar que `coffee_min_ratio = 0.0`.
- [ ] Apresentar as três fontes de ground truth e a decisão manual por Emater 2018.
- [ ] Remover S2DR3/S2DR4 ou tratá-los como trabalho futuro.
- [ ] Trocar AdamW por Adam e descrever CosineAnnealingLR.
- [ ] Fixar a variante `nvidia/mit-b1`.
- [ ] Declarar que a U-Net foi treinada do zero e o SegFormer usou backbone ImageNet.
- [ ] Corrigir a camada-alvo do Grad-CAM para `encoders[-1]`.
- [ ] Preencher k = 5.
- [ ] Inserir a equação da perda total e das três componentes.
- [ ] Descrever a divisão espacial por k-means++ sobre centroides.
- [ ] Descrever a normalização z-score e o rótulo `-1`.
- [ ] Descrever as aumentações e o determinismo.
- [ ] Descrever a avaliação micro/macro e o limiar 0,5.
- [ ] Descrever a comparação estatística (Wilcoxon + Holm, t pareado, bootstrap, \(d_z\), delta de Cliff).
- [ ] Descrever Grad-CAM e Attention Rollout com seus escopos reais.
- [ ] Incluir a seção de limitações (assimetria de pré-treinamento, viés de seleção, normalização global, dependência espacial, dataset pequeno, fonte 2018).
- [ ] Usar os números do Apêndice A apenas no capítulo de Resultados.
