# TCC — Segmentação Semântica de Lavouras Cafeeiras

Análise comparativa entre Redes Neurais Convolucionais (U-Net) e Vision
Transformers (SegFormer) na agricultura de precisão.

Bacharelado em Ciência da Computação — IFSULDEMINAS.

## Sumário

- [Visão Geral](#visão-geral)
- [Metodologia](#metodologia)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Execução](#execução)
- [Status](#status)
- [Contribuição](#contribuição)
- [Licença](#licença)

## Visão Geral

### Objetivo

Estudo comparativo entre duas famílias de arquiteturas para **segmentação
semântica binária** (café vs. não-café) de lavouras cafeeiras.

O foco é avaliar como a transição de um **viés convolucional local** (U-Net)
para a **autoatenção global** (SegFormer) afeta:

- a precisão geométrica das máscaras;
- a interpretabilidade das decisões do modelo;
- o desempenho sob as restrições do relevo montanhoso.

### Recorte territorial e dados

- **Região de estudo:** Região Geográfica Imediata de Guaxupé – MG, Brasil
  (código IBGE `310044`).
- **Imagens:** Sentinel-2 Nível-2A, bandas **B2, B3, B4 e B8**, a **10 m**.
- **Aquisição:** Google Earth Engine (GEE), com mosaicos sem nuvens por ano.
- **Recorte:** patches de **512×512 pixels**.

### Modelos comparados

- **U-Net (CNN):** arquitetura encoder-decoder convolucional, referência local.
- **SegFormer (ViT):** `nvidia/mit-b1` por padrão (configurável para b0–b2),
  com a primeira projeção adaptada de 3 para 4 bandas.
- **Protocolo único:** mesmo split, perda, otimizador, scheduler, sementes e
  métricas — apenas a arquitetura difere.

### Métricas e explicabilidade

- **Validação:** k-fold espacial (k = 5).
- **Métricas em nível de pixel:** IoU, F1-Score, Precisão e Revocação.
- **Explicabilidade:** Grad-CAM (U-Net) e Attention Rollout (SegFormer).

## Metodologia

O pipeline segue seis blocos encadeados, cada um com estágios de notebook
correspondentes.

### 1. Aquisição e ground truth

- Mosaicos Sentinel-2 sem nuvens (QA60) e máscaras de referência por fonte.
- Fontes de referência: **MapBiomas 2025**, **AlphaEarth 2024** e
  **Emater 2018** — cada uma usa o ano mais recente disponível.

### 2. Pré-processamento e dataset

- Composite normalizado por ano e recorte em patches de 512×512.
- Patches com menos de `data.min_valid_fraction` de pixels válidos são
  descartados; os pixels inválidos restantes recebem o rótulo `-1`.
- Cada fonte tem seu próprio conjunto de patches e manifesto.
- Divisão em k-fold espacial para evitar vazamento por autocorrelação espacial.

### 3. Treinamento

- U-Net e SegFormer treinados com o mesmo protocolo (`src/trainer.py`).
- Perda multivariada: **Dice + Focal + Boundary**.
- Pixels com rótulo `-1` são ignorados pela perda e pelas métricas.

### 4. Avaliação e comparação

- Métricas por patch, por dobra e agregadas (estágio `11`).
- Comparação estatística emparelhada entre arquiteturas (estágio `12`):
  - IC bootstrap de 95% e teste t pareado;
  - Wilcoxon com correção de Holm;
  - tamanhos de efeito (d_z de Cohen e delta de Cliff);
  - mapas de erro TP/FP/FN/ignorado.

### 5. Explicabilidade

- **Grad-CAM** da U-Net (estágio `13`), a partir do encoder mais profundo.
- **Attention Rollout** do SegFormer (estágio `14`), na última etapa do
  codificador hierárquico.

### 6. Síntese e disseminação

- Consolidação dos resultados em tabelas CSV/LaTeX e figuras (estágio `15`).
- Publicação do dataset e dos pesos no Hugging Face Hub (estágio `16`).

O [`PLAN.md`](PLAN.md) é o roteiro oficial, com a ordenação das fases, as
entradas/saídas de cada notebook e o rastreador de progresso.

## Estrutura do Repositório

### Código e documentação

```bash
tcc/
├── AGENTS.md                 # regras de código, formatação e infraestrutura
├── PLAN.md                   # roteiro de implementação
├── pyproject.toml            # uv, ruff, mypy, pytest
├── requirements-runtime.txt  # versões pinadas instaladas por notebook
├── src/                      # pacote Python compartilhado
├── tests/                    # pytest sobre src/
├── notebooks/                # estágios .ipynb sequenciais
├── scripts/                  # geradores de tokens OAuth (uso local)
├── secrets/                  # apenas templates versionados
├── data/                     # espelho git-ignored de MyDrive/tcc/data
│   └── external/             # dados de referência versionados (ex.: IBGE)
├── models/                   # espelho git-ignored de MyDrive/tcc/models
├── artifacts/                # espelho git-ignored de MyDrive/tcc/artifacts
└── tcc-docs/                 # submódulo git — material teórico (somente leitura)
```

### Armazenamento canônico (Google Drive)

Dados e artefatos **não** ficam no repositório. A raiz única de armazenamento
é a pasta **`tcc/` na raiz do Google Drive** (`MyDrive/tcc/`), criada ou
acessada pelos próprios notebooks, sem estrutura previamente assumida:

```bash
MyDrive/tcc/
├── data/{raw,interim,processed,external}/
│   └── interim/{mapbiomas,alphaearth,emater,...}/   # ground truth por fonte
├── models/{unet,segformer}/{fonte}/                 # pesos (+ baselines *_v1)
├── artifacts/
│   ├── metrics/{modelo}/{fonte}/                    # treino e avaliação
│   ├── metrics/comparative/{fonte}/                 # comparação estatística
│   ├── runs/{modelo}/{fonte}/                       # logs e metadata de execução
│   ├── xai/gradcam/unet/{fonte}/                    # mapas Grad-CAM
│   ├── xai/attention_rollout/segformer/{fonte}/     # mapas Attention Rollout
│   ├── synthesis/{fonte}/                           # tabelas da síntese
│   ├── release/{fonte}/                             # auditoria da publicação
│   └── figures/                                     # figuras (+ archive_v1)
├── repo/src/                                        # espelho do código-fonte
├── secrets/                                         # tokens OAuth (nunca versionar)
└── ...                                              # subpastas do pipeline
```

## Pré-requisitos

### Ambiente de execução

- Python com PyTorch e Hugging Face Transformers.
- GPU disponível — **Google Colab** ou **Kaggle** (obrigatório).
- Conta no Google Earth Engine, autenticada por OAuth com a conta principal.

### Armazenamento

- Google Drive com a pasta `tcc/` na raiz.
- Se a pasta já existir, ela é reutilizada; caso contrário, os notebooks a criam.

### Credenciais

- **Colab:** autenticação nativa com a conta logada.
- **Kaggle:** tokens OAuth do Drive e do Earth Engine, gerados uma única vez
  pelos scripts em `scripts/`.
- **Hugging Face:** `HF_TOKEN` (Kaggle Secrets, dataset privado ou variável de
  ambiente) para o estágio `16`.
- Nenhum token é versionado. Detalhes em [`secrets/README.md`](secrets/README.md).

## Execução

O projeto usa um **layout híbrido**: os notebooks orquestram e visualizam cada
estágio, enquanto a lógica reutilizável reside em `src/`.

Os notebooks devem ser executados em **ordem numérica** e resolvem todos os
caminhos a partir de `src/config.py` — nenhum caminho é hardcoded.

### Ordem dos estágios

| Fase | Notebooks | Entrega principal |
| --- | --- | --- |
| 0. Setup | `00` | ambiente, `src/` e raiz `tcc/` prontos |
| 1. Aquisição | `01`–`02` | mosaicos Sentinel-2 e máscaras por fonte |
| 2. Ground truth | `03`–`04` | comparação entre fontes e decisão manual |
| 3. Dataset | `05`–`08` | composite, patches, split espacial e EDA |
| 4. Treinamento | `09`–`10` | pesos da U-Net e do SegFormer |
| 5. Avaliação | `11`–`12` | métricas por pixel e comparação estatística |
| 6. XAI | `13`–`14` | mapas Grad-CAM e Attention Rollout |
| 7. Síntese e disseminação | `15`–`16` | tabelas/figuras e publicação no HF Hub |

### Portabilidade entre plataformas

- O **mesmo `.ipynb`** executa no Colab e no Kaggle, com o mesmo protocolo
  (código, sementes, versões e processamento).
- Os resultados são tratados de forma idêntica — não se exige saída bit-a-bit
  entre hardwares diferentes.
- A primeira célula de cada notebook baixa e executa `src/bootstrap.py`, que:
  - extrai `src/`, `data/external/` e `requirements-runtime.txt`;
  - adiciona o workspace ao `sys.path`;
  - grava o SHA do commit em `src/.tcc_commit`;
  - no Colab, cria o espelho `MyDrive/tcc/repo/`.
- Detecção de plataforma, montagem do Drive e resolução de caminhos ficam em
  `src/io.py` — notebooks não usam APIs ou caminhos específicos.
- No Colab o Drive é montado nativamente; no Kaggle o acesso é via Drive API
  (token OAuth + cache local), pela mesma chamada única.

### Autenticação

- Toda autenticação ocorre **dentro das células dos notebooks** (GEE, Hugging
  Face, Kaggle), com as bibliotecas correspondentes.
- As credenciais são lidas apenas de variáveis de ambiente e nunca são
  commitadas, logadas ou expostas.
- Não há etapa de autenticação externa ou manual.

### Dependências e determinismo

- A célula de dependências instala as versões de `requirements-runtime.txt` via
  `src.setup.install_runtime_requirements()`.
- Se uma extensão compilada já carregada divergir da versão pinada, o kernel é
  reiniciado automaticamente e o notebook reexecutado.
- O ambiente completo é registrado por execução: versões, sementes, hash de
  config, commit e fingerprint do manifesto.
- O determinismo do PyTorch é habilitado (`cudnn.deterministic`,
  `benchmark=False`, `use_deterministic_algorithms`).
- Os estágios `10` e `11` usam `warn_only=True`, pois o upsampling bilinear não
  tem backward determinístico no CUDA; a configuração efetiva fica em
  `run.meta.json`.
- Notebooks com Transformers removem um `torchaudio` incompatível quando
  presente (o projeto não usa áudio).
- Os notebooks são versionados com **outputs limpos**, e as convenções
  estruturais são validadas no CI.

### Idempotência e baselines

- As operações de armazenamento são idempotentes e re-executáveis.
- Artefatos derivados são reutilizados apenas enquanto o fingerprint do dataset
  e o hash da configuração coincidem; quando mudam, são regenerados no lugar.
- Baselines históricos são preservados com sufixo `_v1` e nunca sobrescritos
  (ex.: `models/unet/emater_v1`, `data/processed/archive_v1`,
  `artifacts/figures/archive_v1`).

### Qualidade e CI

- Linting, verificação de tipos e testes são validados localmente pelo `.venv`.
- O CI (GitHub Actions) repete os mesmos gates no push.

Regras operacionais completas: [`AGENTS.md`](AGENTS.md).

## Status

Os estágios `00`–`16` estão implementados e versionados.

### Dados

- Aquisição Sentinel-2 no GEE, com mosaicos por ano da fonte.
- Máscaras de referência por fonte (MapBiomas 2025, AlphaEarth 2024,
  Emater 2018), comparação entre fontes e registro da decisão manual (`04`).
- Pré-processamento, geração de patches com manifesto e split espacial k-fold
  (k = 5) por fonte; EDA e estatísticas de normalização.

### Modelos e avaliação

- Treino da **U-Net** (`09`) e do **SegFormer** (`10`) na fonte escolhida
  (Emater 2018), com o protocolo único de `src/trainer.py`.
- **Avaliação por pixel** (`11`): IoU/F1/Precisão/Revocação por patch, por dobra
  e agregados, com predições binárias persistidas.
- **Análise comparativa** (`12`) emparelhada por patch: diferença
  SegFormer − U-Net, IC bootstrap, teste t pareado, Wilcoxon com Holm,
  tamanhos de efeito, figura comparativa e mapas de erro.

### Explicabilidade

- **Grad-CAM da U-Net** (`13`): pesos por dobra, camada-alvo `encoders[-1]`,
  pontuação na região de referência de café e persistência em
  `artifacts/xai/gradcam/unet/<fonte>/`.
- **Attention Rollout do SegFormer** (`14`): última etapa do codificador,
  média das cabeças, conexões residuais, composição das camadas e persistência
  em `artifacts/xai/attention_rollout/segformer/<fonte>/`.

### Síntese e publicação

- **Síntese** (`15`): tabelas CSV/LaTeX e figuras de curvas de treino,
  avaliação/comparação e relevância quantitativa dos mapas, em
  `artifacts/synthesis/<fonte>/` e `artifacts/figures/`.
- **Publicação** (`16`): dataset de patches (CC BY 4.0) e pesos por dobra da
  U-Net e do SegFormer (MIT) no Hugging Face Hub, com model cards, metadados e
  autenticação por `HF_TOKEN`.

### Pacote e infraestrutura

- `src/` cobre `config`, `io`/`bootstrap`/`setup`, `data/*`, `losses`,
  `metrics`, `evaluation`, `comparative`, `trainer`, `models/*`,
  `xai/*`, `synthesis` e `release`.
- `tests/` e CI (ruff, mypy, pytest e convenções de notebook).
- O primeiro treino da U-Net (dataset sem filtro e antes da correção de
  bbox/split) está preservado como `emater_v1`, com metadados em
  `data/processed/archive_v1/` e figuras em `artifacts/figures/archive_v1/`;
  o `BASELINE.json` registra a limitação conhecida.

Consulte o [`PLAN.md`](PLAN.md) para o detalhamento por estágio.

## Contribuição

- Siga as convenções de [`AGENTS.md`](AGENTS.md): layout híbrido
  notebook/pacote, identificadores em en-US, markdown em pt-BR e configuração
  reprodutível.
- Aplique os princípios de engenharia de software (DRY, responsabilidade única,
  KISS, YAGNI, separação de conceitos) a todo o código, incluindo as células
  dos notebooks.
- Notebooks nunca duplicam lógica já existente em `src/`.
- Credenciais nunca devem ser commitadas.

## Licença

Licenciamento duplo:

- **MIT** — código-fonte.
- **CC BY 4.0** — documentação e conjuntos de dados.
