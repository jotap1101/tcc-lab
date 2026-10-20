# Trabalho de Conclusão de Curso – Bacharelado em Ciência da Computação (IFSULDEMINAS)

Segmentação Semântica de Lavouras Cafeeiras: Análise Comparativa entre Redes Neurais Convolucionais (CNN) e Vision Transformers (ViT) na Agricultura de Precisão.

O estudo tem como alvo a Região Geográfica Imediata de Guaxupé – MG, Brasil, utilizando imagens Sentinel-2 Nível-2A para avaliar como a transição de um viés convolucional local para a autoatenção global afeta a precisão geométrica e a interpretabilidade sob as restrições do relevo montanhoso.

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

- **Tarefa:** segmentação semântica binária (café vs. não-café).
- **Modelos:** U-Net (CNN) e SegFormer (ViT, MiT-b0–b2).
- **Dados:** Sentinel-2 Nível-2A, bandas B2/B3/B4/B8 com resolução espacial de 10 m.
- **Função de perda:** combinação de Dice + Focal + Boundary Loss.
- **Validação:** k-fold espacial (k=5) com métricas em nível de pixel: IoU, F1-Score, Precisão e Revocação.
- **Explicabilidade:** Grad-CAM (U-Net) e Attention Rollout (SegFormer).

## Metodologia

O pipeline abrange aquisição, pré-processamento, treinamento, avaliação e explicabilidade:

1. Aquisição no Google Earth Engine de mosaicos Sentinel-2 e máscaras de referência por fonte (MapBiomas 2025 / AlphaEarth 2024 / Emater 2018 — cada fonte usa o ano mais recente com dados disponíveis).
2. Mascaramento de nuvens (QA60), normalização e recorte em patches de 512×512 pixels.
3. Particionamento em k-fold espacial para evitar vazamento por autocorrelação espacial.
4. Protocolo de treinamento idêntico para ambas as arquiteturas — apenas o modelo difere.
5. Avaliação de métricas em nível de pixel e geração de mapas de calor de XAI.

O `PLAN.md` é o roteiro oficial, com ordenação das fases, entradas/saídas de cada notebook e um rastreador de progresso.

## Estrutura do Repositório

```bash
tcc/
├── AGENTS.md              # fonte de verdade para regras de código e formatação
├── PLAN.md                # roteiro de implementação
├── pyproject.toml         # uv, ruff, mypy, pytest
├── requirements-runtime.txt  # versões pinadas instaladas pelo notebook 00
├── src/                   # pacote Python compartilhado (config, io, bootstrap, data, losses, trainer, models, ...)
├── tests/                 # pytest apenas sobre src/
├── notebooks/             # estágios .ipynb sequenciais
├── scripts/               # geradores de tokens OAuth (Drive, GEE) — uso local, para o Kaggle
├── secrets/               # apenas templates versionados (README.md, .env.example) — tokens nunca
├── data/                  # espelho local de MyDrive/tcc/data (ignorado pelo git)
│   └── external/          # dados de referência versionados (ex.: malha IBGE)
├── models/                # espelho local de MyDrive/tcc/models (ignorado pelo git)
├── artifacts/             # espelho local de MyDrive/tcc/artifacts (ignorado pelo git)
└── tcc-docs/              # submodulo git (tcc-docs) — material teórico, somente leitura
```

O **armazenamento canônico** de dados e artefatos não fica no repositório: tudo é gravado dentro da pasta **`tcc/` na raiz do Google Drive** (`MyDrive/tcc/`), acessada ou criada pelos próprios notebooks (junto com todas as subpastas de que precisarem). Nenhuma estrutura pré-criada é assumida:

```bash
MyDrive/tcc/                       # raiz de armazenamento (criada/acessada pelos notebooks)
├── data/{raw,interim,processed,external}/
│   └── interim/{mapbiomas,alphaearth,emater,...}/   # ground truth por fonte (ano próprio)
├── models/{unet,segformer}/
├── artifacts/{metrics,figures,runs}/
├── repo/src/                      # espelho do código-fonte (entrega do src/ ao runtime)
├── secrets/                       # tokens OAuth persistidos — nunca versionar
└── ...                            # qualquer subpasta necessária ao pipeline
```

## Pré-requisitos

- Python com PyTorch e Hugging Face Transformers.
- Conta no Google Earth Engine (autenticação OAuth com a conta principal via `ee.Authenticate()`).
- Google Drive com a pasta `tcc/` na raiz — se já existir é reutilizada; se não existir, os notebooks a criam automaticamente.
- Ambiente de execução com GPU (Google Colab ou Kaggle).
- Apenas para Kaggle: tokens OAuth do Drive e do Earth Engine, gerados uma vez pelos scripts em `scripts/` e armazenados como Secrets (env vars) — nunca versionados (veja `secrets/README.md`).

## Execução

O projeto segue um layout híbrido: os notebooks Jupyter orquestram cada estágio, enquanto a lógica reutilizável reside no pacote `src/`. Os notebooks devem ser executados em ordem numérica e resolver todos os caminhos de entrada a partir de `src/config.py`.

Os notebooks são **agnósticos de plataforma**: o mesmo `.ipynb` executa no Google Colab ou no Kaggle e roda perfeitamente nas duas (mesmo protocolo: código, sementes, versões e processamento — não necessariamente saída bit-a-bit entre hardwares diferentes). A primeira célula de **cada** notebook baixa e executa `src/bootstrap.py` (somente stdlib) do repositório público: o bootstrap extrai `src/`, `data/external/` e `requirements-runtime.txt` para o workspace, adiciona-o ao `sys.path` e, no Colab, cria o espelho `MyDrive/tcc/repo/`; quando o espelho já existe, `src/io.py` (`sync_repo_to_workspace()`) o copia para o workspace — os notebooks geram a estrutura dentro de `tcc/`. Todo dado, arquivo, modelo, métrica, figura e log gerado é armazenado dentro da pasta **`tcc/` na raiz do Google Drive** (`MyDrive/tcc/`); os notebooks acessam essa pasta se já existir ou a criam — assim como todas as subpastas de que precisam — sem assumir estrutura preexistente. Detecção de ambiente, montagem do Drive e resolução de caminhos ficam centralizados em `src/` (ex.: `io.py`); no Colab o Drive é montado nativamente e no Kaggle é acessado via Drive API (token OAuth + cache local) — sempre pela mesma chamada única. Cada fonte de ground truth (MapBiomas, AlphaEarth, Emater, ...) tem sua própria célula de código, subpasta (`data/interim/<fonte>/`) e **conjunto próprio de patches e manifesto** (`manifest_<fonte>.parquet`). O sistema não elege fonte alguma: o estágio 03 compara todas as fontes e o estágio 04 registra a **decisão manual** do pesquisador (`ground_truth.chosen_source`), que seleciona apenas qual conjunto de patches será usado no treinamento.

O treinamento e o armazenamento de artefatos ocorrem no Google Drive (a partir do Colab ou do Kaggle); linting, verificação de tipos e testes são validados localmente pelo `.venv` e novamente no CI (GitHub Actions). Consulte o `AGENTS.md` para as regras operacionais completas.

**Autenticação dentro dos notebooks:** todo processo de autenticação (GEE, Hugging Face, etc.) é executado nas próprias células de código dos notebooks, usando as bibliotecas/frameworks necessários (ex.: autenticação OAuth no GEE com `ee.Authenticate()`, usando a conta principal). As credenciais são lidas apenas de variáveis de ambiente — nunca commitadas, logadas ou expostas. Não há etapa de autenticação externa ou manual fora dos notebooks.

Para garantir o mesmo protocolo de execução nas duas plataformas, o notebook `00` instala versões fixadas das bibliotecas (`requirements-runtime.txt`, já que os ambientes padrão do Colab e do Kaggle diferem) e cada execução registra o ambiente completo (versões, sementes, hash de config/commit/manifest), com flags determinísticas do PyTorch habilitadas (`cudnn.deterministic`, `use_deterministic_algorithms`). Os notebooks são versionados com outputs limpos e as convenções estruturais (nomenclatura, markdown sobre cada célula de código, outputs vazios) são validadas no CI. As operações de armazenamento são idempotentes: reexecuções criam/abrem a árvore `tcc/` com segurança e nunca sobrescrevem silenciosamente artefatos versionados.

## Status

Estágios `00`–`09` implementados e versionados: aquisição Sentinel-2 (GEE, mosaicos por ano da fonte), máscaras de referência por fonte (MapBiomas 2025 / AlphaEarth 2024 / Emater 2018), comparação entre fontes e registro da decisão manual (`04`), pré-processamento (composite por ano), geração de patches por fonte + manifesto, divisão espacial k-fold (k=5) por fonte, EDA e treino do **U-Net** na fonte escolhida com o protocolo único (`src/trainer.py`). O pacote `src/` cobre `config`, `io`/`bootstrap`/`setup`, `data/*`, `losses`, `metrics`, `trainer` e `models/unet`, com `tests/` e CI (ruff, mypy, pytest e convenções de notebook). Pendentes: SegFormer (`10`), avaliação (`11`/`12`), XAI (`13`/`14`) e síntese/disseminação (`15`/`16`) — consulte o `PLAN.md`.

## Contribuição

Siga as convenções definidas no `AGENTS.md`: layout híbrido notebook/pacote, identificadores em en-US, markdown dos notebooks em pt-BR e configuração reprodutível. Princípios de engenharia de software (DRY, responsabilidade única, KISS, YAGNI, separação de conceitos) se aplicam a todo o código — inclusive às células dos notebooks, que nunca duplicam a lógica já existente em `src/`. Credenciais nunca devem ser commitadas.

## Licença

Licenciamento duplo:

- **MIT** — código-fonte.
- **CC BY 4.0** — documentação e conjuntos de dados.
