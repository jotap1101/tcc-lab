# RESULTADOS — Verdade Numérica do Pipeline

Documento de referência para o Capítulo 4 da monografia. Consolida exclusivamente valores extraídos dos artefatos de execução dos estágios 03 a 16, da fonte escolhida **Emater 2018**.

## Metadados da consolidação

| Item | Valor |
| --- | --- |
| Região | RGI de Guaxupé — MG (`310044`) |
| Fonte escolhida (estágio 04) | `emater` (2018) |
| Patches | 125 |
| Dobras espaciais | 5 (`k=5`) |
| Threshold de binarização | 0.5 |
| `config_hash` | `9330341efa14` |
| `manifest_fingerprint_hash` | `adb0b7b0229e` |
| `split_fingerprint_hash` | `ece582ed5e14` |
| Data de consolidação | 2026-11-05 |

**Regra de precisão:** métricas de desempenho, concordância, proporções e relevância são reportadas com 4 casas decimais. Contagens de pixels e patches são inteiros exatos. P-valores são reproduzidos na notação científica original dos CSVs/JSONs, sem arredondamento arbitrário.

---

## 1. Avaliação multicritério das máscaras (Estágios 03/04)

Fonte: `artifacts/metrics/ground_truth/mask_sources_comparison_310044.json`.

### 1.1 Área e cobertura por fonte

| Fonte | Ano | Resolução nativa | Área de café (km²) | Fração de café na grade |
| --- | ---: | ---: | ---: | ---: |
| MapBiomas | 2025 | 30 m | 471.9503 | 0.0804 |
| AlphaEarth | 2024 | 10 m | 542.2271 | 0.0923 |
| Emater | 2018 | 10 m | 603.2446 | 0.1027 |

Grade de referência: `6953 × 8445` pixels, `EPSG:31983`, pixel de 10 m, total de `58.718.085` pixels (`5871.8085 km²`).

### 1.2 Concordância par a par

Convenção do relatório: no par `A_vs_B`, `only_reference` corresponde a pixels de café exclusivos de `A`, enquanto `only_other` corresponde a pixels de café exclusivos de `B`. A precisão é calculada em relação a `A` e a revocação em relação a `B`.

| Par | Concordância global | IoU | F1 | Precisão | Revocação | Kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MapBiomas vs. AlphaEarth | 0.9321 | 0.4358 | 0.6070 | 0.6522 | 0.5677 | 0.5701 |
| MapBiomas vs. Emater | 0.9495 | 0.5677 | 0.7242 | 0.8250 | 0.6454 | 0.6969 |
| AlphaEarth vs. Emater | 0.9286 | 0.4641 | 0.6339 | 0.6696 | 0.6019 | 0.5945 |

### 1.3 Matrizes de confusão de pixels

| Par | Ambos café | Só `reference` | Só `other` | Ambos não café |
| --- | ---: | ---: | ---: | ---: |
| MapBiomas vs. AlphaEarth | 3.078.213 | 1.641.290 | 2.344.058 | 51.654.524 |
| MapBiomas vs. Emater | 3.893.525 | 825.978 | 2.138.921 | 51.859.661 |
| AlphaEarth vs. Emater | 3.630.817 | 1.791.454 | 2.401.629 | 50.894.185 |

**Decisão do estágio 04:** fonte escolhida `emater` (2018), registrada em `src/config.yaml` (`ground_truth.chosen_source`).

---

## 2. Estatísticas do dataset e split espacial (Estágios 06 a 08)

Fontes: `artifacts/synthesis/emater/tables/dataset.csv`, `data/processed/manifest_emater.parquet` (réplica publicada `manifest.parquet` com as mesmas 125 linhas) e `data/processed/split_emater.meta.json`.

### 2.1 Visão geral

| Estatística | Valor |
| --- | ---: |
| Patches totais | 125 |
| Patches com café (`coffee_ratio > 0`) | 124 |
| Dobras | 5 |
| Pixels totais (125 × 512 × 512) | 32.768.000 |
| Pixels válidos totais | 27.767.193 |
| Fração válida global | 0.8474 |
| Fração válida mínima por patch | 0.1330 |
| Média de `coffee_ratio` por patch | 0.2150 |
| Mediana de `coffee_ratio` por patch | 0.1986 |
| Máximo de `coffee_ratio` por patch | 0.5954 |

### 2.2 Distribuição por dobra

A coluna “café (média por patch)” é a média aritmética de `coffee_ratio` entre os patches da dobra — mesma definição da coluna `coffee_ratio_mean` de `dataset.csv`. A coluna “café (ponderado por pixels válidos)” é a razão entre pixels de café e pixels válidos na dobra.

| Dobra | Patches | Pixels válidos | Café (média por patch) | Café (ponderado por pixels válidos) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 18 | 3.792.117 | 0.1032 | 0.1085 |
| 1 | 21 | 4.655.682 | 0.1985 | 0.2168 |
| 2 | 25 | 5.605.627 | 0.2167 | 0.2036 |
| 3 | 30 | 6.308.562 | 0.2918 | 0.2754 |
| 4 | 31 | 7.405.205 | 0.2156 | 0.2246 |
| **Total** | **125** | **27.767.193** | **0.2150** | **0.2147** |

Pixels de café estimados a partir de `coffee_ratio × valid_fraction × 512²`: 411.476 (fold 0), 1.009.271 (fold 1), 1.141.499 (fold 2), 1.737.580 (fold 3) e 1.662.859 (fold 4), totalizando **5.962.685** pixels de café.

---

## 3. Desempenho quantitativo comparativo (Estágios 11/15)

Fontes: `artifacts/metrics/unet/emater/evaluation/summary.json`, `artifacts/metrics/segformer/emater/evaluation/summary.json`, `artifacts/metrics/{unet,segformer}/emater/evaluation/fold_{0..4}.json`, `artifacts/synthesis/emater/tables/evaluation_aggregate.csv` e `evaluation_per_fold.csv`.

### 3.1 Valores agregados (micro)

| Modelo | Patches | VP | FP | FN | IoU | F1-Score | Precisão | Revocação |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net | 125 | 5.060.378 | 2.235.495 | 902.307 | 0.6173 | 0.7633 | 0.6936 | 0.8487 |
| SegFormer | 125 | 4.940.265 | 1.219.662 | 1.022.420 | 0.6878 | 0.8150 | 0.8020 | 0.8285 |

### 3.2 Média e desvio-padrão entre as 5 dobras (macro)

Desvio-padrão populacional (`ddof=0`), conforme `summary.json`.

| Modelo | IoU (média ± desvio) | F1 (média ± desvio) | Precisão (média ± desvio) | Revocação (média ± desvio) |
| --- | ---: | ---: | ---: | ---: |
| U-Net | 0.6016 ± 0.0682 | 0.7489 ± 0.0553 | 0.6890 ± 0.0979 | 0.8357 ± 0.0574 |
| SegFormer | 0.6770 ± 0.0308 | 0.8070 ± 0.0218 | 0.7963 ± 0.0191 | 0.8183 ± 0.0301 |

### 3.3 Métricas por dobra

| Modelo | Dobra | Patches | IoU | F1 | Precisão | Revocação |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net | 0 | 18 | 0.4783 | 0.6471 | 0.5317 | 0.8263 |
| U-Net | 1 | 21 | 0.6222 | 0.7671 | 0.7341 | 0.8033 |
| U-Net | 2 | 25 | 0.6282 | 0.7716 | 0.7781 | 0.7653 |
| U-Net | 3 | 30 | 0.5949 | 0.7460 | 0.6198 | 0.9368 |
| U-Net | 4 | 31 | 0.6844 | 0.8126 | 0.7810 | 0.8469 |
| SegFormer | 0 | 18 | 0.6362 | 0.7777 | 0.7753 | 0.7801 |
| SegFormer | 1 | 21 | 0.6646 | 0.7985 | 0.7750 | 0.8235 |
| SegFormer | 2 | 25 | 0.6645 | 0.7984 | 0.8079 | 0.7892 |
| SegFormer | 3 | 30 | 0.6920 | 0.8180 | 0.7991 | 0.8378 |
| SegFormer | 4 | 31 | 0.7275 | 0.8422 | 0.8244 | 0.8609 |

### 3.4 Matriz de confusão agregada

| Modelo | VP | FP | FN | VN (derivado) | Total de pixels válidos |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net | 5.060.378 | 2.235.495 | 902.307 | 19.569.013 | 27.767.193 |
| SegFormer | 4.940.265 | 1.219.662 | 1.022.420 | 20.584.846 | 27.767.193 |

`VN` foi derivado por `pixels válidos − VP − FP − FN`; VP, FP e FN são exatamente os totais agregados dos artefatos.

---

## 4. Comparação estatística emparelhada (Estágios 12/15)

Fontes: `artifacts/metrics/comparative/emater/comparison.json`, `artifacts/synthesis/emater/tables/comparative_paired.csv` e `comparative_per_fold.csv`.

Direção da diferença: **SegFormer − U-Net**, emparelhada por `patch_id` (125 pares). IC95 por bootstrap com 2.000 iterações.

### 4.1 Testes de hipótese

| Métrica | Diferença média | Diferença mediana | IC95 | Estatística t | p (t pareado) | Estatística W | p (Wilcoxon) | p (Holm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IoU | 0.0666 | 0.0578 | [0.0553; 0.0778] | 11.77120769336805 | 6.165275919413241e-22 | 276.0 | 2.840181089983364e-19 | 1.1360724359933455e-18 |
| F1 | 0.0554 | 0.0446 | [0.0449; 0.0662] | 10.252313312411353 | 3.0831813201035825e-18 | 293.0 | 4.170576725673123e-19 | 1.251173017701937e-18 |
| Precisão | 0.1085 | 0.0645 | [0.0873; 0.1315] | 9.657263988522894 | 8.562100800146225e-17 | 340.0 | 1.1952904920365123e-18 | 2.3905809840730245e-18 |
| Revocação | -0.0209 | -0.0140 | [-0.0354; -0.0057] | -2.84359697324465 | 0.0052189272538352375 | 2677.0 | 0.004139995081974113 | 0.004139995081974113 |

### 4.2 Tamanhos de efeito e contagem de vitórias

| Métrica | Cohen's d_z | Delta de Cliff | SegFormer melhor | Empates | U-Net melhor |
| --- | ---: | ---: | ---: | ---: | ---: |
| IoU | 1.0528 | 0.3620 | 119 | 1 | 5 |
| F1 | 0.9170 | 0.3620 | 119 | 1 | 5 |
| Precisão | 0.8638 | 0.5377 | 118 | 1 | 6 |
| Revocação | -0.2543 | -0.1162 | 46 | 2 | 77 |

### 4.3 Diferenças por dobra (SegFormer − U-Net)

| Dobra | Patches | Δ IoU | Δ F1 | Δ Precisão | Δ Revocação |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 18 | 0.1580 | 0.1306 | 0.2435 | -0.0461 |
| 1 | 21 | 0.0424 | 0.0314 | 0.0408 | 0.0203 |
| 2 | 25 | 0.0363 | 0.0268 | 0.0298 | 0.0239 |
| 3 | 30 | 0.0971 | 0.0719 | 0.1793 | -0.0990 |
| 4 | 31 | 0.0430 | 0.0296 | 0.0433 | 0.0140 |

---

## 5. Complexidade computacional

Fontes: `run.meta.json` dos estágios 09/10, `model_config.json` publicado no Hugging Face Hub e documentação metodológica (`METODOLOGIA.md`, Seção 3.5). O pipeline **não persiste** contagem de FLOPs nem benchmark de latência em nenhum artefato dos estágios 09 a 16.

| Modelo | Parâmetros treináveis | GFLOPs | Tempo de inferência (FPS ou ms/patch) |
| --- | ---: | ---: | ---: |
| U-Net | 7.763.329 | DADO NÃO ENCONTRADO | DADO NÃO ENCONTRADO |
| SegFormer (`nvidia/mit-b1`) | 13.680.641 | DADO NÃO ENCONTRADO | DADO NÃO ENCONTRADO |

Notas:

- Os parâmetros foram confirmados por instanciação reprodutível das arquiteturas com a configuração vigente (`data.bands = [B2, B3, B4, B8]`, `model.unet_channels = [32, 64, 128, 256]`, `model.segformer_variant = nvidia/mit-b1`) e coincidem com os valores registrados na documentação metodológica.
- Não há arquivo de benchmark, contador de FLOPs ou medição de latência nos artefatos. Qualquer valor de GFLOPs ou FPS/ms por patch exigiria uma execução adicional de benchmark, não realizada pelo pipeline.

---

## 6. Quantificação de XAI (Estágio 15)

Fonte: `artifacts/synthesis/emater/tables/xai.csv`. Relevância é a média do mapa normalizado em [0,1], calculada dentro e fora da referência de café; a diferença é `dentro − fora`.

### 6.1 Relevância média por modelo

| Modelo | Relevância dentro | Relevância fora | Diferença |
| --- | ---: | ---: | ---: |
| U-Net (Grad-CAM) | 0.2431 | 0.0629 | 0.1802 |
| SegFormer (Attention Rollout) | 0.0055 | 0.0075 | -0.0021 |

### 6.2 Detalhamento por patch explicado

| Modelo | Patch | Critério | Dobra | IoU | `coffee_ratio` | Dentro | Fora | Diferença |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net | `composite_310044_2018_r0009_c0012` | Maior IoU | 4 | 0.7696 | 0.4469 | 0.2295 | 0.1246 | 0.1049 |
| U-Net | `composite_310044_2018_r0007_c0000` | Menor IoU | 0 | 0.1240 | 0.0059 | 0.4295 | 0.0299 | 0.3996 |
| U-Net | `composite_310044_2018_r0000_c0013` | Maior proporção de café | 3 | 0.6541 | 0.5954 | 0.0934 | 0.0428 | 0.0505 |
| U-Net | `composite_310044_2018_r0002_c0011` | IoU mediano | 3 | 0.6035 | 0.4124 | 0.2199 | 0.0542 | 0.1657 |
| SegFormer | `composite_310044_2018_r0009_c0012` | Maior IoU | 4 | 0.8057 | 0.4469 | 0.0060 | 0.0148 | -0.0088 |
| SegFormer | `composite_310044_2018_r0001_c0007` | Menor IoU | 1 | 0.2065 | 0.0088 | 0.0116 | 0.0048 | 0.0068 |
| SegFormer | `composite_310044_2018_r0000_c0013` | Maior proporção de café | 3 | 0.7106 | 0.5954 | 0.0016 | 0.0020 | -0.0003 |
| SegFormer | `composite_310044_2018_r0003_c0008` | IoU mediano | 1 | 0.6746 | 0.1466 | 0.0026 | 0.0086 | -0.0060 |

Nota interpretativa: os valores do Attention Rollout são aproximadamente uma ordem de grandeza menores que os do Grad-CAM porque, sem token CLS, a relevância de cada token é a média das linhas do rollout sobre uma grade de `16 × 16 = 256` tokens (escala próxima de `1/256`). A comparação válida é entre dentro e fora **dentro do mesmo modelo**, não entre as escalas absolutas dos dois métodos.

---

## 7. Mapeamento de figuras para a monografia

O arquivo do Blueprint não integra este repositório nem a pasta `tcc/`; o mapeamento abaixo segue a sequência de resultados do Capítulo 4 (fontes de máscara → dataset/EDA → desempenho → comparação/erros → XAI). Caminhos relativos à raiz canônica `tcc/`.

| Figura do Blueprint | Conteúdo | Caminho exato |
| --- | --- | --- |
| Figura 6 | Comparação multicritério das máscaras de referência | `artifacts/figures/mask_sources_comparison_310044.png` |
| Figura 7 | EDA do dataset e partição espacial k-fold | `artifacts/figures/eda_summary_310044_2018_emater.png` + `artifacts/figures/split_kfold_310044_2018_emater.png` |
| Figura 8 | Desempenho: curvas de treino e visão geral da avaliação | `artifacts/figures/synthesis_training_curves_310044_2018_emater.png` + `artifacts/figures/synthesis_evaluation_overview_310044_2018_emater.png` |
| Figura 9 | Comparação emparelhada e mapas de erro TP/FP/FN | `artifacts/figures/comparison_310044_2018_emater.png` + `artifacts/figures/error_maps_310044_2018_emater.png` |
| Figura 10 | Mapas de explicabilidade e relevância quantitativa | `artifacts/figures/gradcam_unet_310044_2018_emater.png` + `artifacts/figures/attention_rollout_segformer_310044_2018_emater.png` + `artifacts/figures/synthesis_xai_relevance_310044_2018_emater.png` |

Figuras complementares disponíveis:

- `artifacts/figures/evaluation_310044_2018_emater_unet.png`
- `artifacts/figures/evaluation_310044_2018_emater_segformer.png`
- `artifacts/figures/patch_montage_310044_2018_emater.png`
- `artifacts/figures/composite_310044_2018.png`
- `artifacts/figures/mask_emater_310044_2018_preview.png`
- `artifacts/figures/mask_mapbiomas_310044_2025_preview.png`
- `artifacts/figures/mask_alphaearth_310044_2024_t028_preview.png`

---

## 8. Dados não encontrados

| Item | Situação |
| --- | --- |
| GFLOPs por modelo | DADO NÃO ENCONTRADO |
| Tempo de inferência (FPS ou ms/patch) | DADO NÃO ENCONTRADO |
| Arquivo do Blueprint com a numeração definitiva das Figuras 6 a 10 | DADO NÃO ENCONTRADO (mapeamento inferido pela sequência do Capítulo 4) |

---

## 9. Proveniência dos artefatos consultados

| Conteúdo | Artefato |
| --- | --- |
| Comparação das fontes de máscara | `artifacts/metrics/ground_truth/mask_sources_comparison_310044.json` |
| Dataset por fonte | `artifacts/synthesis/emater/tables/dataset.csv` |
| Manifesto e split | `data/processed/manifest_emater.parquet`, `data/processed/split_emater.meta.json` |
| Avaliação agregada | `artifacts/metrics/{unet,segformer}/emater/evaluation/summary.json` |
| Avaliação por dobra | `artifacts/metrics/{unet,segformer}/emater/evaluation/fold_{0..4}.json` |
| Tabelas de avaliação | `artifacts/synthesis/emater/tables/evaluation_aggregate.csv`, `evaluation_per_fold.csv` |
| Comparação estatística | `artifacts/metrics/comparative/emater/comparison.json`, `artifacts/synthesis/emater/tables/comparative_paired.csv`, `comparative_per_fold.csv` |
| Complexidade | `artifacts/runs/{unet,segformer}/emater/run.meta.json`, `model_config.json` no Hugging Face Hub |
| XAI quantitativo | `artifacts/synthesis/emater/tables/xai.csv` |
| Metadados de XAI | `artifacts/xai/gradcam/unet/emater/gradcam_metadata.json`, `artifacts/xai/attention_rollout/segformer/emater/attention_rollout_metadata.json` |
| Figuras | `artifacts/figures/` |
