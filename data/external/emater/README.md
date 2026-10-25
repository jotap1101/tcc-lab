# Fonte de ground truth — Emater (Geoportal do Café)

Glebas (polígonos) de lavouras de café por município, obtidas do
[Geoportal do Café](https://portaldocafedeminas.emater.mg.gov.br/Mapas)
("Vozes de Minas" — projeto Observatório da Agricultura de Minas Gerais,
Emater-MG / Codemig / Seapa / Epamig / FJP).

## Origem e vintage

- O mapeamento do parque cafeeiro mineiro foi **iniciado em 2016 e concluído
  em 2018**, a partir de imagens de satélite com **validação em campo** pelos
  extensionistas da Emater-MG.
- O conjunto é um **snapshot estático de ~2018** (sem atualização temporal
  conhecida no portal); os arquivos **não** carregam atributo de ano.
- Por isso, em `src/config.yaml` esta fonte define `year: 2018`, sobrescrevendo
  o ano global (2023) e demandando um mosaico/composite Sentinel-2 de 2018.

## Estrutura dos arquivos

- 9 arquivos `GetGlebasGeoJson - <Município>.json` (FeatureCollection, CRS
  EPSG:4326), um por município da RGI de Guaxupé (310044).
- 34.985 polígonos no total (~603 km² de café, ≈10% do AOI — fração registrada em `artifacts/metrics/ground_truth/source_decision_310044.json`).
- Propriedades: `area` (hectares, consistente com a área geométrica),
  `altitude` (m), `variedade` (100% "Café" — camada já binária);
  `anoplantio`/`espacamento` vazios.

## Tratamento aplicado na integração

- Reprojeção para EPSG:31983, reparo de geometrias inválidas (`buffer(0)`),
  clipe ao AOI e rasterização no grid de 10 m (`src/data/emater_masks.py`).
- Dados mantidos imutáveis (não mesclados) para preservar a fidelidade do download.