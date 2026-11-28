# Decisões metodológicas e limitações conhecidas

Registro das decisões que afetam a interpretação dos resultados, com o motivo e
a consequência de cada uma. Mantido junto do código para que quem reproduzir ou
estender o estudo conheça os limites do que foi medido. Para o passo a passo de
execução, veja [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## 1. Dados e referência

### D1 — Fonte de ground truth: Emater 2018 (decisão manual)
- **Decisão:** treinar e avaliar com as glebas de café da Emater-MG (2018),
  registradas manualmente no estágio `04` (`ground_truth.chosen_source`).
- **Motivo:** é a única fonte com ano coincidente com o composite Sentinel-2
  (2018), com validação em campo e cobertura dos 9 municípios da região.
- **Consequência:** as métricas medem a concordância com o **mapa da Emater**, não
  com a verdade de campo. A concordância entre as próprias fontes é moderada
  (IoU de 0,43 a 0,57 entre pares; kappa de 0,57 a 0,70), o que limita o teto
  alcançável por qualquer modelo. MapBiomas (2025) e AlphaEarth (2024) usam anos
  diferentes e permanecem como fontes de comparação.

### D2 — Patches de 512×512 sem sobreposição, filtro de validade
- Patches abaixo de `data.min_valid_fraction = 0,10` de pixels válidos são
  descartados; os pixels inválidos restantes recebem o rótulo `-1` e são
  excluídos da perda e das métricas.
- **Consequência:** a amostra de validação tem **125 recortes** de ~5,1 km de lado.

### D3 — Validação cruzada espacial (k = 5)
- Centroides dos patches agrupados por k-means determinístico em numpy puro,
  para reduzir vazamento por autocorrelação espacial.
- **Consequência:** as dobras são regiões contíguas e heterogêneas (nº de
  recortes e densidade de café variam); essa heterogeneidade é intencional.
- **Histórico:** o primeiro treino da U-Net (dataset sem filtro e antes da
  correção do cálculo de `bbox`/split) está preservado como baseline `emater_v1`
  e **não** entra nos resultados.

## 2. Treinamento

### D4 — Protocolo único entre os modelos
Mesma divisão, perda (Dice + Focal + Boundary, pesos 1:1:1), otimizador (Adam,
`lr = 1e-4`, `weight_decay = 1e-4`), *scheduler* (cosseno), 100 épocas, lote 16,
aumentações geométricas e sementes. Apenas a arquitetura difere.

### D5 — Inicialização assimétrica (limitação conceitual)
- **U-Net:** pesos aleatórios. **SegFormer (`nvidia/mit-b1`):** codificador
  pré-treinado (ImageNet), com a primeira projeção adaptada de 3 para 4 bandas e
  cabeça inicializada do zero.
- **Consequência:** a vantagem do SegFormer **não pode ser atribuída apenas à
  arquitetura**; parte pode vir do pré-treino. Isolar o efeito exigiria um
  controle (U-Net com codificador pré-treinado ou SegFormer sem pré-treino), que
  **não** foi executado e é a extensão mais informativa para trabalhos futuros.

### D6 — Seleção da melhor época na própria dobra de validação
- Os pesos retidos são os da época de maior IoU de validação **da mesma dobra**
  usada na avaliação.
- **Consequência:** viés otimista, que tende a beneficiar o modelo de curva mais
  oscilante. A tabela de treino da síntese (`training`) registra o IoU da época
  final e a diferença para a melhor época (`selection_optimism`) para dimensionar
  o viés. Eliminá-lo exigiria uma partição adicional (e, portanto, retreino).

## 3. Avaliação e inferência estatística

### D7 — Limiar de decisão 0,5 e sensibilidade ao limiar
- As métricas principais usam o limiar fixo de 0,5. Como os modelos podem ter
  calibrações diferentes, o estágio `11` salva histogramas de probabilidade por
  recorte e o estágio `12` calcula, **sem nova inferência**: a precisão média (AP),
  o limiar calibrado nas outras dobras (leave-one-fold-out) e o limiar ótimo
  (limite superior otimista, pois usa a própria amostra).
- A conclusão só é considerada robusta se persistir sob o limiar calibrado e o AP.

### D8 — Recortes não são independentes
- Recortes vizinhos compartilham relevo e condições de imageamento; testes que os
  tratam como independentes (t pareado, Wilcoxon por recorte) subestimam a
  incerteza e geram p-valores excessivamente pequenos.
- **Decisão:** os testes por recorte permanecem no `comparison.json` como
  **descritivos**. A inferência principal usa:
  - **blocos espaciais** de 3×3 recortes (bootstrap de blocos com 10.000
    reamostragens; teste de inversão de sinais; correção de Holm);
  - **dobras** (n = 5; IC pela distribuição t; p exato mínimo de 2/32 = 0,0625).
- A sensibilidade ao tamanho do bloco (1 a 4) é registrada em
  `spatial_inference.block_size_sensitivity_iou`.

### D9 — Agregação
O agregado **micro** (TP/FP/FN somados entre as dobras) é a medida principal; a
média ± desvio entre dobras (desvio populacional) expressa estabilidade.

## 4. Explicabilidade

### D10 — Métodos e limites
- **Grad-CAM (U-Net):** camada `encoders[-1]`, pontuação na região de referência,
  normalização restrita aos pixels válidos. Resolução do nível explicado: 64×64
  posições (manchas arredondadas, não contornos).
- **Attention Rollout (SegFormer):** última etapa do codificador (16×16 tokens),
  média das cabeças, conexões residuais, composição das camadas; os tokens de
  consulta são ponderados pela probabilidade predita de café e a normalização usa
  apenas pixels válidos (a média uniforme original acumulava relevância em um
  token de canto, um *attention sink*).
- **Limites:** o decodificador All-MLP usa os quatro estágios, mas o Rollout cobre
  só o último; mapas de gradiente e de atenção **não têm a mesma escala** nem a
  mesma natureza, então o contraste dentro × fora de um método não é comparável
  diretamente ao do outro. São 4 recortes por modelo (seleção determinística) e a
  métrica é descritiva; não há avaliação de fidelidade (ex.: curvas de apagamento).

## 5. Engenharia e reprodutibilidade

### D11 — Notebooks como orquestração, lógica em `src/`
Só `src/` e `tests/` passam por ruff, mypy e pytest. Notebooks são versionados
sem saídas e executam no Colab e no Kaggle com o mesmo protocolo.

### D12 — Fingerprints e idempotência
Artefatos derivados só são reutilizados se o fingerprint (manifesto, split, hash
da configuração, hash dos pesos, versão de esquema) coincidir; caso contrário são
regenerados no lugar. Alterar a **versão de esquema** de uma etapa (ex.: avaliação,
comparação, XAI, síntese, monografia) força a regeneração dela e das seguintes.

### D13 — Parâmetros de apresentação fora de `config.yaml`
O hash de `config.yaml` é o fingerprint dos artefatos de treino e avaliação;
acrescentar ali chaves editoriais invalidaria o treino. Largura, dpi e formatos
das figuras vivem em `src/figstyle.py`.

### D14 — Entrega de `src/` pelo GitHub público
O primeiro passo de cada notebook baixa o `src/` da `main`; por isso o código
**precisa estar publicado** para valer nos notebooks, e o commit executado fica
registrado em `src/.tcc_commit`.

## 6. Como interpretar os resultados (resumo honesto)

- O SegFormer supera a U-Net em IoU, F1 e precisão, e é mais estável entre as
  dobras; a U-Net tem revocação ligeiramente maior.
- Essa diferença deve ser lida **dentro** dos limites D1 (ruído da referência),
  D5 (pré-treino assimétrico), D6 (seleção otimista) e D8 (dependência espacial).
- Os resultados valem para esta região, sensor, ano e fonte de referência; não
  são estimativas de área para a região inteira.
