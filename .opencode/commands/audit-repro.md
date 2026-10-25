---
description: Audita a reprodutibilidade computacional de um run ou configuração
agent: reproducibility-auditor
subtask: true
---

Audite a reprodutibilidade de `$ARGUMENTS`.

Passos:
1. Leia `src/config.yaml`, `src/config.py`, `pyproject.toml` e o código relevante do alvo `$ARGUMENTS`.
2. Verifique cada item do checklist de reprodutibilidade (seeds, fonte única de config, `uv.lock`, `manifest_<fonte>.parquet`, logging de ambiente e ausência de segredos).
3. Marque cada item como PASS / FAIL / NÃO APLICÁVEL com evidência (`arquivo:linha` ou saída de comando).
4. Para cada FAIL, indique a correção exata necessária.
