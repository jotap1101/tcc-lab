---
description: Cria ou atualiza o notebook de um estágio do PLAN.md
agent: notebook-builder
subtask: true
---

Crie ou atualize o notebook do estágio `$ARGUMENTS` descrito em `PLAN.md`.

Passos:
1. Leia `PLAN.md` e localize o estágio `$ARGUMENTS`; identifique número, nome, entradas e saídas.
2. Leia `src/config.py`, `src/config.yaml` e `src/io.py` para conhecer caminhos, parâmetros e helpers de armazenamento disponíveis.
3. Leia notebooks adjacentes existentes para espelhar a estrutura e o estilo.
4. Produza o `.ipynb` seguindo as convenções: markdown em pt-br impessoal imediatamente acima de cada célula de código, identificadores en-US, uma responsabilidade por célula, caminhos sempre via `src/config.py`, agnosticismo de plataforma (mesmo notebook roda corretamente no Colab e no Kaggle), entrega do `src/` pelo bootstrap da primeira célula (`src/bootstrap.py`, que também grava `src/.tcc_commit`) e pelo espelho `repo/src` quando existir, acesso ao Drive só via `mount_drive()`/`ensure_storage_root()`, armazenamento dentro de `MyDrive/tcc/`, células de código com outputs vazios e autenticação executada dentro das próprias células.

Se o estágio `$ARGUMENTS` não existir ou conflitar com `PLAN.md`, não invente: reporte o conflito.
