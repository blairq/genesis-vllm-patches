# SPDX-License-Identifier: Apache-2.0
"""PN173 — tamano del chunk de prefill segun haya o no pedidos decodificando.

Medido 06-10 (prefill de 125k y, durante el, un turno con cache que genera 64 tokens):
    chunk   prefill solo   turno que decodifica
     880       83,7 s             16 s
    1760       76,3 s             31 s
    2640       74,0 s             49 s
El decode de los demas va al ritmo de los pasos, y el paso lo fija el chunk del prefill. Con nadie decodificando
conviene el chunk grande (prefill -3%); con alguien decodificando (o con poco prefill pendiente), el chico (su decode 2x mas rapido, +10% de prefill
solo mientras dura). Se fija una vez por paso en scheduler_config.long_prefill_token_threshold, que vLLM usa en los
chunks de running y waiting y en la alineacion de Mamba: los tres quedan coherentes. Multiplos de 880 (bloque).
"""
from __future__ import annotations

import os

ACTIVO = os.environ.get("GENESIS_ENABLE_PN173_CHUNK_DINAMICO", "0").strip().lower() in ("1", "true", "yes", "on")
SIN_DECODE = int(os.environ.get("GENESIS_PN173_CHUNK_SOLO", "2640"))
CON_DECODE = int(os.environ.get("GENESIS_PN173_CHUNK_CON_DECODE", "880"))
# Tambien cuenta como interactivo un pedido al que le falta poco prefill (un turno con cache): sin esto su primer
# token salia a 15 s en vez de 12 (su propio prefill corto corria con pasos de 2640 del pedido grande).
POCO = int(os.environ.get("GENESIS_PN173_FALTA_POCO", "8192"))


def ajustar(scheduler) -> None:
    if not ACTIVO:
        return
    try:
        hay = any(r.num_prompt_tokens - r.num_computed_tokens <= POCO for r in scheduler.running)
        scheduler.scheduler_config.long_prefill_token_threshold = CON_DECODE if hay else SIN_DECODE
    except Exception:
        pass
