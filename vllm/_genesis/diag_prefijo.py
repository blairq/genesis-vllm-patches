# SPDX-License-Identifier: Apache-2.0
"""PN165 — diagnostico de prefix cache por pedido (solo numeros, nada del contenido).

Por cada pedido nuevo, despues de la busqueda local del scheduler: cuantos bloques de hash comparte con el pedido
anterior mas parecido (de los ultimos 64), hace cuanto llego ese pedido y cuantos tokens acerto el cache. Separa
las dos causas de un acierto bajo: el prompt cambio cerca del principio (comun bajo) o el cache perdio lo que tenia
(comun alto, acierto bajo).
"""
from __future__ import annotations

import collections
import logging
import os
import time

log = logging.getLogger("genesis.pn165")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN165_DIAG_PREFIJO", "0").strip().lower() in ("1", "true", "yes", "on")
_previos: collections.deque = collections.deque(maxlen=64)
_ids_previos: collections.deque = collections.deque(maxlen=32)


_esp: dict = {}


def _especiales():
    """ids de <|im_start|> y el nombre de cada rol (system/user/assistant/tool): para ubicar la divergencia por
    mensaje sin mirar el contenido."""
    if "ini" not in _esp:
        try:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(os.environ.get("GENESIS_PN165_TOKENIZER",
                                                              "/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86"))
            _esp["ini"] = tk.convert_tokens_to_ids("<|im_start|>")
            _esp["rol"] = {tk.convert_tokens_to_ids(r): r for r in ("system", "user", "assistant", "tool")}
        except Exception as e:
            log.warning("PN165: sin tokenizer (%s)", e)
            _esp["ini"], _esp["rol"] = -1, {}
    return _esp["ini"], _esp["rol"]


def _ubicar(ids, pos):
    """(indice del mensaje, rol, offset dentro del mensaje) de la posicion pos."""
    ini, roles = _especiales()
    k, rol, comienzo = -1, "?", 0
    for i in range(min(pos + 1, len(ids))):
        if ids[i] == ini:
            k += 1; comienzo = i
            rol = roles.get(ids[i + 1], "?") if i + 1 < len(ids) else "?"
    return k, rol, pos - comienzo


def registrar(request, acierto_local: int) -> None:
    if not ACTIVO:
        return
    try:
        hs = [bytes(h) if not isinstance(h, bytes) else h for h in request.block_hashes]
        n_prompt = request.num_prompt_tokens
        tok_por_bloque = max(1, n_prompt // max(1, len(hs))) if hs else 0
        mejor, quien, hace = 0, "-", 0.0
        ahora = time.time()
        for rid, hs2, t in _previos:
            if rid == request.request_id:
                continue                                     # la re-consulta de un diferido no se compara consigo
            k = 0
            for a, b in zip(hs, hs2):
                if a != b:
                    break
                k += 1
            if k > mejor:
                mejor, quien, hace = k, rid, ahora - t
        ids = list(request.prompt_token_ids or [])
        # divergencia EXACTA en tokens contra el pedido previo con el prefijo de tokens mas largo (no por bloques: si
        # algo cambia en los primeros 880 tokens, los hashes encadenados difieren todos)
        dv, dq, dhace, dlen = 0, "-", 0.0, 0
        for rid, ids2, t in _ids_previos:
            if rid == request.request_id:
                continue
            n = min(len(ids), len(ids2)); j = 0
            while j < n and ids[j] == ids2[j]:
                j += 1
            if j > dv:
                dv, dq, dhace, dlen = j, rid, ahora - t, len(ids2)
        if dq != "-":
            k, rol, off = _ubicar(ids, dv)
            dtxt = (f"divergencia en token {dv} (mensaje #{k} {rol}, token {off} del mensaje) con req={str(dq)[-10:]} "
                    f"de {dlen} tok, de hace {dhace:.0f} s")
        else:
            dtxt = "sin pedidos previos"
        log.warning("PN165 req=%s prompt=%d bloques=%d | comun_max=%d bloques (~%d tok) con req=%s de hace %.0f s | "
                    "acierto_local=%d tok (%.0f%%) | %s", str(request.request_id)[-10:], n_prompt, len(hs), mejor,
                    mejor * tok_por_bloque, str(quien)[-10:], hace, acierto_local,
                    100.0 * acierto_local / max(1, n_prompt), dtxt)
        if not any(rid == request.request_id for rid, _, _ in _previos):
            _previos.append((request.request_id, hs, ahora))
            _ids_previos.append((request.request_id, ids, ahora))
    except Exception as e:  # diagnostico: nunca rompe el scheduler
        log.warning("PN165: no se pudo registrar (%s: %s)", type(e).__name__, e)


_vistos_offload: dict = {}


def offload_grupo(req_id, grupo, ventana, inicio, n_claves, n_hit, computados, max_hit) -> None:
    """PN167: por pedido y grupo, que encontro la busqueda del offload (vLLM solo lo dice en debug)."""
    if not ACTIVO:
        return
    clave = (str(req_id), grupo, n_hit)
    if clave in _vistos_offload:
        return
    if len(_vistos_offload) > 5000:
        _vistos_offload.clear()
    _vistos_offload[clave] = 1
    log.warning("PN167 offload req=%s grupo=%s ventana=%s inicio_chunk=%s claves=%s hit_chunks=%s computados=%s max_hit=%s",
                str(req_id)[-10:], grupo, ventana, inicio, n_claves, n_hit, computados, max_hit)
