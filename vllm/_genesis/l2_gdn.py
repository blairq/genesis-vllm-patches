# SPDX-License-Identifier: Apache-2.0
"""PN170 — L2 (offload a RAM) para el hibrido con DFlash: sin esto el offload no acierta NUNCA.

Con EAGLE (DFlash cuenta como EAGLE) y ningun grupo marcado como de borrador, el connector de vLLM 0.29 declara EAGLE a
TODOS los grupos. Para los grupos GDN (ventana de 1 estado) eso suma la ventana extra de EAGLE: el lookup pide 2 estados
consecutivos y despues descarta uno. Con retencion rala (y PN168) los estados estan en P-3 y P-1, nunca consecutivos,
asi que el GDN daba 0 y el acierto externo era 0 aunque la atencion estuviera entera en L2 (medido 06-10: 5 GB
guardados, 0 aciertos). Los grupos GDN del target no tienen la volatilidad de la KV del borrador: sin EAGLE.

Medido despues (06-10): turno de 100k rescatado de L2 con 95920 tokens, TTFT 55,9 s -> 4,3 s; sumas por bloque de los
9 grupos en las dos GPUs identicas entre lo guardado y lo cargado. La salida difiere de la del acierto local solo por el
reparto en chunks del prefill que sigue (igual que local contra desde cero).

Diagnosticos: GENESIS_PN170_DIAG (handoffs, prepare_store), GENESIS_PN170_SUMAS (checksum por bloque al guardar y cargar).
"""
from __future__ import annotations

import os

ACTIVO = os.environ.get("GENESIS_ENABLE_PN170_L2_GDN", "0").strip().lower() in ("1", "true", "yes", "on")



import logging

log = logging.getLogger("genesis.pn170")
DIAG = os.environ.get("GENESIS_PN170_DIAG", "0") == "1"



def tras_prepare(req, nuevas: list, store_output) -> None:
    if DIAG and req.is_finished():
        acepta = None if store_output is None else len(store_output.keys_to_store)
        log.warning("PN170 prepare_store req=%s: %d claves pedidas, %s aceptadas", str(req.request_id)[-10:],
                    len(nuevas), acepta)


def sin_eagle_en_mamba(eagle_groups: set, kv_cache_config) -> set:
    """Los grupos GDN del target no son de borrador: sin la ventana extra de EAGLE (pide 2 estados consecutivos,
    y con retencion rala no los hay). vLLM los marca eagle solo porque ningun grupo esta marcado como borrador."""
    from vllm.v1.kv_cache_interface import MambaSpec
    return {i for i in eagle_groups if not isinstance(kv_cache_config.kv_cache_groups[i].kv_cache_spec, MambaSpec)}


def handoff(req, group_idx: int, block_id: int, boundary: int) -> None:
    if DIAG:
        log.warning("PN170 handoff req=%s grupo=%d bloque=%d borde=%d (chunk %d) prompt=%d computados=%d fin=%s",
                    str(req.request_id)[-10:], group_idx, block_id, boundary, boundary // 880 - 1,
                    req.num_prompt_tokens, req.num_computed_tokens, req.is_finished())


# ---- diagnostico de bytes (GENESIS_PN170_SUMAS=1): checksum por (grupo, indice logico) al guardar y al cargar ----
SUMAS = os.environ.get("GENESIS_PN170_SUMAS", "0") == "1"
_kv = {"cajas": None}
_cargas: dict = {}


def registrar_kv(canonical_kv_caches) -> None:
    if SUMAS:
        _kv["cajas"] = canonical_kv_caches


def _sumas(spec) -> list:
    import torch
    c = _kv["cajas"]
    if c is None:
        return []
    torch.cuda.synchronize()
    out = []
    pos = 0
    for g, (n, base) in enumerate(zip(spec.group_sizes, spec.block_indices)):
        for i in range(n):
            bid = int(spec.block_ids[pos + i])
            s = 0
            for ref in c.group_data_refs[g]:
                t = c.tensors[ref.tensor_idx].tensor
                s = (s * 1000003 + int(t[bid, :ref.page_size_bytes].view(torch.int32).to(torch.int64).sum().item())) % (1 << 61)
            out.append((g, int(base) + i, s))
        pos += n
    return out


def al_guardar(job_id, src_spec) -> None:
    if SUMAS:
        for g, idx, s in _sumas(src_spec):
            log.warning("PN170S guardar g=%d idx=%d suma=%d", g, idx, s)


def al_cargar(job_id, dst_spec) -> None:
    if SUMAS:
        _cargas[job_id] = dst_spec


def carga_lista(job_id) -> None:
    if SUMAS and job_id in _cargas:
        for g, idx, s in _sumas(_cargas.pop(job_id)):
            log.warning("PN170S cargar g=%d idx=%d suma=%d", g, idx, s)
