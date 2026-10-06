# SPDX-License-Identifier: Apache-2.0
"""PN168 — retencion rala del estado GDN con especulacion: el borde de reuso con el retroceso de EAGLE.

Con --prefix-cache-retention-interval > 0, MambaManager.reachable_block_mask conserva el estado de los cortes de
segmento y el del borde de reuso (``num_prompt - 1``: donde engancha el turno siguiente). Pero con EAGLE (DFlash
cuenta como EAGLE) el scheduler corta el ultimo chunk un bloque ANTES (``last_cache_position - block_size``) y el
estado queda en ese bloque, mientras la mascara marca el siguiente, que no tiene estado. Resultado medido en la
sesion real del 05-10: con retencion 14080 cada turno de una conversacion de 45k reusaba 13200 tokens (el corte de
segmento) en vez de 43120 (el denso). Con use_eagle se marca tambien el bloque anterior al borde: un bloque de
estado mas por pedido.
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN168: borde de reuso con EAGLE]"
_OLD = (
    "            boundary_block = aligned // block_size - 1\n"
    "            if start_block <= boundary_block < end_block:\n"
    "                mask[boundary_block - start_block] = True\n"
)
_NEW = (
    "            boundary_block = aligned // block_size - 1\n"
    "            for _b168 in ((boundary_block, boundary_block - 1, boundary_block - 2) if use_eagle else (boundary_block,)):  # " + MARKER + "\n"
    "                if start_block <= _b168 < end_block:\n"
    "                    mask[_b168 - start_block] = True\n"
)
_DIAG_OLD = (
    "            reachable_boundaries=reachable_boundaries,\n"
    "        )\n"
    "        self.block_pool.cache_full_blocks(\n"
)
_DIAG_NEW = (
    "            reachable_boundaries=reachable_boundaries,\n"
    "        )\n"
    "        if block_mask is not None and __import__('os').environ.get('GENESIS_PN168_DIAG') == '1':  # " + MARKER + "\n"
    "            _bl = self.req_to_blocks[request.request_id]\n"
    "            _m = [start for start, v in enumerate(block_mask) if v]\n"
    "            __import__('logging').getLogger('genesis.pn168').warning(\n"
    "                'PN168 diag req=%s g=%d bloques %d..%d prompt=%d marcados %s nulos %s', request.request_id[:10],\n"
    "                self.kv_cache_group_id, num_cached_blocks, num_full_blocks, request.num_prompt_tokens,\n"
    "                [num_cached_blocks + i for i in _m],\n"
    "                [num_cached_blocks + i for i in _m if num_cached_blocks + i < len(_bl) and _bl[num_cached_blocks + i].is_null])\n"
    "        self.block_pool.cache_full_blocks(\n"
)
_LK_OLD = (
    "                hit_length = (i + 1) * block_size\n"
    "                break  # we just need the last match - early stopping\n"
    "\n"
    "        return computed_blocks, hit_length\n"
)
_LK_NEW = (
    "                hit_length = (i + 1) * block_size\n"
    "                break  # we just need the last match - early stopping\n"
    "\n"
    "        if __import__('os').environ.get('GENESIS_PN168_DIAG') == '1':  # " + MARKER + "\n"
    "            __import__('logging').getLogger('genesis.pn168').warning(\n"
    "                'PN168 lookup g=%s max_length=%d max_bloques=%d alin=%d bs=%d hit=%d cacheados_al_final=%s',\n"
    "                kv_cache_group_ids, max_length, max_num_blocks, alignment_tokens, block_size, hit_length,\n"
    "                [j for j in range(max(0, max_num_blocks - 8), min(max_num_blocks + 2, len(block_hashes)))\n"
    "                 if block_pool.get_cached_block(block_hashes[j], kv_cache_group_ids)])\n"
    "        return computed_blocks, hit_length\n"
)
# La ventana deslizante (el borrador) guarda la cola de ``need`` bloques que termina en el borde: con el acierto
# en P-3 la cola tiene que arrancar 2 bloques antes.
_SWA_OLD = (
    "                end = aligned // block_size + shift\n"
    "                for j in range(max(start_block, end - need), min(end_block, end)):\n"
)
_SWA_NEW = (
    "                end = aligned // block_size + shift\n"
    "                for j in range(max(start_block, end - need - (2 if use_eagle else 0)), min(end_block, end)):  # " + MARKER + "\n"
)

# El turno siguiente con EAGLE busca el estado en el bloque P-3 (P = bloques llenos del prompt; el bloque se
# descarta en atencion y otra vez en Mamba). Con chunks de 2 bloques ese bloque puede quedar a mitad de un chunk
# (sin estado): un corte mas en (P-2)*block_size lo garantiza.
_SCH_OLD = (
    "        stops = (\n"
    "            # Same invariant: a chunk starting mid-block stops at the boundary\n"
)
_SCH_NEW = (
    "        stops = (\n"
    "            (request.num_prompt_tokens // block_size - 2) * block_size if self.use_eagle else 0,  # " + MARKER + "\n"
    "            (request.shared_prefix_boundary // block_size - 2) * block_size\n"
    "            if self.use_eagle and request.shared_prefix_boundary else 0,\n"
    "            # Same invariant: a chunk starting mid-block stops at the boundary\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN168")
    log_decision("PN168", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("v1/core/single_type_kv_cache_manager.py")
    if f is None:
        return "skipped", "falta single_type_kv_cache_manager.py"
    p = TextPatcher(patch_name="PN168 retencion con EAGLE", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn168_borde", anchor=_OLD, replacement=_NEW, required=True),
        TextPatch(name="pn168_swa", anchor=_SWA_OLD, replacement=_SWA_NEW, required=True),
        TextPatch(name="pn168_diag", anchor=_DIAG_OLD, replacement=_DIAG_NEW, required=True),
        TextPatch(name="pn168_lookup", anchor=_LK_OLD, replacement=_LK_NEW, required=True)],
        upstream_drift_markers=["_b168"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    f2 = resolve_vllm_file("v1/core/sched/scheduler.py")
    if f2 is None:
        return "skipped", "falta scheduler.py"
    p = TextPatcher(patch_name="PN168 corte de chunk con EAGLE", target_file=str(f2), marker=MARKER, sub_patches=[
        TextPatch(name="pn168_corte", anchor=_SCH_OLD, replacement=_SCH_NEW, required=True)],
        upstream_drift_markers=["(request.num_prompt_tokens // block_size - 2)"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "retencion rala: el borde de reuso conserva tambien el bloque donde EAGLE deja el estado"
