# SPDX-License-Identifier: Apache-2.0
"""PN172 — desalojo por rol en el prefix cache de la GPU; ver ``vllm._genesis.desalojo_sesion``.
Toca v1/core/block_pool.py (free_blocks, get_new_blocks) y v1/core/kv_cache_manager.py (allocate_slots, free)."""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN172: desalojo por rol]"
_BP_LIB_OLD = (
    "                    # FIFO reuse of cached blocks for LRU eviction behavior.\n"
    "                    blocks_to_evict_last.append(block)\n"
)
_BP_LIB_NEW = _BP_LIB_OLD + (
    "                    __import__('vllm._genesis.desalojo_sesion', fromlist=['x']).al_liberar(block)  # " + MARKER + "\n"
)
_BP_TOM_OLD = "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n"
_BP_TOM_NEW = (
    "        ret: list[KVCacheBlock] = __import__('vllm._genesis.desalojo_sesion', fromlist=['x']).tomar(  # " + MARKER + "\n"
    "            self.free_block_queue, num_blocks)\n"
)
_KM_PIDE_OLD = (
    "        # When loading KV data asynchronously, we may have zero new tokens to\n"
    "        # compute while still allocating slots for externally computed tokens.\n"
    "        if num_new_tokens == 0 and num_external_computed_tokens == 0:\n"
)
_KM_PIDE_NEW = (
    "        __import__('vllm._genesis.desalojo_sesion', fromlist=['x']).pidiendo(request)  # " + MARKER + "\n"
    + _KM_PIDE_OLD
)
_KM_LIB_OLD = "        self.coordinator.free(request.request_id)\n"
_KM_LIB_NEW = (
    "        _g172 = __import__('vllm._genesis.desalojo_sesion', fromlist=['x'])  # " + MARKER + "\n"
    "        _g172.liberando(request)\n"
    "        try:\n"
    "            self.coordinator.free(request.request_id)\n"
    "        finally:\n"
    "            _g172.liberando(None)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN172")
    log_decision("PN172", decision, reason)
    if not decision:
        return "skipped", reason
    for rel, subs in (("v1/core/block_pool.py", [("pn172_liberar", _BP_LIB_OLD, _BP_LIB_NEW), ("pn172_tomar", _BP_TOM_OLD, _BP_TOM_NEW)]),
                      ("v1/core/kv_cache_manager.py", [("pn172_pide", _KM_PIDE_OLD, _KM_PIDE_NEW), ("pn172_libera", _KM_LIB_OLD, _KM_LIB_NEW)])):
        f = resolve_vllm_file(rel)
        if f is None:
            return "skipped", f"falta {rel}"
        p = TextPatcher(patch_name=f"PN172 desalojo por rol ({rel.split('/')[-1]})", target_file=str(f), marker=MARKER,
                        sub_patches=[TextPatch(name=n, anchor=a, replacement=r, required=True) for n, a, r in subs],
                        upstream_drift_markers=["desalojo_sesion"])
        r, fl = p.apply()
        e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
        if e[0] != "applied":
            return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "prefix cache: un subagente no desaloja bloques de un hilo principal (salvo que no haya otra cosa)"
