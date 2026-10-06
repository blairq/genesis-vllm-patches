# SPDX-License-Identifier: Apache-2.0
"""PN165 — diagnostico de prefix cache por pedido; ver ``vllm._genesis.diag_prefijo``. Toca el scheduler de v1."""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN165: diagnostico de prefijo]"
_IMP_OLD = "from vllm.logger import init_logger\n"
_IMP = "from vllm._genesis import diag_prefijo as _g165  # " + MARKER + "\n"
_HIT_OLD = "                    ) = self._get_local_prefix_cache_hit(request)\n"
_HIT_NEW = _HIT_OLD + "                    _g165.registrar(request, num_new_local_computed_tokens)  # " + MARKER + "\n"


_OFF_IMP_OLD = "from vllm.logger import init_logger\n"
_OFF_OLD = (
    "                if num_hit_chunks == 0:\n"
    "                    return 0\n"
    "\n"
    "                if num_hit_chunks is None:\n"
    "                    defer_lookup = True\n"
)
_OFF_NEW = (
    "                _g165.offload_grupo(req_status.req.request_id, group_idx, sliding_window_size_in_chunks,  # " + MARKER + "\n"
    "                                    start_chunk_idx, len(offload_keys), num_hit_chunks, num_computed_tokens,\n"
    "                                    max_hit_size_tokens)\n"
    + _OFF_OLD
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN165")
    log_decision("PN165", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("v1/core/sched/scheduler.py")
    if f is None:
        return "skipped", "falta v1/core/sched/scheduler.py"
    p = TextPatcher(patch_name="PN165 diagnostico de prefijo", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn165_imp", anchor=_IMP_OLD, replacement=_IMP_OLD + _IMP, required=True),
        TextPatch(name="pn165_hit", anchor=_HIT_OLD, replacement=_HIT_NEW, required=True)],
        upstream_drift_markers=["_g165"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    fo = resolve_vllm_file("distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py")
    if fo is not None:   # PN167: la busqueda del offload grupo por grupo
        p2 = TextPatcher(patch_name="PN167 diagnostico del offload", target_file=str(fo), marker=MARKER, sub_patches=[
            TextPatch(name="pn167_imp", anchor=_OFF_IMP_OLD, replacement=_OFF_IMP_OLD + _IMP, required=True),
            TextPatch(name="pn167_grupo", anchor=_OFF_OLD, replacement=_OFF_NEW, required=True)],
            upstream_drift_markers=["_g165.offload_grupo"])
        r2, fl2 = p2.apply()
        result_to_wiring_status(r2, fl2, applied_message=p2.patch_name, patch_name=p2.patch_name)
    return "applied", "diagnostico de prefix cache por pedido (solo numeros) + busqueda del offload por grupo"
