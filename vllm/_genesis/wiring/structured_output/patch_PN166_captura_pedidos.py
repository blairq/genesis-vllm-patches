# SPDX-License-Identifier: Apache-2.0
"""PN166 — captura de pedidos de chat antes de P68/P69; ver ``vllm._genesis.captura_pedidos``. Toca serving_chat.py
DESPUES de P68/P69 (se ancla en su bloque y se pone delante)."""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN166: captura de pedidos]"
_OLD = "        # [Genesis P68/P69 long-ctx tool-call adherence] Mutate request\n"
_NEW = (
    "        try:  # " + MARKER + "\n"
    "            from vllm._genesis import captura_pedidos as _g166\n"
    "            _g166.capturar(request, raw_request)\n"
    "        except Exception:\n"
    "            pass\n"
    + _OLD
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN166")
    log_decision("PN166", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("entrypoints/openai/chat_completion/serving.py") or resolve_vllm_file("entrypoints/openai/serving_chat.py")
    if f is None:
        return "skipped", "no encuentro serving de chat"
    p = TextPatcher(patch_name="PN166 captura de pedidos", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn166_captura", anchor=_OLD, replacement=_NEW, required=True)],
        upstream_drift_markers=["_g166"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "captura de pedidos de chat grandes (antes de P68/P69)"
