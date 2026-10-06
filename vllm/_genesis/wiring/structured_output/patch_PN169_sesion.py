# SPDX-License-Identifier: Apache-2.0
"""PN169 — conversacion de cada pedido desde las cabeceras X-Genesis-*; ver ``vllm._genesis.sesion``.
Toca el serving de chat delante del bloque de P68/P69, como PN166."""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN169: sesion por pedido]"
_OLD = "        # [Genesis P68/P69 long-ctx tool-call adherence] Mutate request\n"
_NEW = (
    "        try:  # " + MARKER + "\n"
    "            from vllm._genesis import sesion as _g169\n"
    "            _g169.etiquetar(request, raw_request)\n"
    "        except Exception:\n"
    "            pass\n"
    + _OLD
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN169")
    log_decision("PN169", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("entrypoints/openai/chat_completion/serving.py") or resolve_vllm_file("entrypoints/openai/serving_chat.py")
    if f is None:
        return "skipped", "no encuentro serving de chat"
    p = TextPatcher(patch_name="PN169 sesion por pedido", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn169_etiqueta", anchor=_OLD, replacement=_NEW, required=True)],
        upstream_drift_markers=["_g169"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "cabeceras X-Genesis-* -> kv_transfer_params (conversacion por pedido)"
