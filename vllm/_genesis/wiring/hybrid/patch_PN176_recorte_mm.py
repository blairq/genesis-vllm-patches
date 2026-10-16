# SPDX-License-Identifier: Apache-2.0
"""PN176 — recorte de imagenes por pedido: las que pasan ``--limit-mm-per-prompt`` se reemplazan por una
nota, las mas viejas primero, en vez de rechazar el pedido con 400. Ver ``vllm._genesis.recorte_mm``."""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN176: recorte mm]"
_OLD = (
    "        self._preflight(request.n or 1)\n"
    "\n"
    "        return await self.online_renderer.render_chat(request)\n"
)
_NEW = (
    "        self._preflight(request.n or 1)\n"
    "\n"
    "        from vllm._genesis import recorte_mm as _g176  # " + MARKER + "\n"
    "        _g176.recortar(request, self.model_config)\n"
    "        return await self.online_renderer.render_chat(request)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN176")
    log_decision("PN176", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("entrypoints/openai/chat_completion/serving.py")
    if f is None:
        return "skipped", "falta entrypoints/openai/chat_completion/serving.py"
    p = TextPatcher(patch_name="PN176 recorte mm", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn176_recortar", anchor=_OLD, replacement=_NEW, required=True)],
        upstream_drift_markers=["_g176"])
    r, fl = p.apply()
    return result_to_wiring_status(r, fl, applied_message="imagenes de mas: se omiten las mas viejas",
                                   patch_name=p.patch_name)
