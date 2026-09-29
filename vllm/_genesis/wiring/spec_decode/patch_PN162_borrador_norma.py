# SPDX-License-Identifier: Apache-2.0
"""PN162 — normas del borrador DFlash2 con el cierre de la conv fundido (SK-34); ver ``vllm._genesis.borrador_norma``.
Toca el lazo de capas de DFlashQwen3Model.forward (qwen3_dflash.py): con capas DFlash2 corre
``borrador_norma.forward_capas``, que junta el cierre de la conv del MLP de cada capa con la norma de entrada de la
siguiente (y el de la ultima con la norma final).
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN162: normas del borrador]"
_IMP_OLD = "from vllm.logger import init_logger\n"
_IMP = "from vllm._genesis import borrador_norma as _g162  # " + MARKER + "\n"
_LAZO_OLD = (
    "        residual = None\n"
    "        for layer in self.layers:\n"
    "            hidden_states, residual = layer(\n"
)
_LAZO_NEW = (
    "        if _g162.aplica(self):  # " + MARKER + "\n"
    "            return _g162.forward_capas(self, positions, hidden_states)\n"
    + _LAZO_OLD
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN162")
    log_decision("PN162", decision, reason)
    if not decision:
        return "skipped", reason
    fa = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    if fa is None:
        return "skipped", "falta qwen3_dflash.py"
    p = TextPatcher(patch_name="PN162 normas del borrador", target_file=str(fa), marker=MARKER, sub_patches=[
        TextPatch(name="pn162_imp", anchor=_IMP_OLD, replacement=_IMP_OLD + _IMP, required=True),
        TextPatch(name="pn162_lazo", anchor=_LAZO_OLD, replacement=_LAZO_NEW, required=True)],
        upstream_drift_markers=["_g162"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "cierre de conv + residuo + RMSNorm del borrador en un kernel (SK-34)"
