# SPDX-License-Identifier: Apache-2.0
"""PN150 — el fc y la proyeccion de contexto del borrador DFlash, en grafos CUDA.

Ver ``vllm._genesis.grafo_borrador``. Dos puntos de ``v1/worker/gpu/spec_decode/dflash/speculator.py``
(DFlash2 hereda ``propose`` de DFlash): la combinacion de los estados ocultos y el precompute del contexto.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN150: borrador en grafo]"

_IMP_OLD = "import torch\n"
_IMP_NEW = _IMP_OLD + "from vllm._genesis import grafo_borrador as _g150  # " + MARKER + "\n"

_COMB_OLD = (
    "        if aux_hidden_states:\n"
    "            hidden_states = self.model.combine_hidden_states(\n"
    "                torch.cat(aux_hidden_states, dim=-1)\n"
    "            )\n"
    "        else:\n"
    "            hidden_states = last_hidden_states\n"
    "        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])\n"
)
_COMB_NEW = (
    "        _g150.combinar(self, aux_hidden_states, last_hidden_states, num_target_tokens,"
    " dummy_run)  # " + MARKER + "\n"
)

_CTX_OLD = (
    "        self.model.precompute_and_store_context_kv(\n"
    "            self.hidden_states[:num_target_tokens],\n"
    "            self.context_positions[:num_target_tokens],\n"
    "            context_slots,\n"
    "        )\n"
)
_CTX_NEW = (
    "        _g150.contexto(self, num_target_tokens, context_slots, dummy_run)  # " + MARKER + "\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN150")
    log_decision("PN150", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("v1/worker/gpu/spec_decode/dflash/speculator.py")
    if f is None:
        return "skipped", "dflash/speculator.py no esta en esta version de vLLM"
    p = TextPatcher(
        patch_name="PN150 borrador en grafo", target_file=str(f), marker=MARKER,
        sub_patches=[TextPatch(name="pn150_import", anchor=_IMP_OLD, replacement=_IMP_NEW, required=True),
                     TextPatch(name="pn150_combinar", anchor=_COMB_OLD, replacement=_COMB_NEW, required=True),
                     TextPatch(name="pn150_contexto", anchor=_CTX_OLD, replacement=_CTX_NEW, required=True)],
        upstream_drift_markers=["_g150"])
    r, fl = p.apply()
    return result_to_wiring_status(r, fl, applied_message="fc y contexto del borrador en grafos CUDA",
                                   patch_name=p.patch_name)
