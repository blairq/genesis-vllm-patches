# SPDX-License-Identifier: Apache-2.0
"""PN159 — candidatos del borrador DFlash2 sobre un vocabulario recortado; ver ``vllm._genesis.vocab_borrador``.
El subconjunto lo arma PN139 al cuantizar el lm_head (sin PN139 no hay subconjunto y todo sigue igual).
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN159: vocabulario recortado del borrador]"
_IMP_OLD = "from .utils import maybe_prefix\n"
_IMP_NEW = _IMP_OLD + "from vllm._genesis import vocab_borrador as _g159  # " + MARKER + "\n"
_TK_OLD = "        return self.candidate_logits_processor.get_top_k_tokens(\n"
_TK_NEW = "        return _g159.top_k(self.candidate_logits_processor,  # " + MARKER + "\n"


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN159")
    log_decision("PN159", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("model_executor/models/qwen3_dflash2.py")
    if f is None:
        return "skipped", "falta qwen3_dflash2.py"
    p = TextPatcher(patch_name="PN159 vocabulario del borrador", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn159_imp", anchor=_IMP_OLD, replacement=_IMP_NEW, required=True),
        TextPatch(name="pn159_topk", anchor=_TK_OLD, replacement=_TK_NEW, required=True)],
        upstream_drift_markers=["_g159"])
    r, fl = p.apply()
    return result_to_wiring_status(r, fl, applied_message="candidatos del borrador sobre vocabulario recortado",
                                   patch_name=p.patch_name)
