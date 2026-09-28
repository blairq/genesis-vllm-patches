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


# fase 2: los tokens que entran al modelo alimentan el anillo dinamico (SK-27), al principio del embedding
_E_OLD = (
    "    def forward(self, input_):\n"
    "        if self.tp_size == 1:\n"
    "            return self.quant_method.embedding(self, input_.long())\n"
)
_E_NEW = (
    "    def forward(self, input_):\n"
    "        _g159.observar(input_)  # " + MARKER + " fase 2: tokens del paso -> anillo del borrador\n"
    "        if self.tp_size == 1:\n"
    "            return self.quant_method.embedding(self, input_.long())\n"
)
_E_IMP_OLD = "from vllm.model_executor.utils import set_weight_attrs\n"
_E_IMP_NEW = _E_IMP_OLD + "from vllm._genesis import vocab_borrador as _g159  # " + MARKER + "\n"


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
    e = result_to_wiring_status(r, fl, applied_message="candidatos del borrador sobre vocabulario recortado",
                                patch_name=p.patch_name)
    if e[0] != "applied":
        return e
    from vllm._genesis import vocab_borrador as _vb
    if _vb.DIN <= 0:
        return e
    fe = resolve_vllm_file("model_executor/layers/vocab_parallel_embedding.py")
    if fe is None:
        return "applied", e[1] + " (sin fase 2: falta vocab_parallel_embedding.py)"
    pe = TextPatcher(patch_name="PN159 fase 2 (embedding)", target_file=str(fe), marker=MARKER, sub_patches=[
        TextPatch(name="pn159_e_imp", anchor=_E_IMP_OLD, replacement=_E_IMP_NEW, required=True),
        TextPatch(name="pn159_e_obs", anchor=_E_OLD, replacement=_E_NEW, required=True)],
        upstream_drift_markers=["_g159.observar"])
    r2, fl2 = pe.apply()
    e2 = result_to_wiring_status(r2, fl2, applied_message="fase 2", patch_name=pe.patch_name)
    return ("applied", e[1] + (" + fase 2 (anillo de %d)" % _vb.DIN if e2[0] == "applied" else f" (fase 2 NO: {e2[1]})"))
