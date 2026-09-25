# SPDX-License-Identifier: Apache-2.0
"""PN149 — borrador DFlash en su base original sobre un target con el residuo rotado.

Ver ``vllm._genesis.rot_borrador``. Tres puntos de ``qwen3_dflash.py`` y uno de ``qwen3_dflash2.py``:
buffers en el __init__ del modelo del borrador, desrotar el embedding compartido, y rotar el hidden
antes del lm_head compartido (logits y candidatos del arbol). Inerte si el config del borrador no
trae ``genesis_rotacion``.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN149: borrador sobre target rotado]"

_IMP_OLD = "from vllm.logger import init_logger\n"
_IMP_NEW = _IMP_OLD + "from vllm._genesis import rot_borrador as _g149  # " + MARKER + "\n"

_INIT_OLD = (
    "        self.norm = RMSNorm(\n"
    "            self.config.hidden_size,\n"
    "            eps=self.config.rms_norm_eps,\n"
    "        )\n"
)
_INIT_NEW = _INIT_OLD + "        _g149.preparar(self, self.config)  # " + MARKER + "\n"

_EMB_OLD = "        embeds = self.embed_tokens(input_ids)\n"
_EMB_NEW = _EMB_OLD + "        embeds = _g149.desrotar(self, embeds)  # " + MARKER + "\n"

_LOG_OLD = "        logits = self.logits_processor(self.lm_head, hidden_states)\n"
_LOG_NEW = ("        logits = self.logits_processor(self.lm_head, _g149.rotar(self.model, hidden_states))"
            "  # " + MARKER + "\n")

_IMP2_OLD = "import torch\n"
_IMP2_NEW = _IMP2_OLD + "from vllm._genesis import rot_borrador as _g149  # " + MARKER + "\n"
_CAND_OLD = "            self.lm_head, hidden_states, self.model.candidate_selector.top_k\n"
_CAND_NEW = ("            self.lm_head, _g149.rotar(self.model, hidden_states),"
             " self.model.candidate_selector.top_k  # " + MARKER + "\n")


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN149")
    log_decision("PN149", decision, reason)
    if not decision:
        return "skipped", reason
    f1 = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    f2 = resolve_vllm_file("model_executor/models/qwen3_dflash2.py")
    if f1 is None or f2 is None:
        return "skipped", "qwen3_dflash/qwen3_dflash2 no estan en esta version de vLLM"
    p1 = TextPatcher(
        patch_name="PN149 borrador rotado (dflash)", target_file=str(f1), marker=MARKER,
        sub_patches=[TextPatch(name="pn149_import", anchor=_IMP_OLD, replacement=_IMP_NEW, required=True),
                     TextPatch(name="pn149_init", anchor=_INIT_OLD, replacement=_INIT_NEW, required=True),
                     TextPatch(name="pn149_embed", anchor=_EMB_OLD, replacement=_EMB_NEW, required=True),
                     TextPatch(name="pn149_logits", anchor=_LOG_OLD, replacement=_LOG_NEW, required=True)],
        upstream_drift_markers=["_g149"])
    r, fl = p1.apply()
    e = result_to_wiring_status(r, fl, applied_message="embed/logits del borrador", patch_name=p1.patch_name)
    if e[0] != "applied":
        return e
    p2 = TextPatcher(
        patch_name="PN149 borrador rotado (dflash2)", target_file=str(f2), marker=MARKER,
        sub_patches=[TextPatch(name="pn149_import2", anchor=_IMP2_OLD, replacement=_IMP2_NEW, required=True),
                     TextPatch(name="pn149_candidatos", anchor=_CAND_OLD, replacement=_CAND_NEW, required=True)],
        upstream_drift_markers=["_g149"])
    r, fl = p2.apply()
    return result_to_wiring_status(r, fl, applied_message="embed, logits y candidatos del borrador",
                                   patch_name=p2.patch_name)
