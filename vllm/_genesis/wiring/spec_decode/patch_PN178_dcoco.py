# SPDX-License-Identifier: Apache-2.0
"""PN178 — corrector DCOCO sobre el selector de candidatos del borrador DFlash2; ver ``vllm._genesis.dcoco``.

Tres toques en qwen3_dflash2.py (crear la cabeza si el checkpoint la trae, cargar sus pesos ``dcoco.*``) y uno
en el speculator de DFlash2 (corregir la tabla entre el selector y el recorrido o el arbol). Con un checkpoint
sin cabeza no cambia nada; un checkpoint DFlash2-DCOCO necesita este parche para cargar.
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN178: corrector DCOCO del borrador]"
_IMP = "from vllm._genesis import dcoco as _g178  # " + MARKER + "\n"
_M_IMP_OLD = "from .utils import maybe_prefix\n"
_M_SEL_OLD = (
    '                prefix=maybe_prefix(prefix, "candidate_selector"),\n'
    "            )\n"
)
_M_SEL_NEW = _M_SEL_OLD + (
    "        _g178.crear(self, draft_config, self.config.hidden_size, _g178.modo_env())  # " + MARKER + "\n"
)
_M_CLS_OLD = (
    "class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):\n"
    "    model_cls = DFlash2Qwen3Model\n"
)
_M_CLS_NEW = _M_CLS_OLD + (
    "\n"
    "    def load_weights(self, weights):  # " + MARKER + "\n"
    "        return _g178.cargar(self.model, super().load_weights, weights)\n"
)
_S_IMP_OLD = "from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator\n"
_S_PATH_OLD = "        self._sample_path(candidate_ids, scores, num_reqs)\n"
_S_PATH_NEW = (
    "        scores = _g178.corregir(self.model.model, scores, candidate_ids, unary_logits,  # " + MARKER + "\n"
    "                                hidden_states)\n"
) + _S_PATH_OLD


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN178")
    log_decision("PN178", decision, reason)
    if not decision:
        return "skipped", reason
    fm = resolve_vllm_file("model_executor/models/qwen3_dflash2.py")
    fs = resolve_vllm_file("v1/worker/gpu/spec_decode/dflash2/speculator.py")
    if fm is None or fs is None:
        return "skipped", "faltan qwen3_dflash2.py / dflash2/speculator.py"
    ps = [TextPatcher(patch_name="PN178 cabeza DCOCO", target_file=str(fm), marker=MARKER, sub_patches=[
              TextPatch(name="pn178_m_imp", anchor=_M_IMP_OLD, replacement=_M_IMP_OLD + _IMP, required=True),
              TextPatch(name="pn178_crear", anchor=_M_SEL_OLD, replacement=_M_SEL_NEW, required=True),
              TextPatch(name="pn178_cargar", anchor=_M_CLS_OLD, replacement=_M_CLS_NEW, required=True)],
              upstream_drift_markers=["_g178"]),
          TextPatcher(patch_name="PN178 correccion en el speculator", target_file=str(fs), marker=MARKER, sub_patches=[
              TextPatch(name="pn178_s_imp", anchor=_S_IMP_OLD, replacement=_S_IMP_OLD + _IMP, required=True),
              TextPatch(name="pn178_corregir", anchor=_S_PATH_OLD, replacement=_S_PATH_NEW, required=True)],
              upstream_drift_markers=["_g178"])]
    msgs = []
    for p in ps:
        r, fl = p.apply()
        e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
        if e[0] != "applied":
            return e
        msgs.append(e[1])
    return "applied", "; ".join(msgs)
