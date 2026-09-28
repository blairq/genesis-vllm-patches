# SPDX-License-Identifier: Apache-2.0
"""PN157 — K/V de contexto y kernel_projection del borrador DFlash2 en Marlin; ver
``vllm._genesis.borrador_marlin``. Va con PN142 (que decuantiza la K/V densa: aca se reemplaza).
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN157: borrador en Marlin]"
_IMP = "from vllm._genesis import borrador_marlin as _g157  # " + MARKER + "\n"

_D_IMP_OLD = "from vllm.model_executor.layers.quantization.base_config import QuantizationConfig\n"
_D_BUILD_OLD = "        self._fused_kv_weight = torch.cat(kv_weights, dim=0)\n"
_D_BUILD_NEW = _D_BUILD_OLD + "        _g157.preparar(self, layers_attn)  # " + MARKER + "\n"
_D_LIN_OLD = (
    "        all_kv_flat = F.linear(\n"
    "            normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
    "        )\n"
)
_D_LIN_NEW = (
    "        all_kv_flat = _g157.kv_contexto(  # " + MARKER + "\n"
    "            self, normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
    "        )\n"
)
_C_IMP_OLD = "from .utils import maybe_prefix\n"
_C_KP_OLD = "        coefficients = self.kernel_projection(hidden_states).reshape(\n"
_C_KP_NEW = "        coefficients = _g157.proyectar(self.kernel_projection, hidden_states).reshape(  # " + MARKER + "\n"


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN157")
    log_decision("PN157", decision, reason)
    if not decision:
        return "skipped", reason
    fd = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    fc = resolve_vllm_file("model_executor/models/qwen3_dflash2.py")
    if fd is None or fc is None:
        return "skipped", "faltan qwen3_dflash.py / qwen3_dflash2.py"
    ps = [
        TextPatcher(patch_name="PN157 K/V de contexto", target_file=str(fd), marker=MARKER, sub_patches=[
            TextPatch(name="pn157_imp", anchor=_D_IMP_OLD, replacement=_D_IMP_OLD + _IMP, required=True),
            TextPatch(name="pn157_preparar", anchor=_D_BUILD_OLD, replacement=_D_BUILD_NEW, required=True),
            TextPatch(name="pn157_kv", anchor=_D_LIN_OLD, replacement=_D_LIN_NEW, required=True)],
            upstream_drift_markers=["_g157"]),
        TextPatcher(patch_name="PN157 kernel_projection", target_file=str(fc), marker=MARKER, sub_patches=[
            TextPatch(name="pn157_c_imp", anchor=_C_IMP_OLD, replacement=_C_IMP_OLD + _IMP, required=True),
            TextPatch(name="pn157_kp", anchor=_C_KP_OLD, replacement=_C_KP_NEW, required=True)],
            upstream_drift_markers=["_g157"]),
    ]
    for p in ps:
        r, fl = p.apply()
        e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
        if e[0] != "applied":
            return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "K/V de contexto y kernel_projection del borrador en Marlin"
