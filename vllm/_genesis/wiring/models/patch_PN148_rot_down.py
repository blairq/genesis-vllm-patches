# SPDX-License-Identifier: Apache-2.0
"""PN148 — Hadamard en linea antes de down_proj, para checkpoints con el residuo rotado.

Ver ``vllm._genesis.rot_down``. Engancha ``Qwen2MoeMLP.forward`` (la MLP densa de Qwen3.5 y
Qwen3-Next): entre el SiluAndMul y down_proj. El borrador DFlash usa ``Qwen2MLP`` y no se toca.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN148: Hadamard antes de down_proj]"

_IMP_OLD = "from vllm.logger import init_logger\n"
_IMP_NEW = _IMP_OLD + "from vllm._genesis import rot_down as _g148  # " + MARKER + "\n"

_INIT_OLD = (
    "        self.act_fn = SiluAndMul()\n"
    "        self.expert_gate = expert_gate\n"
)
_INIT_NEW = (
    "        self.act_fn = SiluAndMul()\n"
    "        self.expert_gate = expert_gate\n"
    "        self.register_buffer('_g148_h', _g148.hadamard(), persistent=False)  # " + MARKER + "\n"
)

_FWD_OLD = (
    "        gate_up, _ = self.gate_up_proj(x)\n"
    "        out = self.act_fn(gate_up)\n"
    "        out, _ = self.down_proj(out)\n"
)
_FWD_NEW = (
    "        gate_up, _ = self.gate_up_proj(x)\n"
    "        out = _g148.act_had_down(self, gate_up)  # " + MARKER + "\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN148")
    log_decision("PN148", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("model_executor/models/qwen2_moe.py")
    if f is None:
        return "skipped", "qwen2_moe.py no esta en esta version de vLLM"
    p = TextPatcher(
        patch_name="PN148 Hadamard antes de down_proj", target_file=str(f), marker=MARKER,
        sub_patches=[TextPatch(name="pn148_import", anchor=_IMP_OLD, replacement=_IMP_NEW, required=True),
                     TextPatch(name="pn148_init", anchor=_INIT_OLD, replacement=_INIT_NEW, required=True),
                     TextPatch(name="pn148_forward", anchor=_FWD_OLD, replacement=_FWD_NEW, required=True)],
        upstream_drift_markers=["_g148"])
    r, fl = p.apply()
    return result_to_wiring_status(r, fl, applied_message="Hadamard por bloques antes de down_proj",
                                   patch_name=p.patch_name)
