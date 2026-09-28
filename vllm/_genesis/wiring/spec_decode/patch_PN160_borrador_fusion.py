# SPDX-License-Identifier: Apache-2.0
"""PN160 — lineales del borrador DFlash2 con SK-32 (cuantizacion y SiluAndMul fusionados); ver
``vllm._genesis.borrador_fusion``. Toca la atencion del borrador (qwen3_dflash.py) y Qwen2MLP (qwen2.py), que
en este despliegue solo usa el borrador (el target usa Qwen2MoeMLP, PN148/PN156).
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN160: borrador fusionado]"
_IMP = "from vllm._genesis import borrador_fusion as _g160  # " + MARKER + "\n"
_A_IMP_OLD = "from vllm.logger import init_logger\n"
_A_QKV_OLD = "        qkv, _ = self.qkv_proj(hidden_states)\n"
_A_QKV_NEW = "        qkv = _g160.lineal(self.qkv_proj, hidden_states, _q160)  # " + MARKER + "\n"
_A_SIG_OLD = (
    "        positions: torch.Tensor,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    '        """DFlash attention assumes that the KV cache is already populated\n'
)
_A_SIG_NEW = (
    "        positions: torch.Tensor,\n"
    "        hidden_states: torch.Tensor,\n"
    "        _q160=None,  # " + MARKER + "\n"
    "    ) -> torch.Tensor:\n"
    '        """DFlash attention assumes that the KV cache is already populated\n'
)
# capa DFlash2 (qwen3_dflash2.py): la conv prepara y cuantiza la entrada de qkv / gate_up
_C_IMP_OLD = "from .utils import maybe_prefix\n"
_C_ATT_OLD = (
    "        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)\n"
    "        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)\n"
)
_C_ATT_NEW = (
    "        hidden_states, coefficients, _q160 = _g160.preparar(  # " + MARKER + "\n"
    "            self.attention_conv, hidden_states, self.self_attn.qkv_proj)\n"
    "        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, _q160=_q160)\n"
)
_C_MLP_OLD = (
    "        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)\n"
    "        hidden_states = self.mlp(hidden_states)\n"
)
_C_MLP_NEW = (
    "        hidden_states, coefficients, _q160 = _g160.preparar(  # " + MARKER + "\n"
    "            self.mlp_conv, hidden_states, self.mlp.gate_up_proj)\n"
    "        hidden_states = self.mlp(hidden_states, _q160)\n"
)
_A_O_OLD = "        output, _ = self.o_proj(attn_output)\n"
_A_O_NEW = "        output = _g160.lineal(self.o_proj, attn_output)  # " + MARKER + "\n"
_M_IMP_OLD = "from vllm.model_executor.layers.linear import (\n"
_M_FWD_OLD = (
    "    def forward(self, x):\n"
    "        gate_up, _ = self.gate_up_proj(x)\n"
    "        x = self.act_fn(gate_up)\n"
    "        x, _ = self.down_proj(x)\n"
    "        return x\n"
)
_M_FWD_NEW = (
    "    def forward(self, x, _q160=None):\n"
    "        return _g160.mlp(self, x, _q160)  # " + MARKER + "\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN160")
    log_decision("PN160", decision, reason)
    if not decision:
        return "skipped", reason
    fa = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    fm = resolve_vllm_file("model_executor/models/qwen2.py")
    fc = resolve_vllm_file("model_executor/models/qwen3_dflash2.py")
    if fa is None or fm is None or fc is None:
        return "skipped", "faltan qwen3_dflash.py / qwen2.py / qwen3_dflash2.py"
    ps = [TextPatcher(patch_name="PN160 atencion del borrador", target_file=str(fa), marker=MARKER, sub_patches=[
              TextPatch(name="pn160_a_imp", anchor=_A_IMP_OLD, replacement=_A_IMP_OLD + _IMP, required=True),
              TextPatch(name="pn160_sig", anchor=_A_SIG_OLD, replacement=_A_SIG_NEW, required=True),
              TextPatch(name="pn160_qkv", anchor=_A_QKV_OLD, replacement=_A_QKV_NEW, required=True),
              TextPatch(name="pn160_o", anchor=_A_O_OLD, replacement=_A_O_NEW, required=True)],
              upstream_drift_markers=["_g160"]),
          TextPatcher(patch_name="PN160 MLP del borrador", target_file=str(fm), marker=MARKER, sub_patches=[
              TextPatch(name="pn160_m_imp", anchor=_M_IMP_OLD, replacement=_IMP + _M_IMP_OLD, required=True),
              TextPatch(name="pn160_m_fwd", anchor=_M_FWD_OLD, replacement=_M_FWD_NEW, required=True)],
              upstream_drift_markers=["_g160"]),
          TextPatcher(patch_name="PN160 conv del borrador", target_file=str(fc), marker=MARKER, sub_patches=[
              TextPatch(name="pn160_c_imp", anchor=_C_IMP_OLD, replacement=_C_IMP_OLD + _IMP, required=True),
              TextPatch(name="pn160_c_att", anchor=_C_ATT_OLD, replacement=_C_ATT_NEW, required=True),
              TextPatch(name="pn160_c_mlp", anchor=_C_MLP_OLD, replacement=_C_MLP_NEW, required=True)],
              upstream_drift_markers=["_g160"])]
    # orden: atencion y MLP primero (agregan un argumento opcional, inofensivo solo); la capa al final
    for p in ps:
        r, fl = p.apply()
        e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
        if e[0] != "applied":
            return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "lineales del borrador con SK-32 (cuantizacion y SiluAndMul en un kernel)"
