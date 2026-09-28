# SPDX-License-Identifier: Apache-2.0
"""PN156 — la norma del decoder escribe int8 (SK-26) y la lineal que sigue lo toma directo; ver
``vllm._genesis.norma_q8``. Va DESPUES de PN148, PN154 y PN155 (toca los mismos archivos).
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN156: norma que escribe int8]"
_IMP = "from vllm._genesis import norma_q8 as _g156  # " + MARKER + "\n"

# ── decoder + atencion (qwen3_next.py) ──
_N_IMP_OLD = "from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate\n"
_N_NORM1_OLD = (
    "        if residual is None:\n"
    "            residual = hidden_states\n"
    "            hidden_states = self.input_layernorm(hidden_states)\n"
    "        else:\n"
    "            hidden_states, residual = self.input_layernorm(hidden_states, residual)\n"
)
_N_NORM1_NEW = (
    "        _q156 = None  # " + MARKER + "\n"
    "        if residual is None:\n"
    "            residual = hidden_states\n"
    "            hidden_states = self.input_layernorm(hidden_states)\n"
    "        else:\n"
    "            hidden_states, residual, _q156 = _g156.norma(self, 0, hidden_states, residual)\n"
)
_N_GDN_OLD = "            hidden_states = self.linear_attn(hidden_states=hidden_states)\n"
_N_GDN_NEW = "            hidden_states = self.linear_attn(hidden_states=hidden_states, _q156=_q156)  # " + MARKER + "\n"
_N_ATT_OLD = (
    "            hidden_states = self.self_attn(\n"
    "                hidden_states=hidden_states,\n"
    "                positions=positions,\n"
    "            )\n"
)
_N_ATT_NEW = (
    "            hidden_states = self.self_attn(\n"
    "                hidden_states=hidden_states,\n"
    "                positions=positions,\n"
    "                _q156=_q156,  # " + MARKER + "\n"
    "            )\n"
)
_N_NORM2_OLD = "        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)\n"
_N_NORM2_NEW = "        hidden_states, residual, _q156 = _g156.norma(self, 1, hidden_states, residual)  # " + MARKER + "\n"
_N_MLP_OLD = (
    "        else:\n"
    "            hidden_states = self.mlp(hidden_states)\n"
)
_N_MLP_NEW = (
    "        else:\n"
    "            hidden_states = self.mlp(hidden_states, _q156=_q156)  # " + MARKER + "\n"
)
_A_FWD_OLD = (
    "        positions: torch.Tensor,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        qkv, _ = self.qkv_proj(hidden_states)\n"
)
_A_FWD_NEW = (
    "        positions: torch.Tensor,\n"
    "        hidden_states: torch.Tensor,\n"
    "        _q156=None,  # " + MARKER + "\n"
    "    ) -> torch.Tensor:\n"
    "        qkv = _g156.lineal(self.qkv_proj, hidden_states, _q156)\n"
)

# ── GDN (qwen_gdn_linear_attn.py) ──
_G_IMP_OLD = "from vllm.model_executor.layers.layernorm import RMSNormGated\n"
_G_FWD_OLD = (
    "    def forward(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
    "        return self._forward_method(hidden_states)\n"
)
_G_FWD_NEW = (
    "    def forward(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "        _q156=None,  # " + MARKER + "\n"
    "    ) -> torch.Tensor:\n"
    "        if _q156 is not None:\n"
    "            return self.forward_cuda(hidden_states, _q156=_q156)\n"
    "        return self._forward_method(hidden_states)\n"
)
_G_CUDA_OLD = (
    "    def forward_cuda(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "    ) -> torch.Tensor:\n"
)
_G_CUDA_NEW = (
    "    def forward_cuda(\n"
    "        self,\n"
    "        hidden_states: torch.Tensor,\n"
    "        _q156=None,  # " + MARKER + "\n"
    "    ) -> torch.Tensor:\n"
)
_G_QKVZ_OLD = "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n        if self.in_proj_ba is None:"
_G_QKVZ_NEW = ("        mixed_qkvz = _g156.lineal(self.in_proj_qkvz, hidden_states, _q156)  # " + MARKER + "\n"
               "        if self.in_proj_ba is None:")

# ── MLP (qwen2_moe.py, ya con PN148) ──
_M_IMP_OLD = "from vllm.logger import init_logger\n"
_M_FWD_OLD = (
    "    def forward(self, x):\n"
    "        gate_up, _ = self.gate_up_proj(x)\n"
)
_M_FWD_NEW = (
    "    def forward(self, x, _q156=None):  # " + MARKER + "\n"
    "        gate_up = _g156.lineal(self.gate_up_proj, x, _q156)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN156")
    log_decision("PN156", decision, reason)
    if not decision:
        return "skipped", reason
    fn = resolve_vllm_file("model_executor/models/qwen3_next.py")
    fg = resolve_vllm_file("model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py")
    fm = resolve_vllm_file("model_executor/models/qwen2_moe.py")
    if fn is None or fg is None or fm is None:
        return "skipped", "faltan qwen3_next.py / qwen_gdn_linear_attn.py / qwen2_moe.py"
    ps = [
        TextPatcher(patch_name="PN156 decoder y atencion", target_file=str(fn), marker=MARKER, sub_patches=[
            TextPatch(name="pn156_imp", anchor=_N_IMP_OLD, replacement=_N_IMP_OLD + _IMP, required=True),
            TextPatch(name="pn156_norma1", anchor=_N_NORM1_OLD, replacement=_N_NORM1_NEW, required=True),
            TextPatch(name="pn156_gdn", anchor=_N_GDN_OLD, replacement=_N_GDN_NEW, required=True),
            TextPatch(name="pn156_att", anchor=_N_ATT_OLD, replacement=_N_ATT_NEW, required=True),
            TextPatch(name="pn156_norma2", anchor=_N_NORM2_OLD, replacement=_N_NORM2_NEW, required=True),
            TextPatch(name="pn156_mlp", anchor=_N_MLP_OLD, replacement=_N_MLP_NEW, required=True),
            TextPatch(name="pn156_att_fwd", anchor=_A_FWD_OLD, replacement=_A_FWD_NEW, required=True)],
            upstream_drift_markers=["_g156"]),
        TextPatcher(patch_name="PN156 GDN", target_file=str(fg), marker=MARKER, sub_patches=[
            TextPatch(name="pn156_g_imp", anchor=_G_IMP_OLD, replacement=_G_IMP_OLD + _IMP, required=True),
            TextPatch(name="pn156_g_fwd", anchor=_G_FWD_OLD, replacement=_G_FWD_NEW, required=True),
            TextPatch(name="pn156_g_cuda", anchor=_G_CUDA_OLD, replacement=_G_CUDA_NEW, required=True),
            TextPatch(name="pn156_g_qkvz", anchor=_G_QKVZ_OLD, replacement=_G_QKVZ_NEW, required=True)],
            upstream_drift_markers=["_g156"]),
        TextPatcher(patch_name="PN156 MLP", target_file=str(fm), marker=MARKER, sub_patches=[
            TextPatch(name="pn156_m_imp", anchor=_M_IMP_OLD, replacement=_M_IMP_OLD + _IMP, required=True),
            TextPatch(name="pn156_m_fwd", anchor=_M_FWD_OLD, replacement=_M_FWD_NEW, required=True)],
            upstream_drift_markers=["_g156"]),
    ]
    # Orden: GDN y MLP primero (solo agregan un argumento opcional _q156=None: inofensivos solos) y el
    # decoder AL FINAL, solo si los otros dos quedaron: el decoder les pasa _q156 y sin su parche seria un
    # TypeError. El anclaje del GDN exige PN155 (el checkpoint actual): sin PN155 no se aplica nada.
    gdn, mlp, dec = ps[1], ps[2], ps[0]
    for p in (gdn, mlp, dec):
        r, fl = p.apply()
        e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
        if e[0] != "applied":
            return e[0], f"{p.patch_name}: {e[1]}" + ("" if p is gdn else " (lo anterior quedo aplicado, inofensivo)")
    return "applied", "norma que escribe int8 (SK-26) en qkv, in_proj_qkvz y gate_up"
