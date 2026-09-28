# SPDX-License-Identifier: Apache-2.0
"""PN154 — Hadamard por cabeza en la entrada de o_proj (atencion) y out_proj (GDN); ver
``vllm._genesis.had_salidas``. Solo actua si el checkpoint lo declara (genesis_rotacion.had_entrada).
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN154: Hadamard en la entrada de o_proj/out_proj]"

# ── atencion (qwen3_next.py: la usa Qwen3.5; el borrador DFlash tiene su propia clase) ──
_A_IMP_OLD = "from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate\n"
_A_IMP_NEW = _A_IMP_OLD + "from vllm._genesis import had_salidas as _g154  # " + MARKER + "\n"
_A_INIT_OLD = (
    "            quant_config=quant_config,\n"
    "            prefix=f\"{prefix}.o_proj\",\n"
    "        )\n"
)
_A_INIT_NEW = _A_INIT_OLD + "        _g154.marcar(self.o_proj, \"o_proj\")  # " + MARKER + "\n"
_A_FWD_OLD = (
    "        if gate is not None:\n"
    "            attn_output = attn_output * torch.sigmoid(gate)\n"
    "        output, _ = self.o_proj(attn_output)\n"
)
_A_FWD_NEW = "        output = _g154.salida_atencion(self, attn_output, gate)  # " + MARKER + "\n"

# ── GDN (qwen_gdn_linear_attn.py) ──
_G_IMP_OLD = "from vllm.model_executor.layers.layernorm import RMSNormGated\n"
_G_IMP_NEW = _G_IMP_OLD + "from vllm._genesis import had_salidas as _g154  # " + MARKER + "\n"
_G_INIT_OLD = (
    "            quant_config=self.quant_config,\n"
    "            prefix=f\"{prefix}.out_proj\",\n"
    "        )\n"
)
_G_INIT_NEW = _G_INIT_OLD + "        _g154.marcar(self.out_proj, \"out_proj\")  # " + MARKER + "\n"
_G_FWD_OLD = (
    "        z_shape_og = z.shape\n"
    "        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])\n"
    "        z = z.reshape(-1, z.shape[-1])\n"
    "        core_attn_out = self.norm(core_attn_out, z)\n"
    "        core_attn_out = core_attn_out.reshape(z_shape_og)\n"
    "        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)\n"
    "        output, _ = self.out_proj(core_attn_out)\n"
    "        return output\n"
)
_G_FWD_NEW = "        return _g154.salida_gdn(self, core_attn_out, z)  # " + MARKER + "\n"


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN154")
    log_decision("PN154", decision, reason)
    if not decision:
        return "skipped", reason
    fa = resolve_vllm_file("model_executor/models/qwen3_next.py")
    fg = resolve_vllm_file("model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py")
    if fa is None or fg is None:
        return "skipped", "qwen3_next.py / qwen_gdn_linear_attn.py no estan en esta version de vLLM"
    pa = TextPatcher(
        patch_name="PN154 Hadamard en la entrada de o_proj", target_file=str(fa), marker=MARKER,
        sub_patches=[TextPatch(name="pn154_a_import", anchor=_A_IMP_OLD, replacement=_A_IMP_NEW, required=True),
                     TextPatch(name="pn154_a_init", anchor=_A_INIT_OLD, replacement=_A_INIT_NEW, required=True),
                     TextPatch(name="pn154_a_forward", anchor=_A_FWD_OLD, replacement=_A_FWD_NEW, required=True)],
        upstream_drift_markers=["_g154"])
    pg = TextPatcher(
        patch_name="PN154 Hadamard en la entrada de out_proj", target_file=str(fg), marker=MARKER,
        sub_patches=[TextPatch(name="pn154_g_import", anchor=_G_IMP_OLD, replacement=_G_IMP_NEW, required=True),
                     TextPatch(name="pn154_g_init", anchor=_G_INIT_OLD, replacement=_G_INIT_NEW, required=True),
                     TextPatch(name="pn154_g_forward", anchor=_G_FWD_OLD, replacement=_G_FWD_NEW, required=True)],
        upstream_drift_markers=["_g154"])
    ra, fla = pa.apply()
    rg, flg = pg.apply()
    sa = result_to_wiring_status(ra, fla, applied_message="o_proj", patch_name=pa.patch_name)
    sg = result_to_wiring_status(rg, flg, applied_message="out_proj", patch_name=pg.patch_name)
    if sa[0] == "applied" and sg[0] == "applied":
        return "applied", "Hadamard por cabeza en la entrada de o_proj y out_proj (si el checkpoint la pide)"
    return (sa if sa[0] != "applied" else sg)
