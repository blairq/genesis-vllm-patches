# SPDX-License-Identifier: Apache-2.0
"""PN155 — in_proj_b/a del GDN (W4 en idiotSavant v2) dentro del Marlin de in_proj_qkvz; ver
``vllm._genesis.gdn_ba_qkvz``.
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN155: in_proj_ba dentro de in_proj_qkvz]"

_G_IMP_OLD = "from vllm.model_executor.layers.layernorm import RMSNormGated\n"
_G_IMP_NEW = _G_IMP_OLD + "from vllm._genesis import gdn_ba_qkvz as _g155  # " + MARKER + "\n"
_G_SIZES_OLD = "            else [key_dim, key_dim, value_dim, value_dim]\n"
_G_SIZES_NEW = "            else _g155.tamanos_qkvz(key_dim, value_dim, self.tp_size)  # " + MARKER + "\n"
_G_BA_OLD = "        self.in_proj_ba = self.create_ba_proj(\n"
_G_BA_NEW = "        self.in_proj_ba = None if _g155.ACTIVO else self.create_ba_proj(  # " + MARKER + "\n"
_G_PREP_OLD = "        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)\n"
_G_PREP_NEW = _G_PREP_OLD + "        _g155.preparar(self)  # " + MARKER + "\n"
_G_FWD_OLD = (
    "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "        ba, _ = self.in_proj_ba(hidden_states)\n"
    "\n"
    "        use_fused_gdn_decode = (\n"
)
_G_FWD_NEW = (
    "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "        if self.in_proj_ba is None:  # " + MARKER + "\n"
    "            mixed_qkvz, ba = _g155.separar(self, mixed_qkvz)\n"
    "        else:\n"
    "            ba, _ = self.in_proj_ba(hidden_states)\n"
    "\n"
    "        use_fused_gdn_decode = (\n"
)
_M_OLD = (
    "            \".in_proj_b\": (\".in_proj_ba\", 0),\n"
    "            \".in_proj_a\": (\".in_proj_ba\", 1),\n"
)
_M_NEW = (
    "            \".in_proj_b\": __import__(\"vllm._genesis.gdn_ba_qkvz\", fromlist=[\"x\"]).destino(\"b\"),  # " + MARKER + "\n"
    "            \".in_proj_a\": __import__(\"vllm._genesis.gdn_ba_qkvz\", fromlist=[\"x\"]).destino(\"a\"),\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN155")
    log_decision("PN155", decision, reason)
    if not decision:
        return "skipped", reason
    fg = resolve_vllm_file("model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py")
    fm = resolve_vllm_file("model_executor/models/qwen3_5.py")
    if fg is None or fm is None:
        return "skipped", "qwen_gdn_linear_attn.py / qwen3_5.py no estan en esta version de vLLM"
    pg = TextPatcher(
        patch_name="PN155 in_proj_ba en in_proj_qkvz (GDN)", target_file=str(fg), marker=MARKER,
        sub_patches=[TextPatch(name="pn155_import", anchor=_G_IMP_OLD, replacement=_G_IMP_NEW, required=True),
                     TextPatch(name="pn155_sizes", anchor=_G_SIZES_OLD, replacement=_G_SIZES_NEW, required=True),
                     TextPatch(name="pn155_ba", anchor=_G_BA_OLD, replacement=_G_BA_NEW, required=True),
                     TextPatch(name="pn155_preparar", anchor=_G_PREP_OLD, replacement=_G_PREP_NEW, required=True),
                     TextPatch(name="pn155_forward", anchor=_G_FWD_OLD, replacement=_G_FWD_NEW, required=True)],
        upstream_drift_markers=["_g155"])
    pm = TextPatcher(
        patch_name="PN155 mapeo de in_proj_b/a", target_file=str(fm), marker=MARKER,
        sub_patches=[TextPatch(name="pn155_mapeo", anchor=_M_OLD, replacement=_M_NEW, required=True)],
        upstream_drift_markers=["gdn_ba_qkvz"])
    rg, flg = pg.apply()
    rm, flm = pm.apply()
    sg = result_to_wiring_status(rg, flg, applied_message="GDN", patch_name=pg.patch_name)
    sm = result_to_wiring_status(rm, flm, applied_message="mapeo", patch_name=pm.patch_name)
    if sg[0] == "applied" and sm[0] == "applied":
        return "applied", "in_proj_b/a dentro del Marlin de in_proj_qkvz"
    return (sg if sg[0] != "applied" else sm)
