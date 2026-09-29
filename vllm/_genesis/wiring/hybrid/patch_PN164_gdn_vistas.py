# SPDX-License-Identifier: Apache-2.0
"""PN164 — entradas del GDN como vistas de in_proj, sin la copia por capa; ver ``vllm._genesis.gdn_vistas``.
Toca qwen_gdn_linear_attn.py DESPUES de PN155 (separar) y PN50:
  1. forward_cuda: con PN164, (mixed_qkv, z, b, a) = vistas; se saltea el armado de PyTorch/PN50;
  2. la llamada al op pasa b y a sin .contiguous();
  3. _forward_core: en_op() las vuelve contiguas salvo en el paso solo-spec del arbol PTX.
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN164: vistas del GDN]"
_IMP_OLD = "from vllm import envs\n"
_IMP = "from vllm._genesis import gdn_vistas as _g164  # " + MARKER + "\n"
_SEP_OLD = "            mixed_qkvz, ba = _g155.separar(self, mixed_qkvz)\n"
_SEP_NEW = (
    "            if _g164.ACTIVO:  # " + MARKER + "\n"
    "                mixed_qkv, z, b, a = _g164.vistas(self, mixed_qkvz)\n"
    "                ba = None\n"
    "            else:\n"
    "                mixed_qkvz, ba = _g155.separar(self, mixed_qkvz)\n"
)
_ARM_OLD = (
    "            output, _ = self.out_proj(core_attn_out.flatten(-2))\n"
    "            return output\n"
    "\n"
    "        if self.gqa_interleaved_layout:\n"
)
_ARM_NEW = (
    "            output, _ = self.out_proj(core_attn_out.flatten(-2))\n"
    "            return output\n"
    "\n"
    "        if _g164.ACTIVO and self.in_proj_ba is None:  # " + MARKER + " ya estan armadas\n"
    "            pass\n"
    "        elif self.gqa_interleaved_layout:\n"
)
_OP_OLD = (
    "            b.contiguous(),\n"
    "            a.contiguous(),\n"
    "            core_attn_out,\n"
    "            layer_name=_encode_layer_name(self.prefix),\n"
    "        )\n"
)
_OP_NEW = (
    "            b if _g164.ACTIVO else b.contiguous(),  # " + MARKER + "\n"
    "            a if _g164.ACTIVO else a.contiguous(),\n"
    "            core_attn_out,\n"
    "            layer_name=_encode_layer_name(self.prefix),\n"
    "        )\n"
)
_CORE_OLD = (
    "        a = a[:num_actual_tokens]\n"
    "\n"
    "        # 1. Convolution sequence transformation\n"
    "        conv_weights = self.conv1d.weight.view(\n"
    "            self.conv1d.weight.size(0), self.conv1d.weight.size(2)\n"
    "        )\n"
    "\n"
    "        if spec_sequence_masks is not None:\n"
    "            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:\n"
    "                mixed_qkv_spec = mixed_qkv\n"
)
_CORE_NEW = _CORE_OLD.replace(
    "        a = a[:num_actual_tokens]\n",
    "        a = a[:num_actual_tokens]\n"
    "        mixed_qkv, b, a = _g164.en_op(attn_metadata, mixed_qkv, b, a)  # " + MARKER + "\n", 1)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN164")
    log_decision("PN164", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py")
    if f is None:
        return "skipped", "falta qwen_gdn_linear_attn.py"
    p = TextPatcher(patch_name="PN164 vistas del GDN", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn164_imp", anchor=_IMP_OLD, replacement=_IMP_OLD + _IMP, required=True),
        TextPatch(name="pn164_sep", anchor=_SEP_OLD, replacement=_SEP_NEW, required=True),
        TextPatch(name="pn164_arm", anchor=_ARM_OLD, replacement=_ARM_NEW, required=True),
        TextPatch(name="pn164_op", anchor=_OP_OLD, replacement=_OP_NEW, required=True),
        TextPatch(name="pn164_core", anchor=_CORE_OLD, replacement=_CORE_NEW, required=True)],
        upstream_drift_markers=["_g164"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "GDN: mixed_qkv/z/b/a como vistas de in_proj, sin la copia por capa"
