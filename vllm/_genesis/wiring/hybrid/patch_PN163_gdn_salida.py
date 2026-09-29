# SPDX-License-Identifier: Apache-2.0
"""PN163 — salida del GDN sin cero ni copia en los pasos solo-spec; ver ``vllm._genesis.gdn_salida``.
Toca qwen_gdn_linear_attn.py DESPUES de P28 (buffer persistente de core_attn_out) y PN122 (spec_update):
  1. forward compilado: el buffer de P28 ya no se pone a cero ahi;
  2. _forward_core: preparar_salida() lo pone a cero en todo paso que no sea solo-spec;
  3. spec_update escribe directo en core_attn_out[:num_actual_tokens];
  4. la mezcla saltea la copia si la salida ya es core_attn_out.
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN163: salida del GDN]"
_IMP_OLD = "from vllm import envs\n"
_IMP = "from vllm._genesis import gdn_salida as _g163  # " + MARKER + "\n"
_ALLOC_OLD = "            self._genesis_gdn_core_attn_buf[:num_tokens].zero_()\n"
_ALLOC_NEW = (
    "            (self._genesis_gdn_core_attn_buf[:num_tokens] if _g163.ACTIVO  # " + MARKER + "\n"
    "             else self._genesis_gdn_core_attn_buf[:num_tokens].zero_())\n"
)
_CORE_OLD = (
    "        assert isinstance(attn_metadata, GDNAttentionMetadata)\n"
    "\n"
    "        if (\n"
    "            self.enable_packed_recurrent_decode\n"
)
_CORE_NEW = (
    "        assert isinstance(attn_metadata, GDNAttentionMetadata)\n"
    "        _g163_directo = _g163.preparar_salida(attn_metadata, core_attn_out)  # " + MARKER + "\n"
    "\n"
    "        if (\n"
    "            self.enable_packed_recurrent_decode\n"
)
_SPEC_OLD = (
    "                attn_metadata.g122_slots,\n"
    "            )\n"
    "        elif spec_sequence_masks is not None:\n"
)
_SPEC_NEW = (
    "                attn_metadata.g122_slots,\n"
    "                o_dest=(core_attn_out[:num_actual_tokens].unsqueeze(0)  # " + MARKER + "\n"
    "                        if _g163_directo else None),\n"
    "            )\n"
    "        elif spec_sequence_masks is not None:\n"
)
_MERGE_OLD = (
    "        elif spec_sequence_masks is not None:\n"
    "            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)\n"
)
_MERGE_NEW = (
    "        elif spec_sequence_masks is not None:\n"
    "            if core_attn_out_spec.data_ptr() != core_attn_out.data_ptr():  # " + MARKER + "\n"
    "                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN163")
    log_decision("PN163", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py")
    if f is None:
        return "skipped", "falta qwen_gdn_linear_attn.py"
    p = TextPatcher(patch_name="PN163 salida del GDN", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn163_imp", anchor=_IMP_OLD, replacement=_IMP_OLD + _IMP, required=True),
        TextPatch(name="pn163_alloc", anchor=_ALLOC_OLD, replacement=_ALLOC_NEW, required=True),
        TextPatch(name="pn163_core", anchor=_CORE_OLD, replacement=_CORE_NEW, required=True),
        TextPatch(name="pn163_spec", anchor=_SPEC_OLD, replacement=_SPEC_NEW, required=True),
        TextPatch(name="pn163_merge", anchor=_MERGE_OLD, replacement=_MERGE_NEW, required=True)],
        upstream_drift_markers=["_g163"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "GDN: sin cero ni copia de core_attn_out en los pasos solo-spec"
