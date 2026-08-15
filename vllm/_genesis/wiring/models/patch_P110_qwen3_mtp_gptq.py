# SPDX-License-Identifier: Apache-2.0
"""Wiring for P110 — Fix Qwen3.5 MTP with GPTQ/AWQ quantization.

Backport of vllm-project/vllm#47828.
GPTQ and AWQ checkpoints typically have unquantized BF16 weights for MTP
layers, but vLLM incorrectly tries to load them using the quantized
config, causing a KeyError for 'layers.0.mlp.experts.w2_weight' (because
it looks for '.qweight' instead). This patch bypasses quantization config
for MTP layers automatically when GPTQ/AWQ is detected.

2026-07-31 EXTENSION (root-caused live on genesis-35b-heretic, MTP-Preserved
GPTQ-Int4, 0% spec-decode acceptance across K=2/K=3, thinking on/off):
the original p110_mtp_gptq_bypass sub-patch only reaches `self.quant_config`
on the OUTER `Qwen3_5MTP` wrapper class, which is only consumed by its
`lm_head`. It never reaches `Qwen3_5MultiTokenPredictor` (the INNER class
that actually builds the MTP decoder block) — that class reads
`vllm_config.quant_config` fresh in its own __init__ and passes the
untouched `vllm_config` straight into `Qwen3_5DecoderLayer(...)`. Result:
the MTP decoder's MoE experts get built expecting GPTQ-packed weights
(qweight/qzeros/scales), the checkpoint's `mtp.layers.0.mlp.experts.*`
keys are plain unquantized tensors (confirmed via
model.safetensors.index.json — no quant siblings, same class of gap as
the pre-existing modelopt_fp4-only `mtp.fc` workaround a few lines below
in this same predictor, vllm#38650), and the loader silently skips them:
  WARNING [qwen3_5_mtp.py] Parameter layers.0.mlp.experts.down_proj not
  found in params_dict, skip loading
  WARNING [qwen3_5_mtp.py] Parameter layers.0.mlp.experts.gate_up_proj not
  found in params_dict, skip loading
Leaves the MTP expert FFN at random init -> predicts noise -> verifier
rejects every draft token, 0% acceptance regardless of K or thinking.
New sub-patch `p110_mtp_gptq_predictor_bypass` extends the SAME bypass
condition to the predictor's `fc_quant` (was modelopt_fp4-only) and to a
shallow-copied `vllm_config` (quant_config=None) passed to
`Qwen3_5DecoderLayer`, so its attention + MoE submodules build unquantized
to match what the checkpoint actually stores. Only affects the MTP
predictor's own layers — the main model's 40 real layers keep full GPTQ
quantization untouched (separate `vllm_config` instance, main model
construction never sees the copy).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.p110_qwen3_mtp_gptq")
GENESIS_P110_MARKER = "Genesis P110 Qwen3 MTP GPTQ/AWQ fix v7.63.x guard-v4"
P110_ANCHOR = (
    "        self.quant_config = vllm_config.quant_config\n"
)

P110_REPLACEMENT = (
    "        # [Genesis P110 Qwen3 MTP GPTQ/AWQ fix v7.63.x guard-v4]\n"
    "        # GPTQ/AWQ: MTP decoder layers use unquantized bf16 weights.\n"
    "        bypass_mtp_quant = False\n"
    "        if vllm_config.quant_config and vllm_config.quant_config.get_name() not in (\"modelopt_fp4\",):\n"
    "            hf_qc = getattr(vllm_config.model_config.hf_config, \"quantization_config\", None)\n"
    "            if isinstance(hf_qc, dict):\n"
    "                dynamic = hf_qc.get(\"dynamic\", {})\n"
    "                if any(k.startswith(\"-:\") and \"mtp\" in k for k in dynamic):\n"
    "                    bypass_mtp_quant = True\n"
    "            if not bypass_mtp_quant and vllm_config.quant_config.get_name() in (\n"
    "                \"gptq\", \"auto_gptq\", \"gptq_marlin\", \"awq\", \"awq_marlin\",\n"
    "                \"compressed-tensors\", \"compressed_tensors\", \"marlin\", \"compressed_tensors_wNa16\",\n"
    "            ):\n"
    "                bypass_mtp_quant = True\n"
    "        self.quant_config = None if bypass_mtp_quant else vllm_config.quant_config\n"
)


P110_GUARD_ANCHOR = (
    "                    if is_fused_expert:\n"
    "                        # qwen3.5 no need to transpose\n"
    "                        # loaded_weight = loaded_weight.transpose(-1, -2)\n"
    "                        if \"experts.gate_up_proj\" in name:\n"
)

P110_GUARD_REPLACEMENT = (
    "                    if is_fused_expert:\n"
    "                        # qwen3.5 no need to transpose\n"
    "                        # loaded_weight = loaded_weight.transpose(-1, -2)\n"
    "                        if name_mapped not in params_dict:\n"
    "                            is_expert_weight = False\n"
    "                            continue\n"
    "                        if \"experts.gate_up_proj\" in name:\n"
)

P110_PREDICTOR_ANCHOR = (
    "        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is\n"
    "        # missing from hf_quant_config.json exclude_modules. Force unquantized.\n"
    "        # Ref: https://github.com/vllm-project/vllm/pull/38650\n"
    "        # Ref: https://github.com/NVIDIA/Model-Optimizer/pull/1124\n"
    "        fc_quant = (\n"
    "            None\n"
    "            if (quant_config and quant_config.get_name() == \"modelopt_fp4\")\n"
    "            else quant_config\n"
    "        )\n"
    "        self.fc = ColumnParallelLinear(\n"
    "            self.config.hidden_size * 2,\n"
    "            self.config.hidden_size,\n"
    "            gather_output=True,\n"
    "            bias=False,\n"
    "            return_bias=False,\n"
    "            quant_config=fc_quant,\n"
    "            prefix=f\"{prefix}.fc\",\n"
    "        )\n"
    "\n"
    "        self.layers = torch.nn.ModuleList(\n"
    "            Qwen3_5DecoderLayer(\n"
    "                vllm_config,\n"
    "                layer_type=\"full_attention\",\n"
    "                prefix=f\"{prefix}.layers.{idx}\",\n"
    "            )\n"
    "            for idx in range(self.num_mtp_layers)\n"
    "        )\n"
)

P110_PREDICTOR_REPLACEMENT = (
    "        # [Genesis P110 Qwen3 MTP GPTQ/AWQ fix v7.63.x guard-v4]\n"
    "        # GPTQ/AWQ checkpoints store the WHOLE MTP predictor block (fc +\n"
    "        # attention + MoE experts) as plain unquantized BF16 tensors, not\n"
    "        # just `fc` (which upstream already special-cases for modelopt_fp4\n"
    "        # right below). Building `self.layers` with the real quant_config\n"
    "        # makes the MoE experts expect GPTQ-packed weights that never\n"
    "        # arrive -> silently skipped at load -> MTP head stays randomly\n"
    "        # initialized -> 0% spec-decode acceptance. Reuse the same\n"
    "        # bypass condition as the outer Qwen3_5MTP.quant_config patch.\n"
    "        _genesis_p110_bypass_mtp_quant = False\n"
    "        if quant_config and quant_config.get_name() not in (\"modelopt_fp4\",):\n"
    "            _genesis_p110_hf_qc = getattr(\n"
    "                model_config.hf_config, \"quantization_config\", None\n"
    "            )\n"
    "            if isinstance(_genesis_p110_hf_qc, dict):\n"
    "                _genesis_p110_dynamic = _genesis_p110_hf_qc.get(\"dynamic\", {})\n"
    "                if any(\n"
    "                    k.startswith(\"-:\") and \"mtp\" in k\n"
    "                    for k in _genesis_p110_dynamic\n"
    "                ):\n"
    "                    _genesis_p110_bypass_mtp_quant = True\n"
    "            if not _genesis_p110_bypass_mtp_quant and quant_config.get_name() in (\n"
    "                \"gptq\", \"auto_gptq\", \"gptq_marlin\", \"awq\", \"awq_marlin\",\n"
    "                \"compressed-tensors\", \"compressed_tensors\", \"marlin\", \"compressed_tensors_wNa16\",\n"
    "            ):\n"
    "                _genesis_p110_bypass_mtp_quant = True\n"
    "\n"
    "        # Workaround: mtp.fc is stored as BF16 in NVFP4 checkpoints but is\n"
    "        # missing from hf_quant_config.json exclude_modules. Force unquantized.\n"
    "        # Ref: https://github.com/vllm-project/vllm/pull/38650\n"
    "        # Ref: https://github.com/NVIDIA/Model-Optimizer/pull/1124\n"
    "        fc_quant = (\n"
    "            None\n"
    "            if (\n"
    "                (quant_config and quant_config.get_name() == \"modelopt_fp4\")\n"
    "                or _genesis_p110_bypass_mtp_quant\n"
    "            )\n"
    "            else quant_config\n"
    "        )\n"
    "        self.fc = ColumnParallelLinear(\n"
    "            self.config.hidden_size * 2,\n"
    "            self.config.hidden_size,\n"
    "            gather_output=True,\n"
    "            bias=False,\n"
    "            return_bias=False,\n"
    "            quant_config=fc_quant,\n"
    "            prefix=f\"{prefix}.fc\",\n"
    "        )\n"
    "\n"
    "        # [Genesis P110 extended] Same bypass for the decoder block: build\n"
    "        # it from a shallow-copied VllmConfig with quant_config cleared, so\n"
    "        # Qwen3_5DecoderLayer's attention + MoE submodules match the\n"
    "        # checkpoint's actual (unquantized) MTP weights. Only this local\n"
    "        # copy is affected — the main model's real layers keep their own\n"
    "        # untouched vllm_config/quant_config.\n"
    "        _genesis_p110_mtp_vllm_config = vllm_config\n"
    "        if _genesis_p110_bypass_mtp_quant:\n"
    "            import copy as _genesis_p110_copy\n"
    "\n"
    "            _genesis_p110_mtp_vllm_config = _genesis_p110_copy.copy(vllm_config)\n"
    "            _genesis_p110_mtp_vllm_config.quant_config = None\n"
    "\n"
    "        self.layers = torch.nn.ModuleList(\n"
    "            Qwen3_5DecoderLayer(\n"
    "                _genesis_p110_mtp_vllm_config,\n"
    "                layer_type=\"full_attention\",\n"
    "                prefix=f\"{prefix}.layers.{idx}\",\n"
    "            )\n"
    "            for idx in range(self.num_mtp_layers)\n"
    "        )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/models/qwen3_5_mtp.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P110 Qwen3 MTP GPTQ/AWQ loader fix (vllm#47828 backport)",
        target_file=str(target),
        marker=GENESIS_P110_MARKER,
        sub_patches=[
            TextPatch(
                name="p110_mtp_gptq_bypass",
                anchor=P110_ANCHOR,
                replacement=P110_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="p110_mtp_gptq_guard",
                anchor=P110_GUARD_ANCHOR,
                replacement=P110_GUARD_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="p110_mtp_gptq_predictor_bypass",
                anchor=P110_PREDICTOR_ANCHOR,
                replacement=P110_PREDICTOR_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    """Apply P110."""
    from vllm._genesis.dispatcher import log_decision, should_apply

    from vllm._genesis.dispatcher import log_decision

    decision, reason = True, "Forced by custom patch"
    log_decision("P110", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file not resolvable"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "P110 applied: Qwen3.5 MTP now correctly handles GPTQ/AWQ models by "
            "bypassing quantization for the MTP layers (vllm#47828 backport). "
            "Fixes KeyError 'layers.0.mlp.experts.w2_weight' crash. Extended "
            "2026-07-31: bypass now also reaches the predictor's fc + decoder "
            "layers (was lm_head-only before), fixing silent MTP expert weight "
            "skip -> 0% spec-decode acceptance on GPTQ MTP-Preserved checkpoints."
        ),
        patch_name=patcher.patch_name,
    )
