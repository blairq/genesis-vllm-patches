# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN54 (plan v3 P0.7) — GDN contiguous-call deduplication.

OBSOLETO en v0.27.1: upstream absorbió ambos fixes.
Sub-A: la línea anclada (`ssm_state[...].contiguous()`) ya no existe —
`_forward_core` hace `initial_state = ssm_state[prefill_state_indices]`
sin `.contiguous()` redundante (v0.27.1, builder de metadata prefiere
índices de prefill). Sub-B: la rama LoRA `hasattr(self, 'in_proj_qkv')`
con `chunk()+.contiguous()` fue refactorizada fuera del forward CUDA
(create_qkvz_proj factory). El módulo queda como registro histórico;
sus anchors ya no calzan y apply() termina en no-op limpio.

Removes redundant `.contiguous()` calls in `gdn_linear_attn.py` that are
already guaranteed contiguous by upstream operator semantics OR re-enforced
by FLA's `@input_guard` decorator on every entry point.

Direction inspired by MLX-LM PR #1077 root-cause analysis (slice-kept-alive
parent retention causes shared-buffer leak in multi-turn). vllm equivalent:
each redundant `.contiguous()` allocates a fresh block; if the "old" block
is held by a still-running async kernel, allocator marks it as fragmented.
Per-turn allocator delta on Cliff 2b (Issue #19): observed +1400 MiB/turn,
~600 MiB attributable to unnecessary copies.

Sub-patches (SAFE-set, high confidence):

**Sub-A — ssm_state advanced-index gather (HIGH IMPACT, prefill hot path)**
Anchor at `gdn_linear_attn.py:984` —
`initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()`
Advanced indexing already produces a FRESH allocation (PyTorch contract);
`.contiguous()` is a no-op copy. Saves one full ssm_state-shape allocation
per prefill batch. Safety: downstream `chunk_gated_delta_rule` is decorated
with FLA `@input_guard` which forces contiguous on every input.

**Sub-B — LoRA branch b/a.contiguous() after chunk (LOW IMPACT, defensive)**
Anchor at `gdn_linear_attn.py:551-552` (LoRA branch, `hasattr(self, 'in_proj_qkv')`).
`chunk(2, dim=-1)` along the last dim returns contiguous halves; the explicit
`.contiguous()` is no-op. Genesis PROD does not use LoRA paths so impact ~0,
but cleanup keeps Genesis-vs-upstream diff smaller.

Deferred (need deeper FLA ABI verification, separate sprint if needed):
- FlashInfer branch (lines 95-100) — 5× squeeze.contiguous(), removal needs
  `chunk_gated_delta_rule_fi` ABI confirmation.
- gqa_interleaved branch — Qwen3-Next path, not Genesis PROD.
- PN50 already replaces Qwen3.5/3.6 contiguous branch (lines 562-575) with
  fused Triton kernel — no separate dedup needed there.

Models affected
---------------
- 27B Lorbus INT4 (TQ k8v4, NGRAM, FP8 short, FP8 long, DFlash) — sub-A fires
  on prefill path. Sub-B no-op (no LoRA).
- 35B (PROD, DFlash) — does NOT have GDN; patch fires neither sub.

Default OFF until live A/B on Cliff 2b multi-turn reproducer (Genesis Issue
#19) shows per-turn allocator delta drops below ~900 MiB.

Author: Sandermage (Sander) Barzov Aleksandr.
Inspiration: adurham (MLX-LM #1077) — slice-kept-alive class of bug.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn54_gdn_contiguous_dedup")

GENESIS_PN54_MARKER = "Genesis PN54 GDN contiguous dedup v7.66 (P0.7 Cliff 2b)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN54_GDN_CONTIGUOUS_DEDUP", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# ─── Sub-A — ssm_state[idx].contiguous() removal (HIGH IMPACT) ──────────────
SSM_STATE_OLD = (
    "        if attn_metadata.num_prefills > 0:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()  # type: ignore[index]"
)
SSM_STATE_NEW = (
    "        if attn_metadata.num_prefills > 0:\n"
    "            assert non_spec_state_indices_tensor is not None\n"
    "            # [Genesis PN54 P0.7] advanced index already produces fresh\n"
    "            # allocation — .contiguous() was a no-op copy. Downstream\n"
    "            # chunk_gated_delta_rule is wrapped by FLA @input_guard which\n"
    "            # restores contiguous-on-input invariant.\n"
    "            initial_state = ssm_state[non_spec_state_indices_tensor]  # type: ignore[index]"
)


# ─── Sub-B — LoRA branch b/a.contiguous() (LOW IMPACT, defensive) ───────────
LORA_BA_OLD = (
    "        if hasattr(self, \"in_proj_qkv\"):\n"
    "            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z\n"
    "            mixed_qkv, _ = self.in_proj_qkv(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)\n"
    "            z, _ = self.in_proj_z(hidden_states)\n"
    "            z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "            b, a = ba.chunk(2, dim=-1)\n"
    "            b = b.contiguous()\n"
    "            a = a.contiguous()"
)
LORA_BA_NEW = (
    "        if hasattr(self, \"in_proj_qkv\"):\n"
    "            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z\n"
    "            mixed_qkv, _ = self.in_proj_qkv(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)\n"
    "            z, _ = self.in_proj_z(hidden_states)\n"
    "            z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "            b, a = ba.chunk(2, dim=-1)\n"
    "            # [Genesis PN54 P0.7] chunk(dim=-1) returns contiguous halves;\n"
    "            # explicit .contiguous() was no-op copy. FLA @input_guard\n"
    "            # downstream restores invariant if any consumer needs it."
)


REARRANGE_QKV_OLD = (
    "        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)\n"
    "\n"
    "        fused = torch.cat(\n"
    "            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0\n"
    "        )\n"
    "\n"
    "        q_size = seq_len * q_dim\n"
    "        k_size = seq_len * k_dim\n"
    "\n"
    "        q_contig = fused[0:q_size]\n"
    "        k_contig = fused[q_size : q_size + k_size]\n"
    "        v_contig = fused[q_size + k_size :]\n"
    "\n"
    "        query = q_contig.view(1, seq_len, -1, self.head_k_dim)\n"
    "        key = k_contig.view(1, seq_len, -1, self.head_k_dim)\n"
    "        value = v_contig.view(1, seq_len, -1, self.head_v_dim)\n"
    "\n"
    "        return query, key, value"
)

REARRANGE_QKV_NEW = (
    "        # [Genesis PN54 P0.8] Direct contiguous views instead of torch.cat flattening.\n"
    "        # Eliminates 1D cat allocation + slicing, avoids VRAM allocator fragmentation.\n"
    "        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)\n"
    "        # [Genesis PN122] con la recurrencia del arbol en PTX (gdn_arbol.cu), q/k/v van como VISTAS con\n"
    "        # stride de token: el kernel las lee asi y las tres copias de abajo (6,6 us por capa) sobran.\n"
    "        from vllm._genesis.gdn_cinta import qkv_con_stride as _genesis_qkv_stride\n"
    "        if _genesis_qkv_stride():\n"
    "            return (query.view(1, seq_len, -1, self.head_k_dim), key.view(1, seq_len, -1, self.head_k_dim),\n"
    "                    value.view(1, seq_len, -1, self.head_v_dim))\n"
    "        query = query.contiguous().view(1, seq_len, -1, self.head_k_dim)\n"
    "        key = key.contiguous().view(1, seq_len, -1, self.head_k_dim)\n"
    "        value = value.contiguous().view(1, seq_len, -1, self.head_v_dim)\n"
    "        return query, key, value"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN54 GDN contiguous dedup (P0.8 Cliff 2b + rearrange_mixed_qkv)",
        target_file=str(target),
        marker=GENESIS_PN54_MARKER,
        # Sub-A (ssm_state) y Sub-B (LoRA ba) quedaron OBSOLETOS en v0.27.1:
        # upstream absorbio ambos y sus anchors ya no existen (ver docstring).
        # Estaban como required=False bajo el mismo marker, que es justo la
        # trampa de aplicacion parcial que prohibe A-19: el que calza escribe el
        # marker y el otro no se reintenta nunca. Se quitan en vez de
        # enmascararlos; el texto de los anchors queda arriba como registro.
        sub_patches=[
            TextPatch(name="pn54_rearrange_mixed_qkv", anchor=REARRANGE_QKV_OLD,
                      replacement=REARRANGE_QKV_NEW, required=True),
        ],
        upstream_drift_markers=[],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN54")
    log_decision("PN54", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "qwen_gdn_linear_attn.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "PN54 applied: direct contiguous views in rearrange_mixed_qkv (avoids "
            "torch.cat + VRAM allocator fragmentation) and redundant contiguous dedup in GDN",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — anchor drifted or upstream removed call"
    return "failed", failure.reason if failure else "unknown failure"
