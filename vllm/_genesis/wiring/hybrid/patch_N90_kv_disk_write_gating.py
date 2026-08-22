# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N90 — KV Disk Write Gating for secondary storage tiers."""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN90_MARKER = "[Genesis PN90 kv disk write gating]"

# ─────────────────── tiering/manager.py ───────────────────

A1_OLD = (
    "        # Step 2: Cascade to ALL secondary tiers\n"
    "        # For each secondary tier, call primary.prepare_read() to get the\n"
    "        # LoadStoreSpec AND to increment ref_cnt (protecting blocks from\n"
    "        # eviction during the async transfer). One prepare_read() call per\n"
    "        # secondary tier.\n"
    "        for tier in self.secondary_tiers:\n"
)

A1_NEW = (
    "        # " + GENESIS_PN90_MARKER + "\n"
    "        from vllm._genesis import kv_disk_gate as _g90\n"
    "        if not _g90.should_persist_to_secondary_tiers(req_context.kv_transfer_params):\n"
    "            return\n"
    "\n"
    "        # Step 2: Cascade to ALL secondary tiers\n"
    "        # For each secondary tier, call primary.prepare_read() to get the\n"
    "        # LoadStoreSpec AND to increment ref_cnt (protecting blocks from\n"
    "        # eviction during the async transfer). One prepare_read() call per\n"
    "        # secondary tier.\n"
    "        for tier in self.secondary_tiers:\n"
)


def apply() -> tuple[str, str]:
    target = resolve_vllm_file("v1/kv_offload/tiering/manager.py")
    if target is None:
        return "skipped", "target file v1/kv_offload/tiering/manager.py not found"

    patcher = TextPatcher(
        patch_name="PN90 kv disk write gating",
        target_file=str(target),
        marker=GENESIS_PN90_MARKER,
        sub_patches=[
            TextPatch(
                name="pn90_complete_store_disk_gate",
                anchor=A1_OLD,
                replacement=A1_NEW,
                required=True,
            ),
        ],
    )
    res, failure = patcher.apply()
    return result_to_wiring_status(
        res,
        failure,
        applied_message="PN90 applied: KV disk write gating activo (subagentes no escriben a SSD)",
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    target = resolve_vllm_file("v1/kv_offload/tiering/manager.py")
    if target is None:
        return False
    try:
        with open(target) as f:
            return GENESIS_PN90_MARKER in f.read()
    except Exception:
        return False
