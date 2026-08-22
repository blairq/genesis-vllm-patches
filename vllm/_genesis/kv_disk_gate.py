# SPDX-License-Identifier: Apache-2.0
"""KV Disk Write Gating for multi-tier offloading (Genesis PN90).

Controls which agents and requests are permitted to cascade KV blocks to
secondary storage tiers (NVMe SSD), avoiding unnecessary write churn from
ephemeral subagents.
"""

from __future__ import annotations

import os
from typing import Any

# Primary/Orchestrator agents permitted to persist to secondary tiers (L3 SSD)
DEFAULT_DISK_WRITERS = {
    "build",
    "primary",
    "primary_low",
    "primary_high",
    "primary_nothink",
    "plan",
    "planner",
    "planner_nothink",
    "coach",
}


def should_persist_to_secondary_tiers(
    kv_transfer_params: dict[str, Any] | None,
) -> bool:
    """Determine whether a request is allowed to cascade blocks to secondary tiers (disk).

    Rules:
      1. If `persist_disk` is explicitly provided in `kv_transfer_params`, its boolean
         value takes absolute precedence.
      2. Otherwise, `genesis_agent` (or `agent`) is checked against the allowed list of disk writers
         (env `GENESIS_KV_DISK_WRITERS` or `DEFAULT_DISK_WRITERS`).
      3. Ephemeral subagents (e.g. coder, explorer, verifier, vision, art, utility)
         return False and are kept exclusively in L1 VRAM and L2 RAM.
    """
    if kv_transfer_params and "persist_disk" in kv_transfer_params:
        return bool(kv_transfer_params["persist_disk"])

    agent = None
    if kv_transfer_params:
        agent = kv_transfer_params.get("genesis_agent") or kv_transfer_params.get("agent")

    if agent:
        agent_clean = str(agent).strip().lower()
    else:
        # Strict protection: requests without explicit agent tag default to False (no disk writes)
        return False

    allowed_env = os.environ.get("GENESIS_KV_DISK_WRITERS")
    if allowed_env is not None:
        allowed_set = {a.strip().lower() for a in allowed_env.split(",") if a.strip()}
    else:
        allowed_set = DEFAULT_DISK_WRITERS

    return agent_clean in allowed_set

