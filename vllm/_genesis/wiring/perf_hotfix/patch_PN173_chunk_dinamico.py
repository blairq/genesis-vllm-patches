# SPDX-License-Identifier: Apache-2.0
"""PN173 — chunk de prefill dinamico (chico si hay decodes); ver ``vllm._genesis.chunk_dinamico``."""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN173: chunk de prefill dinamico]"
_OLD = (
    "        self.current_step += 1\n"
    "        # NOTE(woosuk) on the scheduling algorithm:\n"
)
_NEW = (
    "        self.current_step += 1\n"
    "        __import__('vllm._genesis.chunk_dinamico', fromlist=['x']).ajustar(self)  # " + MARKER + "\n"
    "        # NOTE(woosuk) on the scheduling algorithm:\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN173")
    log_decision("PN173", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("v1/core/sched/scheduler.py")
    if f is None:
        return "skipped", "falta scheduler.py"
    p = TextPatcher(patch_name="PN173 chunk de prefill dinamico", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn173_ajustar", anchor=_OLD, replacement=_NEW, required=True)],
        upstream_drift_markers=["chunk_dinamico"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "chunk de prefill: grande sin decodes, chico con decodes"
