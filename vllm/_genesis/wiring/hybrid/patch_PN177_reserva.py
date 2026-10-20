# SPDX-License-Identifier: Apache-2.0
"""PN177 — la memoria que los parches piden despues del perfil se crea dentro del perfil, para que la KV
se dimensione contandola. Ver ``vllm._genesis.reserva``."""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN177: reserva antes del perfil]"
_OLD = (
    "    def profile_run(self) -> None:\n"
    "        if self.supports_mm_inputs and self.is_first_pp_rank:\n"
)
_NEW = (
    "    def profile_run(self) -> None:\n"
    "        from vllm._genesis import reserva as _g177  # " + MARKER + "\n"
    "\n"
    "        _g177.antes_del_perfil(self)\n"
    "        if self.supports_mm_inputs and self.is_first_pp_rank:\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN177")
    log_decision("PN177", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("v1/worker/gpu/model_runner.py")
    if f is None:
        return "skipped", "falta v1/worker/gpu/model_runner.py (model runner v2)"
    p = TextPatcher(patch_name="PN177 reserva antes del perfil", target_file=str(f), marker=MARKER,
                    sub_patches=[TextPatch(name="pn177_profile_run", anchor=_OLD, replacement=_NEW,
                                           required=True)],
                    upstream_drift_markers=["_g177"])
    r, fl = p.apply()
    return result_to_wiring_status(r, fl, applied_message="buffers de SK-18/GDN y cinta dentro del perfil",
                                   patch_name=p.patch_name)
