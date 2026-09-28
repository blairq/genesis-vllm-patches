# SPDX-License-Identifier: Apache-2.0
"""PN151 — la metadata del GDN del decode en arbol en un kernel (ver ``vllm._genesis.gdn_meta``).

Agrega al final de ``v1/attention/backends/gdn_attn.py`` la instalacion del envoltorio de
``GDNAttentionMetadataBuilder.build``: asi corre en cada proceso que importa el backend (los workers).
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN151: metadata GDN en un kernel]"

_FIN_OLD = "        return self.build(0, m, num_accepted_tokens, num_decode_draft_tokens_cpu)\n"
_FIN_NEW = (_FIN_OLD + "\n\nfrom vllm._genesis import gdn_meta as _g151  # " + MARKER + "\n"
            "_g151.instalar(GDNAttentionMetadataBuilder)  # " + MARKER + "\n")


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN151")
    log_decision("PN151", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("v1/attention/backends/gdn_attn.py")
    if f is None:
        return "skipped", "gdn_attn.py no esta en esta version de vLLM"
    p = TextPatcher(
        patch_name="PN151 metadata GDN en un kernel", target_file=str(f), marker=MARKER,
        sub_patches=[TextPatch(name="pn151_instalar", anchor=_FIN_OLD, replacement=_FIN_NEW, required=True)],
        upstream_drift_markers=["_g151"])
    r, fl = p.apply()
    return result_to_wiring_status(r, fl, applied_message="build del GDN con camino rapido", patch_name=p.patch_name)
