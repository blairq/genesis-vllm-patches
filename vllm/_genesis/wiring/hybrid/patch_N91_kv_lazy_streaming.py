# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN91 — deferral acotado de promociones SSD → RAM.

Cuatro anclas en
`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`:

  1. `_maximal_prefix_lookup` — en modo estricto, un bloque en vuelo cuenta como
     MISS y corta el prefijo, en vez de contar como hit y forzar el deferral.
  2. `_sliding_window_lookup` — ídem para los grupos recurrentes / SWA.
  3. `_lookup` — contabiliza cada deferral y arma el modo estricto al pasarse
     del presupuesto.
  4. `request_finished` — limpia el estado por request.

La versión anterior de este parche apuntaba a un ancla que NO EXISTE en el
fuente (`if request.request_id in self._req_status:`), con `required=True`, así
que `TextPatcher` abortaba y el parche nunca se aplicaba — mientras `apply()`
reportaba "applied". Las cuatro anclas de acá están verificadas contra el árbol.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN91_MARKER = "[Genesis PN91: bounded promotion deferral]"

# ─────────── 1. _maximal_prefix_lookup ───────────

PREFIX_LOOKUP_OLD = (
    "                case LookupResult.HIT_PENDING:\n"
    "                    defer_lookup = True\n"
    "                    hit_count += 1\n"
)

PREFIX_LOOKUP_NEW = (
    "                case LookupResult.HIT_PENDING:\n"
    "                    # " + GENESIS_PN91_MARKER + "\n"
    "                    # En modo estricto un bloque en vuelo corta el prefijo: lo que\n"
    "                    # se devuelve contiene SOLO bloques listos, que es lo que\n"
    "                    # prepare_load exige (assert block.is_ready).\n"
    "                    from vllm._genesis import kv_lazy_streaming as _g91\n"
    "                    if _g91.is_strict(req_context.req_id):\n"
    "                        break\n"
    "                    defer_lookup = True\n"
    "                    hit_count += 1\n"
)

# ─────────── 2. _sliding_window_lookup ───────────

SLIDING_LOOKUP_OLD = (
    "                case LookupResult.HIT_PENDING:\n"
    "                    # Block is in cache, just not readable yet — counts\n"
    "                    # as hit for the consecutive streak. Don't break:\n"
    "                    # keep scanning to let manager kick off async lookups.\n"
    "                    defer_lookup = True\n"
    "                    consecutive_hits += 1\n"
)

SLIDING_LOOKUP_NEW = (
    "                case LookupResult.HIT_PENDING:\n"
    "                    # " + GENESIS_PN91_MARKER + "\n"
    "                    from vllm._genesis import kv_lazy_streaming as _g91\n"
    "                    if _g91.is_strict(req_context.req_id):\n"
    "                        consecutive_hits = 0\n"
    "                    else:\n"
    "                        defer_lookup = True\n"
    "                        consecutive_hits += 1\n"
)

# ─────────── 3. contabilidad del deferral ───────────

DEFER_OLD = (
    "        if defer_lookup:\n"
    "            logger.debug(\n"
    "                \"Offloading manager delayed request %s as backend requested\",\n"
    "                req_status.req.request_id,\n"
    "            )\n"
    "            return None\n"
)

DEFER_NEW = (
    "        if defer_lookup:\n"
    "            # " + GENESIS_PN91_MARKER + "\n"
    "            from vllm._genesis import kv_lazy_streaming as _g91\n"
    "            _g91.note_deferral(req_status.req.request_id)\n"
    "            logger.debug(\n"
    "                \"Offloading manager delayed request %s as backend requested\",\n"
    "                req_status.req.request_id,\n"
    "            )\n"
    "            return None\n"
)

# ─────────── 4. limpieza por request ───────────

CLEANUP_OLD = (
    "        req_status = self._req_status.get(request.request_id)\n"
    "\n"
    "        if req_status is None:\n"
)

CLEANUP_NEW = (
    "        # " + GENESIS_PN91_MARKER + "\n"
    "        from vllm._genesis import kv_lazy_streaming as _g91\n"
    "        _g91.clear_request(request.request_id)\n"
    "        req_status = self._req_status.get(request.request_id)\n"
    "\n"
    "        if req_status is None:\n"
)


def _scheduler_patcher() -> TextPatcher | None:
    root = vllm_install_root()
    if root is None:
        return None
    target = os.path.join(
        root,
        "distributed",
        "kv_transfer",
        "kv_connector",
        "v1",
        "offloading",
        "scheduler.py",
    )
    if not os.path.exists(target):
        return None
    return TextPatcher(
        patch_name="PN91 bounded promotion deferral (offloading scheduler)",
        target_file=target,
        marker=GENESIS_PN91_MARKER,
        sub_patches=[
            TextPatch(
                name="pn91_prefix_lookup_strict",
                anchor=PREFIX_LOOKUP_OLD,
                replacement=PREFIX_LOOKUP_NEW,
                required=True,
            ),
            TextPatch(
                name="pn91_sliding_lookup_strict",
                anchor=SLIDING_LOOKUP_OLD,
                replacement=SLIDING_LOOKUP_NEW,
                required=True,
            ),
            TextPatch(
                name="pn91_note_deferral",
                anchor=DEFER_OLD,
                replacement=DEFER_NEW,
                required=True,
            ),
            TextPatch(
                name="pn91_clear_request_state",
                anchor=CLEANUP_OLD,
                replacement=CLEANUP_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN91")
    log_decision("PN91", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    p = _scheduler_patcher()
    if p is None:
        return "skipped", "target de _scheduler_patcher no encontrado"

    result, failure = p.apply()
    status, msg = result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN91 aplicado: deferral acotado de promociones "
            "(pasado GENESIS_PN91_MAX_DEFER_STEPS el request arranca con el "
            "prefijo que ya está listo en vez de seguir esperando)."
        ),
        patch_name=p.patch_name,
    )
    return status, msg


def is_applied() -> bool:
    p = _scheduler_patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN91_MARKER in f.read()
    except Exception:
        return False
