# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN105 — `reset_cache()` del tiering manager, que no existia.

`OffloadingManager.reset_cache()` en base.py es un no-op (`return`).
`CPUOffloadingManager` lo implementa bien, pero **`TieringOffloadingManager`
NO lo sobreescribe**, asi que hereda el vacio. Y como devuelve `None` y no
`False`, el chequeo del scheduler

    if self.connector.reset_cache() is False:
        return False

nunca detecta la falla: el reset reporta exito y no limpia nada.

Consecuencia medida (2026-08-23): el boton "Reset Total" del dashboard
respondia `{"l2_ram_arc_cache": true}` y L2 quedaba intacta. Verificado con
PN104, que atribuye el acierto por tier:

    tras el reset total, mismo prompt -> 1,0 s, cached=24.128
      de L1 (GPU)     :      0   <- L1 SI se limpia
      del tier externo : 24.128   <- L2 NO

O sea que el usuario cree que arranca de cero y sigue midiendo sobre una
cache caliente. Para un panel cuyo proposito es medir, eso es peor que no
tener el boton.

PN105 implementa el metodo delegando en el tier primario, que es el que
tiene la logica correcta. Los secundarios no exponen ningun metodo de reset
en su interfaz (`tiering/base.py` solo tiene `shutdown()`), asi que L3 se
sigue limpiando por borrado de archivos, que es lo que ya hace PN89.

Kill switch: `GENESIS_DISABLE_PN105=1`.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN105_MARKER = "_GENESIS_PN105_TIERING_RESET_CACHE"

# ─────────── 2. el reset deja job ids colgados por request ───────────
#
# Al hacer que el reset LIMPIE DE VERDAD se destapa un segundo bug que nunca
# se habia podido ver, porque el camino no se ejercitaba: `reset_cache()` del
# connector vacia `self._jobs` pero NO limpia `req_status.transfer_jobs`, que
# es donde viven los ids por request. En el paso siguiente:
#
#     any_jid = next(iter(req_status.transfer_jobs))
#     assert self._jobs[any_jid].is_store      -> KeyError: 110
#
# y el EngineCore muere. Medido el 2026-08-24: el engine se cayo en el primer
# request posterior a un reset exitoso.
#
# El loop que sigue ya recorre los requests para resetear next_stored_block_idx,
# asi que alcanza con vaciar ahi tambien los transfer_jobs.

JOBS_OLD = (
    "        # Reset store progress so active requests re-offload from block 0\n"
    "        for status in self._req_status.values():\n"
    "            for group_state in status.group_states:\n"
    "                group_state.next_stored_block_idx = 0\n"
)

JOBS_NEW = (
    "        # Reset store progress so active requests re-offload from block 0\n"
    "        for status in self._req_status.values():\n"
    "            for group_state in status.group_states:\n"
    "                group_state.next_stored_block_idx = 0\n"
    "            # " + GENESIS_PN105_MARKER + "\n"
    "            # Los ids por request tienen que morir con self._jobs. Si no,\n"
    "            # el paso siguiente hace self._jobs[any_jid] sobre un dict ya\n"
    "            # vacio y el EngineCore muere con KeyError.\n"
    "            status.transfer_jobs.clear()\n"
)

ANCHOR_OLD = "    def _flush_pending_promotions(self) -> None:\n"

ANCHOR_NEW = (
    "    # " + GENESIS_PN105_MARKER + "\n"
    "    def reset_cache(self) -> None:\n"
    "        \"\"\"Vacia el tier primario. SIN esto el reset es un no-op silencioso.\n"
    "\n"
    "        OffloadingManager.reset_cache() en base.py es `return`, y esta clase\n"
    "        no lo sobreescribia, asi que heredaba el vacio. El scheduler chequea\n"
    "        `if self.connector.reset_cache() is False`, y como el no-op devuelve\n"
    "        None la falla nunca se detectaba: el reset reportaba exito con L2\n"
    "        intacta.\n"
    "\n"
    "        Los tiers secundarios no exponen reset en su interfaz, asi que L3 se\n"
    "        limpia por borrado de archivos (PN89).\n"
    "        \"\"\"\n"
    "        self.primary_tier.reset_cache()\n"
    "        try:\n"
    "            from vllm._genesis import kv_tier_attribution as _g105_attr\n"
    "            from vllm._genesis import kv_staging_ring as _g105_ring\n"
    "\n"
    "            _g105_attr.reset()\n"
    "            _g105_ring.reset_vistos()\n"
    "            if hasattr(self.primary_tier, '_g100_reg'):\n"
    "                self.primary_tier._g100_reg.clear()\n"
    "            if hasattr(self.primary_tier, '_g102'):\n"
    "                del self.primary_tier._g102\n"
    "        except Exception:\n"
    "            pass\n"
    "        logger.info(\n"
    "            'PN105: cache del tier primario vaciada (%d bloques disponibles)',\n"
    "            self.primary_tier._num_blocks,\n"
    "        )\n"
    "\n"
    "    def _flush_pending_promotions(self) -> None:\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN105", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _patcher() -> TextPatcher | None:
    root = vllm_install_root()
    if root is None:
        return None
    t = os.path.join(root, "v1", "kv_offload", "tiering", "manager.py")
    if not os.path.exists(t):
        return None
    return TextPatcher(
        patch_name="PN105 tiering reset_cache",
        target_file=t,
        marker=GENESIS_PN105_MARKER,
        sub_patches=[
            TextPatch(name="pn105_reset_cache", anchor=ANCHOR_OLD,
                      replacement=ANCHOR_NEW, required=True),
        ],
    )


def _patcher_scheduler() -> TextPatcher | None:
    root = vllm_install_root()
    if root is None:
        return None
    t = os.path.join(
        root, "distributed", "kv_transfer", "kv_connector", "v1",
        "offloading", "scheduler.py",
    )
    if not os.path.exists(t):
        return None
    return TextPatcher(
        patch_name="PN105 reset clears per-request job ids",
        target_file=t,
        marker=GENESIS_PN105_MARKER,
        sub_patches=[
            TextPatch(name="pn105_clear_transfer_jobs", anchor=JOBS_OLD,
                      replacement=JOBS_NEW, required=True),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN105")
    log_decision("PN105", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN105 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    ps = [_patcher(), _patcher_scheduler()]
    if any(p is None for p in ps):
        return "skipped", (
            "targets de tiering/manager.py u offloading/scheduler.py no hallados"
        )

    from vllm._genesis.wiring.text_patch import MultiFilePatchTransaction

    txn = MultiFilePatchTransaction(ps, name="PN105")
    status, reason = txn.apply_or_skip()
    if status != "applied":
        return status, reason
    return "applied", (
        "PN105 aplicado: TieringOffloadingManager.reset_cache() ahora existe y "
        "vacia el tier primario (heredaba un no-op de base.py, asi que el reset "
        "reportaba exito dejando L2 intacta), y el reset del connector limpia "
        "los job ids por request (sin eso el EngineCore moria con KeyError en el "
        "primer request posterior a un reset)."
    )


def is_applied() -> bool:
    for p in (_patcher(), _patcher_scheduler()):
        if p is None:
            return False
        try:
            with open(p.target_file) as f:
                if GENESIS_PN105_MARKER not in f.read():
                    return False
        except Exception:
            return False
    return True
