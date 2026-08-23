# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN95 — ocupación y desalojos del tier L2 (RAM).

================================================================
QUÉ RESUELVE
================================================================

De L2 se publicaba sólo `kv_tier_lookups_total{tier="ram"}` (hits y misses).
Faltaban las dos series que de verdad explican el comportamiento del sistema:

  1. **Ocupación.** `kv_tier_bytes_used` y `kv_tier_capacity_bytes` existían
     únicamente con `tier="disk"`. Cuánto de los 6 GiB de L2 se está usando
     sólo se podía inferir a mano desde `vllm:kv_offload_total_bytes`, y mal.

  2. **Desalojos.** `kv_tier_evictions_total{tier="ram"}` no existía. Es LA
     métrica del sistema tal como quedó después de PN90: si nada se desaloja
     de L2, nada baja al SSD. El desalojo de L2 **es** la escritura al SSD.
     Sin este contador, "el disco recibió 0 bytes" no se puede distinguir de
     "el disco recibió 0 bytes porque la carga fue chica", que es justo la
     duda que dejó la corrida del 2026-08-23.

Las dos series ya existen en el pipeline de PN88 para `tier="disk"`, así que
PN95 no agrega plumbing: sólo las alimenta con `tier="ram"`.

================================================================
DÓNDE
================================================================

`v1/kv_offload/cpu/manager.py`, que corre en el proceso del scheduler (es el
manager del tier primario dentro del TieringOffloadingManager), o sea el mismo
proceso desde el que PN88 ya publica.

  - `prepare_store`: justo después del bucle que libera los bloques evictados.
    Se engancha ahí y no en el `if ... self.events is not None` de abajo
    porque los eventos pueden estar apagados y el desalojo pasa igual.
  - `complete_store`: antes del bloque de eventos, por el mismo motivo. Es el
    único punto por el que pasan todas las altas y bajas del pool.

`CPUOffloadingManager` cuenta BLOQUES, no bytes: no conoce el tamaño de un
bloque. El tamaño lo registra PN93 al arrancar (`kv_bytes_per_offloaded_block`,
28.966.912 B en este despliegue). Si no está disponible se publica sólo el
conteo de desalojos y se omite la ocupación, en vez de publicar un número
inventado.

================================================================
COSTO
================================================================

Dos multiplicaciones y dos escrituras a un dict por `complete_store`. Todo va
adentro de `try/except` y detrás de `kv_tier_metrics.enabled()`.
Kill switch: `GENESIS_DISABLE_PN95=1`.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN95_MARKER = "_GENESIS_PN95_L2_OCCUPANCY"

EVICT_OLD = (
    "            for key, block in evicted:\n"
    "                self._free_block(block)\n"
    "                to_evict.append(key)\n"
)

EVICT_NEW = (
    "            for key, block in evicted:\n"
    "                self._free_block(block)\n"
    "                to_evict.append(key)\n"
    "            # " + GENESIS_PN95_MARKER + "\n"
    "            # El desalojo de L2 ES la escritura al SSD: con PN90, un\n"
    "            # bloque solo baja a disco cuando lo echan de RAM. Sin este\n"
    "            # contador no se puede distinguir 'no hubo presion' de\n"
    "            # 'la carga fue chica'.\n"
    "            try:\n"
    "                from vllm._genesis import kv_sparse_gdn as _g95_gdn\n"
    "                from vllm._genesis import kv_tier_metrics as _g95_m\n"
    "\n"
    "                if to_evict:\n"
    "                    _g95_bb = _g95_gdn.block_write_bytes()\n"
    "                    _g95_m.note_eviction(\n"
    '                        "ram",\n'
    '                        "capacity",\n'
    "                        count=len(to_evict),\n"
    "                        freed_bytes=len(to_evict) * _g95_bb,\n"
    "                    )\n"
    "            except Exception:\n"
    "                pass\n"
)

OCCUP_OLD = "        if stored_keys and self.events is not None:\n"

OCCUP_NEW = (
    "        # " + GENESIS_PN95_MARKER + "\n"
    "        # Unico punto por el que pasan todas las altas y bajas del pool.\n"
    "        # Va ANTES del bloque de eventos a proposito: los eventos pueden\n"
    "        # estar apagados y la ocupacion cambia igual.\n"
    "        try:\n"
    "            from vllm._genesis import kv_sparse_gdn as _g95_gdn\n"
    "            from vllm._genesis import kv_tier_metrics as _g95_m\n"
    "\n"
    "            _g95_bb = _g95_gdn.block_write_bytes()\n"
    "            if _g95_bb:\n"
    "                _g95_libres = self._get_num_free_blocks()\n"
    "                _g95_usados = self._num_blocks - _g95_libres\n"
    "                _g95_m.set_occupancy(\n"
    '                    "ram",\n'
    "                    _g95_usados * _g95_bb,\n"
    "                    self._num_blocks * _g95_bb,\n"
    "                )\n"
    "        except Exception:\n"
    "            pass\n"
    "        if stored_keys and self.events is not None:\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN95", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _patcher() -> TextPatcher | None:
    root = vllm_install_root()
    if root is None:
        return None
    target = os.path.join(root, "v1", "kv_offload", "cpu", "manager.py")
    if not os.path.exists(target):
        return None
    return TextPatcher(
        patch_name="PN95 L2 occupancy metrics",
        target_file=target,
        marker=GENESIS_PN95_MARKER,
        sub_patches=[
            TextPatch(
                name="pn95_l2_evictions",
                anchor=EVICT_OLD,
                replacement=EVICT_NEW,
                required=True,
            ),
            TextPatch(
                name="pn95_l2_occupancy",
                anchor=OCCUP_OLD,
                replacement=OCCUP_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN95")
    log_decision("PN95", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN95 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "v1/kv_offload/cpu/manager.py not found"

    result, failure = p.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN95 aplicado: L2 (RAM) publica kv_tier_bytes_used / "
            "kv_tier_capacity_bytes / kv_tier_evictions_total con tier=\"ram\". "
            "El desalojo de L2 es lo que PN90 convierte en escritura al SSD, "
            "asi que era el unico eslabon sin medir de la cadena. "
            "Kill switch: GENESIS_DISABLE_PN95=1."
        ),
        patch_name="PN95 L2 occupancy metrics",
    )


def is_applied() -> bool:
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN95_MARKER in f.read()
    except Exception:
        return False
