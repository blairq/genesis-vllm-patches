# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN100 — anillo de staging por clase de agente.

================================================================
QUE RESUELVE
================================================================

El patron real de carga no es un hilo de 256k tokens: es un hilo principal de
60.000-100.000 que dispara ~6 subagentes coder/explorer en paralelo. Los
subagentes llenan L2 con material de un solo uso, desalojan el prefijo del
principal, y cuando terminan el cliente reenvia esos 100k y hay que
prefillearlos de nuevo. Ese pico es el que hay que ahorrarse.

Ninguna politica de reemplazo por recencia puede acertar aca: en el momento
del desalojo los bloques del subagente son los MAS recientes, asi que LRU y
ARC eligen exactamente al reves de lo que conviene. Ya lo medimos con
`store_threshold=2` (PN96), que no dio resistencia a escaneo sino que
desactivo L2.

Pero la informacion existe: PN90 ya clasifica cada request por
`genesis_agent`, y un agente efimero es justamente el que no merece cache.
PN100 reserva los ultimos K bloques de L2 como anillo y sirve de ahi los
bloques de agentes efimeros:

  - no salen del presupuesto del cache, asi que **nunca desalojan** al
    principal;
  - siguen visibles para `lookup()`, asi que el subagente acierta si relee su
    propio prefijo;
  - cuando el anillo se llena se recicla el mas viejo (FIFO), que es la
    semantica correcta para material de un solo uso.

================================================================
DONDE
================================================================

1. `cpu/manager.py`, `_free_block`: un id del anillo tiene que volver al
   anillo y no a la free list del cache. Sin esto el cache terminaria
   entregando ids reservados.

2. `cpu/manager.py`, dentro de `prepare_store`: en v0.27.1 el calculo de
   desalojos (`num_blocks_to_evict`) vive ahi — el tiering manager delega
   los stores del tier primario en `CPUOffloadingManager.prepare_store`.
   La persistencia del agente se evalua in situ con el mismo gate de PN90
   (`kv_disk_gate.should_persist_to_secondary_tiers`, via
   `req_context.kv_transfer_params`).

La reserva es perezosa, en el primer `prepare_store`, cuando
`_num_allocated_blocks` todavia es 0: se toman ids del tope del rango sin
asignar y se baja `_num_blocks`. Asi no hace falta tocar el `__init__`, que ya
tiene parches de PN96.

================================================================
QUE CUESTA DEJARLO PRENDIDO
================================================================

**VRAM: cero.** El anillo vive entero en el mmap de L2, que es RAM del host.
El path de GPU no se toca.

**RAM adicional: cero.** El anillo no aloca nada: PARTICIONA la region de L2
que ya estaba reservada. `cpu_bytes_to_use` no cambia.

**Cache: tampoco cuesta nada si no hay trafico efimero.** K es un TOPE, no una
reserva. La PRIMERA VERSION de este parche si reservaba: agarraba K ids y
achicaba `_num_blocks` de forma permanente, asi que con K=240 sobre 790 el
hilo principal perdia el 30% de L2 aunque los subagentes usaran 18 bloques.
Memoria inmovilizada para nada. Corregido el 2026-08-23.

Hoy los bloques efimeros salen del pool normal y solo se llevan la cuenta:
al pasarse de K se recicla el mas viejo DEL ANILLO. Con trafico efimero en
cero, el cache usa todo L2. `kv_tier_capacity_bytes{tier="ram"}` refleja L2
entera, como debe ser.

La regla para K es: (subagentes en paralelo) x (bloques que offloadea cada
uno). Con `max_offload_tokens` al tamano del prompt de sistema, cada subagente
offloadea ~3 bloques, asi que K=32 cubre 10 subagentes en paralelo.

**Limitacion de observabilidad conocida**: hoy solo se publica un contador
cuando el anillo RECICLA. Si nunca recicla no hay forma de distinguir "no hizo
falta" de "no se activo" — que es justo la ambiguedad que PN95 vino a eliminar
para L2. Hay que exponer servidos/en_uso antes de dejarlo prendido en serio.

================================================================
SIMPLIFICACION CONOCIDA
================================================================

Si CUALQUIER clave de la request ya se vio antes (preambulo compartido), la
request entera va al cache, cola incluida. Lo fino seria partirla en dos
asignaciones — preambulo al cache, cola al anillo — pero eso es dos
`_get_load_store_spec` y mas superficie. La version conservadora nunca manda
preambulo al anillo, que es el error que importa evitar.

================================================================
SEGURIDAD
================================================================

- **OFF por defecto**: `GENESIS_PN100_RING_BLOCKS=0`. Con 0 el modulo no
  reserva nada y el camino es identico al actual.
- Si el anillo no se puede reservar (cache ya en uso, K absurdo) se descarta
  para siempre en esa instancia y se sigue por el camino normal.
- Si el anillo se queda sin ids y sin nada que reciclar, cae al camino normal
  con desalojo, que es el comportamiento de hoy.
- Kill switch: `GENESIS_DISABLE_PN100=1`.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher

GENESIS_PN100_MARKER = "_GENESIS_PN100_STAGING_RING"

# ─────────── el unico hook: acotar lo efimero antes de que desaloje ───────────

RING_OLD = (
    "        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()\n"
)

RING_NEW = (
    "        # " + GENESIS_PN100_MARKER + "\n"
    "        # Un agente efimero (coder/explorer/...) no puede ocupar mas de K\n"
    "        # bloques de L2. Antes de calcular desalojos, el anillo recicla lo\n"
    "        # SUYO mas viejo para hacerse lugar; asi el excedente efimero nunca\n"
    "        # se cobra sobre el prefijo del hilo principal.\n"
    "        #\n"
    "        # K es un TOPE, no una reserva: no se inmoviliza memoria. Con\n"
    "        # trafico efimero en cero el cache se queda con todo L2.\n"
    "        #\n"
    "        # Mismo gate de PN90 (kv_disk_gate.should_persist_to_secondary_tiers)\n"
    "        # evaluado aca: cpu/manager.prepare_store no tiene ese nombre en\n"
    "        # scope, y la decision depende de kv_transfer_params de la request.\n"
    "        try:\n"
    "            from vllm._genesis import kv_disk_gate as _g100_gate\n"
    "\n"
    "            _g100_persist = _g100_gate.should_persist_to_secondary_tiers(\n"
    "                getattr(req_context, 'kv_transfer_params', None)\n"
    "            )\n"
    "        except Exception:\n"
    "            _g100_persist = True\n"
    "        if not _g100_persist:\n"
    "            from vllm._genesis import kv_staging_ring as _g100\n"
    "\n"
    "            _g100_lib = _g100.hacer_lugar(self, keys_to_store)\n"
    "            if _g100_lib:\n"
    "                try:\n"
    "                    from vllm._genesis import kv_tier_metrics as _g100_m\n"
    "\n"
    "                    _g100_m.note_eviction(\n"
    "                        'ram', 'anillo', count=_g100_lib, freed_bytes=0\n"
    "                    )\n"
    "                except Exception:\n"
    "                    pass\n"
    "        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN100", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _patcher() -> TextPatcher | None:
    root = vllm_install_root()
    if root is None:
        return None
    # v0.27.1: la decision de desalojo del tier primario vive en
    # cpu/manager.py (el tiering manager delega prepare_store ahi).
    mgr = os.path.join(root, "v1", "kv_offload", "cpu", "manager.py")
    if not os.path.exists(mgr):
        return None
    return TextPatcher(
        patch_name="PN100 staging ring (cpu manager)",
        target_file=mgr,
        marker=GENESIS_PN100_MARKER,
        sub_patches=[
            TextPatch(
                name="pn100_cap_ephemeral_footprint",
                anchor=RING_OLD,
                replacement=RING_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN100")
    log_decision("PN100", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN100 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "target de cpu/manager.py no encontrado"

    result, failure = p.apply()
    from vllm._genesis.wiring.text_patch import result_to_wiring_status

    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN100 aplicado: lo que offloadea un agente efimero queda acotado a "
            "GENESIS_PN100_RING_BLOCKS bloques de L2. Es un TOPE, no una "
            "reserva: no inmoviliza memoria, y con trafico efimero en cero el "
            "cache se queda con todo L2. Al pasarse, el anillo recicla lo suyo "
            "mas viejo en vez de desalojar el prefijo del hilo principal. "
            "Kill switch: GENESIS_DISABLE_PN100=1."
        ),
        patch_name="PN100 staging ring",
    )


def is_applied() -> bool:
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN100_MARKER in f.read()
    except Exception:
        return False
