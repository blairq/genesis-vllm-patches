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

2. `tiering/manager.py`, dentro del `prepare_store` que instala PN90 — que es
   el unico que corre de verdad, y ademas el unico lugar donde ya se sabe si
   el agente es persistente (`should_persist`).

La reserva es perezosa, en el primer `prepare_store`, cuando
`_num_allocated_blocks` todavia es 0: se toman ids del tope del rango sin
asignar y se baja `_num_blocks`. Asi no hace falta tocar el `__init__`, que ya
tiene parches de PN96.

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
from vllm._genesis.wiring.text_patch import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
)

GENESIS_PN100_MARKER = "_GENESIS_PN100_STAGING_RING"

# ─────────── 1. un id del anillo vuelve al anillo ───────────

FREE_OLD = (
    "    def _free_block(self, block: BlockStatus) -> None:\n"
    "        self._free_list.append(block.block_id)\n"
)

FREE_NEW = (
    "    def _free_block(self, block: BlockStatus) -> None:\n"
    "        # " + GENESIS_PN100_MARKER + "\n"
    "        # Los ids del anillo NO son del cache: si cayeran en la free list,\n"
    "        # _allocate_blocks los entregaria como si fueran suyos.\n"
    "        _g100_a = getattr(self, '_g100_anillo', None)\n"
    "        if _g100_a is not None and _g100_a.contiene(block.block_id):\n"
    "            _g100_a.liberar(block.block_id)\n"
    "            return\n"
    "        self._free_list.append(block.block_id)\n"
)

# ─────────── 2. los agentes efimeros se sirven del anillo ───────────

RING_OLD = (
    "        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()\n"
)

RING_NEW = (
    "        # " + GENESIS_PN100_MARKER + "\n"
    "        # Un agente efimero (coder/explorer/...) no merece cache: sus\n"
    "        # bloques salen del anillo, asi no pueden desalojar el prefijo del\n"
    "        # hilo principal. `should_persist` ya lo decidio PN90 mas arriba.\n"
    "        if not should_persist:\n"
    "            from vllm._genesis import kv_staging_ring as _g100\n"
    "\n"
    "            _g100_a = _g100.asegurar(self)\n"
    "            if _g100_a is not None:\n"
    "                # El preambulo compartido de un subagente (system prompt,\n"
    "                # reglas, herramientas) es lo MAS valioso de L2: lo pega\n"
    "                # cada invocacion. Como la clave es hash de contenido, se\n"
    "                # reconoce porque ya aparecio antes. Ese va al cache; al\n"
    "                # anillo va solo la cola, que es de un solo uso.\n"
    "                _g100_cola = [\n"
    "                    _g100_k\n"
    "                    for _g100_k in keys_to_store\n"
    "                    if not _g100.es_compartido(_g100_k)\n"
    "                ]\n"
    "                if len(_g100_cola) != len(keys_to_store):\n"
    "                    _g100_a = None\n"
    "            if _g100_a is not None:\n"
    "                _g100_bloques = []\n"
    "                for _g100_k in keys_to_store:\n"
    "                    _g100_bid, _g100_rec = _g100_a.tomar(_g100_k)\n"
    "                    if _g100_bid is None:\n"
    "                        _g100_bloques = None\n"
    "                        break\n"
    "                    if _g100_rec is not None:\n"
    "                        self._policy.remove(_g100_rec)\n"
    "                    _g100_b = BlockStatus(_g100_bid)\n"
    "                    self._policy.insert(_g100_k, _g100_b)\n"
    "                    _g100_bloques.append(_g100_b)\n"
    "                if _g100_bloques is not None:\n"
    "                    try:\n"
    "                        from vllm._genesis import kv_tier_metrics as _g100_m\n"
    "\n"
    "                        _g100_m.note_eviction(\n"
    "                            'ram',\n"
    "                            'anillo',\n"
    "                            count=_g100_a.reciclados,\n"
    "                            freed_bytes=0,\n"
    "                        )\n"
    "                    except Exception:\n"
    "                        pass\n"
    "                    return PrepareStoreOutput(\n"
    "                        keys_to_store=keys_to_store,\n"
    "                        store_spec=self._get_load_store_spec(\n"
    "                            keys_to_store, _g100_bloques\n"
    "                        ),\n"
    "                        evicted_keys=[],\n"
    "                    )\n"
    "        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN100", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _patchers() -> list[TextPatcher] | None:
    root = vllm_install_root()
    if root is None:
        return None
    cpu = os.path.join(root, "v1", "kv_offload", "cpu", "manager.py")
    tie = os.path.join(root, "v1", "kv_offload", "tiering", "manager.py")
    if not (os.path.exists(cpu) and os.path.exists(tie)):
        return None
    return [
        TextPatcher(
            patch_name="PN100 staging ring (cpu manager)",
            target_file=cpu,
            marker=GENESIS_PN100_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn100_free_block_routes_to_ring",
                    anchor=FREE_OLD,
                    replacement=FREE_NEW,
                    required=True,
                ),
            ],
        ),
        TextPatcher(
            patch_name="PN100 staging ring (tiering manager)",
            target_file=tie,
            marker=GENESIS_PN100_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn100_ephemeral_agents_use_ring",
                    anchor=RING_OLD,
                    replacement=RING_NEW,
                    required=True,
                ),
            ],
        ),
    ]


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
    patchers = _patchers()
    if patchers is None:
        return "skipped", "targets de cpu/manager.py o tiering/manager.py no hallados"

    txn = MultiFilePatchTransaction(patchers, name="PN100")
    status, reason = txn.apply_or_skip()
    if status != "applied":
        return status, reason
    return "applied", (
        "PN100 aplicado: los bloques de agentes efimeros se sirven de un anillo "
        "reservado al final de L2, asi que no pueden desalojar el prefijo del "
        "hilo principal. Se dimensiona con GENESIS_PN100_RING_BLOCKS (0 = "
        "apagado). Kill switch: GENESIS_DISABLE_PN100=1."
    )


def is_applied() -> bool:
    patchers = _patchers()
    if patchers is None:
        return False
    for p in patchers:
        try:
            with open(p.target_file) as f:
                if GENESIS_PN100_MARKER not in f.read():
                    return False
        except Exception:
            return False
    return True
