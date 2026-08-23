# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN97 — una promoción L3→L2 nunca desaloja.

================================================================
QUÉ RESUELVE
================================================================

`TieringOffloadingManager._initiate_promotion()` mete un bloque que viene del
disco por el MISMO camino que una escritura nueva desde la GPU:

    primary_write_result = self.primary_tier.prepare_write([key], req_context)

y en el constructor del tier primario, `self.prepare_write = self.prepare_store`.

O sea que promover pide un slot de L2 y, si no hay, **desaloja**. Con PN90
activo el desalojo de un bloque taggeado lo baja al SSD. El resultado es un
bucle de realimentación:

    leer del SSD  ->  desalojar de L2  ->  escribir al SSD

Cada lectura de disco paga una escritura de disco, y encima destruye una
entrada residente que probablemente hacía falta. Medido el 2026-08-23 con
8 prompts de 25k tokens: 285 desalojos de L2 en una corrida cuyo objetivo era
LEER, y 0% de aciertos externos porque cada promoción se comía a la anterior.

PN97 hace que una promoción sólo pueda usar un slot **realmente libre**. Si no
hay, no promueve. Es la semántica de las cargas no-temporales de una CPU
(`movntdqa`) y del `primarycache=none` de ZFS: **el tráfico de streaming no
alloca**, así que un barrido no destruye el working set.

================================================================
POR QUÉ NO UN ANILLO DE STAGING
================================================================

La idea natural es un anillo reservado tipo line-fill-buffer/MSHR: promover a
un buffer chico fuera de la caché, hacer el DMA a la GPU y liberar. No sirve
acá: en una CPU la unidad de fill es una LÍNEA, pero acá el scheduler necesita
el PREFIJO ENTERO resuelto en L2 antes de emitir el load a la GPU. Un prompt
de 25k tokens son ~36 bloques (~1 GB con PN93 activo), y con max_model_len
262144 el peor caso son 315 bloques por grupo.

Un anillo más chico que un prefijo se llena de promociones en vuelo que nunca
completan — `complete_load()` no llega nunca porque el load no se emite — y no
se libera jamás. Es un deadlock, no una optimización. Reservar un anillo del
tamaño de un prefijo es reservar media L2, que es peor que el problema.

Por eso PN97 no reserva nada: no cambia la capacidad, sólo prohíbe que la
promoción desaloje.

================================================================
QUÉ CAMBIA EN EL COMPORTAMIENTO
================================================================

Cuando no hay slot libre, `_initiate_promotion` devuelve False y `lookup()`
devuelve False — exactamente lo que ya pasaba hoy en el camino "primary full,
block unavailable". O sea que **el peor caso de PN97 es el comportamiento
actual**, sin la escritura al SSD ni la destrucción del residente.

Lo que NO arregla: la relación entre el tamaño de L2 y el footprint de una
request. Con 222 bloques de L2 y ~36 por prompt entran ~6 prompts; con más
que eso el tier no puede servir a todos y no hay política de reemplazo que lo
salve. Eso es capacidad, no lógica.

Kill switch: `GENESIS_DISABLE_PN97=1`.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN97_MARKER = "_GENESIS_PN97_PROMOTION_NEVER_EVICTS"

# ─────────── 1. un prepare_write que se niega a desalojar ───────────

METHOD_OLD = "    def get_kv_memoryview(self) -> memoryview:\n"

METHOD_NEW = (
    "    # " + GENESIS_PN97_MARKER + "\n"
    "    def prepare_write_no_evict(self, keys, req_context):\n"
    "        \"\"\"prepare_write que NUNCA desaloja: si no hay slot libre, None.\n"
    "\n"
    "        Lo usa solo la promocion L3->L2. Sin esto, traer un bloque del\n"
    "        disco desaloja a un residente y —con PN90— lo escribe al SSD, o\n"
    "        sea que LEER del disco cuesta ESCRIBIR al disco.\n"
    "        \"\"\"\n"
    "        faltantes = [k for k in keys if self._policy.get(k) is None]\n"
    "        if faltantes and len(faltantes) > self._get_num_free_blocks():\n"
    "            try:\n"
    "                from vllm._genesis import kv_tier_metrics as _g97_m\n"
    "\n"
    "                _g97_m.note_promotion_refused('ram')\n"
    "            except Exception:\n"
    "                pass\n"
    "            return None\n"
    "        return self.prepare_write(keys, req_context)\n"
    "\n"
    "    def get_kv_memoryview(self) -> memoryview:\n"
)

# ─────────── 2. la promocion usa ese camino ───────────

CALL_OLD = (
    "        primary_write_result = self.primary_tier.prepare_write("
    "[key], req_context)\n"
)

CALL_NEW = (
    "        # " + GENESIS_PN97_MARKER + "\n"
    "        # prepare_write es un alias de prepare_store, asi que promover\n"
    "        # desalojaba igual que una escritura nueva desde la GPU. La\n"
    "        # variante no_evict devuelve None en vez de hacer lugar a la\n"
    "        # fuerza; el caller ya sabe tratar eso como 'bloque no disponible'.\n"
    "        primary_write_result = self.primary_tier.prepare_write_no_evict(\n"
    "            [key], req_context\n"
    "        )\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN97", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _patcher() -> TextPatcher | None:
    root = vllm_install_root()
    if root is None:
        return None
    target = os.path.join(root, "v1", "kv_offload", "tiering", "manager.py")
    if not os.path.exists(target):
        return None
    return TextPatcher(
        patch_name="PN97 promotion never evicts",
        target_file=target,
        marker=GENESIS_PN97_MARKER,
        sub_patches=[
            TextPatch(
                name="pn97_prepare_write_no_evict",
                anchor=METHOD_OLD,
                replacement=METHOD_NEW,
                required=True,
            ),
            TextPatch(
                name="pn97_promotion_uses_no_evict",
                anchor=CALL_OLD,
                replacement=CALL_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN97")
    log_decision("PN97", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN97 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "v1/kv_offload/tiering/manager.py not found"

    result, failure = p.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN97 aplicado: una promocion L3->L2 solo usa slots realmente "
            "libres. prepare_write es alias de prepare_store, asi que traer un "
            "bloque del disco desalojaba a un residente y —con PN90— lo escribia "
            "al SSD: leer del disco costaba escribir al disco. Ahora, sin slot "
            "libre no se promueve (mismo camino 'primary full' que ya existia) y "
            "se cuenta en kv_tier_promotion_refused_total{tier=\"ram\"}. "
            "Kill switch: GENESIS_DISABLE_PN97=1."
        ),
        patch_name="PN97 promotion never evicts",
    )


def is_applied() -> bool:
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN97_MARKER in f.read()
    except Exception:
        return False
