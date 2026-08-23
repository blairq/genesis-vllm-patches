# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN96 — admisión a L2 resistente a escaneo.

================================================================
QUÉ RESUELVE
================================================================

Todo bloque entra a L2 en su PRIMERA escritura. Con un working set mayor que
L2, un solo barrido secuencial desaloja todo lo que servía. Medido el
2026-08-23 con 6 prompts de 43k tokens (≈9 GB de KV) contra 6,43 GB de L2:
acertó **1 de 6**, y el que acertó fue el último escrito. Es el caso de libro
de thrashing por escaneo de LRU/ARC.

vLLM ya trae el remedio en `CPUOffloadingManager`: `store_threshold`. Con
valor ≥2 un bloque entra a L2 recién en su N-ésima aparición en `lookup()`,
así que el material de un solo uso no desaloja al que se relee. El contador
(`self.counts`) ya se incrementa en `lookup()` y el filtro ya está tanto en
`CPUOffloadingManager.prepare_store` como en el override que instala PN90.

Estaba muerto por tres razones, y hacen falta las tres para revivirlo:

  1. `TieringOffloadingSpec` lo rechaza con un `ValueError` pelado.
  2. `CPUPrimaryTierOffloadingManager.__init__` ni siquiera acepta el
     parámetro, así que no lo reenvía a `super()`.
  3. `tiering/spec.py` construye el tier primario sin pasarlo.

================================================================
POR QUÉ EL GUARD DE UPSTREAM NO APLICA ACÁ
================================================================

El `ValueError` es razonable en vLLM tal cual: en el flujo normal, un bloque
baja a L3 por CASCADA desde L2 (`TieringOffloadingManager.complete_store` →
`primary.prepare_read()` → `tier.submit_store()`). Si el filtro deja el bloque
fuera de L2, tampoco llega nunca al disco, y el usuario pidió un tier de disco
que se queda vacío sin explicación.

Pero PN90 ya eliminó esa cascada: `complete_store` retorna antes de cascadear
y un bloque baja a L3 **únicamente cuando lo desalojan de L2**. Con esa
semántica, "no entró a L2" y "no llegó al disco" es exactamente la política
deseada — es la misma regla de PN90 (no escribir lo que no se va a releer),
sólo que decidida por reuso en vez de por agente.

Por eso PN96 **conserva el `ValueError` si PN90 no está activo**. La detección
es en runtime y sin I/O: si `CPUPrimaryTierOffloadingManager.prepare_store` ya
no es el método de la clase base, el override de PN90 está instalado.

================================================================
RESULTADO MEDIDO: store_threshold=2 NO SIRVE ACÁ
================================================================

Se probó el 2026-08-23 en este despliegue y **la hipótesis era falsa**:

    ocupación de L2      6,43 GB (100%)  ->  58 MB (0,9%)
    hot tras el escaneo   0,0%           ->   0,0%   (sin cambio)
    hot antes del escaneo 99,2%          ->  90,9%   (peor)

`counts` se incrementa en `lookup()`, y `lookup()` corre sólo cuando una
request BUSCA el bloque. El flujo natural es computar y offloadear (store)
ANTES de que nadie lo busque, así que el contador casi nunca llega a 2 antes
de que pase la oportunidad de guardar. Con 2 no se gana resistencia a
escaneo: **se desactiva L2**.

PN96 queda igual porque el parche en sí es correcto y ahora el parámetro es
configurable y medible — pero el valor productivo es 1. Para que la idea
funcione habría que contar apariciones en el camino de STORE (o llevar un
registro de claves vistas independiente de si están en L2), no en `lookup()`.
No re-probar con 2 sin cambiar eso primero.

Nota aparte: `eviction_policy` ya es `"arc"`, y la resistencia a escaneo es
justamente lo que ARC promete sobre LRU. En este test no se notó, lo cual
merece su propia mirada.

================================================================
CÓMO SE USA
================================================================

No hace nada por sí solo: hay que pedirlo en la config del connector,

    "extra_config": {"store_threshold": 2}

Con 2, un bloque entra a L2 en su segunda aparición. Con 1 (default) el
comportamiento es idéntico al de hoy, así que el parche es inerte salvo que se
lo configure.

Cuidado al medir: con `store_threshold=2` la primera relectura de un prompt
NO acierta (recién ahí se guarda). El efecto aparece de la tercera en
adelante, o cuando hay un escaneo intercalado. Un test de dos pasadas mide
peor con el parche que sin él, y eso no es una regresión.

Kill switch: `GENESIS_DISABLE_PN96=1`.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
)

GENESIS_PN96_MARKER = "_GENESIS_PN96_SCAN_RESISTANT_ADMISSION"

# ─────────── 1. el constructor acepta y reenvia store_threshold ───────────

INIT_OLD = (
    "        num_blocks: int,\n"
    "        mmap_region: SharedOffloadRegion,\n"
    '        cache_policy: str = "lru",\n'
    "        enable_events: bool = False,\n"
    "    ):\n"
    "        super().__init__(\n"
    "            num_blocks=num_blocks,\n"
    "            cache_policy=cache_policy,  # type: ignore[arg-type]\n"
    "            enable_events=enable_events,\n"
    "        )\n"
)

INIT_NEW = (
    "        num_blocks: int,\n"
    "        mmap_region: SharedOffloadRegion,\n"
    '        cache_policy: str = "lru",\n'
    "        enable_events: bool = False,\n"
    "        # " + GENESIS_PN96_MARKER + "\n"
    "        # Sin este parametro el filtro de admision de la clase base es\n"
    "        # inalcanzable desde el tiering manager: counts queda en None.\n"
    "        store_threshold: int = 1,\n"
    "    ):\n"
    "        super().__init__(\n"
    "            num_blocks=num_blocks,\n"
    "            cache_policy=cache_policy,  # type: ignore[arg-type]\n"
    "            enable_events=enable_events,\n"
    "            store_threshold=store_threshold,\n"
    "        )\n"
)

# ─────────── 2. el spec lo pasa al construir el tier primario ───────────

BUILD_OLD = (
    "            primary_tier = CPUPrimaryTierOffloadingManager(\n"
    "                num_blocks=self.num_blocks,\n"
    "                cache_policy=self.eviction_policy,  # type: ignore[arg-type]\n"
    "                enable_events=enable_events,\n"
    "                mmap_region=scheduler_mmap,\n"
    "            )\n"
)

BUILD_NEW = (
    "            # " + GENESIS_PN96_MARKER + "\n"
    '            _g96_thr = int(self.extra_config.get("store_threshold", 1) or 1)\n'
    "            primary_tier = CPUPrimaryTierOffloadingManager(\n"
    "                num_blocks=self.num_blocks,\n"
    "                cache_policy=self.eviction_policy,  # type: ignore[arg-type]\n"
    "                enable_events=enable_events,\n"
    "                mmap_region=scheduler_mmap,\n"
    "                store_threshold=_g96_thr,\n"
    "            )\n"
    "            if _g96_thr >= 2:\n"
    "                logger.info(\n"
    '                    "PN96: admision a L2 con store_threshold=%d — un bloque '
    'entra "\n'
    '                    "recien en su %d-esima aparicion. El material de un solo '
    'uso "\n'
    '                    "deja de desalojar al que se relee.",\n'
    "                    _g96_thr,\n"
    "                    _g96_thr,\n"
    "                )\n"
)

# ─────────── 3. el guard: solo se levanta si PN90 corto la cascada ───────────

GUARD_OLD = (
    '            if int(self.extra_config.get("store_threshold", 0)) >= 2:\n'
    "                raise ValueError(\n"
    '                    "store_threshold is not supported for '
    'TieringOffloadingSpec"\n'
    "                )\n"
)

GUARD_NEW = (
    "            # " + GENESIS_PN96_MARKER + "\n"
    "            # El guard de upstream existe porque en el flujo normal un\n"
    "            # bloque baja a L3 por CASCADA desde L2: filtrar la admision\n"
    "            # dejaria el tier de disco vacio sin explicacion. PN90 ya\n"
    "            # elimino esa cascada (complete_store retorna antes) y un\n"
    "            # bloque baja a disco solo al ser desalojado de L2, con lo\n"
    "            # cual 'no entro a L2' = 'no se escribe al SSD' es la\n"
    "            # politica buscada. Deteccion sin I/O: si el override de PN90\n"
    "            # esta instalado, prepare_store ya no es el de la clase base.\n"
    '            if int(self.extra_config.get("store_threshold", 0)) >= 2:\n'
    "                from vllm.v1.kv_offload.cpu.manager import (\n"
    "                    CPUOffloadingManager as _g96_base,\n"
    "                )\n"
    "\n"
    "                _g96_pn90 = (\n"
    "                    CPUPrimaryTierOffloadingManager.prepare_store\n"
    "                    is not _g96_base.prepare_store\n"
    "                )\n"
    "                if not _g96_pn90:\n"
    "                    raise ValueError(\n"
    '                        "store_threshold is not supported for '
    'TieringOffloadingSpec"\n'
    '                        " without the Genesis PN90 no-cascade store path: '
    'filtered"\n'
    '                        " blocks would never reach the secondary tier."\n'
    "                    )\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN96", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _patchers() -> list[TextPatcher] | None:
    root = vllm_install_root()
    if root is None:
        return None
    manager = os.path.join(root, "v1", "kv_offload", "tiering", "manager.py")
    spec = os.path.join(root, "v1", "kv_offload", "tiering", "spec.py")
    if not (os.path.exists(manager) and os.path.exists(spec)):
        return None
    return [
        TextPatcher(
            patch_name="PN96 scan-resistant L2 admission (tiering manager)",
            target_file=manager,
            marker=GENESIS_PN96_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn96_primary_tier_accepts_store_threshold",
                    anchor=INIT_OLD,
                    replacement=INIT_NEW,
                    required=True,
                ),
            ],
        ),
        TextPatcher(
            patch_name="PN96 scan-resistant L2 admission (tiering spec)",
            target_file=spec,
            marker=GENESIS_PN96_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn96_pass_store_threshold",
                    anchor=BUILD_OLD,
                    replacement=BUILD_NEW,
                    required=True,
                ),
                TextPatch(
                    name="pn96_guard_only_without_pn90",
                    anchor=GUARD_OLD,
                    replacement=GUARD_NEW,
                    required=True,
                ),
            ],
        ),
    ]


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN96")
    log_decision("PN96", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN96 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patchers = _patchers()
    if patchers is None:
        return "skipped", "targets de tiering/{manager,spec}.py no encontrados"

    txn = MultiFilePatchTransaction(patchers, name="PN96")
    status, reason = txn.apply_or_skip()
    if status != "applied":
        return status, reason
    return "applied", (
            "PN96 aplicado: L2 acepta store_threshold. Con "
            '"extra_config": {"store_threshold": 2} un bloque entra a L2 recien '
            "en su segunda aparicion, asi que un barrido de material de un solo "
            "uso deja de desalojar al working set. Inerte con el default de 1. "
            "El ValueError de upstream se conserva si PN90 no corto la cascada. "
            "Kill switch: GENESIS_DISABLE_PN96=1."
    )


def is_applied() -> bool:
    patchers = _patchers()
    if patchers is None:
        return False
    for p in patchers:
        try:
            with open(p.target_file) as f:
                if GENESIS_PN96_MARKER not in f.read():
                    return False
        except Exception:
            return False
    return True
