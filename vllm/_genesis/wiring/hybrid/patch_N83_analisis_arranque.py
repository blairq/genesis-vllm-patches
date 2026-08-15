# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N83 — informe de arranque en lenguaje humano.

================================================================
QUÉ RESUELVE
================================================================

El engine loguea todo lo que hace falta para entender un arranque, pero
repartido en ~700 líneas, en inglés, en unidades distintas y en dos procesos
distintos. Para responder "¿cuánto contexto me entra?" o "¿por qué murió?"
hay que reconstruirlo a mano cada vez.

Peor: hay riesgos que **no se loguean en ningún lado** y que ya nos costaron
horas:

- El crash intermitente por `cudaHostRegister` (PN82, §3.3). El síntoma
  apunta a la VRAM y no es la VRAM.
- Los 394 MiB que FlashInfer aloca en el **primer request** (§3.4). El
  engine arranca perfecto y muere con el primer prompt.
- El tier de disco sin cuota si PN81 está apagado (§4).

PN83 emite, al final del arranque, un informe que contesta de una lectura:

    A DONDE SE FUE LA VRAM     (desglose con barras, por GPU)
    QUE ENTRA EN ESA CACHE     (traducido al patrón de uso real:
                                1 hilo largo + N agentes)
    CACHE EN DOS CAPAS         (tiers, política, cuánto contexto extra)
    SPECULATIVE DECODING       (método, K, cómo medir la aceptación)
    RIESGOS                    (los tres de arriba, con veredicto ✔/✖)
    COMO CONSEGUIR MAS KV      (flags medidos, con su costo real)

================================================================
CÓMO
================================================================

Dos sub-patches, porque los números viven en procesos distintos:

1. `v1/worker/gpu_worker.py` — el WORKER es quien perfila la memoria
   (pesos, non-torch, pico de activaciones, estimación de CUDA graphs).
   Se engancha después del `logger.info_once("Available KV cache memory")`
   y deja el desglose en un JSON. Solo rank 0.

2. `v1/core/kv_cache_utils.py` — el ENGINE CORE es quien calcula el total
   de tokens de KV. Se engancha después del
   `logger.info_once("GPU KV cache size")`, junta las dos mitades y escribe
   el informe.

La lógica vive en `vllm/_genesis/analisis_arranque.py`, no en el parche, así
que se puede re-ejecutar sin reiniciar nada:

    docker exec <contenedor> python3 -m vllm._genesis.analisis_arranque

================================================================
COSTO Y SEGURIDAD
================================================================

- Default **ON**. Kill switch: `GENESIS_DISABLE_PN83=1`.
- Observación pura: no toca ninguna asignación ni resultado numérico. Solo
  lee config y escribe texto.
- Corre **una vez** por arranque, no en el loop de inferencia.
- Todo el cuerpo va en try/except: un fallo del informe no puede tumbar el
  arranque. Ante cualquier duda se calla.
- Escribe a stderr y no por el logger, porque el logger prefija cada línea
  y rompe el formato del informe.
- Tunables: `GENESIS_ANALISIS_HILO_PRINCIPAL` (default 220000) y
  `GENESIS_ANALISIS_AGENTE` (default 40000) ajustan el patrón de uso con el
  que se traduce la capacidad de la cache.

Ver docs/KV-OFFLOADING.md.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN83_MARKER = "_GENESIS_PN83_ANALISIS_ARRANQUE"


# ── 1. worker: guardar el desglose de memoria ────────────────────────────

ANCHOR_WORKER_OLD = '''        logger.info_once(
            "Available KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
        )
'''

ANCHOR_WORKER_NEW = '''        logger.info_once(
            "Available KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
        )
        # _GENESIS_PN83_ANALISIS_ARRANQUE
        # El desglose de memoria solo existe aca, en el WORKER. El total de
        # tokens de KV se calcula en el ENGINE CORE, otro proceso. Guardamos
        # esta mitad para que el informe pueda juntar las dos.
        try:
            from vllm._genesis import analisis_arranque as _g83

            _g83.guardar_memoria(
                rank=int(getattr(self, "rank", 0) or 0),
                total_memory=int(self.init_snapshot.total_memory),
                free_memory=int(self.init_snapshot.free_memory),
                requested_memory=int(self.requested_memory),
                weights_memory=int(profile_result.weights_memory),
                non_kv_cache_memory=int(profile_result.non_kv_cache_memory),
                torch_peak_increase=int(profile_result.torch_peak_increase),
                non_torch_increase=int(profile_result.non_torch_increase),
                cudagraph_memory_estimate=int(cudagraph_memory_estimate),
                available_kv_cache_memory_bytes=int(
                    self.available_kv_cache_memory_bytes
                ),
                gpu_memory_utilization=float(
                    self.cache_config.gpu_memory_utilization
                ),
            )
        except Exception:
            pass  # el informe es opcional; nunca puede tumbar el arranque
'''


# ── 2. engine core: emitir el informe ────────────────────────────────────

ANCHOR_CORE_OLD = '''    logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}")
'''

ANCHOR_CORE_NEW = '''    logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}")
    # _GENESIS_PN83_ANALISIS_ARRANQUE
    # Ultimo punto del arranque donde ya se sabe TODO: config completa mas el
    # tamaño real de la cache en tokens. Aca se junta con el desglose de
    # memoria que dejo el worker y se escribe el informe para humanos.
    try:
        from vllm._genesis import analisis_arranque as _g83

        _g83.emitir_informe(vllm_config, num_tokens)
    except Exception:
        pass  # el informe es opcional; nunca puede tumbar el arranque
'''


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN83", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _patcher_worker() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_worker.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN83 analisis arranque (worker)",
        target_file=str(target),
        marker=GENESIS_PN83_MARKER,
        sub_patches=[
            TextPatch(
                name="pn83_stash_memory_snapshot",
                anchor=ANCHOR_WORKER_OLD,
                replacement=ANCHOR_WORKER_NEW,
                required=True,
            ),
        ],
    )


def _patcher_core() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/kv_cache_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN83 analisis arranque",
        target_file=str(target),
        marker=GENESIS_PN83_MARKER,
        sub_patches=[
            TextPatch(
                name="pn83_emit_report",
                anchor=ANCHOR_CORE_OLD,
                replacement=ANCHOR_CORE_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN83")
    log_decision("PN83", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN83 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # el desglose de memoria es un extra: si su ancla derivo, el informe se
    # emite igual, solo que sin la seccion de VRAM
    wp = _patcher_worker()
    if wp is not None:
        try:
            wp.apply()
        except Exception:
            pass

    p = _patcher_core()
    if p is None:
        return "skipped", "v1/core/kv_cache_utils.py not found"
    result, failure = p.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN83 applied: al final del arranque se emite un informe en "
            "castellano con el desglose de VRAM, cuanto contexto entra "
            "traducido al patron de uso real (1 hilo largo + N agentes), el "
            "estado de la cache en dos capas, MTP, los riesgos conocidos "
            "(PN82, workspace lazy de FlashInfer, cuota del tier de disco) y "
            "que flags darian mas KV con su costo. Re-ejecutable con "
            "python3 -m vllm._genesis.analisis_arranque. Tunables: "
            "GENESIS_ANALISIS_HILO_PRINCIPAL / _AGENTE. "
            "Kill switch: GENESIS_DISABLE_PN83=1."
        ),
        patch_name="PN83 analisis arranque",
    )
