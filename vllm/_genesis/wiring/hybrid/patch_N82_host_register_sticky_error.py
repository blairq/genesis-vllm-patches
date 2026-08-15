# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N82 — limpia el error pegajoso de un cudaHostRegister fallido.

================================================================
QUÉ RESUELVE
================================================================

Con `OffloadingConnector` + TP>1, el engine muere en el arranque, durante la
captura de CUDA graphs, con un error que no tiene nada que ver con el lugar
donde aparece:

    File "vllm/v1/worker/gpu_model_runner.py", line 5775, in _dummy_run
        sm.fill_(-1)
    torch.AcceleratorError: CUDA error: invalid argument

`sm.fill_(-1)` es válido: `sm` es un slot mapping recién asignado y `fill_`
es una op trivial. El error viene de ANTES. El propio mensaje de CUDA lo
avisa: *"CUDA kernel errors might be asynchronously reported at some other
API call, so the stacktrace below might be incorrect."*

La causa real está dos segundos antes, en `pin_mmap_region()`:

    WARNING gpu_worker.py:141 cudaHostRegister failed for rank=1 (code=1)
            — transfers will still work but may be slower (unpinned DMA)

`code=1` es `cudaErrorInvalidValue`. vLLM chequea el valor de retorno,
loguea el warning y sigue **sin limpiar el estado de error del contexto de
CUDA**. En el runtime de CUDA el último error queda latcheado hasta que
alguien lo consume con `cudaGetLastError()`. PyTorch consulta ese estado en
cada operación, así que la PRIMERA op del rank afectado hereda el error y
explota — sea cual sea.

Correlación medida (w8a16-mtp, TP=2, 2× RTX 3090):

    20:13:49  TP1  Created mmap file /dev/shm/vllm_offload_...  (5.35 GB)
    20:13:49  TP0  Opened existing mmap file (el mismo)
    20:13:51  TP1  cudaHostRegister failed for rank=1 (code=1)
    20:13:53  TP1  sm.fill_(-1) -> CUDA error: invalid argument

Muere **el mismo rank** que falló el registro, y solo ese. TP0 registró bien
y nunca falló.

================================================================
POR QUÉ FALLA EL REGISTRO (el bug de abajo)
================================================================

Los dos ranks mapean el MISMO archivo de `/dev/shm` y los dos llaman
`cudaHostRegister` sobre él. Son las mismas páginas físicas: el que registra
segundo falla. O sea que con TP>1 el fallo no es una rareza, es el caso
normal — lo raro es que a veces el error latcheado lo consuma una llamada
interna que lo tolera en vez de una op de PyTorch, y ahí el arranque
"funciona".

Eso explica la no-determinación que hace imposible tunear el engine: la
misma configuración arranca o no arranca según el timing entre ranks, y el
síntoma (OOM-ish en captura de graphs) apunta a la VRAM, que no tiene nada
que ver. Bajar `--gpu-memory-utilization` no ayuda porque el problema no es
de memoria de GPU.

================================================================
CÓMO
================================================================

Una línea: después de un `cudaHostRegister` fallido, consumir el error con
`cudaGetLastError()` para que no contamine la próxima op.

No cambia la decisión de vLLM —seguir sin pinnear, con DMA más lento— solo
hace que esa decisión sea la que realmente pasa, en vez de un crash tres
llamadas después.

================================================================
COSTO Y SEGURIDAD
================================================================

- Default **ON**: no es una optimización, es un crash de arranque. Kill
  switch con `GENESIS_DISABLE_PN82=1`.
- No-op cuando el registro funciona: el código nuevo vive dentro del
  `if result.value != 0`.
- No enmascara nada: el warning de vLLM se sigue logueando igual, y PN82
  agrega el suyo diciendo qué error limpió.
- `upstream_drift_markers`: si upstream agrega su propio `cudaGetLastError`
  o `synchronize` en esa rama, PN82 se saltea limpio.

Ver docs/KV-OFFLOADING.md §3.4.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN82_MARKER = "_GENESIS_PN82_HOST_REGISTER_STICKY_V2"


ANCHOR_OLD = '''    result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
    if result.value != 0:
        logger.warning(
            "cudaHostRegister failed for rank=%d (code=%d) — "
            "transfers will still work but may be slower (unpinned DMA)",
            rank,
            result,
        )
'''

ANCHOR_NEW = '''    result = torch.cuda.cudart().cudaHostRegister(base_ptr, region.total_size_bytes, 0)
    if result.value != 0:
        logger.warning(
            "cudaHostRegister failed for rank=%d (code=%d) — "
            "transfers will still work but may be slower (unpinned DMA)",
            rank,
            result,
        )
        # _GENESIS_PN82_HOST_REGISTER_STICKY_V2
        # El runtime de CUDA deja el ultimo error LATCHEADO en el contexto
        # hasta que alguien lo consume. Chequear el valor de retorno no lo
        # limpia. PyTorch consulta ese estado en cada op, asi que la primera
        # op de este rank hereda el error y explota lejos de aca -- en la
        # practica, en la captura de CUDA graphs:
        #     gpu_model_runner._dummy_run -> sm.fill_(-1)
        #     torch.AcceleratorError: CUDA error: invalid argument
        # Con TP>1 esto es el caso NORMAL, no una rareza: los dos ranks
        # mapean el mismo archivo de /dev/shm y los dos lo registran, asi
        # que el segundo siempre falla sobre las mismas paginas fisicas.
        # Consumimos el error para que la decision de vLLM (seguir sin
        # pinnear) sea la que realmente pasa.
        #
        # OJO: torch.cuda.cudart() NO expone cudaGetLastError (verificado en
        # torch 2.11.0+cu130: dir(cudart) solo trae cudaError y
        # cudaGetErrorString). Hay que llamarlo por ctypes sobre libcudart.
        try:
            import ctypes as _g82_ct
            import ctypes.util as _g82_ctu
            import glob as _g82_glob

            _g82_lib = None
            _g82_cands = sorted(
                _g82_glob.glob(
                    "/usr/local/lib/python3*/dist-packages/nvidia/**/libcudart.so*",
                    recursive=True,
                )
            )
            _g82_found = _g82_ctu.find_library("cudart")
            if _g82_found:
                _g82_cands.append(_g82_found)
            _g82_cands += ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
            for _g82_c in _g82_cands:
                try:
                    _g82_probe = _g82_ct.CDLL(_g82_c)
                    _g82_probe.cudaGetLastError.restype = _g82_ct.c_int
                    _g82_lib = _g82_probe
                    break
                except Exception:
                    continue

            if _g82_lib is None:
                logger.error(
                    "PN82: no encontre libcudart para limpiar el error de CUDA. "
                    "El proximo kernel de rank=%d va a morir con 'invalid "
                    "argument' durante la captura de CUDA graphs.",
                    rank,
                )
            else:
                _g82_code = _g82_lib.cudaGetLastError()
                logger.warning(
                    "PN82: error pegajoso de CUDA limpiado tras el "
                    "cudaHostRegister fallido de rank=%d (era code=%d). Sin "
                    "esto el proximo kernel de este rank muere con 'invalid "
                    "argument' durante la captura de CUDA graphs.",
                    rank,
                    _g82_code,
                )
        except Exception as _g82_exc:  # nunca puede tumbar el arranque
            logger.error("PN82: no se pudo limpiar el error de CUDA: %s", _g82_exc)
'''


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN82", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/kv_offload/cpu/gpu_worker.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN82 host register sticky error",
        target_file=str(target),
        marker=GENESIS_PN82_MARKER,
        sub_patches=[
            TextPatch(
                name="pn82_clear_sticky_cuda_error",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Si upstream arregla esto por su cuenta, PN82 sobra -> SKIP limpio.
            "cudaGetLastError",
            "cuda_get_last_error",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN82")
    log_decision("PN82", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN82 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "v1/kv_offload/cpu/gpu_worker.py not found"
    result, failure = p.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN82 applied: tras un cudaHostRegister fallido se consume el error "
            "con cudaGetLastError (por ctypes sobre libcudart: torch.cuda.cudart() "
            "no lo expone). Sin esto el error queda latcheado en el contexto y la "
            "primera op del rank afectado muere con 'CUDA error: invalid argument' "
            "en la captura de CUDA graphs (_dummy_run -> sm.fill_(-1)). Con TP>1 "
            "el registro fallido es el caso normal: ambos ranks pinean el mismo "
            "mmap de /dev/shm. Kill switch: GENESIS_DISABLE_PN82=1."
        ),
        patch_name="PN82 host register sticky error",
    )
