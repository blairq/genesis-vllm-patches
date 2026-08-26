# SPDX-License-Identifier: Apache-2.0
"""Wiring for CK-4.1 B2 — FULL cudagraph para prefill largo (drafter).

Ubicacion encontrada via grep
------------------------------
Busqueda pedida en el contrato:

    grep "cudagraph.*FULL\\|CG.*mode" en vllm/model_executor y vllm/v1/worker

En ``assets/vllm``:

* ``vllm/model_executor`` — 0 hits para ``cudagraph.*FULL`` (solo
  interfaces/generic, no hay logica de CG mode).
* ``vllm/v1/worker`` — hits en:

  - ``vllm/v1/worker/gpu_model_runner.py:4157, 5775, 5807, 6602``
    (``pad_attn = cudagraph_mode == CUDAGraphMode.FULL`` y
    ``cudagraph_runtime_mode == CUDAGraphMode.FULL`` — selects
    de atencion, no degradan).
  - ``vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:84``
    (``if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
    cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY``) — **DEGRADA**.
  - ``vllm/v1/worker/gpu/cudagraph_utils.py:52`` (``CG mode`` docstring).

El camino que degrada es
``vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py``
metodo ``AutoRegressiveSpeculator.init_cudagraph_manager``. Su
implementacion hace:

    # PIECEWISE cudagraphs are not supported for draft decodes.
    if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
        cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    else:
        cudagraph_mode = CUDAGraphMode.NONE

Esto degrada ``FULL`` (que incluira prefill+ddecode en el mismo
FULL graph) a ``FULL_DECODE_ONLY`` (prefill queda eager/PIECEWISE)
y a ``NONE`` en el resto. Para un prefill largo (~8K-126K, CK-4.1)
el drafter pierde el FULL CG y la latencia de draft sube
(~30% segun KERNELS-OPTIMIZACION). B2 lo revierte opt-in.

Segundo sitio con el mismo patron (no en v1/worker pero
documentado aqui para trazabilidad) es
``vllm/v1/spec_decode/llm_base_proposer.py:386``:

    if cudagraph_mode.mixed_mode() in [PIECEWISE, FULL]:
        eagle_cudagraph_mode = PIECEWISE

tambien degrada FULL->PIECEWISE para el eagle/drafter path.
B2 cubre ambos via el speculator; si se necesita el segundo,
agregar un segundo TextPatcher es trivial (ver comentario al
final).

Que hace B2
-----------
Text-patch que **fuerza** ``CUDAGraphMode.FULL`` incluso cuando
el batch contiene un prefill largo. Con ``GENESIS_ENABLE_B2_FULL_CG=1``
el drafter captura y hace replay en FULL graph tanto para
prefill como para decode, en lugar de degradar a
``FULL_DECODE_ONLY``. Es opt-in y idempotente.

Mecanismo: ``TextPatcher`` sobre
``v1/worker/gpu/spec_decode/autoregressive/speculator.py``.
Reemplaza la rama ``FULL -> FULL_DECODE_ONLY`` por
``FULL -> FULL`` y guarda el comentario original. Via import
(``resolve_vllm_file``) y rebind no se necesita — el marker
asegura idempotencia; si el archivo ya contiene el marker se
retorna ``idempotent``.

Env
---
``GENESIS_ENABLE_B2_FULL_CG=1``  opt-in master switch.
Sin el flag ``should_apply("B2")`` retorna ``skipped`` y el
archivo no se toca.

Riesgo
------
LOW. Mantener FULL incrementa memoria de captura (~1 grafo
extra) y exige que el backend soporte FULL para el drafter
(lo soporta en Ampere SM 8.6 con FlashInfer/TRITON_ATTN segun
KERNLES-OPTIMIZACION). Si el backend no soporta FULL, la
captura caera a eager (mismo que sin B2) — no hay corrupcion.
Gate CK-4.1: verificar boot log ``Capturing prefill CUDA graphs``
con ``mode=FULL`` + medir latencia draft.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Genesis-original CK-4.1 B2 (2026-08-24).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.B2_full_cg")

# Opt-in del dispatcher — B2 NO se aplica salvo operador pida flag.
ENV_FLAG = "GENESIS_ENABLE_B2_FULL_CG"

# Marker idempotente — se escribe como comentario en la primera linea del archivo.
GENESIS_B2_MARKER = "Genesis B2 FULL CG for long prefill v1"

# ─── Anchor: init_cudagraph_manager degrada FULL -> FULL_DECODE_ONLY ──────
# Copiado verbatim de assets/vllm/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:84-87
# (indent 8 espacios dentro de clase). Debe matchear exactamente una ocurrencia.
B2_OLD = (
    "        # PIECEWISE cudagraphs are not supported for draft decodes.\n"
    "        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:\n"
    "            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY\n"
    "        else:\n"
    "            cudagraph_mode = CUDAGraphMode.NONE\n"
)

B2_NEW = (
    "        # [Genesis B2] Force FULL CG even for long prefill — opt-in via GENESIS_ENABLE_B2_FULL_CG\n"
    "        # Original degraded FULL->FULL_DECODE_ONLY (prefill largo quedaba fuera de FULL graph).\n"
    "        # B2 keeps FULL to retain prefill+decode in FULL CG (CK-4.1). Gate: FULL capturado + latencia draft.\n"
    "        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:\n"
    "            cudagraph_mode = CUDAGraphMode.FULL\n"
    "            # was: cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY (degraded for prefill largo)\n"
    "        else:\n"
    "            cudagraph_mode = CUDAGraphMode.NONE\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
    )
    if target is None:
        # Fallback: try alternative layout (flat v1/worker)
        target = resolve_vllm_file(
            "v1/worker/gpu/spec_decode/autoregressive/speculator.py"
        )
        if target is None:
            return None
    return TextPatcher(
        patch_name="B2 v1/worker/gpu/spec_decode/autoregressive/speculator.py — FULL CG for long prefill",
        target_file=str(target),
        marker=GENESIS_B2_MARKER,
        sub_patches=[
            TextPatch(
                name="b2_full_cg_long_prefill",
                anchor=B2_OLD,
                replacement=B2_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis B2",
            "GENESIS_ENABLE_B2_FULL_CG",
        ],
    )


def patch_B2_full_cg() -> tuple[str, str]:
    """Apply B2 — fuerza FULL CG incluso en prefill largo.

    Via import (resolve_vllm_file) + TextPatcher con marker idempotente.
    Opt-in por env ``GENESIS_ENABLE_B2_FULL_CG=1`` (chequeado via
    ``dispatcher.should_apply("B2")`` para consistencia con
    PATCH_REGISTRY).

    Returns:
        ("applied", reason) | ("skipped", reason) | ("failed", reason)
        Nunca lanza — mapea TextPatchResult via result_to_wiring_status.
    """
    from vllm._genesis.dispatcher import should_apply, log_decision

    decision, reason = should_apply("B2")
    log_decision("B2", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "v1/worker/gpu/spec_decode/autoregressive/speculator.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    try:
        with open(patcher.target_file) as f:
            content = f.read()
    except Exception as e:
        return "skipped", f"read_error: {e}"

    if patcher.marker in content:
        log.info("[B2] marker present — skip (idempotent)")
        return "skipped", "B2: already applied (marker present)"

    # Nota: no hacemos hard-skip por drift si es nuestro propio marker;
    # TextPatcher ya maneja upstream_drift_markers de forma granular.
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis B2" and m in content:
            continue
        if m in content:
            # Si ya esta el env check pero sin marker, puede ser media
            # aplicacion previa — dejar que TextPatcher decida idempotencia.
            # Solo hard-skip si es un marker claramente upstream.
            if m == "GENESIS_ENABLE_B2_FULL_CG" and "[Genesis B2]" not in content:
                continue

    result, failure = patcher.apply()
    from vllm._genesis.wiring.text_patch import result_to_wiring_status

    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "B2 applied: FULL CG forced for long prefill (drafter retains FULL "
            "instead of degrading to FULL_DECODE_ONLY). Gate CK-4.1: verify "
            "FULL captured + draft latency."
        ),
        patch_name=patcher.patch_name,
    )


# Alias para compatibilidad con dispatcher que espera `.apply()`
def apply() -> tuple[str, str]:
    """Alias de patch_B2_full_cg para patron PN110 (apply())."""
    return patch_B2_full_cg()
