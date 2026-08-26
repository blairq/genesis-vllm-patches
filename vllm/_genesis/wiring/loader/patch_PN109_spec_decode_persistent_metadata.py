# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN109 — buffers persistentes (pinned-CPU + GPU) para la metadata
de spec-decode.

Genesis-original 2026-08-24.

El problema
-----------
En cada paso de spec-decode, `GPUModelRunner._calc_spec_decode_metadata`
(`vllm/v1/worker/gpu_model_runner.py:2851-2942` en v0.27.1) calcula 5 arrays numpy en CPU
y, al final, hace **5 copias H2D independientes** vía `async_tensor_h2d(...)`. Cada
llamada re-crea un tensor pinned desde el array numpy (`pin_memory()` por copia,
5 alojamientos nuevos por paso) y luego `to(device, non_blocking=True)`. El
movimiento es asíncrono, pero se re-alocan 5 tensores pinned por paso y no hay
reutilización entre pasos.

La solución
-----------
Separar la MATEMÁTICA (que queda idéntica a upstream, ver
`spec_decode_metadata_numpy`) del MOVIMIENTO de datos. `PersistentMetadataBuffers`
conserva, por clave, una buffer GPU persistente y una staging CPU **pinned**:
cada paso se escribe el array numpy en la staging y se hace un único
`copy_(non_blocking=True)` real (pinned → GPU sí es asíncrono), devolviendo una
**vista** `[:n]` del buffer persistente. Los buffers crecen x2 cuando no
alcancan y se reutilizan entre pasos: cero re-alocaciones en estado estacionario
y 5 H2D de pageable → 5 H2D de pinned sobre memoria ya residente.

El rebind
---------
`patch_PN109_spec_decode_persistent_metadata()` rebinda
`GPUModelRunner._calc_spec_decode_metadata` preservando la firma
`(self, num_draft_tokens, cu_num_scheduled_tokens) -> SpecDecodeMetadata`. El
wrapper: (1) llama `spec_decode_metadata_numpy` con `self._get_cumsum_and_arange`
y `self._arange_scratch` (misma matemática y dtypes que upstream), (2) sube los
5 arrays por la cache persistente, (3) recalcula `draft_token_ids` igual que
upstream (:2913-2914 en v0.27.1 con `self.input_ids.gpu`) y (4) devuelve el
`SpecDecodeMetadata` con `num_draft_tokens.tolist()` intacto.

Restricción de validez intra-paso
---------------------------------
Las vistas devueltas por `upload` apuntan a buffers persistentes que se
reescriben en el paso siguiente. Son válidas **dentro del paso** en el que se
generan (el caller las consume antes del próximo `_calc_spec_decode_metadata`),
que es exactamente el mismo contrato de vida que tenía el tensor pageable de
upstream. No se deben retener entre pasos.

Seguridad
---------
- Default **ON**. Kill switch: `GENESIS_DISABLE_PN109=1` (patrón inverse
  `GENESIS_DISABLE_<X>=1` que usa el resto del stack, ver PN100/PN87).
- Idempotente: marker de clase `_genesis_pn109_installed`; re-aplicar es no-op.
- Nunca lanza hacia el runner: cualquier excepción en el wrapper cae al
  `_calc_spec_decode_metadata` original (el comportamiento de hoy).
- No modifica `assets/vllm`: rebind de método en memoria, sin text-patch.

Author: ox-alpha 2026-08-24.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

log = logging.getLogger("genesis.wiring.pn109_spec_decode_persistent_metadata")

# Kill switch (inverse pattern): default ON, `GENESIS_DISABLE_PN109=1` apaga.
_DISABLE_ENV = "GENESIS_DISABLE_PN109"
_TRUTHY = ("1", "true", "yes", "on")

# Opt-in del dispatcher (misma convención que PN106/PN108): el parche NO se
# aplica salvo que el operador pida explícitamente el flag. Corregido el
# 2026-08-24: la primera versión solo chequeaba el kill switch y PN109
# terminó activo en PROD sin promoción (gate de plan violado).
ENV_FLAG = "GENESIS_ENABLE_PN109_SPEC_DECODE_PERSISTENT_METADATA"

_MARKER_ATTR = "_genesis_pn109_installed"

# Capacidad inicial de cada buffer (tokens). Crecimiento x2 a partir de aquí.
_INITIAL_CAPACITY = 1024

# Claves fijas que produce spec_decode_metadata_numpy (orden de subida).
_METADATA_KEYS = (
    "cu_num_draft_tokens",
    "cu_num_sampled_tokens",
    "logits_indices",
    "target_logits_indices",
    "bonus_logits_indices",
)

_original = None
_installed_class = None


def _is_disabled() -> bool:
    """Devuelve True si el kill switch GENESIS_DISABLE_PN109 está activo."""
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in _TRUTHY


def spec_decode_metadata_numpy(
    num_draft_tokens: np.ndarray,
    cu_num_scheduled_tokens: np.ndarray,
    get_cumsum_and_arange,
    arange_scratch: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calcula en CPU los 5 arrays de metadata de spec-decode.

    Replica EXACTO las líneas 2868-2897 de
    `GPUModelRunner._calc_spec_decode_metadata` (vllm 0.23.0; verificado
    idéntico en v0.27.1): misma
    matemática y mismos dtypes que upstream. Aquí NO hay ninguna copia a GPU;
    eso lo hace `PersistentMetadataBuffers.upload`.

    Alinea la aritmética con upstream para que los dtypes queden idénticos:
    `num_draft_tokens` llega como int32, por lo que `num_sampled_tokens` y
    `logits_indices`/`target_logits_indices` (np.repeat de int32) quedan int32;
    el `+=` in-place contra `arange_scratch` (int64) conserva el dtype del
    operando izquierdo (int32). `cu_*` y `bonus` son int32 por
    `cumsum_dtype=np.int32`.

    :param num_draft_tokens: [num_reqs] drafts por request (int32 en el caller).
    :param cu_num_scheduled_tokens: [num_reqs] cumsum de tokens agendados
        (int32, proviene de `_get_cumsum_and_arange`).
    :param get_cumsum_and_arange: `GPUModelRunner._get_cumsum_and_arange`
        (bound method), usado para no duplicar la lógica de cumsum+arange.
    :param arange_scratch: `self._arange_scratch` (int64), mismo scratch que
        upstream para que las escrituras de arange sean idénticas.
    :returns: dict con las claves `cu_num_draft_tokens`,
        `cu_num_sampled_tokens`, `logits_indices`, `target_logits_indices` y
        `bonus_logits_indices`, cada una un `np.ndarray` (dtypes de upstream).
    """
    # Compute the logits indices.
    # [4, 1, 3, 1, 2]
    num_sampled_tokens = num_draft_tokens + 1

    # Step 1.
    # cu_num_sampled_tokens: [4, 5, 8, 9, 11]
    # _arange_scratch[:11]: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
    cu_num_sampled_tokens = get_cumsum_and_arange(
        num_sampled_tokens, arange_scratch, cumsum_dtype=np.int32
    )
    # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
    logits_indices = np.repeat(
        cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
    )
    # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
    logits_indices += arange_scratch[: cu_num_sampled_tokens[-1]]

    # Compute the bonus logits indices.
    bonus_logits_indices = cu_num_sampled_tokens - 1

    # Compute the draft logits indices.
    # cu_num_draft_tokens: [3, 3, 5, 5, 6]
    # _arange_scratch[:6]: [0, 1, 2, 0, 1, 0]
    cu_num_draft_tokens = get_cumsum_and_arange(
        num_draft_tokens, arange_scratch, cumsum_dtype=np.int32
    )
    # [0, 0, 0, 5, 5, 9]
    target_logits_indices = np.repeat(
        cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
    )
    # [0, 1, 2, 5, 6, 9]
    target_logits_indices += arange_scratch[: cu_num_draft_tokens[-1]]

    return {
        "cu_num_draft_tokens": cu_num_draft_tokens,
        "cu_num_sampled_tokens": cu_num_sampled_tokens,
        "logits_indices": logits_indices,
        "target_logits_indices": target_logits_indices,
        "bonus_logits_indices": bonus_logits_indices,
    }


class PersistentMetadataBuffers:
    """Cache de buffers persistentes (GPU) + staging (CPU pinned) por clave.

    Sustituye las 5 copias H2D `torch.from_numpy(...).to(device,
    non_blocking=True)` de pageable de upstream por copias pinned→GPU sobre
    buffers que se reutilizan entre pasos. Cada clave (los 5 arrays de
    metadata) tiene su propia pareja buffer-GPU/staging-CPU.
    """

    def __init__(self, device: torch.device):
        """Inicializa la cache vacía para el dispositivo dado.

        :param device: dispositivo destino de las buffers GPU persistentes
            (el `self.device` del runner).
        """
        self._device = device
        # key -> torch.Tensor persistente en GPU (nunca se re-aloca salvo crecer).
        self._gpu: dict[str, torch.Tensor] = {}
        # key -> np.ndarray pinned CPU (staging para el copy non_blocking).
        self._cpu: dict[str, np.ndarray] = {}

    def upload(self, key: str, arr: np.ndarray) -> torch.Tensor:
        """Sube `arr` a la buffer GPU persistente de `key` y devuelve una vista.

        Flujo: escribir `arr` en la staging CPU pinned de `key` (creciéndola x2
        si no alcanza), luego `copy_(non_blocking=True)` a la buffer GPU
        persistente (creciéndola x2 si no alcanza) y devolver la vista `[:n]`.
        El copy pinned→GPU es asíncrono de verdad, a diferencia del pageable de
        upstream, y la memoria ya está residente entre pasos.

        La vista devuelta es válida **intra-paso**: apunta a memoria que se
        reescribe en la siguiente llamada a `upload` de la misma clave.

        :param key: nombre de la metadata (una de `_METADATA_KEYS`).
        :param arr: array numpy 1-D a subir (puede no ser contiguo; se copia).
        :returns: `torch.Tensor` 1-D en GPU, vista `[:n]` del buffer
            persistente, con el dtype de `arr`.
        """
        n = arr.shape[0]
        torch_dtype = torch.from_numpy(np.ascontiguousarray(arr)).dtype

        # Staging CPU pinned: crecer x2 si no alcanza (o desde la capacidad base).
        cpu = self._cpu.get(key)
        if cpu is None or cpu.shape[0] < n:
            cap = max(n, _INITIAL_CAPACITY)
            if cpu is not None:
                cap = max(cap, cpu.shape[0] * 2)
            cpu = np.empty(cap, dtype=arr.dtype)
            cpu = np.ascontiguousarray(cpu)
            try:
                torch.from_numpy(cpu).pin_memory()
            except Exception:
                pass  # sin pinned sigue funcionando (copy síncrono, no asíncrono)
            self._cpu[key] = cpu
        cpu[:n] = arr

        # Buffer GPU persistente: crecer x2 si no alcanza.
        gpu = self._gpu.get(key)
        if gpu is None or gpu.shape[0] < n:
            cap = max(n, _INITIAL_CAPACITY)
            if gpu is not None:
                cap = max(cap, gpu.shape[0] * 2)
            gpu = torch.empty(
                cap, dtype=torch_dtype, device=self._device
            )
            self._gpu[key] = gpu
        # copy_ desde staging pinned → GPU, asíncrono; no bloquea el host.
        gpu[:n].copy_(torch.from_numpy(cpu[:n]), non_blocking=True)

        return gpu[:n]


def _make_wrapper(original):
    """Construye el rebind de `_calc_spec_decode_metadata` sobre `original`."""

    def _calc_spec_decode_metadata(self, num_draft_tokens, cu_num_scheduled_tokens):
        # Cache persistente lazily guardada en self (una por runner).
        buffers = getattr(self, "_genesis_pn109_buffers", None)
        if buffers is None:
            buffers = PersistentMetadataBuffers(self.device)
            self._genesis_pn109_buffers = buffers

        # Misma matemática y dtypes que upstream (:2868-2897 en v0.27.1).
        meta = spec_decode_metadata_numpy(
            num_draft_tokens,
            cu_num_scheduled_tokens,
            self._get_cumsum_and_arange,
            self._arange_scratch,
        )

        # 5 H2D sobre buffers persistentes (pinned) en vez de 5 pageable.
        gpu = {key: buffers.upload(key, meta[key]) for key in _METADATA_KEYS}

        # Compute the draft token ids (idéntico a upstream :2913-2914).
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[gpu["logits_indices"]]
        draft_token_ids = draft_token_ids[gpu["target_logits_indices"] + 1]

        from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=gpu["cu_num_draft_tokens"],
            cu_num_sampled_tokens=gpu["cu_num_sampled_tokens"],
            target_logits_indices=gpu["target_logits_indices"],
            bonus_logits_indices=gpu["bonus_logits_indices"],
            logits_indices=gpu["logits_indices"],
        )

    return _calc_spec_decode_metadata


def install(runner_class) -> bool:
    """Rebind de `_calc_spec_decode_metadata` sobre la CLASE del runner.

    :param runner_class: clase `GPUModelRunner` a parchear.
    :returns: True si quedó instalado (o ya lo estaba), False si no aplicó.
    """
    global _original, _installed_class
    if getattr(runner_class, _MARKER_ATTR, False):
        return True
    if not hasattr(runner_class, "_calc_spec_decode_metadata"):
        return False
    original = runner_class._calc_spec_decode_metadata
    runner_class._calc_spec_decode_metadata = _make_wrapper(original)
    setattr(runner_class, _MARKER_ATTR, True)
    _original = original
    _installed_class = runner_class
    log.info("PN109 instalado sobre %s", runner_class.__name__)
    return True


def revert() -> bool:
    """Restaura el método original (para tests y apagado limpio)."""
    global _original, _installed_class
    if _installed_class is None or _original is None:
        return False
    _installed_class._calc_spec_decode_metadata = _original
    setattr(_installed_class, _MARKER_ATTR, False)
    _installed_class = None
    _original = None
    return True


def apply() -> tuple[str, str]:
    """Punto de entrada del orquestador. Nunca lanza.

    :returns: `(status, mensaje)` — status `"applied"` | `"skipped"` |
        `"failed"`, convención del orquestador (igual que PN106/PN108 y el
        resto del stack; `apply_all.py` compara contra esas cadenas).
    """
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN109 set"
    if os.environ.get(ENV_FLAG, "").strip().lower() not in _TRUTHY:
        return "skipped", (
            "opt-in only — set "
            "GENESIS_ENABLE_PN109_SPEC_DECODE_PERSISTENT_METADATA=1")
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:
        return "failed", f"import GPUModelRunner: {e}"
    try:
        if install(GPUModelRunner):
            return "applied", (
                "rebind de _calc_spec_decode_metadata con buffers persistentes "
                "pinned (5 H2D pageable → 5 H2D pinned reutilizados). "
                "Kill switch: GENESIS_DISABLE_PN109=1."
            )
        return "skipped", "ya estaba instalado"
    except Exception as e:
        return "failed", f"install: {e}"
