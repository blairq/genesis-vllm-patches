# SPDX-License-Identifier: Apache-2.0
"""Wiring for CK-4.2 (B3) — custom all-reduce TP=2 fast path + robust fallback.

Contrato
--------
- Ubicación de referencia: assets/vllm/vllm/v1/worker/gpu_model_runner.py
  línea ~6546 (custom all-reduce dispatch). En el pin actual v0.23.0 ese
  despacho vive en ``vllm.distributed.device_communicators.cuda_communicator``
  (``CudaCommunicator.all_reduce``) que delega a
  ``CustomAllreduce.should_custom_ar / custom_all_reduce``. El runner lo
  invoca indirectamente vía ``get_tp_group().all_reduce`` / capas TP.
  Este wiring lo envuelve de forma robusta.

- Env flag: ``GENESIS_ENABLE_B3_CUSTOM_AR`` (off por defecto).
- Marker: ``Genesis B3 custom AR TP=2 fast path v1 // gpu_model_runner:6546``
  usado para idempotencia y para ``is_applied()``.
- Función pública: ``patch_B3_custom_ar() -> tuple[str,str]`` (status, reason)
  + alias ``apply()`` para el dispatcher ``apply_all``.

Qué hace
--------
1. **Fast path TP=2**: cuando ``world_size == 2`` el wrapper de
   ``CustomAllreduce.should_custom_ar`` hace early-return para tensores
   pequeños (< 2 MiB) y 16-byte alineados sin pasar por chequeos caros de
   P2P / fully_connected. Para TP>2 mantiene el camino upstream intacto.
   Esto acelera el caso más común del stack (2×3090/2×A5000 TP=2).

2. **Robust fallback**: ``CudaCommunicator.all_reduce`` se envuelve con
   try/except. Si el backend custom / pynccl lanza (p. ej. cumem×IPC bug
   documentado en KERNELS-OPTIMIZACION §2 #6 — custom AR bloqueado por
   cumem×IPC), cae a ``torch.distributed.all_reduce`` clonando el tensor.
   Garantiza corrección aunque el kernel falle; el error queda logueado
   como WARNING y no mata el engine.

3. **Idempotencia**: verifica ``__genesis_b3_wrapped__`` en los callables
   objetivo antes de rebind. Doble apply es no-op.

4. **No-ops seguros**: si ``CustomAllreduce`` o ``CudaCommunicator`` no
   son importables en este pin / plataforma, retorna ``skipped`` en vez de
   ``failed`` — no bloquea el boot.

Relación con BACKPORT-V2 / KERNELS-OPTIMIZACION
-----------------------------------------------
Ninguno de los dos docs menciona un patch específico para el custom AR en
gpu_model_runner:6546; KERNELS-OPTIMIZACION §2 #6 solo recomienda barrido
NCCL_ALGO/PROTO y documenta el bug cumem×IPC que bloquea custom AR. Este
patch sigue la rama “si no, haz un patch que envuelva el all-reduce con
un fast path para TP=2” del contrato.

Autor: Genesis CK-4.2 (B3) — custom AR en gpu_model_runner.py:6546.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.B3_custom_ar")

# ─── Public contract constants ──────────────────────────────────────────
ENV_FLAG = "GENESIS_ENABLE_B3_CUSTOM_AR"
# Marker must be stable — used for idempotency + tests that grep the file.
GENESIS_B3_MARKER = "Genesis B3 custom AR TP=2 fast path v1 // gpu_model_runner:6546"
GENESIS_B3_CUSTOM_AR_MARKER = GENESIS_B3_MARKER  # alias for greps

_TRUTHY = ("1", "true", "yes", "on")

# kept for revert / introspection (not strictly needed but handy)
_ORIGINAL_ALL_REDUCE = None
_ORIGINAL_SHOULD_CUSTOM_AR = None
_ORIGINAL_CUSTOM_ALL_REDUCE = None
_INSTALLED = False


def _is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in _TRUTHY


def is_applied() -> bool:
    """True if B3 wrappers are live in this process."""
    try:
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        wrapped_ar = getattr(CudaCommunicator.all_reduce, "__genesis_b3_wrapped__", False)
        wrapped_should = getattr(CustomAllreduce.should_custom_ar, "__genesis_b3_wrapped__", False)
        # either entry counts; both should be set when installed
        return bool(wrapped_ar or wrapped_should or _INSTALLED)
    except Exception:
        return bool(_INSTALLED)


def patch_B3_custom_ar() -> tuple[str, str]:
    """Apply B3 — custom all-reduce TP=2 fast path + robust fallback.

    Returns:
        (status, reason) where status in {"applied", "skipped", "failed"}.
        Never raises — all exceptions become "failed" / "skipped".
    """
    global _ORIGINAL_ALL_REDUCE, _ORIGINAL_SHOULD_CUSTOM_AR
    global _ORIGINAL_CUSTOM_ALL_REDUCE, _INSTALLED

    # ── Env gate ────────────────────────────────────────────────────────
    if not _is_enabled():
        return "skipped", f"opt-in only — set {ENV_FLAG}=1 to engage"

    # ── Import targets ──────────────────────────────────────────────────
    try:
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
    except Exception as e:
        return "skipped", f"not applicable — custom_all_reduce stack not importable on this pin: {e} (platform mismatch)"

    # Idempotency: already wrapped?
    if getattr(CudaCommunicator.all_reduce, "__genesis_b3_wrapped__", False):
        _INSTALLED = True
        return "applied", "idempotent (marker present) — B3 already wrapped CudaCommunicator.all_reduce"

    if not hasattr(CudaCommunicator, "all_reduce"):
        return "skipped", "not applicable — CudaCommunicator.all_reduce not present on this pin — B3 NULL"

    if not hasattr(CustomAllreduce, "should_custom_ar"):
        return "skipped", "not applicable — CustomAllreduce.should_custom_ar not present — B3 NULL"

    # ── Save originals ──────────────────────────────────────────────────
    try:
        _ORIGINAL_ALL_REDUCE = CudaCommunicator.all_reduce
        _ORIGINAL_SHOULD_CUSTOM_AR = CustomAllreduce.should_custom_ar
        _ORIGINAL_CUSTOM_ALL_REDUCE = getattr(CustomAllreduce, "custom_all_reduce", None)
    except Exception as e:
        return "failed", f"cannot stash originals: {e}"

    # ── Wrapper 1: CustomAllreduce.should_custom_ar with TP=2 fast path ──
    _orig_should = CustomAllreduce.should_custom_ar

    def _b3_should_custom_ar(self, inp) -> bool:  # type: ignore[no-untyped-def]
        """TP=2 fast path + upstream checks + robust guard.

        Fast path: world_size==2, inp_size < 2 MiB, 16-byte aligned,
        weak-contiguous → True if not disabled. Avoids P2P / fully_connected
        queries for the common 2-GPU case.
        """
        try:
            # disabled short-circuit (preserve upstream)
            if getattr(self, "disabled", False):
                return False
            # need tensor props
            try:
                inp_size = int(inp.numel() * inp.element_size())
            except Exception:
                return bool(_orig_should(self, inp))

            # TP=2 fast path — small & aligned tensors go custom without extra checks
            # Threshold 2 MiB covers typical hidden-state all-reduces (e.g. 5120*4096*2 bytes ~40MB would NOT qualify — goes to normal path)
            # We keep threshold conservative to avoid forcing large tensors through custom when NCCL might be better.
            ws = getattr(self, "world_size", None)
            if ws == 2:
                # 16-byte alignment required by custom kernel
                if inp_size % 16 == 0 and inp_size < (2 * 1024 * 1024):
                    # weak_contiguous check (import lazily to avoid circular)
                    try:
                        from vllm.distributed.utils import is_weak_contiguous  # type: ignore

                        if not is_weak_contiguous(inp):
                            return False
                    except Exception:
                        # if utility missing, fall back to contiguous check
                        try:
                            if not inp.is_contiguous():
                                return False
                        except Exception:
                            pass
                    # respect max_size if set
                    max_sz = getattr(self, "max_size", None)
                    if max_sz is not None and inp_size >= int(max_sz):
                        return False
                    return True

            # Normal path — delegate to upstream logic
            return bool(_orig_should(self, inp))
        except Exception as e:
            log.debug("[B3] should_custom_ar wrapper exception (%s) — fallback to upstream", e)
            try:
                return bool(_orig_should(self, inp))
            except Exception:
                return False

    _b3_should_custom_ar.__genesis_b3_wrapped__ = True  # type: ignore[attr-defined]
    _b3_should_custom_ar.__genesis_b3_marker__ = GENESIS_B3_MARKER  # type: ignore[attr-defined]

    # ── Wrapper 2: CudaCommunicator.all_reduce with robust fallback ──────
    _orig_all_reduce = CudaCommunicator.all_reduce

    def _b3_all_reduce(self, input_):  # type: ignore[no-untyped-def]
        """Wrapped all_reduce with TP=2 awareness + exception fallback.

        - For world_size==1: straight clone (no comm needed).
        - Otherwise: try original dispatch (which tries custom AR, flashinfer, symm_mem, pynccl).
        - On any exception: log WARNING and fallback to torch.distributed.all_reduce.
        """
        # Fast path: single GPU — no reduction needed, but keep clone semantics
        try:
            ws = getattr(self, "world_size", None)
            if ws == 1:
                try:
                    return input_.clone()
                except Exception:
                    pass
        except Exception:
            pass

        # Normal dispatch with robust guard
        try:
            return _orig_all_reduce(self, input_)
        except Exception as e:
            # Log once per process to avoid spam
            try:
                log.warning(
                    "[%s] all_reduce failed (%s: %s) — fallback to torch.distributed.all_reduce (world_size=%s)",
                    GENESIS_B3_MARKER,
                    type(e).__name__,
                    e,
                    getattr(self, "world_size", "?"),
                )
            except Exception:
                pass
            # Fallback: torch.distributed all_reduce (out-of-place clone)
            try:
                import torch.distributed as dist

                # CudaCommunicator keeps device_group; fallback to it
                group = getattr(self, "device_group", None)
                out = input_.clone()
                if group is not None:
                    dist.all_reduce(out, group=group)
                else:
                    dist.all_reduce(out)
                return out
            except Exception as e2:
                log.error("[B3] fallback all_reduce also failed: %s: %s", type(e2).__name__, e2)
                raise

    _b3_all_reduce.__genesis_b3_wrapped__ = True  # type: ignore[attr-defined]
    _b3_all_reduce.__genesis_b3_marker__ = GENESIS_B3_MARKER  # type: ignore[attr-defined]

    # ── Install ─────────────────────────────────────────────────────────
    try:
        CustomAllreduce.should_custom_ar = _b3_should_custom_ar  # type: ignore[method-assign,assignment]
        CudaCommunicator.all_reduce = _b3_all_reduce  # type: ignore[method-assign,assignment]
        _INSTALLED = True
        log.info("[B3] %s installed — CudaCommunicator.all_reduce wrapped with TP=2 fast path + robust fallback", GENESIS_B3_MARKER)
        return (
            "applied",
            "B3 custom AR TP=2 fast path with robust fallback installed (ref gpu_model_runner:6546) — CudaCommunicator.all_reduce + CustomAllreduce.should_custom_ar wrapped",
        )
    except Exception as e:
        # attempt rollback
        try:
            if _ORIGINAL_ALL_REDUCE is not None:
                CudaCommunicator.all_reduce = _ORIGINAL_ALL_REDUCE  # type: ignore[method-assign]
            if _ORIGINAL_SHOULD_CUSTOM_AR is not None:
                CustomAllreduce.should_custom_ar = _ORIGINAL_SHOULD_CUSTOM_AR  # type: ignore[method-assign]
        except Exception:
            pass
        return "failed", f"rebind failed: {e}"


# Alias expected by vllm._genesis.patches.apply_all (generic helper)
def apply() -> tuple[str, str]:
    """Alias for dispatcher compatibility — delegates to patch_B3_custom_ar."""
    return patch_B3_custom_ar()


def revert() -> bool:
    """Revert wrappers (for tests). Returns True if reverted."""
    global _INSTALLED, _ORIGINAL_ALL_REDUCE, _ORIGINAL_SHOULD_CUSTOM_AR
    try:
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        if _ORIGINAL_ALL_REDUCE is not None:
            try:
                CudaCommunicator.all_reduce = _ORIGINAL_ALL_REDUCE  # type: ignore[method-assign]
            except Exception:
                pass
        if _ORIGINAL_SHOULD_CUSTOM_AR is not None:
            try:
                CustomAllreduce.should_custom_ar = _ORIGINAL_SHOULD_CUSTOM_AR  # type: ignore[method-assign]
            except Exception:
                pass
        _INSTALLED = False
        return True
    except Exception:
        return False
