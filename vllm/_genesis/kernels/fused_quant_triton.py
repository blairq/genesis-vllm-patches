# SPDX-License-Identifier: Apache-2.0
"""Kernel fused Triton para quant_activation_per_token — per-token INT8.

Hace en un solo pass (sin intermedios):

    bf16/fp16/fp32 -> amax por fila (reduccion tl.max) -> scale = amax/127
                 -> x_i8 = round(x/scale) clamp [-127,127] -> int8

Contrato:
    - Lee bf16 una vez (tl.load) y escribe int8 + scale, sin buffers fp32
      intermedios extra entre DRAM passes.
    - Un programa por fila (token). BLOCK cubre toda la fila (next_pow2).
    - Usa tl.arange, tl.load, tl.max, tl.where obligatoriamente.
    - Fallback 100% GPU (torch) sin Triton ni copias a host.

Soporta [..., K] (cualquier leading dims) via flatten a [M, K]. Si Triton
no esta o N > BLOCK_MAX, usa fallback torch vectorizado en GPU.

Author: Genesis (fused quant per-token, 2026-08-25)
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("genesis.kernels.fused_quant_triton")

_TRITON_OK = False
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _TRITON_OK = True
except Exception as _e:  # pragma: no cover
    log.warning("fused_quant_triton: Triton no disponible (%s) — solo fallback torch", _e)
    triton = None  # type: ignore
    tl = None  # type: ignore

# BLOCK max soportado en single-load (potencia de 2). 8192 cubre hidden
# tipicos (4096, 5120, 8192). Modelos con intermediate 17408 caen a fallback
# torch que sigue siendo 100% GPU.
BLOCK_MAX = 8192


def is_triton_available() -> bool:
    """True si Triton esta importable."""
    return _TRITON_OK


def is_available() -> bool:
    """True si kernel Triton esta disponible (Triton + CUDA)."""
    if not _TRITON_OK:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# ── Triton kernel — single pass fused ──────────────────────────────────
if _TRITON_OK:

    @triton.jit
    def _fused_quant_per_token_kernel(
        x_ptr,
        out_ptr,
        scale_ptr,
        N,
        stride_xm,
        stride_xn,
        stride_out_m,
        stride_out_n,
        stride_scale,
        BLOCK: tl.constexpr,
    ):
        """Kernel fused per-token: bf16 -> amax (tl.max) -> scale -> int8.

        Un programa = una fila (token).

        Flujo single-pass:
          1. offs = tl.arange(0, BLOCK)
          2. x = tl.load(x_ptr + pid*stride_xm + offs*stride_xn)  # unico load bf16
          3. amax = tl.max(tl.abs(x_f32), axis=0)                 # reduccion fila
          4. scale = tl.where(amax/127 > 0, amax/127, 1.0)
          5. q = round(x / scale) clamp [-127,127] via tl.where
          6. tl.store out int8 + scale

        Sin buffers intermedios DRAM: x vive solo en registros.
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N

        # 1) Unico load bf16/fp16/fp32 por elemento
        x = tl.load(x_ptr + pid * stride_xm + offs * stride_xn, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        # 2) amax por fila via reduccion tl.max
        # tl.abs + tl.max(axis=0) -> escalar por programa
        amax = tl.max(tl.abs(x_f32), axis=0)

        # 3) scale = amax / 127, con tl.where para fila toda-ceros
        scale = amax / 127.0
        scale = tl.where(scale > 0, scale, 1.0)

        # 4) Guardar escala per-token (un escalar por fila)
        tl.store(scale_ptr + pid * stride_scale, scale)

        # 5) Quant: round(x / scale) clamp [-127,127]
        # round via bias 0.5 + cast trunc (Triton cast trunca hacia 0)
        # tl.where elige bias segun signo para round half-away-from-zero
        q_scaled = x_f32 / scale
        bias = tl.where(q_scaled >= 0, 0.5, -0.5)
        q_int = (q_scaled + bias).to(tl.int32)
        # clamp con tl.where (dos veces)
        q_int = tl.where(q_int > 127, 127, q_int)
        q_int = tl.where(q_int < -127, -127, q_int)

        # 6) Store int8 (unico store por elemento)
        q_i8 = q_int.to(tl.int8)
        tl.store(out_ptr + pid * stride_out_m + offs * stride_out_n, q_i8, mask=mask)


# ── Fallback 100% GPU (sin host copies) ────────────────────────────────
def _quant_fallback_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback vectorizado 100% GPU: quant per-token sin Triton.

    Replica exacto: amax/127, scale 1.0 si fila cero, round+clamp.
    Soporta [..., K] (cualquier leading dims) y preserva device/dims.
    Sin copias a host.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    if x.dim() < 1:
        raise ValueError(f"quant_activation_per_token: esperado [...,K] con dim>=1, got {x.dim()}D")
    k = int(orig_shape[-1])
    m = x.numel() // k if k else 0
    x_2d = x.reshape(m, k)
    x_f32 = x_2d.to(torch.float32)
    amax = x_f32.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_i8_2d = (x_f32 / scale).round().clamp(-127, 127).to(torch.int8)
    # restaurar leading dims
    x_i8 = x_i8_2d.reshape(orig_shape)
    scale_out = scale.reshape(orig_shape[:-1] + (1,)).contiguous()
    # mantener orig_dtype no mutado (solo lectura)
    assert x.dtype == orig_dtype
    return x_i8, scale_out


# ── API publica ────────────────────────────────────────────────────────
def quant_activation_per_token(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cuantiza activacion per-token a INT8 — fused Triton single-pass.

    Hace bf16/fp16/fp32 -> amax fila (tl.max) -> scale=amax/127 ->
    x_i8=round(x/scale) clamp [-127,127] -> int8. Lee bf16 una vez y
    escribe int8+scale sin intermedios DRAM cuando Triton disponible.

    Args:
        x: tensor [..., K] en bf16/fp16/fp32 (cualquier leading dims).
           Ultima dim es la de cuantizacion.

    Returns:
        (x_i8, scale) donde x_i8 es [..., K] int8 y scale es [...,1] fp32.
        Pura: no muta x. 100% GPU.

    Camino rapido: Triton kernel _fused_quant_per_token_kernel (1 load bf16,
        1 store int8 + 1 store scale por fila, reduccion tl.max).
    Fallback: torch vectorizado GPU si Triton no disponible, N>BLOCK_MAX,
        o tensor en host.
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError("quant_activation_per_token: x debe ser torch.Tensor")
    if x.dim() < 1:
        raise ValueError(f"quant_activation_per_token: x.dim debe ser >=1, got {x.dim()}")
    if x.numel() == 0:
        # tensor vacio: devolver vacio con misma forma
        scale_shape = x.shape[:-1] + (1,) if x.dim() >= 1 else (1,)
        return torch.empty_like(x, dtype=torch.int8), torch.ones(scale_shape, dtype=torch.float32, device=x.device)

    # flatten para kernel: [...,K] -> [M,K] generico (soporta 1-D, 2-D, 3-D)
    orig_shape = x.shape
    k = int(orig_shape[-1])
    m = x.numel() // k if k else 0
    x_2d = x.reshape(m, k)
    m_2d, n_2d = x_2d.shape

    # Fallback si no hay Triton/CUDA o dims exceden BLOCK_MAX o tensor no CUDA
    use_triton = (
        _TRITON_OK
        and x.is_cuda
        and torch.cuda.is_available()
        and n_2d <= BLOCK_MAX
        and n_2d > 0
    )
    if use_triton:
        try:
            # BLOCK = next power of 2 >= N (constexpr Triton)
            block = triton.next_power_of_2(n_2d)  # type: ignore
            if block > BLOCK_MAX:
                block = BLOCK_MAX
            # si N no potencia, block>N y mask cubre tail
            # Triton exige BLOCK potencia de 2; next_power_of_2 lo garantiza
            out_2d = torch.empty((m_2d, n_2d), dtype=torch.int8, device=x.device)
            scale_2d = torch.empty((m_2d,), dtype=torch.float32, device=x.device)

            grid = (m_2d,)

            # strides: Triton kernel usa stride_xm = stride(0), stride_xn = stride(1)
            stride_xm = x_2d.stride(0)
            stride_xn = x_2d.stride(1)
            stride_out_m = out_2d.stride(0)
            stride_out_n = out_2d.stride(1)
            stride_scale = scale_2d.stride(0)

            _fused_quant_per_token_kernel[grid](
                x_2d,
                out_2d,
                scale_2d,
                n_2d,
                stride_xm,
                stride_xn,
                stride_out_m,
                stride_out_n,
                stride_scale,
                BLOCK=block,
                num_warps=4,
                num_stages=2,
            )
            # restaurar leading dims: out [...,K] int8, scale [...,1] fp32
            out = out_2d.reshape(orig_shape)
            scale = scale_2d.reshape(orig_shape[:-1] + (1,)).contiguous()
            return out, scale
        except Exception as e:
            log.warning(
                "fused_quant_triton Triton fallo (%s: %s) — fallback torch GPU",
                type(e).__name__,
                e,
            )
            # caer a fallback torch
            pass

    # Fallback torch 100% GPU
    return _quant_fallback_torch(x)


# Alias compatibles con wiring PN110
fused_quant_per_token = quant_activation_per_token
fused_quant_triton = quant_activation_per_token

__all__ = [
    "quant_activation_per_token",
    "fused_quant_per_token",
    "fused_quant_triton",
    "is_available",
    "is_triton_available",
]
