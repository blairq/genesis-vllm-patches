# SPDX-License-Identifier: Apache-2.0
"""Kernel custom Diseño C híbrido diádico — PN110 §4C.

Diseño C: ``w = q * 2^shift_b * s_row`` donde q int8, shift_b int8 por bloque
(128×128), s_row fp32 per-channel.

Contrato del kernel (ver fp8_a_int8_ampere.md §4C):
    int_acc = a_i8 @ b_i8  (INT32, 128 productos → ≤127*127*128=2.06M)
    int_acc_shifted = int_acc << shift_b   (>> si shift negativo, aritmético)
    out = int_acc_shifted * a_scale * s_row  (fp32) → cast a fp16/bf16

Overflow (§4C): INT32 max 2.147G, shift ≤10 cabe (2.06M<<10≈2.11G) con margen.
Shift es barato (entero sobre acumulador), no multiply flotante.

Firma requerida:
    a        [M, K] int8
    b        [K, N] int8
    a_scales [M, 1] fp32  (per-token)
    b_scales [N, 1] fp32  (per-channel, s_row)
    shifts   [K/128, N/128] int8  (per-bloque diádico)
    out_dtype fp16/bf16

Acumulación por bloques de K=128 con shift aplicado por bloque sobre el
acumulador INT32, luego epílogo fp32.

Estado actual:
    - Triton kernel ``_int8_hybrid_gemm_kernel`` implementa el camino rápido
      (tl.dot INT8→INT32 + shift INT32 + epílogo fp32) cuando Triton está
      disponible y dims son múltiplo de 128.
    - Fallback Python (``_hybrid_fallback_torch``) emula la semántica exacta
      con matmuls por bloques en fp32 (100× más lento, documentado) — solo
      para validación de exactitud, NO para performance.
    - Si Triton falla o no está, ``int8_hybrid_gemm`` hace WARNING y usa el
      fallback (o delega al camino B per-channel si caller lo pide), dejando
      claro que el kernel custom queda pendiente de optimización.

No toca el camino B (cutlass_scaled_mm per-channel). Solo añade dispatch.

Author: Genesis PN110 — Diseño C híbrido (ox-alpha 2026-08-25)
"""
from __future__ import annotations

import logging
import warnings

import torch

log = logging.getLogger("genesis.kernels.int8_hybrid_gemm")

_TRITON_OK = False
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    _TRITON_OK = True
except Exception as _e:
    log.warning("int8_hybrid_gemm: Triton no disponible (%s) — solo fallback", _e)
    triton = None  # type: ignore
    tl = None  # type: ignore

# ── Configuración de bloque ─────────────────────────────────────────────
# SHIFT_BLOCK es la granularidad del shift diádico (§4C): 128.
# Elegimos BLOCK_M=32, BLOCK_N=64 o 128, BLOCK_K=32 para tl.dot.
# Para alinear shift por bloque 128×128 sin fracciones, el kernel exige
# N % 128 == 0 y K % 128 == 0 (garantizado por §1.4: todas las dims del
# modelo son múltiplo de 128) y usa BLOCK_N=64/128 con mapeo a shifts.
BLOCK_M = 32
BLOCK_N = 64  # se ajusta a 128 si N%128==0; kernel soporta ambos vía constexpr
BLOCK_K = 32
# SHIFT_BLOCK (=128) ahora es tl.constexpr del kernel (fix NameError Triton constexpr)

_HAS_WARNED_FALLBACK = False


def is_triton_available() -> bool:
    """True si Triton está importable."""
    return _TRITON_OK


def is_available() -> bool:
    """True si el kernel custom está disponible (Triton + CUDA)."""
    if not _TRITON_OK:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# ── Triton kernel ───────────────────────────────────────────────────────
if _TRITON_OK:
    @triton.jit
    def _int8_hybrid_gemm_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        a_scale_ptr,
        b_scale_ptr,
        shifts_ptr,
        M,
        N,
        K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_out_m, stride_out_n,
        stride_scales_a,
        stride_scales_b,
        stride_shift_k, stride_shift_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SHIFT_BLOCK: tl.constexpr,
    ):
        """
        Triton kernel Diseño C híbrido.

        Cada programa calcula un bloque [BLOCK_M, BLOCK_N] de la salida.
        Itera sobre K en pasos de SHIFT_BLOCK=128. Por cada chunk de 128:
          - acumula 128 productos int8*int8 en INT32 (4× tl.dot BLOCK_K=32)
          - aplica shift entero sobre el acumulador INT32
          - convierte a fp32, multiplica por a_scale y b_scale, acumula en fp32.

        Requiere M,N,K múltiplo de SHIFT_BLOCK para camino rápido; el wrapper
        hace fallback si no.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        # Escalas per-token [M] y per-channel [N] — load vectorizado
        a_scales = tl.load(a_scale_ptr + offs_m * stride_scales_a, mask=mask_m, other=0.0).to(tl.float32)
        b_scales = tl.load(b_scale_ptr + offs_n * stride_scales_b, mask=mask_n, other=0.0).to(tl.float32)

        # Acumulador final fp32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Número de bloques de 128 en K
        # K es múltiplo de 128 garantizado; usar loop constexpr-friendly
        num_kb = K // SHIFT_BLOCK

        for kb in range(num_kb):
            # shift para este (kb, nb) — BLOCK_N=64 requiere 2 shifts? Simplificamos
            # exigiendo BLOCK_N múltiplo de 128 o manejando sub-bloque.
            # Si BLOCK_N==64, el bloque N cubre media columna de shifts (128/64=2);
            # para exactitud tomamos el shift de la primera mitad y asumimos
            # uniformidad dentro del superbloque 128. El wrapper en Python
            # garantiza que esto solo se usa cuando shifts por Bn es constante
            # dentro del bloque o fallback a torch si no. Para BLOCK_N=128 es 1:1.
            # Cargamos shift correspondiente al bloque N actual:
            #   shift_nb = pid_n  si BLOCK_N==128
            #   shift_nb = pid_n //2 si BLOCK_N==64 y shifts por 128
            # Para generalidad, mapeamos pid_n -> shift col index:
            # shift_col = (pid_n * BLOCK_N) // SHIFT_BLOCK
            shift_col = (pid_n * BLOCK_N) // SHIFT_BLOCK
            shift_val = tl.load(shifts_ptr + kb * stride_shift_k + shift_col * stride_shift_n).to(tl.int32)

            # Acumulador INT32 para este chunk de 128 (4×32)
            int_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

            # K offsets para este chunk: kb*128 + [0,32,64,96]
            k_base = kb * SHIFT_BLOCK
            # 4 sub-iteraciones de BLOCK_K
            for sub in range(SHIFT_BLOCK // BLOCK_K):
                k_offs = k_base + sub * BLOCK_K + tl.arange(0, BLOCK_K)

                # Load A tile [BLOCK_M, BLOCK_K] int8
                a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
                # Mask: offs_m < M y k_offs < K (K sempre múltiplo, pero por seguridad)
                mask_a = mask_m[:, None] & (k_offs[None, :] < K)
                a_tile = tl.load(a_ptrs, mask=mask_a, other=0).to(tl.int8)

                # Load B tile [BLOCK_K, BLOCK_N] int8
                # B es [K,N]: row K, col N
                b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
                mask_b = (k_offs[:, None] < K) & mask_n[None, :]
                b_tile = tl.load(b_ptrs, mask=mask_b, other=0).to(tl.int8)

                # tl.dot int8 -> int32
                # tl.dot espera [M,K] @ [K,N] -> [M,N]
                int_acc = int_acc + tl.dot(a_tile, b_tile)

            # Aplicar shift sobre acumulador INT32 (aritmético)
            # shift >=0: <<  ; shift <0: >> aritmético
            # Triton no tiene where sobre int shift variable? Usamos branching constexpr
            # con tl.where
            shifted = tl.where(
                shift_val >= 0,
                int_acc << shift_val,
                int_acc >> (-shift_val),
            )

            # Convertir a fp32 y aplicar escalas: * a_scale * b_scale
            # a_scales [BLOCK_M], b_scales [BLOCK_N] -> outer product broadcast
            shifted_f = shifted.to(tl.float32)
            scaled = shifted_f * a_scales[:, None] * b_scales[None, :]
            acc = acc + scaled

        # Store acc [BLOCK_M, BLOCK_N] -> out [M,N]
        out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
        mask_out = mask_m[:, None] & mask_n[None, :]
        tl.store(out_ptrs, acc, mask=mask_out)


# ── Fallback vectorizado 100% GPU (sin loops Python, sin .cpu()) ──
def _hybrid_fallback_torch(
    a: torch.Tensor,  # [M,K] int8
    b: torch.Tensor,  # [K,N] int8
    a_scales: torch.Tensor,  # [M,1] or [M] fp32
    b_scales: torch.Tensor,  # [N,1] or [N] fp32
    shifts: torch.Tensor,  # [K/128, N/128] int8
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fallback vectorizado 100% GPU — sin ``for`` ni ``.cpu()``.

    Contrato nuevo: ``int_acc = a_i8 @ b_i8`` en INT32 en GPU
    (``torch.matmul`` con ``int32`` o ``cutlass`` si disponible),
    luego ``int_acc << shift`` vía ``torch.bitwise_left_shift`` en GPU,
    luego ``* a_scale * s_row`` en GPU. Todo ``torch.*`` en GPU.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"K mismatch {K} vs {K2}"
    Bk = 128
    Bn = 128
    num_kb = K // Bk
    num_nb = N // Bn

    # Normalizar escalas a [M] y [N] fp32 1-D GPU
    if a_scales.dim() == 2:
        if a_scales.shape == (M, 1):
            a_vec = a_scales.squeeze(1).to(torch.float32)
        elif a_scales.shape == (1, M):
            a_vec = a_scales.squeeze(0).to(torch.float32)
        else:
            a_vec = a_scales.reshape(-1).to(torch.float32)[:M]
    elif a_scales.dim() == 1:
        a_vec = a_scales.to(torch.float32)[:M]
    else:
        a_vec = a_scales.reshape(-1).to(torch.float32)[:M]

    if b_scales.dim() == 2:
        if b_scales.shape == (N, 1):
            b_vec = b_scales.squeeze(1).to(torch.float32)
        elif b_scales.shape == (1, N):
            b_vec = b_scales.squeeze(0).to(torch.float32)
        else:
            b_vec = b_scales.reshape(-1).to(torch.float32)[:N]
    elif b_scales.dim() == 1:
        b_vec = b_scales.to(torch.float32)[:N]
    else:
        b_vec = b_scales.reshape(-1).to(torch.float32)[:N]

    # ── 100% GPU: int_acc INT32 via matmul por bloques vectorizado ──
    # a [M,K] -> [M,Kb,Bk], b [K,N] -> [Kb,Bk,Nb,Bn] para einsum sin loops
    a_i32 = a.to(torch.int32)
    b_i32 = b.to(torch.int32)
    a_3d = a_i32.view(M, num_kb, Bk)
    b_4d = b_i32.view(num_kb, Bk, num_nb, Bn)
    # einsum GPU vectorizado: [M,Kb,Bk] x [Kb,Bk,Nb,Bn] -> [M,Kb,Nb,Bn] sum Bk
    int_acc_4d = torch.einsum('mkb,kbnt->mknt', a_3d, b_4d)
    # shifts [Kb,Nb] int8 -> broadcast a [M,Kb,Nb,Bn]
    shifts_4d = shifts.view(1, num_kb, num_nb, 1).expand(M, num_kb, num_nb, Bn).to(torch.int32)
    int_acc_4d_i32 = int_acc_4d.to(torch.int32)
    shifted_4d = torch.where(
        shifts_4d >= 0,
        torch.bitwise_left_shift(int_acc_4d_i32, shifts_4d),
        torch.bitwise_right_shift(int_acc_4d_i32, -shifts_4d),
    )
    # Escalas GPU: a_vec [M] -> [M,1,1,1], b_vec [N] -> [Nb,Bn] -> [1,1,Nb,Bn]
    a_bc = a_vec.view(M, 1, 1, 1).expand(M, num_kb, num_nb, Bn)
    b_bc = b_vec.view(num_nb, Bn).view(1, 1, num_nb, Bn).expand(M, num_kb, num_nb, Bn)
    contrib_4d = shifted_4d.to(torch.float32) * a_bc * b_bc
    # Sum sobre Kb y reshape a [M,N]
    out_4d = contrib_4d.sum(dim=1)
    out_fp32 = out_4d.reshape(M, N)
    return out_fp32.to(out_dtype)


# ── API pública ─────────────────────────────────────────────────────────
def int8_hybrid_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    block_k: int = 128,
) -> torch.Tensor:
    """GEMM híbrido Diseño C: INT8 @ INT8 → INT32 → shift → fp32 → cast.

    Contrato §4C: ``int_acc = a_i8 @ b_i8`` (INT32), luego
    ``int_acc_shifted = int_acc << shift_b`` (>> si negativo, aritmético),
    luego ``out = int_acc_shifted * a_scale * s_row`` (fp32) y casteo a
    fp16/bf16. Overflow seguro: 128*127*127=2.06M << 2.1G.

    Args:
        a:        [M, K] int8 (row-major).
        b:        [K, N] int8 (puede ser column-major con stride (1,K)).
        a_scales: [M, 1] fp32 per-token (o [M] 1-D).
        b_scales: [N, 1] fp32 per-channel s_row (o [N] 1-D).
        shifts:   [K/128, N/128] int8 per-bloque diádico.
        out_dtype: dtype de salida (fp16/bf16). Si None, usa bf16 si
                   a es bf16 else fp16.
        block_k:  granularidad del shift (default 128, debe dividir K y N).

    Returns:
        Tensor [M, N] en out_dtype.

    Notas:
        - Si Triton disponible y dims múltiplo de 128, usa kernel Triton
          (shift entero sobre INT32, barato).
        - Si no, WARNING + fallback torch (100× más lento, solo validación
          de exactitud — KL 2000× mejor que B — no para producción).
        - No toca el camino B; el caller decide fallback a B si quiere.
    """
    global _HAS_WARNED_FALLBACK
    if out_dtype is None:
        out_dtype = torch.float16

    # Validaciones básicas
    assert a.dtype == torch.int8, f"a debe ser int8, es {a.dtype}"
    assert b.dtype == torch.int8, f"b debe ser int8, es {b.dtype}"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"K mismatch a {K} vs b {K2}"
    assert K % block_k == 0, f"K {K} no múltiplo de block_k {block_k}"
    assert N % block_k == 0, f"N {N} no múltiplo de block_k {block_k}"
    # shifts debe ser [K/128, N/128]
    expected_kb = K // block_k
    expected_nb = N // block_k
    assert shifts.shape == (expected_kb, expected_nb), (
        f"shifts shape {tuple(shifts.shape)} != esperado {(expected_kb, expected_nb)} "
        f"para K={K} N={N} block_k={block_k}"
    )

    # Camino rápido Triton
    if is_available() and a.is_cuda and b.is_cuda:
        # Requiere múltiplo de SHIFT_BLOCK y Triton OK
        try:
            # Normalizar escalas a 1-D contiguas fp32 para kernel
            # a_scales puede venir como [M,1] o [M]
            if a_scales.dim() == 2 and a_scales.shape[1] == 1:
                a_scales_1d = a_scales.squeeze(1).contiguous().to(torch.float32)
            elif a_scales.dim() == 1:
                a_scales_1d = a_scales.contiguous().to(torch.float32)
            else:
                # Caso [1,M] o [M,1] transpuesto
                a_scales_1d = a_scales.reshape(-1).contiguous().to(torch.float32)[:M]

            if b_scales.dim() == 2 and b_scales.shape[0] == N:
                # [N,1]
                b_scales_1d = b_scales.squeeze(1).contiguous().to(torch.float32)
            elif b_scales.dim() == 2 and b_scales.shape[1] == 1:
                # confuso: b_scales [N,1] ya cubierto; [1,N] sería otro
                b_scales_1d = b_scales.reshape(-1).contiguous().to(torch.float32)[:N]
            elif b_scales.dim() == 1:
                b_scales_1d = b_scales.contiguous().to(torch.float32)
            else:
                b_scales_1d = b_scales.reshape(-1).contiguous().to(torch.float32)[:N]

            # Asegurar contigüidad y device
            # a y b pueden ser column-major (stride 1,K); el kernel maneja strides
            # genéricos, no forzamos .contiguous() para no copiar en hot path.
            # Pero para Triton tl.load con strides genéricos, funciona igual.

            # Lanzar kernel
            out = torch.empty((M, N), dtype=out_dtype, device=a.device)
            # Pero el kernel acumula en fp32 y storea fp32; casteamos después si out_dtype != fp32
            # Para simplificar, el kernel escribe fp32 y casteamos — usamos buffer fp32 intermedio
            # si out_dtype no es fp32. El kernel actual escribe fp32 directo; hacemos buffer fp32
            # y luego cast si hace falta.
            # Si out_dtype es fp16/bf16, el kernel haría store fp32 y perdería perf; hacemos
            # buffer fp32 + cast. Es aceptable para skeleton funcional.
            out_fp32 = torch.empty((M, N), dtype=torch.float32, device=a.device)

            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

            # Strides: para a [M,K] row-major stride_am = K si contiguous, pero calculamos real
            stride_am = a.stride(0)
            stride_ak = a.stride(1)
            stride_bk = b.stride(0)
            stride_bn = b.stride(1)
            stride_out_m = out_fp32.stride(0)
            stride_out_n = out_fp32.stride(1)
            stride_sa = 1  # 1-D contiguous
            stride_sb = 1
            stride_shift_k = shifts.stride(0)
            stride_shift_n = shifts.stride(1)

            # Heurística BLOCK_N: si N%128==0 usar 64 o 128 según N
            # Para máxima compatibilidad con shifts 128×128, usamos BLOCK_N=64
            # y mapeo shift_col = (pid_n*BLOCK_N)//128 dentro del kernel.
            # Lanzar con BLOCK_M=32, BLOCK_N=64 (o 128 si N>=2048 y divisible)
            _block_m = 32
            _block_n = 64 if N % 64 == 0 else 32
            # Si N%128==0 y N>=128, preferir 128 para 1:1 shift mapping y menos grid
            if N % 128 == 0 and N >= 128:
                # Usar 64 sigue siendo válido con mapeo, pero 128 es más eficiente
                # Elegir 64 para Triton occupancy en A5000 (empírico PN50: 64 mejor)
                _block_n = 64

            _int8_hybrid_gemm_kernel[grid](
                a, b, out_fp32,
                a_scales_1d, b_scales_1d, shifts,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_out_m, stride_out_n,
                stride_sa, stride_sb,
                stride_shift_k, stride_shift_n,
                BLOCK_M=_block_m,
                BLOCK_N=_block_n,
                BLOCK_K=BLOCK_K,
                SHIFT_BLOCK=128,
                num_warps=4,
                num_stages=2,
            )
            if out_dtype != torch.float32:
                out = out_fp32.to(out_dtype)
            else:
                out = out_fp32
            return out
        except Exception as e:
            log.warning(
                "int8_hybrid_gemm Triton falló (%s: %s) — fallback torch (solo validación, no performance)",
                type(e).__name__, e,
            )
            if not _HAS_WARNED_FALLBACK:
                warnings.warn(
                    f"int8_hybrid_gemm Triton falló ({type(e).__name__}); "
                    "usando fallback torch 100× más lento — solo validación de exactitud, "
                    "no para producción. Kernel custom pendiente de estabilización.",
                    UserWarning, stacklevel=2,
                )
                _HAS_WARNED_FALLBACK = True
            # Caer al fallback torch
            try:
                return _hybrid_fallback_torch(a, b, a_scales, b_scales, shifts, out_dtype)
            except Exception as e2:
                log.warning("int8_hybrid_gemm fallback torch también falló (%s) — propagando", e2)
                raise

    # Fallback torch si no hay Triton/CUDA o dims no múltiplo
    if not _HAS_WARNED_FALLBACK and not is_available():
        warnings.warn(
            "int8_hybrid_gemm: Triton/CUDA no disponible — usando fallback torch "
            "(100× más lento, 2000× mejor KL que B pero no para performance). "
            "Kernel custom queda pendiente; hybrid por ahora solo validación de exactitud.",
            UserWarning, stacklevel=2,
        )
        log.warning(
            "int8_hybrid_gemm fallback torch activo (Triton no disponible) — "
            "solo validación de exactitud, no para performance. Kernel custom pendiente."
        )
        _HAS_WARNED_FALLBACK = True
    elif not _HAS_WARNED_FALLBACK:
        # Triton disponible pero no se pudo usar (ej. CPU tensor)
        warnings.warn(
            "int8_hybrid_gemm: fallback torch (solo validación, no performance) — "
            "kernel Triton disponible pero no aplicable a estos tensores/dims. "
            "Hybrid sigue siendo validación de exactitud.",
            UserWarning, stacklevel=2,
        )
        _HAS_WARNED_FALLBACK = True

    return _hybrid_fallback_torch(a, b, a_scales, b_scales, shifts, out_dtype)


# Alias para compatibilidad con naming del contrato
hybrid_gemm = int8_hybrid_gemm

__all__ = [
    "int8_hybrid_gemm",
    "hybrid_gemm",
    "is_available",
    "is_triton_available",
    "_hybrid_fallback_torch",
]
