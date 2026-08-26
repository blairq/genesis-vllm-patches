# SPDX-License-Identifier: Apache-2.0
"""SK-06 MLP_DOWN W4A8 — branchless GEMM+residual con peso INT4 sm_86.

Branchless: unicamente tl.load / tl.dot / tl.where / tl.store. Asume shift siempre
habilitado y residual siempre presente. Validacion (K%128,N%128, dtype int8,
packed b [K//2,N] column-major stride(1,K//2), a_scale [M] bf16,
b_scale [K//128,N] bf16, b_zp [K//128,N] int8, shifts [K//128,N//128] int8,
contiguity) movida a warmup_all_kernels en patch_PN110. Peso INT4 empaquetado
2 por byte, dequant branchless via bit ops. Activacion int8 per-token. sm_86 bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

GLOBAL_N: int = 5120
GLOBAL_K: int = 17408
PER_RANK_N: int = 5120
PER_RANK_K: int = 8704
BLOCK: int = 128
SHIFT_BLOCK: int = 128
GROUP_SIZE: int = 128
BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32


@triton.jit
def _sk06_mlp_down_w4a8_kernel(
    a_ptr,
    b_packed_ptr,
    out_ptr,
    resid_ptr,
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    shifts_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bpk,
    stride_bpn,
    stride_out_m,
    stride_out_n,
    stride_resid_m,
    stride_resid_n,
    stride_scale_a,
    stride_scale_b_k,
    stride_scale_b_n,
    stride_zp_k,
    stride_zp_n,
    stride_shift_k,
    stride_shift_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SHIFT_BLOCK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_scales = tl.load(a_scale_ptr + offs_m * stride_scale_a, mask=mask_m, other=0.0).to(tl.bfloat16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    num_kb = K // SHIFT_BLOCK
    for kb in range(num_kb):
        shift_col = (pid_n * BLOCK_N) // SHIFT_BLOCK
        shift_val = tl.load(shifts_ptr + kb * stride_shift_k + shift_col * stride_shift_n).to(tl.int32)
        b_scale_vec = tl.load(b_scale_ptr + kb * stride_scale_b_k + offs_n * stride_scale_b_n, mask=mask_n, other=1.0).to(tl.bfloat16)
        b_zp_vec = tl.load(b_zp_ptr + kb * stride_zp_k + offs_n * stride_zp_n, mask=mask_n, other=0).to(tl.int32)
        int_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        k_base = kb * SHIFT_BLOCK
        for sub in range(SHIFT_BLOCK // BLOCK_K):
            k_offs = k_base + sub * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
            mask_a = mask_m[:, None] & (k_offs[None, :] < K)
            a_tile = tl.load(a_ptrs, mask=mask_a, other=0).to(tl.int8)
            packed_k = k_offs // 2
            b_packed_ptrs = b_packed_ptr + packed_k[:, None] * stride_bpk + offs_n[None, :] * stride_bpn
            mask_b = (packed_k[:, None] < (K // 2)) & mask_n[None, :]
            b_packed = tl.load(b_packed_ptrs, mask=mask_b, other=0).to(tl.int32)
            b_packed_u = b_packed & 0xFF
            low = b_packed_u & 0x0F
            high = (b_packed_u >> 4) & 0x0F
            is_even = (k_offs % 2) == 0
            b_int4 = tl.where(is_even[:, None], low, high)
            b_deq = b_int4 - b_zp_vec[None, :]
            b_tile = b_deq.to(tl.int8)
            int_acc = int_acc + tl.dot(a_tile, b_tile)
        shifted = tl.where(shift_val >= 0, int_acc << shift_val, int_acc >> (-shift_val))
        shifted_f = shifted.to(tl.bfloat16)
        scaled = shifted_f * a_scales[:, None] * b_scale_vec[None, :]
        acc = acc + scaled
    resid_ptrs = resid_ptr + offs_m[:, None] * stride_resid_m + offs_n[None, :] * stride_resid_n
    mask_resid = mask_m[:, None] & mask_n[None, :]
    resid = tl.load(resid_ptrs, mask=mask_resid, other=0.0).to(tl.bfloat16)
    acc = acc + resid
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


def mlp_down_w4a8_scaled_residual(
    x: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    b_zp: torch.Tensor,
    residual: torch.Tensor,
    weight_shifts: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = x.shape[0]
    K = x.shape[1]
    N = b_packed.shape[1]
    x_bf16 = x.to(out_dtype) if x.dtype != out_dtype else x
    amax = x_bf16.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_i8 = (x_bf16 / scale).round().clamp(-127, 127).to(torch.int8)
    a_scales = scale.squeeze(-1).to(out_dtype)
    b_scale_bf16 = b_scale.to(out_dtype) if b_scale.dtype != out_dtype else b_scale
    resid_bf16 = residual.to(out_dtype) if residual.dtype != out_dtype else residual
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk06_mlp_down_w4a8_kernel[grid](
        x_i8,
        b_packed,
        out,
        resid_bf16,
        a_scales,
        b_scale_bf16,
        b_zp,
        weight_shifts,
        M,
        N,
        K,
        x_i8.stride(0),
        x_i8.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        out.stride(0),
        out.stride(1),
        resid_bf16.stride(0),
        resid_bf16.stride(1),
        1,
        b_scale_bf16.stride(0),
        b_scale_bf16.stride(1),
        b_zp.stride(0),
        b_zp.stride(1),
        weight_shifts.stride(0),
        weight_shifts.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SHIFT_BLOCK=SHIFT_BLOCK,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


def mlp_down_w4a8_gemm(
    x: torch.Tensor,
    b_packed: torch.Tensor,
    b_scale: torch.Tensor,
    b_zp: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = x.shape[0]
    N = b_packed.shape[1]
    residual = torch.zeros((M, N), dtype=out_dtype, device=x.device)
    K = x.shape[1]
    num_kb = K // SHIFT_BLOCK
    num_nb = N // SHIFT_BLOCK
    shifts = torch.zeros((num_kb, num_nb), dtype=torch.int8, device=x.device)
    return mlp_down_w4a8_scaled_residual(x, b_packed, b_scale, b_zp, residual, shifts, out_dtype)


mlp_down_w4a8 = mlp_down_w4a8_gemm
sk06_mlp_down_w4a8 = mlp_down_w4a8_gemm
sk06_mlp_down_w4a8_scaled = mlp_down_w4a8_scaled_residual

__all__ = [
    "GLOBAL_N",
    "GLOBAL_K",
    "PER_RANK_N",
    "PER_RANK_K",
    "BLOCK",
    "SHIFT_BLOCK",
    "GROUP_SIZE",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "_sk06_mlp_down_w4a8_kernel",
    "mlp_down_w4a8_scaled_residual",
    "mlp_down_w4a8_gemm",
    "mlp_down_w4a8",
    "sk06_mlp_down_w4a8",
]
