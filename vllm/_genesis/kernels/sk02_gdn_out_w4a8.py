# SPDX-License-Identifier: Apache-2.0
"""SK-02 GDN_OUT_W4A8_SCALED — branchless RowParallel 5120x6144 W4A8 sm_86.

Branchless: solo tl.load/tl.dot/tl.where/tl.store. Validacion movida a
patch_PN110 warmup_all_kernels (dims%16, dtype int8/int4, contiguity,
K%128==0, N%128==0). Cualquier error es bug del pwal, no del kernel.

Geometria: Global [5120,6144]=[40*128,48*128], por-rank TP2 [5120,3072].
W4A8: a [M,K] int8 per-token, b_packed [K,N//2] uint8 int4 packed
(2 valores int4 por byte, low nibble = canal par, high nibble = canal impar),
b_scale [N] bf16 per-channel, shifts [K/128,N/128] int8 diadico.
Diseno C sm_86: w = q * 2^shift * s_row, shift sobre INT32, epilogo
bf16 * a_scale * s_row. Unpack branchless via &0xF, >>4, tl.where.
INT32→bf16 directo, acc bf16, 2x BW vs fp32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK_ID = "SK-02"
SK_NAME = "GDN_OUT_W4A8_SCALED"
GLOBAL_SHAPE = (5120, 6144)
RANK_SHAPE = (5120, 3072)
NUM_LAYERS = 48
ROW_PARALLEL = True
SHIFT_BLOCK = 128
GROUP_SIZE = 128
BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32
PACK_FACTOR = 2


@triton.jit
def _sk02_gdn_out_w4a8_kernel(
    a_ptr,
    b_packed_ptr,
    out_ptr,
    a_scale_ptr,
    b_scale_ptr,
    shifts_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_out_m,
    stride_out_n,
    stride_scale_a,
    stride_scale_b,
    stride_shift_k,
    stride_shift_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SHIFT_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n_half = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_n_half = offs_n_half < (N // 2)
    a_scales = tl.load(a_scale_ptr + offs_m * stride_scale_a, mask=mask_m, other=0.0).to(tl.bfloat16)
    b_scales_even = tl.load(b_scale_ptr + offs_n_half * 2 * stride_scale_b, mask=(offs_n_half * 2 < N), other=1.0).to(tl.bfloat16)
    b_scales_odd = tl.load(b_scale_ptr + (offs_n_half * 2 + 1) * stride_scale_b, mask=((offs_n_half * 2 + 1) < N), other=1.0).to(tl.bfloat16)
    acc_even = tl.zeros((BLOCK_M, BLOCK_N // 2), dtype=tl.bfloat16)
    acc_odd = tl.zeros((BLOCK_M, BLOCK_N // 2), dtype=tl.bfloat16)
    num_kb = K // SHIFT_BLOCK
    for kb in range(num_kb):
        shift_col = (pid_n * BLOCK_N) // SHIFT_BLOCK
        shift_val = tl.load(shifts_ptr + kb * stride_shift_k + shift_col * stride_shift_n).to(tl.int32)
        int_acc_even = tl.zeros((BLOCK_M, BLOCK_N // 2), dtype=tl.int32)
        int_acc_odd = tl.zeros((BLOCK_M, BLOCK_N // 2), dtype=tl.int32)
        k_base = kb * SHIFT_BLOCK
        for sub in range(SHIFT_BLOCK // BLOCK_K):
            k_offs = k_base + sub * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
            mask_a = mask_m[:, None] & (k_offs[None, :] < K)
            a_tile = tl.load(a_ptrs, mask=mask_a, other=0).to(tl.int8)
            b_packed_ptrs = b_packed_ptr + k_offs[:, None] * stride_bk + offs_n_half[None, :] * stride_bn
            mask_b_packed = (k_offs[:, None] < K) & mask_n_half[None, :]
            b_packed_tile = tl.load(b_packed_ptrs, mask=mask_b_packed, other=0).to(tl.int32)
            b_low = b_packed_tile & 0xF
            b_high = (b_packed_tile >> 4) & 0xF
            b_low = tl.where(b_low >= 8, b_low - 16, b_low)
            b_high = tl.where(b_high >= 8, b_high - 16, b_high)
            b_low_i8 = b_low.to(tl.int8)
            b_high_i8 = b_high.to(tl.int8)
            int_acc_even = int_acc_even + tl.dot(a_tile, b_low_i8)
            int_acc_odd = int_acc_odd + tl.dot(a_tile, b_high_i8)
        shifted_even = tl.where(shift_val >= 0, int_acc_even << shift_val, int_acc_even >> (-shift_val))
        shifted_odd = tl.where(shift_val >= 0, int_acc_odd << shift_val, int_acc_odd >> (-shift_val))
        shifted_even_f = shifted_even.to(tl.bfloat16)
        shifted_odd_f = shifted_odd.to(tl.bfloat16)
        scaled_even = shifted_even_f * a_scales[:, None] * b_scales_even[None, :]
        scaled_odd = shifted_odd_f * a_scales[:, None] * b_scales_odd[None, :]
        acc_even = acc_even + scaled_even
        acc_odd = acc_odd + scaled_odd
    out_ptrs_even = out_ptr + offs_m[:, None] * stride_out_m + (offs_n_half * 2)[None, :] * stride_out_n
    out_ptrs_odd = out_ptr + offs_m[:, None] * stride_out_m + (offs_n_half * 2 + 1)[None, :] * stride_out_n
    mask_out_even = mask_m[:, None] & ((offs_n_half * 2)[None, :] < N)
    mask_out_odd = mask_m[:, None] & ((offs_n_half * 2 + 1)[None, :] < N)
    tl.store(out_ptrs_even, acc_even, mask=mask_out_even)
    tl.store(out_ptrs_odd, acc_odd, mask=mask_out_odd)


def sk02_gdn_out_w4a8_gemm(
    a: torch.Tensor,
    b_packed: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = a.shape[0]
    N = b_packed.shape[1] * 2
    K = a.shape[1]
    # asegurar escalas bf16 sin .cpu()
    if a_scales.dtype != out_dtype:
        a_scales = a_scales.to(out_dtype)
    if b_scales.dtype != out_dtype:
        b_scales = b_scales.to(out_dtype)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk02_gdn_out_w4a8_kernel[grid](
        a,
        b_packed,
        out,
        a_scales,
        b_scales,
        shifts,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        out.stride(0),
        out.stride(1),
        1,
        1,
        shifts.stride(0),
        shifts.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SHIFT_BLOCK=SHIFT_BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return out


def sk02_gdn_out_w4a8_proj(
    hidden: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    shifts: torch.Tensor,
    residual: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    hidden_bf16 = hidden.to(out_dtype) if hidden.dtype != out_dtype else hidden
    amax = hidden_bf16.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    hidden_i8 = (hidden_bf16 / scale).round().clamp(-127, 127).to(torch.int8)
    a_scales = scale.squeeze(-1).to(out_dtype)
    w_scale_bf16 = weight_scale.to(out_dtype) if weight_scale.dtype != out_dtype else weight_scale
    M = hidden_i8.shape[0]
    N = weight_packed.shape[1] * 2
    K = hidden_i8.shape[1]
    out = torch.empty((M, N), dtype=out_dtype, device=hidden.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk02_gdn_out_w4a8_kernel[grid](
        hidden_i8,
        weight_packed,
        out,
        a_scales,
        w_scale_bf16,
        shifts,
        M,
        N,
        K,
        hidden_i8.stride(0),
        hidden_i8.stride(1),
        weight_packed.stride(0),
        weight_packed.stride(1),
        out.stride(0),
        out.stride(1),
        1,
        1,
        shifts.stride(0),
        shifts.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SHIFT_BLOCK=SHIFT_BLOCK,
        num_warps=4,
        num_stages=2,
    )
    out = out + residual.to(out_dtype) if residual.dtype != out_dtype else out + residual
    return out


sk02_gdn_out_w4a8 = sk02_gdn_out_w4a8_proj
gdn_out_w4a8_scaled = sk02_gdn_out_w4a8_gemm
sk02_gemm_w4a8 = sk02_gdn_out_w4a8_gemm

__all__ = [
    "SK_ID",
    "SK_NAME",
    "GLOBAL_SHAPE",
    "RANK_SHAPE",
    "NUM_LAYERS",
    "ROW_PARALLEL",
    "SHIFT_BLOCK",
    "GROUP_SIZE",
    "PACK_FACTOR",
    "sk02_gdn_out_w4a8_gemm",
    "sk02_gdn_out_w4a8_proj",
    "sk02_gdn_out_w4a8",
    "gdn_out_w4a8_scaled",
    "sk02_gemm_w4a8",
]
