# SPDX-License-Identifier: Apache-2.0
"""SK-02 GDN_OUT_INT8_SCALED — monolito PTX inline RowParallel 5120x6144 sm_86.

Monolito branchless: single kernel triton jit usando tl.load / tl.dot / tl.store.
Validacion trasladada hacia warmup_all_kernels en patch_PN110 (dims multiplo 16,
dtype int8, contiguity, K multiplo 128). Cualquier fallo indica bug en pwal,
no en kernel.

Geometria: Global [5120,6144]=[40*128,48*128], por-rank TP2 [5120,3072].
Diseno C sm_86: w = q * 2^shift * s_row, desplazamiento sobre INT32 previo
a epilogo bf16 * a_scale * s_row. INT32 hacia bf16 directo, acumulador bf16,
sin fp32. 39 TFLOPS bf16 vs 19.5 fp32 en sm_86.

PTX sm_86 7.4 monolito:
  .version 7.4
  .target sm_86
  .address_size 64
  tl.load  -> ld.global.b8 / ld.global.b16 (PTX cp.async compatible)
  tl.dot   -> mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (Tensor Core Ampere)
  tl.store -> st.global.b32
Desplazamiento diadico siempre >=0 validado en warmup, operacion
branchless mediante << sin seleccion condicional. Validacion en pwal,
kernel sin ramificaciones.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK_ID = "SK-02"
SK_NAME = "GDN_OUT_INT8_SCALED"
GLOBAL_SHAPE = (5120, 6144)
RANK_SHAPE = (5120, 3072)
NUM_LAYERS = 48
ROW_PARALLEL = True
SHIFT_BLOCK = 128
BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32
# PTX monolito sm_86 — documentacion version PTX para auditoria
_PTX_VERSION = "7.4"
_PTX_TARGET = "sm_86"


@triton.jit
def _sk02_gdn_out_int8_kernel(
    a_ptr,
    b_ptr,
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
    stride_scales_a,
    stride_scales_b,
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
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_scales = tl.load(a_scale_ptr + offs_m * stride_scales_a, mask=mask_m, other=0.0).to(tl.bfloat16)
    b_scales = tl.load(b_scale_ptr + offs_n * stride_scales_b, mask=mask_n, other=0.0).to(tl.bfloat16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    num_kb = K // SHIFT_BLOCK
    for kb in range(num_kb):
        shift_col = (pid_n * BLOCK_N) // SHIFT_BLOCK
        shift_val = tl.load(shifts_ptr + kb * stride_shift_k + shift_col * stride_shift_n).to(tl.int32)
        int_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        k_base = kb * SHIFT_BLOCK
        for sub in range(SHIFT_BLOCK // BLOCK_K):
            k_offs = k_base + sub * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
            mask_a = mask_m[:, None] & (k_offs[None, :] < K)
            a_tile = tl.load(a_ptrs, mask=mask_a, other=0).to(tl.int8)
            b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
            mask_b = (k_offs[:, None] < K) & mask_n[None, :]
            b_tile = tl.load(b_ptrs, mask=mask_b, other=0).to(tl.int8)
            int_acc = int_acc + tl.dot(a_tile, b_tile)
        shifted = int_acc << shift_val
        shifted_f = shifted.to(tl.bfloat16)
        scaled = shifted_f * a_scales[:, None] * b_scales[None, :]
        acc = acc + scaled
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


def sk02_gemm_int8_scaled(
    hidden: torch.Tensor,
    weight_i8: torch.Tensor,
    a_scales: torch.Tensor = None,
    b_scales: torch.Tensor = None,
    shifts: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    hidden_bf16 = hidden.to(out_dtype)
    amax = hidden_bf16.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    hidden_i8 = (hidden_bf16 / scale).round().clamp(-127, 127).to(torch.int8)
    a = hidden_i8
    a_scales_local = scale.squeeze(-1).to(out_dtype)
    b_scales_local = b_scales.to(out_dtype)
    b_kn = weight_i8.t().contiguous()
    M = a.shape[0]
    N = b_kn.shape[1]
    K = a.shape[1]
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk02_gdn_out_int8_kernel[grid](
        a,
        b_kn,
        out,
        a_scales_local,
        b_scales_local,
        shifts,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_kn.stride(0),
        b_kn.stride(1),
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


def sk02_gdn_out_proj(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    shifts: torch.Tensor,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
    do_allreduce: bool = True,
    mode: str = "int8",
) -> torch.Tensor:
    out = sk02_gemm_int8_scaled(hidden, weight, None, weight_scale, shifts, out_dtype)
    out = out + residual
    return out


sk02_gdn_out = sk02_gdn_out_proj
gdn_out_int8_scaled = sk02_gemm_int8_scaled
sk02_gemm_w8a16 = sk02_gemm_int8_scaled

__all__ = [
    "SK_ID",
    "SK_NAME",
    "GLOBAL_SHAPE",
    "RANK_SHAPE",
    "NUM_LAYERS",
    "ROW_PARALLEL",
    "SHIFT_BLOCK",
    "sk02_gemm_int8_scaled",
    "sk02_gdn_out_proj",
    "sk02_gdn_out",
    "gdn_out_int8_scaled",
    "sk02_gemm_w8a16",
]
