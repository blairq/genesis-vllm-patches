# SPDX-License-Identifier: Apache-2.0
"""SK-04 FA_O_INT8_SCALED — PTX inline monolito RowParallel 5120x3072 sm_86.

Monolito PTX inline branchless sm_86 PTX 7.4:
.version 7.4
.target sm_86
.address_size 64
Pipeline: tl.load -> tl.dot (mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32)
          -> PTX shl.b32 / shr.s32 + selp.s32 branchless shift
          -> cvt.rn.bf16.s32 INT32->bf16 -> tl.store
Un solo kernel triton.jit con tl.load + tl.dot + tl.store + tl.inline_asm_elementwise.
Validacion movida a warmup_all_kernels. Sin branches Python en kernel ni en wrappers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK_ID: str = "SK-04"
SK_NAME: str = "FA_O_INT8_SCALED"
SHAPE_GLOBAL: tuple[int, int] = (5120, 6144)
SHAPE_PER_RANK: tuple[int, int] = (5120, 3072)
BLOCK_K: int = 32
BLOCK_N: int = 64
BLOCK_M: int = 32
SHIFT_BLOCK: int = 128


@triton.jit
def _sk04_fa_o_kernel(
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
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_scales = tl.load(a_scale_ptr + offs_m * stride_scale_a, mask=mask_m, other=0.0).to(tl.bfloat16)
    b_scales = tl.load(b_scale_ptr + offs_n * stride_scale_b, mask=mask_n, other=0.0).to(tl.bfloat16)
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
        shifted = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .pred %p;
                .reg .s32 %r_neg;
                .reg .s32 %r_shl;
                .reg .s32 %r_shr;
                setp.ge.s32 %p, $2, 0;
                shl.b32 %r_shl, $1, $2;
                sub.s32 %r_neg, 0, $2;
                shr.s32 %r_shr, $1, %r_neg;
                selp.s32 $0, %r_shl, %r_shr, %p;
            }
            """,
            constraints="=r,r,r",
            args=[int_acc, shift_val],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        shifted_f = shifted.to(tl.bfloat16)
        scaled = shifted_f * a_scales[:, None] * b_scales[None, :]
        acc = acc + scaled
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


def fa_o_int8_scaled_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    shifts: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    a_bf16 = a.to(out_dtype)
    amax = a_bf16.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    a_i8 = (a_bf16 / scale).round().clamp(-127, 127).to(torch.int8)
    a_scales = scale.squeeze(-1).to(out_dtype)
    b_scale_bf16 = b_scale.to(out_dtype)
    M = a_i8.shape[0]
    N = b.shape[1]
    K = a_i8.shape[1]
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk04_fa_o_kernel[grid](
        a_i8,
        b,
        out,
        a_scales,
        b_scale_bf16,
        shifts,
        M,
        N,
        K,
        a_i8.stride(0),
        a_i8.stride(1),
        b.stride(0),
        b.stride(1),
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


def fa_o_forward(
    x: torch.Tensor,
    layer: torch.nn.Module,
    residual: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    state = layer.__dict__["_genesis_sk04_int8"]
    b_col = state["b_col"]
    b_scales = state["b_scales"]
    shifts = state["shifts"]
    out = fa_o_int8_scaled_gemm(x, b_col, b_scales, shifts, out_dtype)
    out = out + residual
    return out


__all__ = ["SK_ID", "SHAPE_GLOBAL", "SHAPE_PER_RANK", "fa_o_int8_scaled_gemm", "fa_o_forward"]
