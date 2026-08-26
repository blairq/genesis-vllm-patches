# SPDX-License-Identifier: Apache-2.0
"""SK-10 MTP_DRAFT PTX inline monolito — branchless fused quant+GEMM sm_86.

Monolito: 1 solo kernel @triton.jit con tl.load / tl.dot / tl.store.
Branchless: solo tl.load, tl.store, tl.dot, tl.where, tl.max, tl.abs.
Validacion movida a warmup_all_kernels en patch_PN110. Cualquier error
es bug del pwal (draft no validado). Espejo de SK-05 fused_quant_gemm
single launch sin buffers DRAM intermedios.

Geometria MTP draft (mirror target):
  qkv 14336x5120  gateup 34816x5120  down 5120x17408  o 5120x6144  fc 5120x10240
Por rank TP2: qkv 7168x5120  gateup 17408x5120  down 5120x8704
Grupo 128: todas dims multiplo 128, TP divide sin resto, 0 padding.
sm_86 39 TFLOPS bf16 epilogo directo INT32->bf16 sin fp32 intermedio.

Pipeline por programa [BLOCK_M, BLOCK_N] — single launch PTX 7.4 sm_86:
  1. cvt.rn.f32.bf16 — bf16 load via tl.load -> tl.float32
  2. abs.f32 + max.f32 — amax por fila via tl.max(tl.abs, axis=1) reduccion global K
  3. rcp.approx + mul.f32 — scale = amax / 127 via tl.where para cero
  4. mul.f32 + cvt.rni.s32.f32 + cvt.sat.s8.s32 — quant round clamp via tl.where
  5. mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 — tl.dot int8 -> int32 TC
  6. cvt.bf16 epilogo — acc.to(bf16) * a_scale * b_scale -> tl.store

PTX doc sm_86:
  .version 7.4
  .target sm_86
  cvt.rn.f32.bf16 %f, %h;
  abs.f32 %f, %f;
  max.f32 %f, %f, %f;
  mul.f32 %f, %f, %f;
  cvt.rni.s32.f32 %r, %f;
  cvt.sat.s8.s32 %r, %r;
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%r0,%r1,%r2,%r3}, {%r4}, {%r5}, {%r6,%r7,%r8,%r9};
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32
BLOCK_SIZE: tuple[int, int] = (128, 128)
MTP_HIDDEN: int = 5120
MTP_INTERMEDIATE: int = 17408
GROUP_SIZE: int = 128


@triton.jit
def _sk10_mtp_draft_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak
        mask_a = mask_m[:, None] & mask_k[None, :]
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
        a_f32 = a_tile.to(tl.float32)
        row_amax = tl.max(tl.abs(a_f32), axis=1)
        amax = tl.maximum(amax, row_amax)
    scale = amax / 127.0
    scale = tl.where(scale > 0, scale, 1.0)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.bfloat16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak
        mask_a = mask_m[:, None] & mask_k[None, :]
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
        a_f32 = a_tile.to(tl.float32)
        q_scaled = a_f32 / scale[:, None]
        bias = tl.where(q_scaled >= 0, 0.5, -0.5)
        q_int = (q_scaled + bias).to(tl.int32)
        q_int = tl.where(q_int > 127, 127, q_int)
        q_int = tl.where(q_int < -127, -127, q_int)
        q_i8 = q_int.to(tl.int8)
        b_ptrs = b_ptr + cur_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = mask_k[:, None] & mask_n[None, :]
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0).to(tl.int8)
        acc = acc + tl.dot(q_i8, b_tile)
    acc_f = acc.to(tl.bfloat16) * scale.to(tl.bfloat16)[:, None] * b_scale[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc_f, mask=mask_out)


def mtp_draft_fused_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    b_scale_bf16 = b_scale.to(torch.bfloat16)
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk10_mtp_draft_kernel[grid](
        a,
        b,
        out,
        b_scale_bf16,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out


def mtp_draft_linear(
    x: torch.Tensor,
    b_col: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return mtp_draft_fused_gemm(x, b_col, b_scales, out_dtype)


__all__ = ["mtp_draft_fused_gemm", "mtp_draft_linear", "BLOCK_SIZE", "BLOCK_M", "BLOCK_N", "BLOCK_K"]
