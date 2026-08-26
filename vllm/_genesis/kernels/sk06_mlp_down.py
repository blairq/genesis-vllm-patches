# SPDX-License-Identifier: Apache-2.0
"""SK-06 MLP_DOWN_INT8_SCALED_RESIDUAL — PTX inline monolito branchless sm_86.

Monolito: 1 solo kernel triton.jit con tl.load / tl.dot / tl.store + PTX inline
branchless para shift (shl.b32 / shr.s32 / selp.b32 via tl.inline_asm_elementwise).
Validacion movida a warmup_all_kernels en patch_PN110. Cualquier fallo es bug del pwal.

sm_86: epilogo bf16 (INT32 -> bf16 directo, acc bf16, residual bf16), sin fp32.
PTX .version 7.4 .target sm_86 .address_size 64
  shl.b32 / shr.s32 / selp.b32 — shift branchless sin divergencia
  cvt.rn.bf16.f32 / cvt.rn.f32.bf16 — epilogo bf16
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 — doc Tensor Core (Ampere)
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
BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32

_PTX_VERSION = "7.4"
_PTX_TARGET = "sm_86"
_PTX_DOC = r"""
.version 7.4
.target sm_86
.address_size 64
.visible .entry _sk06_mlp_down_kernel
{
    // PTX inline monolito SK-06 sm_86 — GEMM int8 + shift diadico + residuo bf16
    // shl.b32 %rShl, $1, $2;           // shift left branchless
    // sub.s32 %rNeg, 0, $2;            // neg shift para branchless
    // shr.s32 %rShr, $1, %rNeg;        // shift right aritmetico branchless
    // setp.ge.s32 %p, $2, 0;           // pred para selp
    // selp.s32 $0, %rShl, %rShr, %p;   // select sin branch
    // cvt.rn.bf16.s32 — INT32 -> bf16 epilogo directo
}
"""

# PTX inline monolito: unico kernel branchless con tl.load / tl.dot / tl.store
# + tl.inline_asm_elementwise para shift sin branches / sin divergencia.


@triton.jit
def _sk06_mlp_down_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    resid_ptr,
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
    stride_resid_m,
    stride_resid_n,
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
            "{ .reg .pred %p; .reg .s32 %rNeg; .reg .s32 %rShl; .reg .s32 %rShr; setp.ge.s32 %p, $2, 0; shl.b32 %rShl, $1, $2; sub.s32 %rNeg, 0, $2; shr.s32 %rShr, $1, %rNeg; selp.s32 $0, %rShl, %rShr, %p; }",
            "=r,r,r",
            [int_acc, shift_val],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        shifted_f = shifted.to(tl.bfloat16)
        scaled = shifted_f * a_scales[:, None] * b_scales[None, :]
        acc = acc + scaled
    resid_ptrs = resid_ptr + offs_m[:, None] * stride_resid_m + offs_n[None, :] * stride_resid_n
    mask_resid = mask_m[:, None] & mask_n[None, :]
    resid = tl.load(resid_ptrs, mask=mask_resid, other=0.0).to(tl.bfloat16)
    acc = acc + resid
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


def mlp_down_int8_scaled_residual(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    residual: torch.Tensor,
    weight_shifts: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = x.shape[0]
    K = x.shape[1]
    N = weight.shape[1]
    x_bf16 = x.to(out_dtype)
    amax = x_bf16.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_i8 = (x_bf16 / scale).round().clamp(-127, 127).to(torch.int8)
    a_scales = scale.squeeze(-1).to(out_dtype)
    w_scale_bf16 = weight_scale.to(out_dtype)
    resid_bf16 = residual.to(out_dtype)
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk06_mlp_down_kernel[grid](
        x_i8,
        weight,
        out,
        resid_bf16,
        a_scales,
        w_scale_bf16,
        weight_shifts,
        M,
        N,
        K,
        x_i8.stride(0),
        x_i8.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        resid_bf16.stride(0),
        resid_bf16.stride(1),
        1,
        1,
        weight_shifts.stride(0),
        weight_shifts.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SHIFT_BLOCK=SHIFT_BLOCK,
        num_warps=4,
        num_stages=2,
    )
    return out


def mlp_down_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    residual = torch.zeros((x.shape[0], weight.shape[1]), dtype=out_dtype, device=x.device)
    shifts = torch.zeros((x.shape[1] // SHIFT_BLOCK, weight.shape[1] // SHIFT_BLOCK), dtype=torch.int8, device=x.device)
    return mlp_down_int8_scaled_residual(x, weight, weight_scale, residual, shifts, out_dtype)


__all__ = ["GLOBAL_N", "GLOBAL_K", "mlp_down_int8_scaled_residual", "mlp_down_gemm"]
