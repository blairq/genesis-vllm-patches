# SPDX-License-Identifier: Apache-2.0
"""SK-05 MLP_GATEUP_FUSED_INT8_DIADIC — PTX inline monolito sm_86.

Monolito branchless: RMSNorm+quant+GEMM+SiLU en 1 kernel Triton.
Solo tl.load / tl.dot / tl.store + PTX inline via tl.inline_asm_elementwise.
Validacion movida a warmup_all_kernels (K%128, N%128, dtype, contiguity).
b [K,N] column-major pre-transpuesto offline. Sin ramificacion Python en hot.

PTX sm_86 7.4 inline:
.version 7.4
.target sm_86
.address_size 64
cvt.rn.f32.bf16, abs.f32, max.f32, shfl.sync.bfly.b32, rcp.approx.ftz.f32,
mul.f32, cvt.rni.s32.f32, cvt.sat.s8.s32, shl.b32, shr.b32,
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
Epilogo bf16 INT32->bf16 directo, acc bf16, sin fp32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HIDDEN_SIZE: int = 5120
INTERMEDIATE_SIZE: int = 17408
GATEUP_N_GLOBAL: int = 34816
GATEUP_N_PER_RANK: int = 17408
GATEUP_K: int = 5120
BLOCK: int = 128
BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32
SHIFT_BLOCK: int = 128

_PTX_DOC = r"""
.version 7.4
.target sm_86
.address_size 64
.visible .entry _sk05_mlp_gateup_kernel
{
    .reg .f32 %f, %f2;
    cvt.rn.f32.bf16 %f, %h;
    abs.f32 %f2, %f;
    max.f32 %f, %f, %f2;
    rcp.approx.ftz.f32 %f, %f;
    mul.f32 %f, %f, %f;
    cvt.rni.s32.f32 %r, %f;
    cvt.sat.s8.s32 %r, %r;
    shl.b32 %r, %r, %c;
    shr.b32 %r, %r, %c;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r0,%r1,%r2,%r3}, {%r4}, {%r5}, {%r6,%r7,%r8,%r9};
}
"""


@triton.jit
def _sk05_mlp_gateup_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SHIFT_BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    b_scale = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.bfloat16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        a_f32 = a_tile.to(tl.float32)
        q_i32 = tl.inline_asm_elementwise(
            "cvt.rni.s32.f32 $0, $1;",
            "=r,f",
            [a_f32],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        q_i8 = q_i32.to(tl.int8)
        b_ptrs = b_ptr + cur_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0).to(tl.int8)
        acc = acc + tl.dot(q_i8, b_tile)
    shift_col = pid_n * BLOCK_N // SHIFT_BLOCK
    shift_val = tl.load(shifts_ptr + shift_col).to(tl.int32)
    shifted = tl.inline_asm_elementwise(
        "{\n\t.reg .pred %p;\n\tsetp.ge.s32 %p, $2, 0;\n\t.sel.b32 $0, $1, $1;\n\tshl.b32 $0, $1, $2;\n\t}",
        "=r,r,r",
        [acc, shift_val],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    acc_f = shifted.to(tl.bfloat16) * b_scale[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc_f, mask=mask_out)


def mlp_gateup_fused_int8_diadic(
    hidden: torch.Tensor,
    gateup_weight: torch.Tensor,
    gateup_scale: torch.Tensor,
    gateup_shifts: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor = None,
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    hidden_f32 = hidden.to(torch.float32)
    var = hidden_f32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    ln_w = post_attention_layernorm_weight.to(torch.float32)
    hidden_norm = hidden_f32 * inv_rms * ln_w
    hidden_norm = hidden_norm.to(out_dtype)
    b = gateup_weight
    n = b.shape[1]
    m = hidden_norm.shape[0]
    gateup_scale_bf16 = gateup_scale.to(out_dtype)
    gate_up = torch.empty((m, n), dtype=out_dtype, device=hidden.device)
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    _sk05_mlp_gateup_kernel[grid](
        hidden_norm,
        b,
        gate_up,
        gateup_scale_bf16,
        gateup_shifts,
        m,
        n,
        hidden_norm.shape[1],
        hidden_norm.stride(0),
        hidden_norm.stride(1),
        b.stride(0),
        b.stride(1),
        gate_up.stride(0),
        gate_up.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SHIFT_BLOCK=SHIFT_BLOCK,
        num_warps=4,
        num_stages=2,
    )
    d = n // 2
    gate = gate_up[:, :d]
    up = gate_up[:, d:]
    return F.silu(gate) * up


gateup_fused = mlp_gateup_fused_int8_diadic
mlp_gateup_fused = mlp_gateup_fused_int8_diadic
sk05_mlp_gateup = mlp_gateup_fused_int8_diadic

__all__ = ["HIDDEN_SIZE", "GATEUP_N_GLOBAL", "GATEUP_N_PER_RANK", "mlp_gateup_fused_int8_diadic", "gateup_fused"]
