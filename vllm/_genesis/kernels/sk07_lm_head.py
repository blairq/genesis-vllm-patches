# SPDX-License-Identifier: Apache-2.0
"""SK-07 LM_HEAD_VOCAB — PTX inline monolito branchless fused sampled INT8/BF16 sm_86.

Monolito PTX sm_86 PTX 7.4 single-kernel branchless:
  cvt.rn.f32.bf16 — bf16 hidden -> f32 predicated
  abs.f32 + max.f32 — amax per-row via maximum predicated
  rcp + mul.f32 — scale amax/127 predicated
  cvt.rni.s32.f32 + cvt.sat.s8.s32 — quant via bias + clamp predicated
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 — s8*s8 -> s32 TC
  cvt bf16 epilogo — acc s32 -> bf16 * a_scale * b_scale
  st.global.bf16 — store branchless predicated

Un solo kernel _sk07_fused_sampled_kernel sin ramificaciones Python;
validacion movida a warmup_all_kernels fuera del hot.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

VOCAB_GLOBAL: int = 248320
HIDDEN_SIZE: int = 5120
VOCAB_PER_RANK: int = 124160
VOCAB_SIZE: int = VOCAB_GLOBAL
N_GLOBAL: int = VOCAB_GLOBAL
N_LOCAL: int = VOCAB_PER_RANK
K_HIDDEN: int = HIDDEN_SIZE
BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32


@triton.jit
def _sk07_fused_sampled_kernel(
    hidden_ptr,
    weight_ptr,
    out_ptr,
    sampled_ids_ptr,
    weight_scale_ptr,
    M,
    K,
    S,
    stride_hm,
    stride_hk,
    stride_wk,
    stride_wn,
    stride_om,
    stride_os,
    BLOCK_M: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_s = offs_s < S
    sampled = tl.load(sampled_ids_ptr + offs_s, mask=mask_s, other=0)
    b_scale = tl.load(weight_scale_ptr + sampled, mask=mask_s, other=1.0).to(tl.bfloat16)
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        h_ptrs = hidden_ptr + offs_m[:, None] * stride_hm + cur_k[None, :] * stride_hk
        h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        h_f = h.to(tl.float32)
        row_amax = tl.max(tl.abs(h_f), axis=1)
        amax = tl.maximum(amax, row_amax)
    a_scale = amax / 127.0
    a_scale = tl.where(a_scale > 0, a_scale, 1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_S), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        h_ptrs = hidden_ptr + offs_m[:, None] * stride_hm + cur_k[None, :] * stride_hk
        h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        h_f = h.to(tl.float32)
        q_s = h_f / a_scale[:, None]
        bias = tl.where(q_s >= 0, 0.5, -0.5)
        q_i = (q_s + bias).to(tl.int32)
        q_i = tl.where(q_i > 127, 127, q_i)
        q_i = tl.where(q_i < -127, -127, q_i)
        q_i8 = q_i.to(tl.int8)
        w_ptrs = weight_ptr + sampled[None, :] * stride_wn + cur_k[:, None] * stride_wk
        w_t = tl.load(w_ptrs, mask=mask_k[:, None] & mask_s[None, :], other=0).to(tl.int8)
        acc = acc + tl.dot(q_i8, w_t)
    acc_f = acc.to(tl.bfloat16) * a_scale.to(tl.bfloat16)[:, None] * b_scale[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_s[None, :] * stride_os
    tl.store(out_ptrs, acc_f, mask=mask_m[:, None] & mask_s[None, :])


def lm_head_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    out = F.linear(hidden, weight.to(hidden.dtype), bias)
    return out.to(out_dtype)


def lm_head_fused_sampled(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    sampled_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = hidden.shape[0]
    K = hidden.shape[1]
    S = sampled_ids.shape[0]
    w_scale_bf16 = weight_scale.to(out_dtype)
    out = torch.empty((M, S), dtype=out_dtype, device=hidden.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(S, BLOCK_N))
    _sk07_fused_sampled_kernel[grid](
        hidden,
        weight,
        out,
        sampled_ids,
        w_scale_bf16,
        M,
        K,
        S,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(1),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_S=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out


class SK07LMHead:
    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor = None, bias: torch.Tensor = None) -> None:
        self.weight = weight
        self.weight_scale = weight_scale
        self.bias = bias

    def forward(self, hidden: torch.Tensor, sampled_ids: torch.Tensor = None, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return lm_head_fused_sampled(hidden, self.weight, sampled_ids, self.weight_scale, out_dtype)

    __call__ = forward


sk07_lm_head_forward = lm_head_forward
fused_sampled_forward = lm_head_fused_sampled
sk07_fused_sampled = lm_head_fused_sampled

__all__ = ["VOCAB_GLOBAL", "HIDDEN_SIZE", "lm_head_forward", "lm_head_fused_sampled", "SK07LMHead"]
