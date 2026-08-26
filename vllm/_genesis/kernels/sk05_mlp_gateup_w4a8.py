# SPDX-License-Identifier: Apache-2.0
"""SK-05 W4A8 MLP_GATEUP — branchless Triton puro para zipperlein W4A8.

Branchless: solo tl.load / tl.dot / tl.where / tl.store / tl.max.
Toda validacion (dims%128, dtype, contiguity, K%GROUP_SIZE==0, N%128==0,
packing INT4 low nibble, escalas contiguas, strides) se hace UNA vez por
capa en pwal (warmup_all_kernels), nunca por forward.
Cuando este kernel falla, indica bug del pwal (capa no validada), nunca
condicion del kernel. Sin ramificacion Python en hot path.

Geometria: Global N=34816 (2x17408)=272*128, K=5120=40*128. Per-rank
TP=2: N=17408, K=5120. W4A8: peso INT4 GPTQ sym g128 empaquetado
low nibble UINT4 (0..15, zero 8, valor -8..7) + escala per-grupo fp32
[G,N] donde G=K//128=40. Activacion INT8 per-token quantizada dinamica
dentro del kernel tras RMSNorm. Compute INT8 TC mma.m16n8k32.s8.

Contrato hot (asume validado):
    hidden       [M,K] bf16/fp16/fp32 — row-major
    w_packed     [K,N] uint8 — INT4 low nibble por byte, fila K col N
    w_scale      [G,N] fp32 — per-grupo, contiguo, fp32, G=40
    ln_weight    [K] bf16/fp32 — RMSNorm weight post_attention
    out          [M,N] fp32/bf16 — salida gate_up sin SiLU

Triton: solo tl.load / tl.dot / tl.where / tl.store / tl.max,
sin ramificacion Python. Unpack branchless via (b & 15) - 8 -> int8
con clamp via tl.where. Dot INT8 -> INT32, epilogo fp32 a_scale * w_scale
por grupo, acumulacion fp32. RMSNorm + quant fusionados.
Validacion movida a arranque, hot path 100% branchless.
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
GATEUP_GROUPS: int = 40
GROUP_SIZE: int = 128
BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32
BLOCK: int = 128


@triton.jit
def _sk05_mlp_gateup_w4a8_kernel(
    hidden_ptr,
    w_packed_ptr,
    w_scale_ptr,
    ln_weight_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_hidden_m,
    stride_hidden_k,
    stride_wpack_k,
    stride_wpack_n,
    stride_wscale_g,
    stride_wscale_n,
    stride_ln,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        h_ptrs = hidden_ptr + offs_m[:, None] * stride_hidden_m + cur_k[None, :] * stride_hidden_k
        h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        h_f = h.to(tl.float32)
        sum_sq = sum_sq + tl.sum(h_f * h_f, axis=1)
    mean_sq = sum_sq / K
    rsqrt = 1.0 / tl.sqrt(mean_sq + EPS)
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        h_ptrs = hidden_ptr + offs_m[:, None] * stride_hidden_m + cur_k[None, :] * stride_hidden_k
        h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        h_f = h.to(tl.float32)
        w_ln = tl.load(ln_weight_ptr + cur_k * stride_ln, mask=mask_k, other=1.0).to(tl.float32)
        h_norm = h_f * rsqrt[:, None] * w_ln[None, :]
        row_amax = tl.max(tl.abs(h_norm), axis=1)
        amax = tl.maximum(amax, row_amax)
    a_scale = amax / 127.0
    a_scale = tl.where(a_scale > 0, a_scale, 1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    num_groups = K // GROUP_SIZE
    for g in range(num_groups):
        w_scale = tl.load(w_scale_ptr + g * stride_wscale_g + offs_n * stride_wscale_n, mask=mask_n, other=1.0).to(tl.bfloat16)
        int_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        k_base = g * GROUP_SIZE
        for sub in range(GROUP_SIZE // BLOCK_K):
            k_offs = k_base + sub * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = k_offs < K
            h_ptrs = hidden_ptr + offs_m[:, None] * stride_hidden_m + k_offs[None, :] * stride_hidden_k
            h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            h_f = h.to(tl.float32)
            w_ln = tl.load(ln_weight_ptr + k_offs * stride_ln, mask=mask_k, other=1.0).to(tl.float32)
            h_norm = h_f * rsqrt[:, None] * w_ln[None, :]
            q_s = h_norm / a_scale[:, None]
            bias = tl.where(q_s >= 0, 0.5, -0.5)
            q_i = (q_s + bias).to(tl.int32)
            q_i = tl.where(q_i > 127, 127, q_i)
            q_i = tl.where(q_i < -127, -127, q_i)
            q_i8 = q_i.to(tl.int8)
            w_ptrs = w_packed_ptr + k_offs[:, None] * stride_wpack_k + offs_n[None, :] * stride_wpack_n
            w_byte = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0).to(tl.int32)
            w_low = w_byte & 15
            w_val = w_low - 8
            w_val = tl.where(w_val > 7, 7, w_val)
            w_val = tl.where(w_val < -8, -8, w_val)
            w_i8 = w_val.to(tl.int8)
            int_acc = int_acc + tl.dot(q_i8, w_i8)
        acc_f = int_acc.to(tl.bfloat16) * a_scale.to(tl.bfloat16)[:, None] * w_scale[None, :]
        acc = acc + acc_f
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


@triton.jit
def _sk05_mlp_gateup_w4a8_silu_kernel(
    gateup_ptr,
    out_ptr,
    M,
    N,
    stride_gateup_m,
    stride_gateup_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_gate = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_up = offs_gate + (N // 2)
    mask_m = offs_m < M
    mask_gate = offs_gate < (N // 2)
    mask_up = offs_up < N
    gate_ptrs = gateup_ptr + offs_m[:, None] * stride_gateup_m + offs_gate[None, :] * stride_gateup_n
    up_ptrs = gateup_ptr + offs_m[:, None] * stride_gateup_m + offs_up[None, :] * stride_gateup_n
    gate = tl.load(gate_ptrs, mask=mask_m[:, None] & mask_gate[None, :], other=0.0).to(tl.bfloat16)
    up = tl.load(up_ptrs, mask=mask_m[:, None] & mask_up[None, :], other=0.0).to(tl.bfloat16)
    # sm_86: sigmoid en fp32 para precision, convertir resultado a bf16
    silu = gate.to(tl.float32) * tl.sigmoid(gate.to(tl.float32))
    silu = silu.to(tl.bfloat16)
    out = silu * up
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_gate[None, :] * stride_out_n
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_gate[None, :])


def mlp_gateup_w4a8(
    hidden: torch.Tensor,
    gateup_weight_packed: torch.Tensor,
    gateup_scale: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = hidden.shape[0]
    K = hidden.shape[1]
    N = gateup_weight_packed.shape[1]
    w_scale_bf16 = gateup_scale.to(out_dtype) if gateup_scale.dtype != out_dtype else gateup_scale
    gate_up = torch.empty((M, N), dtype=out_dtype, device=hidden.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk05_mlp_gateup_w4a8_kernel[grid](
        hidden,
        gateup_weight_packed,
        w_scale_bf16,
        post_attention_layernorm_weight,
        gate_up,
        M,
        N,
        K,
        hidden.stride(0),
        hidden.stride(1),
        gateup_weight_packed.stride(0),
        gateup_weight_packed.stride(1),
        w_scale_bf16.stride(0),
        w_scale_bf16.stride(1),
        1,
        gate_up.stride(0),
        gate_up.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE=GROUP_SIZE,
        EPS=eps,
        num_warps=4,
        num_stages=2,
    )
    d = N // 2
    gate = gate_up[:, :d]
    up = gate_up[:, d:]
    return F.silu(gate) * up


def mlp_gateup_w4a8_silu_fused(
    gateup: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = gateup.shape[0]
    N = gateup.shape[1]
    d = N // 2
    out = torch.empty((M, d), dtype=out_dtype, device=gateup.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(d, BLOCK_N))
    _sk05_mlp_gateup_w4a8_silu_kernel[grid](
        gateup,
        out,
        M,
        N,
        gateup.stride(0),
        gateup.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out


gateup_w4a8 = mlp_gateup_w4a8
mlp_gateup_fused_w4a8 = mlp_gateup_w4a8
sk05_mlp_gateup_w4a8 = mlp_gateup_w4a8
SK05_MLP_GATEUP_W4A8 = mlp_gateup_w4a8

__all__ = [
    "HIDDEN_SIZE",
    "INTERMEDIATE_SIZE",
    "GATEUP_N_GLOBAL",
    "GATEUP_N_PER_RANK",
    "GATEUP_K",
    "GATEUP_GROUPS",
    "GROUP_SIZE",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "mlp_gateup_w4a8",
    "mlp_gateup_w4a8_silu_fused",
    "gateup_w4a8",
    "mlp_gateup_fused_w4a8",
    "sk05_mlp_gateup_w4a8",
    "SK05_MLP_GATEUP_W4A8",
]
