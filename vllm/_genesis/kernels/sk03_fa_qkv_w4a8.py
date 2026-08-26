# SPDX-License-Identifier: Apache-2.0
"""SK-03 FA_QKV_W4A8 — branchless monolito RMSNorm+quant+W4unpack+GEMM+split sm_86.

W4A8: peso INT4 empaquetado low nibble UINT4 por byte + escalas por grupo
128, activacion INT8 per-token dinamica. Solo tl.load / tl.dot /
tl.where / tl.store. Validacion dims%128, dtype, contiguity, K==5120,
N%128==0 movida a warmup_all_kernels. Fallo es bug del pwal.
Geometria: Global N=14336 (12288 Q+gate +1024 K+1024 V), K=5120.
Per rank TP2: N=7168 (3072 Q+3072 gate+512 K+512 V), K=5120.
B es uint8 [K,N] low nibble UINT4 (valor 0..15, zero 8), w_scale
[G,N] bf16 per grupo G=K//128, a_scale per token.
GEMM INT8 TC: tl.dot(q_i8, w_i8) -> INT32, luego escala bf16 sm_86.
INT32→bf16 directo, acc bf16, sin fp32 en epilogo. 1 launch por capa Full.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK03_W4A8_Q_ROWS: int = 12288
SK03_W4A8_K_ROWS: int = 1024
SK03_W4A8_V_ROWS: int = 1024
SK03_W4A8_N_GLOBAL: int = 14336
SK03_W4A8_K: int = 5120
SK03_W4A8_N_PER_RANK: int = 7168
SK03_W4A8_Q_PER_RANK: int = 3072
SK03_W4A8_GATE_PER_RANK: int = 3072
SK03_W4A8_K_PER_RANK: int = 512
SK03_W4A8_V_PER_RANK: int = 512
SK03_W4A8_GROUP: int = 128
SK03_W4A8_GROUPS: int = 40
SK03_W4A8_NUM_LAYERS: int = 16
SK03_W4A8_BLOCK: int = 128

BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32
GROUP_SIZE: int = 128


@triton.jit
def _sk03_fa_qkv_w4a8_kernel(
    hidden_ptr,
    w_packed_ptr,
    w_scale_ptr,
    ln_weight_ptr,
    qkv_ptr,
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
    stride_qkv_m,
    stride_qkv_n,
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
            w_low = w_byte & 0xF
            w_val = w_low - 8
            w_val = tl.where(w_val > 7, 7, w_val)
            w_val = tl.where(w_val < -8, -8, w_val)
            w_i8 = w_val.to(tl.int8)
            int_acc = int_acc + tl.dot(q_i8, w_i8)
        acc_f = int_acc.to(tl.bfloat16) * a_scale.to(tl.bfloat16)[:, None] * w_scale[None, :]
        acc = acc + acc_f
    out_ptrs = qkv_ptr + offs_m[:, None] * stride_qkv_m + offs_n[None, :] * stride_qkv_n
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def sk03_fa_qkv_w4a8_forward(
    hidden: torch.Tensor,
    qweight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden.shape[0]
    K = hidden.shape[1]
    N = qweight_packed.shape[1]
    n_q = 3072
    n_gate = 3072
    n_k = 512
    n_v = 512
    qkv_out = torch.empty((M, N), dtype=out_dtype, device=hidden.device)
    w_scale_bf16 = weight_scale.to(out_dtype) if weight_scale.dtype != out_dtype else weight_scale
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk03_fa_qkv_w4a8_kernel[grid](
        hidden,
        qweight_packed,
        w_scale_bf16,
        input_layernorm_weight,
        qkv_out,
        M,
        N,
        K,
        hidden.stride(0),
        hidden.stride(1),
        qweight_packed.stride(0),
        qweight_packed.stride(1),
        w_scale_bf16.stride(0),
        w_scale_bf16.stride(1),
        1,
        qkv_out.stride(0),
        qkv_out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE=GROUP_SIZE,
        EPS=eps,
        num_warps=4,
        num_stages=2,
    )
    qkv = qkv_out
    q = qkv[:, 0:3072]
    gate = qkv[:, 3072:6144]
    k_t = qkv[:, 6144:6656]
    v_t = qkv[:, 6656:7168]
    return q, gate, k_t, v_t, qkv


FA_QKV_W4A8 = sk03_fa_qkv_w4a8_forward
fa_qkv_w4a8 = sk03_fa_qkv_w4a8_forward
sk03_w4a8_forward = sk03_fa_qkv_w4a8_forward
qkv_w4a8_forward = sk03_fa_qkv_w4a8_forward

__all__ = [
    "SK03_W4A8_N_GLOBAL",
    "SK03_W4A8_N_PER_RANK",
    "SK03_W4A8_K",
    "SK03_W4A8_GROUP",
    "sk03_fa_qkv_w4a8_forward",
    "FA_QKV_W4A8",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_SIZE",
]
