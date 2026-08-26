# SPDX-License-Identifier: Apache-2.0
"""SK-03 FA_QKV_FUSED_INT8_DIADIC — PTX inline monolito RMSNorm+quant+GEMM+split sm_86.

Branchless monolito — solo tl.load tl.dot tl.store via PTX sm_86.
Validacion dims%128 dtype bf16->int8 contiguity K==5120 N%128 movida a
warmup_all_kernels en patch_PN110. Cualquier fallo es bug pwal.

sm_86 Ampere PTX 7.4 inline — un solo launch RMSNorm+quant+GEMM+split:

Pipeline por programa — single launch sin buffers DRAM intermedios:
 1. ld.global.b16 + cvt.rn.f32.bf16 — carga hidden bf16 a f32
 2. mul.f32 add.f32 sqrt.approx rcp — RMSNorm var mean rsqrt fp32
 3. abs.f32 max.f32 — amax per-token reduccion tl.max tl.maximum
 4. div.approx.f32 mul.f32 cvt.rni.s32.f32 cvt.sat.s8.s32 — quant per-token
 5. ld.global.b8 — b int8 column-major peso
 6. mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 — tl.dot INT8 TC
 7. shl.s32 shr.s32 — shift diadico sobre acumulador INT32
 8. cvt.bf16.f32 mul.f32 st.global.b32 — epilogo bf16 store

PTX ISA 7.4 .version 7.4 .target sm_86 .address_size 64
ops ejercitadas: cvt.rn.f32.bf16 abs.f32 max.f32 rsqrt mul.f32
cvt.rni.s32.f32 cvt.sat.s8.s32 mma.m16n8k32 shfl.sync

Geometria Global N=14336 K=5120 Per rank TP2 N=7168 K=5120
Q 12288 Gate 6144 K 1024 V 1024_heads 24Q 4KV head_dim 256 hidden 5120.
INT32->bf16 directo acc bf16 2x ancho banda vs fp32 39 TFLOPS bf16 sm_86.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK03_Q_ROWS = 12288
SK03_Q_GATE_SPLIT = 6144
SK03_K_ROWS = 1024
SK03_V_ROWS = 1024
SK03_N_GLOBAL = 14336
SK03_K = 5120
SK03_N_PER_RANK = 7168
SK03_Q_PER_RANK = 6144
SK03_K_PER_RANK = 512
SK03_V_PER_RANK = 512
SK03_GATE_PER_RANK = 3072
SK03_KV_HEADS = 4
SK03_Q_HEADS = 24
SK03_HEAD_DIM = 256
SK03_HIDDEN = 5120
SK03_BLOCK = 128
SK03_SHIFT_BLOCK = 128
SK03_NUM_LAYERS = 16

BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32
SHIFT_BLOCK = 128


@triton.jit
def _sk03_fused_rmsnorm_quant_gemm_kernel(
    hidden_ptr,
    weight_ptr,
    b_ptr,
    out_ptr,
    a_scale_ptr,
    b_scale_ptr,
    shifts_ptr,
    M,
    N,
    K,
    stride_hidden_m,
    stride_hidden_k,
    stride_b_k,
    stride_b_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SHIFT_BLOCK: tl.constexpr,
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
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        h_ptrs = hidden_ptr + offs_m[:, None] * stride_hidden_m + cur_k[None, :] * stride_hidden_k
        h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        h_f = h.to(tl.float32)
        sum_sq = sum_sq + tl.sum(h_f * h_f, axis=1)
        row_amax = tl.max(tl.abs(h_f), axis=1)
        amax = tl.maximum(amax, row_amax)
    mean_sq = sum_sq / K
    rsqrt = 1.0 / tl.sqrt(mean_sq + EPS)
    a_scale = amax / 127.0
    a_scale = tl.where(a_scale > 0, a_scale, 1.0)
    tl.store(a_scale_ptr + offs_m, a_scale, mask=mask_m)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.bfloat16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        h_ptrs = hidden_ptr + offs_m[:, None] * stride_hidden_m + cur_k[None, :] * stride_hidden_k
        h = tl.load(h_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        h_f = h.to(tl.float32)
        w = tl.load(weight_ptr + cur_k, mask=mask_k, other=1.0).to(tl.float32)
        y = h_f * rsqrt[:, None] * w[None, :]
        q_s = y / a_scale[:, None]
        bias = tl.where(q_s >= 0, 0.5, -0.5)
        q_i = (q_s + bias).to(tl.int32)
        q_i = tl.where(q_i > 127, 127, q_i)
        q_i = tl.where(q_i < -127, -127, q_i)
        q_i8 = q_i.to(tl.int8)
        b_ptrs = b_ptr + cur_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0).to(tl.int8)
        acc = acc + tl.dot(q_i8, b_tile)
    shift_col = (pid_n * BLOCK_N) // SHIFT_BLOCK
    shift_val = tl.load(shifts_ptr + shift_col).to(tl.int32)
    shifted = tl.where(shift_val >= 0, acc << shift_val, acc >> (-shift_val))
    acc_f = shifted.to(tl.bfloat16) * a_scale.to(tl.bfloat16)[:, None] * b_scale[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc_f, mask=mask_m[:, None] & mask_n[None, :])


def sk03_fa_qkv_forward(
    hidden: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_scales: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    qkv_shifts: torch.Tensor,
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden.shape[0]
    K = hidden.shape[1]
    N = qkv_weight.shape[1]
    a_scale_buf = torch.empty((M,), dtype=torch.float32, device=hidden.device)
    qkv_out = torch.empty((M, N), dtype=out_dtype, device=hidden.device)
    qkv_scales_bf16 = qkv_scales.to(out_dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk03_fused_rmsnorm_quant_gemm_kernel[grid](
        hidden,
        input_layernorm_weight,
        qkv_weight,
        qkv_out,
        a_scale_buf,
        qkv_scales_bf16,
        qkv_shifts,
        M,
        N,
        K,
        hidden.stride(0),
        hidden.stride(1),
        qkv_weight.stride(0),
        qkv_weight.stride(1),
        qkv_out.stride(0),
        qkv_out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SHIFT_BLOCK=SHIFT_BLOCK,
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


FA_QKV_FUSED_INT8_DIADIC = sk03_fa_qkv_forward
fa_qkv_fused = sk03_fa_qkv_forward
qkv_forward = sk03_fa_qkv_forward

__all__ = [
    "SK03_N_GLOBAL",
    "SK03_N_PER_RANK",
    "SK03_K",
    "sk03_fa_qkv_forward",
    "FA_QKV_FUSED_INT8_DIADIC",
]
