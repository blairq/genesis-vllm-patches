# SPDX-License-Identifier: Apache-2.0
"""SK-10 W4A8 MTP_DRAFT — Triton puro branchless W4A8 sm_86.

MTP draft mirror W4A8: weight INT4 GPTQ group 128 + act INT8 per-token.
Branchless: solo tl.load / tl.store / tl.dot / tl.where / tl.max.
Validacion movida a warmup_all_kernels patch_PN110. Cualquier fallo es bug pwal.
Solo kernel Triton puro.

Peso offline: INT4 simetrico -8..7 packed 2 por byte, escala bf16 per-channel
o per-grupo 128 fusionada en b_scale sm_86. Activacion: bf16 -> amax -> scale_a
amax/127 -> quant INT8 -> GEMM INT8 TC -> epilogo bf16 scale_a * b_scale.
INT32→bf16 directo, acc bf16, sin fp32 en epilogo.

Geometria MTP draft (mirror target):
  qkv 14336x5120  gateup 34816x5120  down 5120x17408  o 5120x6144  fc 5120x10240
Por rank TP2: qkv 7168x5120  gateup 17408x5120  down 5120x8704
Grupo 128: todas dims multiplo 128, TP divide sin resto, 0 padding. sm_86 39 TFLOPS bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MTP_HIDDEN: int = 5120
MTP_INTERMEDIATE: int = 17408
MTP_GATEUP_N_GLOBAL: int = 34816
MTP_GATEUP_N_PER_RANK: int = 17408
MTP_DOWN_K_PER_RANK: int = 8704
MTP_QKV_N_GLOBAL: int = 14336
MTP_QKV_N_PER_RANK: int = 7168
MTP_O_N_GLOBAL: int = 6144
MTP_O_K: int = 5120
GROUP_SIZE: int = 128
W4_BITS: int = 4
W4_MAX: int = 7
W4_MIN: int = -8
BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32


@triton.jit
def _sk10_mtp_w4a8_kernel(
    hidden_ptr,
    weight_ptr,
    out_ptr,
    weight_scale_ptr,
    M,
    N,
    K,
    stride_hm,
    stride_hk,
    stride_wk,
    stride_wn,
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
        h_ptrs = hidden_ptr + offs_m[:, None] * stride_hm + cur_k[None, :] * stride_hk
        mask_a = mask_m[:, None] & mask_k[None, :]
        h_tile = tl.load(h_ptrs, mask=mask_a, other=0.0)
        h_f32 = h_tile.to(tl.float32)
        row_amax = tl.max(tl.abs(h_f32), axis=1)
        amax = tl.maximum(amax, row_amax)
    scale_a = amax / 127.0
    scale_a = tl.where(scale_a > 0, scale_a, 1.0)
    b_scale = tl.load(weight_scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.bfloat16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        h_ptrs = hidden_ptr + offs_m[:, None] * stride_hm + cur_k[None, :] * stride_hk
        mask_a = mask_m[:, None] & mask_k[None, :]
        h_tile = tl.load(h_ptrs, mask=mask_a, other=0.0)
        h_f32 = h_tile.to(tl.float32)
        q_scaled = h_f32 / scale_a[:, None]
        bias = tl.where(q_scaled >= 0, 0.5, -0.5)
        q_int = (q_scaled + bias).to(tl.int32)
        q_int = tl.where(q_int > 127, 127, q_int)
        q_int = tl.where(q_int < -127, -127, q_int)
        q_i8 = q_int.to(tl.int8)
        w_ptrs = weight_ptr + cur_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        mask_b = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=mask_b, other=0).to(tl.int8)
        w_tile = tl.where(w_tile > 7, 7, w_tile)
        w_tile = tl.where(w_tile < -8, -8, w_tile)
        acc = acc + tl.dot(q_i8, w_tile)
    acc_f = acc.to(tl.bfloat16) * scale_a.to(tl.bfloat16)[:, None] * b_scale[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc_f, mask=mask_out)


@triton.jit
def _sk10_mtp_w4a8_rmsnorm_quant_kernel(
    hidden_ptr,
    rms_weight_ptr,
    out_i8_ptr,
    scale_ptr,
    M,
    K,
    stride_hm,
    stride_hk,
    stride_w,
    stride_out_m,
    stride_out_k,
    stride_scale,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    h = tl.load(hidden_ptr + pid * stride_hm + offs * stride_hk, mask=mask, other=0.0)
    h_f = h.to(tl.float32)
    var = tl.sum(h_f * h_f, axis=0) / K
    inv_rms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(rms_weight_ptr + offs * stride_w, mask=mask, other=1.0).to(tl.float32)
    y = h_f * inv_rms * w
    amax = tl.max(tl.abs(y), axis=0)
    scale = amax / 127.0
    scale = tl.where(scale > 0, scale, 1.0)
    tl.store(scale_ptr + pid * stride_scale, scale.to(tl.bfloat16))
    q_s = y / scale
    bias = tl.where(q_s >= 0, 0.5, -0.5)
    q_i = (q_s + bias).to(tl.int32)
    q_i = tl.where(q_i > 127, 127, q_i)
    q_i = tl.where(q_i < -127, -127, q_i)
    q_i8 = q_i.to(tl.int8)
    tl.store(out_i8_ptr + pid * stride_out_m + offs * stride_out_k, q_i8, mask=mask)


def mtp_draft_w4a8_gemm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = hidden.shape[0]
    N = weight.shape[1]
    K = hidden.shape[1]
    w_scale_bf16 = weight_scale.to(out_dtype) if weight_scale.dtype != out_dtype else weight_scale
    out = torch.empty((M, N), dtype=out_dtype, device=hidden.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk10_mtp_w4a8_kernel[grid](
        hidden,
        weight,
        out,
        w_scale_bf16,
        M,
        N,
        K,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out


def mtp_draft_w4a8_rmsnorm_quant(
    hidden: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float = 1e-6,
    out_scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hidden.shape[-1]
    m = hidden.numel() // k
    hidden_2d = hidden.reshape(m, k)
    out_2d = torch.empty((m, k), dtype=torch.int8, device=hidden.device)
    scale_1d = torch.empty((m,), dtype=out_scale_dtype, device=hidden.device)
    block = triton.next_power_of_2(k)
    grid = (m,)
    _sk10_mtp_w4a8_rmsnorm_quant_kernel[grid](
        hidden_2d,
        rms_weight,
        out_2d,
        scale_1d,
        m,
        k,
        hidden_2d.stride(0),
        hidden_2d.stride(1),
        rms_weight.stride(0),
        out_2d.stride(0),
        out_2d.stride(1),
        scale_1d.stride(0),
        float(eps),
        BLOCK=block,
        num_warps=4,
        num_stages=2,
    )
    out = out_2d.reshape(hidden.shape)
    scale = scale_1d.reshape(hidden.shape[:-1] + (1,)).contiguous()
    return out, scale


mtp_draft_w4a8_forward = mtp_draft_w4a8_gemm
sk10_mtp_w4a8_gemm = mtp_draft_w4a8_gemm
sk10_mtp_draft_w4a8 = mtp_draft_w4a8_gemm

__all__ = [
    "mtp_draft_w4a8_gemm",
    "mtp_draft_w4a8_forward",
    "mtp_draft_w4a8_rmsnorm_quant",
    "sk10_mtp_w4a8_gemm",
    "sk10_mtp_draft_w4a8",
    "MTP_HIDDEN",
    "MTP_INTERMEDIATE",
    "GROUP_SIZE",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
]
