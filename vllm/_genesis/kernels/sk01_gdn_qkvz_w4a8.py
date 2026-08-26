# SPDX-License-Identifier: Apache-2.0
"""SK-01 W4A8 — GDN_QKVZ_FUSED_W4A8 — branchless super kernel QKVZ para zipperlein W4A8.

Branchless: kernel puro sin branches Python. Toda validacion (dims%128,
dtype, contiguity, K%128==0, N%128==0, K%GROUP_SIZE==0, packing 2xINT4 por byte,
scales contiguas, strides) se hace UNA vez por capa en pwal
(patch_PN110 warmup_all_kernels), no por forward.
Cuando este kernel falla, indica bug del pwal (capa no validada), no del kernel.

Geometria: Global N=16384 (=10240+6144)=128*128, K=5120=40*128. Per-rank
TP=2: N=8192. W4A8: peso INT4 GPTQ sym g128 empaquetado uint8 2 por byte
(q 0..15, zp 8, valor -8..7) + escala per-grupo bf16 [K/GROUP,N]. Activacion
INT8 per-token quantizada previa. Compute INT8 TC mma.m16n8k32.s8 sm_86.

Contrato hot (asume validado) — sm_86 Ampere bf16:
    a_packed  [M,K] int8   — activacion ya quantizada per-token
    w_packed  [K//2,N] uint8 — peso INT4 empaquetado low nibble k par, high nibble k impar
    a_scales  [M] bf16     — per-token amax/127, contiguo, bf16
    w_scales  [K//GROUP,N] bf16 — per-grupo, contiguo, bf16, GROUP=128
    out       [M,N] bf16/fp16 — salida nativa sm_86

Triton sm_86: solo tl.load / tl.dot / tl.where / tl.store, sin branches Python.
Unpack branchless via tl.where(is_even, low, high) - 8 -> int8.
Dot INT8 -> INT32, epilogo bf16 a_scale * w_scale por bloque, acumulacion bf16.
INT32→bf16 directo sin pasar por fp32. Validacion movida a arranque.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK01_W4A8_N_GLOBAL: int = 16384
SK01_W4A8_K: int = 5120
SK01_W4A8_QKV_N: int = 10240
SK01_W4A8_Z_N: int = 6144
SK01_W4A8_Q_END: int = 2048
SK01_W4A8_K_END: int = 4096
SK01_W4A8_V_END: int = 10240
SK01_W4A8_N_PER_RANK: int = 8192
SK01_W4A8_K_PER_RANK: int = 5120
SK01_W4A8_GROUP_SIZE: int = 128
SK01_W4A8_K_PACKED: int = 2560

BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32
GROUP_SIZE: int = 128


@triton.jit
def _sk01_gdn_qkvz_w4a8_kernel(
    a_ptr,
    w_packed_ptr,
    out_ptr,
    a_scale_ptr,
    w_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_out_m,
    stride_out_n,
    stride_scale_a,
    stride_scale_g,
    stride_scale_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_scales = tl.load(a_scale_ptr + offs_m * stride_scale_a, mask=mask_m, other=0.0).to(tl.bfloat16)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        mask_k = cur_k < K
        group = k // GROUP_SIZE
        w_scales = tl.load(w_scale_ptr + group * stride_scale_g + offs_n * stride_scale_n, mask=mask_n, other=1.0).to(tl.bfloat16)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak
        mask_a = mask_m[:, None] & mask_k[None, :]
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0).to(tl.int8)
        packed_k = cur_k // 2
        is_even = (cur_k % 2) == 0
        w_packed_ptrs = w_packed_ptr + packed_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        mask_w = mask_k[:, None] & mask_n[None, :]
        packed = tl.load(w_packed_ptrs, mask=mask_w, other=0).to(tl.int32)
        w_low = packed & 15
        w_high = (packed >> 4) & 15
        is_even_bc = is_even[:, None]
        w_q = tl.where(is_even_bc, w_low, w_high)
        w_int8 = (w_q - 8).to(tl.int8)
        int_acc = tl.dot(a_tile, w_int8)
        scaled = int_acc.to(tl.bfloat16) * a_scales[:, None] * w_scales[None, :]
        acc = acc + scaled
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


def sk01_gdn_qkvz_w4a8_gemm(
    a: torch.Tensor,
    w_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = a.shape[0]
    N = w_packed.shape[1]
    K = a.shape[1]
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk01_gdn_qkvz_w4a8_kernel[grid](
        a,
        w_packed,
        out,
        a_scales,
        w_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        w_packed.stride(0),
        w_packed.stride(1),
        out.stride(0),
        out.stride(1),
        a_scales.stride(0),
        w_scales.stride(0),
        w_scales.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


def sk01_gdn_qkvz_w4a8_forward(
    hidden: torch.Tensor,
    w_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qkvz = sk01_gdn_qkvz_w4a8_gemm(hidden, w_packed, a_scales, w_scales, out_dtype=out_dtype)
    N = qkvz.shape[1]
    q_n = N // 8
    k_n = N // 8
    v_n = (N * 3) // 8
    q = qkvz[:, 0:q_n]
    k = qkvz[:, q_n : q_n + k_n]
    v = qkvz[:, q_n + k_n : q_n + k_n + v_n]
    z = qkvz[:, q_n + k_n + v_n :]
    return q, k, v, z


gdn_qkvz_w4a8 = sk01_gdn_qkvz_w4a8_gemm
gdn_qkvz_fused_w4a8 = sk01_gdn_qkvz_w4a8_gemm
sk01_w4a8_forward = sk01_gdn_qkvz_w4a8_forward
SK01_GDN_QKVZ_FUSED_W4A8 = sk01_gdn_qkvz_w4a8_gemm

__all__ = [
    "sk01_gdn_qkvz_w4a8_gemm",
    "gdn_qkvz_w4a8",
    "gdn_qkvz_fused_w4a8",
    "sk01_gdn_qkvz_w4a8_forward",
    "sk01_w4a8_forward",
    "SK01_GDN_QKVZ_FUSED_W4A8",
    "SK01_W4A8_N_GLOBAL",
    "SK01_W4A8_K",
    "SK01_W4A8_N_PER_RANK",
    "SK01_W4A8_QKV_N",
    "SK01_W4A8_Z_N",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_SIZE",
]
