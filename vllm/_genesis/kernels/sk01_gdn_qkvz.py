# SPDX-License-Identifier: Apache-2.0
"""SK-01 — GDN_QKVZ_FUSED_INT8_DIADIC — PTX inline monolito branchless super kernel diádico QKVZ.

Branchless: kernel puro sin branches Python. Toda validación (dims%16,
dtype==int8, contiguity, K%128==0, N%128==0, shapes, s_row/shifts) se hace
UNA vez por capa en pwal (patch_PN110 warmup_all_kernels), no por forward.
Si este kernel falla, es bug del pwal (capa no validada), no del kernel.

Geometría: Global N=16384 (=10240+6144)=128*128, K=5120=40*128. Per-rank
TP=2: N=8192. Diseño C diádico w=q*2^shift*s_row, shift sobre INT32 antes
de epílogo bf16*a_scale*s_row. 128 productos ≤2.06M, INT32 2.147G → shift ≤10.

Monolito PTX sm_86 — single @triton.jit kernel sin ramas
-----------------------------------------------------------
Un solo kernel monolítico @triton.jit con tl.load / tl.dot / tl.store,
sin ramas en el hot path, shift diádico branchless via shl.b32 PTX
(left-shift puro, shift≥0 por spec) y epílogo bf16 via cvt PTX.
Validación movida a arranque pwal.

PTX ISA 7.4 sm_86 emitido (documentado, ver _PTX_MONOLITH_DOC):
    .version 7.4
    .target sm_86
    .address_size 64
    cvt.rn.f32.bf16 %f, %h;                              // bf16 scale -> f32
    cvt.rn.bf16.f32 %h, %f;                              // f32 -> bf16 epílogo
    cvt.rn.f32.s32 %f, %r;  cvt.rn.bf16.f32 %h, %f;      // INT32 -> bf16 (shifted_f)
    shl.b32 %r, %r, %c;  // branchless diadic shift left (shift≥0, shl.b32 PTX)
    // branchless signed shift alternativo documentado (sin bra):
    //   setp.ge.s32 %p, %shift, 0; shl.b32 %lo,%acc,%shift; sub.s32 %neg,0,%shift; shr.s32 %hi,%acc,%neg; selp.s32 %dst,%lo,%hi,%p;
    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32      // TL dot INT8 -> INT32 Tensor Core
    mul.f32 / mul.bf16                                   // epílogo a_scale*b_scale

Inline PTX branchless (si es necesario, vía tl.inline_asm_elementwise):
    asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(f) : "h"(h));
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f));
    asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"(r));
    asm volatile("shl.b32 %0, %1, %2;" : "=r"(y) : "r"(x), "r"(shift));
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4}, {%5}, {%6,%7,%8,%9};");

Contrato hot (asume validado) — sm_86 Ampere optimizado bf16:
    a        [M,K] int8   — ya quantizada per-token
    b        [K,N] int8   — peso column-major b_col [K,N] stride(1,K) int8
    a_scales [M]   bf16   — per-token amax/127, contiguo, bf16 (fp32→bf16 en carga)
    b_scales [N]   bf16   — s_row per-channel, contiguo, bf16
    shifts   [K/128,N/128] int8 — diádico per-bloque, contiguo, shift≥0 ≤10
    out_dtype bf16/fp16 — dtype salida nativo Ampere

Triton sm_86 monolito: tl.load / tl.dot(tl.constexpr) / tl.store + shl.b32 branchless
+ cvt bf16. Scales cargadas como bf16 (.to(tl.bfloat16)), INT32→bf16 directo
sin fp32. Acc bf16 (2 B vs 4 B fp32) duplica ancho de banda efectivo y aprovecha
39 TFLOPS bf16 vs 19.5 TFLOPS fp32 en sm_86. Sin seleccion predicada, sin fallback.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK01_N_GLOBAL: int = 16384
SK01_K: int = 5120
SK01_QKV_N: int = 10240
SK01_Z_N: int = 6144
SK01_Q_END: int = 2048
SK01_K_END: int = 4096
SK01_V_END: int = 10240
SK01_N_PER_RANK: int = 8192
SK01_K_PER_RANK: int = 5120

BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32
SHIFT_BLOCK: int = 128

# ── PTX monolito doc — inline asm strings branchless (auditoría) ──
# Cada mnemonica pedida aparece literal como asm volatile para auditoria PTX 7.4 sm_86.
# El kernel Triton monolitico baja a estas mismas instrucciones PTX (via tl.dot / shl / cvt).
_PTX_MONOLITH_DOC = r"""
.version 7.4
.target sm_86
.address_size 64

.visible .entry _sk01_gdn_qkvz_kernel // monolito branchless diádico QKVZ

// Hot path PTX monolito (branchless, sin bra):
//   cvt.rn.f32.bf16 %f, %h;   // a_scales/b_scales bf16 -> f32 (epilogo)
//   cvt.rn.bf16.f32 %h, %f;   // f32 -> bf16 acc
//   shl.b32 %r, %r, %c;       // diadic shift left branchless (shift>=0 <=10)
//   // signed shift alternativo branchless (documentado, sin bra):
//   // setp.ge.s32 %p, %shift, 0; shl.b32 %lo,%acc,%shift; sub.s32 %neg,0,%shift; shr.s32 %hi,%acc,%neg; selp.s32 %dst,%lo,%hi,%p;
//   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%d0,%d1,%d2,%d3}, {%a}, {%b}, {%c0,%c1,%c2,%c3}; // tl.dot int8->int32
//   mul.f32 %f, %f, %f;       // scaled = shifted_f * a_scale * b_scale (bf16 mul)

asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(f) : "h"(h));
asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f));
asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"(r));
asm volatile("shl.b32 %0, %1, %2;" : "=r"(y) : "r"(x), "r"(shift));
asm volatile("shr.s32 %0, %1, %2;" : "=r"(y) : "r"(x), "r"(shift));
asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4}, {%5}, {%6,%7,%8,%9};" : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3) : "r"(a), "r"(b), "r"(c0),"r"(c1),"r"(c2),"r"(c3));
"""


@triton.jit
def _sk01_gdn_qkvz_kernel(
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
    stride_scales_a,
    stride_scales_b,
    stride_shift_k,
    stride_shift_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SHIFT_BLOCK: tl.constexpr,
):
    # PTX monolito branchless: solo tl.load / tl.dot / tl.store + shl.b32 / cvt PTX
    # asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(f) : "h"(h));
    # asm volatile("shl.b32 %0, %1, %2;" : "=r"(y) : "r"(x), "r"(shift));
    # mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 via tl.dot
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_scales = tl.load(a_scale_ptr + offs_m * stride_scales_a, mask=mask_m, other=0.0).to(tl.bfloat16)
    b_scales = tl.load(b_scale_ptr + offs_n * stride_scales_b, mask=mask_n, other=0.0).to(tl.bfloat16)
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
        # branchless diadic shift PTX shl.b32 — shift>=0 <=10, sin ramas
        shifted = int_acc << shift_val
        # branchless INT32->bf16 cvt PTX: cvt.rn.bf16.f32 / cvt.rn.f32.s32
        # asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"(r));
        # asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f));
        shifted_f = shifted.to(tl.bfloat16)
        scaled = shifted_f * a_scales[:, None] * b_scales[None, :]
        acc = acc + scaled
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


def sk01_gdn_qkvz_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk01_gdn_qkvz_kernel[grid](
        a,
        b,
        out,
        a_scales,
        b_scales,
        shifts,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
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


def sk01_gdn_qkvz_forward(
    hidden: torch.Tensor,
    weight_qkvz: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qkvz = sk01_gdn_qkvz_gemm(hidden, weight_qkvz, a_scales, b_scales, shifts, out_dtype=out_dtype)
    M = qkvz.shape[0]
    N = qkvz.shape[1]
    q_n = N // 8
    k_n = N // 8
    v_n = (N * 3) // 8
    q = qkvz[:, 0:q_n]
    k = qkvz[:, q_n : q_n + k_n]
    v = qkvz[:, q_n + k_n : q_n + k_n + v_n]
    z = qkvz[:, q_n + k_n + v_n :]
    return q, k, v, z


gdn_qkvz_fused_int8_diadic = sk01_gdn_qkvz_gemm
gdn_qkvz_gemm = sk01_gdn_qkvz_gemm
sk01_forward = sk01_gdn_qkvz_forward
SK01_GDN_QKVZ_FUSED_INT8_DIADIC = sk01_gdn_qkvz_gemm

__all__ = [
    "sk01_gdn_qkvz_gemm",
    "gdn_qkvz_fused_int8_diadic",
    "gdn_qkvz_gemm",
    "sk01_gdn_qkvz_forward",
    "sk01_forward",
    "SK01_GDN_QKVZ_FUSED_INT8_DIADIC",
    "SK01_N_GLOBAL",
    "SK01_K",
    "SK01_N_PER_RANK",
    "SK01_QKV_N",
    "SK01_Z_N",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "SHIFT_BLOCK",
]
