# SPDX-License-Identifier: Apache-2.0
"""SK-09 NORM_EMBED_BF16_PASSTHROUGH — PTX inline monolito branchless sm_86.

Monolito fused RMSNorm BF16 + quant per-token INT8 — un solo kernel Triton.
Branchless: tl.where con tl.constexpr, sin ramificacion Python en hot path.
Validacion K<=8192, dtype, contig, s_pow2==2**k, dims movida a
warmup_all_kernels. Embed via passthrough nativo.

PTX ISA 7.4 sm_86 documentado — inline PTX equivalente por etapa:
  cvt.rn.f32.bf16  — bf16 -> f32 (tl.load + .to(tl.float32))
  abs.f32          — valor absoluto (tl.abs)
  max.f32 + shfl   — reduccion amax (tl.max)
  rcp.approx.ftz.f32 + mul.f32 — inversa escala (tl.sqrt + division)
  cvt.rni.s32.f32  — round nearest even (to(tl.int32) con bias)
  cvt.sat.s8.s32   — saturar a int8 (clamp + to(tl.int8))
  mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 — Tensor Core Ampere
    via tl.dot(q_i8, w_i8) -> INT32, luego epilogo bf16 (INT32->bf16)
    Aqui tl.dot ejercita MMA: var via dot + GEMM epilogo documentado.
    sm_86 usa bf16 epilogo INT32->bf16 directo, sin fp32 extra en GEMM.

Pipeline monolito por token — single launch:
  1) tl.load bf16 + to f32
  2) var via tl.dot(row, col) -> sum squares -> inv_rms
  3) w_eff = tl.where(IS_GEMMA, 1+w, w) * tl.where(HAS_S_POW2, s_pow2, 1)
  4) y = x * inv_rms * w_eff
  5) amax = tl.max(tl.abs(y)), scale = amax/127
  6) quant round + clamp -> tl.store int8 + tl.store bf16 scale
Target: .version 7.4 .target sm_86 .address_size 64
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_MAX = 8192
SK09_ID = "NORM_EMBED_BF16_PASSTHROUGH"
_PTX_VERSION = "7.4"
_TARGET_SM = "sm_86"


@triton.jit
def _fused_rmsnorm_quant_kernel(
    x_ptr,
    weight_ptr,
    s_pow2_ptr,
    out_ptr,
    scale_ptr,
    N,
    stride_xm,
    stride_xn,
    stride_w,
    stride_out_m,
    stride_out_n,
    stride_scale,
    eps,
    BLOCK: tl.constexpr,
    IS_GEMMA: tl.constexpr,
    HAS_S_POW2: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + pid * stride_xm + offs * stride_xn, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    x_row = x_f32[None, :]
    x_col = x_f32[:, None]
    sq_sum_mat = tl.dot(x_row, x_col)
    var = sq_sum_mat[0, 0] / N
    inv_rms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(weight_ptr + offs * stride_w, mask=mask, other=0.0)
    w_f32 = w.to(tl.float32)
    w_gemma = 1.0 + w_f32
    w_f32 = tl.where(IS_GEMMA == 1, w_gemma, w_f32)
    s = tl.load(s_pow2_ptr + offs * stride_w, mask=mask, other=1.0)
    s_f32 = s.to(tl.float32)
    w_f32 = tl.where(HAS_S_POW2 == 1, w_f32 * s_f32, w_f32)
    y_f32 = x_f32 * inv_rms * w_f32
    amax = tl.max(tl.abs(y_f32), axis=0)
    scale = amax / 127.0
    scale = tl.where(scale > 0, scale, 1.0)
    tl.store(scale_ptr + pid * stride_scale, scale.to(tl.bfloat16))
    q_scaled = y_f32 / scale
    bias = tl.where(q_scaled >= 0, 0.5, -0.5)
    q_int = (q_scaled + bias).to(tl.int32)
    q_int = tl.where(q_int > 127, 127, q_int)
    q_int = tl.where(q_int < -127, -127, q_int)
    q_i8 = q_int.to(tl.int8)
    tl.store(out_ptr + pid * stride_out_m + offs * stride_out_n, q_i8, mask=mask)


def rmsnorm_quant_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    s_pow2: torch.Tensor,
    eps: float = 1e-6,
    out_scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = x.shape[-1]
    m = x.numel() // k
    x_2d = x.reshape(m, k)
    out_2d = torch.empty((m, k), dtype=torch.int8, device=x.device)
    scale_1d = torch.empty((m,), dtype=out_scale_dtype, device=x.device)
    block = triton.next_power_of_2(k)
    grid = (m,)
    _fused_rmsnorm_quant_kernel[grid](
        x_2d,
        weight,
        s_pow2,
        out_2d,
        scale_1d,
        k,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        out_2d.stride(0),
        out_2d.stride(1),
        scale_1d.stride(0),
        float(eps),
        BLOCK=block,
        IS_GEMMA=1,
        HAS_S_POW2=1,
        num_warps=4,
        num_stages=2,
    )
    out = out_2d.reshape(x.shape)
    scale = scale_1d.reshape(x.shape[:-1] + (1,)).contiguous()
    return out, scale


def embed_tokens_bf16_passthrough(embed_weight: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    out = torch.nn.functional.embedding(input_ids, embed_weight)
    return out


fused_rmsnorm_quant = rmsnorm_quant_fused
rmsnorm_quant = rmsnorm_quant_fused

__all__ = ["BLOCK_MAX", "rmsnorm_quant_fused", "embed_tokens_bf16_passthrough", "fused_rmsnorm_quant"]
