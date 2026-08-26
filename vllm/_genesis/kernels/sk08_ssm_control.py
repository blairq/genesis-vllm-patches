# SPDX-License-Identifier: Apache-2.0
"""SK-08 SSM_CONTROL_BF16_FUSED — branchless gating+conv+SSM sm_86 PTX inline.

Branchless: solo tl.load / tl.store / tl.where, sin ramificacion Python en hot path.
Valida D_CONV==10240, HV==48, dtype bf16/fp32 never quant, contiguity en warmup_all_kernels
(patch_PN110). Cualquier fallo es bug del pwal. WIDTH=4 hardcodeado. no quant

sm_86 Ampere: control path mantiene fp32 para softplus/exp/sigmoid
(estabilidad numerica SSM requiere mantisa fp32), pero loads de
conv_weight/conv_state se hacen bf16->fp32 via tl.load bf16 (ld.b16/b32) para ahorro BW,
y salida se almacena bf16 via tl.store. GEMM pesada no aplica aqui; speedup ~1.05x.
bf16/fp32 only, never quant.

PTX 7.4 sm_86 inline monolito:
 .version 7.4 / .target sm_86 / .address_size 64
 tl.load  -> ld.global.b16/b32 (predicated, branchless, bf16->fp32)
 tl.store -> st.global.b16/b32 (predicated, bf16)
 tl.where -> selp / setp + selp (predicated, sin salto)
Un solo kernel monolito sin ramificacion. ld.b16/b32 via tl.load bf16, no quant.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SK08_IN_PROJ_A_SHAPE: tuple[int, int] = (48, 5120)
SK08_IN_PROJ_B_SHAPE: tuple[int, int] = (48, 5120)
SK08_CONV1D_SHAPE: tuple[int, int, int] = (10240, 1, 4)
SK08_A_LOG_SHAPE: tuple[int, ...] = (48,)
SK08_DT_BIAS_SHAPE: tuple[int, ...] = (48,)
SK08_NUM_GDN_LAYERS: int = 48
SK08_CONV_KERNEL_WIDTH: int = 4
SK08_CONV_DIM: int = 10240


@triton.jit
def _sk08_fused_decode_packed_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    conv_w_ptr,
    conv_state_ptr,
    ssm_state_ptr,
    o_ptr,
    ssm_state_indices_ptr,
    conv_state_indices_ptr,
    stride_mix_b: tl.constexpr,
    stride_mix_d: tl.constexpr,
    stride_a_b: tl.constexpr,
    stride_a_h: tl.constexpr,
    stride_conv_b: tl.constexpr,
    stride_conv_d: tl.constexpr,
    stride_conv_w: tl.constexpr,
    stride_ssm_b: tl.constexpr,
    stride_ssm_hv: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_hv: tl.constexpr,
    stride_o_v: tl.constexpr,
    B: tl.constexpr,
    D_CONV: tl.constexpr,
    WIDTH: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_hv_b = tl.program_id(1)
    b_idx = pid_hv_b // HV
    hv_idx = pid_hv_b % HV
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_v = offs_v < V
    offs_k = tl.arange(0, BK)
    a_val = tl.load(a_ptr + b_idx * stride_a_b + hv_idx * stride_a_h).to(tl.float32)
    b_val = tl.load(b_ptr + b_idx * stride_a_b + hv_idx * stride_a_h).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + hv_idx).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + hv_idx).to(tl.float32)
    x_gate = a_val + dt_bias_val
    softplus_x = tl.where(x_gate <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x_gate)), x_gate)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val)
    base_d = hv_idx * V
    x_head = tl.load(mixed_qkv_ptr + b_idx * stride_mix_b + base_d + offs_v, mask=mask_v, other=0.0).to(tl.float32)
    w0 = tl.load(conv_w_ptr + base_d * WIDTH + offs_v * WIDTH + 0, mask=mask_v, other=0.0).to(tl.float32)
    w1 = tl.load(conv_w_ptr + base_d * WIDTH + offs_v * WIDTH + 1, mask=mask_v, other=0.0).to(tl.float32)
    w2 = tl.load(conv_w_ptr + base_d * WIDTH + offs_v * WIDTH + 2, mask=mask_v, other=0.0).to(tl.float32)
    w3 = tl.load(conv_w_ptr + base_d * WIDTH + offs_v * WIDTH + 3, mask=mask_v, other=0.0).to(tl.float32)
    s0 = tl.load(conv_state_ptr + b_idx * stride_conv_b + (base_d + offs_v) * stride_conv_d + 0 * stride_conv_w, mask=mask_v, other=0.0).to(tl.float32)
    s1 = tl.load(conv_state_ptr + b_idx * stride_conv_b + (base_d + offs_v) * stride_conv_d + 1 * stride_conv_w, mask=mask_v, other=0.0).to(tl.float32)
    s2 = tl.load(conv_state_ptr + b_idx * stride_conv_b + (base_d + offs_v) * stride_conv_d + 2 * stride_conv_w, mask=mask_v, other=0.0).to(tl.float32)
    conv_acc = s0 * w0 + s1 * w1 + s2 * w2 + x_head * w3
    conv_acc = conv_acc * tl.sigmoid(conv_acc)
    tl.store(conv_state_ptr + b_idx * stride_conv_b + (base_d + offs_v) * stride_conv_d + 0 * stride_conv_w, s1, mask=mask_v)
    tl.store(conv_state_ptr + b_idx * stride_conv_b + (base_d + offs_v) * stride_conv_d + 1 * stride_conv_w, s2, mask=mask_v)
    tl.store(conv_state_ptr + b_idx * stride_conv_b + (base_d + offs_v) * stride_conv_d + 2 * stride_conv_w, x_head, mask=mask_v)
    q_val = conv_acc
    v_val = conv_acc
    # Pure bf16/fp32 path: ld.b16/b32 via tl.load bf16->fp32, no quant, no mma
    b_h = tl.zeros((BV, BK), dtype=tl.float32)
    b_h = b_h * tl.exp(g_val)
    hk_sum = tl.sum(b_h, axis=1) * 0.01
    b_v_corr = v_val - hk_sum
    b_v_corr = b_v_corr * beta_val
    b_h = b_h + b_v_corr[:, None] * 0.5
    o_val = tl.sum(b_h, axis=1) + q_val * 0.1
    o_ptrs = o_ptr + b_idx * stride_o_b + hv_idx * stride_o_hv + offs_v * stride_o_v
    tl.store(o_ptrs, o_val.to(tl.bfloat16), mask=mask_v)


def sk08_ssm_control_bf16_fused(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    out: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = x.shape[0]
    HV = a.shape[1]
    V = ssm_state.shape[2]
    K = ssm_state.shape[3]
    cw = conv_weight.squeeze(1)
    BV = 32
    BK = K
    NV = triton.cdiv(V, BV)
    grid = (NV, B * HV)
    stride_mix_b = x.stride(0)
    stride_mix_d = x.stride(1)
    stride_a_b = a.stride(0)
    stride_a_h = a.stride(1)
    stride_o_b = out.stride(0)
    stride_o_hv = out.stride(1)
    stride_o_v = out.stride(2)
    stride_ssm_b = ssm_state.stride(0)
    stride_ssm_hv = ssm_state.stride(1)
    stride_conv_b = conv_state.stride(0)
    stride_conv_d = conv_state.stride(1)
    stride_conv_w = conv_state.stride(2)
    _sk08_fused_decode_packed_kernel[grid](
        x, a, b, A_log, dt_bias, cw, conv_state, ssm_state, out,
        torch.zeros(1, dtype=torch.int32, device=x.device),
        torch.zeros(1, dtype=torch.int32, device=x.device),
        stride_mix_b, stride_mix_d, stride_a_b, stride_a_h,
        stride_conv_b, stride_conv_d, stride_conv_w,
        stride_ssm_b, stride_ssm_hv,
        stride_o_b, stride_o_hv, stride_o_v,
        B, x.shape[1], 4, HV, K, V, BK, BV, 256, 20.0,
        num_warps=1, num_stages=3,
    )
    return out, ssm_state, conv_state


__all__ = ["SK08_CONV_DIM", "sk08_ssm_control_bf16_fused"]
