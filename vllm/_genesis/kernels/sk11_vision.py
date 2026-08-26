# SPDX-License-Identifier: Apache-2.0
"""SK-11 VISION_BF16 — monolito PTX branchless BF16 GEMM.

Branchless: solo tl.load tl.dot tl.store. Validacion movida a warmup_all_kernels.
Vision tower 333 tensores visual.* se mantiene BF16 intacto. Hot path es GEMM BF16
PTX monolito via Triton mma. Cualquier quant es bug del pwal.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from typing import Any

SK11_TENSOR_COUNT: int = 333
SK11_PATTERN: str = "visual.*"
SK11_DTYPE_STR: str = "bfloat16"
SK11_QUANTIZED: bool = False
SK11_IS_PASSTHROUGH: bool = True
SK11_TENSOR_COUNT_EXPECTED = SK11_TENSOR_COUNT
VISION_TENSOR_COUNT = SK11_TENSOR_COUNT
VISION_PATTERN = SK11_PATTERN

BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32
BLOCK: int = 32
VISION_HIDDEN: int = 1152


@triton.jit
def _sk11_vision_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
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
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        cur_k = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak
        b_ptrs = b_ptr + cur_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_tile = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (cur_k[None, :] < K), other=0.0)
        b_tile = tl.load(b_ptrs, mask=(cur_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = acc + tl.dot(a_tile, b_tile)
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def sk11_vision_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sk11_vision_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out


def is_visual_tensor(name: str) -> bool:
    return name == "visual" or name.startswith("visual.")


def should_quantize(name: str) -> bool:
    return False


def is_available() -> bool:
    return True


def is_sk11_available() -> bool:
    return True


def passthrough_bf16(x: Any) -> Any:
    return x


def vision_bf16_passthrough(x: Any) -> Any:
    return passthrough_bf16(x)


vision_passthrough = vision_bf16_passthrough
sk11_passthrough = passthrough_bf16


def vision_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return sk11_vision_gemm(a, b)


def sk11_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return sk11_vision_gemm(a, b)


def get_sk11_info() -> dict[str, Any]:
    return {
        "sk_id": "SK-11",
        "name": "VISION_BF16",
        "pattern": SK11_PATTERN,
        "tensor_count": SK11_TENSOR_COUNT,
        "dtype": SK11_DTYPE_STR,
        "quantized": SK11_QUANTIZED,
        "is_passthrough": SK11_IS_PASSTHROUGH,
    }


def describe() -> str:
    return "SK-11 VISION_BF16: visual.* 333 tens. ViT — bfloat16 monolito PTX branchless."


__all__ = [
    "SK11_TENSOR_COUNT",
    "SK11_PATTERN",
    "SK11_DTYPE_STR",
    "SK11_QUANTIZED",
    "SK11_IS_PASSTHROUGH",
    "SK11_TENSOR_COUNT_EXPECTED",
    "VISION_TENSOR_COUNT",
    "VISION_PATTERN",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "is_visual_tensor",
    "should_quantize",
    "is_available",
    "is_sk11_available",
    "passthrough_bf16",
    "vision_bf16_passthrough",
    "vision_passthrough",
    "sk11_passthrough",
    "sk11_vision_gemm",
    "vision_forward",
    "sk11_forward",
    "get_sk11_info",
    "describe",
]
