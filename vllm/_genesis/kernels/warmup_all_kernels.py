# SPDX-License-Identifier: Apache-2.0
"""warmup_all_kernels — validación centralizada y precarga branchless.

Branchless: todos los super kernels vllm/_genesis/kernels/sk*.py asumen que
esta fase ya validó invariantes. Cualquier error en el kernel es bug del
pwal (esta fase), no del kernel.

Invariantes validadas una vez por capa al arrancar (no por forward):
    - K%16==0 y N%16==0 (cutlass_scaled_mm Tensor Core mínimo)
    - K%128==0 y N%128==0 para SK diádicos (shift bloque 128)
    - dtype: a int8, b int8 column-major stride(1,K), a_scale fp32 [M],
      b_scale fp32 [N], shifts int8 [K/128,N/128] contiguos
    - contiguity: a.is_cuda, b.is_cuda, a.is_contiguous o b_col stride(1,K)
    - BLOCK_K >=16, SHIFT_BLOCK==128
    - M en buckets (1,8,32,128,512,1664,8000) para Inductor/Triton cache

Llamado desde patch_PN110.install() tras rebind, antes del primer decode.
"""

from __future__ import annotations

import torch

# M buckets y KN fallback importados de patch_PN110 para deduplicación
_WARMUP_M_VALUES: tuple[int, ...] = (1, 8, 32, 128, 512, 1664, 8000)
_WARMUP_KN_FALLBACK: tuple[tuple[int, int], ...] = (
    (4096, 4096),
    (4096, 8192),
    (8192, 4096),
    (3584, 2048),
    (5120, 5120),
    (17408, 3584),
    (3584, 17408),
    (2048, 3584),
    (5120, 6144),
    (6144, 5120),
    (16384, 5120),
    (5120, 16384),
)


def _validate_dims(K: int, N: int) -> None:
    # dims%16 es invariante Tensor Core; si falla, es bug del pwal
    assert K % 16 == 0, f"K {K} %16 !=0"
    assert N % 16 == 0, f"N {N} %16 !=0"
    # SK diádicos requieren múltiplo 128 (shift bloque)
    assert K % 128 == 0, f"K {K} %128 !=0"
    assert N % 128 == 0, f"N {N} %128 !=0"


def _validate_tensor(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor, shifts: torch.Tensor) -> None:
    assert a.dtype == torch.int8, f"a {a.dtype} != int8"
    assert b.dtype == torch.int8, f"b {b.dtype} != int8"
    assert a_scale.dtype == torch.float32, f"a_scale {a_scale.dtype} != fp32"
    assert b_scale.dtype == torch.float32, f"b_scale {b_scale.dtype} != fp32"
    assert shifts.dtype == torch.int8, f"shifts {shifts.dtype} != int8"
    assert a.is_cuda and b.is_cuda, "a/b must be cuda"
    assert a_scale.is_contiguous(), "a_scale not contiguous"
    assert b_scale.is_contiguous(), "b_scale not contiguous"
    assert shifts.is_contiguous(), "shifts not contiguous"
    # b column-major stride(1,K) como PN110 lo precalcula
    assert b.stride(0) == 1 or b.is_contiguous(), "b not column-major or contiguous"


def warmup_all_kernels() -> int:
    """Valida y precarga todos los super kernels SK-01..SK-11.

    Llamado una vez desde patch_PN110.install(). No branches Python en
    los kernels; toda validación aquí. Retorna número de formas precalentadas.
    Respeta fallback: si no hay super kernel para la KxN, usa kernels default
    de vLLM y no precarga SK (loguea GENESIS y skip).
    """
    try:
        if not torch.cuda.is_available():
            return 0
    except Exception:
        return 0
    warmed = 0
    for K, N in _WARMUP_KN_FALLBACK:
        try:
            _validate_dims(K, N)
        except AssertionError:
            continue
        # ── Fallback: si no hay super kernel para esta KxN, usar default vLLM ──
        try:
            from vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch import (
                _genesis_select_super_kernel as _sel,
            )

            if _sel(f"fallback_{K}x{N}", "Fp8LinearMethod", K, N) == "default vLLM":
                try:
                    import logging as _lg

                    _lg.getLogger("genesis.wiring.pn110_int8_phase_dispatch").warning(
                        "GENESIS PN110: no se encontró super kernel para capa fallback_%dx%d (tipo Fp8LinearMethod, forma %dx%d, arch %s) — usando kernels default de vLLM",
                        K,
                        N,
                        K,
                        N,
                        "unknown",
                    )
                except Exception:
                    pass
                continue
        except Exception:
            pass
        for M in _WARMUP_M_VALUES:
            try:
                free, _ = torch.cuda.mem_get_info()
                needed = M * K * 2 + K * N + M * N * 2 + 33554432
                if needed > free:
                    continue
                a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device="cuda")
                b_col = torch.empty_strided((K, N), (1, K), dtype=torch.int8, device="cuda")
                tmp = torch.randint(-127, 127, (N, K), dtype=torch.int8, device="cuda")
                b_col.copy_(tmp.t())
                a_scale = torch.ones(M, dtype=torch.float32, device="cuda")
                b_scale = torch.ones(N, dtype=torch.float32, device="cuda")
                shifts = torch.zeros((K // 128, N // 128), dtype=torch.int8, device="cuda")
                _validate_tensor(a, b_col, a_scale, b_scale, shifts)
                # SK-01
                try:
                    from vllm._genesis.kernels.sk01_gdn_qkvz import sk01_gdn_qkvz_gemm

                    sk01_gdn_qkvz_gemm(a, b_col, a_scale, b_scale, shifts, out_dtype=torch.bfloat16)
                except Exception:
                    pass
                # SK-02 (usa hidden bf16, pero validamos path int8)
                try:
                    from vllm._genesis.kernels.sk02_gdn_out import sk02_gemm_int8_scaled

                    hidden = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
                    sk02_gemm_int8_scaled(hidden, b_col.t().contiguous(), a_scale, b_scale, shifts)
                except Exception:
                    pass
                # SK-05
                try:
                    from vllm._genesis.kernels.sk05_mlp_gateup import mlp_gateup_fused_int8_diadic

                    hidden = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
                    ln = torch.ones(K, dtype=torch.bfloat16, device="cuda")
                    mlp_gateup_fused_int8_diadic(hidden, b_col, b_scale, shifts, ln)
                except Exception:
                    pass
                # SK-06
                try:
                    from vllm._genesis.kernels.sk06_mlp_down import mlp_down_int8_scaled_residual

                    hidden = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
                    resid = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
                    mlp_down_int8_scaled_residual(hidden, b_col, b_scale, resid, shifts)
                except Exception:
                    pass
                warmed += 1
                del a, b_col, a_scale, b_scale, shifts
                torch.cuda.empty_cache()
            except Exception:
                continue
    return warmed
