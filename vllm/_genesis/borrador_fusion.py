# SPDX-License-Identifier: Apache-2.0
"""PN160 — fusiones de las lineales del borrador DFlash2 (inventario del 28-09: el borrador no tenia ninguna).

Cada lineal W4A8 del borrador corria per_token_quant_int8 + la escala por la global (inductor) + Marlin, y
delante de down_proj ademas act_and_mul: 4 kernels chicos por lineal. Con SK-32 la cuantizacion (y el
SiluAndMul) es UN kernel y el int8 entra directo a Marlin por ``rot_down._gemm_int8`` (como el target).
Si la lineal no corre en W4A8 (sin input_global_scale / otro kernel), todo sigue por el camino de siempre.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn160")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN160_BORRADOR_FUSION", "0").strip().lower() in ("1", "true", "yes", "on")
_k: dict = {}


def _kern(nombre: str):
    clave = (torch.cuda.current_device(), nombre)
    if clave not in _k:
        from vllm._genesis.kernels.ptx_lab import Kernel
        k = Kernel("sk32_borrador_q8.cu", nombre, warps=8)
        k.cargar()
        _k[clave] = k
    return _k[clave]


def _q8_cuda(x: torch.Tensor, gscale: torch.Tensor, silu: bool):
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    T = x2.shape[0]
    N = x2.shape[1] // 2 if silu else x2.shape[1]
    q = torch.empty((T, N), dtype=torch.int8, device=x.device)
    esc = torch.empty((T, 1), dtype=torch.float32, device=x.device)
    if T:
        g = gscale.reshape(-1)
        if g.dtype != torch.float32:
            g = g.float()
        _kern("sk32_silu_q8" if silu else "sk32_q8").lanzar((T, 1), [x2, q, esc, g, N, x2.stride(0), q.stride(0)])
    return q, esc


@torch.library.custom_op("genesis::pn160_q8", mutates_args=())
def q8_op(x: torch.Tensor, gscale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _q8_cuda(x, gscale, False)


@q8_op.register_fake
def _q8_fake(x, gscale):
    T = x.numel() // x.shape[-1]
    return x.new_empty((T, x.shape[-1]), dtype=torch.int8), x.new_empty((T, 1), dtype=torch.float32)


@torch.library.custom_op("genesis::pn160_silu_q8", mutates_args=())
def silu_q8_op(x: torch.Tensor, gscale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _q8_cuda(x, gscale, True)


@silu_q8_op.register_fake
def _silu_q8_fake(x, gscale):
    T = x.numel() // x.shape[-1]
    N = x.shape[-1] // 2
    return x.new_empty((T, N), dtype=torch.int8), x.new_empty((T, 1), dtype=torch.float32)


def _marlin(lin):
    if not ACTIVO or lin is None or getattr(lin, "bias", None) is not None:
        return None
    from vllm._genesis import rot_down as _g148
    return _g148._marlin_int8(lin)


def lineal(lin, x: torch.Tensor) -> torch.Tensor:
    """Reemplaza ``y, _ = lin(x)``: con W4A8, SK-32 + Marlin (y el all-reduce si la lineal es de fila)."""
    k = _marlin(lin)
    if k is None:
        y, _ = lin(x)
        return y
    from vllm._genesis import rot_down as _g148
    q, esc = torch.ops.genesis.pn160_q8(x, lin.input_global_scale)
    out = _g148._gemm_int8(lin, k, q, esc).reshape(*x.shape[:-1], -1)
    if getattr(lin, "reduce_results", False):
        out = _g148._reducir(lin, out)
    return out


def mlp(m, x: torch.Tensor) -> torch.Tensor:
    """Reemplaza el forward de Qwen2MLP (gate_up -> SiluAndMul -> down_proj)."""
    gate_up = lineal(m.gate_up_proj, x)
    d = m.down_proj
    k = _marlin(d)
    if k is None:
        out, _ = d(m.act_fn(gate_up))
        return out
    from vllm._genesis import rot_down as _g148
    q, esc = torch.ops.genesis.pn160_silu_q8(gate_up, d.input_global_scale)
    out = _g148._gemm_int8(d, k, q, esc).reshape(*gate_up.shape[:-1], -1)
    return _g148._reducir(d, out)
