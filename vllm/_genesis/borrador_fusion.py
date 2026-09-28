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


def _conv_q8_cuda(h, coef, base, gscale, bsq: int):
    h2 = h.reshape(-1, h.shape[-1])
    T, N = h2.shape
    q = torch.empty((T, N), dtype=torch.int8, device=h.device)
    esc = torch.empty((T, 1), dtype=torch.float32, device=h.device)
    if T:
        g = gscale.reshape(-1)
        if g.dtype != torch.float32:
            g = g.float()
        G = N // 16                                    # grupos de la conv (coef trae 2 lados x 2 taps x G)
        _kern("sk32_conv_q8").lanzar((T, 1), [h2, coef, base, q, esc, g, N, G, bsq, h2.stride(0), coef.stride(0), q.stride(0)])
    return q, esc


@torch.library.custom_op("genesis::pn160_conv_q8", mutates_args=())
def conv_q8_op(h: torch.Tensor, coef: torch.Tensor, base: torch.Tensor, gscale: torch.Tensor,
               bsq: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _conv_q8_cuda(h, coef, base, gscale, bsq)


@conv_q8_op.register_fake
def _conv_q8_fake(h, coef, base, gscale, bsq):
    T = h.numel() // h.shape[-1]
    return h.new_empty((T, h.shape[-1]), dtype=torch.int8), h.new_empty((T, 1), dtype=torch.float32)


def preparar(conv, h: torch.Tensor, lin):
    """Reemplaza ``h, coef = conv.prepare(h)`` cuando lo que sigue es ``lin`` (qkv o gate_up): la conv y la
    cuantizacion en un kernel (SK-32 conv_q8). Devuelve (h sin convolucionar -solo forma-, coef del lado 1,
    (q, esc)); si no aplica, (salida de prepare, coef, None)."""
    if (_marlin(lin) is None or getattr(conv, "taps", 0) != 2 or getattr(conv, "group_size", 0) != 16
            or h.dtype != torch.float16 or conv.base_kernel.dtype != torch.float16):
        y, c1 = conv.prepare(h)
        return y, c1, None
    from vllm._genesis import borrador_marlin as _g157
    T = h.shape[0]
    coef = _g157.proyectar(conv.kernel_projection, h).reshape(T, 2 * conv.taps * conv.num_groups)
    q, esc = torch.ops.genesis.pn160_conv_q8(h, coef, conv.base_kernel[0], lin.input_global_scale, conv.block_size)
    c1 = coef.view(T, 2, conv.taps, conv.num_groups)[:, 1]
    return h, c1, (q, esc)


def _marlin(lin):
    if not ACTIVO or lin is None or getattr(lin, "bias", None) is not None:
        return None
    from vllm._genesis import rot_down as _g148
    return _g148._marlin_int8(lin)


def lineal(lin, x: torch.Tensor, q160=None) -> torch.Tensor:
    """Reemplaza ``y, _ = lin(x)``: con W4A8, SK-32 + Marlin (y el all-reduce si la lineal es de fila). Con
    q160 = (q, esc) ya cuantizado por preparar(), x solo da la forma."""
    k = _marlin(lin)
    if k is None:
        y, _ = lin(x)
        return y
    from vllm._genesis import rot_down as _g148
    q, esc = q160 if q160 is not None else torch.ops.genesis.pn160_q8(x, lin.input_global_scale)
    out = _g148._gemm_int8(lin, k, q, esc).reshape(*x.shape[:-1], -1)
    if getattr(lin, "reduce_results", False):
        out = _g148._reducir(lin, out)
    return out


def mlp(m, x: torch.Tensor, q160=None) -> torch.Tensor:
    """Reemplaza el forward de Qwen2MLP (gate_up -> SiluAndMul -> down_proj)."""
    gate_up = lineal(m.gate_up_proj, x, q160)
    d = m.down_proj
    k = _marlin(d)
    if k is None:
        out, _ = d(m.act_fn(gate_up))
        return out
    from vllm._genesis import rot_down as _g148
    q, esc = torch.ops.genesis.pn160_silu_q8(gate_up, d.input_global_scale)
    out = _g148._gemm_int8(d, k, q, esc).reshape(*gate_up.shape[:-1], -1)
    return _g148._reducir(d, out)
