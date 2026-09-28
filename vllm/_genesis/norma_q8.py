# SPDX-License-Identifier: Apache-2.0
"""PN156 — la norma del decoder escribe int8 (SK-26): residuo + RMSNorm + int8 por token en un kernel.

Antes de qkv / in_proj_qkvz / gate_up corrian tres kernels por sitio: la RMSNorm de inductor (con la suma
del residuo), per_token_quant_int8 y la escala por la global de Marlin. SK-26 hace las tres cosas y Marlin
W4A8 recibe el int8 directo, como con SK-23 (down_proj) y SK-25 (o_proj/out_proj).

Medido aislado (reloj fijo 1395 MHz, grafos): 7,31 -> 2,99 us con 9 filas, 605 -> 351 us con 8192. La
normalizacion se cancela en la cuantizacion (q = round(r*g*127/max|r*g|)): el rstd solo entra en la
escala de salida. Maximo y suma de cuadrados se reducen juntos (una barrera, no cinco).

Por que NO como PN135: aquel pasaba el int8 por un diccionario indexado por data_ptr, que torch.compile
hornea al trazar (FakeTensor sin puntero). Aca el int8 y la escala son SALIDAS de una custom op funcional
y viajan como argumento explicito (``_q156``) del decoder a la lineal consumidora: el decoder llama
``norma()``, y la atencion / el GDN / la MLP llaman ``lineal()`` en vez de la lineal. Si algo no cierra
(no es Marlin W4A8, tiene bias, la norma no es Gemma), todo corre por el camino de siempre.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn156")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN156_NORMA_Q8", "0").strip().lower() in ("1", "true", "yes", "on")
_k26: dict = {}


def _kernel(h: int):
    clave = (torch.cuda.current_device(), h)
    if clave not in _k26:
        from vllm._genesis.kernels.ptx_lab import Kernel
        k = Kernel("sk26_norma_q8.cu", "sk26_norma_q8", defs=["-DSUMA=3", f"-DH={h}"], warps=h // 8 // 32)
        k.cargar()
        _k26[clave] = k
    return _k26[clave]


def norma_q8_cuda(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor, gscale: torch.Tensor,
                  eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    r2 = res.reshape(-1, res.shape[-1]).contiguous()
    T, H = x2.shape
    ro = torch.empty_like(r2)
    q = torch.empty((T, H), dtype=torch.int8, device=x.device)
    esc = torch.empty((T, 1), dtype=torch.float32, device=x.device)
    if T:
        g = gscale.reshape(-1)
        if g.dtype != torch.float32:
            g = g.float()
        _kernel(H).lanzar((T, 1), [x2, r2, ro, w, q, esc, g, q.stride(0), float(eps)])
    return ro.view(res.shape), q, esc


@torch.library.custom_op("genesis::pn156_norma_q8", mutates_args=())
def norma_q8_op(x: torch.Tensor, res: torch.Tensor, w: torch.Tensor, gscale: torch.Tensor,
                eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x + res, int8 [T, H], escala fp32 [T, 1] ya por gscale) de GemmaRMSNorm(x + res)."""
    return norma_q8_cuda(x, res, w, gscale, eps)


@norma_q8_op.register_fake
def _norma_q8_fake(x, res, w, gscale, eps):
    T = x.numel() // x.shape[-1]
    return (torch.empty_like(res), x.new_empty((T, x.shape[-1]), dtype=torch.int8),
            x.new_empty((T, 1), dtype=torch.float32))


def _consumidor(capa, cual: int):
    if cual == 1:
        return getattr(getattr(capa, "mlp", None), "gate_up_proj", None)
    if hasattr(capa, "self_attn"):
        return capa.self_attn.qkv_proj
    la = getattr(capa, "linear_attn", None)
    return getattr(la, "in_proj_qkvz", None)


def _marlin(lin):
    """Kernel Marlin W4A8 de la lineal o None. Solo lee atributos: estatico al trazar."""
    if not ACTIVO or lin is None or getattr(lin, "bias", None) is not None:
        return None
    from vllm._genesis import rot_down as _g148
    return _g148._marlin_int8(lin)


def norma(capa, cual: int, h: torch.Tensor, r: torch.Tensor):
    """Reemplaza ``h, r = norma(h, r)`` del decoder. Devuelve (h, r, _q156): con PN156, h es la entrada
    SIN normalizar (solo sirve de forma/dtype: la consumidora usa _q156) y _q156 = (int8, escala)."""
    n = capa.input_layernorm if cual == 0 else capa.post_attention_layernorm
    lin = _consumidor(capa, cual)
    k = _marlin(lin)
    if (k is None or type(n).__name__ != "GemmaRMSNorm" or n.weight.dtype != torch.float16
            or h.dtype != torch.float16 or h.shape[-1] % 256):
        hh, rr = n(h, r)
        return hh, rr, None
    rr, q, esc = torch.ops.genesis.pn156_norma_q8(h, r, n.weight, lin.input_global_scale,
                                                  float(n.variance_epsilon))
    return h, rr, (q, esc)


def lineal(lin, x: torch.Tensor, q156=None) -> torch.Tensor:
    """Reemplaza ``y, _ = lin(x)`` en la consumidora: con _q156, Marlin W4A8 sobre el int8 ya hecho."""
    if q156 is None:
        y, _ = lin(x)
        return y
    from vllm._genesis import rot_down as _g148
    q, esc = q156
    return _g148._gemm_int8(lin, _marlin(lin), q, esc)
