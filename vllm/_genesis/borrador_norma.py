# SPDX-License-Identifier: Apache-2.0
"""PN162 — normas del borrador DFlash2: cierre de la conv + suma del residuo + RMSNorm en un kernel (SK-34).

Cada capa DFlash2 cierra dos convs (attention_conv.finish, mlp_conv.finish) y cada cierre va seguido de una
RMSNorm con residuo (post_attention_layernorm; input_layernorm de la capa siguiente; la norma final). Inductor
las funde en un kernel por norma (~4,4 us con 9 tokens, 11 por paso) con una CTA chica por fila. SK-34: un bloque
de 640 hilos por fila, una pasada en registros (3,5 us), igual a inductor salvo el orden de la suma (~1e-4 de los
valores a 1 LSB). Como el cierre de la conv del MLP de una capa se junta con la norma de entrada de la siguiente,
el lazo de capas del modelo pasa a ``forward_capas`` (misma cuenta que DFlash2Qwen3DecoderLayer.forward, con
PN160 si esta prendido).
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn162")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN162_BORRADOR_NORMA", "0").strip().lower() in ("1", "true", "yes", "on")
_k: dict = {}
N_KERNEL = 5120


def _kern():
    clave = torch.cuda.current_device()
    if clave not in _k:
        from vllm._genesis.kernels.ptx_lab import Kernel
        k = Kernel("sk34_borrador_fin_norma.cu", "sk34_fin_norma", warps=20)
        k.cargar()
        _k[clave] = k
    return _k[clave]


def _cuda(h, coef, base, res, w, eps: float, bsq: int):
    T, N = h.shape
    out = torch.empty_like(h)
    ro = torch.empty_like(h)
    if T:
        if h.stride(-1) != 1:
            h = h.contiguous()
        if res.stride(-1) != 1:
            res = res.contiguous()
        _kern().lanzar((T, 1), [h, h.stride(0), coef, coef.stride(0), base, res, res.stride(0), w, float(eps), N, bsq,
                                out, out.stride(0), ro, ro.stride(0)])
    return out, ro


@torch.library.custom_op("genesis::pn162_fin_norma", mutates_args=())
def fin_norma_op(h: torch.Tensor, coef: torch.Tensor, base: torch.Tensor, res: torch.Tensor, w: torch.Tensor,
                 eps: float, bsq: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _cuda(h, coef, base, res, w, eps, bsq)


@fin_norma_op.register_fake
def _fin_norma_fake(h, coef, base, res, w, eps, bsq):
    return torch.empty_like(h), torch.empty_like(h)


def aplica(model) -> bool:
    """En el lazo del modelo (constante para dynamo): solo capas DFlash2 con lo que asume SK-34."""
    if not ACTIVO or not len(model.layers) or not hasattr(model.layers[0], "mlp_conv"):
        return False
    c = model.layers[0].mlp_conv
    ok = (c.taps == 2 and c.group_size == 16 and c.num_groups * 16 == N_KERNEL
          and c.base_kernel.dtype == torch.float16 and model.layers[0].input_layernorm.weight is not None)
    if not ok and not getattr(model, "_pn162_avisado", False):
        log.warning("PN162: el borrador no cumple lo que asume SK-34 (taps 2, grupos de 16, N %d, fp16)", N_KERNEL)
        model._pn162_avisado = True
    return ok


def fin_norma(conv, h: torch.Tensor, coef: torch.Tensor, residual: torch.Tensor, norma):
    """``norma(conv.finish(h, coef), residual)`` en un kernel."""
    return torch.ops.genesis.pn162_fin_norma(h, coef, conv.base_kernel[1], residual, norma.weight,
                                             norma.variance_epsilon, conv.block_size)


def forward_capas(model, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """El lazo de DFlashQwen3Model.forward para capas DFlash2, con los cierres de conv fundidos en las normas."""
    from vllm._genesis import borrador_fusion as _g160
    residual = None
    pendiente = None                                           # (conv, coeficientes) del MLP de la capa anterior
    for layer in model.layers:
        if pendiente is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fin_norma(pendiente[0], hidden_states, pendiente[1], residual, layer.input_layernorm)
        hidden_states, coef, q160 = _g160.preparar(layer.attention_conv, hidden_states, layer.self_attn.qkv_proj)
        if q160 is None:
            hidden_states = layer.self_attn(positions=positions, hidden_states=hidden_states)
        else:
            hidden_states = layer.self_attn(positions=positions, hidden_states=hidden_states, _q160=q160)
        hidden_states, residual = fin_norma(layer.attention_conv, hidden_states, coef, residual,
                                            layer.post_attention_layernorm)
        hidden_states, coef, q160 = _g160.preparar(layer.mlp_conv, hidden_states, layer.mlp.gate_up_proj)
        hidden_states = layer.mlp(hidden_states) if q160 is None else layer.mlp(hidden_states, q160)
        pendiente = (layer.mlp_conv, coef)
    hidden_states, _ = fin_norma(pendiente[0], hidden_states, pendiente[1], residual, model.norm)
    return hidden_states
