# SPDX-License-Identifier: Apache-2.0
"""PN157 — las dos lineales densas del borrador DFlash2, a Marlin.

Medido en el perfil de decode (28-09), por paso:

* K/V del contexto: PN142 decuantiza las filas k/v de las 5 capas a UNA matriz fp16 [5120 x 5120] por
  rango (52 MB) y ``_project_context_kv`` hace F.linear: 69 us, en el techo de DRAM. Con Marlin W4A16
  sobre los MISMOS int4 y escalas del checkpoint son 13 MB: los valores del peso son identicos, solo
  cambia el orden de la suma.
* ``kernel_projection`` de las dos convs de cada capa (10 por paso, [1280 x 5120], bf16 en el checkpoint
  e ignoradas por la cuantizacion): 23 us cada una = 229 us. Se cuantizan al cargar con RTN simetrico
  (int8 por canal, o int4 g128) y corren en Marlin W8A16 / W4A16.

Todo se arma al final de load_weights (desde ``_build_context_kv_buffers``), antes de torch.compile:
en el forward solo se leen atributos, estaticos al trazar.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn157")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN157_BORRADOR_MARLIN", "0").strip().lower() in ("1", "true", "yes", "on")
BITS_CONV = int(os.environ.get("GENESIS_PN157_BITS_CONV", "0"))     # 8, 4, o 0 (densa: int8 por canal cuesta ~1% de aceptacion)


def _empacar_gptq(q: torch.Tensor, bits: int) -> torch.Tensor:
    """q sin signo [N, K] (0..2^bits-1) -> formato GPTQ [K * bits / 32, N] int32, bits bajos primero."""
    n, k = q.shape
    por = 32 // bits
    qt = q.t().contiguous().to(torch.int64).view(k // por, por, n)
    w = torch.zeros(k // por, n, dtype=torch.int64, device=q.device)
    for i in range(por):
        w |= qt[:, i, :] << (bits * i)
    return torch.where(w >= 1 << 31, w - (1 << 32), w).to(torch.int32)


class _Marlin:
    """Peso en Marlin (W{4,8}A16, simetrico, sin zero point) listo para apply_gptq_marlin_linear."""

    def __init__(self, q_con_signo: torch.Tensor, escala: torch.Tensor, bits: int, grupo: int):
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
        from vllm.scalar_type import scalar_types

        n, k = q_con_signo.shape
        dev = q_con_signo.device
        q = (q_con_signo.to(torch.int32) + (1 << (bits - 1))).clamp_(0, (1 << bits) - 1)
        self.w = ops.gptq_marlin_repack(_empacar_gptq(q, bits), torch.empty(0, dtype=torch.int, device=dev),
                                        k, n, bits)
        g = k if grupo <= 0 else grupo
        s = escala.to(torch.float16).reshape(n, k // g).t().contiguous()      # [K/g, N]
        self.s = mu.marlin_permute_scales(s, k, n, grupo if grupo > 0 else -1)
        self.zp = mu.marlin_make_empty_g_idx(dev)
        self.gi = mu.marlin_make_empty_g_idx(dev)
        self.gs = mu.marlin_make_empty_g_idx(dev)
        self.ws = mu.marlin_make_workspace_new(dev)
        self.tipo = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
        self.n, self.k = n, k

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear
        return apply_gptq_marlin_linear(x, self.w, self.s, self.zp, self.gi, self.gs, self.ws, self.tipo,
                                        self.n, self.k, True)


def _rtn(w: torch.Tensor, bits: int, grupo: int):
    """RTN simetrico: (q con signo [N, K], escala [N, K/g])."""
    n, k = w.shape
    g = k if grupo <= 0 else grupo
    wf = w.float().view(n, k // g, g)
    qmax = (1 << (bits - 1)) - 1
    s = wf.abs().amax(-1).clamp_min(1e-8) / qmax
    q = torch.round(wf / s[..., None]).clamp_(-qmax - 1, qmax)
    return q.view(n, k).to(torch.int8), s


def marlin_de_empacado(packed: torch.Tensor, escala: torch.Tensor, in_f: int) -> _Marlin:
    """Desde compressed-tensors pack-quantized (int4/int8 simetrico por grupo, empacado sobre la entrada)."""
    from compressed_tensors.compressors.pack_quantized.base import unpack_from_int32
    out_f = int(packed.shape[0])
    bits = 32 * packed.shape[1] // in_f
    q = unpack_from_int32(packed.data, bits, torch.Size([out_f, in_f]), packed_dim=1)
    return _Marlin(q, escala, bits, in_f // escala.shape[1])


def preparar(modelo, layers_attn) -> None:
    """Desde _build_context_kv_buffers (fin de load_weights): arma los pesos Marlin del borrador."""
    if not ACTIVO:
        return
    try:
        qkv = [a.qkv_proj for a in layers_attn]
        if all(hasattr(p, "weight_packed") and hasattr(p, "weight_scale") for p in qkv):
            qs = layers_attn[0].q_size
            packed = torch.cat([p.weight_packed.data[qs:] for p in qkv], 0)
            escala = torch.cat([p.weight_scale.data[qs:] for p in qkv], 0)
            modelo._g157_kv = marlin_de_empacado(packed, escala, int(qkv[0].input_size))
            modelo._fused_kv_weight = modelo._fused_kv_weight[:0]     # libera los 52 MB densos
            log.info("PN157: K/V de contexto en Marlin W%dA16 [%d x %d]", 32 * packed.shape[1] // modelo._g157_kv.k,
                     modelo._g157_kv.n, modelo._g157_kv.k)
        if BITS_CONV in (4, 8):
            n = 0
            for capa in modelo.layers:
                for nombre in ("attention_conv", "mlp_conv"):
                    conv = getattr(capa, nombre, None)
                    kp = getattr(conv, "kernel_projection", None)
                    if kp is None or getattr(kp, "bias", None) is not None:
                        continue
                    q, s = _rtn(kp.weight.data, BITS_CONV, 128 if BITS_CONV == 4 else -1)
                    kp._g157 = _Marlin(q, s, BITS_CONV, 128 if BITS_CONV == 4 else -1)
                    n += 1
            log.info("PN157: %d kernel_projection en Marlin W%dA16", n, BITS_CONV)
    except Exception as e:  # sin Marlin armado todo sigue por el camino denso
        log.warning("PN157: no se armo (%s: %s); camino denso", type(e).__name__, e)
        for a in ("_g157_kv",):
            if hasattr(modelo, a):
                delattr(modelo, a)


def kv_contexto(modelo, x: torch.Tensor, w: torch.Tensor, b):
    m = getattr(modelo, "_g157_kv", None)
    if m is None or b is not None:
        return torch.nn.functional.linear(x, w, b)
    return m(x)


def proyectar(kp, x: torch.Tensor) -> torch.Tensor:
    m = getattr(kp, "_g157", None)
    if m is None:
        return kp(x)
    return m(x)
