# SPDX-License-Identifier: Apache-2.0
"""SK-31: atencion del borrador DFlash2 en una pasada (kernels/cuda/sk31_borrador_attn.cu), desde PN124.

El borrador verifica bloques de 9 queries NO causales con ventana simetrica de 2048 sobre la KV
int8_per_token_head de vLLM (registro de 264 B por slot y cabeza: K 128 | escala fp32 | V 128 | escala fp32).
Hoy va por kernel_unified_attention (Triton, PN124 3D). SK-31: softmax en linea en fp32, Q.K en fp16 (K int8
-> fp16 en shared), P.V fp16 con acumulador fp16 por tile; reparto en GMAX grupos de keys dentro de la ventana
segun el seq_len real (vale en grafo). Error contra float con la KV decuantizada 0,06% (Triton 0,03%).

Se engancha en ``triton_attn_ampere.llamar`` y solo toma la llamada si es exactamente la del borrador; si no,
sigue Triton. ``GENESIS_PN124_SK31=1``.
"""
from __future__ import annotations

import ctypes
import logging
import math
import os

import torch

log = logging.getLogger("genesis.sk31")
ACTIVO = os.environ.get("GENESIS_PN124_SK31", "0") == "1"
GMAX = int(os.environ.get("GENESIS_PN124_SK31_GMAX", "32"))
BMAX = int(os.environ.get("GENESIS_PN124_SK31_BMAX", "16"))
RB, QD = 64, 128
SH = 2 * 64 * 144 + 2 * 64 * 128 + 2 * 64 * 136 * 2 + 64 * 72 * 2 + 2 * 64 * 2 * 4 + 2 * 64 * 4
_est: dict = {}
_aviso = [True]


def _kernels():
    if "k" not in _est:
        from vllm._genesis.kernels.ptx_lab import Kernel
        k = Kernel("sk31_borrador_attn.cu", "sk31_borrador", warps=8)
        u = Kernel("sk31_borrador_attn.cu", "sk31_union", warps=4)
        k.cargar(); u.cargar()
        _est["k"], _est["u"] = k, u
    return _est["k"], _est["u"]


def _bufs(dev, nkv):
    clave = (dev.index, nkv)
    b = _est.get(clave)
    if b is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        b = _est[clave] = (torch.empty(GMAX * BMAX * nkv * RB * QD, dtype=torch.float16, device=dev),
                           torch.empty(GMAX * BMAX * nkv * RB, dtype=torch.float32, device=dev),
                           torch.empty(GMAX * BMAX * nkv * RB, dtype=torch.float32, device=dev))
        log.warning("SK-31: atencion del borrador en una pasada (GMAX=%d, hasta %d pedidos, %.0f MiB)",
                    GMAX, BMAX, b[0].numel() * 2 / 2**20)
    return b


def intentar(kw) -> bool:
    """True si la llamada la resolvio SK-31 (kw['out'] escrito); False para seguir con Triton."""
    if not ACTIVO:
        return False
    try:
        from vllm.v1.kv_cache_interface import KVQuantMode
        q, kc, out = kw["q"], kw["k"], kw["out"]
        ws = kw.get("window_size")
        if (kw.get("kv_quant_mode") != KVQuantMode.INT8_PER_TOKEN_HEAD or q.dim() != 3 or q.shape[-1] != QD
                or kc.dtype != torch.int8 or kc.dim() != 4 or kc.shape[-1] != QD + 4 or kw.get("causal", True)
                or ws is None or ws[0] < 0 or kw.get("alibi_slopes") is not None or kw.get("sinks") is not None
                or (kw.get("softcap") or 0) != 0 or kw.get("mm_prefix_range") is not None
                or kw.get("rswa_prefix_lens") is not None or q.dtype != torch.float16 or out.dtype != torch.float16):
            return False
        L = int(kw["max_seqlen_q"])
        seq = kw["seqused_k"]
        B = int(seq.shape[0])
        nkv = int(kc.shape[2])
        G = int(q.shape[1]) // nkv
        if q.shape[0] != B * L or L * G > RB or B > BMAX or q.stride(-1) != 1 or out.stride(-1) != 1:
            return False
        bufs = _bufs(q.device, nkv)
        if bufs is None:
            return False
        k31, ku = _kernels()
        Op, Mp, Lp = bufs
        bt = kw["block_table"]
        esc = float(kw["softmax_scale"]) * math.log2(math.e)
        k31.lanzar((GMAX, B * nkv), [q, q.stride(0), kc, ctypes.c_int64(kc.stride(0)), ctypes.c_int64(kc.stride(2)),
                                     ctypes.c_int64(kc.stride(1)), bt, bt.stride(0), seq, nkv, G, L, int(kc.shape[1]),
                                     int(ws[0]) + 1, esc, GMAX, Op, Mp, Lp], shared=SH)
        ku.lanzar((RB, B * nkv), [Op, Mp, Lp, GMAX, B, nkv, L, G, out, out.stride(0)])
        if _aviso[0] and not torch.cuda.is_current_stream_capturing():
            _aviso[0] = False
            log.warning("SK-31: primera llamada del borrador (B=%d, L=%d, G=%d, ventana %d)", B, L, G, int(ws[0]) + 1)
        return True
    except Exception as e:       # nunca romper el paso: sigue Triton
        if _aviso[0]:
            _aviso[0] = False
            log.error("SK-31: no se uso (%s: %s); sigue Triton", type(e).__name__, e)
        return False
