# SPDX-License-Identifier: Apache-2.0
"""Genesis PN124: TRITON_ATTN rápido en Ampere (SM86) para head_dim 256.

Por qué
-------
Medido con las formas del Qwen3.8 (12 q / 2 kv por rank, head 256, bloque 832):

* **Prefill**: ``unified_attention`` lanza con ``BLOCK_M=16``, o sea
  ``BLOCK_Q = 16 // 6 = 2`` queries por programa y tiles de 32: cada programa
  relee todo el contexto para 2 queries. Upstream tiene un afinado para head 256
  pero lo ata a Blackwell (``is_device_capability_family(100)``) y además pide
  155.648 de SRAM compartida contra un límite de 101.376 en Ampere.
  ``BLOCK_M=64, TILE=64, stages=1`` entra y da 29,1 ms contra 82,4 ms (2,84x),
  igual que FlashAttention paginada (28,4 ms).

* **Decode con MTP**: el kernel 3D (softmax por segmentos, paralelo a lo largo
  de la KV) exige ``max_seqlen_q == 1``. Con K=3 hay 4 queries por request y
  siempre cae al 2D, que recorre la KV en serie. Con 1 query @57k: 3,40 ms en
  2D contra 0,31 ms en 3D (10,8x).

Qué hace
--------
1. Parámetros de lanzamiento del camino 2D para SM8x + head 256 (texto en
   ``triton_unified_attention.py``). Si la compilación no entra en la SRAM
   (modos de KV cuantizados agregan código al kernel), baja UNA vez a la
   siguiente configuración y la deja fija.
2. ``llamar``: envuelve la llamada del backend. Si el batch es uniforme
   (``q.shape[0] == N * Q``, con ``Q = max_seqlen_q`` chico), lo aplana en
   ``N*Q`` pseudo-secuencias de 1 query: la i-ésima query de un request con
   ``L`` tokens ve ``L - Q + i + 1`` tokens, con la misma fila del block table.
   Es exactamente la máscara causal del caso original, y entra al kernel 3D.
"""

from __future__ import annotations

import logging
import os

import torch

from vllm._genesis import borrador_attn as _sk31

log = logging.getLogger("genesis.pn124")

_ACTIVO = os.environ.get("GENESIS_ENABLE_PN124_TRITON_AMPERE", "0") == "1"
_APLANAR = os.environ.get("GENESIS_PN124_APLANAR_SPEC", "1") == "1"
# 16 (28-09): el borrador DFlash2 verifica bloques de 9 tokens y con 8 caia siempre al 2D (12 bloques
# en 82 SM, ~1 ms por paso en sus 5 capas).
_Q_MAX = int(os.environ.get("GENESIS_PN124_Q_MAX", "16"))
_VERIFICAR = [int(os.environ.get("GENESIS_PN124_VERIFICAR", "0"))]
# Queries por secuencia con las que se deja entrar al kernel 3D SIN aplanar (upstream: 1). 1 = apagado.
_Q3D = [int(os.environ.get("GENESIS_PN124_Q3D", "1")) if _ACTIVO else 1]


def q_max_3d() -> int:
    return _Q3D[0]


# Configuraciones de prefill de mejor a peor (BLOCK_M, TILE, num_stages, num_warps).
# La primera es la medida; las siguientes son para cuando el kernel crece
# (per-token-head) y no entra en la SRAM compartida de Ampere.
_CONFIGS = [(64, 64, 1, 8), (32, 64, 2, 8), (32, 32, 2, 4), (16, 32, 3, 4)]
_nivel = 0
_es_ampere = None


def activo() -> bool:
    return _ACTIVO


def ampere_head256(head_size: int) -> bool:
    global _es_ampere
    if not _ACTIVO or head_size != 256:
        return False
    if _es_ampere is None:
        try:
            _es_ampere = torch.cuda.get_device_capability()[0] == 8
        except Exception:
            _es_ampere = False
    return _es_ampere


# INT4 per-token-head tiene su propio kernel (split-dot sobre nibbles): barrido
# aparte (tests/proto/pn124_int4_barrido.py): BLOCK_M=32 TILE=32 stages=1
# warps=4 da 1,35x; BLOCK_M=64 TILE=64 lo hace 0,5-0,65x.
_CONFIGS_INT4 = [(32, 32, 1, 4), (16, 32, 3, 4)]
_nivel_int4 = 0


def params() -> tuple[int, int, int, int]:
    return _CONFIGS[_nivel]


def params_int4() -> tuple[int, int, int, int]:
    return _CONFIGS_INT4[_nivel_int4]


def tile_decode_int4() -> int:
    """Tile del kernel INT4 en el camino 3D (decode). Upstream elige 16 porque la
    query es fp16; medido @57k con decode spec aplanado: 16 -> 0,78 ms,
    32 -> 0,40 ms, 64 -> 0,44 ms, 128 -> 0,59 ms."""
    return 32


def _es_sin_sram(e: Exception) -> bool:
    return "out of resource" in str(e) and "shared memory" in str(e)


def _aplanar(kw: dict) -> dict | None:
    q = kw["q"]
    Q = int(kw["max_seqlen_q"])
    seqused = kw["seqused_k"]
    N = int(seqused.shape[0])
    if Q <= 1 or Q > _Q_MAX or q.shape[0] != N * Q:
        return None
    thr = kw.get("seq_threshold_3D")
    if thr is None or kw.get("softmax_segm_output") is None or N * Q > thr:
        return None
    if kw.get("alibi_slopes") is not None or kw.get("sinks") is not None:
        return None
    ws = kw.get("window_size")
    if ws is not None and ws[0] >= 0:
        return None
    if kw.get("mm_prefix_range") is not None or kw.get("rswa_prefix_lens") is not None:
        return None
    dev = q.device
    nuevo = dict(kw)
    if kw.get("causal", True):
        desp = torch.arange(Q, device=dev, dtype=seqused.dtype)
        nuevo["seqused_k"] = (seqused.unsqueeze(1) - (Q - 1) + desp.unsqueeze(0)).reshape(-1).clamp_min_(0)
    else:
        # No causal (el borrador DFlash: cada query del bloque ve todo el contexto y todo el bloque): cada
        # pseudo-secuencia de 1 query ve los L tokens enteros. Con 1 query por secuencia la mascara causal
        # del kernel ya deja ver todo, asi que se lanza como causal.
        nuevo["seqused_k"] = seqused.repeat_interleave(Q)
        nuevo["causal"] = True
    nuevo["cu_seqlens_q"] = torch.arange(N * Q + 1, device=dev, dtype=kw["cu_seqlens_q"].dtype)
    nuevo["max_seqlen_q"] = 1
    nuevo["block_table"] = kw["block_table"].repeat_interleave(Q, dim=0)
    for nombre in ("k_descale", "v_descale"):
        t = kw.get(nombre)
        if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == N:
            nuevo[nombre] = t.repeat_interleave(Q, dim=0)
    return nuevo


def llamar(fn, **kw):
    """Reemplazo de ``unified_attention(**kw)`` en el backend Triton."""
    global _nivel, _nivel_int4
    if _sk31.ACTIVO and _sk31.intentar(kw):          # SK-31: atencion del borrador en una pasada
        return None
    if _ACTIVO and _APLANAR:
        plano = _aplanar(kw)
        Qk = int(kw["max_seqlen_q"])
        thr = kw.get("seq_threshold_3D")
        if (plano is None and 1 < Qk <= _Q3D[0] and _VERIFICAR[0] > 0 and thr is not None
                and kw["q"].shape[0] <= thr and not torch.cuda.is_current_stream_capturing()
                and int(kw["seqused_k"].max()) > 64):              # solo llamadas reales que van al 3D
            # 3D con varias queries contra el 2D de siempre, sobre la misma llamada
            _VERIFICAR[0] -= 1
            q3d, _Q3D[0] = _Q3D[0], 1
            fn(**kw)
            ref = kw["out"].clone()
            _Q3D[0] = q3d
            fn(**kw)
            d = float((kw["out"].float() - ref.float()).abs().max())
            m = float(ref.float().abs().max())
            log.warning("[PN124 verificar 3D] Q=%d N=%d causal=%s ws=%s: max|dif| %.3e (max|ref| %.3e)",
                        Qk, int(kw["seqused_k"].shape[0]), kw.get("causal"), kw.get("window_size"), d, m)
            return None
        if plano is None and _VERIFICAR[0] > 0 and not torch.cuda.is_current_stream_capturing():
            _VERIFICAR[0] -= 1
            log.warning("[PN124 diag] sin aplanar: Q=%s q=%s N=%s thr=%s segm=%s causal=%s head=%s alibi=%s sinks=%s ws=%s mm=%s rswa=%s",
                        kw.get("max_seqlen_q"), tuple(kw["q"].shape), int(kw["seqused_k"].shape[0]),
                        kw.get("seq_threshold_3D"), kw.get("softmax_segm_output") is not None, kw.get("causal"),
                        kw["q"].shape[-1], kw.get("alibi_slopes") is not None, kw.get("sinks") is not None,
                        kw.get("window_size"), kw.get("mm_prefix_range") is not None, kw.get("rswa_prefix_lens") is not None)
        if plano is not None:
            if _VERIFICAR[0] > 0 and not torch.cuda.is_current_stream_capturing():
                _VERIFICAR[0] -= 1
                fn(**kw)
                ref = kw["out"].clone()
                fn(**plano)
                d = float((plano["out"].float() - ref.float()).abs().max())
                m = float(ref.float().abs().max())
                log.warning("[PN124 verificar] Q=%d N=%d causal=%s: max|dif| %.3e (max|ref| %.3e)",
                            int(kw["max_seqlen_q"]), int(kw["seqused_k"].shape[0]), kw.get("causal"), d, m)
                return None
            kw = plano
    while True:
        try:
            return fn(**kw)
        except Exception as e:
            if not (_ACTIVO and _es_sin_sram(e)):
                raise
            # Se degradan las dos listas un escalon: la llamada no dice cual de los
            # dos kernels (fp16/int8 o INT4 empaquetado) fue el que no entro.
            bajo = False
            if _nivel_int4 + 1 < len(_CONFIGS_INT4):
                _nivel_int4 += 1
                bajo = True
            if _nivel + 1 < len(_CONFIGS):
                _nivel += 1
                bajo = True
            if not bajo:
                raise
            log.warning("[PN124] el kernel no entra en la SRAM de Ampere; bajo a %s / int4 %s",
                        _CONFIGS[_nivel], _CONFIGS_INT4[_nivel_int4])


__all__ = ["activo", "ampere_head256", "params", "params_int4", "tile_decode_int4", "llamar"]
