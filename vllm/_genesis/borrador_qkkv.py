# SPDX-License-Identifier: Apache-2.0
"""PN161 — atencion del borrador DFlash2: de qkv a la atencion en un kernel (SK-33).

Por capa del borrador corrian, entre el GEMM de qkv y la atencion: q_norm + k_norm + rope (inductor), la FWHT de
PN126 sobre q y sobre k (sk_fwht128 x2) y la escritura de la KV int8 por token-cabeza de vLLM: 4 kernels, ~7,8 us.
SK-33 hace todo eso en uno y escribe k y v en la cache; la atencion se llama directo por el op de vLLM, sin el
paso de escritura de Attention.forward (TritonAttentionImpl solo usa key/value en atencion de encoder).
Si la capa no cumple lo que el kernel asume (cabeza 128, rope neox completo, KV int8 por token-cabeza), sigue
por el camino de siempre.
"""
from __future__ import annotations

import ctypes
import logging
import os

import torch

log = logging.getLogger("genesis.pn161")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN161_BORRADOR_QKKV", "0").strip().lower() in ("1", "true", "yes", "on")
_k: dict = {}
_cs: dict = {}


def _kern(rot: bool):
    clave = (torch.cuda.current_device(), rot)
    if clave not in _k:
        from vllm._genesis.kernels.ptx_lab import Kernel
        k = Kernel("sk33_borrador_qk_kv.cu", "sk33_qk_kv", defs=[f"-DROT={int(rot)}"], warps=4)
        k.cargar()
        _k[clave] = k
    return _k[clave]


def aplicable(attn) -> bool:
    if not ACTIVO:
        return False
    try:
        from vllm.v1.attention.backends.triton_attn import KVQuantMode
        impl = attn.attn.impl
        re = attn.rotary_emb
        ok = (attn.head_dim == 128 and re.is_neox_style and re.rotary_dim == 128 and re.head_size == 128
              and getattr(impl, "_is_per_token_head_quant", False)
              and getattr(impl, "_kv_quant_mode", None) == KVQuantMode.INT8_PER_TOKEN_HEAD
              and attn.attn.kv_sharing_target_layer_name is None
              and getattr(attn.qkv_proj, "bias", None) is None)
    except Exception as e:                                    # otra backend o version: camino de siempre
        log.warning("PN161: no aplica (%s)", e)
        return False
    if not ok and not getattr(attn, "_pn161_avisado", False):
        log.warning("PN161: la capa %s no cumple lo que asume SK-33; queda el camino de siempre", attn.layer_name)
        attn._pn161_avisado = True
    return ok


def _cos_sin(cs: torch.Tensor) -> torch.Tensor:
    clave = (cs.data_ptr(), cs.device)
    if clave not in _cs:
        _cs[clave] = cs.to(dtype=torch.float16).contiguous()
    return _cs[clave]


def _cuda(qkv: torch.Tensor, positions: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, cs: torch.Tensor,
          layer_name: str, nhq: int, nkv: int, eps: float, rot: bool) -> torch.Tensor:
    T = qkv.shape[0]
    qo = torch.empty((T, nhq * 128), dtype=qkv.dtype, device=qkv.device)
    if T == 0:
        return qo
    if qkv.stride(-1) != 1:
        qkv = qkv.contiguous()
    from vllm._genesis import rot_qk as _g126
    from vllm.model_executor.layers.attention.attention import get_attention_context
    signos = _g126._signos_dev(128, qkv.device)
    _, capa, kv_cache, slots = get_attention_context(layer_name)
    nulo = ctypes.c_uint64(0)
    if slots is None or kv_cache is None or kv_cache.numel() == 0:
        cache = [nulo, 1] + [nulo, ctypes.c_int64(0), ctypes.c_int64(0), ctypes.c_int64(0)] * 4
    else:
        impl = capa.impl
        kc, vc = impl._pth_key_value_caches(kv_cache)          # [bloques, BS, nkv, hs] (vistas)
        ks, vs = impl._k_scale_cache, impl._v_scale_cache      # [bloques, BS, nkv] float32
        if slots.dtype != torch.int64:
            slots = slots.to(torch.int64)
        c64 = lambda t: [t] + [ctypes.c_int64(int(x)) for x in t.stride()[:3]]
        cache = [slots, kc.shape[1]] + c64(kc) + c64(vc) + c64(ks) + c64(vs)
    pos = positions if positions.dtype == torch.int64 else positions.to(torch.int64)
    _kern(rot).lanzar((T, -(-(nhq + nkv) // 4)),
                      [qkv, qkv.stride(0), pos, wq, wk, float(eps), _cos_sin(cs), signos, nhq, nkv, qo, qo.stride(0)]
                      + cache)
    return qo


@torch.library.custom_op("genesis::pn161_qk_kv", mutates_args=())
def qk_kv_op(qkv: torch.Tensor, positions: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, cs: torch.Tensor,
             layer_name: str, nhq: int, nkv: int, eps: float, rot: bool) -> torch.Tensor:
    return _cuda(qkv, positions, wq, wk, cs, layer_name, nhq, nkv, eps, rot)


@qk_kv_op.register_fake
def _qk_kv_fake(qkv, positions, wq, wk, cs, layer_name, nhq, nkv, eps, rot):
    return qkv.new_empty((qkv.shape[0], nhq * 128))


def preparar(attn) -> None:
    """Al final de DFlashQwen3Attention.__init__: la decision queda en atributos (constantes para dynamo)."""
    from vllm._genesis import rot_qk as _g126
    attn._pn161 = aplicable(attn)
    attn._pn161_rot = _g126.activo(128)


def qk_kv(attn, qkv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """q lista para la atencion (normada, con rope y rotada); k y v ya quedan en la cache."""
    return torch.ops.genesis.pn161_qk_kv(
        qkv, positions, attn.q_norm.weight, attn.k_norm.weight, attn.rotary_emb.cos_sin_cache,
        attn.attn.layer_name, attn.num_heads, attn.num_kv_heads, attn.q_norm.variance_epsilon, attn._pn161_rot)


def o_proj(attn, x: torch.Tensor) -> torch.Tensor:
    from vllm._genesis import borrador_fusion as _g160
    return _g160.lineal(attn.o_proj, x)


def atencion(attn, q: torch.Tensor) -> torch.Tensor:
    """Attention.forward sin el paso de escritura de la KV (ya la hizo SK-33). El op de vLLM exige key y value
    como Tensor aunque en decoder no los lee (TritonAttentionImpl solo los usa en encoder): va q de relleno."""
    from vllm.model_executor.layers.attention import attention as _a
    a = attn.attn
    out = torch.empty((q.shape[0], a.num_heads * a.head_size_v), dtype=q.dtype, device=q.device)
    qv = q.view(-1, a.num_heads, a.head_size)
    ov = out.view(-1, a.num_heads, a.head_size_v)
    if a.use_direct_call:
        _a.unified_attention_with_output(qv, qv, qv, ov, a.layer_name)
    else:
        torch.ops.vllm.unified_attention_with_output(qv, qv, qv, ov, _a._encode_layer_name(a.layer_name))
    return out
