# SPDX-License-Identifier: Apache-2.0
"""PN148: Hadamard en linea por bloques antes de down_proj (checkpoints con residuo rotado).

El checkpoint rotado (entrenamiento/cuant/rotacion.py) guarda down_proj como  A = R W Hd, con Hd la
Hadamard por bloques de B sobre la dimension intermedia. Para que la cuenta de exacta, la entrada de
down_proj tiene que llegar multiplicada por Hd:  y = (u Hd) A^T. Hd es simetrica y ortogonal.

Con TP=2 cada rango tiene su mitad de la dimension intermedia (8704 = 17 x 512): los bloques no
cruzan la particion, asi que cada rango aplica su parte sola, sin comunicacion. Y el int8 por token
de Marlin cuantiza la entrada YA rotada, que es donde la cresta baja de ~27 a ~4.

B sale de GENESIS_PN148_BLOQUE (512 por defecto). Sin checkpoint rotado este parche ROMPE el modelo:
solo se prende con GENESIS_ENABLE_PN148_ROT_DOWN=1 junto con el checkpoint que lo pide.
"""
from __future__ import annotations

import os

import torch

B = int(os.environ.get("GENESIS_PN148_BLOQUE", "512"))
# Kernel fusionado SiluAndMul + Hadamard (silu_had.py): 624 us contra 2042 del camino de torch a 8192
# tokens, y la mitad de error. Solo existe para B = 512.
FUSIONADO = os.environ.get("GENESIS_PN148_FUSIONADO", "1") == "1" and B == 512
# SK-23: ademas sale int8 por token y down_proj (Marlin W4A8) lo toma sin volver a cuantizar.
INT8 = FUSIONADO and os.environ.get("GENESIS_PN148_INT8", "1") == "1"
if FUSIONADO:
    from vllm._genesis import silu_had  # noqa: F401  (registra las ops genesis::pn148_*)


def hadamard() -> torch.Tensor:
    """H de B x B normalizada, en el dispositivo y dtype por defecto del contexto: se llama desde el
    __init__ de la MLP, que vLLM construye bajo el dispositivo y el dtype del modelo. Va como buffer
    del modulo (nada que cachear en el forward: torch.compile + grafos no lo permiten)."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < B:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / B ** 0.5).to(torch.get_default_dtype())


def act_had(mlp, gate_up: torch.Tensor) -> torch.Tensor:
    """Lo que va entre gate_up y down_proj: silu(g)*u y la Hadamard por bloques."""
    if FUSIONADO:
        return torch.ops.genesis.pn148_silu_had(gate_up)
    return aplicar(mlp.act_fn(gate_up), mlp._g148_h)


def _marlin_int8(down):
    """El kernel Marlin de down_proj si corre con activacion int8 (W4A8); si no, None. Se evalua al
    trazar: es estatico por capa."""
    sch = getattr(down, "scheme", None)
    k = getattr(sch, "kernel", None)
    c = getattr(k, "config", None)
    if (k is None or getattr(c, "act_type", None) != torch.int8
            or getattr(down, "input_global_scale", None) is None or not hasattr(k, "_get_weight_params")):
        return None
    return k


_FP32_REDUCE = None


def _gemm_int8(down, k, q: torch.Tensor, esc: torch.Tensor) -> torch.Tensor:
    """Lo mismo que apply_gptq_marlin_linear con input_dtype=int8, sin el paso de cuantizar."""
    global _FP32_REDUCE
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
    if _FP32_REDUCE is None:
        import inspect
        _FP32_REDUCE = inspect.signature(mu.apply_gptq_marlin_linear).parameters["use_fp32_reduce"].default
    c = k.config
    w_q, w_s, w_zp, w_gidx = k._get_weight_params(down)
    n_part = c.partition_weight_shape[1]
    padded_n, padded_k = mu.marlin_repacked_nk(w_q, c.weight_type.size_bits)
    assert padded_k == q.shape[1], (padded_k, q.shape)
    m = q.shape[0]
    atom = mu.should_use_atomic_add_reduce(m=m, n=padded_n, k=padded_k, device=q.device, dtype=torch.float16)
    try:
        from vllm._genesis import marlin_s16 as _g130
        gemm = _g130.marlin_gemm if _g130.ACTIVO_Y_CARGADO else None
    except Exception:  # noqa: BLE001
        gemm = None
    if gemm is None:
        from vllm import _custom_ops as ops
        gemm = ops.marlin_gemm
    out = gemm(q, None, w_q, None, w_s, esc, None, w_zp, w_gidx, down.g_idx_sort_indices, k.workspace,
               c.weight_type, size_m=m, size_n=padded_n, size_k=padded_k, is_k_full=k.is_k_full,
               use_atomic_add=atom, use_fp32_reduce=_FP32_REDUCE, is_zp_float=False)
    return mu.marlin_unpad_output(out, n_part, padded_n)


def _reducir(down, out: torch.Tensor) -> torch.Tensor:
    """El all-reduce de RowParallelLinear, con PN120 si esta activo (igual que su parche)."""
    if down.reduce_results and down.tp_size > 1:
        from vllm._genesis import ar_int8 as _g120
        if _g120.activo():
            return _g120.all_reduce_int8(out)
        from vllm.distributed import tensor_model_parallel_all_reduce
        return tensor_model_parallel_all_reduce(out)
    return out


VERIFICAR = os.environ.get("GENESIS_PN148_VERIFICAR", "0") == "1"   # solo con --enforce-eager
_n_verif: dict = {}


def _verificar(mlp, gate_up, out):
    """Compara la salida local (antes del all-reduce) con el camino de siempre: fp16 -> down_proj
    cuantizando adentro de Marlin. Registra las primeras 3 llamadas de cada capa."""
    import logging
    i = id(mlp)
    if _n_verif.get(i, 0) >= 3:
        return
    _n_verif[i] = _n_verif.get(i, 0) + 1
    d = mlp.down_proj
    ref = d.quant_method.apply(d, act_had(mlp, gate_up), None).reshape(out.shape)
    rel = float((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12))
    logging.getLogger("genesis.pn148").warning("[PN148 verificar] %s M=%d err rel int8 directo vs camino fp16: %.2e",
                                               d.prefix if hasattr(d, "prefix") else "?", out.shape[0], rel)


def act_had_down(mlp, gate_up: torch.Tensor) -> torch.Tensor:
    """gate_up -> salida de down_proj (ya reducida). Con SK-23 el int8 va directo a Marlin."""
    d = mlp.down_proj
    k = _marlin_int8(d) if INT8 else None
    if k is not None and d.bias is None:
        q, esc = torch.ops.genesis.pn148_silu_had_q8(gate_up, d.input_global_scale)
        out = _gemm_int8(d, k, q, esc)
        if VERIFICAR:
            _verificar(mlp, gate_up, out)
        return _reducir(d, out.reshape(*gate_up.shape[:-1], out.shape[-1]))
    out, _ = d(act_had(mlp, gate_up))
    return out


def aplicar(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return (x.view(*x.shape[:-1], -1, B) @ h).view(x.shape)
