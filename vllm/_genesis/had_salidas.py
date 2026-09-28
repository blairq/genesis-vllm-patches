# SPDX-License-Identifier: Apache-2.0
"""PN154 — Hadamard por cabeza en la ENTRADA de o_proj (atencion) y out_proj (GDN), para idiotSavant v2.

El checkpoint v2 (entrenamiento/idiotsavant/idiotsavant.py, HAD_ENTRADA) guarda esas dos lineales como
A = R W Hc, con Hc la Hadamard por bloques de una cabeza (256 en la atencion, 128 en el GDN). Para que la
cuenta de exacta, la entrada tiene que llegar multiplicada por Hc: y = (x Hc) A^T. Hc es simetrica y
ortogonal. Por que:

  * o_proj / out_proj son las unicas lineales que quedaban con la entrada sin rotar (cresta 12-21): el int8
    por token de Marlin W4A8 les costaba 2-3% de error, y con la Hadamard ~1%;
  * el bloque es UNA cabeza: cada rango de TP tiene sus cabezas enteras (sin comunicacion) y el kernel que
    produce la entrada (la salida de la atencion, la norma con compuerta del GDN) tiene la cabeza entera
    en registros: ahi la Hadamard y el int8 salen gratis y desaparece el _per_token_quant de Marlin.

Se prende SOLO si el checkpoint lo declara (config.json -> genesis_rotacion.had_entrada): con un checkpoint
v1 este parche no hace nada. GENESIS_PN154_APAGAR=1 lo apaga igual (solo para depurar: con un v2 rompe).
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn154")
_APAGAR = os.environ.get("GENESIS_PN154_APAGAR", "0") == "1"
_cfg = {"leido": False, "had": {}}


def _had_entrada() -> dict:
    """{"o_proj": 256, "out_proj": 128} del checkpoint, o {} si no lo declara."""
    if not _cfg["leido"]:
        _cfg["leido"] = True
        try:
            from vllm.config import get_current_vllm_config
            hf = get_current_vllm_config().model_config.hf_config
            rot = getattr(hf, "genesis_rotacion", None) or {}
            _cfg["had"] = dict(rot.get("had_entrada", {})) if isinstance(rot, dict) else {}
        except Exception as e:  # noqa: BLE001
            log.warning("[PN154] no se pudo leer genesis_rotacion del config: %s", e)
    return {} if _APAGAR else _cfg["had"]


def hadamard(b: int, dtype=None) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < b:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / b ** 0.5).to(dtype or torch.get_default_dtype())


def marcar(lineal, nombre: str) -> None:
    """Desde el __init__ de la capa (bajo el dispositivo y dtype del modelo): si el checkpoint pide la
    Hadamard en la entrada de ``nombre`` (o_proj / out_proj), la deja como buffer de la lineal."""
    b = int(_had_entrada().get(nombre, 0))
    lineal._g154_b = b
    if b:
        lineal.register_buffer("_g154_h", hadamard(b), persistent=False)
        if not getattr(marcar, "_avisado", False):
            marcar._avisado = True
            log.warning("[PN154] Hadamard por cabeza en la entrada de %s (bloque %d)", _had_entrada(), b)


def rotar(lineal, x: torch.Tensor) -> torch.Tensor:
    """x @ Hc por bloques (una cabeza). Sin marca, x tal cual."""
    b = getattr(lineal, "_g154_b", 0)
    if not b:
        return x
    return (x.reshape(*x.shape[:-1], -1, b) @ lineal._g154_h).reshape(x.shape)


# ─── SK-25: la operacion por cabeza + Hadamard + int8 en un kernel, y Marlin W4A8 directo ──────────
FUSIONADO = os.environ.get("GENESIS_PN154_FUSIONADO", "1") == "1"
_k25: dict = {}


def _kernel25(modo: int, d: int, nh: int, w32: bool):
    clave = (torch.cuda.current_device(), modo, d, nh, w32)
    if clave not in _k25:
        from vllm._genesis.kernels.ptx_lab import Kernel
        nw = min(nh, 32)
        porw = -(-nh // nw)
        k = Kernel("sk25_cabeza_had_q8.cu", "sk25_cabeza_had_q8",
                   defs=[f"-DMODO={modo}", f"-DD={d}", f"-DNW={nw}", f"-DPORW={porw}", f"-DW32={int(w32)}"],
                   warps=nw)
        k.cargar()
        _k25[clave] = k
    return _k25[clave]


def cabeza_q8_cuda(x: torch.Tensor, a: torch.Tensor, w: torch.Tensor, gscale: torch.Tensor,
                   modo: int, d: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    x2 = x.reshape(-1, x.shape[-1]) if x.dim() == 2 else x.reshape(x.shape[0], -1)
    a2 = a.reshape(x2.shape[0], -1) if a.dim() != 2 else a
    if x2.stride(-1) != 1 or x2.stride(0) % 8:
        x2 = x2.contiguous()
    if a2.stride(-1) != 1 or a2.stride(0) % 8:
        a2 = a2.contiguous()
    T, N = x2.shape
    nh = N // d
    assert N % d == 0 and a2.shape[-1] >= N and x2.dtype == torch.float16 and a2.dtype == torch.float16
    q = torch.empty((T, N), dtype=torch.int8, device=x.device)
    esc = torch.empty((T, 1), dtype=torch.float32, device=x.device)
    if T:
        _kernel25(modo, d, nh, w.dtype == torch.float32).lanzar((T, 1), [x2, a2, w, q, esc, gscale, nh, x2.stride(0), a2.stride(0),
                                               q.stride(0), float(eps)])
    return q, esc


@torch.library.custom_op("genesis::pn154_cabeza_q8", mutates_args=())
def cabeza_q8(x: torch.Tensor, a: torch.Tensor, w: torch.Tensor, gscale: torch.Tensor,
              modo: int, d: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(int8 [T, N], escala fp32 [T, 1] ya por gscale) de Hadamard_d(op(x, a)) por cabeza."""
    return cabeza_q8_cuda(x, a, w, gscale, modo, d, eps)


@cabeza_q8.register_fake
def _cabeza_q8_fake(x, a, w, gscale, modo, d, eps):
    T = x.shape[0]
    N = x.numel() // T
    return x.new_empty((T, N), dtype=torch.int8), x.new_empty((T, 1), dtype=torch.float32)


def _marlin(lineal):
    """El kernel Marlin W4A8 de la lineal, o None. Solo lee atributos: al trazar es estatico por capa
    (nada de guardar cosas en el modulo desde el forward: torch.compile + grafos no lo permiten)."""
    if not FUSIONADO or not getattr(lineal, "_g154_b", 0) or lineal.bias is not None:
        return None
    from vllm._genesis import rot_down as _g148
    return _g148._marlin_int8(lineal)


def salida_gdn(gdn, core_attn_out: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Reemplaza _output_projection del GDN: RMSNormGated -> (Hadamard 128) -> out_proj."""
    lin = gdn.out_proj
    k = _marlin(lin)
    if k is not None and gdn.norm.norm_before_gate and gdn.norm.activation == "silu" \
            and gdn.norm.group_size is None:
        from vllm._genesis import rot_down as _g148
        T = core_attn_out.shape[0]
        q, esc = torch.ops.genesis.pn154_cabeza_q8(core_attn_out.reshape(T, -1), z.reshape(T, -1), gdn.norm.weight,
                                                   lin.input_global_scale, 0, lin._g154_b,
                                                   float(gdn.norm.eps))
        out = _g148._gemm_int8(lin, k, q, esc)
        return _g148._reducir(lin, out)
    z_shape_og = z.shape
    x = gdn.norm(core_attn_out.reshape(-1, core_attn_out.shape[-1]), z.reshape(-1, z.shape[-1]))
    x = x.reshape(z_shape_og).flatten(-2)
    output, _ = lin(rotar(lin, x))
    return output


def salida_atencion(attn, attn_output: torch.Tensor, gate) -> torch.Tensor:
    """Reemplaza el final del forward de la atencion: x*sigmoid(gate) -> (Hadamard 256) -> o_proj."""
    lin = attn.o_proj
    k = _marlin(lin)
    if k is not None and gate is not None:
        from vllm._genesis import rot_down as _g148
        T = attn_output.shape[0]
        q, esc = torch.ops.genesis.pn154_cabeza_q8(attn_output.reshape(T, -1), gate.reshape(T, -1),
                                                   lin.input_global_scale, lin.input_global_scale, 1,
                                                   lin._g154_b, 0.0)
        out = _g148._gemm_int8(lin, k, q, esc)
        return _g148._reducir(lin, out)
    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)
    output, _ = lin(rotar(lin, attn_output))
    return output
