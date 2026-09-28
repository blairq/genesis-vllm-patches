# SPDX-License-Identifier: Apache-2.0
"""PN155 — in_proj_b / in_proj_a (compuertas del GDN) dentro del Marlin de in_proj_qkvz.

En idiotSavant v1 las compuertas iban en fp16: cada capa GDN corria un cutlass fp16 + split-K (~7 us) para
48 salidas. En el v2 van en W4 como el resto, pero Marlin no puede con ellas solas: con TP=2 son 24 filas
por rango (Marlin pide N >= 64). Por eso se suman a in_proj_qkvz como un quinto trozo:

    por rango: [q | k | v | z | b (24) + relleno (40) | a (24) + relleno (40)]  = 8192 + 128 = 65 x 128

(dos trozos de 64 por rango, los shards 4 y 5 del MergedColumnParallelLinear: vLLM solo acepta ids de
trozo enteros) y el forward separa b y a de la salida. El relleno son pesos cero (nibble 8) con escala 1.

Solo con checkpoints que traen in_proj_a/b cuantizadas (config: no estan en ``ignore``) y
GENESIS_ENABLE_PN155_BA_EN_QKVZ=1. El mapeo de pesos se decide al importar (variable de entorno); si
el checkpoint no coincide, falla al cargar con un mensaje claro.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn155")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN155_BA_EN_QKVZ", "0").strip().lower() in ("1", "true", "yes", "on")
TROZO = 64                       # filas por rango de cada trozo (b y a), con su relleno
RELLENO_POR_RANGO = 2 * TROZO
_CERO_PACK = -2004318072         # int32 0x88888888: ocho nibbles "8" = ocho pesos cero en uint4b8


def destino(cual: str):
    """Entrada de ``orig_to_new_stacked`` para ".in_proj_b" / ".in_proj_a"."""
    if ACTIVO:
        return (".in_proj_qkvz", 4 if cual == "b" else 5)
    return (".in_proj_ba", 0 if cual == "b" else 1)


def tamanos_qkvz(key_dim: int, value_dim: int, tp: int) -> list:
    base = [key_dim, key_dim, value_dim, value_dim]
    return base + [TROZO * tp, TROZO * tp] if ACTIVO else base


def preparar(gdn) -> None:
    """Desde el __init__ del GDN, despues de crear in_proj_qkvz: engancha el cargador de los trozos b/a
    y escribe el relleno (los parametros de compressed-tensors nacen con torch.empty)."""
    if not ACTIVO:
        return
    lin = gdn.in_proj_qkvz
    nv = gdn.num_v_heads // gdn.tp_size
    base = (gdn.key_dim * 2 + gdn.value_dim * 2) // gdn.tp_size
    gdn._g155 = (base, nv)
    rango = gdn.tp_rank
    for nombre, p in lin.named_parameters():
        if nombre == "weight_packed":
            p.data[base:base + RELLENO_POR_RANGO].fill_(_CERO_PACK)
        elif nombre == "weight_scale":
            p.data[base:base + RELLENO_POR_RANGO].fill_(1.0)
        elif nombre == "weight":
            raise RuntimeError("[PN155] in_proj_qkvz no esta cuantizada: este parche es para idiotSavant v2")
        cargar0 = p.weight_loader

        def cargar(param, w, sid=None, _c0=cargar0, _n=nombre):
            if sid not in (4, 5):
                return _c0(param, w, sid) if sid is not None else _c0(param, w)
            if _n == "weight_shape":
                return                                   # la forma la fija el trozo principal
            if w.shape[0] != nv * gdn.tp_size:
                raise RuntimeError(f"[PN155] {_n} de in_proj_{'b' if sid == 4 else 'a'}: {tuple(w.shape)}, se esperaban "
                                   f"{nv * gdn.tp_size} filas")
            ini = base + (0 if sid == 4 else TROZO)
            param.data[ini:ini + nv].copy_(w[rango * nv:(rango + 1) * nv])
        p.weight_loader = cargar
    if not getattr(preparar, "_avisado", False):
        preparar._avisado = True
        log.warning("[PN155] in_proj_b/a dentro de in_proj_qkvz: %d + %d filas por rango (b y a de %d)",
                    base, RELLENO_POR_RANGO, nv)


def separar(gdn, mixed: torch.Tensor):
    """Salida de in_proj_qkvz -> (qkvz, ba) con ba = [b | a] como la daria in_proj_ba."""
    base, nv = gdn._g155
    return mixed[..., :base], torch.cat([mixed[..., base:base + nv], mixed[..., base + TROZO:base + TROZO + nv]], -1)
