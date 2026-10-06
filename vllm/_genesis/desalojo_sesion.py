# SPDX-License-Identifier: Apache-2.0
"""PN172 — desalojo del prefix cache de la GPU por rol: un subagente no desplaza al hilo principal.

Regla (06-10, del usuario): el hilo principal no se desaloja para hacerle lugar a subagentes, ni propios ni ajenos;
solo otro hilo principal puede desplazarlo cuando falta VRAM (y lo desplazado queda en L2, que guarda al vuelo).

- Al liberar un pedido principal, sus bloques cacheados quedan marcados como protegidos por TTL segundos. Si despues
  los libera un subagente (prefijo comun), la marca se conserva.
- Cuando un subagente pide bloques, se saltean los protegidos (van al final de la cola LRU). Un principal toma en orden
  LRU, como siempre. Si no hay otra cosa, el subagente toma protegidos igual (nunca se traba).

Rol: con el plugin genesis-sesion (PN169), subagente = tiene padre. Sin plugin, por la etiqueta genesis_agent del
perfil de opencode (coder, explorer, verifier, utility, vision, art son subagentes).
"""
from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("genesis.pn172")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN172_DESALOJO_SESION", "0").strip().lower() in ("1", "true", "yes", "on")
TTL = float(os.environ.get("GENESIS_PN172_TTL_S", "1200"))
SUBAGENTES = {a.strip() for a in os.environ.get(
    "GENESIS_PN172_SUBAGENTES", "coder,explorer,verifier,utility,vision,art").split(",") if a.strip()}

_prot: dict[int, float] = {}         # block_id -> protegido hasta (monotonic)
_estado = {"libera": None, "pide": None}
_n = {"salteados": 0, "forzados": 0}


def rol(request) -> str:
    ktp = getattr(request, "kv_transfer_params", None) or {}
    if "genesis_sesion" in ktp:
        return "sub" if ktp.get("genesis_padre") else "pri"
    return "sub" if str(ktp.get("genesis_agent", "")).strip().lower() in SUBAGENTES else "pri"


def liberando(request) -> None:
    if ACTIVO:
        _estado["libera"] = None if request is None else rol(request)


def pidiendo(request) -> None:
    if ACTIVO:
        _estado["pide"] = rol(request)


def al_liberar(block) -> None:
    """Un bloque cacheado vuelve a la cola: si lo libera un principal, queda protegido."""
    if ACTIVO and _estado["libera"] == "pri":
        _prot[block.block_id] = time.monotonic() + TTL


def tomar(q, n: int) -> list:
    """popleft_n de la cola libre, salteando protegidos si pide un subagente."""
    if not ACTIVO or _estado["pide"] != "sub" or not _prot or n == 0:
        ret = q.popleft_n(n)
        if _prot:
            for b in ret:
                _prot.pop(b.block_id, None)
        return ret
    ahora = time.monotonic()
    ret, salteados = [], []
    b = q.fake_free_list_head.next_free_block
    vistos, total = 0, q.num_free_blocks
    while len(ret) < n and b is not None and b is not q.fake_free_list_tail and vistos < total:
        sig = b.next_free_block
        vistos += 1
        if _prot.get(b.block_id, 0.0) > ahora:
            salteados.append(b)
        else:
            _prot.pop(b.block_id, None)
            ret.append(b)
        b = sig
    falta = n - len(ret)
    if falta > 0:                                       # no alcanza sin tocar protegidos: se toman los mas viejos
        ret.extend(salteados[:falta])
        _n["forzados"] += falta
        salteados = salteados[falta:]
    for x in ret:
        q.remove(x)
        _prot.pop(x.block_id, None)
    for x in salteados:                                 # los protegidos salteados pasan al final de la cola
        q.remove(x)
        q.append(x)
    _n["salteados"] += len(salteados)
    if (_n["salteados"] + _n["forzados"]) and (_n["salteados"] + _n["forzados"]) % 2000 < len(salteados) + max(falta, 0):
        log.warning("PN172: subagentes salteando bloques del principal: %d salteados, %d tomados igual (sin lugar)",
                    _n["salteados"], _n["forzados"])
    return ret
