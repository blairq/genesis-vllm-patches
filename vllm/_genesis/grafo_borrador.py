# SPDX-License-Identifier: Apache-2.0
"""PN150 — el tramo del borrador DFlash que vLLM corre fuera de grafo, grabado en grafos CUDA.

En cada paso de decode, entre el forward del target y el del borrador, ``DFlashSpeculator.propose``
lanza de a uno, desde Python:

  1. ``combine_hidden_states(cat(aux))`` y la copia a ``self.hidden_states`` (el fc del borrador);
  2. ``precompute_and_store_context_kv``: la proyeccion K/V del contexto, norma, RoPE y la escritura en la
     cache de las 5 capas del borrador.

vLLM lo deja afuera de sus grafos "porque la forma del contexto cambia por paso". Pero el mismo codigo
mantiene ``num_target_tokens`` igual al del target (rellena los rechazados), y en el decode en arbol eso es
N pedidos x (K+1): una forma por cantidad de pedidos. Las entradas son buffers persistentes (direcciones
fijas). En el perfil del 27-09 esos ~12 kernels dejaban la GPU ociosa entre 20 y 70 us antes de cada uno:
~600 us por paso.

Cada tramo se graba en un grafo indexado por las formas y DIRECCIONES de sus entradas: la primera vez
corre en eager (calienta JIT/autotune), la segunda se graba, y despues se reproduce. Lo que no se repite
(prefills mezclados, formas nuevas) sigue en eager. Nunca se graba dentro de otra captura ni en las
corridas de prueba (dummy) de vLLM.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import torch

log = logging.getLogger("genesis.pn150")
_ACTIVO = os.environ.get("GENESIS_ENABLE_PN150_GRAFO_BORRADOR", "0").strip().lower() in ("1", "true", "yes", "on")
_MAX_GRAFOS = 48
_grafos: "OrderedDict[tuple, torch.cuda.CUDAGraph]" = OrderedDict()
_vistos: dict = {}
_pool = None
_avisado = False


def _correr(clave, fn):
    """Corre ``fn`` (sin argumentos; lee y escribe solo buffers persistentes) grabado en un grafo."""
    global _pool, _avisado
    g = _grafos.get(clave)
    if g is not None:
        _grafos.move_to_end(clave)
        g.replay()
        return
    n = _vistos.get(clave, 0) + 1
    _vistos[clave] = n
    if n < 2 or len(_grafos) >= _MAX_GRAFOS:
        fn()
        return
    if _pool is None:
        _pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=_pool):
        fn()
    _grafos[clave] = g
    if not _avisado:
        _avisado = True
        log.warning("[PN150] tramo del borrador grabado en grafo (primera clave: %s)", clave[:2])
    g.replay()                     # la captura no ejecuta: este paso corre ahora


def _en_grafo(dummy_run: bool) -> bool:
    return _ACTIVO and not dummy_run and not torch.cuda.is_current_stream_capturing()


def combinar(spec, aux_hidden_states, last_hidden_states, n: int, dummy_run: bool) -> None:
    """``hidden_states = combine_hidden_states(cat(aux))`` y copia a ``spec.hidden_states[:n]``."""
    def fn():
        if aux_hidden_states:
            hs = spec.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))
        else:
            hs = last_hidden_states
        spec.hidden_states[:n].copy_(hs[:n])
    if not _en_grafo(dummy_run):
        fn()
        return
    fuentes = tuple(aux_hidden_states) if aux_hidden_states else (last_hidden_states,)
    clave = ("combinar", n) + tuple((t.data_ptr(), tuple(t.shape), t.stride(0)) for t in fuentes) \
        + (spec.hidden_states.data_ptr(),)
    _correr(clave, fn)


def contexto(spec, n: int, context_slots, dummy_run: bool) -> None:
    """``precompute_and_store_context_kv`` sobre ``spec.hidden_states[:n]`` y ``spec.context_positions[:n]``."""
    hs, pos = spec.hidden_states[:n], spec.context_positions[:n]
    def fn():
        spec.model.precompute_and_store_context_kv(hs, pos, context_slots)
    if not _en_grafo(dummy_run) or context_slots is None:
        fn()
        return
    slots = context_slots if isinstance(context_slots, (list, tuple)) else [context_slots]
    clave = ("contexto", n, hs.data_ptr(), pos.data_ptr()) + tuple(
        (s.data_ptr(), tuple(s.shape)) if s is not None else None for s in slots)
    _correr(clave, fn)
