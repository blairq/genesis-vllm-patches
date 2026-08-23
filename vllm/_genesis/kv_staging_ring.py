# SPDX-License-Identifier: Apache-2.0
"""Anillo de staging de L2: bloques transitorios que NO desalojan al cache.

================================================================
EL PROBLEMA QUE RESUELVE
================================================================

El patron real de carga no es un hilo de 256k tokens. Es un hilo principal de
60.000-100.000 tokens que dispara ~6 subagentes coder/explorer en paralelo.
Los subagentes llenan L2 con material de un solo uso, desalojan el prefijo del
principal, y cuando terminan el cliente reenvia esos 100k y hay que
prefillearlos de nuevo. Ese pico es el que hay que ahorrarse.

La caché no puede distinguir esos dos tipos de bloque por frecuencia de uso:
en el momento del desalojo los del subagente son los MAS recientes, asi que
LRU y ARC eligen exactamente al revés de lo que conviene.

Pero sí tenemos la información: PN90 ya clasifica cada request por
`genesis_agent`. Un agente efímero (coder, explorer, verifier...) es
justamente el que no merece ocupar caché.

================================================================
COMO
================================================================

K es un **TOPE ELASTICO, no una reserva**. Esto es importante y la primera
version lo hacia mal: reservaba K ids de bloque y achicaba `_num_blocks` de
forma permanente, asi que con K=240 sobre 790 el hilo principal perdia el 30%
de L2 aunque los subagentes usaran 18 bloques. Memoria inmovilizada.

Ahora no se reserva nada. Los bloques efimeros salen del pool normal, pero se
anotan en un registro FIFO acotado a K: cuando llega el K+1, se recicla el mas
viejo **del propio registro** en vez de desalojar del cache. Consecuencias:

  - con trafico efimero en cero, el cache se queda con el 100% de L2;
  - lo efimero nunca puede ocupar mas de K bloques, asi que **no desaloja** al
    hilo principal;
  - los bloques siguen siendo normales y visibles para `lookup()`, asi que el
    subagente acierta si relee su preambulo.

O sea: K deja de ser "cuanto le saco al cache" y pasa a ser "cuanto le permito
ocupar a lo efimero".
"""

from __future__ import annotations

import os
from collections import OrderedDict

_MAX_VISTOS = 65536
_VISTOS: OrderedDict = OrderedDict()


def es_compartido(key) -> bool:
    """True si esta clave ya aparecio antes: es preambulo, no cola.

    Los subagentes comparten el arranque del prompt — system prompt, reglas,
    herramientas. Si se llama 100 veces al mismo agente, esos primeros bloques
    son identicos las 100 veces, y como la clave es un hash del contenido, la
    MISMA clave vuelve a aparecer. Ese material es lo mas valioso que hay en
    L2: lo pega cada invocacion.

    La cola (el pedido concreto de cada subagente) aparece una sola vez y es
    la que tiene que ir al anillo.

    Sin esta distincion el anillo recicla el preambulo junto con la basura, o
    sea que empeora exactamente el caso que queremos mejorar.
    """
    k = bytes(key) if isinstance(key, (bytes, bytearray, memoryview)) else key
    if k in _VISTOS:
        _VISTOS.move_to_end(k)
        return True
    if len(_VISTOS) >= _MAX_VISTOS:
        _VISTOS.popitem(last=False)
    _VISTOS[k] = True
    return False


def reset_vistos() -> None:
    _VISTOS.clear()

_DEFAULT_BLOQUES = 0  # 0 = apagado


def bloques_pedidos() -> int:
    try:
        return max(0, int(os.environ.get("GENESIS_PN100_RING_BLOCKS", "0") or 0))
    except ValueError:
        return _DEFAULT_BLOQUES


def habilitado() -> bool:
    return bloques_pedidos() > 0


def _registro(mgr) -> OrderedDict:
    reg = getattr(mgr, "_g100_reg", None)
    if reg is None:
        reg = OrderedDict()
        mgr._g100_reg = reg
    return reg


def hacer_lugar(mgr, keys_nuevas) -> int:
    """Recicla bloques del propio anillo hasta que las nuevas entren en el tope.

    Devuelve cuantos bloques libero. No toca nada que no sea del anillo: si el
    cupo alcanza no libera nada, y si una clave del anillo esta en uso
    (`ref_cnt != 0`) se la deja al cache en vez de forzar.
    """
    K = bloques_pedidos()
    if K <= 0:
        return 0
    reg = _registro(mgr)
    nuevas = [k for k in keys_nuevas if k not in reg and not es_compartido(k)]
    if not nuevas:
        return 0

    liberados = 0
    while reg and len(reg) + len(nuevas) > K:
        vieja, _ = reg.popitem(last=False)
        b = mgr._policy.get(vieja)
        if b is None:
            continue
        if b.ref_cnt != 0:
            # en uso por una transferencia: no se puede reciclar ahora.
            # Sale del registro y queda como bloque normal del cache.
            continue
        mgr._policy.remove(vieja)
        mgr._free_block(b)
        liberados += 1

    for k in nuevas:
        reg[k] = True
    return liberados


def en_uso(mgr) -> int:
    return len(getattr(mgr, "_g100_reg", ()) or ())


def stats(mgr) -> dict:
    return {"tope": bloques_pedidos(), "en_uso": en_uso(mgr)}
