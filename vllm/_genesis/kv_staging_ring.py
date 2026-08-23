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

Se reservan los ultimos K bloques de L2 como anillo. Los bloques de agentes
efimeros se sirven de ahi:

  - no salen del presupuesto del cache, asi que **nunca desalojan** al hilo
    principal;
  - siguen siendo visibles para `lookup()`, asi que el propio subagente
    acierta si relee su prefijo;
  - cuando el anillo se llena se recicla el mas viejo (FIFO), que es la
    semantica correcta para material de un solo uso.

La reserva se hace perezosamente, en el primer `prepare_store`, cuando
`_num_allocated_blocks` todavia es 0: se toman los ids del tope del rango sin
asignar y se baja `_num_blocks`. Asi no hace falta tocar el `__init__`, que ya
tiene parches de PN96.
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


class Anillo:
    """Pool FIFO de ids de bloque reservados fuera del cache."""

    def __init__(self, base: int, n: int) -> None:
        self.base = base
        self.n = n
        self.libres: list[int] = list(range(base, base + n))
        # key -> block_id, en orden de llegada (para reciclar el mas viejo)
        self.en_uso: OrderedDict = OrderedDict()
        self.reciclados = 0
        self.servidos = 0

    def contiene(self, block_id: int) -> bool:
        return self.base <= int(block_id) < self.base + self.n

    def tomar(self, key):
        """Devuelve un id para `key`, reciclando el mas viejo si hace falta.

        Devuelve (block_id, key_reciclada|None). `key_reciclada` hay que
        sacarla de la politica: su bloque se reusa.
        """
        reciclada = None
        if not self.libres:
            if not self.en_uso:
                return None, None
            reciclada, bid = self.en_uso.popitem(last=False)
            self.reciclados += 1
            self.libres.append(bid)
        bid = self.libres.pop()
        self.en_uso[key] = bid
        self.servidos += 1
        return bid, reciclada

    def liberar(self, block_id: int) -> None:
        bid = int(block_id)
        if not self.contiene(bid):
            return
        for k, v in list(self.en_uso.items()):
            if v == bid:
                del self.en_uso[k]
                break
        if bid not in self.libres:
            self.libres.append(bid)

    def stats(self) -> dict:
        return {
            "bloques": self.n,
            "en_uso": len(self.en_uso),
            "libres": len(self.libres),
            "servidos": self.servidos,
            "reciclados": self.reciclados,
        }


def asegurar(mgr):
    """Reserva el anillo en `mgr` la primera vez. Devuelve el Anillo o None.

    Solo reserva si el cache todavia no asigno nada: los ids salen del tope
    del rango sin usar, asi que no pueden chocar con nada ya entregado.
    """
    anillo = getattr(mgr, "_g100_anillo", None)
    if anillo is not None:
        return anillo
    if getattr(mgr, "_g100_descartado", False):
        return None

    n = bloques_pedidos()
    disponible = mgr._num_blocks - mgr._num_allocated_blocks
    if n <= 0 or n >= mgr._num_blocks or n > disponible:
        mgr._g100_descartado = True
        return None

    base = mgr._num_blocks - n
    mgr._num_blocks = base
    anillo = Anillo(base, n)
    mgr._g100_anillo = anillo
    return anillo
