# SPDX-License-Identifier: Apache-2.0
"""Score por bloque con envejecimiento, para elegir mejor a quien desalojar.

================================================================
LA IDEA
================================================================

Score de un bloque = en cuantos requests distintos aparecio. Eso tiene una
propiedad que hace innecesario todo el andamiaje que uno imaginaria:

    **el score ya es monotono decreciente a lo largo del prefijo.**

El bloque 3 no puede haber aparecido en mas requests que el bloque 2, porque
su hash INCLUYE al bloque 2. Es imposible por construccion.

Consecuencia: ordenando las victimas por score ascendente, el desalojo se va
solo a las COLAS de los prefijos y nunca parte un prefijo por el medio. Sin
arbol radix, sin comparar contra el request anterior, sin logica de reseteo,
sin bookkeeping por agente. Un entero por clave.

Cortar un prefijo por el medio es el peor error posible: el lookup para en el
primer miss, asi que los bloques que quedan despues conservan score alto y
valor CERO — ocupan L2 sin poder servir nunca.

================================================================
QUE LE FALTA A ARC Y ESTO AGREGA
================================================================

ARC ya tiene una dimension de frecuencia: `touch()` mueve de T1 a T2 la
segunda vez que se usa una clave, y `evict()` busca victimas primero en T1.
Le faltan dos cosas:

  1. **Envejecimiento.** T2 es permanente: una vez adentro, un bloque queda
     privilegiado hasta que lo desalojan. El preambulo de un agente que no se
     usa hace horas sigue tan protegido como el del agente que se esta
     martillando ahora. Aca se halvan los scores cada N observaciones, que es
     el aging clasico de LFU (y lo que hace W-TinyLFU con su sketch).
  2. **Granularidad.** T1/T2 es de 1 bit: no distingue un bloque visto 2 veces
     de uno visto 200.

Esto se monta SOBRE ARC, no en su lugar: ARC sigue decidiendo la particion
recencia/frecuencia y esto solo ordena a quien sacrificar dentro de cada lista.

================================================================
POR QUE NO SE REUSA `counts`
================================================================

`CPUOffloadingManager.counts` cuenta casi lo mismo, pero `prepare_store` lo
usa para filtrar: `counts.get(k, 0) >= store_threshold`. Y
`_maximal_prefix_lookup` **corta en el primer miss**, asi que los bloques
nuevos —los que justamente hay que guardar— nunca se miran y quedan en 0.
Prender `counts` de forma global haria que ese filtro los descarte a todos y
no se guarde nada. Por eso el score va en su propia estructura.

================================================================
COSTO
================================================================

Un dict acotado por manager y un entero por clave. La eleccion de victima mira
solo las primeras `GENESIS_PN102_SCAN` candidatas elegibles (default 64) en vez
de ordenar las 790: costo fijo, y captura casi todo el beneficio porque las
listas de ARC ya vienen ordenadas por recencia.
"""

from __future__ import annotations

import os
from collections import OrderedDict

_MAX_CLAVES = 65536


def habilitado() -> bool:
    v = os.environ.get("GENESIS_ENABLE_PN102_BLOCK_SCORE", "0")
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int_env(nombre: str, defecto: int) -> int:
    try:
        return max(1, int(os.environ.get(nombre, "") or defecto))
    except ValueError:
        return defecto


def periodo_envejecimiento() -> int:
    return _int_env("GENESIS_PN102_AGING_EVERY", 50000)


def candidatas_a_mirar() -> int:
    return _int_env("GENESIS_PN102_SCAN", 64)


def _estado(mgr):
    st = getattr(mgr, "_g102", None)
    if st is None:
        st = {"scores": OrderedDict(), "n": 0, "envejecidas": 0}
        mgr._g102 = st
        # La politica es la que elige victima, asi que necesita ver los scores.
        try:
            mgr._policy._g102_scores = st["scores"]
        except Exception:
            pass
    return st


def anotar(mgr, key) -> None:
    """Suma una aparicion. Se llama desde `lookup()`. Nunca puede levantar."""
    if not habilitado():
        return
    try:
        st = _estado(mgr)
        sc = st["scores"]
        if key in sc:
            sc.move_to_end(key)
            sc[key] += 1
        else:
            if len(sc) >= _MAX_CLAVES:
                sc.popitem(last=False)
            sc[key] = 1

        st["n"] += 1
        if st["n"] >= periodo_envejecimiento():
            st["n"] = 0
            st["envejecidas"] += 1
            # Halving clasico de LFU: lo que dejo de repetirse pierde
            # proteccion solo, sin logica especial.
            for k in list(sc):
                v = sc[k] >> 1
                if v:
                    sc[k] = v
                else:
                    del sc[k]
    except Exception:
        pass


def elegir(items, protected, ya_elegidas, scores, limite: int | None = None):
    """Devuelve (key, block) de menor score entre las primeras candidatas.

    Sin `scores` (PN102 apagado) devuelve la primera elegible, que es
    exactamente lo que hace vLLM hoy.
    """
    tope = limite if limite is not None else candidatas_a_mirar()
    mejor = None
    mejor_score = None
    vistas = 0
    for key, block in items:
        if block.ref_cnt != 0 or key in protected or key in ya_elegidas:
            continue
        if scores is None:
            return key, block
        s = scores.get(key, 0)
        if mejor_score is None or s < mejor_score:
            mejor, mejor_score = (key, block), s
            if s == 0:
                # No hay nada mejor que un bloque que no se repitio nunca.
                return mejor
        vistas += 1
        if vistas >= tope:
            break
    return mejor


def stats(mgr) -> dict:
    st = getattr(mgr, "_g102", None)
    if st is None:
        return {"claves": 0, "envejecidas": 0}
    return {"claves": len(st["scores"]), "envejecidas": st["envejecidas"]}
