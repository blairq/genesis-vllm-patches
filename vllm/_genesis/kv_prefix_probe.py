# SPDX-License-Identifier: Apache-2.0
"""Mide el prefijo REALMENTE compartido entre requests de un mismo agente.

================================================================
POR QUE
================================================================

`max_offload_tokens` se venia estimando a mano: tokenizar el prompt del agente
tal como esta en `opencode.jsonc` y usar ese numero. Eso mide la cosa
equivocada. Lo que importa es el prefijo que de verdad se repite entre dos
invocaciones del mismo agente, que incluye lo que el cliente ponga ANTES del
prompt del agente (framing global, esquemas de herramientas) y va en bloques,
no en tokens sueltos.

Y no hace falta adivinarlo: las claves de offload SON hashes de contenido de
prefijos alineados a bloque. La cantidad de claves iniciales en comun entre
dos requests del mismo agente **es** el prefijo compartido, exacto y en la
misma unidad en la que trabaja el tier.

Eso ademas responde una pregunta que no se puede contestar leyendo la config:
si el cliente inyecta algo variable al principio —un timestamp, un id de
sesion, el cwd— el prefijo se rompe en el bloque 0 y no se comparte NADA. Con
esta medicion se ve al toque: el valor queda en 0.

================================================================
QUE PUBLICA
================================================================

    kv_prefix_shared_blocks{agent}      ultima medicion, en bloques
    kv_prefix_shared_tokens{agent}      lo mismo en tokens
    kv_prefix_shared_blocks_max{agent}  maximo visto para ese agente
    kv_prefix_observations{agent}       cuantas comparaciones se hicieron

El maximo es el que sirve para dimensionar: es el techo de lo que ese agente
puede llegar a reusar. La ultima medicion sirve para detectar que algo rompio
el prefijo.
"""

from __future__ import annotations

from collections import OrderedDict

_MAX_AGENTES = 64
_MAX_CLAVES = 4096  # techo por agente; un prefijo mas largo que esto no se mide

_PREV: OrderedDict = OrderedDict()
_MAX: dict = {}
_MIN: dict = {}
_OBS: dict = {}


def _guardar(agent: str, keys: list) -> None:
    if agent in _PREV:
        _PREV.move_to_end(agent)
    elif len(_PREV) >= _MAX_AGENTES:
        viejo, _ = _PREV.popitem(last=False)
        _MAX.pop(viejo, None)
        _MIN.pop(viejo, None)
        _OBS.pop(viejo, None)
    _PREV[agent] = keys


def observar(agent: str, keys) -> int | None:
    """Compara con la request anterior del mismo agente.

    Devuelve la cantidad de bloques iniciales en comun, o None si es la
    primera vez que se ve ese agente (no hay con que comparar).
    """
    if not agent:
        return None
    actuales = list(keys)[:_MAX_CLAVES]
    if not actuales:
        return None

    previas = _PREV.get(agent)
    _guardar(agent, actuales)
    if previas is None:
        return None

    n = 0
    for a, b in zip(previas, actuales):
        if a != b:
            break
        n += 1

    _MAX[agent] = max(_MAX.get(agent, 0), n)
    anterior = _MIN.get(agent)
    _MIN[agent] = n if anterior is None else min(anterior, n)
    _OBS[agent] = _OBS.get(agent, 0) + 1
    return n


def maximo(agent: str) -> int:
    return _MAX.get(agent, 0)


def minimo(agent: str) -> int:
    """El prefijo que se comparte SIEMPRE, no el que se comparte a veces.

    Es el estadistico que hay que usar para dimensionar. El maximo incluye
    bloques que solo coincidieron una vez: guardarlos cuesta L2 en cada
    invocacion y casi nunca se releen. MEDIDO el 2026-08-23: pasar el limite
    de coder de 3 a 12 bloques (su maximo) bajo el hit rate del tier de 21,4%
    a 9,0%, porque 48 invocaciones x 9 bloques de mas dan vuelta L2 entera.
    """
    return _MIN.get(agent, 0)


def observaciones(agent: str) -> int:
    return _OBS.get(agent, 0)


def publicar(agent: str, comun: int, tokens_por_bloque: int) -> None:
    """Manda las series al sink de PN88. Nunca puede levantar."""
    try:
        from vllm._genesis import kv_tier_metrics as M

        if not M.enabled():
            return
        s = M.sink()
        lab = (("agent", M.agent_label({"genesis_agent": agent})),)
        s.set("kv_prefix_shared_blocks", lab, float(comun))
        s.set("kv_prefix_shared_tokens", lab, float(comun * max(tokens_por_bloque, 0)))
        s.set("kv_prefix_shared_blocks_max", lab, float(maximo(agent)))
        s.set("kv_prefix_observations", lab, float(observaciones(agent)))
    except Exception:
        pass


def reset() -> None:
    _PREV.clear()
    _MAX.clear()
    _MIN.clear()
    _OBS.clear()


def snapshot() -> dict:
    """Para el endpoint de PN89 / el dashboard."""
    return {
        a: {
            "max_bloques": _MAX.get(a, 0),
            "min_bloques": _MIN.get(a, 0),
            "observaciones": _OBS.get(a, 0),
        }
        for a in _PREV
    }
