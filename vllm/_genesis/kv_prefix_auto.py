# SPDX-License-Identifier: Apache-2.0
"""Deriva `max_offload_tokens` de lo que PN101 midio, en vez de configurarlo.

PN101 mide el prefijo que cada agente comparte de verdad entre invocaciones.
Teniendo ese numero, poner `max_offload_tokens` a mano deja de tener sentido:
es un valor que hay que descubrir, mantener y que se desactualiza solo cuando
cambian las herramientas o el prompt del agente.

Lo que medimos el 2026-08-23 muestra por que la estimacion manual falla:

    agente          estimado a mano   MEDIDO
    coder                     2.496    9.984   (4x corto)
    explorer                    832    4.992   (6x corto)
    primary_high                  -   59.072

El estimado salia de tokenizar el prompt del agente del archivo de config. Lo
que faltaba es todo lo que el cliente pone alrededor: esquemas de
herramientas, framing global. Eso no se puede saber leyendo la config; se mide.

================================================================
COMO SE COMPORTA
================================================================

  - **Sin datos suficientes: sin limite.** Con menos de `MIN_OBS`
    observaciones devuelve None y se offloadea todo, que es justamente lo que
    permite APRENDER cual es el prefijo. Arrancar con un limite chico seria
    profecia autocumplida: nunca se veria compartir mas de lo que se dejo
    guardar.
  - **Con datos: el MINIMO observado.** La primera version usaba el maximo,
    razonando que era "el techo de lo reusable". La medicion lo refuto: subir
    el limite de coder de 3 a 12 bloques (su maximo) bajo el hit rate del tier
    de 21,4% a 9,0%. El maximo incluye bloques que coincidieron una sola vez;
    guardarlos cuesta L2 en CADA invocacion y casi nunca se releen, y con 48
    invocaciones esos bloques de mas dan vuelta L2 entera. El minimo es lo que
    se comparte SIEMPRE, que es lo unico que paga su lugar.
  - **Un valor explicito del cliente siempre gana.** Esto solo actua cuando no
    se mando `max_offload_tokens`.

Nunca devuelve 0: eso apagaria el offload del agente por completo, que no es
una conclusion que esta heuristica deba poder sacar sola.
"""

from __future__ import annotations

import os

_MIN_OBS_DEFECTO = 3
_MARGEN_DEFECTO = 0


def habilitado() -> bool:
    v = os.environ.get("GENESIS_ENABLE_PN103_AUTO_PREFIX", "1")
    return v.strip().lower() not in ("0", "false", "no", "off")


def _int_env(nombre: str, defecto: int) -> int:
    try:
        return max(0, int(os.environ.get(nombre, "") or defecto))
    except ValueError:
        return defecto


def min_observaciones() -> int:
    return max(1, _int_env("GENESIS_PN103_MIN_OBS", _MIN_OBS_DEFECTO))


def margen_bloques() -> int:
    # 0 por defecto: el minimo ya es conservador y agregarle margen vuelve a
    # meter bloques que no siempre se comparten, que es lo que se midio mal.
    return _int_env("GENESIS_PN103_MARGIN_BLOCKS", _MARGEN_DEFECTO)


def limite_para(agent, tokens_por_bloque: int) -> int | None:
    """Tokens a offloadear para `agent`, o None para no poner limite."""
    if not habilitado() or not agent or tokens_por_bloque <= 0:
        return None
    try:
        from vllm._genesis import kv_prefix_probe as P

        if P.observaciones(agent) < min_observaciones():
            return None  # todavia aprendiendo: no limitar
        bloques = P.minimo(agent)
        if bloques <= 0:
            # El prefijo se rompe en el primer bloque. Puede ser que el cliente
            # inyecte algo variable adelante. No es una conclusion para apagar
            # el offload: se deja sin limite y que decida la politica de cache.
            return None
        return (bloques + margen_bloques()) * tokens_por_bloque
    except Exception:
        return None


def explicacion(agent, tokens_por_bloque: int) -> str:
    from vllm._genesis import kv_prefix_probe as P

    obs = P.observaciones(agent)
    if obs < min_observaciones():
        return f"{agent}: {obs} observaciones, todavia aprendiendo (sin limite)"
    lim = limite_para(agent, tokens_por_bloque)
    return (
        f"{agent}: min {P.minimo(agent)} (max {P.maximo(agent)}) + margen "
        f"{margen_bloques()}"
        f" -> {lim} tokens ({obs} observaciones)"
    )
