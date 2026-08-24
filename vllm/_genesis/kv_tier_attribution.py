# SPDX-License-Identifier: Apache-2.0
"""Atribucion EXACTA de que tier sirvio cada acierto, por request.

================================================================
POR QUE
================================================================

El dashboard clasificaba el tier ADIVINANDO por latencia: estimaba cuanto
deberia tardar el prefill de los tokens nuevos, le restaba eso y 0,8 s fijos
al TTFT, y bucketeaba el sobrante. Ese metodo tiene tres problemas y los tres
se vieron en produccion:

  1. Depende de un throughput de prefill que estaba 20x mal, y con eso TODO
     caia en el bucket mas lento: la tabla mostraba L3 = 99,1% y L1 = 0
     mientras el engine reportaba 1.389.440 aciertos de GPU y CERO de disco.
  2. Asigna el request ENTERO a un solo tier. Un request real toma bloques de
     L1, L2 y L3 a la vez.
  3. Convierte tokens a bytes con una constante hardcodeada (64 KB/token) que
     no coincide con nada: lo medido es 139.264 B/token crudo, ~40 KB con PN93
     y ~23 KB con PN99.

El engine SI sabe la respuesta exacta: `TieringOffloadingManager.lookup()`
distingue si el bloque lo sirvio el tier primario (RAM) o hubo que promoverlo
de un secundario (disco). Esto lleva la cuenta por request y publica el
reparto real.

================================================================
COMO
================================================================

Se anota por request cuantos bloques respondio cada tier. Cuando el scheduler
decide `num_hit_tokens`, esos tokens se reparten en la misma proporcion y se
publican en `kv_tier_hit_tokens_total{tier}`.

El reparto es proporcional y no exacto token a token porque el hit es un
PREFIJO: el scheduler acepta los primeros N tokens y no una seleccion de
bloques sueltos. Repartir proporcionalmente es la atribucion correcta para ese
prefijo, y es exacta a nivel agregado, que es lo que la tabla necesita.

Los tokens de L1 no salen de aca: son los del prefix cache de la GPU, que vLLM
ya publica como `prefix_cache_hits_total` menos `external_prefix_cache_hits_total`.
"""

from __future__ import annotations

from collections import OrderedDict

_MAX_REQUESTS = 4096
_POR_REQUEST: OrderedDict = OrderedDict()


def _clave(req_context) -> str | None:
    rid = getattr(req_context, "req_id", None)
    return str(rid) if rid is not None else None


def anotar(req_context, tier: str) -> None:
    """Un bloque mas servido por `tier` para esta request. Nunca levanta."""
    try:
        k = _clave(req_context)
        if k is None:
            return
        d = _POR_REQUEST.get(k)
        if d is None:
            if len(_POR_REQUEST) >= _MAX_REQUESTS:
                _POR_REQUEST.popitem(last=False)
            d = _POR_REQUEST[k] = {}
        else:
            _POR_REQUEST.move_to_end(k)
        from vllm._genesis.kv_tier_metrics import tier_name

        t = tier_name(tier)
        d[t] = d.get(t, 0) + 1
    except Exception:
        pass


def reparto(req_context) -> dict:
    k = _clave(req_context)
    return dict(_POR_REQUEST.get(k) or {}) if k else {}


def publicar_y_limpiar(req_context, num_hit_tokens) -> dict:
    """Reparte `num_hit_tokens` entre los tiers que respondieron y publica.

    Devuelve el reparto {tier: tokens} para quien lo quiera usar.
    """
    salida: dict = {}
    try:
        k = _clave(req_context)
        if k is None:
            return salida
        d = _POR_REQUEST.pop(k, None)
        if not d or not num_hit_tokens:
            return salida
        total = sum(d.values())
        if total <= 0:
            return salida

        from vllm._genesis import kv_tier_metrics as M

        if not M.enabled():
            return salida
        s = M.sink()
        restante = int(num_hit_tokens)
        items = sorted(d.items())
        for i, (t, n) in enumerate(items):
            # El ultimo se lleva el resto para que la suma cierre exacta.
            tk = restante if i == len(items) - 1 else int(num_hit_tokens * n / total)
            restante -= tk
            if tk > 0:
                salida[t] = tk
                s.inc("kv_tier_hit_tokens_total", (("tier", t),), float(tk))
    except Exception:
        pass
    return salida


def reset() -> None:
    _POR_REQUEST.clear()


def stats() -> dict:
    return {"requests_en_vuelo": len(_POR_REQUEST)}
