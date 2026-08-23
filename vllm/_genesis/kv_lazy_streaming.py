# SPDX-License-Identifier: Apache-2.0
"""Genesis PN91: deferral acotado de promociones (L3 SSD → L2 RAM).

El problema real
----------------
`OffloadingConnectorScheduler._lookup()` devuelve `None` —"volvé a preguntar el
paso que viene"— en cuanto CUALQUIER bloque del prefijo está en vuelo. El
request queda parado pasos enteros mientras se promueven bloques desde el SSD.
Con un prefijo grande eso son varios segundos de TTFT en los que el request ni
siquiera entra al batch.

Por qué no se puede simplemente "seguir con lo que haya"
--------------------------------------------------------
`_maximal_prefix_lookup` cuenta un `None` COMO HIT (`result = True`) a propósito,
para seguir disparando promociones más allá del primer bloque en vuelo. Es decir
que `num_hit_tokens` incluye bloques que todavía no están listos. Devolverlo
haría que `update_state_after_alloc` llamara a `prepare_load` sobre bloques no
listos y reventara el `assert block.is_ready` de `CPUOffloadingManager`.

Qué hace este módulo
--------------------
Mide cuánto TIEMPO lleva cada request difiriendo. Pasado el presupuesto (o un
tope secundario en pasos, para el caso de pasos instantáneos), arma el **modo
estricto** para ese request: en el paso siguiente los lookups
tratan `None` como MISS en vez de como hit, con lo cual el prefijo que se
devuelve contiene sólo bloques efectivamente listos. El request arranca con un
hit más corto en vez de seguir esperando, y las promociones que quedaron en
vuelo sirven para el turno siguiente.

Es seguro por construcción: un hit más corto siempre es válido; lo único que se
paga es recómputo de la cola.

Techo esperado
--------------
Acotado. La medición del propio compose dice que el tráfico de offloading es
1,0-1,9% del wall time y está fuera del camino crítico. Esto no ataca ese
tráfico sino la LATENCIA DE ADMISIÓN de un request con prefijo grande. Si te
importa el TTFT de hilos largos, sirve; si te importa el throughput agregado,
no esperes nada.
"""

from __future__ import annotations

import logging
import os
import threading
import time


def _get_logger(name: str):
    """Logger que SÍ se ve desde dentro del EngineCore.

    vLLM configura sus handlers sobre los loggers que crea `init_logger`; un
    `logging.getLogger` pelado no propaga a ninguno, así que todo lo que se
    logueara acá era invisible en `docker logs` (verificado: las líneas de
    registro de grupos de PN93 no aparecían nunca). Se cae al logger estándar
    si vLLM no está disponible, para no romper los tests.
    """
    try:
        from vllm.logger import init_logger

        return init_logger(name)
    except Exception:
        return logging.getLogger(name)


log = _get_logger("genesis.pn91.lazy_streaming")

# El presupuesto va en SEGUNDOS, no en pasos del scheduler. Un paso mide entre
# ~30 ms (motor ocioso) y ~1,6 s (con un prefill largo en el batch), medido en
# este rig: un presupuesto de "4 pasos" era en realidad entre 0,12 s y 6,4 s
# segun la carga, o sea que no significaba nada.
_DEFAULT_BUDGET_SECONDS = 2.0
# Tope secundario por cantidad de pasos, para el caso patologico de pasos
# instantaneos donde el reloj nunca llega al presupuesto.
_DEFAULT_BUDGET_STEPS = 32
_MAX_TRACKED = 4096

_LOCK = threading.RLock()
# req_id -> (primer diferimiento en perf_counter, pasos diferidos)
_DEFERRALS: dict[str, tuple[float, int]] = {}
_STRICT: set[str] = set()

_STATS = {
    "deferrals": 0,
    "requests_forced_strict": 0,
}


def is_lazy_streaming_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN91_KV_LAZY_STREAMING", "1"
    ).strip().lower() in ("1", "true", "yes", "on")


def deferral_budget_seconds() -> float:
    """Cuánto puede esperar un request antes de arrancar con lo que ya está listo.

    `0` fuerza el modo estricto desde el primer diferimiento. Un valor muy alto
    equivale al comportamiento de upstream.
    """
    raw = os.environ.get("GENESIS_PN91_MAX_DEFER_SECONDS")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            log.warning(
                "PN91: presupuesto inválido %r, se usan %.1f s",
                raw,
                _DEFAULT_BUDGET_SECONDS,
            )
    return _DEFAULT_BUDGET_SECONDS


def deferral_budget_steps() -> int:
    """Tope secundario en pasos, por si los pasos son instantáneos."""
    raw = os.environ.get("GENESIS_PN91_MAX_DEFER_STEPS")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return _DEFAULT_BUDGET_STEPS


def note_deferral(req_id: str) -> int:
    """Registra que este request difirió un paso más. Devuelve el acumulado.

    Al superar el presupuesto (tiempo, o el tope de pasos) arma el modo estricto
    para el paso siguiente.
    """
    if not is_lazy_streaming_enabled():
        return 0
    now = time.perf_counter()
    armed = False
    waited = 0.0
    with _LOCK:
        _trim()
        t0, count = _DEFERRALS.get(req_id, (now, 0))
        count += 1
        _DEFERRALS[req_id] = (t0, count)
        _STATS["deferrals"] += 1

        waited = now - t0
        if req_id not in _STRICT and (
            waited >= deferral_budget_seconds() or count > deferral_budget_steps()
        ):
            _STRICT.add(req_id)
            _STATS["requests_forced_strict"] += 1
            armed = True

    if armed:
        log.debug(
            "PN91: request %s esperó %.2f s (%d pasos); se fuerza hit sólo-listos",
            req_id,
            waited,
            count,
        )
        _publish_arm(waited)
    return count


def _trim() -> None:
    """Acota el dict descartando los MÁS VIEJOS, no vaciándolo entero.

    El `clear()` anterior tiraba el estado de todos los requests vivos de un
    saque — el mismo patrón que ya se había corregido en PN88.
    """
    excess = len(_DEFERRALS) - _MAX_TRACKED
    if excess <= 0:
        return
    for k in list(_DEFERRALS)[:excess]:
        _DEFERRALS.pop(k, None)
        _STRICT.discard(k)


def _publish_arm(waited_seconds: float) -> None:
    """Publica el evento en el sink de PN88.

    PN91 es un tradeoff puro —arrancar antes con menos prefijo— y hasta ahora no
    publicaba nada: no había forma de saber si se disparaba ni cuánto se esperó
    antes de rendirse.
    """
    try:
        from vllm._genesis import kv_tier_metrics as _g88

        if not _g88.enabled():
            return
        s = _g88.sink()
        s.inc("kv_promotion_strict_armed_total", (("tier", "disk"),))
        s.observe("kv_promotion_defer_seconds", (("tier", "disk"),), waited_seconds)
    except Exception:
        pass


def is_strict(req_id: str) -> bool:
    """¿Este request tiene que tratar los bloques en vuelo como miss?"""
    if not is_lazy_streaming_enabled():
        return False
    with _LOCK:
        return req_id in _STRICT


def clear_request(req_id: str) -> None:
    with _LOCK:
        _DEFERRALS.pop(req_id, None)
        _STRICT.discard(req_id)


def stats() -> dict[str, int]:
    with _LOCK:
        out = dict(_STATS)
        out["tracked_requests"] = len(_DEFERRALS)
        out["strict_requests"] = len(_STRICT)
        out["budget_seconds"] = deferral_budget_seconds()
        out["budget_steps"] = deferral_budget_steps()
        return out


def reset_lazy_streaming() -> None:
    """Resetea todo el estado. Lo usan los tests."""
    with _LOCK:
        _DEFERRALS.clear()
        _STRICT.clear()
        for k in _STATS:
            _STATS[k] = 0
