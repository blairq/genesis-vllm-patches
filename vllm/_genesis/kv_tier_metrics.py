# SPDX-License-Identifier: Apache-2.0
"""Sink de telemetría de los tiers del cache de KV — soporte de PN88.

================================================================
POR QUÉ EXISTE ESTE MÓDULO
================================================================

vLLM instrumenta el offloading a medias. Lo que publica hoy en Prometheus
sale todo de `OffloadingConnectorStats` y cubre **solo el tier de RAM**:

    vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"|"CPU_to_GPU"}
    vllm:kv_offload_total_time_total{transfer_type=...}
    vllm:kv_offload_size{transfer_type=...}

El tier de disco (`FileSystemTierManager`) no aporta un solo número: su
`JobResult` lleva `job_id` y `success` y nada más, así que ni los bytes ni
el tiempo de una lectura o escritura de NVMe llegan a ningún lado. Tampoco
hay hit rate por tier — `external_prefix_cache_hits` fusiona RAM y disco —
ni ocupación, ni desalojos, ni errores.

Este módulo es el acumulador donde el código inyectado por PN88 deposita
esas observaciones. Se mantiene aparte del parche de texto a propósito: la
lógica vive en Python normal, versionado y testeable, y las anclas de PN88
quedan en dos o tres líneas cada una.

================================================================
CÓMO LLEGA A /metrics
================================================================

El tier de disco corre en el proceso **scheduler**, y el endpoint /metrics
lo sirve el proceso de la API. El único canal que los une es
`KVConnectorStats`, que ya viaja scheduler -> logger en cada step.

PN88 mete lo de acá dentro de `OffloadingConnectorStats.data` bajo la clave
`GENESIS_BUCKET`, y parchea los tres puntos que asumen que todo valor de
ese dict es una lista de operaciones (`aggregate`, `reduce` y
`OffloadPromMetrics.observe`) para que la reconozcan y la ruteen.

Por eso todo lo que sale de `drain()` tiene que ser **serializable como
JSON**: cruza un límite de proceso. De ahí que las labels se aplanen a un
string en vez de usar tuplas como clave.

================================================================
CARDINALIDAD
================================================================

`agent_label()` colapsa cualquier nombre fuera de `GENESIS_KV_AGENTS` al
bucket `other`. Es deliberado: el nombre del agente entra por
`kv_transfer_params`, o sea desde el cliente, y sin lista cerrada un typo
en un `extraBody` de opencode crea una serie de Prometheus nueva para
siempre. Los identificadores de alta cardinalidad —`task_id`, `session_id`,
hashes de bloque— NO tienen lugar acá; van al registro por request del
proxy.
"""

from __future__ import annotations

import os
import threading

# Clave bajo la que PN88 cuelga este bucket dentro de
# OffloadingConnectorStats.data. Empieza con guion bajo para no chocar nunca
# con un transfer_type real, que siempre tiene la forma "<SRC>_to_<DST>".
GENESIS_BUCKET = "_genesis_tiers"

# Etiqueta usada cuando no se pudo determinar el agente: el request no traía
# `genesis_agent`, o traía uno que no está en la allowlist.
AGENT_UNKNOWN = "unknown"
AGENT_OTHER = "other"

_DEFAULT_AGENTS = (
    "coach",
    "primary",
    "planner",
    "coder",
    "verifier",
    "utility",
    "explorer",
    "vision",
    "art",
    "sculptor",
)


def _sep(name: str, labels: tuple[tuple[str, str], ...]) -> str:
    """Aplana (nombre, labels) a una clave string serializable.

    Formato: ``nombre|k=v,k=v`` con las labels ordenadas, para que dos
    observaciones con las mismas labels en distinto orden caigan en la
    misma serie.
    """
    if not labels:
        return name
    body = ",".join(f"{k}={v}" for k, v in sorted(labels))
    return f"{name}|{body}"


def parse_key(key: str) -> tuple[str, dict[str, str]]:
    """Inversa de `_sep`: de ``nombre|k=v,k=v`` a (nombre, {k: v})."""
    if "|" not in key:
        return key, {}
    name, _, body = key.partition("|")
    labels: dict[str, str] = {}
    for pair in body.split(","):
        if not pair:
            continue
        k, _, v = pair.partition("=")
        labels[k] = v
    return name, labels


class TierStatsSink:
    """Acumulador de observaciones entre dos `drain()`.

    Vive en el proceso scheduler. `drain()` lo vacía: cada observación se
    entrega exactamente una vez, igual que hace
    `OffloadingConnectorWorker.get_kv_connector_stats`, que también devuelve
    y resetea.

    Es thread-safe porque el pool de I/O del tier fs corre en hilos aparte y
    puede reportar mientras el scheduler drena.
    """

    __slots__ = ("_lock", "_counters", "_hist", "_gauges")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._hist: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}

    def inc(self, name: str, labels=(), value: float = 1.0) -> None:
        """Suma a un contador monótono."""
        k = _sep(name, tuple(labels))
        with self._lock:
            self._counters[k] = self._counters.get(k, 0.0) + value

    def observe(self, name: str, labels=(), value: float = 0.0) -> None:
        """Registra una muestra para un histograma."""
        k = _sep(name, tuple(labels))
        with self._lock:
            self._hist.setdefault(k, []).append(value)

    def set(self, name: str, labels=(), value: float = 0.0) -> None:
        """Fija el valor de un gauge (gana la última escritura)."""
        k = _sep(name, tuple(labels))
        with self._lock:
            self._gauges[k] = value

    def is_empty(self) -> bool:
        with self._lock:
            return not (self._counters or self._hist or self._gauges)

    def drain(self) -> dict | None:
        """Devuelve lo acumulado y resetea. `None` si no hubo nada."""
        with self._lock:
            if not (self._counters or self._hist or self._gauges):
                return None
            payload = {
                "counters": self._counters,
                "hist": self._hist,
                "gauges": self._gauges,
            }
            self._counters = {}
            self._hist = {}
            self._gauges = {}
        return payload


_SINK: TierStatsSink | None = None
_SINK_LOCK = threading.Lock()


def sink() -> TierStatsSink:
    """Sink global del proceso. Lo comparten el tier fs y el tiering manager."""
    global _SINK
    if _SINK is None:
        with _SINK_LOCK:
            if _SINK is None:
                _SINK = TierStatsSink()
    return _SINK


def enabled() -> bool:
    return os.environ.get("GENESIS_ENABLE_PN88_KV_TIER_METRICS") == "1"


_AGENTS_CACHE: frozenset[str] | None = None


def _allowlist() -> frozenset[str]:
    global _AGENTS_CACHE
    if _AGENTS_CACHE is None:
        raw = os.environ.get("GENESIS_KV_AGENTS", "")
        names = tuple(n.strip() for n in raw.split(",") if n.strip())
        _AGENTS_CACHE = frozenset(names or _DEFAULT_AGENTS)
    return _AGENTS_CACHE


def agent_label(kv_transfer_params) -> str:
    """Extrae el agente de `kv_transfer_params`, acotado a la allowlist.

    El nombre lo manda el cliente (opencode lo pone en el `extraBody` de cada
    canal, como `kv_transfer_params.genesis_agent`). Cualquier valor que no
    esté en la lista cerrada cae en `other`, y su ausencia en `unknown`, para
    que la cardinalidad de las labels quede acotada pase lo que pase.
    """
    if not isinstance(kv_transfer_params, dict):
        return AGENT_UNKNOWN
    raw = kv_transfer_params.get("genesis_agent")
    if raw is None:
        return AGENT_UNKNOWN
    name = str(raw).strip().lower()
    if not name:
        return AGENT_UNKNOWN
    return name if name in _allowlist() else AGENT_OTHER


# ---------------------------------------------------------------------------
# Helpers que llama el código inyectado por PN88.
#
# Viven acá y no en las anclas para que cada punto de inyección sea una sola
# línea: cuanto más corta el ancla, menos probable que derive con una versión
# nueva de vLLM. Todos son best-effort — envueltos en try/except, porque una
# métrica NUNCA puede tumbar el scheduler ni el pool de I/O.
# ---------------------------------------------------------------------------


def note_job(mgr, job_metadata, direction: str) -> None:
    """Anota el arranque de un job del tier fs para poder medirlo al terminar.

    Se llama desde `submit_store` / `submit_load`, que corren en el hilo del
    scheduler. Guarda bytes, t0 y agente en un dict del propio manager; lo
    consume `finish_job`.
    """
    if not enabled():
        return
    try:
        import time as _t

        jobs = getattr(mgr, "_genesis_pn88_jobs", None)
        if jobs is None:
            jobs = mgr._genesis_pn88_jobs = {}
        # Cota superior: `store_block` saltea el archivo si ya existe, así que
        # los bytes REALMENTE escritos pueden ser menos. Se documenta en la
        # métrica; medir el valor exacto exigiría instrumentar cada callback.
        nbytes = len(job_metadata.keys) * getattr(mgr, "_block_size", 0)
        ctx = getattr(job_metadata, "req_context", None)
        agent = agent_label(getattr(ctx, "kv_transfer_params", None))
        jobs[job_metadata.job_id] = (nbytes, _t.perf_counter(), direction, agent)
        if len(jobs) > 4096:  # red de seguridad contra una fuga por jobs perdidos
            jobs.clear()
    except Exception:
        pass


def finish_job(mgr, result):
    """Registra un job terminado del tier fs. Devuelve `result` sin tocarlo.

    Se usa como envoltorio dentro del generador de `get_finished_jobs`, así
    que tiene que ser transparente: cualquier cosa que devuelva es lo que ve
    el `TieringOffloadingManager`.
    """
    if not enabled():
        return result
    try:
        import time as _t

        jobs = getattr(mgr, "_genesis_pn88_jobs", None)
        if not jobs:
            return result
        entry = jobs.pop(getattr(result, "job_id", None), None)
        if entry is None:
            return result
        nbytes, t0, direction, agent = entry
        dt = _t.perf_counter() - t0
        s = sink()
        tier = (("tier", "disk"),)
        if getattr(result, "success", True):
            s.inc("kv_tier_bytes_total", tier + (("direction", direction), ("agent", agent)), nbytes)
            s.inc("kv_tier_ops_total", tier + (("direction", direction), ("agent", agent)))
            s.observe("kv_tier_op_seconds", tier + (("direction", direction),), dt)
        else:
            s.inc("kv_tier_errors_total", tier + (("direction", direction),))
    except Exception:
        pass
    return result


# El nombre del tier llega de dos fuentes que no coinciden: el tiering manager
# pasa `tier.tier_type`, que es el tipo REGISTRADO en SecondaryTierFactory
# ("fs", "obj", "example"), mientras PN81 y las gauges de ocupacion hablan de
# "disk". Sin normalizar, `kv_tier_lookups_total` sale con tier="fs" y
# `kv_tier_evictions_total` con tier="disk": el mismo tier en dos series que no
# se pueden cruzar, y el dashboard —que busca "disk"— no encuentra los lookups.
# Se normaliza al nombre del MEDIO, no al de la implementacion.
_TIER_ALIASES = {
    "fs": "disk",
    "obj": "object",
}


def tier_name(tier: str) -> str:
    """Normaliza el tipo de tier al nombre del medio que usan las métricas."""
    return _TIER_ALIASES.get(tier, tier)


def note_lookup(tier: str, result) -> None:
    """Clasifica un lookup de tier. `True` hit, `None` en vuelo, `False` miss."""
    if not enabled():
        return
    try:
        label = "hit" if result is True else ("inflight" if result is None else "miss")
        sink().inc(
            "kv_tier_lookups_total", (("tier", tier_name(tier)), ("result", label))
        )
    except Exception:
        pass


def note_promotion_refused(tier: str) -> None:
    """El tier primario está lleno y no se pudo promover un bloque que SÍ estaba.

    Es el caso peor y hoy es invisible: `lookup()` devuelve `False`, o sea lo
    mismo que un miss frío, y el scheduler corta el hit de prefijo en el
    primer `False` — con lo cual no se pierde ese bloque, se pierde **todo el
    prefijo restante**. Sin este contador no hay forma de distinguir "no lo
    teníamos" de "lo teníamos y no lo pudimos usar".
    """
    if not enabled():
        return
    try:
        sink().inc("kv_tier_promotion_refused_total", (("tier", tier_name(tier)),))
    except Exception:
        pass


def note_eviction(tier: str, reason: str, count: float = 1.0, freed_bytes: float = 0.0) -> None:
    """Desalojos. Lo llama PN81 al podar por cuota o por directorio huérfano."""
    if not enabled():
        return
    try:
        s = sink()
        s.inc("kv_tier_evictions_total", (("tier", tier_name(tier)), ("reason", reason)), count)
        if freed_bytes:
            s.inc("kv_tier_evicted_bytes_total", (("tier", tier_name(tier)), ("reason", reason)), freed_bytes)
    except Exception:
        pass


def set_occupancy(tier: str, used_bytes: float, capacity_bytes: float = 0.0) -> None:
    """Ocupación de un tier. Gauge: gana la última medición."""
    if not enabled():
        return
    try:
        s = sink()
        s.set("kv_tier_bytes_used", (("tier", tier_name(tier)),), used_bytes)
        if capacity_bytes:
            s.set("kv_tier_capacity_bytes", (("tier", tier_name(tier)),), capacity_bytes)
    except Exception:
        pass


def merge(dst: dict, src: dict) -> dict:
    """Fusiona dos payloads de `drain()`. Usado por `aggregate()`.

    Contadores se suman, histogramas se concatenan, gauges los pisa el
    último — que es la semántica correcta para una medición de ocupación.
    """
    for bucket, combine in (
        ("counters", lambda a, b: a + b),
        ("gauges", lambda a, b: b),
    ):
        for k, v in src.get(bucket, {}).items():
            acc = dst.setdefault(bucket, {})
            acc[k] = combine(acc[k], v) if k in acc else v
    for k, v in src.get("hist", {}).items():
        dst.setdefault("hist", {}).setdefault(k, []).extend(v)
    return dst


# Buckets por métrica. Los de tiempo cubren de 1 ms a 30 s: una lectura de
# NVMe de un bloque de 27,6 MiB ronda los 10-40 ms, y el techo alto atrapa el
# caso patológico de un pool de I/O saturado.
_BUCKETS: dict[str, tuple[float, ...]] = {
    "kv_tier_op_seconds": (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
    ),
    "kv_tier_block_residency_seconds": (
        60.0, 300.0, 900.0, 1800.0, 3600.0, 7200.0, 21600.0, 86400.0,
    ),
    "kv_tier_time_to_first_reuse_seconds": (
        1.0, 10.0, 60.0, 300.0, 900.0, 1800.0, 3600.0, 7200.0, 21600.0,
    ),
}

_DOC = {
    "kv_tier_bytes_total": (
        "Bytes movidos por tier de offloading. Para el tier de disco en "
        "direction=write es una COTA SUPERIOR: store_block saltea el archivo "
        "si el bloque ya existe."
    ),
    "kv_tier_ops_total": "Operaciones de I/O completadas por tier.",
    "kv_tier_op_seconds": (
        "Latencia de una operación del tier, en segundos. Wall-clock del pool "
        "de I/O — NO comparable con vllm:kv_offload_total_time, que son CUDA "
        "events sobre el stream de copia y no camino crítico."
    ),
    "kv_tier_errors_total": "Operaciones fallidas por tier (short read/write, archivo ilegible).",
    "kv_tier_lookups_total": (
        "Lookups por tier y resultado. Separa el hit rate de RAM del de disco, "
        "que vllm:external_prefix_cache_hits fusiona."
    ),
    "kv_tier_promotion_refused_total": (
        "Bloques presentes en un tier secundario que NO se pudieron promover "
        "por tier primario lleno. El scheduler corta el hit de prefijo en el "
        "primer fallo, así que cada uno trunca todo el prefijo restante."
    ),
    "kv_tier_evictions_total": "Bloques desalojados por tier y motivo.",
    "kv_tier_evicted_bytes_total": "Bytes liberados por desalojo, por tier y motivo.",
    "kv_tier_bytes_used": "Bytes ocupados actualmente por el tier.",
    "kv_tier_capacity_bytes": "Capacidad configurada del tier.",
    "kv_tier_block_residency_seconds": "Vida de un bloque en el tier antes de ser desalojado.",
    "kv_tier_time_to_first_reuse_seconds": "Tiempo desde que se guarda un bloque hasta su primera lectura.",
}


def observe_bucket(prom, payload: dict, engine_idx: int = 0) -> None:
    """Vuelca el bucket Genesis a métricas de Prometheus.

    `prom` es la instancia de `OffloadPromMetrics`; de ahí salen las clases de
    métrica (`_counter_cls`/`_gauge_cls`/`_histogram_cls`), los `labelnames`
    base y los valores por engine. Las métricas se crean perezosamente la
    primera vez que aparece cada nombre, igual que hace vLLM con
    `transfer_type`.

    Nunca lanza: una métrica rota no puede tumbar el logger.
    """
    try:
        cache = getattr(prom, "_genesis_metrics", None)
        if cache is None:
            cache = prom._genesis_metrics = {}
        base_labels = list(prom._labelnames)
        base_values = list(prom.per_engine_labelvalues[engine_idx])

        def _metric(kind: str, name: str, label_keys: tuple[str, ...]):
            entry = cache.get(name)
            if entry is None:
                if kind == "counter":
                    cls, kwargs = prom._counter_cls, {}
                elif kind == "gauge":
                    cls, kwargs = prom._gauge_cls, {}
                else:
                    cls = prom._histogram_cls
                    kwargs = {"buckets": list(_BUCKETS.get(name, (0.01, 0.1, 1.0, 10.0)))}
                metric = cls(
                    name=f"vllm:{name}",
                    documentation=_DOC.get(name, "Genesis KV tier metric."),
                    labelnames=base_labels + list(label_keys),
                    **kwargs,
                )
                entry = cache[name] = (metric, label_keys, {})
            metric, keys, children = entry
            # Un nombre con otro juego de labels rompería la serie: se ignora
            # en vez de explotar. No debería pasar — cada emisor es consistente.
            if keys != label_keys:
                return None
            return metric, children

        def _child(kind, name, labels: dict[str, str]):
            label_keys = tuple(sorted(labels))
            got = _metric(kind, name, label_keys)
            if got is None:
                return None
            metric, children = got
            values = tuple(labels[k] for k in label_keys)
            ck = (engine_idx, values)
            child = children.get(ck)
            if child is None:
                child = children[ck] = metric.labels(*(base_values + list(values)))
            return child

        for key, value in payload.get("counters", {}).items():
            name, labels = parse_key(key)
            child = _child("counter", name, labels)
            if child is not None:
                child.inc(value)

        for key, value in payload.get("gauges", {}).items():
            name, labels = parse_key(key)
            child = _child("gauge", name, labels)
            if child is not None:
                child.set(value)

        for key, samples in payload.get("hist", {}).items():
            name, labels = parse_key(key)
            child = _child("histogram", name, labels)
            if child is not None:
                for s in samples:
                    child.observe(s)
    except Exception:
        pass


def reduce_for_log(payload: dict) -> dict[str, float]:
    """Resumen legible para la línea 'KV Transfer metrics' del log."""
    out: dict[str, float] = {}
    for k, v in payload.get("counters", {}).items():
        out[k] = v
    for k, v in payload.get("gauges", {}).items():
        out[k] = v
    for k, samples in payload.get("hist", {}).items():
        if samples:
            out[f"{k}#n"] = len(samples)
            out[f"{k}#avg"] = sum(samples) / len(samples)
    return out
