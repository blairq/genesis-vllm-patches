# SPDX-License-Identifier: Apache-2.0
"""KV Disk Write Gating & Tiered Demotion for multi-tier offloading (Genesis PN90).

Dos mecanismos independientes:

1. **Gating** — decide qué agentes pueden persistir bloques KV al tier
   secundario (L3 SSD). Los subagentes efímeros (coder, explorer, verifier…)
   se quedan en L1 VRAM + L2 RAM y no ensucian el SSD.

2. **Democión por desalojo** — vLLM upstream cascadea a disco en
   `complete_store` (write-through). PN90 lo reemplaza por una L3 *exclusiva*:
   el bloque baja a SSD sólo cuando lo desalojan de L2 RAM.

   El slot de L2 se reutiliza en el mismo `prepare_store` que lo desaloja, así
   que hay que copiar la fila del mmap ANTES de liberar el bloque. Esa copia
   queda en `_PENDING_DEMOTIONS` y la drena el `TieringOffloadingManager`.

Notas de auditoría (2026-08-23)
-------------------------------
- El drenaje ocurre en DOS sitios: `TieringOffloadingManager.prepare_store`
  (inmediato) y `on_schedule_end` (red de seguridad). El segundo es
  imprescindible: `_initiate_promotion` llama a `primary_tier.prepare_write`
  directamente, sin pasar por el manager, y esas evicciones también generan
  snapshots. Sin la red de seguridad quedaban huérfanos para siempre.
- El tag de persistencia se consume SÓLO si efectivamente se toma el snapshot.
  La versión anterior lo popeaba incluso cuando el bloque seguía en L1, con lo
  cual el bloque no se persistía nunca más.
- Los job ids de democión son NEGATIVOS. El contador de `TieringOffloadingManager`
  es positivo y monótono, así que los dos espacios no se pueden cruzar y el
  `assert job_metadata is not None` de `_process_finished_jobs` sigue vivo para
  los jobs reales.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable
from typing import Any


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


log = _get_logger("genesis.pn90.kv_disk_gate")

# Agentes primarios/orquestadores habilitados a persistir en L3 SSD.
DEFAULT_DISK_WRITERS = {
    "build",
    "primary",
    "primary_low",
    "primary_high",
    "primary_nothink",
    "plan",
    "planner",
    "planner_nothink",
    "coach",
}

# Presupuesto de RAM para snapshots pendientes de democión. Un bloque de este
# modelo pesa 27,6 MB: un tope por CANTIDAD (el viejo `> 1024`) permitía hasta
# 28 GB de heap. El tope va en bytes, y es RAM que compite directamente con L2.
_DEFAULT_MAX_PENDING_BYTES = 256 * 1024 * 1024

_LOCK = threading.RLock()
_PERSISTED_KEYS: set[Any] = set()
_PENDING_DEMOTIONS: list[tuple[Any, bytes]] = []
_PENDING_BYTES: int = 0

# Los ids de democión bajan desde -1. Ver nota de auditoría arriba.
_DEMOTE_JOB_ID: int = 0

# Contadores de diagnóstico (los lee PN88/PN89; no son métricas Prometheus).
_STATS = {
    "demotions_recorded": 0,
    "demotions_submitted": 0,
    "snapshots_refused_budget": 0,
    "demotions_failed": 0,
    "tagged_by_use": 0,
}


# ───────────────────────────── gating ─────────────────────────────


def should_persist_to_secondary_tiers(
    kv_transfer_params: dict[str, Any] | None,
) -> bool:
    """¿Este request puede bajar bloques al tier secundario (disco)?

    Reglas:
      1. `persist_disk` explícito en `kv_transfer_params` manda sobre todo.
      2. Si no, se compara `genesis_agent` (o `agent`) contra la allowlist
         (`GENESIS_KV_DISK_WRITERS`, o `DEFAULT_DISK_WRITERS`).
      3. Sin tag de agente → False (protección estricta: no se escribe).
    """
    if kv_transfer_params and "persist_disk" in kv_transfer_params:
        return bool(kv_transfer_params["persist_disk"])

    agent = None
    if kv_transfer_params:
        agent = kv_transfer_params.get("genesis_agent") or kv_transfer_params.get(
            "agent"
        )

    if not agent:
        return False

    agent_clean = str(agent).strip().lower()

    allowed_env = os.environ.get("GENESIS_KV_DISK_WRITERS")
    if allowed_env is not None:
        allowed_set = {a.strip().lower() for a in allowed_env.split(",") if a.strip()}
    else:
        allowed_set = DEFAULT_DISK_WRITERS

    return agent_clean in allowed_set


# Tope del set de claves tagueadas. Sólo pueden demotarse bloques residentes en
# L2, así que más allá del tamaño de L2 el tag no sirve para nada. Es una red de
# seguridad: el tagueo por uso filtra por residencia antes de llegar acá.
_MAX_TAGGED_KEYS = 16384


def tag_keys_persistence(keys: Iterable[Any]) -> None:
    """Marca claves habilitadas a democión a disco cuando las desalojen de L2."""
    with _LOCK:
        _PERSISTED_KEYS.update(keys)
        excess = len(_PERSISTED_KEYS) - _MAX_TAGGED_KEYS
        if excess > 0:
            for k in list(_PERSISTED_KEYS)[:excess]:
                _PERSISTED_KEYS.discard(k)


def untag_keys_persistence(keys: Iterable[Any]) -> None:
    with _LOCK:
        _PERSISTED_KEYS.difference_update(keys)


def note_persistent_use(keys: Iterable[Any]) -> int:
    """Taguea bloques que un agente persistente está USANDO, no sólo escribiendo.

    Por qué existe
    --------------
    Un bloque de KV es COMPARTIDO — ése es el punto del prefix caching. Si el
    tag se decide sólo en `prepare_store`, queda atado a quién lo escribió
    primero, y eso da los dos errores simétricos:

      - un bloque que un `coder` efímero guardó primero NO se persiste nunca,
        aunque `primary` lo reutilice en cada turno para siempre;
      - un bloque tagueado por `primary` se persiste aunque después sólo lo
        toquen agentes efímeros.

    `touch()` lo llama el scheduler con todas las claves del request, así que
    tagueando ahí el criterio pasa a ser "lo usa alguien que persiste".

    El llamador tiene que filtrar por residencia en L2 antes de llamar: sólo un
    bloque residente puede ser desalojado, y sin ese filtro el set crecería con
    cada clave de cada request.
    """
    n = 0
    with _LOCK:
        for k in keys:
            if k not in _PERSISTED_KEYS:
                _PERSISTED_KEYS.add(k)
                n += 1
        _STATS["tagged_by_use"] += n
        excess = len(_PERSISTED_KEYS) - _MAX_TAGGED_KEYS
        if excess > 0:
            for k in list(_PERSISTED_KEYS)[:excess]:
                _PERSISTED_KEYS.discard(k)
    return n


def is_key_persisted(key: Any) -> bool:
    with _LOCK:
        return key in _PERSISTED_KEYS


# ─────────────────────── snapshots de democión ───────────────────────


def max_pending_bytes() -> int:
    raw = os.environ.get("GENESIS_PN90_MAX_PENDING_MB")
    if raw:
        try:
            return max(1, int(raw)) * 1024 * 1024
        except ValueError:
            pass
    return _DEFAULT_MAX_PENDING_BYTES


def take_demotion_snapshot(key: Any) -> bool:
    """¿Hay que copiar este bloque desalojado de L2 antes de liberar el slot?

    Consume el tag SÓLO si la respuesta es sí.

    Si la cola de pendientes ya llegó al presupuesto se responde que no y **el
    tag se conserva**: no se hace la copia de 28 MB. La versión anterior copiaba
    primero y descartaba el más viejo después, o sea que pagaba la RAM y encima
    perdía un bloque distinto del que causó la presión. Conservar el tag deja
    que el bloque se persista en un desalojo futuro si vuelve a L2.
    """
    with _LOCK:
        if key not in _PERSISTED_KEYS:
            return False
        if _PENDING_BYTES >= max_pending_bytes():
            _STATS["snapshots_refused_budget"] += 1
            return False  # tag conservado a propósito
        _PERSISTED_KEYS.discard(key)
        return True


def record_demotion(key: Any, snapshot: bytes) -> None:
    """Encola la copia cruda de un bloque desalojado para bajarla a disco."""
    global _PENDING_BYTES
    with _LOCK:
        _PENDING_DEMOTIONS.append((key, snapshot))
        _PENDING_BYTES += len(snapshot)
        _STATS["demotions_recorded"] += 1


def drain_demotions() -> list[tuple[Any, bytes]]:
    """Devuelve y vacía la cola de snapshots pendientes."""
    global _PENDING_BYTES
    with _LOCK:
        if not _PENDING_DEMOTIONS:
            return []
        pending = list(_PENDING_DEMOTIONS)
        _PENDING_DEMOTIONS.clear()
        _PENDING_BYTES = 0
        return pending


def pending_demotion_bytes() -> int:
    with _LOCK:
        return _PENDING_BYTES


def submit_pending_demotions(secondary_tiers: Any, req_context: Any = None) -> int:
    """Drena la cola y la entrega a cada tier secundario que sepa demotar.

    Se llama desde `TieringOffloadingManager.prepare_store` (inmediato) y desde
    `on_schedule_end` (red de seguridad para el camino de promoción).
    """
    pending = drain_demotions()
    if not pending:
        return 0
    submitted = failed = 0
    for key, snapshot in pending:
        # Un snapshot cuenta UNA vez aunque haya varios tiers secundarios; antes
        # se sumaba (snapshot × tier) y con dos tiers el contador duplicaba.
        delivered = False
        for tier in secondary_tiers or ():
            fn = getattr(tier, "submit_demote", None)
            if fn is None:
                continue
            try:
                fn(key, snapshot, req_context)
                delivered = True
            except Exception:
                log.warning("PN90: submit_demote falló", exc_info=True)
        if delivered:
            submitted += 1
        else:
            # Ningún tier lo aceptó. El snapshot ya salió de la cola, así que se
            # devuelve el tag: si el bloque vuelve a L2 y lo desalojan otra vez,
            # se reintenta. Antes se perdía en silencio.
            failed += 1
            with _LOCK:
                _PERSISTED_KEYS.add(key)
    with _LOCK:
        _STATS["demotions_submitted"] += submitted
        _STATS["demotions_failed"] += failed
    return submitted


# ─────────────────────── job ids de democión ───────────────────────


def next_demote_job_id() -> int:
    """Id de job para una democión. Negativo y monótono decreciente.

    El `_job_id_counter` del `TieringOffloadingManager` arranca en 0 y sólo
    sube, así que los dos espacios nunca se cruzan.
    """
    global _DEMOTE_JOB_ID
    with _LOCK:
        _DEMOTE_JOB_ID -= 1
        return _DEMOTE_JOB_ID


def is_demote_job_id(job_id: Any) -> bool:
    return isinstance(job_id, int) and job_id < 0


# ─────────────────────────── diagnóstico ───────────────────────────


def stats() -> dict[str, int]:
    with _LOCK:
        out = dict(_STATS)
        out["pending_snapshots"] = len(_PENDING_DEMOTIONS)
        out["pending_bytes"] = _PENDING_BYTES
        out["tagged_keys"] = len(_PERSISTED_KEYS)
        return out


def clear_all() -> None:
    """Resetea TODO el estado. Lo usan los tests y `reset_cache`."""
    global _PENDING_BYTES, _DEMOTE_JOB_ID
    with _LOCK:
        _PERSISTED_KEYS.clear()
        _PENDING_DEMOTIONS.clear()
        _PENDING_BYTES = 0
        _DEMOTE_JOB_ID = 0
        for k in _STATS:
            _STATS[k] = 0
