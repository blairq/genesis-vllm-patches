# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N88 — telemetría de los tiers del cache de KV.

================================================================
QUÉ RESUELVE
================================================================

vLLM instrumenta el offloading a medias. Todo lo que hoy sale a Prometheus
viene de `OffloadingConnectorStats`, que solo ve el tier de **RAM**:

    vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"|"CPU_to_GPU"}
    vllm:kv_offload_total_time_total{transfer_type=...}
    vllm:kv_offload_size{transfer_type=...}

El tier de **disco** no aporta un solo número. Verificado leyendo el código:
`FileSystemTierManager.get_finished_jobs()` construye un `JobResult` con
`job_id` y `success` y nada más, y `record_transfer` solo se llama desde
`offloading/worker.py:279` con datos de `cpu/gpu_worker.py`, que únicamente
maneja GPU↔CPU. O sea: cuántos bytes se escriben al NVMe, cuánto tardan y
cuántos fallan es literalmente invisible.

Medido en este rig antes del parche, sin poder verlo desde /metrics:
PN81 podó 598 veces en 66 h — una cada 6,6 min, ~6 GiB por vez, del orden
de 1,3 TiB/día escritos al SSD. La única forma de enterarse era contar
líneas del log.

Faltan además tres cosas que no son del tier de disco pero que se miden en
el mismo sitio y sin las cuales los números no se interpretan:

 - **Hit rate por tier.** `vllm:external_prefix_cache_hits` fusiona RAM y
   disco, y no es lo mismo un hit de RAM (µs) que uno de NVMe (ms + dos
   steps del scheduler).
 - **Promociones rechazadas.** Cuando el tier primario está lleno,
   `_initiate_promotion` falla y `lookup()` devuelve `False` — lo mismo que
   un miss frío. Pero el scheduler corta el hit de prefijo en el primer
   `False`, así que **no se pierde ese bloque: se pierde todo el prefijo
   restante**. Es un acantilado y hoy es indistinguible de un miss.
 - **Desalojos y ocupación**, que hoy solo existen como texto en stderr.

================================================================
CÓMO
================================================================

La lógica vive en `vllm/_genesis/kv_tier_metrics.py` (Python normal,
versionado y testeable). Este parche solo inyecta las llamadas, de una línea
cada una, para que las anclas sean lo más cortas posible y no deriven con
una versión nueva de vLLM.

Ocho sub-parches sobre cuatro archivos:

  fs/manager.py         A1 submit_store   -> anota bytes/t0/agente
                        A2 submit_load    -> idem
                        A3 get_finished_jobs -> cierra la medición
  tiering/manager.py    B1 lookup primario   -> hit/miss/inflight de RAM
                        B2 lookup secundario -> idem por tier + rechazo
  offloading_connector  C  get_kv_connector_stats -> drena el sink
  offloading/metrics.py D1 aggregate  \\
                        D2 reduce      >- rutean el bucket Genesis
                        D3 observe    /

EL CANAL. El tier de disco corre en el proceso **scheduler** y /metrics lo
sirve el proceso de la API. Lo único que los une es `KVConnectorStats`. Por
eso C abre la rama que hoy dice literalmente *"We only emit stats from the
worker-side"*, y D1/D2/D3 enseñan a los tres puntos que asumen
`isinstance(ops, list)` a reconocer el bucket Genesis.

⚠️ `v1/core/sched/scheduler.py:1363` solo llama a
`self.connector.get_kv_connector_stats()` **si el worker ya produjo stats
ese step**. No se parchea a propósito: el sink acumula y se drena entero en
el primer step que sí pase, así que no se pierde nada — solo se demora. Con
tráfico real siempre hay copias GPU↔CPU, así que la demora es de un step.

================================================================
CARDINALIDAD
================================================================

La label `agent` sale de `kv_transfer_params.genesis_agent`, o sea del
cliente. `agent_label()` la acota contra `GENESIS_KV_AGENTS` (lista cerrada)
y manda todo lo demás a `other`, con `unknown` cuando el request no la trae.
Sin eso, un typo en un `extraBody` de opencode crearía una serie nueva para
siempre. `task_id`, `session_id` y hashes de bloque NO van a labels.

================================================================
COSTO Y SEGURIDAD
================================================================

- Default OFF (`GENESIS_ENABLE_PN88_KV_TIER_METRICS=1`). Con la env apagada
  cada helper devuelve en el primer `if` y el costo es una llamada vacía.
- Todos los helpers van en try/except: una métrica rota no puede tumbar el
  scheduler ni el pool de I/O.
- A3 es un envoltorio transparente: `finish_job` devuelve el mismo
  `JobResult` que recibe.
- No toca el camino de datos: no interviene en store, load ni lookup más
  allá de observar el valor que ya se calculó.
- Si el tier de disco no está configurado, A1-A3 nunca se ejecutan y B1/B2
  solo cuentan lookups de RAM.

================================================================
COMPOSICIÓN
================================================================

Ortogonal a PN81 (cuota del tier de disco): comparten archivo pero no ancla
— PN81 engancha en `shutdown`, PN88 en `submit_*` y `get_finished_jobs`. Se
complementan: PN88 le da a PN81 la métrica `kv_tier_evictions_total` que su
poda hoy solo escribe a stderr.
Ortogonal también a PN87 y PN84, que tocan el camino de corrección del
offloading y no su telemetría.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN88_MARKER = "[Genesis PN88 kv tier metrics]"

_IMPORT = "from vllm._genesis import kv_tier_metrics as _g88"


# ─────────────────────────── fs/manager.py ───────────────────────────

A1_OLD = (
    "        self._pool.enqueue_store(job_metadata.job_id, "
    "len(job_metadata.keys), tasks)\n"
)
A1_NEW = (
    "        # " + GENESIS_PN88_MARKER + "\n"
    "        " + _IMPORT + "\n"
    "        _g88.note_job(self, job_metadata, 'write')\n"
    + A1_OLD
)

A2_OLD = (
    "        self._pool.enqueue_load(job_metadata.job_id, "
    "len(job_metadata.keys), tasks)\n"
)
A2_NEW = (
    "        " + _IMPORT + "\n"
    "        _g88.note_job(self, job_metadata, 'read')\n"
    + A2_OLD
)

A3_OLD = (
    "        return (\n"
    "            JobResult(job_id=job_id, success=success)\n"
    "            for job_id, success in self._pool.get_finished()\n"
    "        )\n"
)
A3_NEW = (
    "        " + _IMPORT + "\n"
    "        return (\n"
    "            _g88.finish_job(self, JobResult(job_id=job_id, success=success))\n"
    "            for job_id, success in self._pool.get_finished()\n"
    "        )\n"
)


# ───────────────────────── tiering/manager.py ─────────────────────────

B1_OLD = (
    "        primary_hit = self.primary_tier.lookup(key, req_context)\n"
    "        if primary_hit is True:\n"
)
B1_NEW = (
    "        # " + GENESIS_PN88_MARKER + "\n"
    "        " + _IMPORT + "\n"
    "        primary_hit = self.primary_tier.lookup(key, req_context)\n"
    "        _g88.note_lookup('ram', primary_hit, key)\n"
    "        if primary_hit is True:\n"
)

B2_OLD = (
    "            result = tier.lookup(key, req_context)\n"
    "            if result is True:\n"
    "                if not self._initiate_promotion(tier, key, req_context):\n"
    "                    return False  # primary full, block unavailable\n"
)
B2_NEW = (
    "            result = tier.lookup(key, req_context)\n"
    "            _g88.note_lookup(\n"
    "                getattr(tier, 'tier_type', 'secondary'), result, key)\n"
    "            if result is True:\n"
    "                if not self._initiate_promotion(tier, key, req_context):\n"
    "                    _g88.note_promotion_refused(\n"
    "                        getattr(tier, 'tier_type', 'secondary'))\n"
    "                    return False  # primary full, block unavailable\n"
)


# ──────────────────────── offloading_connector.py ────────────────────────

C_OLD = (
    "    def get_kv_connector_stats(self) -> KVConnectorStats | None:\n"
    "        if self.connector_worker is None:\n"
    "            return None  # We only emit stats from the worker-side\n"
    "        return self.connector_worker.get_kv_connector_stats()\n"
)
C_NEW = (
    "    # " + GENESIS_PN88_MARKER + "\n"
    "    # El tier de disco vive en el proceso scheduler, donde connector_worker\n"
    "    # es None y upstream devolvia None sin mas. Se drena el sink de PN88 y\n"
    "    # se manda por el mismo canal de stats que ya viaja a /metrics.\n"
    "    def get_kv_connector_stats(self) -> KVConnectorStats | None:\n"
    "        if self.connector_worker is None:\n"
    "            " + _IMPORT + "\n"
    "            if not _g88.enabled():\n"
    "                return None\n"
    "            _payload = _g88.sink().drain()\n"
    "            if _payload is None:\n"
    "                return None\n"
    "            return OffloadingConnectorStats(\n"
    "                data={_g88.GENESIS_BUCKET: _payload})\n"
    "        return self.connector_worker.get_kv_connector_stats()\n"
)


# ────────────────────── offloading/metrics.py ──────────────────────

D1_OLD = (
    "                if k not in self.data:\n"
    "                    self.data[k] = v\n"
    "                else:\n"
    "                    accumulator = self.data[k]\n"
    "                    assert isinstance(accumulator, list)\n"
    "                    accumulator.extend(v)\n"
)
D1_NEW = (
    "                # " + GENESIS_PN88_MARKER + "\n"
    "                if k not in self.data:\n"
    "                    self.data[k] = v\n"
    "                elif k == _G88_BUCKET:\n"
    "                    from vllm._genesis import kv_tier_metrics as _g88m\n"
    "                    _g88m.merge(self.data[k], v)\n"
    "                else:\n"
    "                    accumulator = self.data[k]\n"
    "                    assert isinstance(accumulator, list)\n"
    "                    accumulator.extend(v)\n"
)

D2_OLD = (
    "        for transfer_type, ops_list in self.data.items():\n"
    "            assert isinstance(ops_list, list)\n"
)
D2_NEW = (
    "        for transfer_type, ops_list in self.data.items():\n"
    "            if transfer_type == _G88_BUCKET:\n"
    "                from vllm._genesis import kv_tier_metrics as _g88m\n"
    "                return_dict.update(_g88m.reduce_for_log(ops_list))\n"
    "                continue\n"
    "            assert isinstance(ops_list, list)\n"
)

D3_OLD = (
    "        for transfer_type, ops in transfer_stats_data.items():\n"
    "            # Cache:\n"
)
D3_NEW = (
    "        for transfer_type, ops in transfer_stats_data.items():\n"
    "            if transfer_type == _G88_BUCKET:\n"
    "                from vllm._genesis import kv_tier_metrics as _g88m\n"
    "                _g88m.observe_bucket(self, ops, engine_idx)\n"
    "                continue\n"
    "            # Cache:\n"
)

# La constante del bucket se define una vez a nivel de modulo para que las
# tres ramas de arriba no dependan de importar Genesis solo para comparar.
D0_OLD = "logger = init_logger(__name__)\n"
D0_NEW = (
    "logger = init_logger(__name__)\n"
    "\n"
    "# " + GENESIS_PN88_MARKER + "\n"
    "# Clave del bucket de telemetria de tiers. Empieza con guion bajo para no\n"
    "# chocar con un transfer_type real, que siempre es '<SRC>_to_<DST>'.\n"
    "_G88_BUCKET = \"_genesis_tiers\"\n"
)


# ─────────────────────────── fs/io.py ───────────────────────────
#
# `kv_tier_bytes_total` se calcula AL ENCOLAR, como `bloques × block_size`, y
# se desvía de la realidad por dos motivos independientes: `store_block`
# saltea el archivo si el bloque ya existe (0 bytes, un bloque contado), y con
# PN92 lo que se escribe es el bloque comprimido (~0,56×). Para estimar
# desgaste del SSD hace falta el número del syscall, que es lo que miden E1/E2.

E1_OLD = (
    "        finally:\n"
    "            os.close(fd)\n"
    "        os.replace(tmp_path, dest_path)\n"
)
E1_NEW = (
    "        finally:\n"
    "            os.close(fd)\n"
    "        os.replace(tmp_path, dest_path)\n"
    "        # " + GENESIS_PN88_MARKER + "\n"
    "        " + _IMPORT + "\n"
    "        _g88.note_disk_write(written)\n"
)

E2_OLD = (
    "        bytes_read = os.readv(fd, [view_slice])\n"
    "        if bytes_read < block_size:\n"
    "            raise OSError(f\"Short read: expected {block_size} bytes, read {bytes_read}\")\n"
)
E2_NEW = (
    "        bytes_read = os.readv(fd, [view_slice])\n"
    "        if bytes_read < block_size:\n"
    "            raise OSError(f\"Short read: expected {block_size} bytes, read {bytes_read}\")\n"
    "        " + _IMPORT + "\n"
    "        _g88.note_disk_read(bytes_read)\n"
)


def _fs_io_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/kv_offload/tiering/fs/io.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN88 kv tier metrics (fs io: bytes reales al syscall)",
        target_file=str(target),
        marker=GENESIS_PN88_MARKER,
        sub_patches=[
            TextPatch(name="pn88_io_bytes_written", anchor=E1_OLD,
                      replacement=E1_NEW, required=True),
            TextPatch(name="pn88_io_bytes_read", anchor=E2_OLD,
                      replacement=E2_NEW, required=True),
        ],
    )


def _is_enabled() -> bool:
    import os

    return os.environ.get("GENESIS_ENABLE_PN88_KV_TIER_METRICS") == "1"


def _fs_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/kv_offload/tiering/fs/manager.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN88 kv tier metrics (fs)",
        target_file=str(target),
        marker=GENESIS_PN88_MARKER,
        sub_patches=[
            TextPatch(name="pn88_fs_submit_store", anchor=A1_OLD,
                      replacement=A1_NEW, required=True),
            TextPatch(name="pn88_fs_submit_load", anchor=A2_OLD,
                      replacement=A2_NEW, required=True),
            TextPatch(name="pn88_fs_finished", anchor=A3_OLD,
                      replacement=A3_NEW, required=True),
        ],
        upstream_drift_markers=[
            # Si upstream le pone telemetria propia al tier de disco, este
            # parche sobra y se pelearia con la de ellos -> SKIP limpio.
            "transfer_size",
            "record_transfer",
        ],
    )


def _tiering_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/kv_offload/tiering/manager.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN88 kv tier metrics (tiering)",
        target_file=str(target),
        marker=GENESIS_PN88_MARKER,
        sub_patches=[
            TextPatch(name="pn88_lookup_primary", anchor=B1_OLD,
                      replacement=B1_NEW, required=True),
            TextPatch(name="pn88_lookup_secondary", anchor=B2_OLD,
                      replacement=B2_NEW, required=True),
        ],
    )


def _connector_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "distributed/kv_transfer/kv_connector/v1/offloading_connector.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN88 kv tier metrics (connector)",
        target_file=str(target),
        marker=GENESIS_PN88_MARKER,
        sub_patches=[
            TextPatch(name="pn88_scheduler_stats", anchor=C_OLD,
                      replacement=C_NEW, required=True),
        ],
    )


def _prom_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "distributed/kv_transfer/kv_connector/v1/offloading/metrics.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN88 kv tier metrics (prom)",
        target_file=str(target),
        marker=GENESIS_PN88_MARKER,
        sub_patches=[
            TextPatch(name="pn88_bucket_const", anchor=D0_OLD,
                      replacement=D0_NEW, required=True),
            TextPatch(name="pn88_aggregate", anchor=D1_OLD,
                      replacement=D1_NEW, required=True),
            TextPatch(name="pn88_reduce", anchor=D2_OLD,
                      replacement=D2_NEW, required=True),
            TextPatch(name="pn88_observe", anchor=D3_OLD,
                      replacement=D3_NEW, required=True),
        ],
    )


_PATCHERS = (_fs_patcher, _fs_io_patcher, _tiering_patcher,
             _connector_patcher, _prom_patcher)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN88")
    log_decision("PN88", decision, reason)
    if not decision:
        return "skipped", reason
    if not _is_enabled():
        return "skipped", (
            "GENESIS_ENABLE_PN88_KV_TIER_METRICS not set; default OFF. "
            "Publica en Prometheus la telemetria del tier de disco del cache "
            "de KV (bytes, latencia, errores), el hit rate separado por tier, "
            "las promociones rechazadas por tier primario lleno, y los "
            "desalojos de PN81 — nada de lo cual expone vLLM hoy."
        )
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    applied: list[str] = []
    for factory in _PATCHERS:
        p = factory()
        if p is None:
            return "skipped", f"target de {factory.__name__} no encontrado"
        result, failure = p.apply()
        status, _msg = result_to_wiring_status(
            result, failure,
            applied_message="ok", patch_name=p.patch_name,
        )
        if status == "failed":
            return "failed", f"{p.patch_name}: {failure}"
        applied.append(f"{p.patch_name}={status}")

    return "applied", (
        "PN88 applied: telemetria de tiers del cache de KV. Publica "
        "vllm:kv_tier_{bytes,ops,op_seconds,errors,lookups,promotion_refused,"
        "evictions,bytes_used,capacity_bytes} con labels tier/direction/agent. "
        "El tier de disco pasa de cero metricas a medido. [" +
        ", ".join(applied) + "]"
    )


def is_applied() -> bool:
    """Reporter para verify_live_rebinds en apply_all.py."""
    if vllm_install_root() is None:
        return False
    for factory in _PATCHERS:
        p = factory()
        if p is None:
            return False
        try:
            with open(p.target_file) as f:
                if GENESIS_PN88_MARKER not in f.read():
                    return False
        except Exception:
            return False
    return True
