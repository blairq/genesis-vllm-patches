# SPDX-License-Identifier: Apache-2.0
"""Wiring del parche N90 — KV Disk Write Gating & Tiered Demotion (L2 → L3 al desalojar).

Reemplaza el write-through de vLLM (cascada a disco en `complete_store`) por una
L3 exclusiva: el bloque baja a SSD sólo cuando lo desalojan de L2 RAM, y sólo si
el agente está habilitado a persistir.

Sitios parcheados
-----------------
`v1/kv_offload/tiering/fs/manager.py`
    + `FileSystemTierManager.submit_demote()` — escribe un snapshot suelto, con
      job id negativo (espacio disjunto del contador del tiering manager).

`v1/kv_offload/tiering/manager.py`
    + `CPUPrimaryTierOffloadingManager.prepare_store()` — override que toma el
      snapshot del bloque desalojado ANTES de liberar el slot.
    ~ `TieringOffloadingManager.prepare_store()` — drena y entrega los snapshots.
    ~ `TieringOffloadingManager.on_schedule_end()` — drenaje de red de seguridad
      para las evicciones que dispara `_initiate_promotion`.
    ~ `TieringOffloadingManager.complete_store()` — se elimina la cascada.
    ~ `TieringOffloadingManager._process_finished_jobs()` — ignora los jobs de
      democión SIN perder el assert de los jobs reales.

`v1/core/block_pool.py`
    ~ registra el `BlockPool` para el chequeo opcional de residencia en L1.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
)

GENESIS_PN90_MARKER = "[Genesis PN90 kv disk write gating & tiered demotion]"

# ─────────────────── fs/manager.py ───────────────────

FS_ANCHOR_OLD = (
    "    @override\n"
    "    def get_finished_jobs(self) -> Iterable[JobResult]:\n"
)

FS_ANCHOR_NEW = (
    "    # " + GENESIS_PN90_MARKER + "\n"
    "    def submit_demote(\n"
    "        self,\n"
    "        key: OffloadKey,\n"
    "        snapshot: bytes,\n"
    "        req_context: ReqContext | None = None,\n"
    "    ) -> None:\n"
    "        \"\"\"Baja a disco un bloque ya desalojado de L2, desde una copia suelta.\n"
    "\n"
    "        A diferencia de submit_store, el dato NO vive mas en el primary tier:\n"
    "        el slot ya fue reutilizado. Por eso se escribe desde `snapshot`.\n"
    "        \"\"\"\n"
    "        from vllm._genesis import kv_disk_gate as _g90\n"
    "        dest_path = self.file_mapper.get_file_name(key)\n"
    "        if os.path.exists(dest_path):\n"
    "            return\n"
    "        buf = memoryview(snapshot)\n"
    "        task = functools.partial(store_block, dest_path, buf, 0, len(buf))\n"
    "        _job_id = _g90.next_demote_job_id()\n"
    "        from vllm._genesis import kv_tier_metrics as _g88\n"
    "        if _g88.enabled():\n"
    "            try:\n"
    "                _params = getattr(req_context, 'kv_transfer_params', None)\n"
    "                _g88.note_demote(self, _job_id, len(buf), _g88.agent_label(_params))\n"
    "            except Exception:\n"
    "                pass\n"
    "        self._pool.enqueue_store(_job_id, 1, [task])\n"
    "\n"
) + FS_ANCHOR_OLD


# ─────────────────── tiering/manager.py ───────────────────

PRIMARY_STORE_OLD = (
    "    def get_kv_memoryview(self) -> memoryview:\n"
    "        \"\"\"Return the memoryview over the primary tier's KV cache buffer.\n"
    "\n"
    "        The view has shape (num_blocks, row_stride_bytes) and is backed by the\n"
    "        SharedOffloadRegion mmap.  Secondary tiers address block *b* as\n"
    "        ``view[b]``.\n"
    "        \"\"\"\n"
    "        return self._kv_memoryview\n"
)

PRIMARY_STORE_NEW = (
    "    # " + GENESIS_PN90_MARKER + "\n"
    "    @override\n"
    "    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:\n"
    "        \"\"\"Taguea por USO, no por primer escritor.\n"
    "\n"
    "        Un bloque de KV es compartido. Decidir la persistencia solo en\n"
    "        prepare_store lo ata a quien lo escribio primero: un bloque que un\n"
    "        agente efimero guardo primero no se persistia nunca, aunque un\n"
    "        agente persistente lo reutilizara en cada turno.\n"
    "\n"
    "        Se filtra por residencia en L2 (`self._policy.get`) porque solo un\n"
    "        bloque residente puede ser desalojado, y sin ese filtro el set de\n"
    "        tags crece con cada clave de cada request.\n"
    "        \"\"\"\n"
    "        from vllm._genesis import kv_disk_gate as _g90\n"
    "        if _g90.should_persist_to_secondary_tiers(\n"
    "            getattr(req_context, 'kv_transfer_params', None)\n"
    "        ):\n"
    "            _g90.note_persistent_use(\n"
    "                k for k in keys if self._policy.get(k) is not None\n"
    "            )\n"
    "        self._policy.touch(keys, req_context)\n"
    "\n"
    "    # " + GENESIS_PN90_MARKER + "\n"
    "    @override\n"
    "    def prepare_store(\n"
    "        self,\n"
    "        keys: Collection[OffloadKey],\n"
    "        req_context: ReqContext,\n"
    "    ) -> PrepareStoreOutput | None:\n"
    "        \"\"\"Igual que CPUOffloadingManager.prepare_store, mas dos cosas:\n"
    "\n"
    "        - copia la fila del mmap de cada bloque desalojado ANTES de liberar el\n"
    "          slot (el slot se reutiliza en esta misma llamada);\n"
    "        - marca las claves nuevas como persistibles si el agente lo permite.\n"
    "        \"\"\"\n"
    "        from vllm._genesis import kv_disk_gate as _g90\n"
    "        should_persist = _g90.should_persist_to_secondary_tiers(\n"
    "            getattr(req_context, 'kv_transfer_params', None)\n"
    "        )\n"
    "        if self.counts is not None:\n"
    "            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]\n"
    "        keys_to_store = [k for k in keys if self._policy.get(k) is None]\n"
    "        if not keys_to_store:\n"
    "            return PrepareStoreOutput(\n"
    "                keys_to_store=[],\n"
    "                store_spec=self._get_load_store_spec([], []),\n"
    "                evicted_keys=[],\n"
    "            )\n"
    "        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()\n"
    "        to_evict: list[OffloadKey] = []\n"
    "        if num_blocks_to_evict > 0:\n"
    "            protected = set(keys)\n"
    "            evicted = self._policy.evict(num_blocks_to_evict, protected)\n"
    "            if evicted is None:\n"
    "                return None\n"
    "            _region = self._mmap_region\n"
    "            _mm = getattr(_region, 'mmap_obj', None)\n"
    "            for key, block in evicted:\n"
    "                if _mm is not None and _g90.take_demotion_snapshot(key):\n"
    "                    _stride = _region._row_stride\n"
    "                    _off = int(block.block_id) * _stride\n"
    "                    _g90.record_demotion(key, bytes(_mm[_off : _off + _stride]))\n"
    "                self._free_block(block)\n"
    "                to_evict.append(key)\n"
    "            # [Genesis PN95: desalojos de L2]\n"
    "            # Va ACA y no en CPUOffloadingManager.prepare_store porque\n"
    "            # este override lo reemplaza: el hook de la clase base nunca\n"
    "            # se ejecuta con el tiering manager activo. El desalojo de L2\n"
    "            # es lo que las lineas de arriba convierten en escritura al\n"
    "            # SSD, asi que es el contador que cierra la cadena.\n"
    "            if to_evict:\n"
    "                try:\n"
    "                    from vllm._genesis import kv_sparse_gdn as _g95_gdn\n"
    "                    from vllm._genesis import kv_tier_metrics as _g95_m\n"
    "\n"
    "                    _g95_bb = _g95_gdn.block_write_bytes()\n"
    "                    _g95_m.note_eviction(\n"
    "                        'ram',\n"
    "                        'capacity',\n"
    "                        count=len(to_evict),\n"
    "                        freed_bytes=len(to_evict) * _g95_bb,\n"
    "                    )\n"
    "                except Exception:\n"
    "                    pass\n"
    "        if to_evict and self.events is not None:\n"
    "            self.events.append(\n"
    "                OffloadingEvent(\n"
    "                    keys=to_evict,\n"
    "                    medium=self.medium,\n"
    "                    removed=True,\n"
    "                )\n"
    "            )\n"
    "        blocks = self._allocate_blocks(keys_to_store)\n"
    "        assert len(blocks) == len(keys_to_store)\n"
    "        for key, block in zip(keys_to_store, blocks):\n"
    "            self._policy.insert(key, block)\n"
    "        if should_persist:\n"
    "            _g90.tag_keys_persistence(keys_to_store)\n"
    "        store_spec = self._get_load_store_spec(keys_to_store, blocks)\n"
    "        return PrepareStoreOutput(\n"
    "            keys_to_store=keys_to_store,\n"
    "            store_spec=store_spec,\n"
    "            evicted_keys=to_evict,\n"
    "        )\n"
    "\n"
) + PRIMARY_STORE_OLD


TIER_PREPARE_OLD = (
    "        primary_result = self.primary_tier.prepare_store(keys, req_context)\n"
    "\n"
    "        if primary_result is None:\n"
    "            return None\n"
)

TIER_PREPARE_NEW = (
    "        primary_result = self.primary_tier.prepare_store(keys, req_context)\n"
    "\n"
    "        # " + GENESIS_PN90_MARKER + "\n"
    "        # Se drena SIEMPRE, incluso si primary_result es None: los snapshots\n"
    "        # ya fueron tomados y el slot de L2 ya se reutilizo.\n"
    "        from vllm._genesis import kv_disk_gate as _g90\n"
    "        _g90.submit_pending_demotions(self.secondary_tiers, req_context)\n"
    "\n"
    "        if primary_result is None:\n"
    "            return None\n"
)

TIER_SCHEDULE_END_OLD = (
    "        self._processed_jobs_this_step = False\n"
    "\n"
    "        self._flush_pending_promotions()\n"
)

TIER_SCHEDULE_END_NEW = (
    "        # " + GENESIS_PN90_MARKER + "\n"
    "        # Red de seguridad: _initiate_promotion llama a\n"
    "        # primary_tier.prepare_write() directamente, sin pasar por\n"
    "        # TieringOffloadingManager.prepare_store. Las evicciones de ese camino\n"
    "        # dejan snapshots que nadie drenaria.\n"
    "        from vllm._genesis import kv_disk_gate as _g90\n"
    "        _g90.submit_pending_demotions(self.secondary_tiers, None)\n"
    "        self._processed_jobs_this_step = False\n"
    "\n"
    "        self._flush_pending_promotions()\n"
)

TIER_COMPLETE_OLD = (
    "        if success:\n"
    "            # Step 2: Cascade to ALL secondary tiers\n"
    "            # For each secondary tier, call primary.prepare_read() to get the\n"
    "            # LoadStoreSpec AND to increment ref_cnt (protecting blocks from\n"
    "            # eviction during the async transfer). One prepare_read() call per\n"
    "            # secondary tier.\n"
    "            for tier in self.secondary_tiers:\n"
    "                job_metadata = self.create_store_job(keys, req_context)\n"
    "                tier.submit_store(job_metadata)\n"
)

TIER_COMPLETE_NEW = (
    "        if success:\n"
    "            # " + GENESIS_PN90_MARKER + "\n"
    "            # Democion por desalojo: NO hay cascada al guardar. El bloque baja\n"
    "            # a L3 SSD unicamente cuando lo desalojan de L2 RAM (ver el override\n"
    "            # de CPUPrimaryTierOffloadingManager.prepare_store). El cuerpo de la\n"
    "            # cascada se elimina pero el metodo sigue: abajo queda la contabilidad\n"
    "            # de pending_primary_stores y _maybe_finalize_request.\n"
    "            pass\n"
)

FINISHED_JOBS_OLD = (
    "            for completed_job in tier.get_finished_jobs():\n"
    "                job_id = completed_job.job_id\n"
    "                job_metadata = self._transfer_jobs.pop(job_id, None)\n"
    "                assert job_metadata is not None, (\n"
    "                    f\"Finished job_id {job_id} from tier #{i}\"\n"
    "                    f\" ({tier.tier_type}) not in _transfer_jobs\"\n"
    "                )\n"
)

FINISHED_JOBS_NEW = (
    "            # " + GENESIS_PN90_MARKER + "\n"
    "            # Los jobs de democion usan ids NEGATIVOS y no tienen JobMetadata\n"
    "            # (no tomaron ref_cnt: el bloque ya no esta en primary). Se filtran\n"
    "            # por signo para no perder el assert de los jobs reales.\n"
    "            from vllm._genesis import kv_disk_gate as _g90\n"
    "            for completed_job in tier.get_finished_jobs():\n"
    "                job_id = completed_job.job_id\n"
    "                if _g90.is_demote_job_id(job_id):\n"
    "                    continue\n"
    "                job_metadata = self._transfer_jobs.pop(job_id, None)\n"
    "                assert job_metadata is not None, (\n"
    "                    f\"Finished job_id {job_id} from tier #{i}\"\n"
    "                    f\" ({tier.tier_type}) not in _transfer_jobs\"\n"
    "                )\n"
)


# ─────────────────── camino REQUEST_LEVEL ───────────────────

REQUEST_LEVEL_OLD = (
    "        # Filter out keys that are not ready in primary (e.g. in-flight)\n"
    "        ready_keys = tuple(\n"
    "            k\n"
    "            for k in keys\n"
    "            if self.primary_tier.lookup(k, req_context) is LookupResult.HIT\n"
    "        )\n"
)

REQUEST_LEVEL_NEW = (
    "        # " + GENESIS_PN90_MARKER + "\n"
    "        # Este camino tambien escribe a un tier secundario, asi que pasa por\n"
    "        # el mismo gate. Esta dormido para el tier fs (politica BLOCK_LEVEL),\n"
    "        # pero dejarlo sin gatear era una via de escritura sin control.\n"
    "        from vllm._genesis import kv_disk_gate as _g90\n"
    "        if not _g90.should_persist_to_secondary_tiers(\n"
    "            getattr(req_context, 'kv_transfer_params', None)\n"
    "        ):\n"
    "            return\n"
    "        # Filter out keys that are not ready in primary (e.g. in-flight)\n"
    "        ready_keys = tuple(\n"
    "            k\n"
    "            for k in keys\n"
    "            if self.primary_tier.lookup(k, req_context) is LookupResult.HIT\n"
    "        )\n"
)


# ─────────────────── patchers ───────────────────


def _fs_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/kv_offload/tiering/fs/manager.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN90 kv disk tiered demotion (fs)",
        target_file=str(target),
        marker=GENESIS_PN90_MARKER,
        sub_patches=[
            TextPatch(
                name="pn90_fs_submit_demote",
                anchor=FS_ANCHOR_OLD,
                replacement=FS_ANCHOR_NEW,
                required=True,
            ),
        ],
    )


def _tiering_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/kv_offload/tiering/manager.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN90 kv disk tiered demotion (tiering)",
        target_file=str(target),
        marker=GENESIS_PN90_MARKER,
        sub_patches=[
            TextPatch(
                name="pn90_primary_tier_prepare_store",
                anchor=PRIMARY_STORE_OLD,
                replacement=PRIMARY_STORE_NEW,
                required=True,
            ),
            TextPatch(
                name="pn90_tiering_prepare_demote",
                anchor=TIER_PREPARE_OLD,
                replacement=TIER_PREPARE_NEW,
                required=True,
            ),
            TextPatch(
                name="pn90_tiering_schedule_end_drain",
                anchor=TIER_SCHEDULE_END_OLD,
                replacement=TIER_SCHEDULE_END_NEW,
                required=True,
            ),
            TextPatch(
                name="pn90_tiering_no_cascade_store",
                anchor=TIER_COMPLETE_OLD,
                replacement=TIER_COMPLETE_NEW,
                required=True,
            ),
            TextPatch(
                name="pn90_gate_request_level_cascade",
                anchor=REQUEST_LEVEL_OLD,
                replacement=REQUEST_LEVEL_NEW,
                required=True,
            ),
            TextPatch(
                name="pn90_tiering_finished_jobs_allow_demote",
                anchor=FINISHED_JOBS_OLD,
                replacement=FINISHED_JOBS_NEW,
                required=True,
            ),
        ],
    )


def _patchers() -> list[TextPatcher] | None:
    out: list[TextPatcher] = []
    for factory in (_fs_patcher, _tiering_patcher):
        p = factory()
        if p is None:
            return None
        out.append(p)
    return out


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN90")
    log_decision("PN90", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patchers = _patchers()
    if patchers is None:
        return "skipped", "algún target de PN90 no encontrado"

    # Tres archivos distintos: se aplican con validate-all-then-write-all para
    # no dejar un árbol a medio parchear si el ancla del segundo derivó.
    txn = MultiFilePatchTransaction(patchers, name="PN90")
    status, reason = txn.apply_or_skip()
    if status != "applied":
        return status, reason

    return "applied", (
        "PN90 aplicado: gating de escritura a disco + democión por desalojo "
        "(las escrituras a SSD ocurren sólo cuando el bloque sale de L2 RAM, "
        "y sólo para agentes persistentes)."
    )


def is_applied() -> bool:
    patchers = _patchers()
    if patchers is None:
        return False
    for p in patchers:
        try:
            with open(p.target_file) as f:
                if GENESIS_PN90_MARKER not in f.read():
                    return False
        except Exception:
            return False
    return True
