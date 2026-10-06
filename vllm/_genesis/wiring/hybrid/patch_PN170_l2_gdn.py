# SPDX-License-Identifier: Apache-2.0
"""PN170 — L2 con el estado GDN: los grupos GDN del target no cuentan como EAGLE en el connector de offload;
ver ``vllm._genesis.l2_gdn``. Mas diagnosticos opcionales (handoffs, prepare_store, sumas por bloque)."""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN170: L2 con estado GDN]"
_IMP_OLD = "from vllm.v1.core.sched.output import SchedulerOutput\n"
_IMP = "from vllm._genesis import l2_gdn as _g170  # " + MARKER + "\n"
_P_OLD = (
    "            store_output = self.manager.prepare_store(\n"
    "                new_offload_keys, req_status.req_context\n"
    "            )\n"
)
_P_NEW = _P_OLD + "            _g170.tras_prepare(req, new_offload_keys, store_output)  # " + MARKER + "\n"
_E_OLD = (
    "        if use_eagle and not eagle_groups:\n"
    "            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))\n"
)
_E_NEW = _E_OLD + (
    "        if _g170.ACTIVO:  # " + MARKER + "\n"
    "            eagle_groups = _g170.sin_eagle_en_mamba(eagle_groups, kv_cache_config)\n"
)
_H_OLD = (
    "                key = self._make_boundary_key(req, group_idx, boundary)\n"
    "                store_output = self.manager.prepare_store([key], req_status.req_context)\n"
)
_H_NEW = (
    "                _g170.handoff(req, group_idx, block_id, boundary)  # " + MARKER + "\n"
    + _H_OLD
)
_S1_OLD = (
    "    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:\n"
    "        self.worker = self.spec.get_worker(kv_caches)\n"
)
_S1_NEW = _S1_OLD + (
    "        from vllm._genesis import l2_gdn as _g170  # " + MARKER + "\n"
    "        _g170.registrar_kv(kv_caches)\n"
)
_S2_OLD = (
    "        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:\n"
    "            success = self.worker.submit_store(job_id, src_spec, dst_spec)\n"
    "            assert success\n"
    "        self._unsubmitted_store_jobs.clear()\n"
    "\n"
    "        for job_id, entry in metadata.load_jobs.items():\n"
    "            self._load_jobs[job_id] = entry.req_id\n"
)
_S2_NEW = (
    "        from vllm._genesis import l2_gdn as _g170  # " + MARKER + "\n"
    "        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:\n"
    "            _g170.al_guardar(job_id, src_spec)\n"
    "            success = self.worker.submit_store(job_id, src_spec, dst_spec)\n"
    "            assert success\n"
    "        self._unsubmitted_store_jobs.clear()\n"
    "\n"
    "        for job_id, entry in metadata.load_jobs.items():\n"
    "            _g170.al_cargar(job_id, entry.dst_spec)\n"
    "            self._load_jobs[job_id] = entry.req_id\n"
)
_S3_OLD = (
    "            self._connector_worker_meta.mark_completed(job_id)\n"
    "            req_id = self._load_jobs.pop(job_id, None)\n"
)
_S3_NEW = (
    "            self._connector_worker_meta.mark_completed(job_id)\n"
    "            from vllm._genesis import l2_gdn as _g170  # " + MARKER + "\n"
    "            _g170.carga_lista(job_id)\n"
    "            req_id = self._load_jobs.pop(job_id, None)\n"
)
_S0_OLD = (
    "        # Submit deferred stores from previous step (and jobs_to_flush above).\n"
    "        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:\n"
    "            assert isinstance(src_spec, GPULoadStoreSpec)\n"
)
_S0_NEW = _S0_OLD + (
    "            from vllm._genesis import l2_gdn as _g170  # " + MARKER + "\n"
    "            _g170.al_guardar(job_id, src_spec)\n"
)
# PN171 (GENESIS_ENABLE_PN171_BORRADOR_RECORTADO): del grupo del borrador solo los chunks que puede pedir un lookup
_R_OLD = (
    "                        group_config.sliding_window_size_in_chunks,\n"
    "                        group_config.is_eagle_group,\n"
    "                    ):\n"
    "                        continue\n"
    "                    new_offload_keys.append(offload_key)\n"
)
_R_NEW = (
    "                        group_config.sliding_window_size_in_chunks,\n"
    "                        group_config.is_eagle_group,\n"
    "                    ):\n"
    "                        continue\n"
    "                    if _g170.saltear_ventana(req, group_config, abs_chunk_idx):  # " + MARKER + "\n"
    "                        continue\n"
    "                    new_offload_keys.append(offload_key)\n"
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN170")
    log_decision("PN170", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py")
    if f is None:
        return "skipped", "falta offloading/scheduler.py"
    p = TextPatcher(patch_name="PN170 L2 con estado GDN", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn170_imp", anchor=_IMP_OLD, replacement=_IMP_OLD + _IMP, required=True),
        TextPatch(name="pn170_prepare", anchor=_P_OLD, replacement=_P_NEW, required=True),
        TextPatch(name="pn170_eagle", anchor=_E_OLD, replacement=_E_NEW, required=True),
        TextPatch(name="pn170_handoff", anchor=_H_OLD, replacement=_H_NEW, required=True),
        TextPatch(name="pn171_recorte", anchor=_R_OLD, replacement=_R_NEW, required=True)],
        upstream_drift_markers=["_g170"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    f2 = resolve_vllm_file("distributed/kv_transfer/kv_connector/v1/offloading/worker.py")
    if f2 is not None:
        p2 = TextPatcher(patch_name="PN170 diagnostico (worker)", target_file=str(f2), marker=MARKER, sub_patches=[
            TextPatch(name="pn170_s0", anchor=_S0_OLD, replacement=_S0_NEW, required=True),
            TextPatch(name="pn170_s1", anchor=_S1_OLD, replacement=_S1_NEW, required=True),
            TextPatch(name="pn170_s2", anchor=_S2_OLD, replacement=_S2_NEW, required=True),
            TextPatch(name="pn170_s3", anchor=_S3_OLD, replacement=_S3_NEW, required=True)],
            upstream_drift_markers=["_entera170"])
        p2.apply()
    return "applied", "offload: grupos GDN sin la ventana de EAGLE (L2 acierta)"
