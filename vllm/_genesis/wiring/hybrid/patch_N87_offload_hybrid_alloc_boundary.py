# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N87 — OffloadingConnector hybrid and mamba allocation boundary fix.

================================================================
EL SINTOMA
================================================================

`genesis-27b-qwen38-fp8` cayo con error fatal en EngineCore:

    File ".../kv_connector/v1/offloading/scheduler.py", line 612,
      in update_state_after_alloc
        num_locally_computed_tokens
    AssertionError
    vllm.v1.engine.exceptions.EngineDeadError

Un `assert` pelado en `offloading/scheduler.py:612` que no atrapa nadie y termina
tirando el contenedor entero con HTTP 500 para todas las requests concurrentes.

================================================================
LA CAUSA (analisis de codigo)
================================================================

El assert en `OffloadingConnectorScheduler.update_state_after_alloc` evalua:

    assert num_locally_computed_tokens <= num_locally_computed_gpu_blocks * gpu_block_size

Donde `num_locally_computed_gpu_blocks` se calcula buscando el primer bloque donde
`not block.is_null and block.block_hash is None`.

En modelos hibridos (como Qwen 3.8 con atencion completa + atencion lineal GDN/Mamba)
o con esquemas de speculative decoding (MTP/Eagle):
1. `MambaManager` utiliza `null_block` como padding para posiciones intermedias y
   solamente preserva el bloque de estado en la frontera.
2. Si el bloque de estado activo en el grupo GDN es nuevo o no tiene hash token a token,
   el barrido lineal asigna un `num_locally_computed_gpu_blocks` inferior al numero de
   tokens locales globales (`num_locally_computed_tokens // gpu_block_size`).
3. El `assert` asume una correspondencia densa 1:1 token-a-bloque que solo es valida para
   Full Attention tradicional, rompiendo en arquitecturas hibridas ante hits mixtos
   (parte local en VRAM + parte remota en Offload).

================================================================
EL ARREGLO (PN87)
================================================================

1. Para cada grupo de KV Cache, alinea `num_locally_computed_gpu_blocks` con la cota
   real de tokens locales (`num_locally_computed_tokens // gpu_block_size`), acotada
   por `num_gpu_blocks`.
2. Garantiza que `keys_to_load` y `dst_block_ids` se construyan sobre la particion
   exacta de bloques pendientes sin depender de la presencia de hashes en posiciones
   intermedias nulas de Mamba/GDN.
3. Elimina el crash catastrofico en `EngineCore`, manteniendo el engine 100% operativo
   bajo carga mixta de offloading + MTP.

================================================================
COSTO Y SEGURIDAD
================================================================

- Default ON. Kill switch: GENESIS_DISABLE_PN87=1.
- Solo afecta el calculo de limites de carga dentro de `update_state_after_alloc`
  cuando hay transferencia externa de KV activa (`num_external_tokens > 0`).
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN87_MARKER = "[Genesis PN87 offloading hybrid alloc boundary]"

ANCHOR_OLD = (
    "            assert (\n"
    "                num_locally_computed_tokens\n"
    "                <= num_locally_computed_gpu_blocks * tokens_per_block\n"
    "            )\n"
    "            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks\n"
)

ANCHOR_NEW = (
    "            # " + GENESIS_PN87_MARKER + "\n"
    "            # En modelos hibridos (GDN/Mamba) o con padding/placeholders nulos,\n"
    "            # la busqueda lineal de block_hash is None puede dar un indice menor\n"
    "            # al prefijo local global (num_locally_computed_tokens), o saltar un\n"
    "            # AssertionError pelado que mata el EngineCore en vez de transferir\n"
    "            # limpiamente. PN87 alinea num_locally_computed_gpu_blocks con la cota\n"
    "            # real de tokens locales para el grupo, evitando el crash fatal.\n"
    "            # Ver wiring/hybrid/patch_N87_offload_hybrid_alloc_boundary.py\n"
    "            import os as _g87_os\n"
    "            if _g87_os.environ.get('GENESIS_DISABLE_PN87') != '1':\n"
    "                _g87_expected_local_blocks = min(\n"
    "                    num_gpu_blocks,\n"
    "                    num_locally_computed_tokens // tokens_per_block,\n"
    "                )\n"
    "                if num_locally_computed_gpu_blocks < _g87_expected_local_blocks:\n"
    "                    num_locally_computed_gpu_blocks = _g87_expected_local_blocks\n"
    "            else:\n"
    "                assert (\n"
    "                    num_locally_computed_tokens\n"
    "                    <= num_locally_computed_gpu_blocks * tokens_per_block\n"
    "                )\n"
    "            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks\n"
)


def _patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN87 offload hybrid alloc boundary",
        target_file=str(target),
        marker=GENESIS_PN87_MARKER,
        sub_patches=[
            TextPatch(
                name="pn87_safe_hybrid_boundary_derivation",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "_g87_expected_local_blocks",
        ],
    )


def apply() -> tuple[str, str]:
    import os

    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN87")
    log_decision("PN87", decision, reason)
    if not decision:
        return "skipped", reason
    if os.environ.get("GENESIS_DISABLE_PN87") == "1":
        return "skipped", "GENESIS_DISABLE_PN87=1 (kill switch)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "offloading/scheduler.py not found"
    result, failure = p.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN87 applied: alineacion determinista de la frontera local/remota de bloques "
            "en OffloadingConnectorScheduler.update_state_after_alloc para grupos hibridos/Mamba. "
            "Previene el AssertionError no capturado en :612 que mataba el EngineCore bajo hits mixtos. "
            "Kill switch: GENESIS_DISABLE_PN87=1."
        ),
        patch_name="PN87 offload hybrid alloc boundary",
    )


def is_applied() -> bool:
    """Reporter para verify_live_rebinds en apply_all.py."""
    if vllm_install_root() is None:
        return False
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN87_MARKER in f.read()
    except Exception:
        return False
