# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN93 — checkpointing disperso del estado recurrente (GDN/Mamba).

Un solo archivo, dos anclas, ambas en el scheduler del connector de offloading:

  1. `SchedulerOffloadConfig.from_spec` — registra qué grupos son recurrentes.
     Es el único punto donde conviven el índice del grupo y su `KVCacheSpec`.

  2. El filtro de `new_offload_keys` en `_get_offload_jobs` — descarta los
     estados recurrentes intermedios ANTES de `manager.prepare_store`, con lo
     cual el bloque no entra ni a L2 RAM ni a L3 SSD.

Se parchea el scheduler y NO `fs/io.py` a propósito: filtrar en el scheduler
cubre las dos capas con un solo hook, y deja `fs/io.py` libre para PN92.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN93_MARKER = "[Genesis PN93: Sparse GDN Checkpointing]"

# ─────────── 1. registro de grupos recurrentes ───────────

REGISTER_OLD = (
    "        for idx, gpu_block_size in enumerate(spec.gpu_block_size):\n"
    "            kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec\n"
)

REGISTER_NEW = (
    "        for idx, gpu_block_size in enumerate(spec.gpu_block_size):\n"
    "            kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec\n"
    "            # " + GENESIS_PN93_MARKER + "\n"
    "            from vllm._genesis import kv_sparse_gdn as _g93\n"
    "            # El tercer argumento es lo que store_block escribe por bloque.\n"
    "            # page_size_bytes mide el ESTADO recurrente (851.968 B aca) y no\n"
    "            # el bloque en disco (28.966.912 B): con el primero el contador de\n"
    "            # bytes ahorrados iba 34x corto.\n"
    "            _g93.register_group_spec(\n"
    "                idx, kv_spec, getattr(spec, 'kv_bytes_per_offloaded_block', 0)\n"
    "            )\n"
)

# ─────────── 2. filtro de bloques a offloadear ───────────

FILTER_OLD = (
    "                    if alignment_block_count is not None:\n"
    "                        assert tail is not None\n"
    "                        abs_block_idx = start_block_idx + key_idx\n"
    "                        pos_in_segment = abs_block_idx % alignment_block_count\n"
    "                        if pos_in_segment < alignment_block_count - tail:\n"
    "                            continue\n"
    "                    new_offload_keys.append(offload_key)\n"
)

FILTER_NEW = (
    "                    if alignment_block_count is not None:\n"
    "                        assert tail is not None\n"
    "                        abs_block_idx = start_block_idx + key_idx\n"
    "                        pos_in_segment = abs_block_idx % alignment_block_count\n"
    "                        if pos_in_segment < alignment_block_count - tail:\n"
    "                            continue\n"
    "                    # " + GENESIS_PN93_MARKER + "\n"
    "                    # Los grupos recurrentes (Mamba/GDN) se resuelven con\n"
    "                    # _sliding_window_lookup(window=1): solo se lee el ULTIMO\n"
    "                    # bloque presente. Guardar uno de cada `stride` hace que el\n"
    "                    # hit redondee hacia abajo al checkpoint anterior, que es\n"
    "                    # correcto por construccion.\n"
    "                    #\n"
    "                    # El tercer argumento es el total de bloques del PROMPT, no\n"
    "                    # `num_blocks`: ese ultimo vale num_offloadable_tokens // bs y\n"
    "                    # CRECE en cada paso del prefill fragmentado, con lo cual la\n"
    "                    # regla 'conservar el ultimo' disparaba una vez por PASO y el\n"
    "                    # stride efectivo colapsaba a ~1,33 con chunk de 2 bloques.\n"
    "                    from vllm._genesis import kv_sparse_gdn as _g93\n"
    "                    if _g93.should_skip_recurrent_block(\n"
    "                        group_config.group_idx,\n"
    "                        start_block_idx + key_idx,\n"
    "                        req.num_prompt_tokens // group_config.offloaded_block_size,\n"
    "                    ):\n"
    "                        continue\n"
    "                    new_offload_keys.append(offload_key)\n"
)


# ─────────── 3. costo: tokens de hit perdidos por el redondeo ───────────
#
# PN93 publicaba sólo el lado del ahorro. Cuando un grupo recurrente trunca el
# prefijo, la diferencia son tokens que se van a RECOMPUTAR: ése es el precio
# del stride y sin medirlo no se puede calibrar.

COST_OLD = (
    "                if num_hit_blocks is None:\n"
    "                    defer_lookup = True\n"
    "                else:\n"
    "                    max_hit_size_tokens = min(\n"
    "                        max_hit_size_tokens,\n"
    "                        offloaded_block_size * (start_block_idx + num_hit_blocks),\n"
    "                    )\n"
)

COST_NEW = (
    "                if num_hit_blocks is None:\n"
    "                    defer_lookup = True\n"
    "                else:\n"
    "                    # " + GENESIS_PN93_MARKER + "\n"
    "                    _g93_before = max_hit_size_tokens\n"
    "                    max_hit_size_tokens = min(\n"
    "                        max_hit_size_tokens,\n"
    "                        offloaded_block_size * (start_block_idx + num_hit_blocks),\n"
    "                    )\n"
    "                    if max_hit_size_tokens < _g93_before:\n"
    "                        from vllm._genesis import kv_sparse_gdn as _g93\n"
    "                        _g93.note_hit_truncated(\n"
    "                            group_idx, _g93_before - max_hit_size_tokens\n"
    "                        )\n"
)


def _scheduler_patcher() -> TextPatcher | None:
    root = vllm_install_root()
    if root is None:
        return None
    target = os.path.join(
        root,
        "distributed",
        "kv_transfer",
        "kv_connector",
        "v1",
        "offloading",
        "scheduler.py",
    )
    if not os.path.exists(target):
        return None
    return TextPatcher(
        patch_name="PN93 sparse GDN checkpointing (offloading scheduler)",
        target_file=target,
        marker=GENESIS_PN93_MARKER,
        sub_patches=[
            TextPatch(
                name="pn93_register_group_spec",
                anchor=REGISTER_OLD,
                replacement=REGISTER_NEW,
                required=True,
            ),
            TextPatch(
                name="pn93_skip_intermediate_recurrent_blocks",
                anchor=FILTER_OLD,
                replacement=FILTER_NEW,
                required=True,
            ),
            TextPatch(
                name="pn93_note_hit_truncation_cost",
                anchor=COST_OLD,
                replacement=COST_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN93")
    log_decision("PN93", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    p = _scheduler_patcher()
    if p is None:
        return "skipped", "target de _scheduler_patcher no encontrado"

    result, failure = p.apply()
    status, msg = result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN93 aplicado: checkpointing disperso de estado recurrente "
            "(se guarda 1 de cada N fronteras de bloque en los grupos GDN/Mamba; "
            "el hit redondea hacia abajo al checkpoint anterior)."
        ),
        patch_name=p.patch_name,
    )
    # Devolver el status REAL: un `skipped` por deriva de ancla no se reporta
    # como `applied`.
    return status, msg


def is_applied() -> bool:
    p = _scheduler_patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN93_MARKER in f.read()
    except Exception:
        return False
