# SPDX-License-Identifier: Apache-2.0
"""Genesis PN93: checkpointing disperso del estado recurrente (GDN / Mamba).

Por qué
-------
En el híbrido Qwen3.8-27B los cuatro grupos de KV cache pesan lo mismo por
bloque (28.966.912 B, medido), pero no valen lo mismo:

    g0, g1, g2 → `linear_attn`      → estado recurrente GDN   (102 KB/token)
    g3         → `self_attn.attn`   → K/V de atención + MTP    (34 KB/token)

El estado recurrente es **acumulativo**: una página Mamba es un estado, no
tokens. `MambaSpec.page_size_bytes` ni siquiera depende de `block_size`. Y del
lado de la lectura vLLM ya lo sabe — `scheduler.py`::

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1                      # sliding_window_size_in_blocks

Es decir: los grupos GDN se resuelven con `_sliding_window_lookup(window=1)`,
que escanea **desde el final hacia atrás** y se queda con el ÚLTIMO bloque
presente. Todos los bloques GDN intermedios se guardan y no se leen nunca.

Qué hace este módulo
--------------------
Guarda una de cada `S` fronteras de bloque de los grupos recurrentes (más
siempre la última del prompt, que es el punto de reanudación natural del turno
siguiente). Si falta el bloque `k`, el lookup **redondea hacia abajo** al
checkpoint anterior y recomputa el resto: es correcto por construcción, no hay
riesgo de leer un estado equivocado.

Es la misma optimización que vLLM ya trae para SWA (`alignment_block_count`,
*"for DeepSeek V4 with 100K tokens this reduces SWA stores by ~78%"*), que acá
queda inerte porque `_alignment_block_count()` devuelve `None` cuando todos los
grupos comparten `block_size` (832 ≤ 832).

Economía (medida sobre la config real)
--------------------------------------
    sin PN93      136,0 KB/token  ->  46k tokens en 6 GiB de L2
    S = 4          61,6 KB/token  -> 102k
    S = 8          51,0 KB/token  -> 123k

El filtro se aplica **antes** de `manager.prepare_store`, así que el bloque no
entra ni a L2 RAM ni a L3 SSD. Eso importa más de lo que parece: `_initiate_promotion`
trunca el hit cuando la primary se llena, o sea que **la L2 es la que topea el
hit máximo**. Y con PN90 la tasa de escritura al SSD ES la tasa de desalojo de
L2, así que achicar el costo por token ataca los dos problemas a la vez. En esta
máquina (30 GB de RAM) no se puede comprar capacidad de L2: ésta es la única vía.

El costo se publica en `kv_tier_hit_truncated_tokens_total{recurrent="1"}`: son
los tokens que hay que recomputar cuando el hit redondea hacia abajo. En el
patrón agentico suele ser cero, porque el último bloque del prompt —el punto de
reanudación del turno siguiente— siempre se conserva.
"""

from __future__ import annotations

import logging
import os
import threading


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


log = _get_logger("genesis.pn93.sparse_gdn")

_DEFAULT_STRIDE = 4

_lock = threading.RLock()

# group_idx → True si el grupo guarda estado recurrente (Mamba/GDN).
_RECURRENT_GROUPS: dict[int, bool] = {}
# group_idx → bytes por página del KVCacheSpec. NO es lo que se escribe a disco:
# ver `_BLOCK_WRITE_BYTES`.
_GROUP_PAGE_BYTES: dict[int, int] = {}

# Bytes que `store_block` escribe por cada bloque salteado, o sea
# `kv_bytes_per_offloaded_block` del spec (28.966.912 en este modelo).
#
# Antes se reportaba `MambaSpec.page_size_bytes` (851.968), que es el tamaño del
# ESTADO, no el del bloque en disco: el contador iba 34x corto. El conteo de
# bloques siempre estuvo bien; el de bytes no.
_BLOCK_WRITE_BYTES: int = 0

_STATS = {
    "blocks_skipped": 0,
    "bytes_skipped": 0,
    "blocks_kept": 0,
    "hit_truncated_tokens": 0,
}


def is_sparse_gdn_enabled() -> bool:
    return os.environ.get("GENESIS_ENABLE_PN93_SPARSE_GDN", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def checkpoint_stride() -> int:
    """Cada cuántas fronteras de bloque se conserva un checkpoint recurrente.

    `1` desactiva el salteo (se guardan todos, comportamiento de upstream).
    """
    raw = os.environ.get("GENESIS_PN93_GDN_CHECKPOINT_STRIDE")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            log.warning("PN93: stride inválido %r, se usa %d", raw, _DEFAULT_STRIDE)
    return _DEFAULT_STRIDE


def _looks_recurrent(kv_cache_spec: object) -> bool:
    """¿Este KVCacheSpec guarda estado recurrente?

    Se prueba `isinstance(..., MambaSpec)` primero; si el import falla (versión
    de vLLM distinta) se cae a duck-typing sobre el nombre de la clase y el
    atributo `mamba_type`, que sólo existe en MambaSpec.
    """
    try:
        from vllm.v1.kv_cache_interface import MambaSpec

        return isinstance(kv_cache_spec, MambaSpec)
    except Exception:
        pass
    return type(kv_cache_spec).__name__ == "MambaSpec" or hasattr(
        kv_cache_spec, "mamba_type"
    )


def register_group_spec(
    group_idx: int,
    kv_cache_spec: object,
    block_write_bytes: int = 0,
) -> None:
    """Registra la naturaleza de un grupo de KV cache.

    Lo llama el código inyectado en `SchedulerOffloadConfig.from_spec`, que es
    el único sitio donde conviven el índice del grupo, su `KVCacheSpec` y el
    `OffloadingSpec` del que sale `kv_bytes_per_offloaded_block`.

    Args:
        block_write_bytes: bytes que `store_block` escribe por bloque. Es lo
            único válido para contabilizar el ahorro; `page_size_bytes` mide el
            estado recurrente, que en este modelo es 34x más chico.
    """
    global _BLOCK_WRITE_BYTES
    recurrent = _looks_recurrent(kv_cache_spec)
    page_bytes = 0
    try:
        page_bytes = int(getattr(kv_cache_spec, "page_size_bytes", 0) or 0)
    except Exception:
        page_bytes = 0
    with _lock:
        _RECURRENT_GROUPS[int(group_idx)] = recurrent
        _GROUP_PAGE_BYTES[int(group_idx)] = page_bytes
        if block_write_bytes:
            _BLOCK_WRITE_BYTES = int(block_write_bytes)
    log.info(
        "PN93: grupo %d = %s (estado %d B, bloque en disco %d B, stride %d)",
        group_idx,
        "recurrente (GDN/Mamba)" if recurrent else "atencion",
        page_bytes,
        _BLOCK_WRITE_BYTES,
        checkpoint_stride(),
    )


def block_write_bytes(group_idx: int = 0) -> int:
    """Bytes ahorrados por saltear un bloque. Cae a `page_size_bytes` si el
    spec no lo informó."""
    with _lock:
        return _BLOCK_WRITE_BYTES or _GROUP_PAGE_BYTES.get(int(group_idx), 0)


def is_recurrent_group(group_idx: int) -> bool:
    with _lock:
        return _RECURRENT_GROUPS.get(int(group_idx), False)


def should_skip_recurrent_block(
    group_idx: int,
    abs_block_idx: int,
    total_blocks: int,
) -> bool:
    """¿Saltear el guardado de este bloque por ser un estado recurrente intermedio?

    Args:
        group_idx: índice del grupo de KV cache.
        abs_block_idx: índice ABSOLUTO del bloque dentro del request. Tiene que
            ser absoluto (y no relativo a la ventana de este paso) para que dos
            requests que comparten prefijo coincidan en qué checkpoints existen.
        total_blocks: bloques offloadeables TOTALES del request — o sea
            ``num_prompt_tokens // offloaded_block_size``.

            ⚠️ NO pasar el `num_blocks` del paso actual. Ese valor crece paso a
            paso con el prefill fragmentado, así que la regla "conservar el
            último" dispararía una vez POR PASO en vez de una vez por prompt.
            Medido con la config real (prompt 40k, bloque 832, chunk 1664 = 2
            bloques/paso, stride 4): se conservaban 36 de 48 bloques, o sea un
            stride efectivo de **1,33** en lugar de 4 — el ahorro caía de 2,2x
            a 1,23x. Ver `test_stride_efectivo_bajo_prefill_fragmentado`.

    Returns:
        True si el bloque es un estado recurrente intermedio y se puede saltear.
    """
    if not is_sparse_gdn_enabled():
        return False
    if not is_recurrent_group(group_idx):
        return False

    stride = checkpoint_stride()
    if stride <= 1:
        return False

    idx = int(abs_block_idx)
    is_checkpoint = (idx % stride) == 0
    # Igualdad EXACTA, no `>=`: con `offload_prompt_only=False` los bloques de
    # decodificación caen más allá de `total_blocks` y con `>=` se conservarían
    # todos. Pasado el final del prompt manda sólo el stride.
    is_last = idx == int(total_blocks) - 1

    if is_checkpoint or is_last:
        with _lock:
            _STATS["blocks_kept"] += 1
        return False

    saved = block_write_bytes(group_idx)
    with _lock:
        _STATS["blocks_skipped"] += 1
        _STATS["bytes_skipped"] += saved
    _publish_skip(int(group_idx), saved)
    return True


def _publish_skip(group_idx: int, saved_bytes: int) -> None:
    """Publica el salteo en el sink de PN88, si está activo.

    Sin esto el ahorro de PN93 es invisible en Prometheus, y peor: los bloques
    salteados aparecen como `miss` en `kv_tier_lookups_total` y hunden el hit
    rate del disco sin que haya ningún problema de caché. La label `group` de
    esa serie y estos contadores son las dos mitades de esa lectura.
    """
    try:
        from vllm._genesis import kv_tier_metrics as _g88

        if not _g88.enabled():
            return
        labels = (("tier", "disk"), ("group", str(group_idx)))
        s = _g88.sink()
        s.inc("kv_tier_recurrent_blocks_skipped_total", labels)
        if saved_bytes:
            s.inc("kv_tier_recurrent_bytes_skipped_total", labels, saved_bytes)
    except Exception:
        pass


def note_hit_truncated(group_idx: int, tokens: int) -> None:
    """Un grupo truncó el prefijo: esos tokens se van a recomputar.

    Es el COSTO de PN93 y hasta ahora no se medía. Para los grupos recurrentes
    la causa es el redondeo al checkpoint anterior; para el de atención es un
    miss genuino, y por eso se etiqueta por grupo en vez de sumarlo todo junto.

    Si `kv_tier_hit_truncated_tokens_total` de los grupos recurrentes crece,
    bajá `GENESIS_PN93_GDN_CHECKPOINT_STRIDE`.
    """
    if tokens <= 0:
        return
    with _lock:
        _STATS["hit_truncated_tokens"] += int(tokens)
    try:
        from vllm._genesis import kv_tier_metrics as _g88

        if not _g88.enabled():
            return
        _g88.sink().inc(
            "kv_tier_hit_truncated_tokens_total",
            (
                ("group", str(group_idx)),
                ("recurrent", "1" if is_recurrent_group(group_idx) else "0"),
            ),
            int(tokens),
        )
    except Exception:
        pass


def get_sparse_gdn_stats() -> dict[str, int]:
    """Métricas reales de PN93 (los bytes salen de `page_size_bytes`, no de una
    constante hardcodeada)."""
    with _lock:
        skipped = _STATS["blocks_skipped"]
        kept = _STATS["blocks_kept"]
        total = skipped + kept
        return {
            "blocks_skipped": skipped,
            "blocks_kept": kept,
            "bytes_skipped": _STATS["bytes_skipped"],
            "mb_skipped": _STATS["bytes_skipped"] // (1024 * 1024),
            "skip_ratio_pct": (100 * skipped // total) if total else 0,
            "hit_truncated_tokens": _STATS["hit_truncated_tokens"],
            "block_write_bytes": block_write_bytes(),
            "stride": checkpoint_stride(),
            "recurrent_groups": sum(1 for v in _RECURRENT_GROUPS.values() if v),
        }


def reset_state() -> None:
    """Resetea registro y contadores. Lo usan los tests.

    Tiene que limpiar TODO el estado de módulo, incluido `_BLOCK_WRITE_BYTES`:
    olvidarlo hacía que un test filtrara el tamaño de bloque al siguiente.
    """
    global _BLOCK_WRITE_BYTES
    with _lock:
        _RECURRENT_GROUPS.clear()
        _GROUP_PAGE_BYTES.clear()
        _BLOCK_WRITE_BYTES = 0
        for k in _STATS:
            _STATS[k] = 0
