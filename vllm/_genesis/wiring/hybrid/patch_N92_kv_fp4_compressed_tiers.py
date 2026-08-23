# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN92 — cuantización agrupada a 4 bits del tier de disco.

Dos anclas en `v1/kv_offload/tiering/fs/io.py`:

  `store_block` — sustituye la vista cruda por el bloque comprimido. El buffer
      comprimido viene de un mmap anónimo y con longitud múltiplo de 4096, que
      es lo que `O_DIRECT` exige (la versión anterior escribía un `bytes` de
      longitud arbitraria: `os.write` fallaba con EINVAL en el 100% de los
      casos, verificado sobre el NVMe real).

  `load_block` — detecta el magic ANTES de abrir con `O_DIRECT`. La detección
      NO depende de la env var: si dependiera, apagar la compresión haría que
      cada archivo comprimido se leyera como KV crudo, diera short read, y
      `load_block` lo borrara. Apagar la env destruía el caché entero.

El parche se aplica siempre; lo que decide si se comprime es
`GENESIS_ENABLE_PN92_KV_FP4_COMPRESSION` (default **0**), en tiempo de
ejecución. Así la lectura de archivos ya comprimidos sigue funcionando aunque
la compresión esté apagada.
"""

from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN92_MARKER = "[Genesis PN92: KV 4-bit grouped compressed disk tier]"

STORE_BLOCK_HOOK_OLD = (
    "    # Write block atomically. Cast to a flat byte view so the slice uses byte\n"
    "    # indices; the raw memoryview may be multi-dimensional with itemsize > 1.\n"
    '    view_slice = buffer.cast("B")[offset : offset + block_size]\n'
)

STORE_BLOCK_HOOK_NEW = (
    "    # Write block atomically. Cast to a flat byte view so the slice uses byte\n"
    "    # indices; the raw memoryview may be multi-dimensional with itemsize > 1.\n"
    '    view_slice = buffer.cast("B")[offset : offset + block_size]\n'
    "    # " + GENESIS_PN92_MARKER + "\n"
    "    # compress_block devuelve None cuando no conviene o no se puede comprimir;\n"
    "    # en ese caso se escribe el bloque crudo, sin cambios.\n"
    "    from vllm._genesis import kv_fp4_codec as _g92\n"
    "    if _g92.is_fp4_compression_enabled():\n"
    "        _compressed = _g92.compress_block(view_slice)\n"
    "        if _compressed is not None:\n"
    "            view_slice = _compressed\n"
)

LOAD_BLOCK_HOOK_OLD = (
    "    fd: int | None = None\n"
    '    view_slice = view.cast("B")[offset : offset + block_size]\n'
)

LOAD_BLOCK_HOOK_NEW = (
    "    # " + GENESIS_PN92_MARKER + "\n"
    "    # La deteccion NO mira la env var a proposito: un archivo comprimido tiene\n"
    "    # que poder leerse aunque la compresion este apagada.\n"
    "    from vllm._genesis import kv_fp4_codec as _g92\n"
    "    if _g92.probe_compressed(source_path):\n"
    '        _g92.load_compressed_block(source_path, view.cast("B")[offset : offset + block_size])\n'
    "        return\n"
    "    fd: int | None = None\n"
    '    view_slice = view.cast("B")[offset : offset + block_size]\n'
)


def _get_fs_io_patcher() -> TextPatcher | None:
    path = resolve_vllm_file("v1/kv_offload/tiering/fs/io.py")
    if not path:
        return None
    return TextPatcher(
        patch_name="PN92 kv 4-bit compressed disk tier (fs io)",
        target_file=path,
        marker=GENESIS_PN92_MARKER,
        sub_patches=[
            TextPatch(
                name="pn92_fs_io_store_compress",
                anchor=STORE_BLOCK_HOOK_OLD,
                replacement=STORE_BLOCK_HOOK_NEW,
                required=True,
            ),
            TextPatch(
                name="pn92_fs_io_load_decompress",
                anchor=LOAD_BLOCK_HOOK_OLD,
                replacement=LOAD_BLOCK_HOOK_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN92")
    log_decision("PN92", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    p = _get_fs_io_patcher()
    if p is None:
        return "skipped", "target de _get_fs_io_patcher no encontrado"

    result, failure = p.apply()
    status, msg = result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN92 aplicado: codec de 4 bits agrupado cableado en el tier de disco "
            "(la compresión en sí la enciende "
            "GENESIS_ENABLE_PN92_KV_FP4_COMPRESSION, default 0)."
        ),
        patch_name=p.patch_name,
    )
    return status, msg


def is_applied() -> bool:
    p = _get_fs_io_patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN92_MARKER in f.read()
    except Exception:
        return False
