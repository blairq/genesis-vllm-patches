# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN98 — avisa el tamaño mínimo recomendado de L2 al arrancar.

El engine loguea `primary tier (arc, 222 blocks)` y ese número no se puede
comparar con nada sin hacer la cuenta a mano. PN98 la hace y la imprime, con
la regla de cualquier jerarquía de cachés: **cada nivel tiene que ser más
grande que el de arriba**.

Acá se violaba por 2,5x sin que nada lo dijera:

    configurado    : 222 bloques = 5,99 GiB = 155.540 tokens
    L1 (KV en GPU) : 390.444 tokens  -> L2/L1 = 0,40x

Con L2 más chica que L1, nada de lo que la GPU desaloja entra completo, y
toda promoción desde disco termina desalojando.

Es puramente informativo: no cambia ninguna decisión ni asignación.
Kill switch: `GENESIS_DISABLE_PN98=1`.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN98_MARKER = "_GENESIS_PN98_L2_SIZING_ADVISORY"

ANCHOR_OLD = (
    '                "Created TieringOffloadingManager with primary tier "\n'
    '                "(%s, %s blocks) and %s secondary tier(s)",\n'
    "                self.eviction_policy,\n"
    "                self.num_blocks,\n"
    "                len(secondary_tiers),\n"
    "            )\n"
)

ANCHOR_NEW = (
    '                "Created TieringOffloadingManager with primary tier "\n'
    '                "(%s, %s blocks) and %s secondary tier(s)",\n'
    "                self.eviction_policy,\n"
    "                self.num_blocks,\n"
    "                len(secondary_tiers),\n"
    "            )\n"
    "            # " + GENESIS_PN98_MARKER + "\n"
    "            # El numero de bloques solo no dice nada. Lo traducimos a\n"
    "            # tokens y lo comparamos con L1, que es la regla que importa.\n"
    "            try:\n"
    "                import os as _g98_os\n"
    "\n"
    "                from vllm._genesis import l2_sizing as _g98\n"
    "\n"
    "                _g98_grupos = len(self.kv_cache_config.kv_cache_groups)\n"
    "                _g98_rec = 0\n"
    "                for _g98_g in self.kv_cache_config.kv_cache_groups:\n"
    "                    if type(_g98_g.kv_cache_spec).__name__.startswith('Mamba'):\n"
    "                        _g98_rec += 1\n"
    "                _g98_stride = 1\n"
    "                if _g98_os.environ.get(\n"
    "                    'GENESIS_ENABLE_PN93_SPARSE_GDN', '1'\n"
    "                ) not in ('0', 'false', 'no'):\n"
    "                    _g98_stride = int(\n"
    "                        _g98_os.environ.get(\n"
    "                            'GENESIS_PN93_GDN_CHECKPOINT_STRIDE', '1'\n"
    "                        )\n"
    "                        or 1\n"
    "                    )\n"
    "                _g98_cc = self.vllm_config.cache_config\n"
    "                _g98_tpb = _g98_cc.block_size * self.block_size_factor\n"
    "                _g98_l1 = int(\n"
    "                    (getattr(_g98_cc, 'num_gpu_blocks', 0) or 0)\n"
    "                    * _g98_cc.block_size\n"
    "                )\n"
    "                for _g98_ln in _g98.informe(\n"
    "                    num_blocks=self.num_blocks,\n"
    "                    bytes_por_bloque=self.kv_bytes_per_offloaded_block,\n"
    "                    tokens_por_bloque=_g98_tpb,\n"
    "                    num_grupos=_g98_grupos,\n"
    "                    max_model_len=self.vllm_config.model_config.max_model_len,\n"
    "                    tokens_en_l1=_g98_l1,\n"
    "                    grupos_recurrentes=_g98_rec,\n"
    "                    stride=_g98_stride,\n"
    "                ):\n"
    "                    logger.info('%s', _g98_ln)\n"
    "            except Exception as _g98_exc:  # nunca puede tumbar el arranque\n"
    "                logger.debug('PN98: no pude calcular el informe: %s', _g98_exc)\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN98", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _patcher() -> TextPatcher | None:
    root = vllm_install_root()
    if root is None:
        return None
    target = os.path.join(root, "v1", "kv_offload", "tiering", "spec.py")
    if not os.path.exists(target):
        return None
    return TextPatcher(
        patch_name="PN98 L2 sizing advisory",
        target_file=target,
        marker=GENESIS_PN98_MARKER,
        sub_patches=[
            TextPatch(
                name="pn98_sizing_report",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN98")
    log_decision("PN98", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN98 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "v1/kv_offload/tiering/spec.py not found"
    result, failure = p.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN98 aplicado: al crear el TieringOffloadingManager se loguea el "
            "tamano de L2 traducido a TOKENS, comparado con L1 (KV en GPU) y "
            "con el costo de una request de max_model_len, mas el minimo "
            "recomendado. Informativo puro. Kill switch: GENESIS_DISABLE_PN98=1."
        ),
        patch_name="PN98 L2 sizing advisory",
    )


def is_applied() -> bool:
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN98_MARKER in f.read()
    except Exception:
        return False
