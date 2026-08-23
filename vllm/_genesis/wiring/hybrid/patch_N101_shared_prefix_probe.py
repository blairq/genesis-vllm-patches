# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN101 — publica el prefijo compartido real de cada agente.

`max_offload_tokens` se venia estimando a mano, tokenizando el prompt del
agente del archivo de config. Eso mide la cosa equivocada: lo que importa es
el prefijo que de verdad se repite entre invocaciones, que incluye lo que el
cliente ponga ANTES (framing global, esquemas de herramientas) y va en
bloques.

Y se puede medir exacto: las claves de offload SON hashes de contenido de
prefijos alineados a bloque, asi que la cantidad de claves iniciales en comun
entre dos requests del mismo `genesis_agent` **es** el prefijo compartido.

Engancha justo despues de `update_offload_keys()`, que es donde las claves de
la request quedan armadas y todavia se tiene el `req_context` con el tag del
agente.

Publica `kv_prefix_shared_{blocks,tokens}`, `..._blocks_max` y
`kv_prefix_observations`, todas con label `agent`. El pipeline de PN88 maneja
gauges por nombre generico, asi que no hace falta tocarlo.

Observacion pura: no cambia ninguna decision. Kill switch:
`GENESIS_DISABLE_PN101=1`.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN101_MARKER = "_GENESIS_PN101_SHARED_PREFIX_PROBE"

ANCHOR_OLD = (
    "        req_status.update_offload_keys()\n"
    "        req_status.num_locally_computed_tokens = num_computed_tokens\n"
)

ANCHOR_NEW = (
    "        req_status.update_offload_keys()\n"
    "        # " + GENESIS_PN101_MARKER + "\n"
    "        # Las claves son hashes de contenido alineados a bloque, asi que\n"
    "        # las iniciales en comun con la request anterior del mismo agente\n"
    "        # SON el prefijo compartido. Es el numero que hay que usar para\n"
    "        # max_offload_tokens, medido en vez de estimado.\n"
    "        try:\n"
    "            from vllm._genesis import kv_prefix_probe as _g101\n"
    "\n"
    "            _g101_p = (\n"
    "                getattr(req_status.req_context, 'kv_transfer_params', None) or {}\n"
    "            )\n"
    "            _g101_a = _g101_p.get('genesis_agent')\n"
    "            if _g101_a and req_status.group_states:\n"
    "                _g101_n = _g101.observar(\n"
    "                    _g101_a, req_status.group_states[0].offload_keys\n"
    "                )\n"
    "                if _g101_n is not None:\n"
    "                    _g101.publicar(\n"
    "                        _g101_a,\n"
    "                        _g101_n,\n"
    "                        self.config.kv_group_configs[0].offloaded_block_size,\n"
    "                    )\n"
    "        except Exception:  # observacion pura: nunca puede romper nada\n"
    "            pass\n"
    "        req_status.num_locally_computed_tokens = num_computed_tokens\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN101", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _patcher() -> TextPatcher | None:
    root = vllm_install_root()
    if root is None:
        return None
    t = os.path.join(
        root, "distributed", "kv_transfer", "kv_connector", "v1",
        "offloading", "scheduler.py",
    )
    if not os.path.exists(t):
        return None
    return TextPatcher(
        patch_name="PN101 shared prefix probe",
        target_file=t,
        marker=GENESIS_PN101_MARKER,
        sub_patches=[
            TextPatch(
                name="pn101_observe_shared_prefix",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN101")
    log_decision("PN101", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN101 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "offloading/scheduler.py no encontrado"
    result, failure = p.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN101 aplicado: se mide el prefijo REALMENTE compartido entre "
            "requests de un mismo agente contando las claves iniciales en comun "
            "(que son hashes de contenido alineados a bloque) y se publica en "
            "kv_prefix_shared_{blocks,tokens}, ..._blocks_max y "
            "kv_prefix_observations, con label agent. Es el valor que hay que "
            "usar para max_offload_tokens. Kill switch: GENESIS_DISABLE_PN101=1."
        ),
        patch_name="PN101 shared prefix probe",
    )


def is_applied() -> bool:
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN101_MARKER in f.read()
    except Exception:
        return False
