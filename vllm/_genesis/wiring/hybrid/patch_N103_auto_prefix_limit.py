# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN103 — `max_offload_tokens` automatico, medido por PN101.

PN101 mide el prefijo que cada agente comparte de verdad. Teniendo ese numero,
configurar `max_offload_tokens` a mano deja de tener sentido: es un valor que
hay que descubrir, mantener, y que se desactualiza solo cuando cambian las
herramientas o el prompt del agente.

Lo medido el 2026-08-23 muestra por que la estimacion manual falla:

    agente          estimado a mano   MEDIDO
    coder                     2.496    9.984   (4x corto)
    explorer                    832    4.992   (6x corto)

El estimado salia de tokenizar el prompt del agente del archivo de config; lo
que faltaba es todo lo que el cliente pone alrededor (esquemas de
herramientas, framing). Eso no se puede saber leyendo la config.

Engancha en `RequestOffloadState.__post_init__`, justo despues de que vLLM
parsea el valor del cliente, y solo actua si el cliente NO mando ninguno: un
valor explicito siempre gana.

Arranca SIN limite y lo pone recien cuando tiene observaciones. Eso es a
proposito: limitar antes de saber seria profecia autocumplida — nunca se veria
compartir mas de lo que se dejo guardar.

Kill switch: `GENESIS_DISABLE_PN103=1`. Tambien
`GENESIS_ENABLE_PN103_AUTO_PREFIX=0` apaga solo el comportamiento.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

GENESIS_PN103_MARKER = "_GENESIS_PN103_AUTO_PREFIX_LIMIT"

ANCHOR_OLD = (
    "        elif raw is not None:\n"
    "            logger.warning(\n"
    '                "max_offload_tokens must be a non-negative int, got %r; '
    'ignoring", raw\n'
    "            )\n"
    "\n"
    "    def update_offload_keys(self) -> None:\n"
)

ANCHOR_NEW = (
    "        elif raw is not None:\n"
    "            logger.warning(\n"
    '                "max_offload_tokens must be a non-negative int, got %r; '
    'ignoring", raw\n'
    "            )\n"
    "        # " + GENESIS_PN103_MARKER + "\n"
    "        # Si el cliente no mando un limite, se usa el que PN101 MIDIO para\n"
    "        # este agente. Un valor explicito siempre gana. Sin observaciones\n"
    "        # suficientes se deja sin limite, que es lo que permite aprender\n"
    "        # cual es el prefijo: limitar antes de saber seria profecia\n"
    "        # autocumplida.\n"
    "        if self.max_offload_tokens is None:\n"
    "            try:\n"
    "                from vllm._genesis import kv_prefix_auto as _g103\n"
    "\n"
    "                _g103_a = (params or {}).get('genesis_agent')\n"
    "                _g103_v = _g103.limite_para(\n"
    "                    _g103_a,\n"
    "                    self.config.kv_group_configs[0].offloaded_block_size,\n"
    "                )\n"
    "                if _g103_v:\n"
    "                    self.max_offload_tokens = _g103_v\n"
    "            except Exception:  # nunca puede romper una request\n"
    "                pass\n"
    "\n"
    "    def update_offload_keys(self) -> None:\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN103", "").strip().lower() in (
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
        patch_name="PN103 auto prefix limit",
        target_file=t,
        marker=GENESIS_PN103_MARKER,
        sub_patches=[
            TextPatch(
                name="pn103_auto_max_offload_tokens",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN103")
    log_decision("PN103", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN103 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    p = _patcher()
    if p is None:
        return "skipped", "offloading/scheduler.py no encontrado"
    result, failure = p.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN103 aplicado: max_offload_tokens se deriva de lo que PN101 midio "
            "para cada agente (maximo observado + margen), en vez de "
            "configurarse a mano. Un valor explicito del cliente siempre gana, y "
            "sin observaciones suficientes no se pone limite para poder "
            "aprender. Kill switch: GENESIS_DISABLE_PN103=1."
        ),
        patch_name="PN103 auto prefix limit",
    )


def is_applied() -> bool:
    p = _patcher()
    if p is None:
        return False
    try:
        with open(p.target_file) as f:
            return GENESIS_PN103_MARKER in f.read()
    except Exception:
        return False
