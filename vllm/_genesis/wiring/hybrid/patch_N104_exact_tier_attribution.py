# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN104 — atribucion EXACTA de tier por acierto.

El dashboard clasificaba el tier adivinando por latencia y daba cualquier
cosa: mostraba L3 = 99,1% de los aciertos y L1 = 0, con 75,1 GB leidos de un
SSD cuyo directorio tenia 12 KB, mientras el engine reportaba 1.389.440
aciertos de prefix cache de GPU y CERO de disco.

El engine si sabe la respuesta: `TieringOffloadingManager.lookup()` distingue
si el bloque lo sirvio el tier primario (RAM) o hubo que promoverlo de un
secundario (disco). PN104 lleva la cuenta por request y, cuando el scheduler
decide cuantos tokens acepta del prefijo, los reparte en esa proporcion y los
publica en `kv_tier_hit_tokens_total{tier}`.

Tres hooks:

  1. `tiering/manager.py`, rama de hit del primario.
  2. `tiering/manager.py`, rama de hit de un secundario.
  3. `offloading/scheduler.py`, cuando `_lookup` devuelve `num_hit_tokens`.

Los dos primeros anclan sobre el resultado de PN88, que ya toca esas lineas,
asi que PN104 se aplica DESPUES.

Los tokens de L1 no salen de aca: son los del prefix cache de la GPU, que vLLM
ya publica como `prefix_cache_hits_total` menos
`external_prefix_cache_hits_total`.

Observacion pura. Kill switch: `GENESIS_DISABLE_PN104=1`.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
)

GENESIS_PN104_MARKER = "_GENESIS_PN104_EXACT_TIER_ATTRIBUTION"

PRIM_OLD = "        if primary_hit is True:\n            return True\n"

PRIM_NEW = (
    "        if primary_hit is True:\n"
    "            # " + GENESIS_PN104_MARKER + "\n"
    "            from vllm._genesis import kv_tier_attribution as _g104\n"
    "\n"
    "            _g104.anotar(req_context, 'ram')\n"
    "            return True\n"
)

SEC_OLD = "                return None  # promotion started, retry later\n"

SEC_NEW = (
    "                # " + GENESIS_PN104_MARKER + "\n"
    "                # El bloque estaba en un secundario: se le atribuye a ese\n"
    "                # tier aunque la lectura efectiva ocurra despues de la\n"
    "                # promocion, porque es el tier que lo TENIA.\n"
    "                from vllm._genesis import kv_tier_attribution as _g104\n"
    "\n"
    "                _g104.anotar(\n"
    "                    req_context, getattr(tier, 'tier_type', 'secondary')\n"
    "                )\n"
    "                return None  # promotion started, retry later\n"
)

SCHED_OLD = "        num_hit_tokens = self._lookup(req_status)\n"

SCHED_NEW = (
    "        num_hit_tokens = self._lookup(req_status)\n"
    "        # " + GENESIS_PN104_MARKER + "\n"
    "        # Aca se sabe cuantos tokens del prefijo se aceptaron; se reparten\n"
    "        # entre los tiers que respondieron y se publican. El reparto es\n"
    "        # proporcional porque el hit es un PREFIJO, no una seleccion de\n"
    "        # bloques sueltos: es la atribucion correcta para ese prefijo y es\n"
    "        # exacta a nivel agregado.\n"
    "        try:\n"
    "            from vllm._genesis import kv_tier_attribution as _g104\n"
    "\n"
    "            _g104.publicar_y_limpiar(req_status.req_context, num_hit_tokens)\n"
    "        except Exception:  # observacion pura\n"
    "            pass\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN104", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _patchers() -> list[TextPatcher] | None:
    root = vllm_install_root()
    if root is None:
        return None
    tie = os.path.join(root, "v1", "kv_offload", "tiering", "manager.py")
    sch = os.path.join(
        root, "distributed", "kv_transfer", "kv_connector", "v1",
        "offloading", "scheduler.py",
    )
    if not (os.path.exists(tie) and os.path.exists(sch)):
        return None
    return [
        TextPatcher(
            patch_name="PN104 exact tier attribution (tiering manager)",
            target_file=tie,
            marker=GENESIS_PN104_MARKER,
            sub_patches=[
                TextPatch(name="pn104_primary_hit", anchor=PRIM_OLD,
                          replacement=PRIM_NEW, required=True),
                TextPatch(name="pn104_secondary_hit", anchor=SEC_OLD,
                          replacement=SEC_NEW, required=True),
            ],
        ),
        TextPatcher(
            patch_name="PN104 exact tier attribution (scheduler)",
            target_file=sch,
            marker=GENESIS_PN104_MARKER,
            sub_patches=[
                TextPatch(name="pn104_publish_split", anchor=SCHED_OLD,
                          replacement=SCHED_NEW, required=True),
            ],
        ),
    ]


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN104")
    log_decision("PN104", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN104 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patchers = _patchers()
    if patchers is None:
        return "skipped", "targets no encontrados"
    txn = MultiFilePatchTransaction(patchers, name="PN104")
    status, reason = txn.apply_or_skip()
    if status != "applied":
        return status, reason
    return "applied", (
        "PN104 aplicado: se publica kv_tier_hit_tokens_total{tier} con la "
        "atribucion EXACTA de que tier sirvio cada acierto, contada en el "
        "engine en vez de adivinada por latencia en el dashboard."
    )


def is_applied() -> bool:
    patchers = _patchers()
    if patchers is None:
        return False
    for p in patchers:
        try:
            with open(p.target_file) as f:
                if GENESIS_PN104_MARKER not in f.read():
                    return False
        except Exception:
            return False
    return True
