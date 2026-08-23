# SPDX-License-Identifier: Apache-2.0
"""Wiring de Genesis PN102 — desalojo por score de bloque, con envejecimiento.

ARC ya tiene frecuencia (T1 -> T2 al segundo uso) y ya prefiere victimas de
T1. Le faltan dos cosas y esto se las agrega SIN reemplazarlo:

  1. **Envejecimiento**: T2 es permanente, asi que el preambulo de un agente
     que no se usa hace horas sigue tan protegido como el del que se esta
     martillando ahora.
  2. **Granularidad**: T1/T2 es de 1 bit, no distingue 2 usos de 200.

El score es "en cuantos requests distintos aparecio el bloque", y tiene una
propiedad que evita todo el andamiaje: **ya es monotono decreciente a lo largo
del prefijo**, porque el hash del bloque 3 incluye al bloque 2. Ordenando por
score ascendente, el desalojo se va solo a las COLAS y nunca parte un prefijo
por el medio — que es el peor error posible, porque el lookup corta en el
primer miss y lo que queda detras pasa a valer cero ocupando lugar.

Dos hooks:

  1. `cpu/manager.py`, `lookup()`: anota la aparicion.
  2. `cpu/policies/arc.py`, `evict()`: elige la victima de menor score entre
     las primeras N elegibles, en vez de la primera que aparece.

Lo de "primeras N" (default 64) es a proposito: ordenar las 790 candidatas en
el camino caliente del scheduler costaria mas de lo que ahorra, y las listas de
ARC ya vienen ordenadas por recencia, asi que mirar el frente captura casi todo
el beneficio a costo fijo.

**OFF por defecto**: `GENESIS_ENABLE_PN102_BLOCK_SCORE=1` para activarlo. Con
el flag apagado `elegir()` devuelve la primera elegible, que es exactamente el
comportamiento actual de vLLM.

Kill switch: `GENESIS_DISABLE_PN102=1`.
"""

from __future__ import annotations

import os

from vllm._genesis.guards import vllm_install_root
from vllm._genesis.wiring.text_patch import (
    MultiFilePatchTransaction,
    TextPatch,
    TextPatcher,
)

GENESIS_PN102_MARKER = "_GENESIS_PN102_BLOCK_SCORE"

# ─────────── 1. anotar la aparicion ───────────

LOOKUP_OLD = (
    "                self.counts[key] = 1\n"
    "        block = self._policy.get(key)\n"
)

LOOKUP_NEW = (
    "                self.counts[key] = 1\n"
    "        # " + GENESIS_PN102_MARKER + "\n"
    "        # Score aparte de `counts` a proposito: prepare_store filtra por\n"
    "        # counts >= store_threshold, y _maximal_prefix_lookup corta en el\n"
    "        # primer miss, asi que los bloques NUEVOS nunca se miran. Si el\n"
    "        # score fuera counts, ese filtro los descartaria y no se guardaria\n"
    "        # nada.\n"
    "        from vllm._genesis import kv_block_score as _g102\n"
    "\n"
    "        _g102.anotar(self, key)\n"
    "        block = self._policy.get(key)\n"
)

# ─────────── 2. elegir victima por score ───────────

T1_OLD = (
    "            if virtual_t1_size >= int(self.target_t1_size):\n"
    "                for key, block in self.t1.items():\n"
    "                    if (\n"
    "                        block.ref_cnt == 0\n"
    "                        and key not in protected\n"
    "                        and key not in already_selected\n"
    "                    ):\n"
    "                        candidate = (key, block, True)\n"
    "                        virtual_t1_size -= 1\n"
    "                        break\n"
)

T1_NEW = (
    "            if virtual_t1_size >= int(self.target_t1_size):\n"
    "                # " + GENESIS_PN102_MARKER + "\n"
    "                from vllm._genesis import kv_block_score as _g102\n"
    "\n"
    "                _g102_c = _g102.elegir(\n"
    "                    self.t1.items(),\n"
    "                    protected,\n"
    "                    already_selected,\n"
    "                    getattr(self, '_g102_scores', None),\n"
    "                )\n"
    "                if _g102_c is not None:\n"
    "                    candidate = (_g102_c[0], _g102_c[1], True)\n"
    "                    virtual_t1_size -= 1\n"
)

T2_OLD = (
    "            if candidate is None:\n"
    "                for key, block in self.t2.items():\n"
    "                    if (\n"
    "                        block.ref_cnt == 0\n"
    "                        and key not in protected\n"
    "                        and key not in already_selected\n"
    "                    ):\n"
    "                        candidate = (key, block, False)\n"
    "                        break\n"
    "                if candidate is None:\n"
    "                    return None\n"
)

T2_NEW = (
    "            if candidate is None:\n"
    "                # " + GENESIS_PN102_MARKER + "\n"
    "                from vllm._genesis import kv_block_score as _g102\n"
    "\n"
    "                _g102_c = _g102.elegir(\n"
    "                    self.t2.items(),\n"
    "                    protected,\n"
    "                    already_selected,\n"
    "                    getattr(self, '_g102_scores', None),\n"
    "                )\n"
    "                if _g102_c is not None:\n"
    "                    candidate = (_g102_c[0], _g102_c[1], False)\n"
    "                if candidate is None:\n"
    "                    return None\n"
)


def _is_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN102", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _patchers() -> list[TextPatcher] | None:
    root = vllm_install_root()
    if root is None:
        return None
    mgr = os.path.join(root, "v1", "kv_offload", "cpu", "manager.py")
    arc = os.path.join(root, "v1", "kv_offload", "cpu", "policies", "arc.py")
    if not (os.path.exists(mgr) and os.path.exists(arc)):
        return None
    return [
        TextPatcher(
            patch_name="PN102 block score (cpu manager)",
            target_file=mgr,
            marker=GENESIS_PN102_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn102_score_on_lookup",
                    anchor=LOOKUP_OLD,
                    replacement=LOOKUP_NEW,
                    required=True,
                ),
            ],
        ),
        TextPatcher(
            patch_name="PN102 block score (arc policy)",
            target_file=arc,
            marker=GENESIS_PN102_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn102_t1_victim_by_score",
                    anchor=T1_OLD,
                    replacement=T1_NEW,
                    required=True,
                ),
                TextPatch(
                    name="pn102_t2_victim_by_score",
                    anchor=T2_OLD,
                    replacement=T2_NEW,
                    required=True,
                ),
            ],
        ),
    ]


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN102")
    log_decision("PN102", decision, reason)
    if not decision:
        return "skipped", reason
    if _is_disabled():
        return "skipped", "GENESIS_DISABLE_PN102 set"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patchers = _patchers()
    if patchers is None:
        return "skipped", "targets de cpu/manager.py o policies/arc.py no hallados"

    txn = MultiFilePatchTransaction(patchers, name="PN102")
    status, reason = txn.apply_or_skip()
    if status != "applied":
        return status, reason
    return "applied", (
        "PN102 aplicado: la victima de desalojo se elige por score de bloque "
        "(en cuantos requests distintos aparecio) entre las primeras "
        "GENESIS_PN102_SCAN candidatas, con halving cada "
        "GENESIS_PN102_AGING_EVERY observaciones. El score es monotono a lo "
        "largo del prefijo, asi que el desalojo se va a las colas y no parte "
        "prefijos por el medio. Se activa con "
        "GENESIS_ENABLE_PN102_BLOCK_SCORE=1; apagado el comportamiento es "
        "identico al de vLLM. Kill switch: GENESIS_DISABLE_PN102=1."
    )


def is_applied() -> bool:
    patchers = _patchers()
    if patchers is None:
        return False
    for p in patchers:
        try:
            with open(p.target_file) as f:
                if GENESIS_PN102_MARKER not in f.read():
                    return False
        except Exception:
            return False
    return True
