# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN108 — FP8 lm_head del DRAFT MTP (extensión del mecanismo PN77).

Genesis-original 2026-08-24 (checkpoint CK-1.1 del plan de optimización).

El problema
-----------
El drafter del MTP (`runner.drafter.model`, clase Qwen3_5MTP) pesa
~2,78 GB/rank en FP16 de los cuales ~1,27 GB son su lm_head propio
(vocab 248320 × hidden 5120 / TP). Ese lm_head carga por el camino
específico del drafter (`SpecDecodeBaseProposer.load_model` →
`_get_model()`), que NO pasa por el walker de
`process_weights_after_loading` que PN77 text-patcheó — por eso el
swap FP8 de PN77 cubre el lm_head del TARGET pero nunca llegó al del
draft (verificado empíricamente con sonda de árbol: dtype=fp16,
qm=UnquantizedEmbeddingMethod, boot 2026-08-24).

La solución
-----------
Rebind de `SpecDecodeBaseProposer.load_model` (clase base compartida
por EagleProposer/Step3p5MTPProposer): tras el original, localizar el
lm_head del drafter y aplicar el MISMO método de PN77
(`Genesis_FP8_LMHead_EmbeddingMethod` + su `process_weights_after_loading`,
idempotente por marker). Sin duplicar lógica de compresión.

Seguridad de calidad
--------------------
El draft sólo PROPONE: el rejection sampler verifica contra el target,
así que cualquier error introducido por la precisión del lm_head del
draft se traduce en tasa de aceptancia, NUNCA en calidad de salida.
Único riesgo medible: caída de aceptancia (gate del plan: <5% relativo).

Seguridad de implementación
---------------------------
- Env-gated: GENESIS_ENABLE_PN108_DRAFT_FP8_LM_HEAD=1 (default OFF).
- Sólo swap si el método actual es UnquantizedEmbeddingMethod (no
  pisa configs de cuantización reales).
- Idempotente (marker de clase + marker PN77_APPLIED del método).
- Nunca lanza hacia el runner.

Author: ox-alpha 2026-08-24 (checkpoint CK-1.1).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn108_draft_fp8_lm_head")

ENV_FLAG = "GENESIS_ENABLE_PN108_DRAFT_FP8_LM_HEAD"
_MARKER_ATTR = "_genesis_pn108_installed"

_TRUTHY = ("1", "true", "yes", "on")

_ORIGINAL = None
_INSTALLED_CLASS = None


def _find_lm_head(model):
    # El drafter Qwen3_5MTP expone .lm_head directamente (qwen3_5_mtp.py);
    # el scan por nombre de clase queda como fallback para otras arquitecturas.
    lh = getattr(model, "lm_head", None)
    if lh is not None:
        return lh
    try:
        for _name, mod in model.named_modules():
            if "LMHead" in type(mod).__name__:
                return mod
    except Exception:
        pass
    return None


def _quantize_draft_lm_head(proposer) -> tuple[str, str]:
    from vllm._genesis.kernels_legacy.lm_head_fp8_method import (
        Genesis_FP8_LMHead_EmbeddingMethod,)
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,)

    model = getattr(proposer, "model", None)
    if model is None:
        return "skipped", "drafter sin modelo"
    lm_head = _find_lm_head(model)
    if lm_head is None:
        return "skipped", "drafter sin lm_head propio"
    qm = getattr(lm_head, "quant_method", None)
    if isinstance(qm, Genesis_FP8_LMHead_EmbeddingMethod):
        return "skipped", "ya cuantizado (idempotente)"
    if qm is not None and not isinstance(qm, UnquantizedEmbeddingMethod):
        return "skipped", "quant config real presente — no override"

    new_method = Genesis_FP8_LMHead_EmbeddingMethod()
    lm_head.quant_method = new_method
    new_method.process_weights_after_loading(lm_head)
    w = getattr(lm_head, "weight", None)
    return "applied", (
        f"lm_head del drafter comprimido a {getattr(w, 'dtype', '?')}")


def install(proposer_class) -> bool:
    """Rebind de load_model sobre la CLASE del proposer."""
    global _ORIGINAL, _INSTALLED_CLASS
    if getattr(proposer_class, _MARKER_ATTR, False):
        return True
    original = proposer_class.load_model

    def load_model(self, target_model):
        original(self, target_model)
        try:
            status, reason = _quantize_draft_lm_head(self)
            log.info("PN108 %s: %s", status, reason)
        except Exception as e:  # jamás romper el arranque del drafter
            log.warning("PN108 falló (%s) — drafter queda en FP16",
                        type(e).__name__)

    proposer_class.load_model = load_model
    setattr(proposer_class, _MARKER_ATTR, True)
    _ORIGINAL = original
    _INSTALLED_CLASS = proposer_class
    log.info("PN106-style instalado sobre %s", proposer_class.__name__)
    return True


def revert() -> bool:
    global _ORIGINAL, _INSTALLED_CLASS
    if _INSTALLED_CLASS is None or _ORIGINAL is None:
        return False
    _INSTALLED_CLASS.load_model = _ORIGINAL
    setattr(_INSTALLED_CLASS, _MARKER_ATTR, False)
    _INSTALLED_CLASS = None
    _ORIGINAL = None
    return True


def apply():
    """Punto de entrada del orquestador. Nunca lanza."""
    if os.environ.get(ENV_FLAG, "").lower() not in _TRUTHY:
        return "skipped", (
            "opt-in only — set GENESIS_ENABLE_PN108_DRAFT_FP8_LM_HEAD=1")
    try:
        from vllm.v1.spec_decode.llm_base_proposer import (
            SpecDecodeBaseProposer as Base,)
    except Exception as e:
        return "failed", f"import SpecDecodeBaseProposer: {e}"
    try:
        # Instalar en la base y también en las subclases que definan su
        # propio load_model (hoy ninguna, pero el rebind base no las cubre
        # si upstream agrega overrides).
        installed = []
        if install(Base):
            installed.append(Base.__name__)
        try:
            from vllm.v1.spec_decode.eagle import EagleProposer
            if "load_model" in vars(EagleProposer):
                install(EagleProposer)
                installed.append(EagleProposer.__name__)
        except Exception:
            pass
        return "applied", f"rebind de load_model en {installed}"
    except Exception as e:
        return "failed", f"install: {e}"
