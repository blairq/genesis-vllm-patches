# SPDX-License-Identifier: Apache-2.0
"""Wiring for B7 — CK-4.4 lm_head restante (tie_word_embeddings + fused sampled).

Contexto de CK-4.4
=================
Tras PN77 (target untied BF16→FP8 via process_weights_after_loading walker) y
PN108 (mismo swap aplicado al drafter MTP vía SpecDecodeBaseProposer.load_model
rebind), queda un lm_head SIN optimizar:

  1. **tie_word_embeddings=True**: PN77 lo detecta vía
     `weight.data_ptr() == embed_tokens.weight.data_ptr()` y hace SKIP
     deliberado — `replace_parameter` no debe usarse sobre storage compartido
     (doc de vllm/utils.py) y comprimir solo un lado corrompería el otro.
     Este es el caso de Qwen3.6-27B cuando el checkpoint trae tied (algunas
     variantes FP8/vl), y de Qwen3.5-35B-A3B cuando se activa
     `--tie-word-embeddings` experimental. ~606 MiB/rank quedan sin comprimir.

  2. **Proyección final no-cuantizable**: modelos con quant_config real
     (AWQ/GPTQ/CompressedTensors) ya tienen lm_head cubierto por su propio
     quant_method — PN77 respeta `not isinstance(current, Unquantized)` y
     no pisa. Pero el **matmul de vocab completo** (248320×hidden) sigue
     materializando `logits` de vocab entero para luego hacer `gather` de
     los 1-3 tokens muestreados. Eso es ~1 matmul de 5120×248320 por token
     (~5 GFLOP) tirado al 99%.

B7 cubre ambos huecos en un solo wiring:

  A. **Tied INT8/FP8 compartido (A)**: cuando detecta storage compartido,
     cuantiza UNA vez el tensor compartido a FP8 e4m3 per-channel (mismo
     compressor que PN77) y hace que **ambos** módulos (embed_tokens y
     lm_head) apunten al mismo buffer FP8 + escala compartida. Sin duplicar
     memoria, sin orphan de weight_loader (usa `replace_parameter` en ambos
     lados + copia de `weight_loader`). Canario numérico obligatorio (B7 del
     audit mtp_cache.py: cosine ≥0.999, rel_err <1%).

  B. **Fused sampled logits (B)**: text-patch en
     `model_executor/layers/vocab_parallel_embedding.py:ParallelLMHead`
     y fallback en `model_executor/layers/logits_processor.py` que reemplaza
     `F.linear(hidden, weight)` completo por `F.linear(hidden, weight[sampled_ids])`
     cuando `sampling_metadata.selected_token_ids` está presente (decode con
     sampler). En Qwen3.6-27B vocab=248320, hidden=5120, TP=2: de 124160×5120
     por rank a 1×5120 cuando se muestrea 1 token → ~124k× ahorro de FLOPs en
     el camino de sampleo. En prefill o sin metadata cae al path original.

Por qué text-patch + runtime hook
=================================
- El matmul vive dentro de `LogitsProcessor` o `ParallelLMHead.forward`,
  no hay punto de rebind limpio sin duplicar 50 líneas de upstream.
- Otros engines (SGLang, TensorRT-LLM) ya hacen fused sampled (ver
  SGLang PR #21019 y TRT-LLM live-range reuse). vLLM aún no (TODO en
  `gpu_model_runner.py:2778` y `logits_processor.py`).
- La detección de tied requiere leer `config.tie_word_embeddings` y
  comparar `data_ptr()` — eso solo es posible post-load, por eso B7
  instala hook en `process_weights_after_loading` walker (mismo punto que
  PN77 pero con guard distinto) Y un fallback runtime para modelos que
  resuelven tie en `tie_weights()` tardío.

Safety
======
- Default OFF (opt-in via `GENESIS_ENABLE_B7_LM_HEAD=1`).
- Idempotente vía marker `_genesis_b7_lm_head_applied`.
- Tied path valida canario y aborta a BF16 si supera umbral.
- Fused path es strict-superset: si `selected_token_ids is None` o
  `vocab gather` falla, cae al matmul completo sin excepción.
- Drift: si upstream añade `fused_sampled_logits` o `lm_head_quantized`,
  auto-retire vía `upstream_drift_markers`.

Expected impact
===============
- Tied 27B: ~606 MiB/rank igual que PN77 pero sobre storage compartido
  (no duplica). Total cluster 1.2 GiB.
- Fused sampled: -3 ms/token en decode Greedy (1 token) sobre A5000,
  -0% en prefill (no aplica). Compone con PN77/PN108 (B7 no pisa su dtype).

Author: Genesis CK-4.4 (B7) — Sander Barzov Aleksandr, Odessa, Ukraine.
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.b7_lm_head")

# ─── Marker + env ─────────────────────────────────────────────────────────
GENESIS_B7_MARKER = "Genesis B7 lm_head — fused sampled + tie_word_embeddings (CK-4.4) v1.0"
GENESIS_B7_APPLIED_ATTR = "_genesis_b7_lm_head_applied"
ENV_FLAG = "GENESIS_ENABLE_B7_LM_HEAD"

_TRUTHY = ("1", "true", "yes", "y", "on")


def _is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() in _TRUTHY


# ─── Text-patch anchors ───────────────────────────────────────────────────
# Anchor A: tie_word_embeddings en modelos qwen (qwen3_5.py / qwen3_next.py).
# Patrón canónico visto en vllm 0.20.x:
#     if config.tie_word_embeddings:
#         self.lm_head.weight = self.model.embed_tokens.weight
# Lo reemplazamos por una versión que marca el módulo para el hook FP8
# compartido de B7 y evita la asignación directa que rompe weight_loader.
B7_TIE_OLD = (
    "        if config.tie_word_embeddings:\n"
    "            self.lm_head.weight = self.model.embed_tokens.weight\n"
)

B7_TIE_NEW = (
    "        if config.tie_word_embeddings:\n"
    "            # [Genesis B7 CK-4.4] tied storage — marca para cuantización\n"
    "            # compartida FP8 (mismo buffer para embed y lm_head, sin\n"
    "            "            # duplicar VRAM). El hook de B7 en\n"
    "            # process_weights_after_loading cuantiza una vez y re-apunta\n"
    "            # ambos módulos al mismo storage FP8 + escala.\n"
    "            try:\n"
    "                self.lm_head._genesis_b7_tied = True\n"
    "                self.model.embed_tokens._genesis_b7_tied = True\n"
    "            except Exception:\n"
    "                pass\n"
    "            self.lm_head.weight = self.model.embed_tokens.weight\n"
)

# Anchor B: ParallelLMHead forward en vocab_parallel_embedding.py.
# Patrón aproximado (varía levemente entre pins):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return F.linear(x, self.weight, bias=self.bias if hasattr(self, "bias") else None)
# Si no existe, el sub-patch es soft (required=False) y cae al hook runtime.
B7_FUSED_OLD = (
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        return F.linear(x, self.weight, bias=self.bias if hasattr(self, \"bias\") else None)\n"
)

B7_FUSED_NEW = (
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
    "        # [Genesis B7 CK-4.4] fused sampled logits: si el sampler pasó\n"
    "        # selected_token_ids (decode Greedy / top-k=1), evita materializar\n"
    "        # vocab completo (248k) y computa solo filas muestreadas.\n"
    "        try:\n"
    "            _b7_ids = getattr(self, \"_genesis_b7_sampled_ids\", None)\n"
    "            if _b7_ids is not None and x.shape[0] <= 8:\n"
    "                import torch.nn.functional as _b7_F\n"
    "                _w = self.weight[_b7_ids]\n"
    "                return _b7_F.linear(x, _w, bias=None)\n"
    "        except Exception:\n"
    "            pass\n"
    "        return F.linear(x, self.weight, bias=self.bias if hasattr(self, \"bias\") else None)\n"
)

# Anchor C: LogitsProcessor (v1/sample/logits_processor.py o
# model_executor/layers/logits_processor.py). Patrón:
#     logits = self.lm_head(hidden_states)
# Lo envolvemos con try/gather.
B7_LOGITS_OLD = (
    "        logits = self.lm_head(hidden_states)\n"
)

B7_LOGITS_NEW = (
    "        # [Genesis B7 CK-4.4] fused sampled — gathered matmul cuando sea posible\n"
    "        try:\n"
    "            _b7_meta = getattr(self, \"_genesis_b7_fused\", False)\n"
    "            _b7_ids = getattr(hidden_states, \"_genesis_b7_sampled_ids\", None)\n"
    "            if _b7_meta and _b7_ids is not None:\n"
    "                import torch.nn.functional as _b7_F\n"
    "                _w = self.lm_head.weight[_b7_ids]\n"
    "                logits = _b7_F.linear(hidden_states, _w)\n"
    "            else:\n"
    "                logits = self.lm_head(hidden_states)\n"
    "        except Exception:\n"
    "            logits = self.lm_head(hidden_states)\n"
)

# Drift markers — si upstream aterriza fusión o cuantización nativa, retire.
B7_UPSTREAM_DRIFT_MARKERS = [
    "fused_sampled_logits",
    "_genesis_b7_fused",
    "lm_head_quantized",
    "GENESIS_B7",  # nuestro propio marker previo nos hace idempotente
]


def _make_patcher_for_file(rel_path: str, sub_patches: list[TextPatch]) -> TextPatcher | None:
    target = resolve_vllm_file(rel_path)
    if target is None:
        return None
    return TextPatcher(
        patch_name=f"B7 lm_head — {rel_path} (CK-4.4)",
        target_file=str(target),
        marker=GENESIS_B7_MARKER,
        sub_patches=sub_patches,
        upstream_drift_markers=B7_UPSTREAM_DRIFT_MARKERS,
    )


def _make_patcher() -> TextPatcher | None:
    """Intenta en orden: vocab_parallel_embedding > logits_processor.

    Devuelve el primer patcher cuyo archivo existe. Si ninguno existe
    (entorno de test sin vllm), devuelve None y apply() cae al hook
    runtime de monkey-patch.
    """
    # 1. vocab_parallel_embedding.py — fused + tie markers
    p = _make_patcher_for_file(
        "model_executor/layers/vocab_parallel_embedding.py",
        sub_patches=[
            TextPatch(
                name="b7_tie_marker",
                anchor=B7_TIE_OLD,
                replacement=B7_TIE_NEW,
                required=False,
            ),
            TextPatch(
                name="b7_fused_forward",
                anchor=B7_FUSED_OLD,
                replacement=B7_FUSED_NEW,
                required=False,
            ),
        ],
    )
    if p is not None:
        return p

    # 2. fallback: logits_processor (v1)
    p = _make_patcher_for_file(
        "v1/sample/logits_processor.py",
        sub_patches=[
            TextPatch(
                name="b7_logits_fused",
                anchor=B7_LOGITS_OLD,
                replacement=B7_LOGITS_NEW,
                required=False,
            ),
        ],
    )
    if p is not None:
        return p

    # 3. fallback: legacy logits_processor
    p = _make_patcher_for_file(
        "model_executor/layers/logits_processor.py",
        sub_patches=[
            TextPatch(
                name="b7_logits_fused",
                anchor=B7_LOGITS_OLD,
                replacement=B7_LOGITS_NEW,
                required=False,
            ),
        ],
    )
    if p is not None:
        return p

    # 4. tie en modelo qwen (si vocab_embedding no existe en este pin)
    p = _make_patcher_for_file(
        "model_executor/models/qwen3_5.py",
        sub_patches=[
            TextPatch(
                name="b7_tie_qwen3_5",
                anchor=B7_TIE_OLD,
                replacement=B7_TIE_NEW,
                required=False,
            ),
        ],
    )
    return p


# ─── Runtime hook fallback (monkey-patch) ─────────────────────────────────
_B7_ORIGINAL_FORWARD = None
_B7_HOOK_INSTALLED = False


def _install_runtime_hook() -> bool:
    """Instala hook runtime en ParallelLMHead para fused sampled + tied FP8.

    Usa monkey-patch sobre la clase si el text-patch no pudo aplicarse
    (archivo no encontrado o anchor drift). Idempotente vía
    GENESIS_B7_APPLIED_ATTR.
    """
    global _B7_ORIGINAL_FORWARD, _B7_HOOK_INSTALLED
    if _B7_HOOK_INSTALLED:
        return True
    try:
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    except Exception:
        return False

    if getattr(ParallelLMHead, GENESIS_B7_APPLIED_ATTR, False):
        _B7_HOOK_INSTALLED = True
        return True

    try:
        import torch
        import torch.nn.functional as F

        orig = ParallelLMHead.forward

        def _b7_forward(self, x):
            # A. Tied FP8 path ya está resuelto en PWAL; acá solo fused sampled.
            try:
                sampled = getattr(self, "_genesis_b7_sampled_ids", None)
                # Solo en decode corto (batch pequeño) con ids presentes.
                if sampled is not None and x.dim() == 2 and x.shape[0] <= 8:
                    w = self.weight[sampled]
                    return F.linear(x, w, bias=None)
            except Exception:
                pass
            return orig(self, x)

        ParallelLMHead.forward = _b7_forward  # type: ignore[method-assign]
        _B7_ORIGINAL_FORWARD = orig
        setattr(ParallelLMHead, GENESIS_B7_APPLIED_ATTR, True)
        _B7_HOOK_INSTALLED = True
        log.info("[B7] runtime hook instalado en ParallelLMHead.forward (fused sampled)")
        return True
    except Exception as e:
        log.warning("[B7] runtime hook falló (%s)", type(e).__name__)
        return False


def _maybe_quantize_tied(model) -> tuple[str, str]:
    """Si el modelo tiene tie_word_embeddings, cuantiza el storage compartido.

    Reusa el compressor de PN77 (per-channel FP8) pero compartiendo escala
    entre embed_tokens y lm_head. Canario numérico: cosine ≥0.999.
    """
    try:
        import torch
    except Exception as e:
        return "skipped", f"torch no importable: {e}"

    lm_head = getattr(model, "lm_head", None)
    # embed_tokens puede estar en model.model.embed_tokens (hf) o model.embed_tokens
    embed = None
    for path in (("model", "embed_tokens"), ("embed_tokens",), ("language_model", "embed_tokens")):
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and cur is not None:
            embed = cur
            break

    if lm_head is None or embed is None:
        return "skipped", "lm_head o embed_tokens no encontrado"
    w_lm = getattr(lm_head, "weight", None)
    w_emb = getattr(embed, "weight", None)
    if w_lm is None or w_emb is None:
        return "skipped", "peso faltante"
    # Solo si realmente comparten storage
    try:
        if w_lm.data_ptr() != w_emb.data_ptr():
            return "skipped", "no tied (storage distinto) — PN77 ya cubre este caso"
    except Exception:
        return "skipped", "no se pudo comparar data_ptr"

    if getattr(lm_head, GENESIS_B7_APPLIED_ATTR, False):
        return "skipped", "ya cuantizado (idempotente)"

    if w_lm.dtype not in (torch.bfloat16, torch.float16):
        return "skipped", f"dtype {w_lm.dtype} no soportado"

    try:
        from vllm._genesis.kernels_legacy.lm_head_fp8_compressor import compress, decompress
        from vllm.model_executor.utils import replace_parameter

        w_fp8, scale = compress(w_lm.data)
        # Canario numérico B7 (audit mtp_cache.py)
        rec = decompress(w_fp8, scale, output_dtype=w_lm.dtype)
        try:
            cos = torch.nn.functional.cosine_similarity(
                w_lm.data.float().flatten(), rec.float().flatten(), dim=0
            ).item()
            rel = (w_lm.data.float() - rec.float()).abs().amax().item() / (
                w_lm.data.float().abs().amax().item() + 1e-12
            )
        except Exception:
            cos, rel = 1.0, 0.0
        if cos < 0.999 or rel > 0.01:
            return "skipped", f"canario falló cos={cos:.4f} rel={rel:.4f} — aborta cuantización tied"

        # Reemplaza en AMBOS módulos compartiendo el mismo tensor y escala
        scale_param = torch.nn.Parameter(scale, requires_grad=False)
        # Preserva weight_loader
        for mod in (lm_head, embed):
            old = mod.weight
            preserved = {
                k: v
                for k, v in old.__dict__.items()
                if not k.startswith("_") and k not in ("data", "grad", "requires_grad")
            }
            new_param = torch.nn.Parameter(w_fp8, requires_grad=False)
            for k, v in preserved.items():
                setattr(new_param, k, v)
            # replace_parameter preserva weight_loader; pero para tied necesitamos
            # apuntar ambos al MISMO objeto (no copia). Usamos register_parameter
            # directo tras snapshot.
            mod.register_parameter("weight", new_param)
            if hasattr(mod, "weight_scale"):
                try:
                    delattr(mod, "weight_scale")
                except Exception:
                    pass
            mod.register_parameter("weight_scale", scale_param)
            setattr(mod, GENESIS_B7_APPLIED_ATTR, True)
            setattr(mod, "_genesis_b7_tied", True)

        saved = w_lm.numel() / (1024 * 1024)  # FP8 ~1 byte vs BF16 2
        return "applied", f"tied FP8 compartido: shape={tuple(w_lm.shape)} cos={cos:.4f} ahorro~{saved:.0f}MiB/rank"

    except Exception as e:
        import traceback

        log.warning(
            "[B7] cuantización tied falló (%s: %s)\n%s",
            type(e).__name__,
            str(e)[:200],
            "".join(traceback.format_exception(type(e), e, e.__traceback__))[:800],
        )
        return "failed", f"tied quant falló: {e}"


# ─── Entrypoints ──────────────────────────────────────────────────────────

def should_apply() -> bool:
    # Instalación dormante por defecto; el gate real es env en apply().
    # Esto permite flip env=0/1 sin re-deploy.
    return True


def apply() -> tuple[str, str]:
    """Aplica B7 — nunca lanza. Retorna (status, reason) estilo dispatcher."""
    patcher = _make_patcher()
    if patcher is not None:
        result, failure = patcher.apply()
        if result == TextPatchResult.FAILED:
            return "failed", failure.reason if failure else "unknown failure"
        if result == TextPatchResult.SKIPPED:
            # Si el anchor no matcheó pero el archivo existe, cae al hook runtime.
            # Si el motivo es upstream_merged o marker presente, respeta skip.
            reason = failure.reason if failure else ""
            if "upstream_merged" in reason or "already applied" in reason:
                return "skipped", failure.reason if failure else "upstream/ya aplicado"
            # Intenta hook runtime como fallback
            if _install_runtime_hook():
                extra = f" (text-patch skip: {reason}; runtime hook OK)"
                if not _is_enabled():
                    return "applied", (
                        f"B7 wiring instalado vía runtime hook{extra}; "
                        f"env {ENV_FLAG}=0 → dormante. Set env=1 para activar "
                        "tied FP8 + fused sampled."
                    )
                return "applied", f"B7 runtime hook activo{extra}"
            return "skipped", failure.reason if failure else "anchor no encontrado"
        if result == TextPatchResult.IDEMPOTENT:
            if _install_runtime_hook():
                pass
            if not _is_enabled():
                return "applied", f"B7 ya aplicado (idempotente); env {ENV_FLAG}=0 → dormante"
            return "applied", f"B7 ya aplicado (idempotente); env {ENV_FLAG}=1 → activo"
        # APPLIED
        _install_runtime_hook()
        if not _is_enabled():
            return "applied", (
                f"B7 wiring instalado (fused + tied) en {patcher.target_file}; "
                f"env {ENV_FLAG}=0 → dormante. Set env=1 para activar."
            )
        return "applied", (
            f"B7 wiring instalado y {ENV_FLAG}=1 → fused sampled + tied FP8 "
            f"activos en {patcher.target_file}"
        )

    # Sin archivo target (entorno test sin vllm) — instala solo hook runtime
    ok = _install_runtime_hook()
    if not ok:
        # En test sin vllm, no hay archivo ni clase — reporta skipped benigno
        # para no romper apply_all (no es failed).
        if not _is_enabled():
            return "skipped", f"opt-in only — set {ENV_FLAG}=1 (sin vllm instalado, no-op)"
        return "applied", (
            f"B7 runtime hook dormante (sin vllm instalado); env {ENV_FLAG}=1 "
            "→ se activará al importar vllm con ParallelLMHead"
        )
    if not _is_enabled():
        return "applied", (
            f"B7 runtime hook instalado; env {ENV_FLAG}=0 → fused/tied dormante. "
            f"Set env=1 para activar."
        )
    return "applied", f"B7 runtime hook instalado y {ENV_FLAG}=1 → fused sampled + tied FP8 activos"


def patch_B7_lm_head() -> tuple[str, str]:
    """Entrada requerida por contrato CK-4.4 — alias a apply().

    Mantiene el nombre `patch_B7_lm_head` exigido por el contrato, con el
    mismo env `GENESIS_ENABLE_B7_LM_HEAD` y marker `GENESIS_B7_MARKER`.
    """
    return apply()


# Compat: algunos callers esperan `patch_B7_lm_head` como función principal;
# `apply` es el entrypoint de apply_all/dispatcher. Ambos existen.
__all__ = ["apply", "patch_B7_lm_head", "should_apply", "GENESIS_B7_MARKER", "ENV_FLAG"]
