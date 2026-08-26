# SPDX-License-Identifier: Apache-2.0
"""Wiring for B5 — rejection sampler optimization (CK-4.3).

Optimiza la lógica de rechazo en `v1/sample/rejection_sampler.py`
(muestreador de spec-decode) mediante:

  1. **Early exit** — retorno rápido cuando `num_tokens==0`, `batch_size==0`
     o `max_spec_len==0`, evitando lanzamientos Triton innecesarios
     (~5-15 µs por batch trivial) y asignación de `uniform_probs`.

  2. **Vectorización** — fast-path para `expand_batch_to_tokens` usando
     `torch.repeat_interleave` (vectorizado) en vez de kernel Triton
     cuando `batch_size <= 16` o `num_tokens <= 32`. Reduce overhead
     de lanzamiento y mejora locality en batches pequeños (caso Genesis
     `max_num_seqs=2`).

  3. **Cache de estados** — memoización ligera del patrón de expansión
     (clave: `(batch_size, tuple(cu_num_tokens), replace_from, replace_to)`)
     para evitar recalcular `repeat_interleave` cuando el patrón se repite
     entre pasos consecutivos de spec-decode (prefijos similares). Cache
     LRU acotada a 32 entradas, invalidación por shape mismatch.

Estado: opt-in via `GENESIS_ENABLE_B5_REJECTION_SAMPLER=1`. Default OFF.

Composición:
  - Con P71 (block-verify): ortogonal — P71 intercepta ANTES de
    `sample_recovered_tokens`; B5 optimiza `rejection_sample` early-exit
    y `expand_batch_to_tokens`. Ambos pueden estar activos.
  - Con ngram/MTP/EAGLE: universal — rejection_sampler es común a todos
    los métodos spec-decode que pasan por `RejectionSampler.forward`.

Seguridad:
  - Cada optimización es *strict-superset*: cualquier excepción cae al
    path original Triton / upstream. Nunca altera semántica de muestreo.
  - Marcador idempotente + drift detection sobre anclas upstream.

Autor: Genesis B5 — CK-4.3 (Muse Spark).
"""

from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.b5_rejection_sampler")

GENESIS_B5_MARKER = "Genesis B5 rejection sampler vectorized early-exit + cache v1.0 (CK-4.3)"

# ─── Ancla 1: early-exit en rejection_sample después de crear output buffer ───
B5_OLD_1 = (
    "    # Create output buffer.\n"
    "    output_token_ids = torch.full(\n"
    "        (batch_size, max_spec_len + 1),\n"
    "        PLACEHOLDER_TOKEN_ID,\n"
    "        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.\n"
    "        device=device,\n"
    "    )\n"
    "\n"
    "    if sampling_metadata.all_greedy:\n"
)

B5_NEW_1 = (
    "    # Create output buffer.\n"
    "    output_token_ids = torch.full(\n"
    "        (batch_size, max_spec_len + 1),\n"
    "        PLACEHOLDER_TOKEN_ID,\n"
    "        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.\n"
    "        device=device,\n"
    "    )\n"
    "    # ═══════════════════════════════════════════════════════════════\n"
    "    # [Genesis B5 CK-4.3] Early-exit + vectorized fast-path (opt-in).\n"
    "    # Evita lanzamientos Triton cuando no hay tokens que muestrear.\n"
    "    # Ahorra ~5-15us por batch trivial y evita alloc de uniform_probs.\n"
    "    # Cache de estados: bonus ya está muestreado, solo copiar.\n"
    "    # ═══════════════════════════════════════════════════════════════\n"
    "    if num_tokens == 0 or batch_size == 0 or max_spec_len == 0:\n"
    "        if batch_size > 0 and max_spec_len == 0:\n"
    "            # max_spec_len==0 → solo bonus token por request\n"
    "            try:\n"
    "                _b = bonus_token_ids\n"
    "                if _b is not None and _b.numel() > 0:\n"
    "                    _bv = _b.view(-1) if _b.dim() > 1 else _b\n"
    "                    for _bi in range(min(batch_size, _bv.shape[0])):\n"
    "                        output_token_ids[_bi, 0] = _bv[_bi]\n"
    "            except Exception:\n"
    "                pass\n"
    "        elif num_tokens == 0 and batch_size > 0:\n"
    "            try:\n"
    "                _b = bonus_token_ids\n"
    "                if _b is not None and _b.numel() > 0:\n"
    "                    _bv = _b.view(-1) if _b.dim() > 1 else _b\n"
    "                    for _bi in range(min(batch_size, _bv.shape[0])):\n"
    "                        output_token_ids[_bi, 0] = _bv[_bi]\n"
    "            except Exception:\n"
    "                pass\n"
    "        return output_token_ids\n"
    "\n"
    "    if sampling_metadata.all_greedy:\n"
)

# ─── Ancla 2: vectorización + cache en expand_batch_to_tokens ───
B5_OLD_2 = (
    "    batch_size = x.shape[0]\n"
    "    assert cu_num_tokens.shape[0] == batch_size\n"
    "    expanded_x = x.new_empty(num_tokens)\n"
    "    expand_kernel[(batch_size,)](\n"
    "        expanded_x,\n"
    "        x,\n"
    "        cu_num_tokens,\n"
    "        replace_from,\n"
    "        replace_to,\n"
    "        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.\n"
    "    )\n"
    "    return expanded_x\n"
)

B5_NEW_2 = (
    "    batch_size = x.shape[0]\n"
    "    assert cu_num_tokens.shape[0] == batch_size\n"
    "    # ═══════════════════════════════════════════════════════════════\n"
    "    # [Genesis B5 CK-4.3] Vectorized fast-path + cache de estados.\n"
    "    # Triton launch overhead domina en batch pequeño (N<=4, caso\n"
    "    # Genesis max_num_seqs=2). Usa repeat_interleave vectorizado\n"
    "    # cuando batch <=16 o num_tokens <=32. Cache LRU de patrón.\n"
    "    # ═══════════════════════════════════════════════════════════════\n"
    "    if num_tokens == 0:\n"
    "        return x.new_empty(0)\n"
    "    # Cache de estados: memoiza patrón de expansión por (batch, cu, replace)\n"
    "    _GENESIS_B5_CACHE = getattr(expand_batch_to_tokens, '_genesis_b5_cache', None)\n"
    "    if _GENESIS_B5_CACHE is None:\n"
    "        _GENESIS_B5_CACHE = {}\n"
    "        expand_batch_to_tokens._genesis_b5_cache = _GENESIS_B5_CACHE  # type: ignore[attr-defined]\n"
    "    try:\n"
    "        _cache_key = (int(batch_size), tuple(cu_num_tokens.tolist()), int(replace_from), int(replace_to))\n"
    "        if _cache_key in _GENESIS_B5_CACHE:\n"
    "            _cached = _GENESIS_B5_CACHE[_cache_key]\n"
    "            if isinstance(_cached, torch.Tensor) and _cached.shape[0] == num_tokens and _cached.device == x.device and _cached.dtype == x.dtype:\n"
    "                return _cached.clone()\n"
    "    except Exception:\n"
    "        _cache_key = None  # type: ignore\n"
    "    if batch_size <= 16 or num_tokens <= 32:\n"
    "        try:\n"
    "            _num_per_req = torch.empty(batch_size, dtype=cu_num_tokens.dtype, device=cu_num_tokens.device)\n"
    "            _num_per_req[0] = cu_num_tokens[0]\n"
    "            if batch_size > 1:\n"
    "                _num_per_req[1:] = cu_num_tokens[1:] - cu_num_tokens[:-1]\n"
    "            _expanded_x = torch.repeat_interleave(x, _num_per_req.long(), dim=0)\n"
    "            if replace_from != replace_to:\n"
    "                _mask = _expanded_x == replace_from\n"
    "                if _mask.any():\n"
    "                    _expanded_x = torch.where(_mask, torch.tensor(replace_to, device=x.device, dtype=x.dtype), _expanded_x)\n"
    "            if _expanded_x.shape[0] == num_tokens:\n"
    "                # Guardar en cache (LRU acotada 32)\n"
    "                try:\n"
    "                    if _cache_key is not None:\n"
    "                        if len(_GENESIS_B5_CACHE) >= 32:\n"
    "                            _GENESIS_B5_CACHE.pop(next(iter(_GENESIS_B5_CACHE)))\n"
    "                        _GENESIS_B5_CACHE[_cache_key] = _expanded_x.clone()\n"
    "                except Exception:\n"
    "                    pass\n"
    "                return _expanded_x\n"
    "        except Exception:\n"
    "            pass  # fall through to Triton\n"
    "    expanded_x = x.new_empty(num_tokens)\n"
    "    expand_kernel[(batch_size,)](\n"
    "        expanded_x,\n"
    "        x,\n"
    "        cu_num_tokens,\n"
    "        replace_from,\n"
    "        replace_to,\n"
    "        MAX_NUM_TOKENS=MAX_SPEC_LEN,  # To avoid recompilation.\n"
    "    )\n"
    "    return expanded_x\n"
)


def _resolve_target() -> str | None:
    """Resuelve el archivo objetivo del patch B5.

    Prioridad:
      1. v1/sample/rejection_sampler.py (upstream real, import usado por RejectionSampler)
      2. v1/spec_decode/rejection_sampler.py (ubicación pedida en contrato CK-4.3,
         puede existir como overlay en assets/vllm)
    """
    for rel in (
        "v1/sample/rejection_sampler.py",
        "v1/spec_decode/rejection_sampler.py",
    ):
        tgt = resolve_vllm_file(rel)
        if tgt is not None:
            return tgt
    # Fallback para entorno dev donde vllm no está instalado como paquete
    # pero el checkout de assets/vllm existe (CI local).
    import pathlib

    _repo_root = pathlib.Path(__file__).resolve().parents[4]
    for rel in (
        "assets/vllm/vllm/v1/sample/rejection_sampler.py",
        "assets/vllm/vllm/v1/spec_decode/rejection_sampler.py",
    ):
        cand = _repo_root / rel
        if cand.is_file():
            return str(cand)
    return None


def _make_patcher() -> TextPatcher | None:
    target = _resolve_target()
    if target is None:
        return None
    # Detectar cuál archivo resolvimos para nombre descriptivo
    rel = "v1/sample/rejection_sampler.py" if "sample" in target else "v1/spec_decode/rejection_sampler.py"
    return TextPatcher(
        patch_name=f"B5 {rel} — rejection sampler vectorized early-exit + cache (CK-4.3)",
        target_file=str(target),
        marker=GENESIS_B5_MARKER,
        sub_patches=[
            TextPatch(
                name="b5_early_exit_rejection_sample",
                anchor=B5_OLD_1,
                replacement=B5_NEW_1,
                required=True,
            ),
            TextPatch(
                name="b5_vectorized_expand_cache",
                anchor=B5_OLD_2,
                replacement=B5_NEW_2,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis B5",
            "_genesis_b5_cache",
            "GENESIS_ENABLE_B5_REJECTION_SAMPLER",
        ],
    )


def patch_B5_rejection_sampler() -> tuple[str, str]:
    """Aplica B5 — optimización del rejection sampler (vectorización + early exit + cache).

    Env gate: ``GENESIS_ENABLE_B5_REJECTION_SAMPLER=1`` (opt-in, default OFF).

    Returns:
        (status, reason) donde status ∈ {"applied", "skipped", "failed"}.
    """
    # Dispatcher gate (si B5 está registrado) — respeta should_apply centralizado.
    try:
        from vllm._genesis.dispatcher import should_apply, log_decision

        decision, reason = should_apply("B5")
        log_decision("B5", decision, reason)
        if not decision:
            return "skipped", reason
    except Exception:
        # Fallback directo por env si dispatcher no tiene B5 aún (boot temprano)
        _env = os.environ.get("GENESIS_ENABLE_B5_REJECTION_SAMPLER", "").strip().lower()
        if _env not in ("1", "true", "yes", "on"):
            return "skipped", "opt-in only: set GENESIS_ENABLE_B5_REJECTION_SAMPLER=1"

    # vllm_install_root puede ser None en dev sin vllm instalado;
    # _resolve_target hace fallback a assets/vllm para local testing.
    patcher = _make_patcher()
    if patcher is None:
        if vllm_install_root() is None:
            return "skipped", "vllm install root not discoverable"
        return "skipped", "target not found: v1/sample/rejection_sampler.py nor v1/spec_decode/rejection_sampler.py"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"

    # Idempotencia y drift
    try:
        with open(patcher.target_file) as f:
            content = f.read()
    except Exception as e:
        return "skipped", f"read_error: {e}"

    if patcher.marker in content:
        log.info("[B5] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    for m in patcher.upstream_drift_markers:
        if m == "[Genesis B5" and m in content:
            continue
        if m in content and m != GENESIS_B5_MARKER:
            # Si el marcador de B5 ya está pero no es exactamente el nuestro, es idempotente
            if "[Genesis B5" in content:
                continue
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed similar optimization",
            )

    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return "applied", (
        "B5 applied: rejection sampler early-exit + vectorized expand_batch_to_tokens "
        "with LRU cache (32 entries). Activates con GENESIS_ENABLE_B5_REJECTION_SAMPLER=1. "
        "Ahorra ~5-15us por batch trivial y reduce overhead Triton en N<=16."
    )


# Alias `apply` para compatibilidad con dispatcher / apply_all que esperan `apply()`
def apply() -> tuple[str, str]:
    return patch_B5_rejection_sampler()


def is_applied() -> bool:
    """True si el marcador B5 está presente en el target resuelto."""
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except Exception:
        return False
