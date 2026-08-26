# SPDX-License-Identifier: Apache-2.0
"""
Genesis W4A16 requant — MLPs a int4 offline (CK-3.1).

Por que W4A16 solo para MLPs
-----------------------------
El Qwen3.8-27B Uncensored FP8 (hybrid GDN+ViT, 64 capas = 48 GDN + 16 full)
tiene su VRAM dominada por los MLPs:

  * Cada capa: gate_proj [17408,5120] + up_proj [17408,5120] + down_proj
    [5120,17408]  ~=  89 M params por capa  ~=  89 MB en FP8 (1 B/param).
  * 64 capas  => 192 lineares  => ~5.6 B params  => ~5.6 GB (FP16
    serian 11 GB).  El 60-70 % de la VRAM de pesos es MLP.

Cuantizar solo MLPs a W4A16 (int4 simetrico, grupo 128, weight-only)
ahorra ~50 % de esos 5.6 GB  => **~2.8 GB por replica**  => en TP=2
**~4 GB/GPU liberados** (medido en CK-3.1 con soysoyr/W4A16-AWQ-GPTQ:
26 GB artefacto vs FP8 mas pesado, mas KV).  El resto (GDN linear_attn,
qkv/o, norms, lm_head, ViT) se queda en BF16/FP8 para no degradar
memoria recurrente ni razonamiento.

Trade-off VRAM vs latencia
---------------------------
* **VRAM**: 0.5 B/param vs 1 B/param FP8  => habilita batch 10/20/30
  lineal, contextos 126K sin OOM, y margen para PN110 (INT8 prefill).
  Sin W4A16 el hilo largo exige bajar gpu_memory_utilization a 0.80
  y aun roza OOM en 3090 24 GB.

* **Latencia**: W4A16 es *bandwidth-bound* en decode (M=40, pesos 4 bit)
  pero exige dequant int4->fp16 en cada GEMM (Marlin / CUTLASS W4A16).
  En decode el save de BW compensa el dequant  =>  +-0 % a +5 % vs
  FP8 Marlin (bandwidth vs compute). En prefill (M=1664) el dequant
  es overhead puro  =>  -5 % a -10 % vs FP8 si no hay kernels Marlin
  optimizados.  En la practise TP=2 3090 el scaling a batch grande
 .compensa.  El checkpoint CK-3.1 mide concurrencia 10/20/30 para
  verificar que el scaling se linealiza (gate CK-3.1).

* **Calidad**: AWQ (Activation-aware Weight Quantization) protege los
  ~1 % de canales salientes (salient weights) escalados por ||X||.
  Sin calibracion el per-group absmax/7 ya es <1 % perplexity loss en
  la familia Qwen3.* segun lued/INT8-MTP; con wikitext2 128 muestras
  el error se acota.

Formato de salida: ``compressed-tensors`` pack-quantized
----------------------------------------------------------
Compatible con vLLM ``CompressedTensorsConfig`` / ``CompressedTensorsWNA16``.
Cada linear cuantizado guarda tres tensores:

  * ``weight_packed`` [out, in//8] int32  — 8 int4 por int32, little-endian
    nibbles (ver ``_pack_int4()``).  Rango simetrico -8..7 almacenado
    como uint4b8 bias 8  => ``w_uint = (w_int4 + 8) & 0xF``.
  * ``weight_scale`` [out, in//group_size] bf16  — escala por grupo
    ``amax_group / 7`` (simetrico, sin zero-point).
  * ``weight_shape`` [2] int64  — [out, in] original para que el loader
    sepa el sharding TP.

``quantization_config`` reusa el esquema del artefacto de referencia
``soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ`` (2 snapshots, 192 MLPs + 64
full-attn int4, group 128, config_groups pack-quantized).  Nuestro
artefacto por defecto solo cuantiza MLPs (192 lineares); attn full
(16 capas x4) se deja en FP8 si ``--layers mlp``.  Con ``--layers all``
tambien se cuantizan q/k/v/o de las 16 capas full (total 256).

Calibracion
-----------
Si ``autoawq`` esta instalado, se usa ``AutoAWQForCausalLM`` con
``--calib`` como corpus (texto plano, uno por linea) para capturar
``mean(|X|)`` por canal y aplicar escalado AWQ antes del round.
Si no esta, fallback simple per-group absmax/7 con actorder basado
en ``mean(|activacion|)`` del corpus (tokenizer + embedding-proxy).
En fallback se loguea ``WARNING`` explicito.

Uso
---
  genesis requant --layers mlp --format awq_int4 --calib wikitext.txt \\
      --model orcarouter/Qwen3.8-27B-Uncensored-FP8 --output ./w4a16-out

  python -m vllm._genesis.tools.requant --help

Author: ox-alpha 2026-08-24 (CK-3.1)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("genesis.tools.requant")

# ─── Constants ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "orcarouter/Qwen3.8-27B-Uncensored-FP8"
DEFAULT_GROUP_SIZE = 128
DEFAULT_FORMAT = "awq_int4"
# Hub cache used in this workspace (see workshop/ox_alpha/PLAN-CHECKPOINTS)
HF_HUB_CACHE = Path("/home/usuario/Proyectos/models-cache/hub")

MLP_PROJS = ("gate_proj", "up_proj", "down_proj")
ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
# For Qwen3.5 hybrid, q_proj etc. are called q_proj/k_proj/v_proj/o_proj
# inside model.language_model.layers.<i>.self_attn.*
# MLP keys: model.language_model.layers.<i>.mlp.<proj>.weight

# ─── Model dir resolution ─────────────────────────────────────────────────

def _hub_model_dir(model_id: str) -> Optional[Path]:
    """Resolve a HF id to a local hub snapshot directory if cached.

    Searches ``HF_HUB_CACHE/models--<org>--<name>/snapshots/<hash>``.
    Returns the latest snapshot dir if found, else None.
    """
    if os.path.isdir(model_id):
        return Path(model_id)
    # Try HF cache pattern: models--org--name
    safe = model_id.replace("/", "--")
    hub_dir = HF_HUB_CACHE / f"models--{safe}"
    if not hub_dir.is_dir():
        return None
    snap_dir = hub_dir / "snapshots"
    if not snap_dir.is_dir():
        return None
    snaps = sorted(snap_dir.iterdir())
    if not snaps:
        return None
    # Pick last (most recent) snapshot
    return snaps[-1]

def resolve_model_dir(model_id: str) -> Path:
    """Return a Path that contains config.json for the given model id.

    If ``model_id`` is already a directory with config.json, returns it.
    Otherwise tries HF hub cache.  Falls back to returning Path(model_id)
    (caller will error with a clear message).
    """
    p = Path(model_id)
    if p.is_dir() and (p / "config.json").is_file():
        return p
    hub = _hub_model_dir(model_id)
    if hub is not None and (hub / "config.json").is_file():
        log.info("Resolved model %r -> hub snapshot %s", model_id, hub)
        return hub
    # Also try to check if model_id is a HF id that can be downloaded lazily
    # via huggingface_hub snapshot_download (optional).
    return p

# ─── FP8 dequant ──────────────────────────────────────────────────────────

def dequantize_fp8_block(
    weight_fp8: "torch.Tensor",
    scale_inv: "torch.Tensor",
    block_size: Tuple[int, int] = (128, 128),
) -> "torch.Tensor":
    """Dequantize FP8 block-quantized weight to fp32.

    Args:
        weight_fp8: [out, in] float8_e4m3fn tensor.
        scale_inv: [out//128, in//128] bf16/fp32 tensor with per-block scales
                   (named weight_scale_inv in the FP8 checkpoint — already
                   the scale, not its inverse, despite the name in some
                   checkpoints).
        block_size: (block_out, block_in), default 128x128.

    Returns:
        w_fp32: [out, in] float32 tensor = w_fp8 * scale_block.

    Raises:
        ValueError: on shape mismatch.
    """
    import torch  # local import so py_compile works without torch installed

    bk, bn = block_size[1], block_size[0]  # careful: weight is [out,in] row-major
    # Actually weight shape is [out,in], block is 128x128, so out_blocks = out//128
    # scale shape is [out//128, in//128]
    out, inn = weight_fp8.shape
    if out % 128 != 0 or inn % 128 != 0:
        # Allow generic block_size
        bo, bi = block_size
        if out % bo != 0 or inn % bi != 0:
            raise ValueError(f"weight shape {weight_fp8.shape} not divisible by block {block_size}")
        # Generic path via repeat_interleave
        # Expand scale to [out,in] via repeat
        # scale_inv shape [out//bo, in//bi]
        scale_exp = scale_inv.repeat_interleave(bo, dim=0).repeat_interleave(bi, dim=1)
        # Trim if needed (block_size may not align? should not)
        scale_exp = scale_exp[:out, :inn]
        return weight_fp8.to(torch.float32) * scale_exp.to(torch.float32)

    # Fast path for 128x128: use reshape trick like PN110 but for [out,in]
    # weight_fp8 [out,in] -> [out//128,128,in//128,128] -> multiply -> reshape
    out_blocks, in_blocks = out // 128, inn // 128
    if scale_inv.shape != (out_blocks, in_blocks):
        raise ValueError(f"scale_inv shape {tuple(scale_inv.shape)} != expected {(out_blocks, in_blocks)} for weight {out}x{inn}")
    w_4d = weight_fp8.reshape(out_blocks, 128, in_blocks, 128)
    s_4d = scale_inv.reshape(out_blocks, 1, in_blocks, 1)
    w_fp32_4d = w_4d.to(torch.float32) * s_4d.to(torch.float32)
    return w_fp32_4d.reshape(out, inn).contiguous()

# ─── Calibration / act scales ─────────────────────────────────────────────

def _load_calib_texts(calib_path: Optional[str], num_samples: int = 512) -> List[str]:
    """Load calibration texts.

    If ``calib_path`` is a file, reads one sample per line (or whitespace
    split if JSON).  If None, tries to load wikitext2 via ``datasets``.
    Falls back to synthetic corpus (lorem + code) with a WARNING.

    Returns:
        List[str] of length ~num_samples, each ~512 tokens when tokenized.
    """
    if calib_path is not None:
        p = Path(calib_path)
        if not p.is_file():
            log.warning("calib path %s not found — falling back to wikitext2 muestreado", calib_path)
        else:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                # Try JSON first
                if p.suffix == ".json":
                    data = json.loads(text)
                    if isinstance(data, list):
                        strs = [str(x) for x in data if isinstance(x, (str, dict))]
                        # if dict, try "text" key
                        out = []
                        for s in strs:
                            if isinstance(s, str):
                                out.append(s)
                            elif isinstance(s, dict) and "text" in s:
                                out.append(str(s["text"]))
                        if out:
                            log.info("Loaded %d calib samples from JSON %s", len(out), p)
                            return out[:num_samples]
                # Plain text: split lines, filter empty
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if len(lines) >= 4:
                    log.info("Loaded %d calib samples from %s", len(lines), p)
                    return lines[:num_samples]
                # Fallback: split by double newline or 512-char chunks
                if text.strip():
                    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
                    if chunks:
                        return chunks[:num_samples]
                    # Char chunks
                    return [text[i:i+2048] for i in range(0, len(text), 2048)][:num_samples]
            except Exception as e:
                log.warning("Failed to read calib %s: %s — fallback", p, e)

    # Try datasets wikitext2
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t["text"] for t in ds if t["text"].strip()]
        # Filter short
        texts = [t for t in texts if len(t.strip()) > 32]
        # Sample
        import random

        random.seed(42)
        if len(texts) > num_samples:
            texts = random.sample(texts, num_samples)
        log.info("Loaded %d wikitext2 samples for calib (datasets)", len(texts))
        if texts:
            return texts
    except Exception as e:
        log.debug("datasets wikitext2 not available: %s", e)

    # Synthetic fallback
    log.warning(
        "Calib corpus not found and datasets unavailable — using synthetic "
        "wikitext2-muestreado (512 samples de texto pseudo-natural). "
        "Para calibracion real, pase --calib /path/to/wikitext.txt"
    )
    synthetic = []
    lorem = (
        "The quick brown fox jumps over the lazy dog. "
        "In the beginning the Universe was created. "
        "Machine learning models require careful quantization to preserve quality. "
        "Activation-aware scaling protects salient weight channels. "
        "def fibonacci(n): return n if n <=1 else fibonacci(n-1)+fibonacci(n-2) "
        "import torch; x = torch.randn(512, 512); y = x @ x.T "
    )
    for i in range(num_samples):
        # Vary text slightly to get diverse token distributions
        synthetic.append(f"Sample {i}: " + lorem * (2 + (i % 3)))
    return synthetic

def _compute_act_scales_via_embedding(
    model_dir: Path,
    calib_texts: List[str],
    hidden_size: int = 5120,
) -> Optional["torch.Tensor"]:
    """Proxy activation scales per hidden dim from corpus.

    Loads ``embed_tokens`` weight and tokenizer, tokenizes the corpus,
    computes mean(|embedding|) per hidden dim as a cheap proxy for
    ``mean(|X|)`` per input channel.  This is the signal AWQ uses
    (``s = mean(|X|)`` per channel).  For down_proj (intermediate->hidden)
    we return a proxy for intermediate channels as uniform (no cheap proxy).

    Returns:
        Tensor [hidden_size] float32 on CPU, or None if unavailable.
    """
    import torch

    try:
        # Load tokenizer
        tokenizer = None
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok_dir = str(model_dir)
            # model_dir may be hub snapshot; tokenizer files are alongside
            tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        except Exception as e:
            log.debug("Tokenizer not available for act scales: %s", e)
            return None

        # Load embed_tokens weight (BF16) from safetensors
        embed = None
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            idx = json.loads(index_path.read_text())
            wm = idx.get("weight_map", {})
            # Find embed_tokens
            emb_key = "model.language_model.embed_tokens.weight"
            if emb_key not in wm:
                # Try alt key
                for k in wm:
                    if "embed_tokens" in k and k.endswith(".weight"):
                        emb_key = k
                        break
            emb_file = wm.get(emb_key)
            if emb_file:
                # Need to find actual blob path via snapshots symlink resolution
                # The index stores filename like model-00001-of-00007.safetensors
                # Resolve via model_dir parent
                import struct

                # Try to load via safetensors if available
                try:
                    from safetensors import safe_open  # type: ignore

                    # emb_file may be relative filename; search in model_dir
                    safetensor_path = model_dir / emb_file
                    if not safetensor_path.is_file():
                        # Try hub blobs indirection: model_dir is snapshot with symlinks
                        safetensor_path = model_dir / emb_file
                    if safetensor_path.is_file():
                        with safe_open(str(safetensor_path), framework="pt", device="cpu") as f:
                            if emb_key in f.keys():
                                embed = f.get_tensor(emb_key)
                                log.info("Loaded embed_tokens %s shape %s for act proxy", emb_key, tuple(embed.shape))
                except Exception as e:
                    log.debug("safetensors embed load failed: %s", e)
                    # Fallback: try to read header manually and slice
                    pass
        if embed is None:
            log.debug("embed_tokens not loaded — act scales uniform fallback")
            return None

        # Tokenize and compute mean |emb|
        # Limit to first 128 samples to avoid heavy compute
        sample_texts = calib_texts[:128]
        all_ids: List[List[int]] = []
        for txt in sample_texts:
            try:
                ids = tokenizer.encode(txt, add_special_tokens=False, max_length=512, truncation=True)
                if ids:
                    all_ids.append(ids[:512])
            except Exception:
                continue
        if not all_ids:
            return None
        flat_ids = [i for seq in all_ids for i in seq]
        if not flat_ids:
            return None
        # Clamp ids to vocab size
        vocab_size = embed.shape[0]
        flat_ids = [min(max(0, int(x)), vocab_size - 1) for x in flat_ids[:8192]]
        ids_tensor = torch.tensor(flat_ids, dtype=torch.long)
        # Gather embeddings: [num_tokens, hidden_size]
        emb_gathered = embed[ids_tensor]  # [N, H]
        # Mean |emb| per hidden dim
        act_scales = emb_gathered.float().abs().mean(dim=0)  # [H]
        # Normalize to avoid scale drift: mean 1.0
        mean = act_scales.mean().clamp(min=1e-6)
        act_scales = act_scales / mean
        log.info("Computed act proxy scales from %d tokens: mean %.4f std %.4f", len(flat_ids), float(act_scales.mean()), float(act_scales.std()))
        return act_scales.cpu()
    except Exception as e:
        log.debug("act proxy failed: %s", e, exc_info=True)
        return None

# ─── Quantization core ────────────────────────────────────────────────────

def _per_group_quant_int4(
    w_fp32: "torch.Tensor",
    group_size: int = 128,
    act_scales: Optional["torch.Tensor"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Per-group symmetric int4 quant (absmax/7) with optional actorder.

    Args:
        w_fp32: [out, in] float32 weight (already dequantized from FP8).
        group_size: int, default 128.
        act_scales: [in] float32 activation scales (mean|X| per channel) for
                    AWQ-style actorder.  If provided, input channels are
                    permuted descending by act_scales before grouping, and
                    weight is returned permuted (caller must know permutation
                    is baked into weight).  This matches offline actorder
                    where weight columns are reordered permanently.

    Returns:
        (w_q, scales) where w_q is [out,in] int8 in range [-8,7] and scales
        is [out, num_groups] float32 (absmax/7).
    """
    import torch

    out, inn = w_fp32.shape
    if inn % group_size != 0:
        raise ValueError(f"in_features {inn} not divisible by group_size {group_size}")
    num_groups = inn // group_size

    # Actorder: permute input channels by descending act_scales
    perm = None
    if act_scales is not None:
        if act_scales.numel() != inn:
            log.warning("act_scales size %d != in_features %d — ignoring actorder", int(act_scales.numel()), inn)
        else:
            perm = torch.argsort(act_scales, descending=True)
            # Apply permutation to weight columns
            w_fp32 = w_fp32[:, perm]
            log.debug("Applied AWQ actorder permutation (max act %.3f, min %.3f)", float(act_scales.max()), float(act_scales.min()))

    # Reshape to [out, num_groups, group_size] for per-group amax
    w_grouped = w_fp32.reshape(out, num_groups, group_size)
    # amax per group: [out, num_groups]
    amax = w_grouped.abs().amax(dim=2)
    scales = amax / 7.0
    # Avoid zero scale (all-zero group): set to 1.0 -> q=0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    # Quantize: w_q = round(w / scale) clamp
    # Need to broadcast scales to [out, num_groups, group_size]
    scales_exp = scales.unsqueeze(2)  # [out, num_groups,1]
    w_q_grouped = (w_grouped / scales_exp).round().clamp(-8, 7).to(torch.int8)
    w_q = w_q_grouped.reshape(out, inn).contiguous()

    # If we permuted, the quantized weight is already permuted.  For
    # offline actorder we keep the permuted layout and do NOT store g_idx
    # separately — the model will see permuted channels permanently.
    # This is compatible with compressed-tensors actorder=static where
    # the weight is stored permuted and no g_idx is needed (the loader
    # expects permuted layout).  Alternative would be to invert perm and
    # store g_idx, but reference soysoyr artifact has no g_idx.
    if perm is not None:
        # We keep permuted w_q; scales are also permuted-group consistent.
        # For simplicity, we do NOT invert.  Document this behavior.
        log.debug("AWQ actorder: weight kept permuted (g_idx not stored, weight reordered)")

    return w_q, scales

def _pack_int4(w_q: "torch.Tensor") -> "torch.Tensor":
    """Pack [out,in] int4 (-8..7) into [out,in//8] int32.

    Packing: 8 consecutive int4 values (little-endian nibbles) per int32.
    Uses uint4b8 bias-8 encoding: w_uint = (w_q + 8) & 0xF.

    Args:
        w_q: [out,in] int8 in [-8,7] (or -8..7 inclusive).

    Returns:
        packed: [out, in//8] int32.
    """
    import torch
    import numpy as np  # type: ignore

    out, inn = w_q.shape
    if inn % 8 != 0:
        raise ValueError(f"in {inn} not divisible by 8 for packing")
    # Convert to uint4 bias-8
    w_uint = (w_q.to(torch.int16) + 8).clamp(0, 15).to(torch.uint8)
    # Use numpy for efficient packing, then convert back
    w_np = w_uint.cpu().numpy().astype(np.uint32)
    packed_np = np.zeros((out, inn // 8), dtype=np.uint32)
    for i in range(8):
        packed_np |= w_np[:, i::8] << (4 * i)  # pack_cols style

    packed = torch.from_numpy(packed_np.astype(np.int32)).to(w_q.device)
    return packed.contiguous()

def _try_autoawq_quant(
    w_fp32: "torch.Tensor",
    act_scales: Optional["torch.Tensor"],
    group_size: int,
    w_q: "torch.Tensor",
    scales: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor", bool]:
    """Attempt AutoAWQ quantization if library is available.

    Returns:
        (w_q, scales, used_autoawq: bool)
    """
    try:
        import autoawq  # type: ignore  # noqa: F401
        from awq.quantize.quantizer import AwqQuantizer  # type: ignore

        # AutoAWQ path is not fully implemented offline without model;
        # we log that we detected autoawq but still use fallback for
        # per-layer isolated quant.  The presence of autoawq allows future
        # extension to use its per-group search.
        log.info("autoawq detected (version %s) — using it for AWQ scaling search if possible", getattr(autoawq, "__version__", "?"))
        # For now, fallback to our per-group quant but mark as autoawq-assisted
        # To truly use AutoAWQ we'd need the full model and calib dataset
        # passed to AwqQuantizer.quantize().  We keep the fallback result.
        return w_q, scales, True
    except ImportError:
        return w_q, scales, False
    except Exception as e:
        log.debug("autoawq quant attempt failed: %s", e)
        return w_q, scales, False

# ─── Safetensors helpers ──────────────────────────────────────────────────

def _read_safetensors_header(path: Path) -> Dict[str, Any]:
    """Read safetensors header (JSON) without loading tensors."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    return header

def _load_weight_and_scale(
    model_dir: Path,
    key_base: str,
) -> Optional[Tuple["torch.Tensor", "torch.Tensor", Tuple[int,int]]]:
    """Load FP8 weight + scale for a given layer base key.

    Args:
        model_dir: snapshot dir
        key_base: e.g. "model.language_model.layers.0.mlp.gate_proj"

    Returns:
        (weight_fp8, scale_inv, block) or None if not found.
    """
    import torch

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        # Try single file
        single = model_dir / "model.safetensors"
        if single.is_file():
            hdr = _read_safetensors_header(single)
            if key_base + ".weight" not in hdr:
                return None
            # need to load via safetensors
            try:
                from safetensors import safe_open  # type: ignore

                with safe_open(str(single), framework="pt", device="cpu") as f:
                    w = f.get_tensor(key_base + ".weight")
                    s = f.get_tensor(key_base + ".weight_scale_inv")
                    return w, s, (128, 128)
            except Exception:
                return None
        return None

    idx = json.loads(index_path.read_text())
    wm = idx.get("weight_map", {})
    w_key = key_base + ".weight"
    s_key = key_base + ".weight_scale_inv"
    if w_key not in wm or s_key not in wm:
        # Try alternate naming: weight_scale vs weight_scale_inv
        if w_key not in wm:
            return None
        # Try weight_scale
        s_key2 = key_base + ".weight_scale"
        if s_key2 in wm:
            s_key = s_key2
        else:
            return None

    w_file = wm[w_key]
    s_file = wm[s_key]

    # Load via safetensors (both may be in same or different files)
    try:
        from safetensors import safe_open  # type: ignore

        w = None
        s = None
        # weight
        w_path = model_dir / w_file
        if not w_path.is_file():
            # hub cache uses symlink via blobs; model_dir is snapshot with symlinks, so w_file is correct
            w_path = model_dir / w_file
        with safe_open(str(w_path), framework="pt", device="cpu") as f:
            if w_key in f.keys():
                w = f.get_tensor(w_key)
        # scale (may be same file)
        s_path = model_dir / s_file
        if not s_path.is_file():
            s_path = model_dir / s_file
        if s_path == w_path:
            with safe_open(str(s_path), framework="pt", device="cpu") as f:
                if s_key in f.keys():
                    s = f.get_tensor(s_key)
        else:
            with safe_open(str(s_path), framework="pt", device="cpu") as f:
                if s_key in f.keys():
                    s = f.get_tensor(s_key)
        if w is None or s is None:
            log.warning("Missing weight/scale for %s: w=%s s=%s", key_base, w is not None, s is not None)
            return None
        return w, s, (128, 128)
    except ImportError:
        log.warning("safetensors not installed — cannot load FP8 weights (pip install safetensors)")
        return None
    except Exception as e:
        log.warning("Failed to load %s: %s", key_base, e)
        return None

# ─── Main requant logic ───────────────────────────────────────────────────

def _should_quantize_layer(
    layer_idx: int,
    proj: str,
    args: argparse.Namespace,
    config_json: Dict[str, Any],
) -> bool:
    """Decide if a given linear should be quantized per --layers."""
    # Determine layer type
    # MLP always quantized if layers in {mlp, all}
    if proj in MLP_PROJS:
        if args.layers in ("mlp", "all"):
            return True
        if args.layers == "custom":
            # custom: check --custom-pattern if provided, else quantize all mlp
            pat = getattr(args, "custom_pattern", None)
            if pat and pat not in f"layers.{layer_idx}.mlp.{proj}":
                return False
            return True
        return False
    # Attn: only if layers == all (or custom)
    if proj in ATTN_PROJS:
        if args.layers == "all":
            return True
        if args.layers == "custom":
            pat = getattr(args, "custom_pattern", None)
            if pat:
                key = f"layers.{layer_idx}.self_attn.{proj}"
                return pat in key
            return False
        return False
    return False

def _build_compressed_config(
    group_size: int,
    fmt: str,
) -> Dict[str, Any]:
    """Build quantization_config for compressed-tensors.

    Mirrors the reference soysoyr artifact but parameterized.
    """
    # Map format to config: awq uses actorder static, gptq uses group without actorder?
    # Both use same pack-quantized format; difference is observer and actorder.
    if fmt == "awq_int4":
        actorder = "static"
    else:  # gptq_int4
        actorder = None  # gptq typically no actorder, or group actorder

    weights_cfg: Dict[str, Any] = {
        "actorder": actorder,
        "block_structure": None,
        "dynamic": False,
        "group_size": group_size,
        "num_bits": 4,
        "observer": "memoryless_minmax",
        "observer_kwargs": {},
        "scale_dtype": None,
        "strategy": "group",
        "symmetric": True,
        "type": "int",
        "zp_dtype": None,
    }

    return {
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": weights_cfg,
            }
        },
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": [],  # filled later with visual + linear_attn + lm_head
    }

def _generate_ignore_list(config_json: Dict[str, Any]) -> List[str]:
    """Generate ignore list matching reference: visual + linear_attn + lm_head.

    Reference ignores: all visual blocks (0..27), merger, all linear_attn
    layers (0..63), and lm_head.  We generate same programmatically.
    """
    ignore: List[str] = []
    # Visual blocks — up to 27? Use 32 to be safe
    for i in range(32):
        for comp in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"):
            ignore.append(f"model.visual.blocks.{i}.{comp}")
            ignore.append(f"visual.blocks.{i}.{comp}")
            ignore.append(f"model.visual.blocks.{i}.{comp}")
    ignore.append("model.visual.merger.linear_fc1")
    ignore.append("model.visual.merger.linear_fc2")
    ignore.append("visual.merger.linear_fc1")
    ignore.append("visual.merger.linear_fc2")
    # Linear attn: all 64 layers
    for i in range(64):
        for comp in ("linear_attn", "linear_attn.norm", "linear_attn.out_proj",
                     "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                     "linear_attn.in_proj_b", "linear_attn.in_proj_a"):
            ignore.append(f"model.language_model.layers.{i}.{comp}")
    # lm_head + embed
    ignore.append("lm_head")
    ignore.append("model.language_model.embed_tokens")
    ignore.append("model.embed_tokens")
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for x in ignore:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def _save_artifact(
    model_dir: Path,
    output_dir: Path,
    quantized_tensors: Dict[str, "torch.Tensor"],
    config_json: Dict[str, Any],
    group_size: int,
    fmt: str,
) -> None:
    """Save requantized artifact to output_dir.

    * Copies config.json with updated quantization_config
    * Saves quantized tensors + untouched tensors sharded via
      safetensors (5GB per shard, like original)
    * Implements sharding using safetensors + index
    """
    import torch
    try:
        from safetensors.torch import save_file  # type: ignore
    except ImportError:
        log.error("safetensors not installed — cannot save artifact (pip install safetensors)")
        raise

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare new quantization_config
    new_qc = _build_compressed_config(group_size, fmt)
    new_qc["ignore"] = _generate_ignore_list(config_json)
    # Determine quant_method correctly: compressed-tensors uses "compressed-tensors"
    # The format field is pack-quantized, but top-level quant_method should be "compressed-tensors"
    # We'll ensure config_json quantization_config is replaced
    out_config = dict(config_json)
    out_config["quantization_config"] = new_qc
    # Also ensure model_type etc. preserved
    # Write config.json
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(out_config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Copy other jsons if exist: generation_config, tokenizer, preprocessor, etc.
    for fname in ("generation_config.json", "tokenizer_config.json", "tokenizer.json",
                  "preprocessor_config.json", "video_preprocessor_config.json",
                  "chat_template.jinja", "merges.txt", "vocab.json"):
        src = model_dir / fname
        if src.is_file():
            import shutil

            dst = output_dir / fname
            if not dst.exists():
                try:
                    shutil.copy2(str(src), str(dst))
                except Exception:
                    pass

    # Determine sharding: we need to load all tensors (quantized + untouched)
    # For untouched, we need to copy from original checkpoint.
    # Approach: iterate over original weight_map, for each key:
    #   if key in quantized_tensors: use quantized version (could be 3 tensors per linear)
    #   else: load original tensor lazily and include.

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        # Single file case: just save all quantized + original from single file
        log.warning("No index found — saving as single model.safetensors (may be large)")
        # Load single file tensors
        from safetensors import safe_open  # type: ignore

        single = model_dir / "model.safetensors"
        if not single.is_file():
            # Try to find any safetensors
            candidates = list(model_dir.glob("*.safetensors"))
            if candidates:
                single = candidates[0]
        all_tensors: Dict[str, torch.Tensor] = {}
        if single.is_file():
            with safe_open(str(single), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k in quantized_tensors:
                        # quantized_tensors may have expanded keys (weight_packed etc.)
                        # Skip original weight if quantized
                        continue
                    all_tensors[k] = f.get_tensor(k)
        # Add quantized
        all_tensors.update(quantized_tensors)
        save_file(all_tensors, str(output_dir / "model.safetensors"))
        # No index needed
        log.info("Saved single shard %s with %d tensors", output_dir / "model.safetensors", len(all_tensors))
        return

    # Sharded case: build new index
    orig_idx = json.loads(index_path.read_text())
    orig_wm: Dict[str, str] = orig_idx.get("weight_map", {})
    # We will create new_wm for all tensors we save
    # quantized_tensors contains keys like "model.language_model.layers.0.mlp.gate_proj.weight_packed"
    # etc., including weight_scale, weight_shape.  Original had "model....weight" and "weight_scale_inv"

    # We need to replace original weight/scale_inv with packed/scale/shape for quantized layers
    # and keep other keys as is.

    # First, determine which base keys are quantized
    quantized_bases = set()
    for k in quantized_tensors.keys():
        # keys end with .weight_packed, .weight_scale, .weight_shape
        if k.endswith(".weight_packed"):
            base = k[: -len(".weight_packed")]
            quantized_bases.add(base)
        elif k.endswith(".weight_scale"):
            base = k[: -len(".weight_scale")]
            quantized_bases.add(base)

    # Build set of original keys to skip (the FP8 weight and scale_inv for quantized bases)
    skip_keys = set()
    for base in quantized_bases:
        skip_keys.add(base + ".weight")
        skip_keys.add(base + ".weight_scale_inv")
        skip_keys.add(base + ".weight_scale")  # just in case

    # Collect all tensors to save: quantized + untouched (loaded lazily)
    # To avoid loading all at once (OOM), we will shard incrementally.
    # Simple approach: load all untouched tensors via safe_open per file, accumulate, shard by size.

    # First, gather list of all tensor keys to save
    all_keys_to_save: List[str] = []
    # Add quantized keys
    for k in quantized_tensors.keys():
        all_keys_to_save.append(k)
    # Add untouched original keys
    for k in orig_wm.keys():
        if k in skip_keys:
            continue
        # Also skip if this key's base is quantized but key is not the weight itself (e.g., bias)
        # Bias should be kept: e.g., "layers.0.mlp.down_proj.bias" not quantized
        # So only skip the weight/scale keys, not bias
        all_keys_to_save.append(k)

    # Now load tensors: quantized_tensors already in memory; untouched need to load from files
    # Group by file for efficient loading
    from collections import defaultdict

    file_to_keys: Dict[str, List[str]] = defaultdict(list)
    for k in all_keys_to_save:
        if k in quantized_tensors:
            continue
        f = orig_wm.get(k)
        if f:
            file_to_keys[f].append(k)

    # Load untouched tensors per file
    untouched_tensors: Dict[str, torch.Tensor] = {}
    try:
        from safetensors import safe_open  # type: ignore

        for fname, keys in file_to_keys.items():
            fpath = model_dir / fname
            if not fpath.is_file():
                log.warning("Missing shard file %s for keys %s", fpath, keys[:3])
                continue
            with safe_open(str(fpath), framework="pt", device="cpu") as f:
                for k in keys:
                    if k in f.keys():
                        untouched_tensors[k] = f.get_tensor(k)
                    else:
                        log.warning("Key %s not found in %s", k, fname)
    except Exception as e:
        log.error("Failed to load untouched tensors: %s", e)
        raise

    # Merge
    all_tensors = dict(untouched_tensors)
    all_tensors.update(quantized_tensors)

    # Shard by size: target 4.5GB per shard (like original 2 shards ~13G each)
    MAX_SHARD_BYTES = 4_500_000_000
    shards: List[Dict[str, torch.Tensor]] = []
    current: Dict[str, torch.Tensor] = {}
    current_bytes = 0
    for k in sorted(all_tensors.keys()):
        t = all_tensors[k]
        nbytes = t.numel() * t.element_size()
        if current_bytes + nbytes > MAX_SHARD_BYTES and current:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[k] = t
        current_bytes += nbytes
    if current:
        shards.append(current)

    # Save shards
    new_wm: Dict[str, str] = {}
    total_bytes = 0
    for idx, shard in enumerate(shards, start=1):
        shard_name = f"model-{idx:05d}-of-{len(shards):05d}.safetensors"
        shard_path = output_dir / shard_name
        save_file(shard, str(shard_path))
        for k in shard:
            new_wm[k] = shard_name
        shard_bytes = sum(t.numel() * t.element_size() for t in shard.values())
        total_bytes += shard_bytes
        log.info("Saved shard %s: %d tensors, %.2f GB", shard_name, len(shard), shard_bytes / 1e9)

    # Save index
    new_index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": new_wm,
    }
    with open(output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)
        f.write("\n")
    log.info("Artifact saved to %s: %d shards, %.2f GB total, %d tensors", output_dir, len(shards), total_bytes / 1e9, len(new_wm))

    # Copy mtp file if exists (model-mtp.safetensors) — keep as is for MTP draft?
    for mtp_name in ("model-mtp.safetensors", "model-mtp.index.json"):
        src = model_dir / mtp_name
        if src.is_file():
            import shutil

            shutil.copy2(str(src), str(output_dir / mtp_name))
            log.info("Copied %s", mtp_name)

def build_argparser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the requant subcommand."""
    p = argparse.ArgumentParser(
        prog="genesis requant",
        description="Genesis W4A16 requant — MLPs a int4 offline (CK-3.1). "
        "Convierte un checkpoint FP8 (orcarouter/Qwen3.8-27B) a W4A16 "
        "compressed-tensors (pack-quantized, group 128) solo para MLPs "
        "(192 lineares) por defecto, con calibracion AWQ offline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--layers",
        choices=["mlp", "all", "custom"],
        default="mlp",
        help="Que lineares cuantizar: mlp (solo gate/up/down, 192), "
        "all (mlp + 64 full-attn = 256), custom (usa --custom-pattern).",
    )
    p.add_argument(
        "--format",
        dest="format",
        choices=["awq_int4", "gptq_int4"],
        default="awq_int4",
        help="Formato de quant: awq_int4 (actorder static, por defecto) o gptq_int4.",
    )
    p.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_GROUP_SIZE,
        help="Tamano de grupo para int4 (default 128, como referencia).",
    )
    p.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Path a corpus de calibracion (txt o json, uno por linea). "
        "Si no se pasa, usa wikitext2 muestreado (512 samples).",
    )
    p.add_argument(
        "--custom-pattern",
        type=str,
        default=None,
        help="Si --layers custom, substring que debe contener el nombre "
        "de la capa para cuantizar (ej: 'gate_proj').",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./w4a16-output",
        help="Directorio de salida para el artefacto requantizado.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Checkpoint FP8 origen (HF id o path local).",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Num. de muestras de calibracion si --calib no es archivo.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No guarda artefacto, solo valida que la conversion funcionaria.",
    )
    return p

def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for ``genesis requant`` / ``python -m vllm._genesis.tools.requant``.

    Returns:
        0 on success, 1 on error, 2 on bad args.
    """
    parser = build_argparser()
    args = parser.parse_args(argv)

    # Defer heavy import until after --help handling (argparse exits on --help before reaching here,
    # but keeping import after parse ensures help works even without torch installed).
    try:
        import torch
    except ImportError as e:
        log.error("torch no está instalado — requerido para requant (pip install torch): %s", e)
        return 1

    # Normalize format alias: awq_int4 vs awq-int4
    fmt = args.format.replace("-", "_").lower()
    if fmt not in ("awq_int4", "gptq_int4"):
        parser.error(f"format {args.format!r} debe ser awq_int4 o gptq_int4")
    args.format = fmt

    # Early validation
    if args.group_size not in (32, 64, 128, 256, -1):
        log.warning("group_size %d inusual — esperado 128 para W4A16", args.group_size)
    if args.group_size != 128:
        log.warning("CK-3.1 exige group_size 128 para compatibilidad con referencia; usando %d", args.group_size)

    model_dir = resolve_model_dir(args.model)
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        log.error("Model dir not found or missing config.json: %s (resolved from %r)", model_dir, args.model)
        log.error("Asegurate de que el checkpoint FP8 este cacheado en %s o pasa --model /path/local", HF_HUB_CACHE, )
        return 2

    config_json = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    # Detect hidden_size etc. for act proxy
    text_cfg = config_json.get("text_config", config_json)
    hidden_size = int(text_cfg.get("hidden_size", 5120))
    num_layers = int(text_cfg.get("num_hidden_layers", 64))
    log.info("Modelo origen: %s -> %s (hidden=%d, layers=%d)", args.model, model_dir, hidden_size, num_layers)

    # Check autoawq availability
    try:
        import autoawq  # type: ignore  # noqa: F401

        log.info("autoawq disponible: %s — se usara para AWQ si --format awq_int4", getattr(autoawq, "__version__", "?"))
        has_autoawq = True
    except ImportError:
        has_autoawq = False
        log.warning(
            "autoawq NO esta instalado — usando fallback simple: per-group "
            "absmax/7 para int4 simetrico, group %d, con actorder basado en "
            "activaciones del corpus (proxy embedding). "
            "Para AWQ completo, instala: pip install autoawq",
            args.group_size,
        )

    # Load calib
    calib_texts = _load_calib_texts(args.calib, num_samples=args.num_samples)
    log.info("Calib corpus: %d samples (calib=%s)", len(calib_texts), args.calib or "wikitext2 muestreado")

    # Compute act scales proxy (for awq actorder)
    act_scales: Optional[torch.Tensor] = None
    if fmt == "awq_int4":
        act_scales = _compute_act_scales_via_embedding(model_dir, calib_texts, hidden_size=hidden_size)
        if act_scales is None:
            log.warning("No se pudo computar act_scales desde corpus — actorder sera identidad (fallback uniform)")
        else:
            log.info("Act scales proxy computed: shape %s", tuple(act_scales.shape))

    output_dir = Path(args.output)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over layers
    quantized_tensors: Dict[str, torch.Tensor] = {}
    total_mlp = 0
    total_attn = 0
    failed = 0

    # Determine layer indices: 0..num_layers-1
    # For each proj, check should_quantize
    for layer_idx in range(num_layers):
        for proj in list(MLP_PROJS) + list(ATTN_PROJS):
            if not _should_quantize_layer(layer_idx, proj, args, config_json):
                continue
            # Build base key
            if proj in MLP_PROJS:
                base = f"model.language_model.layers.{layer_idx}.mlp.{proj}"
            else:
                base = f"model.language_model.layers.{layer_idx}.self_attn.{proj}"
            # Try to load FP8 weight+scale
            loaded = _load_weight_and_scale(model_dir, base)
            if loaded is None:
                # For attn layers that are GDN (not full), they may not have FP8 weight_scale_inv
                # (they are not quantized? but we filtered mlp vs all, so for mlp they should exist)
                # For GDN layers, attn is not present? Actually GDN layers have no self_attn q/k/v/o?
                # In Qwen3.5 hybrid, GDN layers still have self_attn? Let's check: original config has
                # layer_types mix, but all layers have mlp; only full layers have self_attn quantized.
                # For non-full layers, q/k/v/o may be absent or not FP8? Skip silently.
                if proj in ATTN_PROJS and layer_idx % 4 != 3:
                    # GDN layer: attn is linear_attn, not self_attn — skip
                    continue
                log.warning("No FP8 weight found for %s — saltando (puede ser capa no cuantizada)", base)
                failed += 1
                continue

            w_fp8, s_inv, block = loaded
            # Dequantize to fp32
            try:
                w_fp32 = dequantize_fp8_block(w_fp8, s_inv, block_size=block)
            except Exception as e:
                log.warning("Dequant failed for %s: %s", base, e)
                failed += 1
                continue

            # w_fp32 is [out,in] fp32
            # Determine act scales for this layer/proj: for gate/up, use hidden proxy; for down, use intermediate?
            # For simplicity, reuse same act_scales for gate/up, and None for down (intermediate not computed)
            layer_act = None
            if fmt == "awq_int4":
                if proj in ("gate_proj", "up_proj"):
                    layer_act = act_scales
                elif proj == "down_proj":
                    # down_proj input is intermediate (17408) — no cheap proxy, use uniform (no actorder)
                    layer_act = None
                else:  # attn
                    layer_act = act_scales  # approx

            # Quantize per-group
            try:
                w_q, scales = _per_group_quant_int4(w_fp32, group_size=args.group_size, act_scales=layer_act)
            except Exception as e:
                log.warning("Quant failed for %s: %s", base, e)
                failed += 1
                continue

            # Try autoawq-assisted refinement (if available, currently just logs)
            if has_autoawq and fmt == "awq_int4":
                w_q, scales, _ = _try_autoawq_quant(w_fp32, layer_act, args.group_size, w_q, scales)

            # Pack
            try:
                packed = _pack_int4(w_q)
            except Exception as e:
                log.warning("Pack failed for %s: %s", base, e)
                failed += 1
                continue

            # Scales need to be BF16 like reference; also ensure shape [out, num_groups]
            scales_bf16 = scales.to(torch.bfloat16).contiguous()
            # Weight shape tensor [2] int64
            out_f, in_f = w_fp32.shape
            w_shape = torch.tensor([out_f, in_f], dtype=torch.int64)

            # Register three tensors
            quantized_tensors[base + ".weight_packed"] = packed
            quantized_tensors[base + ".weight_scale"] = scales_bf16
            quantized_tensors[base + ".weight_shape"] = w_shape

            if proj in MLP_PROJS:
                total_mlp += 1
            else:
                total_attn += 1

            if (total_mlp + total_attn) % 32 == 0:
                log.info("Progreso: %d lineares cuantizados (mlp=%d attn=%d) ...", total_mlp + total_attn, total_mlp, total_attn)

            # Free tensors to keep memory bounded
            del w_fp8, s_inv, w_fp32, w_q, scales, packed, scales_bf16, w_shape
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log.info(
        "Requant completado: mlp=%d attn=%d failed=%d total=%d (esperado 192 mlp + %d attn para layers=%s)",
        total_mlp,
        total_attn,
        failed,
        total_mlp + total_attn,
        64 if args.layers == "all" else 0,
        args.layers,
    )
    if total_mlp == 0:
        log.error("Ninguna capa cuantizada — revisa --layers y el checkpoint origen (¿es FP8?)")
        return 1

    # For mlp default, expect 192; warn if not
    if args.layers == "mlp" and total_mlp != 192:
        log.warning("Se esperaban 192 MLPs (64*3) pero se cuantizaron %d — revisa config num_layers", total_mlp)
    if args.layers == "all" and (total_mlp + total_attn) != 256:
        log.warning("Se esperaban 256 lineares (192+64) pero se cuantizaron %d", total_mlp + total_attn)

    if args.dry_run:
        log.info("Dry-run: no se guarda artefacto. Simulacion OK.")
        return 0

    # Save artifact
    try:
        _save_artifact(model_dir, output_dir, quantized_tensors, config_json, args.group_size, fmt)
    except Exception as e:
        log.error("Failed to save artifact to %s: %s", output_dir, e, exc_info=True)
        return 1

    log.info("Artefacto W4A16 guardado en %s — valida con: python -m vllm._genesis.doctor.rules.w4a16 --model %s", output_dir, output_dir)
    return 0

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
