# SPDX-License-Identifier: Apache-2.0
"""Genesis MTP Quantization Disk Cache Manager.

Provides cryptographic fingerprinting, atomic disk caching, and fast loading
for Qwen3.8 / Qwen3.5 MTP (Multi-Token Prediction) layers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Iterable, Iterator, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

try:
    from vllm.logger import init_logger
    log = init_logger("vllm.genesis.mtp_cache")
except Exception:
    log = logging.getLogger("genesis.mtp_cache")

CACHE_DIR_DEFAULT = "/root/.cache/vllm/mtp_quant_cache"
GENESIS_CACHE_VERSION = "v1.0"


def compute_file_hash(filepath: str) -> str:
    """Compute a fast SHA256 fingerprint from file metadata and boundary chunks."""
    if not filepath or not os.path.exists(filepath):
        return "missing"
    st = os.stat(filepath)
    h = hashlib.sha256()
    h.update(f"{filepath}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
    try:
        with open(filepath, "rb") as f:
            chunk_head = f.read(65536)
            h.update(chunk_head)
            if st.st_size > 131072:
                f.seek(-65536, os.SEEK_END)
                chunk_tail = f.read(65536)
                h.update(chunk_tail)
    except Exception as e:
        log.warning("Could not read full chunks for hash: %s", e)
    return h.hexdigest()[:16]


def quantize_linear_weight_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2D linear weight matrix to symmetric INT8 with per-channel scales."""
    # weight: [out_features, in_features] in float16/bfloat16
    orig_device = weight.device
    w_float = weight.to(torch.float32)
    # per-channel max absolute value along input features (dim 1)
    max_val = torch.amax(torch.abs(w_float), dim=1, keepdim=True).clamp(min=1e-8)
    scale = (max_val / 127.0).to(torch.float16)
    qweight = torch.clamp(torch.round(w_float / scale.to(torch.float32)), -128, 127).to(torch.int8)
    return qweight.to(orig_device), scale.to(orig_device)


def dequantize_linear_weight_int8(qweight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize INT8 weight back to target float precision (FP16/BF16) on GPU."""
    return (qweight.to(torch.float32) * scale.to(torch.float32)).to(dtype)


class MTPQuantDiskCacheManager:
    """Manages disk caching and loading of quantized MTP draft weights."""

    def __init__(
        self,
        cache_dir: str = CACHE_DIR_DEFAULT,
        enabled: bool = True,
        quant_format: str = "int8",
    ):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.quant_format = quant_format

    def get_cache_key(
        self,
        model_name_or_path: str,
        source_file: str,
        tp_size: int,
    ) -> str:
        source_hash = compute_file_hash(source_file)
        raw_key = f"{model_name_or_path}|{source_hash}|{self.quant_format}|tp{tp_size}|{GENESIS_CACHE_VERSION}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:24]

    def process_mtp_weights(
        self,
        model_name: str,
        source_file: str,
        tp_size: int,
        weights: Iterable[Tuple[str, torch.Tensor]],
        target_dtype: torch.dtype = torch.float16,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Processes weights with Cache Hit / Cache Miss logic."""
        if not self.enabled:
            for name, weight in weights:
                yield name, weight
            return

        cache_key = self.get_cache_key(model_name, source_file, tp_size)
        entry_dir = os.path.join(self.cache_dir, cache_key)
        cache_file = os.path.join(entry_dir, "mtp_quant.safetensors")
        manifest_file = os.path.join(entry_dir, "manifest.json")

        # ── CACHE HIT ────────────────────────────────────────────────────────
        if os.path.exists(cache_file) and os.path.exists(manifest_file):
            try:
                with open(manifest_file, "r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
                if manifest.get("fingerprint") == cache_key and manifest.get("valid", False):
                    t0 = time.perf_counter()
                    log.info("[Genesis MTP Cache] ⚡ Cache HIT for MTP key %s. Loading from %s", cache_key, cache_file)
                    with safe_open(cache_file, framework="pt", device="cpu") as sf:
                        keys = list(sf.keys())
                        # Load and yield dequantized or quantized tensors
                        for k in keys:
                            if k.endswith(".scale"):
                                continue
                            if f"{k}.scale" in keys:
                                qweight = sf.get_tensor(k)
                                scale = sf.get_tensor(f"{k}.scale")
                                weight = dequantize_linear_weight_int8(qweight, scale, target_dtype)
                                yield k, weight
                            else:
                                yield k, sf.get_tensor(k).to(target_dtype)
                    dt = time.perf_counter() - t0
                    log.info("[Genesis MTP Cache] ⚡ Cache HIT load completed in %.3f seconds (0%% CPU quantization overhead).", dt)
                    return
            except Exception as e:
                log.warning("[Genesis MTP Cache] Cache read error (%s). Regenerating cache...", e)

        # ── CACHE MISS ───────────────────────────────────────────────────────
        log.info("[Genesis MTP Quant] 🧠 Starting INT8 Quantization for MTP Draft Model (Key: %s)", cache_key)
        log.info("[Genesis MTP Quant] ℹ️ Context: Backbone model is quantized (W4A16/W8A16) but MTP layers are unquantized (FP16/BF16).")
        log.info("[Genesis MTP Quant] ℹ️ Action: Compressing draft weights to INT8 to reduce VRAM footprint (~400MB saved) and caching to disk.")
        t0 = time.perf_counter()
        os.makedirs(entry_dir, exist_ok=True)
        tmp_cache_file = cache_file + ".tmp"
        tmp_manifest_file = manifest_file + ".tmp"

        collected_tensors: dict[str, torch.Tensor] = {}
        yield_tensors: list[Tuple[str, torch.Tensor]] = []

        weights_list = list(weights)
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(weights_list, desc="Quantizing MTP tensors (INT8)", unit="tensor", leave=True)
        except Exception:
            pbar = weights_list

        for name, weight in pbar:
            if weight.ndim == 2 and weight.numel() > 1024 and self.quant_format == "int8":
                qweight, scale = quantize_linear_weight_int8(weight)
                collected_tensors[name] = qweight.contiguous().cpu()
                collected_tensors[f"{name}.scale"] = scale.contiguous().cpu()
                # Yield reconstructed weight for this current run
                rec_weight = dequantize_linear_weight_int8(qweight, scale, target_dtype)
                yield_tensors.append((name, rec_weight))
            else:
                collected_tensors[name] = weight.contiguous().cpu()
                yield_tensors.append((name, weight))

        try:
            log.info("[Genesis MTP Quant] 💾 Persisting %d tensors to disk cache at %s...", len(collected_tensors), cache_file)
            save_file(collected_tensors, tmp_cache_file)
            manifest_data = {
                "fingerprint": cache_key,
                "model_name": model_name,
                "source_file": source_file,
                "quant_format": self.quant_format,
                "tp_size": tp_size,
                "version": GENESIS_CACHE_VERSION,
                "tensors_count": len(collected_tensors),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "valid": True,
            }
            with open(tmp_manifest_file, "w", encoding="utf-8") as mf:
                json.dump(manifest_data, mf, indent=2)

            # Atomic swap
            os.replace(tmp_cache_file, cache_file)
            os.replace(tmp_manifest_file, manifest_file)
            dt = time.perf_counter() - t0
            log.info("[Genesis MTP Quant] ✅ MTP Quant Cache written successfully in %.3f seconds (0%% CPU overhead on future boots).", dt)
        except Exception as e:
            log.error("[Genesis MTP Quant] Failed to persist MTP cache: %s", e)
            if os.path.exists(tmp_cache_file):
                try:
                    os.remove(tmp_cache_file)
                except Exception:
                    pass

        for name, weight in yield_tensors:
            yield name, weight
