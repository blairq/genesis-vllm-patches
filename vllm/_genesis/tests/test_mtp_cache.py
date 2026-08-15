# SPDX-License-Identifier: Apache-2.0
"""Test suite for Genesis MTP Quantization Disk Cache."""
import os
import shutil
import tempfile
import time
import torch

from vllm._genesis.mtp_cache import (
    MTPQuantDiskCacheManager,
    compute_file_hash,
    quantize_linear_weight_int8,
    dequantize_linear_weight_int8,
)


def test_mtp_cache_roundtrip():
    temp_dir = tempfile.mkdtemp()
    try:
        # Create dummy weights
        dummy_weights = [
            ("model.fc.weight", torch.randn(5120, 10240, dtype=torch.float16)),
            ("model.layers.0.mlp.down_proj.weight", torch.randn(5120, 17408, dtype=torch.float16)),
            ("model.norm.weight", torch.randn(5120, dtype=torch.float16)),
        ]

        mgr = MTPQuantDiskCacheManager(cache_dir=temp_dir, enabled=True, quant_format="int8")
        
        # 1. First run (Cache MISS)
        t0 = time.perf_counter()
        loaded_1 = list(mgr.process_mtp_weights(
            model_name="test-qwen-27b",
            source_file=os.path.join(temp_dir, "dummy.safetensors"),
            tp_size=2,
            weights=dummy_weights,
            target_dtype=torch.float16,
        ))
        t_miss = time.perf_counter() - t0
        assert len(loaded_1) == 3, f"Expected 3 weights, got {len(loaded_1)}"

        # 2. Second run (Cache HIT)
        t0 = time.perf_counter()
        loaded_2 = list(mgr.process_mtp_weights(
            model_name="test-qwen-27b",
            source_file=os.path.join(temp_dir, "dummy.safetensors"),
            tp_size=2,
            weights=dummy_weights,
            target_dtype=torch.float16,
        ))
        t_hit = time.perf_counter() - t0
        assert len(loaded_2) == 3, f"Expected 3 weights on cache hit, got {len(loaded_2)}"
        
        print(f"Test passed! Miss time: {t_miss:.4f}s | Hit time: {t_hit:.4f}s (Speedup: {t_miss/max(t_hit, 1e-6):.1f}x)")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_mtp_cache_roundtrip()
