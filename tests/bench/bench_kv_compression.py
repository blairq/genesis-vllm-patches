#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark compression ratios and throughput on real KV cache blocks."""

import glob
import os
import time
import zlib

try:
    import zstandard as zstd
except ImportError:
    zstd = None

try:
    import lz4.frame as lz4_frame
except ImportError:
    lz4_frame = None


def main():
    bin_files = glob.glob("/home/usuario/Proyectos/kv-offload/**/*.bin", recursive=True)
    if not bin_files:
        print("No .bin files found in /home/usuario/Proyectos/kv-offload")
        return

    sample_files = bin_files[:5]
    print(f"Probando compresión sobre {len(sample_files)} bloques reales de KV cache (27.6 MB cada uno)...\n")

    algorithms = []
    
    # 1. Zlib / Deflate
    algorithms.append(("zlib (lvl 1)", lambda data: zlib.compress(data, level=1), lambda comp: zlib.decompress(comp)))
    algorithms.append(("zlib (lvl 6)", lambda data: zlib.compress(data, level=6), lambda comp: zlib.decompress(comp)))

    # 2. Zstandard
    if zstd is not None:
        cctx_1 = zstd.ZstdCompressor(level=1)
        dctx = zstd.ZstdDecompressor()
        algorithms.append(("zstd (lvl 1 - fast)", lambda data: cctx_1.compress(data), lambda comp: dctx.decompress(comp)))
        cctx_3 = zstd.ZstdCompressor(level=3)
        algorithms.append(("zstd (lvl 3 - default)", lambda data: cctx_3.compress(data), lambda comp: dctx.decompress(comp)))

    # 3. LZ4
    if lz4_frame is not None:
        algorithms.append(("lz4 (ultra fast)", lambda data: lz4_frame.compress(data), lambda comp: lz4_frame.decompress(comp)))

    results = {}
    for name, comp_fn, decomp_fn in algorithms:
        results[name] = {
            "orig_bytes": 0,
            "comp_bytes": 0,
            "comp_time": 0.0,
            "decomp_time": 0.0,
        }

    for fpath in sample_files:
        with open(fpath, "rb") as f:
            raw_data = f.read()

        for name, comp_fn, decomp_fn in algorithms:
            # Measure compression
            t0 = time.perf_counter()
            compressed = comp_fn(raw_data)
            t_comp = time.perf_counter() - t0

            # Measure decompression
            t1 = time.perf_counter()
            decompressed = decomp_fn(compressed)
            t_decomp = time.perf_counter() - t1

            assert len(decompressed) == len(raw_data), f"Mismatch in {name}"

            res = results[name]
            res["orig_bytes"] += len(raw_data)
            res["comp_bytes"] += len(compressed)
            res["comp_time"] += t_comp
            res["decomp_time"] += t_decomp

    print("=" * 88)
    print(f"{'Algoritmo':<22} | {'Tamaño Orig':<11} | {'Comprimido':<11} | {'Ratio':<7} | {'Comp Speed':<12} | {'Decomp Speed':<12}")
    print("=" * 88)

    for name, res in results.items():
        orig_mb = res["orig_bytes"] / (1024 * 1024)
        comp_mb = res["comp_bytes"] / (1024 * 1024)
        ratio = (1 - (res["comp_bytes"] / res["orig_bytes"])) * 100
        comp_speed = orig_mb / res["comp_time"] if res["comp_time"] > 0 else 0
        decomp_speed = orig_mb / res["decomp_time"] if res["decomp_time"] > 0 else 0

        print(f"{name:<22} | {orig_mb:>8.2f} MB | {comp_mb:>8.2f} MB | {ratio:>5.1f}% | {comp_speed:>9.1f} MB/s | {decomp_speed:>9.1f} MB/s")

    print("=" * 88)


if __name__ == "__main__":
    main()
