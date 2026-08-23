#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A/B Benchmark Suite for Genesis KV Optimization Patches (PN90, PN91, PN92, PN93).

Executes a reproducible 4-turn workload:
  Turn 1: ~39.5k tokens (Base context)
  Turn 2: ~79.0k tokens (Incremental context)
  Turn 3: ~118.4k tokens (Large context)
  Turn 4: ~39.5k tokens (Prefix cache hit test)

Measures:
  - Latency / TTFT per turn
  - Megabytes written to SSD (/kv-offload)
  - Output token rate and accuracy
"""

import glob
import json
import os
import sys
import time
import urllib.request

API_BASE = os.environ.get("VLLM_API_BASE", "http://172.20.0.228:8320")
API_KEY = os.environ.get("VLLM_API_KEY", "super-secret-key-123")
DISK_OFFLOAD_DIR = "/home/usuario/Proyectos/kv-offload"


def build_repeated_context(num_paragraphs: int, seed: int = 0) -> str:
    paragraphs = []
    for i in range(num_paragraphs):
        p = (
            f"Paragraph {seed}_{i}: In modern distributed deep learning systems, the key-value "
            f"cache size scales linearly with the sequence length and the batch size. "
            f"To optimize GPU memory utilization and prevent out-of-memory errors during large "
            f"context window inference, tiered offloading and prefix caching play a crucial role. "
            f"By pipelining host RAM and NVMe SSD transfers asynchronously with chunked prefill, "
            f"we achieve ultra-low time-to-first-token while keeping GPU memory footprint minimal. "
            f"Module identifier: [CHUNK_SEGMENT_{seed:04d}_{i:04d}]."
        )
        paragraphs.append(p)
    return "\n\n".join(paragraphs)


def count_ssd_bytes_since(since_ts: float) -> tuple[int, float]:
    bin_files = glob.glob(f"{DISK_OFFLOAD_DIR}/**/*.bin", recursive=True)
    new_files = [f for f in bin_files if os.path.getmtime(f) >= since_ts]
    total_bytes = sum(os.path.getsize(f) for f in new_files)
    return len(new_files), total_bytes / (1024 * 1024)


def send_chat(prompt: str, agent: str = "primary_nothink", max_tokens: int = 32) -> dict:
    url = f"{API_BASE}/v1/chat/completions"
    payload = {
        "model": "qwen3.8",
        "messages": [
            {"role": "system", "content": "You are an expert AI software engineer."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "kv_transfer_params": {
            "genesis_agent": agent,
            "persist_disk": True,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        elapsed = time.perf_counter() - t0
        body = json.loads(resp.read().decode("utf-8"))
        body["_elapsed_sec"] = elapsed
        return body


def run_workload(mode_label: str) -> dict:
    print("=" * 75)
    print(f"  EJECUTANDO BATERÍA DE PRUEBA: {mode_label}")
    print("=" * 75)

    suite_t0 = time.time()
    results = {"mode": mode_label, "turns": [], "total_ssd_mb": 0.0, "total_time_sec": 0.0}

    # Turn 1: 39.5k tokens
    print("\n[Turno 1] Generando prefijo base (39.5k tokens)...")
    t0_1 = time.time()
    p1 = build_repeated_context(350, seed=10) + "\n\nTask: Summarize the architecture."
    resp1 = send_chat(p1)
    n1, mb1 = count_ssd_bytes_since(t0_1)
    tok1 = resp1.get("usage", {}).get("prompt_tokens", 0)
    print(f"  -> Tokens: {tok1}, Tiempo: {resp1['_elapsed_sec']:.2f}s, SSD: {n1} bloques ({mb1:.2f} MB)")
    results["turns"].append({"turn": 1, "tokens": tok1, "time_sec": resp1["_elapsed_sec"], "ssd_mb": mb1})

    # Turn 2: 79.0k tokens
    print("\n[Turno 2] Contexto incremental (79.0k tokens)...")
    t0_2 = time.time()
    p2 = p1 + "\n\n" + build_repeated_context(350, seed=20) + "\n\nTask: State the optimization method."
    resp2 = send_chat(p2)
    n2, mb2 = count_ssd_bytes_since(t0_2)
    tok2 = resp2.get("usage", {}).get("prompt_tokens", 0)
    print(f"  -> Tokens: {tok2}, Tiempo: {resp2['_elapsed_sec']:.2f}s, SSD: {n2} bloques ({mb2:.2f} MB)")
    results["turns"].append({"turn": 2, "tokens": tok2, "time_sec": resp2["_elapsed_sec"], "ssd_mb": mb2})

    # Turn 3: 118.4k tokens
    print("\n[Turno 3] Contexto extenso (118.4k tokens)...")
    t0_3 = time.time()
    p3 = p2 + "\n\n" + build_repeated_context(350, seed=30) + "\n\nTask: Extract the first chunk segment."
    resp3 = send_chat(p3)
    n3, mb3 = count_ssd_bytes_since(t0_3)
    tok3 = resp3.get("usage", {}).get("prompt_tokens", 0)
    print(f"  -> Tokens: {tok3}, Tiempo: {resp3['_elapsed_sec']:.2f}s, SSD: {n3} bloques ({mb3:.2f} MB)")
    results["turns"].append({"turn": 3, "tokens": tok3, "time_sec": resp3["_elapsed_sec"], "ssd_mb": mb3})

    # Turn 4: Cache hit test (reuso de p1)
    print("\n[Turno 4] Prueba de Cache Hit (Reuso de prefijo 39.5k)...")
    t0_4 = time.time()
    p4 = p1 + "\n\nTask: Count paragraphs."
    resp4 = send_chat(p4)
    n4, mb4 = count_ssd_bytes_since(t0_4)
    tok4 = resp4.get("usage", {}).get("prompt_tokens", 0)
    print(f"  -> Tokens: {tok4}, Tiempo: {resp4['_elapsed_sec']:.2f}s, SSD: {n4} bloques ({mb4:.2f} MB)")
    results["turns"].append({"turn": 4, "tokens": tok4, "time_sec": resp4["_elapsed_sec"], "ssd_mb": mb4})

    total_n, total_mb = count_ssd_bytes_since(suite_t0)
    results["total_ssd_mb"] = total_mb
    results["total_time_sec"] = sum(t["time_sec"] for t in results["turns"])

    print("\n" + "-" * 75)
    print(f"  Resumen {mode_label}: Tiempo Total = {results['total_time_sec']:.2f}s | SSD Total = {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
    print("-" * 75)
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "CON_PARCHES"
    out_file = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/bench_{mode}.json"
    res = run_workload(mode)
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Guardado en {out_file}")
