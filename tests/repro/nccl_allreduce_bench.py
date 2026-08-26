#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark de all-reduce por NCCL entre 2 GPUs, con verificacion de correccion.

Equivale a `all_reduce_perf` de nccl-tests pero sin compilar nada: usa el mismo
NCCL que carga vLLM (via torch.distributed), asi que mide exactamente el camino
de produccion.

POR QUE HACE FALTA ADEMAS DE p2p_ipc_allreduce.py
--------------------------------------------------
Ese test valida `CustomAllreduce` (el camino propio de vLLM, IPC + kernel
propio). Este valida NCCL, que vLLM usa igual para el resto de las colectivas y
para todo all-reduce que supere `max_size` del custom. Son dos pilas distintas
sobre el mismo hardware.

COMO DETECTA SI EL P2P SE USA DE VERDAD
---------------------------------------
Corre la misma medicion dos veces:

    NCCL_P2P_DISABLE=0   NCCL usa P2P si puede
    NCCL_P2P_DISABLE=1   NCCL forzado a SHM (staging por host)

Si el P2P funciona, la primera tiene que ser sensiblemente mas rapida. Si dan
igual, NCCL no lo esta aprovechando aunque el driver diga `topo -p2p: OK`.

    docker run --rm --gpus all --ipc=host --shm-size=2gb \
      -v $PWD/tests:/tests --entrypoint python3 vllm/vllm-openai:v0.27.1 \
      /tests/repro/nccl_allreduce_bench.py

busbw: ancho de banda de bus para ring all-reduce = algbw * 2*(n-1)/n.
Es la metrica de nccl-tests y la comparable contra el techo del enlace.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# tamaños en tokens de hidden=5120 fp16; 1600 tok = un chunk de prefill
TAMAÑOS = [
    ("4 KiB", 2 * 1024),
    ("1 MiB", 512 * 1024),
    ("15,6 MiB (chunk prefill)", 1600 * 5120),
    ("64 MiB", 32 * 1024 * 1024),
]


def worker(rank: int, world: int, reps: int, cola):
    try:
        torch.cuda.set_device(rank)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29597")
        dist.init_process_group("nccl", rank=rank, world_size=world)

        filas = []
        for etiqueta, n in TAMAÑOS:
            x = torch.full((n,), float(rank + 1), dtype=torch.float16,
                           device=f"cuda:{rank}")
            for _ in range(5):  # calentar (NCCL arma canales en la 1a llamada)
                dist.all_reduce(x.clone())
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(reps):
                y = x.clone()
                dist.all_reduce(y)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps

            # correccion: cada rank aporta (rank+1) -> suma = 1+2 = 3
            esperado = float(sum(r + 1 for r in range(world)))
            ok = bool((y == esperado).all().item())

            bytes_ = n * 2
            algbw = bytes_ / dt / 2**30
            busbw = algbw * 2 * (world - 1) / world
            filas.append((etiqueta, bytes_, dt, algbw, busbw, ok))

        if rank == 0:
            cola.put(("ok", filas))
        dist.destroy_process_group()
    except Exception:
        if rank == 0:
            cola.put(("error", traceback.format_exc()[-1200:]))
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    if torch.cuda.device_count() < 2:
        print("hacen falta 2 GPUs")
        return 1

    p2p = os.environ.get("NCCL_P2P_DISABLE", "0")
    print(f"NCCL all-reduce, 2 GPUs, {args.reps} repeticiones  "
          f"(NCCL_P2P_DISABLE={p2p})\n")

    ctx = mp.get_context("spawn")
    cola = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, 2, args.reps, cola)) for r in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=args.timeout)

    vivos = [p for p in ps if p.is_alive()]
    if vivos:
        for p in vivos:
            p.kill()
        print(f"*** SE COLGO *** (>{args.timeout}s)")
        return 1
    if cola.empty():
        print("terminaron sin reportar; revisar stderr")
        return 1

    estado, datos = cola.get()
    if estado != "ok":
        print(datos)
        return 1

    print(f"{'tamaño':<26}{'ms':>9}{'algbw':>11}{'busbw':>11}   correcto")
    todo_ok = True
    for etiqueta, bytes_, dt, algbw, busbw, ok in datos:
        todo_ok &= ok
        print(f"{etiqueta:<26}{dt * 1000:>9.3f}{algbw:>9.2f} GiB/s"
              f"{busbw:>9.2f} GiB/s   {'OK' if ok else '*** MAL ***'}")
    if not todo_ok:
        print("\n*** HAY RESULTADOS NUMERICOS INCORRECTOS: no usar esta config ***")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
