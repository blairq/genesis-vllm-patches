#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""¿Cuanto cuesta que el mmap de offloading no este pinneado?

CONTEXTO
--------
El offloading de KV escribe en un mmap de /dev/shm compartido por todos los
workers (v1/kv_offload/cpu/shared_offload_region.py). vLLM intenta pinnearlo
con cudaHostRegister (gpu_worker.py:139), pero con TP>1 los dos ranks
registran LAS MISMAS paginas fisicas y el segundo falla:

    cudaHostRegister failed for rank=0 (code=1) -- transfers will still work
    but may be slower (unpinned DMA)

Asi que un rank transfiere pinneado y el otro no. Este test mide cuanto cuesta
eso, para saber si vale la pena arreglarlo antes de tocar nada.

Compara GPU -> CPU con tres destinos:
    1. pinned de verdad   (torch pin_memory=True)
    2. mmap de /dev/shm registrado con cudaHostRegister   <- rank que gana
    3. mmap de /dev/shm SIN registrar                     <- rank que pierde

    python3 pinned_vs_mmap.py [--mib 512] [--reps 20]
"""

from __future__ import annotations

import argparse
import mmap
import os
import sys

import torch


def medir(gpu, cpu, reps):
    for _ in range(3):
        cpu.copy_(gpu, non_blocking=True)
    torch.cuda.synchronize()
    ini = torch.cuda.Event(enable_timing=True)
    fin = torch.cuda.Event(enable_timing=True)
    ini.record()
    for _ in range(reps):
        cpu.copy_(gpu, non_blocking=True)
    fin.record()
    fin.synchronize()
    ms = ini.elapsed_time(fin) / reps
    return gpu.nbytes / (ms / 1000) / (1 << 30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=512)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    n = args.mib * (1 << 20)
    gpu = torch.empty(n, dtype=torch.int8, device="cuda:0")
    print(f"GPU -> CPU, {args.mib} MiB, {args.reps} repeticiones\n")

    pin = torch.empty(n, dtype=torch.int8, pin_memory=True)
    bw_pin = medir(gpu, pin, args.reps)
    del pin

    ruta = "/dev/shm/vllm_prueba_pin.mmap"
    fd = os.open(ruta, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, n)
        mm = mmap.mmap(fd, n, flags=mmap.MAP_SHARED,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE)
        mm.madvise(getattr(mmap, "MADV_POPULATE_WRITE", 23), 0, n)
        cpu = torch.frombuffer(memoryview(mm), dtype=torch.int8)

        bw_sin = medir(gpu, cpu, args.reps)

        rt = torch.cuda.cudart()
        r = rt.cudaHostRegister(cpu.data_ptr(), n, 0)
        registrado = r.value == 0
        bw_con = medir(gpu, cpu, args.reps) if registrado else None
        if registrado:
            rt.cudaHostUnregister(cpu.data_ptr())

        print(f"  1. pinned de torch                 {bw_pin:6.2f} GiB/s")
        if registrado:
            print(f"  2. mmap + cudaHostRegister         {bw_con:6.2f} GiB/s")
        else:
            print(f"  2. mmap + cudaHostRegister         FALLO (code={r.value})")
        print(f"  3. mmap SIN registrar              {bw_sin:6.2f} GiB/s")
        print()
        ref = bw_con if registrado else bw_pin
        print(f"  => no pinnear cuesta {100 * (1 - bw_sin / ref):.0f}% "
              f"({ref:.2f} -> {bw_sin:.2f} GiB/s)")
        del cpu
        mm.close()
    finally:
        os.close(fd)
        if os.path.exists(ruta):
            os.unlink(ruta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
