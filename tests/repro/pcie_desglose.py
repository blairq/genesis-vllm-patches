#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""¿Por que medimos 10,9 GiB/s si el enlace es de 15,75 GB/s?

Descompone el hueco en pedazos MEDIDOS en vez de estimarlos. El all-reduce de
NCCL mezcla tres cosas distintas y hay que separarlas:

    velocidad de linea      lo que dice el spec
      - protocolo PCIe      headers, CRC, framing, DLLPs  (calculable)
      = techo practico
      - overhead de NCCL    kernel, algoritmo, sincronizacion  (MEDIBLE:
                            copia P2P cruda vs all-reduce)
      = lo que vemos

La copia cruda se hace con cudaMemcpyPeerAsync por ctypes, NO con
torch.copy_: torch hace su propio staging y su numero no representa el
enlace (por eso tests/repro/p2p_bandwidth.py da razon 1,00x aunque el P2P
ande).

    python3 pcie_desglose.py [--mib 256] [--reps 20]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Gen4 x8: 8 carriles x 16 GT/s = 128 Gb/s de simbolos; el codigo 128b/130b
# deja pasar 128 de cada 130 bits.
LINEA_GBS = 8 * 16e9 * (128 / 130) / 8 / 1e9  # GB/s decimales
LINEA_GIBS = LINEA_GBS * 1e9 / (1 << 30)

# Desglose de un memory write TLP con payload de 256 B y direccion de 64 bits.
# (ECRC verificado APAGADO en este equipo: ECRCGenEn- en los cuatro extremos.)
TLP = {
    "STP (framing)": 4,
    "n. de secuencia": 2,
    "header (64 bits)": 16,
    "LCRC": 4,
}
MPS = 256


def worker(rank, n, reps, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group("nccl", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    buf = torch.empty(n, dtype=torch.uint8, device=f"cuda:{rank}")
    otro = 1 - rank

    def uni():
        if rank == 0:
            dist.send(buf, 1)
        else:
            dist.recv(buf, 0)

    def bidi():
        ops = [dist.P2POp(dist.isend, buf, otro),
               dist.P2POp(dist.irecv, torch.empty_like(buf), otro)]
        for r in dist.batch_isend_irecv(ops):
            r.wait()

    res = {}
    for nombre, fn, factor in (("uni", uni, 1), ("bidi", bidi, 2)):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        ini = torch.cuda.Event(enable_timing=True)
        fin = torch.cuda.Event(enable_timing=True)
        ini.record()
        for _ in range(reps):
            fn()
        fin.record()
        fin.synchronize()
        ms = ini.elapsed_time(fin) / reps
        res[nombre] = factor * n / (ms / 1000) / (1 << 30)
    if rank == 0:
        q.put(res)
    dist.destroy_process_group()


def cudart():
    for nombre in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
        try:
            return ctypes.CDLL(nombre)
        except OSError:
            continue
    raise RuntimeError("no encuentro libcudart")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=256)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    if torch.cuda.device_count() < 2:
        print("hacen falta 2 GPUs")
        return 1

    rt = cudart()
    rt.cudaMemcpyPeerAsync.restype = ctypes.c_int
    rt.cudaMemcpyPeerAsync.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_void_p,
    ]

    n = args.mib * (1 << 20)
    a = torch.empty(n, dtype=torch.uint8, device="cuda:0")
    b = torch.empty(n, dtype=torch.uint8, device="cuda:1")
    a.random_(0, 256)
    b.zero_()

    s0 = torch.cuda.Stream(device=0)
    s1 = torch.cuda.Stream(device=1)

    def copiar(dst, ddev, src, sdev, stream):
        err = rt.cudaMemcpyPeerAsync(
            ctypes.c_void_p(dst.data_ptr()), ddev,
            ctypes.c_void_p(src.data_ptr()), sdev,
            ctypes.c_size_t(n), ctypes.c_void_p(stream.cuda_stream))
        if err != 0:
            raise RuntimeError(f"cudaMemcpyPeerAsync fallo: {err}")

    def medir(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        ini = torch.cuda.Event(enable_timing=True)
        fin = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(0):
            ini.record()
            for _ in range(args.reps):
                fn()
            torch.cuda.synchronize(0); torch.cuda.synchronize(1)
            fin.record()
            fin.synchronize()
        return ini.elapsed_time(fin) / args.reps

    print(f"buffer de {args.mib} MiB, {args.reps} repeticiones\n")

    ms = medir(lambda: copiar(b, 1, a, 0, s0))
    uni01 = n / (ms / 1000) / (1 << 30)
    ms = medir(lambda: copiar(a, 0, b, 1, s1))
    uni10 = n / (ms / 1000) / (1 << 30)

    # Integridad: despues de 0->1 y 1->0 los dos buffers tienen que coincidir.
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    integridad = torch.equal(a.cpu(), b.cpu())

    def ambos():
        copiar(b, 1, a, 0, s0)
        copiar(a, 0, b, 1, s1)

    ms = medir(ambos)
    bidi = 2 * n / (ms / 1000) / (1 << 30)

    print(f"  copia cruda 0 -> 1        {uni01:6.2f} GiB/s")
    print(f"  copia cruda 1 -> 0        {uni10:6.2f} GiB/s")
    print(f"  las dos a la vez (suma)   {bidi:6.2f} GiB/s")
    print(f"  integridad de los datos:  {'OK' if integridad else '*** MAL ***'}")

    crudo = max(uni01, uni10)

    # --- NCCL punto a punto: el MISMO enlace pero por el camino de los SM ---
    # cudaMemcpyPeer usa el motor de copia (DMA) de la placa. NCCL no: sus
    # kernels escriben directo en la ventana BAR1 del vecino desde los SM. Son
    # dos motores distintos sobre el mismo cable, y hay que medir los dos para
    # saber cual limita.
    del a, b
    torch.cuda.empty_cache()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, n, args.reps, q)) for r in (0, 1)]
    for p in procs:
        p.start()
    nccl = q.get(timeout=300)
    for p in procs:
        p.join(timeout=60)

    print(f"\n  NCCL p2p (SM -> BAR1) uni  {nccl['uni']:6.2f} GiB/s")
    print(f"  NCCL p2p bidi (suma)       {nccl['bidi']:6.2f} GiB/s")

    # --- desglose ---
    total_tlp = MPS + sum(TLP.values())
    ef_tlp = MPS / total_tlp

    print(f"\n{'=' * 62}\nDESGLOSE DEL HUECO (unidireccional)\n{'=' * 62}")
    print(f"  velocidad de linea Gen4 x8      {LINEA_GIBS:6.2f} GiB/s"
          f"   ({LINEA_GBS:.2f} GB/s)")
    print(f"    (8 carriles x 16 GT/s, ya descontado el codigo 128b/130b)")
    print(f"\n  overhead de TLP con MPS={MPS} B:")
    for k, v in TLP.items():
        print(f"      {k:<22}{v:>4} B")
    print(f"      {'payload':<22}{MPS:>4} B")
    print(f"      {'-' * 26}")
    print(f"      {'total en el cable':<22}{total_tlp:>4} B"
          f"   -> eficiencia {100 * ef_tlp:.1f}%")
    techo = LINEA_GIBS * ef_tlp
    print(f"\n  techo tras el overhead de TLP   {techo:6.2f} GiB/s   "
          f"(100%, por sentido)")
    print(f"  motor de copia (cudaMemcpyPeer) {crudo:6.2f} GiB/s   "
          f"({100 * crudo / techo:3.0f}%)")
    print(f"  NCCL p2p, SM -> BAR1            {nccl['uni']:6.2f} GiB/s   "
          f"({100 * nccl['uni'] / techo:3.0f}%)")
    print()
    if nccl["uni"] > crudo * 1.2:
        print("  => El motor de COPIA es el que se queda corto, no el enlace.")
        print("     cudaMemcpyPeer usa el DMA de la placa; NCCL escribe desde")
        print("     los SM directo a la BAR1 del vecino y le saca")
        print(f"     {nccl['uni'] / crudo:.1f}x. Cualquier medicion de 'ancho de banda P2P'")
        print("     hecha con memcpy subestima el enlace.")
    else:
        print("  => Los dos motores dan parecido: el limite es el enlace.")
    print()
    print(f"  lo que queda hasta el techo ({techo - nccl['uni']:.2f} GiB/s) son DLLPs de")
    print("  ACK y de credito, SKP ordered sets, y el turnaround del enlace.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
