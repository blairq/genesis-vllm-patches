#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verifica que la transferencia GPU->GPU sea DIRECTA, no rebotando por el host.

    docker run --rm --gpus all -v $PWD/tests:/tests \
      --entrypoint python3 vllm/vllm-openai:v0.27.1 /tests/repro/p2p_bandwidth.py

POR QUE NO ALCANZA CON `nvidia-smi topo -p2p r`
-----------------------------------------------
Ese comando dice lo que el driver DECLARA, no lo que hace. Puede decir OK y el
trafico igual rebotar por la RAM del host. Y `torch.Tensor.copy_` entre dos
GPUs funciona con o sin P2P: sin P2P el runtime hace el staging por host solo,
en silencio. Asi que medir una copia y ver un numero "alto" no prueba nada si
no hay contra que comparar.

LA DISCRIMINACION: EL CONTROL EXPLICITO
---------------------------------------
Este test mide DOS caminos en la misma corrida:

  directo:      gpu0 -> gpu1
  por host:     gpu0 -> cpu (pinned) -> gpu1     <- el control

Sin P2P los dos hacen lo mismo por debajo y dan parecido. Con P2P el directo
cruza el switch PCIe una sola vez y el de host cruza dos, asi que la razon
entre ambos se va cerca de 2x. Esa RAZON es la prueba, no el valor absoluto:
no depende de la generacion de PCIe ni del ancho del link.

Medido en este rig (2x RTX 3090, PCIe Gen4 x8, sin NVLink):

    sin P2P (CNS):   directo ~6,4 GiB/s    razon ~1,0x
    con P2P (OK):    directo 12,58 GiB/s   razon ~2x     <- techo Gen4 x8: 15,75 GB/s

Ver docs/P2P-SIN-PARCHAR-EL-DRIVER.md
"""

from __future__ import annotations

import argparse
import time

import torch


def cronometrar(fn, reps: int) -> float:
    """Devuelve segundos por repeticion, sincronizando de verdad."""
    for _ in range(3):  # calentar
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=256, help="tamaño de la transferencia")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--src", type=int, default=0)
    ap.add_argument("--dst", type=int, default=1)
    args = ap.parse_args()

    if torch.cuda.device_count() < 2:
        print("hacen falta 2 GPUs")
        return 1

    s, d = args.src, args.dst
    print(f"GPU{s}: {torch.cuda.get_device_name(s)}")
    print(f"GPU{d}: {torch.cuda.get_device_name(d)}")
    p_sd = torch.cuda.can_device_access_peer(s, d)
    p_ds = torch.cuda.can_device_access_peer(d, s)
    print(f"can_device_access_peer  {s}->{d}: {p_sd}   {d}->{s}: {p_ds}\n")

    n = args.mib * 1024 * 1024 // 2  # fp16
    bytes_ = n * 2
    gib = bytes_ / 2**30

    # patron reconocible, para poder verificar integridad al final
    a = torch.arange(n, dtype=torch.int32, device=f"cuda:{s}").to(torch.float16)
    b = torch.empty(n, dtype=torch.float16, device=f"cuda:{d}")
    host = torch.empty(n, dtype=torch.float16, device="cpu", pin_memory=True)

    t_dir = cronometrar(lambda: b.copy_(a), args.reps)

    def por_host():
        host.copy_(a, non_blocking=True)
        b.copy_(host, non_blocking=True)

    t_host = cronometrar(por_host, args.reps)

    print(f"transferencia de {args.mib} MiB, {args.reps} repeticiones\n")
    print(f"  directo   gpu{s} -> gpu{d}          {gib / t_dir:7.2f} GiB/s"
          f"   ({t_dir * 1000:.2f} ms)")
    print(f"  por host  gpu{s} -> cpu -> gpu{d}   {gib / t_host:7.2f} GiB/s"
          f"   ({t_host * 1000:.2f} ms)   <- control")
    razon = t_host / t_dir
    print(f"\n  razon host/directo: {razon:.2f}x")
    if razon >= 1.5:
        print("  => el directo NO pasa por el host: P2P ACTIVO")
    else:
        print("  => los dos caminos rinden igual: el 'directo' esta")
        print("     haciendo staging por host. P2P NO activo.")

    # ── integridad: el P2P tiene que mover los datos BIEN, no solo rapido ──
    b.zero_()
    b.copy_(a)
    torch.cuda.synchronize()
    ok = torch.equal(b.to(f"cuda:{s}"), a)
    print(f"\n  integridad de los datos: {'OK' if ok else '*** CORRUPCION ***'}")

    # ── latencia: transferencias chicas, donde manda el overhead ──
    chico_a = torch.zeros(1024, dtype=torch.float16, device=f"cuda:{s}")
    chico_b = torch.zeros(1024, dtype=torch.float16, device=f"cuda:{d}")
    t_lat = cronometrar(lambda: chico_b.copy_(chico_a), 200)
    print(f"  latencia (2 KiB): {t_lat * 1e6:.1f} us")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
