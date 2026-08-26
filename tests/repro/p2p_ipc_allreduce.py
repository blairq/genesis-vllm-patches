#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""¿Sirve el P2P por mailbox para el all-reduce propio de vLLM (sin NCCL)?

vLLM tiene DOS caminos de all-reduce:

  NCCL              registra buffers ARBITRARIOS del usuario para P2P.
                    Con ForceP2P=529 y BAR1 chico, se cuelga en el init
                    (medido: +10 min parado en pynccl.py:113).
  CustomAllreduce   asigna UN buffer FIJO (max_size, 8 MiB por defecto), lo
                    comparte por handle IPC de CUDA y escribe directo en el
                    del vecino con su propio kernel. NO usa NCCL.

La diferencia que importa: el buffer del custom es chico y fijo. Con 256 MB de
BAR1 en la placa limitada, mapear 8-32 MiB por IPC deberia entrar aunque NCCL
no pueda registrar buffers grandes.

Este test instancia la clase REAL de vLLM en dos procesos (uno por GPU, como
hace el engine) y le pide un all-reduce de verdad, verificando el resultado
numerico. No reimplementa nada: si esto anda, el camino de produccion anda.

    docker run --rm --gpus all --ipc=host --shm-size=2gb \\
      -v $PWD/tests:/tests --entrypoint python3 vllm/vllm-openai:v0.27.1 \\
      /tests/repro/p2p_ipc_allreduce.py --mib 32

REQUIERE ForceP2P=529 en /etc/modprobe.d y los modulos recargados.
Ver docs/P2P-SIN-PARCHAR-EL-DRIVER.md
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world: int, mib: int, tokens: int, cola):
    try:
        torch.cuda.set_device(rank)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29593")
        # gloo: CustomAllreduce EXIGE un grupo que no sea NCCL
        dist.init_process_group("gloo", rank=rank, world_size=world)
        grupo = dist.group.WORLD

        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )

        ca = CustomAllreduce(
            group=grupo, device=torch.device(f"cuda:{rank}"), max_size=mib * 1024 * 1024
        )
        if ca.disabled:
            if rank == 0:
                cola.put(("error", "CustomAllreduce quedo DESHABILITADO "
                                   "(mirar el log: p2p check, world size, o lib faltante)"))
            dist.destroy_process_group()
            return

        # tamaño real de un all-reduce del engine:
        #   tokens x hidden(5120) x 2 bytes
        n = tokens * 5120
        x = torch.full((n,), float(rank + 1), dtype=torch.float16,
                       device=f"cuda:{rank}")
        bytes_ = x.numel() * x.element_size()

        usa_custom = ca.should_custom_ar(x)
        out = ca.custom_all_reduce(x)

        if out is None:
            if rank == 0:
                cola.put(("error",
                          f"custom_all_reduce devolvio None para {bytes_/2**20:.1f} MiB "
                          f"(should_custom_ar={usa_custom}, max_size="
                          f"{ca.max_size/2**20:.0f} MiB) -> caeria a NCCL"))
            dist.destroy_process_group()
            return

        torch.cuda.synchronize()
        esperado = float(sum(r + 1 for r in range(world)))  # 1 + 2 = 3
        ok = bool((out == esperado).all().item())
        obtenido = float(out[0].item())

        if rank == 0:
            cola.put((
                "ok",
                f"  tensor: {tokens} tokens x 5120 = {bytes_/2**20:.1f} MiB\n"
                f"  max_size del comunicador: {ca.max_size/2**20:.0f} MiB\n"
                f"  should_custom_ar: {usa_custom}\n"
                f"  resultado: {obtenido} (esperado {esperado})\n"
                f"  correcto: {ok}",
            ))
        dist.destroy_process_group()
    except Exception:
        if rank == 0:
            cola.put(("error", traceback.format_exc()[-1500:]))
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=32,
                    help="max_size del comunicador (vLLM usa 8 por defecto)")
    ap.add_argument("--tokens", type=int, default=1600,
                    help="tokens del all-reduce; 1600 = un chunk de prefill = 16,4 MiB")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    if torch.cuda.device_count() < 2:
        print("hacen falta 2 GPUs")
        return 1

    print(f"CustomAllreduce con max_size={args.mib} MiB, "
          f"all-reduce de {args.tokens} tokens\n")

    ctx = mp.get_context("spawn")
    cola = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, 2, args.mib, args.tokens, cola))
          for r in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=args.timeout)

    vivos = [p for p in ps if p.is_alive()]
    if vivos:
        for p in vivos:
            p.terminate()
        print(f"*** SE COLGO *** (>{args.timeout}s)")
        print("    Mismo sintoma que NCCL: el P2P por mailbox tampoco sirve para IPC.")
        return 1

    if cola.empty():
        print("los procesos terminaron sin reportar; revisar stderr")
        return 1
    estado, msg = cola.get()
    print(msg)
    if estado == "ok":
        print("\n=> el camino CustomAllreduce (sin NCCL) es VIABLE")
    return 0 if estado == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
