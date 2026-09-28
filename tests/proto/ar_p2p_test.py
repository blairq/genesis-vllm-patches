"""SK-24 (PN152, all-reduce P2P de decode) contra NCCL, con las dos placas. Un proceso por GPU.

1) exactitud bit a bit contra dist.all_reduce (NCCL) con M = 1, 9, 36, 54 filas de 5120 fp16, 300 iteraciones
   seguidas (ejercita la paridad de epoca y el doble buffer), en eager y grabado en un grafo CUDA;
2) tiempos dentro de grafos CUDA (100 all-reduce por grafo): SK-24 contra NCCL.

  docker run --rm --gpus all --ipc=host --entrypoint python3 -v $PWD/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis \
     -v $PWD/tests/proto:/t vllm/vllm-openai:v0.29.0 /t/ar_p2p_test.py
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 5120


def trabajador(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29531", GENESIS_ENABLE_PN152_AR_P2P="1")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    cpu = dist.new_group(backend="gloo")
    import vllm._genesis.ar_p2p as ap
    ap._ACTIVO = True
    ap.inicializar(rank, 2, cpu)
    dev = f"cuda:{rank}"
    g = torch.Generator(device=dev); g.manual_seed(1234 + rank)
    ok = True
    # 1) exactitud en eager
    for M in (1, 9, 36, 54):
        malos = 0
        for it in range(300):
            x = (torch.randn(M, H, device=dev, generator=g) * 4).half()
            ref = x.clone(); dist.all_reduce(ref)
            out = ap.all_reduce(x)
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                malos += 1
        torch.cuda.synchronize()
        ok &= malos == 0
        if rank == 0:
            print(f"eager M={M:3d}: {300 - malos}/300 identicos a NCCL", flush=True)
    # 1b) grabado en grafo: el mismo lanzamiento, datos nuevos en cada replay
    for M in (9, 36):
        xs = torch.zeros(M, H, device=dev, dtype=torch.half)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            ap.all_reduce(xs)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            os_ = ap.all_reduce(xs)
        malos = 0
        for it in range(200):
            xs.copy_((torch.randn(M, H, device=dev, generator=g) * 4).half())
            ref = xs.clone(); dist.all_reduce(ref)
            gr.replay()
            if not torch.equal(os_.view(torch.int16), ref.view(torch.int16)):
                malos += 1
        torch.cuda.synchronize()
        ok &= malos == 0
        if rank == 0:
            print(f"grafo M={M:3d}: {200 - malos}/200 identicos a NCCL", flush=True)
    # 2) tiempos en grafos: 100 all-reduce seguidos
    for M in (9, 36, 54):
        res = {}
        for nombre in ("nccl", "sk24"):
            x = torch.randn(M, H, device=dev).half()
            f = (lambda: dist.all_reduce(x)) if nombre == "nccl" else (lambda: ap.all_reduce(x))
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): f()
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(100): f()
            for _ in range(5): gr.replay()
            torch.cuda.synchronize(); dist.barrier()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(20): gr.replay()
            e1.record(); torch.cuda.synchronize()
            res[nombre] = e0.elapsed_time(e1) * 1000 / 2000
        if rank == 0:
            kb = M * H * 2 / 1024
            print(f"M={M:3d} ({kb:5.0f} KB): NCCL {res['nccl']:6.2f} us | SK-24 {res['sk24']:6.2f} us | x{res['nccl'] / res['sk24']:.2f}", flush=True)
    # 3) barrido de bloques de SK-24 (la grilla tiene que caber entera: <= 82)
    for M in (9, 36, 54):
        linea = []
        for b in (4, 8, 16, 24, 32, 48):
            ap._BLOQUES = b
            x = torch.randn(M, H, device=dev).half()
            f = lambda: ap.all_reduce(x)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): f()
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(100): f()
            for _ in range(5): gr.replay()
            torch.cuda.synchronize(); dist.barrier()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(20): gr.replay()
            e1.record(); torch.cuda.synchronize()
            linea.append(f"{b}b {e0.elapsed_time(e1) * 1000 / 2000:5.1f}")
        if rank == 0:
            print(f"barrido M={M:3d}: " + "  ".join(linea), flush=True)
    if rank == 0:
        print("RESULTADO:", "OK, bit a bit" if ok else "FALLA", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(trabajador, nprocs=2)
