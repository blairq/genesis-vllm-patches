"""SK-24/i8: all-reduce P2P con parciales int8 por grupo de 64. Dos procesos, una placa cada uno.

1) las DOS placas dan exactamente lo mismo (el residuo de TP tiene que quedar identico);
2) el error contra el all-reduce exacto (fp16), y contra lo que da PN120 (la misma cuenta en torch);
3) tiempos en grafos: fp16 (SK-24) contra int8 (SK-24/i8), con 9, 18, 36 y 54 filas, y bloques 4/8/16.
"""
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 5120


def ref_i8(xs):
    """La cuenta de PN120 en fp32: escala fp16 = amax/127, q = redondeo, suma en orden de rango."""
    out = None
    for x in xs:
        v = x.float().view(x.shape[0], -1, 64)
        s = (v.abs().amax(-1).clamp_min(1e-6) / 127.0).half().float()
        q = torch.round(v / s[..., None]).clamp(-127, 127)
        t = q * s[..., None]
        out = t if out is None else out + t
    return out.view(xs[0].shape)


def trabajador(rank):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29532")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    cpu = dist.new_group(backend="gloo")
    import vllm._genesis.ar_p2p as ap
    ap._ACTIVO = True
    ap.inicializar(rank, 2, cpu)
    dev = f"cuda:{rank}"
    g = torch.Generator(device=dev); g.manual_seed(77 + rank)
    ok = True
    for M in (9, 18, 36, 54):
        dif_placas, err_exacto, err_ref = 0, 0.0, 0.0
        for it in range(200):
            x = (torch.randn(M, H, device=dev, generator=g) * 3).half()
            out = ap.all_reduce_i8(x)
            exacto = x.clone(); dist.all_reduce(exacto)
            todos = [torch.empty_like(x) for _ in range(2)]; dist.all_gather(todos, x)
            r = ref_i8(todos)
            otras = [torch.empty_like(out) for _ in range(2)]; dist.all_gather(otras, out)
            dif_placas += int(not torch.equal(otras[0].view(torch.int16), otras[1].view(torch.int16)))
            err_exacto = max(err_exacto, ((out.float() - exacto.float()).norm() / exacto.float().norm()).item())
            err_ref = max(err_ref, (out.float() - r).abs().max().item())
        torch.cuda.synchronize()
        ok &= dif_placas == 0
        if rank == 0:
            print(f"M={M:3d}: placas distintas {dif_placas}/200 | error rel. contra exacto {err_exacto:.2e} | "
                  f"max |dif| contra la cuenta de PN120 {err_ref:.2e}", flush=True)
    for M in (9, 18, 36, 54):
        linea = []
        for b in (4, 8, 16):
            ap._BLOQUES = b
            for nombre, f in (("fp16", ap.all_reduce), ("i8", ap.all_reduce_i8)):
                x = torch.randn(M, H, device=dev).half()
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3): f(x)
                torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); dist.barrier()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    for _ in range(100): f(x)
                for _ in range(5): gr.replay()
                torch.cuda.synchronize(); dist.barrier()
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for _ in range(20): gr.replay()
                e1.record(); torch.cuda.synchronize()
                linea.append(f"{nombre}/{b}b {e0.elapsed_time(e1) * 1000 / 2000:5.1f}")
        if rank == 0:
            print(f"tiempos M={M:3d} ({M * H * 2 // 1024} KB): " + "  ".join(linea), flush=True)
    if rank == 0:
        print("RESULTADO:", "OK (placas identicas)" if ok else "FALLA", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(trabajador, nprocs=2)
