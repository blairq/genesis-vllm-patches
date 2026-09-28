"""SK-25 (PN154): la operacion por cabeza + Hadamard + int8 por token, contra la referencia en torch.

  docker run --rm --gpus '"device=0"' --entrypoint python3 -v $PWD/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis \
     -v $PWD/tests/proto:/t vllm/vllm-openai:v0.29.0 /t/sk25_test.py
"""
import torch
from vllm._genesis import had_salidas as hs

torch.manual_seed(0)
dev = "cuda"


def had(b):
    return hs.hadamard(b, torch.float32).to(dev)


def cuant(y):
    am = y.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    return torch.round(y * (127.0 / am)).to(torch.int8), am / 127.0


def ref(x, a, w, modo, d, eps):
    T = x.shape[0]
    xf, af = x.float().view(T, -1, d), a.float().view(T, -1, d)
    if modo == 0:
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float() * torch.nn.functional.silu(af)
    else:
        y = xf * torch.sigmoid(af)
    y = y.half().float()
    y = (y @ had(d)).reshape(T, -1)
    return cuant(y)


ok = True
for modo, d, nh, wdt in ((0, 128, 24, torch.float16), (0, 128, 24, torch.float32), (1, 256, 12, torch.float16)):
    for T in (1, 9, 36, 300):
        x = (torch.randn(T, nh * d, device=dev) * 3).half()
        a_ = (torch.randn(T, nh * d + 64, device=dev) * 2).half()[:, :nh * d]    # con paso de fila
        w = (1 + 0.2 * torch.randn(d, device=dev)).to(wdt)
        g = torch.tensor([0.37], device=dev)
        q, esc = hs.cabeza_q8_cuda(x, a_, w, g, modo, d, 1e-6)
        qr, er = ref(x, a_, w, modo, d, 1e-6)
        dif = (q.int() - qr.int()).abs()
        rel = float(((esc / 0.37) - er).abs().max() / er.abs().max())
        # la salida reconstruida (lo que importa): error relativo contra la referencia fp32 sin cuantizar
        print(f"modo {modo} d {d} nh {nh} w {str(wdt)[6:]} T {T:3d}: int8 distintos {int((dif > 0).sum())}/{dif.numel()} "
              f"(max {int(dif.max())}), escala rel {rel:.1e}")
        ok &= int(dif.max()) <= 1 and float((dif > 0).float().mean()) < 0.01 and rel < 1e-3

# tiempo (grafo de 100 llamadas) contra el camino de torch que reemplaza
for modo, d, nh in ((0, 128, 24), (1, 256, 12)):
    for T in (9, 36):
        x = torch.randn(T, nh * d, device=dev).half(); a_ = torch.randn(T, nh * d, device=dev).half()
        w = torch.ones(d, device=dev).half(); g = torch.tensor([1.0], device=dev)
        f = lambda: hs.cabeza_q8_cuda(x, a_, w, g, modo, d, 1e-6)
        for _ in range(3): f()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            f()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(100): f()
        gr.replay(); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(10): gr.replay()
        e1.record(); torch.cuda.synchronize()
        print(f"tiempo modo {modo} T {T}: {e0.elapsed_time(e1) * 1000 / 1000:.2f} us")
print("RESULTADO:", "OK" if ok else "FALLA")
