"""Proyeccion de la conv del borrador (kernel_projection, 5120 -> 1280, fp16 replicada): error y tiempo de variantes.
Pesos reales del checkpoint; entrada = randn * peso de la norma que la precede (magnitudes de la salida de RMSNorm)."""
import torch, re
from safetensors import safe_open
torch.zeros(1, device="cuda")
from vllm._genesis import borrador_marlin as g157
P = "/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86_dflash2/model.safetensors"
f = safe_open(P, "pt", device="cuda")
kps = sorted(k for k in f.keys() if "kernel_projection" in k)
print(len(kps), kps[:2])


def t_grafo(fn, it=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(5): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / (5 * it)


res = {}
for i, k in enumerate(kps):
    w = f.get_tensor(k).half()                                            # [1280, 5120]
    capa, lado = re.search(r"layers\.(\d+)\.(\w+_conv)", k).groups()
    nk = f"model.layers.{capa}." + ("input_layernorm.weight" if lado == "attention_conv" else "post_attention_layernorm.weight")
    nw = f.get_tensor(nk).half() if nk in f.keys() else torch.ones(w.shape[1], device="cuda", dtype=torch.float16)
    x = (torch.randn(9, w.shape[1], device="cuda") * nw.float()).half()
    ref = (x.float() @ w.float().t())
    base = x @ w.t()
    er = lambda y: float((y.float() - ref).norm() / ref.norm())
    res.setdefault("fp16 cuBLAS", []).append(er(base))
    for g in (-1, 128, 64, 32):
        q, s = g157._rtn(w, 8, g)
        m = g157._Marlin(q, s, 8, g)
        res.setdefault(f"int8 g{g if g > 0 else 'canal'}", []).append(er(m(x)))
    if i == 0:
        print(f"forma {tuple(w.shape)}, norma previa {nk.split('.')[-2]} |w| max {float(w.abs().max()):.3f}")
        tiempos = {"fp16 cuBLAS": t_grafo(lambda: x @ w.t()), "fp16 mitad de N (TP)": t_grafo(lambda: x @ w[:640].t())}
        for g in (-1, 128, 64, 32):
            q, s = g157._rtn(w, 8, g); m = g157._Marlin(q, s, 8, g)
            tiempos[f"int8 g{g if g > 0 else 'canal'}"] = t_grafo(lambda: m(x))
        q, s = g157._rtn(w[:640], 8, 128); m = g157._Marlin(q, s, 8, 128)
        tiempos["int8 g128 mitad de N (TP)"] = t_grafo(lambda: m(x))
for n, v in res.items():
    print(f"{n:14s} error relativo de la salida: media {sum(v) / len(v):.2e}  max {max(v):.2e}")
for n, v in tiempos.items():
    print(f"GRAFO M=9 {n:26s} {v:6.1f} us")
