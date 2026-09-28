"""SK-26 (PN156): residuo + RMSNorm Gemma + int8 por token en un kernel. Variantes SUMA 0 (todo fp32),
1 (half2 + suma de cuadrados por DP4A sobre el int8), 2 (half2 + suma de cuadrados fp32), contra la
referencia en torch (el camino de hoy) y contra los 3 kernels de vLLM que reemplaza. Reloj fijo afuera.
"""
import torch
from vllm._genesis.kernels.ptx_lab import Kernel

dev = "cuda"; H = 5120; eps = 1e-6
torch.manual_seed(0)
torch.zeros(1, device=dev)          # contexto CUDA antes de cargar los .cu


def ref(x, r, w, g):
    s = x.float() + r.float()
    rn = s.half()
    y = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * (1 + w.float())).half().float()
    am = y.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    return rn, torch.round(y * (127 / am)).to(torch.int8), (am / 127 * g).squeeze(-1)


ks = {v: Kernel("sk26_norma_q8.cu", "sk26_norma_q8", defs=[f"-DSUMA={v}", f"-DH={H}"], warps=H // 8 // 32) for v in (0, 1, 2, 3)}
for k in ks.values(): k.cargar()


def correr(v, x, r, w, g, q, e):
    ks[v].lanzar((x.shape[0], 1), [x, r, r, w, q, e, g, q.stride(0), float(eps)])


def base(x, r, w1, g):   # los 3 kernels de hoy: fused_add_rms_norm + dynamic int8 + escala por la global
    torch.ops._C.fused_add_rms_norm(x, r, w1, eps)
    q = torch.empty(x.shape, dtype=torch.int8, device=dev); s = torch.empty(x.shape[0], 1, device=dev)
    torch.ops._C.dynamic_scaled_int8_quant(q, x, s, None)
    return q, s * g


for T in (9, 36, 8192):
    # entradas con la forma del residuo rotado (cresta ~3) y un peso de norma cero (plegado) o al azar
    for nombre_w, w in (("w=0", torch.zeros(H, device=dev).half()), ("w al azar", (0.1 * torch.randn(H, device=dev)).half())):
        x = (torch.randn(T, H, device=dev) * 2).half(); r0 = (torch.randn(T, H, device=dev) * 20).half()
        g = torch.tensor([0.37], device=dev)
        rn_ref, q_ref, e_ref = ref(x, r0, w, g)
        linea = f"T={T:5d} {nombre_w:9s}"
        for v in (0, 3):
            r = r0.clone(); q = torch.empty(T, H, dtype=torch.int8, device=dev); e = torch.empty(T, device=dev)
            correr(v, x, r, w, g, q, e); torch.cuda.synchronize()
            dq = (q.int() - q_ref.int()).abs()
            linea += (f" | S{v}: res {'ok' if torch.equal(r, rn_ref) else 'DIF'} q!= {float((dq > 0).float().mean())*100:.2f}% "
                      f"max {int(dq.max())} esc {float(((e - e_ref).abs() / e_ref).max())*100:.3f}%")
        print(linea, flush=True)

print("tiempos (grafo de 100 llamadas, us por llamada):")
for T in (9, 36, 512, 8192):
    x = torch.randn(T, H, device=dev).half(); r = torch.randn(T, H, device=dev).half(); w = torch.zeros(H, device=dev).half()
    w1 = (w.float() + 1).half(); g = torch.tensor([1.0], device=dev)
    q = torch.empty(T, H, dtype=torch.int8, device=dev); e = torch.empty(T, device=dev)
    fs = {"hoy (3 kernels)": lambda: base(x, r, w1, g)}
    for v in (0, 3):
        fs[f"SK-26 S{v}"] = (lambda v=v: correr(v, x, r, w, g, q, e))
    out = []
    for nom, f in fs.items():
        for _ in range(3): f()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s): f()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(100): f()
        for _ in range(3): gr.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
        for _ in range(20): gr.replay()
        e1.record(); torch.cuda.synchronize()
        out.append(f"{nom} {e0.elapsed_time(e1) * 1000 / 2000:7.2f}")
    print(f"T={T:5d}: " + " | ".join(out), flush=True)
