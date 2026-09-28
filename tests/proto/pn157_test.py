"""PN157: Marlin del borrador contra F.linear denso (K/V de contexto desde el empacado; kernel_projection RTN)."""
import torch
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
from vllm._genesis import borrador_marlin as bm
dev = "cuda"; torch.manual_seed(0)


def tiempo(f):
    for _ in range(5): f()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20): f()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(5): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / 100


# K/V de contexto: 5 capas x (k+v) 1024 filas por rango, entrada 5120, int4 g128
N, K = 5120, 5120
q = torch.randint(-8, 8, (N, K), device=dev, dtype=torch.int8)
s = (torch.rand(N, K // 128, device=dev) * 0.01 + 0.001).half()
packed = pack_to_int32(q.cpu(), 4, packed_dim=1).to(dev)
m = bm.marlin_de_empacado(packed, s, K)
wd = (q.float().view(N, K // 128, 128) * s.float()[..., None]).view(N, K).half()
for M in (9, 36, 300):
    x = torch.randn(M, K, device=dev).half()
    ref = torch.nn.functional.linear(x, wd); y = m(x)
    print(f"kv M={M:4d}: rel {float((y - ref).abs().max() / ref.abs().max()):.1e} | denso {tiempo(lambda: torch.nn.functional.linear(x, wd)):6.1f} us  marlin {tiempo(lambda: m(x)):6.1f} us")

# kernel_projection: [1280 x 5120] RTN
w = (torch.randn(1280, 5120, device=dev) * 0.02).half()
for bits in (8, 4):
    g = 128 if bits == 4 else -1
    qq, ss = bm._rtn(w, bits, g); mm = bm._Marlin(qq, ss, bits, g)
    for M in (9, 36):
        x = torch.randn(M, 5120, device=dev).half()
        ref = torch.nn.functional.linear(x, w)
        wq = (qq.float().view(1280, -1, 5120 if g < 0 else g) * ss[..., None]).view(1280, 5120).half()
        refq = torch.nn.functional.linear(x, wq); y = mm(x)
        print(f"conv W{bits} M={M:3d}: vs denso cuantizado rel {float((y - refq).abs().max() / refq.abs().max()):.1e}, "
              f"vs original rel {float((y - ref).norm() / ref.norm()):.1e} | denso {tiempo(lambda: torch.nn.functional.linear(x, w)):5.1f} us  marlin {tiempo(lambda: mm(x)):5.1f} us")

# prefill: la K/V de contexto con chunks grandes (el borrador procesa el contexto del prompt)
for M in (1024, 2048, 4096, 8192):
    x = torch.randn(M, K, device=dev).half()
    print(f"kv prefill M={M:5d}: denso {tiempo(lambda: torch.nn.functional.linear(x, wd)):7.1f} us  marlin {tiempo(lambda: m(x)):7.1f} us", flush=True)
