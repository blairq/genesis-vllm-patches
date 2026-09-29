"""SK-30 v1 (sk30_v1_ref.cu) contra la version actual, en el mismo proceso, intercaladas (reloj real, GPU compartida)."""
import os, sys
src = open("/t/sk30_test.py").read().split("rel = lambda")[0]
exec(compile(src, "sk30", "exec"))
import torch
from vllm._genesis.kernels.ptx_lab import Kernel
defs = [f"-DNQ={NQ}", "-DPV8=1", "-DPHL=1"]
defs2 = defs + [f"-DPFIJA={os.environ.get('PFIJA', '2')}"]
k1 = Kernel("sk30_v1_ref.cu", "sk30_decode", defs=defs, warps=4 * NQ); k1.cargar()
k2 = Kernel("sk30_decode_1pasada.cu", "sk30_decode", defs=defs2, warps=4 * NQ); k2.cargar()
args_new = [Qi, qs, kv, bt, bt.stride(0), sl, lim_, abase_, amask_, NH, BS, BLK, c.refs, GMAX, Op, Mp, Lp]
# v1: su firma tomaba vsc float en vez de refs
args_v1 = list(args_new)
fprep()
Op.zero_(); k1.lanzar((GMAX, NH), args_v1, shared=SH); torch.cuda.synchronize(); o1 = Op.clone(); m1 = Mp.clone(); l1 = Lp.clone()
Op.zero_(); k2.lanzar((GMAX, NH), args_new, shared=SH); torch.cuda.synchronize(); o2 = Op.clone(); m2 = Mp.clone(); l2 = Lp.clone()
ok = l1 > 0
print("salida v2 vs v1 (grupos con l > 0): max|dO|", float((o1.float() - o2.float())[ok].abs().max()), " max|dM|",
      float((m1[ok] - m2[ok]).abs().max()), " max rel dL", float(((l1 - l2).abs() / l1.clamp_min(1e-30))[ok].max()))
def t(k, a, it=20):
    g = torch.cuda.CUDAGraph()
    s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        k.lanzar((GMAX, NH), a, shared=SH)
    torch.cuda.current_stream().wait_stream(s_); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(it): k.lanzar((GMAX, NH), a, shared=SH)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(5): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / (5 * it)
r1, r2 = [], []
for _ in range(6):
    r1.append(t(k1, args_v1)); r2.append(t(k2, args_new))
r1.sort(); r2.sort()
print(f"N={N}: v1 mediana {r1[3]:.1f} us (min {r1[0]:.1f}) | v2 mediana {r2[3]:.1f} us (min {r2[0]:.1f}) | {100 * (r2[3] / r1[3] - 1):+.1f}%", flush=True)
