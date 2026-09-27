"""Barrido FPT x NHILOS del PTX de la conv del arbol (con estado), grafos CUDA; bit a bit contra Triton."""
import sys, ctypes, torch, triton
exec(open("/t/arbol_salidas_par.py").read().split("ok = True")[0])
from vllm._genesis.kernels.ptx_lab import Kernel
def med(N, fpt, nh, reps=2000):
    x, cs, w, sidx, nacc, cu, anc = lote([9] * N); anc = como_servidor(anc, [9] * N)
    out = torch.empty_like(x); dim = x.shape[1]
    k = Kernel("arbol_conv.cu", "arbol_conv", warps=max(1, nh // 32),
               defs=[f"-DTOK=9", f"-DFPT={fpt}", f"-DNHILOS={nh}", "-DSILU=1", "-DESCRIBIR=1"])
    cs0 = cs.clone()
    f = lambda c: k.lanzar((N, triton.cdiv(dim, nh * fpt)), [x, x.stride(0), out, out.stride(0), c,
            ctypes.c_longlong(c.stride(0)), c.stride(1), w, cu, sidx, nacc, anc, anc.stride(0), dim])
    # exactitud contra Triton desenrollado
    c1, c2 = cs0.clone(), cs0.clone(); o1 = torch.empty_like(x)
    ac.salidas(x, c1, w, "silu", sidx, nacc, cu, anc, out=o1, escribir_estado=True, par=True)
    f(c2); torch.cuda.synchronize()
    exacto = torch.equal(o1.view(torch.int16), out.view(torch.int16)) and torch.equal(c1.view(torch.int16), c2.view(torch.int16))
    for _ in range(3): f(cs)
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): f(cs)
    for _ in range(20): g.replay()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(reps // 10): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / reps, exacto
for N in (1, 4):
    r = []
    for fpt in (1, 2, 4):
        for nh in (32, 64, 128):
            t, ex = med(N, fpt, nh); r.append((t, f"FPT{fpt}/NH{nh}", ex))
    r.sort()
    print(f"N={N}: " + "  ".join(f"{n} {t:.2f}{'' if e else ' DISTINTO'}" for t, n, e in r))
