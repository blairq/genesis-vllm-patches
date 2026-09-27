"""Barrido de BN y num_warps de _k_salidas_par (escribir_estado 0 y 1), con grafos CUDA."""
import sys, torch, triton
sys.argv = ["x"]
exec(open("/t/arbol_salidas_par.py").read().split("ok = True")[0])   # reusa lote()
def med(N, BN, nw, esc, reps=2000):
    x, cs, w, sidx, nacc, cu, anc = lote([9] * N)
    out = torch.empty_like(x); T = anc.shape[0] // N
    def f():
        if BN == 0:
            ac.salidas(x, cs, w, "silu", sidx, nacc, cu, anc, out=out, escribir_estado=esc, par=False); return
        ac._k_salidas_par[(N, triton.cdiv(DIM, BN))](x, x.stride(0), out, out.stride(0), cs, cs.stride(0), cs.stride(1),
            cs.stride(2), w, w.stride(0), w.stride(1), cu, sidx, nacc, anc, anc.stride(0), DIM,
            BN=BN, SILU=True, ESCRIBIR=esc, SL=cs.shape[-1], T=T, num_warps=nw)
    for _ in range(3): f()
    torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): f()
    for _ in range(20): g.replay()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(reps // 10): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / reps
for esc in (False, True):
    for N in (1, 4):
        r = [("viejo", med(N, 0, 4, esc))]
        for BN in (64, 128, 256, 512):
            for nw in (1, 2, 4):
                if BN // (32 * nw) >= 1: r.append((f"BN{BN}/w{nw}", med(N, BN, nw, esc)))
        r.sort(key=lambda z: z[1])
        print(f"escribir={int(esc)} N={N}: " + "  ".join(f"{a} {b:.2f}" for a, b in [z for z in r if z[0] == 'viejo'] + r[:5]))
