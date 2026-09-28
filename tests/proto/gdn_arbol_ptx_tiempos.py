"""Tiempos de _k_spec_arbol (Triton) y gdn_arbol (PTX) con la GPU caliente, por forma del arbol."""
import torch
exec(open("/t/gdn_arbol_ptx.py").read().split("ok = True")[0])
def calentar():
    x = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    for _ in range(200): x = x @ x * 1e-3
    torch.cuda.synchronize()
def tiempo(f, reps=5000):
    for _ in range(3): f()
    torch.cuda.synchronize(); gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(10): f()
    calentar()
    for _ in range(50): gr.replay()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(reps // 10): gr.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / reps
def saltos(anc):
    m = anc.view(-1, T).tolist(); n = 0
    for fila in m:
        for t in range(2, T):
            if fila[t] != (fila[t - 1] | (1 << (t - 1))): n += 1
    return n / len(m)
print(f"{'forma':30} {'saltos/pedido':>13} {'pedidos':>8} {'Triton us':>10} {'PTX us':>8}")
for nom, cad, rr in [("cadena, r=0", True, 0), ("cadena, r=5", True, 5), ("arbol al azar, r=5", False, 5)]:
    for N in (1, 4):
        c = caso(N, torch.float16, cad, rr)
        # solo el kernel: buffers fijos (clonar el estado de 13 MB por llamada dominaba la medicion)
        h = c["h"].clone(); o = torch.empty(1, c["q"].shape[1], HV, V, device=dev, dtype=torch.float16)
        qc, kc, vc = c["q"].contiguous(), c["k"].contiguous(), c["v"].contiguous()
        def ft():
            g._k_spec_arbol[(triton.cdiv(V, BV), c["N"] * HV)](
                c["A_log"], c["a"], c["b"], c["dt_bias"], 1.0, 20.0, qc, kc, vc, o, h, h.stride(0), c["cu"], c["sidx"],
                c["nacc"], c["slots"], c["cinta"], c["camino"], c["anc"], K ** -0.5, c["N"],
                H=H, HV=HV, K=K, V=V, BK=K, BV=BV, TM=TM, ROW=ROW, IS_L2=True, num_warps=4, num_stages=3)
        correr_ptx(c)
        kern = _kern[0]; L = ctypes.c_longlong
        args = [c["A_log"], c["a"], c["b"], c["dt_bias"], c["q"], c["k"], c["v"], L(c["q"].stride(1)), L(c["k"].stride(1)),
                L(c["v"].stride(1)), o, h, L(h.stride(0)), c["cu"], c["sidx"], c["nacc"], c["slots"], c["cinta"],
                c["camino"], c["anc"], K ** -0.5, c["N"]]
        fp = lambda: kern.lanzar((V // (4 * RPW), c["N"] * HV), args)
        print(f"{nom:30} {saltos(c['anc']):13.1f} {N:8d} {tiempo(ft):10.2f} {tiempo(fp):8.2f}")
