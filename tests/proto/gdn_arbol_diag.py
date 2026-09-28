import torch
exec(open("/t/gdn_arbol_ptx.py").read().split("ok = True")[0])
x = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
for _ in range(200): x = x @ x * 1e-3
for rr in (0, 5):
    c = caso(1, torch.float16, True, rr)
    k = Kernel("gdn_arbol.cu", "gdn_arbol", warps=4, defs=[f"-DH={H}", f"-DHV={HV}", f"-DTM={TM}", f"-DTMAX={T}", "-DHDT=0", "-DDIAG=1", f"-DRPW={RPW}"])
    diag = torch.zeros(8, dtype=torch.int64, device=dev); L = ctypes.c_longlong
    h = c["h"].clone(); o = torch.empty(1, c["q"].shape[1], HV, V, device=dev, dtype=torch.float16)
    args = [c["A_log"], c["a"], c["b"], c["dt_bias"], c["q"], c["k"], c["v"], L(c["q"].stride(1)), L(c["k"].stride(1)),
            L(c["v"].stride(1)), o, h, L(h.stride(0)), c["cu"], c["sidx"], c["nacc"], c["slots"], c["cinta"],
            c["camino"], c["anc"], K ** -0.5, c["N"], diag]
    for _ in range(50): k.lanzar((V // (4 * RPW), HV), args)
    torch.cuda.synchronize()
    d = diag.tolist()
    print(f"r={rr}: prologo+estado {d[1]}  barrera {d[2]-d[1]}  reproducir {d[3]-d[2]}  tokens {d[4]-d[3]}  total {d[4]} ciclos")
