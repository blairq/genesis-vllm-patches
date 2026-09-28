"""_k_spec_arbol (Triton) contra gdn_arbol.cu (PTX), los dos contra una referencia en fp64.

El PTX suma en otro orden (reduce-scatter dentro del warp, h.q algebraico), asi que NO es identico bit a
bit a Triton: el criterio es que su error contra fp64 sea <= al de Triton, en la salida o y en el estado
que escribe. Casos: arboles al azar (con saltos de rama), cadena, camino a reproducir de 0..8 filas de
cinta, 1/4/6 pedidos, estado fp16 y fp32, q/k/v como vistas con stride de un tensor mezclado (como las
deja la conv en el servidor). Despues, tiempos con grafos CUDA.
"""
import ctypes, random, math, os
import torch, triton
import vllm._genesis.gdn_cinta as g
from vllm._genesis.kernels.ptx_lab import Kernel

dev = "cuda"
H, HV, K, V, TM, T = 8, 24, 128, 128, 8, 9
ROW = H * K + HV * V + 2 * HV
BV = 32
RPW = int(os.environ.get("RPW", "8"))     # filas por warp del PTX; su bloque cubre 4*RPW filas
NSL = 16
random.seed(0); torch.manual_seed(0)


def arbol(T, cadena=False):
    par = [-1] + [(t - 1) if cadena else random.randrange(0, t) for t in range(1, T)]
    orden = []            # preorden (el kernel exige topologico, mejor preorden)
    hijos = {i: [c for c in range(T) if par[c] == i] for i in range(T)}
    def dfs(n):
        orden.append(n)
        for c in hijos[n]:
            dfs(c)
    dfs(0)
    nuevo = {o: i for i, o in enumerate(orden)}
    par2 = [-1] + [nuevo[par[orden[i]]] for i in range(1, T)]
    m = []
    for t in range(T):
        bits, c = 0, t
        while c > 0:
            bits |= 1 << (c - 1)
            c = par2[c]
        m.append(bits)
    return m


def caso(N, hdt, cadena=False, rr=None):
    Tt = N * T
    ancho = 2 * H * K + HV * V                       # q | k | v mezclados por token (como la salida de la conv)
    mix = torch.randn(Tt, ancho, device=dev, dtype=torch.float16)
    q = mix[:, :H * K].view(1, Tt, H, K)
    k = mix[:, H * K:2 * H * K].view(1, Tt, H, K)
    v = mix[:, 2 * H * K:].view(1, Tt, HV, V) * 0.5
    a = torch.randn(Tt, HV, device=dev, dtype=torch.float16) * 2
    b = torch.randn(Tt, HV, device=dev, dtype=torch.float16)
    A_log = torch.randn(HV, device=dev) * 0.5
    dt_bias = torch.randn(HV, device=dev) * 0.5
    h = (torch.randn(NSL + 1, HV, V, K, device=dev) * 0.05).to(hdt)
    cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
    sidx = torch.tensor(random.sample(range(1, NSL + 1), N), device=dev, dtype=torch.int32)
    slots = torch.tensor(random.sample(range(NSL), N), device=dev, dtype=torch.int32)
    nacc = torch.tensor([(rr if rr is not None else random.randint(0, TM)) + 1 for _ in range(N)], device=dev,
                        dtype=torch.int32)
    # cinta con filas plausibles: k normalizada, v, g <= 0, beta en (0,1)
    cinta = torch.zeros(NSL, TM, ROW, device=dev)
    kk = torch.randn(NSL, TM, H, K, device=dev)
    cinta[..., :H * K] = (kk / kk.norm(dim=-1, keepdim=True)).view(NSL, TM, H * K)
    cinta[..., H * K:H * K + HV * V] = torch.randn(NSL, TM, HV * V, device=dev) * 0.5
    cinta[..., H * K + HV * V:H * K + HV * V + HV] = -torch.rand(NSL, TM, HV, device=dev) * 2
    cinta[..., H * K + HV * V + HV:] = torch.rand(NSL, TM, HV, device=dev)
    camino = torch.stack([torch.randperm(TM) for _ in range(NSL)]).to(dev, torch.int32).contiguous()
    anc = torch.tensor([m for _ in range(N) for m in arbol(T, cadena)], device=dev, dtype=torch.int32)
    return dict(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias, h=h, cu=cu, sidx=sidx, slots=slots,
                nacc=nacc, cinta=cinta, camino=camino, anc=anc, N=N)


def ref64(c):
    """La semantica de _k_spec_arbol en fp64."""
    q, k, v = c["q"][0].double(), c["k"][0].double(), c["v"][0].double()
    a, b = c["a"].double(), c["b"].double()
    h = c["h"].double().clone()
    o = torch.zeros(q.shape[0], HV, V, dtype=torch.float64, device=dev)
    scale = K ** -0.5
    for n in range(c["N"]):
        bos = int(c["cu"][n]); s = int(c["sidx"][n]); slot = int(c["slots"][n]); r = int(c["nacc"][n]) - 1
        for hv in range(HV):
            ih = hv // (HV // H)
            S = h[s, hv].clone()
            eA = -math.exp(float(c["A_log"][hv])); db = float(c["dt_bias"][hv])
            def delta(S, kv, vv, gg, bb):
                S = S * math.exp(gg)
                d = (vv - S @ kv) * bb
                return S + torch.outer(d, kv)
            for j in range(r):
                jj = int(c["camino"][slot, j]); row = c["cinta"][slot, jj].double()
                S = delta(S, row[ih * K:(ih + 1) * K], row[H * K + hv * V:H * K + (hv + 1) * V],
                          float(row[H * K + HV * V + hv]), float(row[H * K + HV * V + HV + hv]))
            def tok(t):
                src = bos + t
                kn = k[src, ih] / torch.sqrt((k[src, ih] ** 2).sum() + 1e-6)
                x = float(a[src, hv]) + db
                sp = math.log1p(math.exp(x)) if x <= 20 else x
                bb = 1 / (1 + math.exp(-float(b[src, hv])))
                return kn, v[src, hv], eA * sp, bb
            raiz, m_prev = None, 0
            for t in range(T):
                m = int(c["anc"][bos + t])
                if t >= 2 and m != (m_prev | (1 << (t - 1))):
                    S = raiz.clone()
                    for j in range(1, t):
                        if (m >> (j - 1)) & 1:
                            S = delta(S, *tok(j))
                m_prev = m
                kn, vv, gg, bb = tok(t)
                S = delta(S, kn, vv, gg, bb)
                qn = q[bos + t, ih] / torch.sqrt((q[bos + t, ih] ** 2).sum() + 1e-6) * scale
                o[bos + t, hv] = S @ qn
                if t == 0:
                    raiz = S.clone(); h[s, hv] = S
    return o, h


def correr_triton(c):
    h = c["h"].clone(); o = torch.empty(1, c["q"].shape[1], HV, V, device=dev, dtype=torch.float16)
    g._k_spec_arbol[(triton.cdiv(V, BV), c["N"] * HV)](
        c["A_log"], c["a"], c["b"], c["dt_bias"], 1.0, 20.0, c["q"].contiguous(), c["k"].contiguous(),
        c["v"].contiguous(), o, h, h.stride(0), c["cu"], c["sidx"], c["nacc"], c["slots"], c["cinta"],
        c["camino"], c["anc"], K ** -0.5, c["N"],
        H=H, HV=HV, K=K, V=V, BK=K, BV=BV, TM=TM, ROW=ROW, IS_L2=True, num_warps=4, num_stages=3)
    return o, h


_kern = {}
def correr_ptx(c):
    hdt = 0 if c["h"].dtype == torch.float16 else 1
    kern = _kern.get(hdt)
    if kern is None:
        kern = _kern[hdt] = Kernel("gdn_arbol.cu", "gdn_arbol", warps=4,
                                   defs=[f"-DH={H}", f"-DHV={HV}", f"-DTM={TM}", f"-DTMAX={T}", f"-DHDT={hdt}", f"-DRPW={RPW}"])
    h = c["h"].clone(); o = torch.empty(1, c["q"].shape[1], HV, V, device=dev, dtype=torch.float16)
    L = ctypes.c_longlong
    kern.lanzar((V // (4 * RPW), c["N"] * HV), [c["A_log"], c["a"], c["b"], c["dt_bias"], c["q"], c["k"], c["v"],
                L(c["q"].stride(1)), L(c["k"].stride(1)), L(c["v"].stride(1)), o, h, L(h.stride(0)),
                c["cu"], c["sidx"], c["nacc"], c["slots"], c["cinta"], c["camino"], c["anc"], K ** -0.5, c["N"]])
    return o, h


def err(x, ref):
    return ((x.double() - ref).norm() / ref.norm()).item()


ok = True
print(f"{'caso':36} {'o Triton':>10} {'o PTX':>10} {'h Triton':>10} {'h PTX':>10}")
for nom, N, hdt, cad, rr in [("1 pedido, arbol, h fp16", 1, torch.float16, False, None),
                             ("4 pedidos, arbol, h fp16", 4, torch.float16, False, None),
                             ("6 pedidos, arbol, h fp32", 6, torch.float32, False, None),
                             ("2 pedidos, cadena, sin cinta", 2, torch.float16, True, 0),
                             ("2 pedidos, arbol, cinta llena", 2, torch.float16, False, TM)]:
    c = caso(N, hdt, cad, rr)
    o_r, h_r = ref64(c)
    o_t, h_t = correr_triton(c); o_p, h_p = correr_ptx(c)
    torch.cuda.synchronize()
    ss = c["sidx"].long()
    eo_t, eo_p = err(o_t[0], o_r), err(o_p[0], o_r)
    eh_t, eh_p = err(h_t[ss], h_r[ss]), err(h_p[ss], h_r[ss])
    bien = eo_p <= max(eo_t * 1.1, 1e-4) and eh_p <= max(eh_t * 1.1, 1e-4)
    ok &= bien
    print(f"{nom:36} {eo_t:10.2e} {eo_p:10.2e} {eh_t:10.2e} {eh_p:10.2e}  {'OK' if bien else 'PEOR'}")


def tiempo(f, reps=1000):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(10):
            f()
    for _ in range(10):
        gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps // 10):
        gr.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / reps


print(f"\n{'pedidos':>8} {'Triton us':>10} {'PTX us':>8}")
for N in (1, 4, 6):
    c = caso(N, torch.float16, False, 4)
    print(f"{N:8d} {tiempo(lambda: correr_triton(c)):10.2f} {tiempo(lambda: correr_ptx(c)):8.2f}")
print("\nRESULTADO:", "OK (PTX no peor que Triton contra fp64)" if ok else "FALLA")
