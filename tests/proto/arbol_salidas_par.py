"""_k_salidas (lazo de tokens variable, BN 1024) contra _k_salidas_par (desenrollado, BN 256).

1) exactitud bit a bit de las salidas Y del estado conv que deja (escribir_estado), con arboles
   al azar, lotes uniformes (T = 9) e irregulares (incluye un pedido mas largo que el promedio,
   que cae en la cola dinamica) y pedidos sin slot spec;
2) tiempo con grafos CUDA, 1 / 4 / 6 pedidos.

Forma de idiotSavant por rango: dim de la conv = (2*16*128 + 48*128) / 2 = 5120, ancho 4,
state_len = 3 + K (K = 8 nodos del arbol).
"""
import random
import torch
import vllm._genesis.arbol_conv as ac

dev, dt = "cuda", torch.float16
DIM, KSPEC = 5120, 8
SL = 3 + KSPEC
NBLOQ = 24
random.seed(0); torch.manual_seed(0)


def anc3_arbol(T):
    par = [-1] + [random.randrange(0, t) for t in range(1, T)]
    filas = []
    for t in range(T):
        cur, fila = t, []
        for _ in range(3):
            cur = (par[cur] if cur >= 0 else cur - 1) if cur >= 0 else cur - 1
            fila.append(cur)
        filas.append(fila)
    return filas


def lote(Ts, sid=None, trasp=False, sl=None):
    sl = sl or SL
    N = len(Ts)
    cu = torch.tensor([0] + list(torch.tensor(Ts).cumsum(0)), device=dev, dtype=torch.int32)
    Tt = int(cu[-1])
    x = torch.randn(Tt, DIM, device=dev, dtype=dt)
    if trasp:                                                # estado con la dim contigua
        cs = torch.randn(NBLOQ, sl, DIM, device=dev, dtype=dt).transpose(1, 2)
    else:
        cs = torch.randn(NBLOQ, DIM, sl, device=dev, dtype=dt)
    w = torch.randn(DIM, 4, device=dev, dtype=dt) * 0.5
    sidx = torch.tensor(sid if sid is not None else random.sample(range(1, NBLOQ), N), device=dev,
                        dtype=torch.int32)
    nacc = torch.tensor([random.randint(1, KSPEC + 1) for _ in range(N)], device=dev, dtype=torch.int32)
    anc = torch.tensor([f for T in Ts for f in anc3_arbol(T)], device=dev, dtype=torch.int32)
    return x, cs, w, sidx, nacc, cu, anc


NSLOTS = 16


def como_servidor(anc, Ts):
    """El servidor pasa anc3 como un buffer FIJO de n_slots*(K+1) filas (grafos CUDA), no del
    largo del lote: es la forma que rompio la primera version (T = shape[0] // N)."""
    import vllm._genesis.gdn_cinta as g
    buf = torch.full((NSLOTS * (KSPEC + 1), 3), -1, device=dev, dtype=torch.int32)
    buf[:anc.shape[0]] = anc
    g._anc3_gpu, g._n_slots = buf, NSLOTS
    return buf


def correr(args, par, escribir):
    x, cs, w, sidx, nacc, cu, anc = args
    cs = cs.clone()
    out = torch.full_like(x, float("nan"))
    ac.salidas(x, cs, w, "silu", sidx, nacc, cu, anc, out=out, escribir_estado=escribir, par=par)
    torch.cuda.synchronize()
    return out, cs


def igual(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


ok = True
casos = [("1 pedido, arbol T=9", [9], None, False), ("4 pedidos", [9] * 4, None, False),
         ("6 pedidos", [9] * 6, None, False), ("estado traspuesto", [9] * 3, None, True),
         ("irregular (uno > promedio)", [9, 2, 1, 9, 3], None, False),
         # el estado conv tiene 3 + K columnas: un pedido de 12 tokens necesita K >= 12 (con SL = 11
         # escribiria fuera de la columna, en Triton y en PTX; en el servidor no pasa)
         ("mas largo que el arbol (cola)", [12, 3], None, False, 15),
         ("sin slot spec", [9, 9, 9], [5, 0, -1], False)]
for nom, Ts, sid, tr, *sl in casos:
    args = lote(Ts, sid, tr, *sl)
    args = args[:-1] + (como_servidor(args[-1], Ts),)
    for esc in (False, True):
        ov, cv = correr(args, False, esc)
        for modo in (True, "ptx"):
            on, cn = correr(args, modo, esc)
            r = igual(ov, on) and igual(cv.contiguous(), cn.contiguous())
            ok &= r
            print(f"{nom:28} escribir={int(esc)} {str(modo):5} {'IDENTICA' if r else 'DISTINTA'}"
                  + ("" if r else f"  salida {int((ov.view(torch.int16) != on.view(torch.int16)).sum())}"
                                  f" estado {int((cv.contiguous().view(torch.int16) != cn.contiguous().view(torch.int16)).sum())}"))


def tiempo(N, par, reps=2000):
    x, cs, w, sidx, nacc, cu, anc = lote([9] * N)
    anc = como_servidor(anc, [9] * N)
    out = torch.empty_like(x)
    f = lambda: ac.salidas(x, cs, w, "silu", sidx, nacc, cu, anc, out=out, escribir_estado=True, par=par)
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            f()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps // 10):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / reps


print(f"\n{'pedidos':>8} {'viejo us':>9} {'triton us':>10} {'ptx us':>8}")
for N in (1, 4, 6):
    tv, tn, tp = tiempo(N, False), tiempo(N, True), tiempo(N, "ptx")
    print(f"{N:8d} {tv:9.2f} {tn:10.2f} {tp:8.2f}")
print("\nRESULTADO:", "OK, bit a bit" if ok else "FALLA")
