"""(base: gdn_arbol.py) Hito 3 del arbol de borrador: GDN en arbol sobre la cinta de PN122.

Compara ``gdn_cinta.spec_update`` con GENESIS_ENABLE_ARBOL=1 contra una referencia en fp32
escrita aparte (estado por nodo = regla delta sobre el estado de SU padre), a lo largo de varios
pasos con aceptaciones al azar: asi se prueba tambien que lo que se reproduce al paso siguiente
es el CAMINO aceptado y no las primeras filas de la cinta. Y el control: con la mascara de
cadena la salida tiene que ser identica bit a bit a la del kernel de produccion.

Correr adentro del contenedor:  GENESIS_ENABLE_ARBOL=1 GENESIS_ENABLE_PN122_GDN_CINTA=1 python3 gdn_arbol.py
"""
import os, sys, types, torch
os.environ["GENESIS_ENABLE_ARBOL"] = "1"
os.environ["GENESIS_ENABLE_PN122_GDN_CINTA"] = "1"
os.environ.setdefault("GENESIS_PN122_ARBOL_PTX", "1")
os.environ.setdefault("GENESIS_PN122_ESCRIBIR_PAR", "ptx")
import vllm._genesis.gdn_cinta as g
import vllm._genesis.arbol_borrador as ab

torch.manual_seed(3)
dev, dt = "cuda", torch.float16
H, HV, K, V, SPEC, N = 8, 24, 128, 128, 8, 4
T = SPEC + 1
PASOS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NEG = int(os.environ.get("NEG", "0"))
S = 1 + N
raw = torch.randn(S, HV * V * K + 1000, device=dev, dtype=dt) * 0.05
h = raw[:, :HV * V * K].view(S, HV, V, K)
A_log = torch.randn(HV, device=dev) * 0.5
dt_bias = torch.randn(HV, device=dev) * 0.5
cols = torch.arange(1, S, device=dev, dtype=torch.int32).view(N, 1)
g._init_slots(8, dev)
layer = types.SimpleNamespace(tp_size=2, num_k_heads=2 * H, num_v_heads=2 * HV, head_k_dim=K,
                              head_v_dim=V, num_spec=SPEC, prefix="t")
g.enlazar(layer, dev)
slots = torch.tensor([3, 7, 1, 5], device=dev, dtype=torch.int32)
cu = torch.arange(0, (N + 1) * T, T, device=dev, dtype=torch.int32)
anc_buf = torch.zeros(N * T, dtype=torch.int32, device=dev)
g.fijar_ancestros(anc_buf)
camino = g.camino_gpu()
assert camino is not None and camino.shape[1] == SPEC


def delta(Sx, kk, vv, gg, bb):
    Sx = Sx * torch.exp(gg)[:, None, None]
    d = (vv - torch.einsum("hvk,hk->hv", Sx, kk)) * bb[:, None]
    return Sx + d[:, :, None] * kk[:, None, :]


def entradas(q, k, v, a, b, i):
    kk = k[0, i].float(); qq = q[0, i].float()
    kk = kk * torch.rsqrt((kk * kk).sum(-1, keepdim=True) + 1e-6)
    qq = qq * torch.rsqrt((qq * qq).sum(-1, keepdim=True) + 1e-6) * K ** -0.5
    rep = HV // H
    x = a[i].float() + dt_bias
    sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    return (qq.repeat_interleave(rep, 0), kk.repeat_interleave(rep, 0), v[0, i].float(),
            -torch.exp(A_log) * sp, torch.sigmoid(b[i].float()))


def arbol_al_azar(n, gen, dfs):
    """Padres de [ancla]+n nodos. dfs=False deja un orden topologico cualquiera."""
    pad = torch.tensor([[int(torch.randint(-1, i, (1,), generator=gen)) for i in range(n)]])
    if dfs:
        prof = torch.zeros_like(pad)
        for i in range(n):
            prof[0, i] = 0 if pad[0, i] < 0 else prof[0, pad[0, i]] + 1
        _, pad, _, _, _ = ab.orden_dfs(pad.clone(), pad, prof, pad.clone())
    return ab.padres_verificacion(pad)[0]          # [T], el 0 es el ancla



# PN164: a y b como vistas de una fila ancha (stride de fila de la salida de in_proj) contra contiguas:
# spec_update (arbol PTX + cinta PTX) tiene que dar lo MISMO bit a bit (salida, estado y cinta).
gen = torch.Generator().manual_seed(11)
print("arbol PTX:", g._ARBOL_PTX, "| cinta:", g._ESCRIBIR_MODO)
ANCHO = 6208                                          # fila de in_proj por rango (qkvz + trozos b/a con relleno)
todo_igual = True
for paso in range(6):
    q = torch.randn(1, N * T, H, K, device=dev, dtype=dt); k = torch.randn_like(q)
    v = torch.randn(1, N * T, HV, V, device=dev, dtype=dt)
    fila = torch.randn(N * T, ANCHO, device=dev, dtype=dt)
    b_v, a_v = fila[:, 6080:6080 + HV], fila[:, 6144:6144 + HV]        # vistas, stride ANCHO
    a_c, b_c = a_v.contiguous(), b_v.contiguous()
    for n in range(N):
        p = torch.arange(-1, T - 1) if n == 0 else arbol_al_azar(SPEC, gen, dfs=(n != 3))
        anc_buf[n * T:(n + 1) * T] = ab.bits_ancestros(p[None])[0].to(dev)
    acc = torch.randint(1, T + 1, (N,), device=dev, dtype=torch.int32)
    h0 = h.clone(); cinta0 = layer._g122_cinta.clone()
    oC, _ = g.spec_update(layer, A_log, a_c, b_c, dt_bias, q, k, v, h, cu, cols, acc, slots)
    hC, cintaC = h.clone(), layer._g122_cinta.clone()
    h.copy_(h0); layer._g122_cinta.copy_(cinta0)
    oV, _ = g.spec_update(layer, A_log, a_v, b_v, dt_bias, q, k, v, h, cu, cols, acc, slots)
    torch.cuda.synchronize()
    igual = torch.equal(oC, oV) and torch.equal(hC, h) and torch.equal(cintaC, layer._g122_cinta)
    todo_igual &= igual
    print(f"paso {paso}: salida {torch.equal(oC, oV)} estado {torch.equal(hC, h)} cinta {torch.equal(cintaC, layer._g122_cinta)}")
print("PN164 vistas == contiguas bit a bit:", todo_igual, "| kernels del arbol:", sorted(g._k_arbol_ptx), "| cinta:", sorted(g._k_cinta_ptx))
