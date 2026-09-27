"""_k_escribir (1 programa por pedido, en serie) contra _k_escribir_par (grilla N x TM x (H+HV)).

1) exactitud: la cinta tiene que salir IDENTICA bit a bit, en lotes uniformes (arbol: T = 9) y
   en lotes irregulares (T distintos, T = 1, pedidos sin slot spec).
2) tiempo: cada variante grabada en un grafo CUDA y repetida, con 1, 4 y 6 pedidos.

Formas de idiotSavant por rango (TP=2): H = 8 cabezas k, HV = 24 cabezas v, K = V = 128.
Correr en la imagen de vLLM con el _genesis montado (ver el pie del archivo).
"""
import torch
import vllm._genesis.gdn_cinta as g

dev, dt = "cuda", torch.float16
H, HV, K, V = 8, 24, 128, 128
TM = 8                                   # filas de cinta por slot (arbol de 8 nodos -> T = 9)
ROW = H * K + HV * V + 2 * HV
NSLOTS = 16
torch.manual_seed(0)
A_log = torch.randn(HV, device=dev) * 0.5
dt_bias = torch.randn(HV, device=dev) * 0.5


def lote(Ts, sid=None):
    cu = torch.tensor([0] + list(torch.tensor(Ts).cumsum(0)), device=dev, dtype=torch.int32)
    Tt = int(cu[-1])
    k = torch.randn(1, Tt, H, K, device=dev, dtype=dt) * 3
    v = torch.randn(1, Tt, HV, V, device=dev, dtype=dt)
    a = torch.randn(Tt, HV, device=dev, dtype=dt) * 4          # cubre las dos ramas del softplus
    b = torch.randn(Tt, HV, device=dev, dtype=dt)
    N = len(Ts)
    sidx = torch.tensor(sid if sid is not None else [i + 1 for i in range(N)], device=dev, dtype=torch.int32)
    slots = torch.randperm(NSLOTS, device=dev)[:N].to(torch.int32)
    return A_log, a, b, dt_bias, k, v, cu, sidx, slots, N


def correr(args, par):
    A_log_, a, b, dtb, k, v, cu, sidx, slots, N = args
    cinta = torch.full((NSLOTS, TM, ROW), float("nan"), device=dev)
    g.escribir_cinta(A_log_, a, b, dtb, k, v, cu, sidx, slots, cinta, N, H, HV, K, V, par=par)
    torch.cuda.synchronize()
    return cinta


casos = {
    "1 pedido, arbol T=9": [9],
    "4 pedidos, arbol": [9] * 4,
    "6 pedidos, arbol": [9] * 6,
    "irregular": [9, 1, 5, 3, 9, 2],
}
ok = True
for nom, Ts in casos.items():
    args = lote(Ts)
    viejo = correr(args, False)
    for modo in (True, "ptx"):
        nuevo = correr(args, modo)
        igual = torch.equal(viejo.view(torch.int32), nuevo.view(torch.int32))   # bits, NaN incluido
        ok &= igual
        print(f"{nom:24} {str(modo):5} {'IDENTICA' if igual else 'DISTINTA'}  filas escritas: "
              f"{int((~viejo.isnan()).any(-1).sum())}"
              + ("" if igual else f"  difieren {int((viejo.view(torch.int32) != nuevo.view(torch.int32)).sum())}"))
# A_log / dt_bias en bf16 (como pueden venir del checkpoint): el PTX usa una copia fp32 exacta
args = lote([9] * 3)
args = (args[0].bfloat16(),) + args[1:3] + (args[3].bfloat16(),) + args[4:]
for modo in (True, "ptx"):
    igual = torch.equal(correr(args, False).view(torch.int32), correr(args, modo).view(torch.int32))
    ok &= igual
    print(f"{'A_log/dt_bias bf16':24} {str(modo):5} {'IDENTICA' if igual else 'DISTINTA'}")
args = lote([9, 9, 9], sid=[1, 0, -1])                                        # pedidos sin slot spec
for modo in (True, "ptx"):
    igual = torch.equal(correr(args, False).view(torch.int32), correr(args, modo).view(torch.int32))
    ok &= igual
    print(f"{'sin slot spec':24} {str(modo):5} {'IDENTICA' if igual else 'DISTINTA'}")


def tiempo(N, par, reps=2000):
    args = lote([9] * N)
    A_log_, a, b, dtb, k, v, cu, sidx, slots, N = args
    cinta = torch.zeros((NSLOTS, TM, ROW), device=dev)
    f = lambda: g.escribir_cinta(A_log_, a, b, dtb, k, v, cu, sidx, slots, cinta, N, H, HV, K, V, par=par)
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(10):
            f()
    for _ in range(20):
        gr.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps // 10):
        gr.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000 / reps


print(f"\n{'pedidos':>8} {'serie us':>9} {'triton us':>10} {'ptx us':>8}")
for N in (1, 4, 6):
    tv, tn, tp = tiempo(N, False), tiempo(N, True), tiempo(N, "ptx")
    print(f"{N:8d} {tv:9.2f} {tn:10.2f} {tp:8.2f}")
print("\nRESULTADO:", "OK, bit a bit" if ok else "FALLA")

# docker run --rm --gpus '"device=0"' --entrypoint python3 \
#   -v $PWD/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis \
#   -v $PWD/tests/proto:/t vllm/vllm-openai:v0.29.0 /t/pn122_escribir_par.py
