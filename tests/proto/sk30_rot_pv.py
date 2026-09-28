"""Error de P.V segun como se cuantiza P' (por fila y tile de 64 keys): directo int8, con Hadamard-64 en las
keys (P H)(H^T V), y 16 bits (hi/lo). Sobre atencion REAL-ista: q.k con k de la KV del harness; tambien una
version con picos (logits escalados x4) que se parece mas al softmax de un modelo entrenado."""
import torch, math
torch.manual_seed(0); dev = "cuda"
N, D, R, T = 8192, 256, 54, 64
def had(n):
    H = torch.ones(1, 1, device=dev)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)
H64 = had(T)
K = torch.randn(N, D, device=dev) * 2; V = torch.randn(N, D, device=dev)
V[:, :4] *= 8                                                   # canales con outliers en V
def cuant(x, bits):
    m = x.abs().amax(-1, keepdim=True).clamp_min(1e-30); n = 2 ** (bits - 1) - 1
    return (x / m * n).round() * m / n
Vq = (V / V.abs().amax(-1, keepdim=True) * 127).round() * V.abs().amax(-1, keepdim=True) / 127
for pico in (1.0, 4.0):
    q = torch.randn(R, D, device=dev)
    s = q @ K.t() / 16 * pico
    p = (s - s.max(-1, keepdim=True).values).exp()               # como el kernel: p <= 1 respecto del maximo
    ref = (p @ V) / p.sum(-1, keepdim=True)
    res = {}
    for nombre in ("int8", "int8+H64", "16 bits"):
        num = torch.zeros(R, D, device=dev)
        for t in range(0, N, T):
            pt, vt = p[:, t:t + T], Vq[t:t + T]
            if nombre == "int8": num += cuant(pt, 8) @ vt
            elif nombre == "16 bits": num += cuant(pt, 16) @ vt
            else: num += cuant(pt @ H64, 8) @ (H64.t() @ vt)
        o = num / p.sum(-1, keepdim=True)
        res[nombre] = float((o - ref).norm() / ref.norm())
    base = float(((p @ Vq) / p.sum(-1, keepdim=True) - ref).norm() / ref.norm())
    print(f"picos x{pico}: error de P.V (solo por P) int8 {res['int8']:.3%} | int8+Hadamard64 {res['int8+H64']:.3%} | "
          f"16 bits {res['16 bits']:.3%}   (referencia: el de V int8 {base:.3%})")
# V: rotar en d al escribir baja su error de cuantizacion?
H256 = had(D)
for nom, VV in (("gaussiana", torch.randn(N, D, device=dev)), ("con outliers", V)):
    e_dir = float((cuant(VV, 8) - VV).norm() / VV.norm()); Vr = VV @ H256
    e_rot = float(((cuant(Vr, 8) @ H256.t()) - VV).norm() / VV.norm())
    print(f"V {nom}: error int8 por token {e_dir:.3%}, rotada en d {e_rot:.3%}")
