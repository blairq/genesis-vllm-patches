"""SK-30 (atencion de decode en una pasada): referencia en torch sobre la KV int8 REAL de PN131 (la escribe
P.escribir, como en el servidor). Decuantiza K (rotada) y V del pool, ajusta las constantes contra los k/v
originales y compara la atencion en float contra el decode actual (batch2 + union + salida)."""
import os, sys, types, torch
os.environ.setdefault("GENESIS_PN131_MAXTOK", "9")
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; BS = 880; L = 9; HQ = 12
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
torch.manual_seed(0)
impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, head_size=D, num_heads=HQ,
                             alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                             scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev), layer_name="capa_sk30")
nb = (N + BS - 1) // BS + 2
kv = torch.zeros(nb, BS, NH, 520, dtype=torch.int8, device=dev).permute(0, 2, 1, 3)
perm = torch.randperm(nb)[: (N + BS - 1) // BS].to(torch.int32)
slots = (perm[torch.arange(N) // BS].to(torch.int64) * BS + torch.arange(N) % BS).to(dev)
K0 = torch.empty(N, NH, D, device=dev).half(); V0 = torch.empty_like(K0)
for t0 in range(0, N, 8192):
    t1 = min(N, t0 + 8192)
    K0[t0:t1] = (torch.randn(t1 - t0, NH, D, device=dev) * 2).half(); V0[t0:t1] = torch.randn(t1 - t0, NH, D, device=dev).half()
    P.escribir(impl, layer, K0[t0:t1], V0[t0:t1], kv, slots[t0:t1])
c = P._capa(impl, torch.device(dev)); ek, ev = [int(x) for x in c.refs.tolist()]
raw = kv.permute(0, 2, 1, 3).contiguous().view(nb, -1)                       # bloque crudo [nb, BS*NH*520]
KOFF, EOFF = BS * NH * D, 2 * BS * NH * D
Ki = raw[:, :KOFF].view(nb, BS, NH, D).float()
Vi = raw[:, KOFF:EOFF].view(nb, NH, D, BS).permute(0, 3, 1, 2).float()       # -> [nb, BS, NH, D]
Es = raw[:, EOFF:EOFF + BS * NH * 4].contiguous().view(torch.int16).view(nb, BS, NH, 2).float()
pos = slots.cpu()
kq = Ki.view(-1, NH, D)[pos.to(dev)] * Es.view(-1, NH, 2)[pos.to(dev)][..., :1]
vq = Vi.reshape(-1, NH, D)[pos.to(dev)] * Es.view(-1, NH, 2)[pos.to(dev)][..., 1:]
H = P._hadamard(torch.device(dev), torch.float32) / 16
Kr = (K0.float() * P._signos_dev(torch.device(dev)).float()) @ H
ck = float((Kr * kq).sum() / (kq * kq).sum()); cv = float((V0.float() * vq).sum() / (vq * vq).sum())
ek_rel = float(((kq * ck - Kr).norm() / Kr.norm())); ev_rel = float(((vq * cv - V0.float()).norm() / V0.float().norm()))
print(f"refs ek={ek} ev={ev}: ck={ck:.6g} (= 2^{torch.log2(torch.tensor(ck)):.3f}), cv={cv:.6g} (= 2^{torch.log2(torch.tensor(cv)):.3f}); error K {ek_rel:.3%}, V {ev_rel:.3%}")
torch.save(dict(ck=ck, cv=cv, ek=ek, ev=ev), os.environ.get("SALIDA", "/tmp/sk30_const.pt"))
