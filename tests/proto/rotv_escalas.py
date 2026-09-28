"""V rotada (ROTV) en el pool de PN131: error de decuantizacion por token y distribucion de svf."""
import os, types, torch
os.environ.setdefault("GENESIS_PN131_MAXTOK", "9")
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; BS = 880; N = int(os.environ.get("N", "20000"))
torch.manual_seed(0)
impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, head_size=D, num_heads=12, num_kv_heads=NH,
                             alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0, scale=1 / 16,
                             chunk_lookback=-1, use_td=False)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev), layer_name="rotv")
nb = (N + BS - 1) // BS + 1
kv = torch.zeros(nb, BS, NH, 520, dtype=torch.int8, device=dev).permute(0, 2, 1, 3)
slots = torch.arange(N, device=dev)
K0 = (torch.randn(N, NH, D, device=dev) * 2).half(); V0 = torch.randn(N, NH, D, device=dev)
if os.environ.get("VOUT") == "1": V0[:, :, :4] *= 8
V0 = V0.half()
for t0 in range(0, N, 8192):
    P.escribir(impl, layer, K0[t0:t0 + 8192], V0[t0:t0 + 8192], kv, slots[t0:t0 + 8192])
c = P._capa(impl, torch.device(dev)); ek, ev = [int(x) for x in c.refs.tolist()]
raw = kv.permute(0, 2, 1, 3).contiguous().view(nb, -1)
KOFF, EOFF = BS * NH * D, 2 * BS * NH * D
Vi = raw[:, KOFF:EOFF].view(nb, NH, D, BS).permute(0, 3, 1, 2).reshape(-1, NH, D)[:N].float()
Es = raw[:, EOFF:EOFF + BS * NH * 4].contiguous().view(torch.int16).view(-1, NH, 2)[:N].float()
H = P._hadamard(torch.device(dev), torch.float32) / 16
sg = P._signos_dev(torch.device(dev)).float()
Vr = (V0.float() * sg) @ H if P.ROTV_MAIN else V0.float()
vq = Vi * Es[..., 1:] * 2.0 ** (ev - 15)
e = ((vq - Vr).norm(dim=-1) / Vr.norm(dim=-1))
svf = Es[..., 1]
print(f"ROTV={int(P.ROTV_MAIN)} VOUT={os.environ.get('VOUT','0')} ev={ev}: error de V por token mediana {e.median():.3%} p99 {e.quantile(0.99):.3%} | "
      f"svf min {svf.min():.0f} mediana {svf.median():.0f} max {svf.max():.0f} (tope {1 << P.VSH}), saturadas {(svf >= (1 << P.VSH)).float().mean():.2%}")
