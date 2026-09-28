import os, runpy, torch
os.environ["GENESIS_PN131_ROTV"] = "1"
from vllm._genesis import sk18_attn as P
# union SIN des-rotar (salida en la base rotada), el resto igual
orig = P._kernels
def k2(md="int8"):
    ks = orig(md)
    if not ks.get("_sin_rot"):
        from vllm._genesis.kernels.ptx_lab import Kernel
        ks["sk30u"] = Kernel("sk30_decode_1pasada.cu", "sk30_union", warps=8); ks["sk30u"].cargar(); ks["_sin_rot"] = True
    return ks
P._kernels = k2
g = runpy.run_path("/t/sk30_integrado.py", run_name="sin_main")
kv, K0s, V0s, LENS, q, out, BS, NH, D, L, G = (g[x] for x in ("kv", "K0s", "V0s", "LENS", "q", "out", "BS", "NH", "D", "L", "G"))
bt = g["bt"]; dev = "cuda"
c = P._capa(g["impl"], torch.device(dev)); ek, ev = [int(x) for x in c.refs.tolist()]
nb = kv.shape[0]
raw = kv.permute(0, 2, 1, 3).contiguous().view(nb, -1)
KOFF, EOFF = BS * NH * D, 2 * BS * NH * D
Ki = raw[:, :KOFF].view(nb, BS, NH, D).float()
Vi = raw[:, KOFF:EOFF].view(nb, NH, D, BS).permute(0, 3, 1, 2).float()
Es = raw[:, EOFF:EOFF + BS * NH * 4].contiguous().view(torch.int16).view(nb, BS, NH, 2).float()
H = P._hadamard(torch.device(dev), torch.float32) / 16; sg = P._signos_dev(torch.device(dev)).float()
for b, n in enumerate(LENS):
    pos = torch.arange(n, device=dev)
    blk = bt[b].long()[pos // BS]; sl = pos % BS
    kd = Ki[blk, sl] * Es[blk, sl][..., :1] * 2.0 ** (ek - 15)            # k rotada
    vd = Vi[blk, sl] * Es[blk, sl][..., 1:] * 2.0 ** (ev - 15)            # v rotada
    ctx = n - L
    qr = ((q[b * L:(b + 1) * L].float() * sg) @ H).view(L, NH, G, D)
    sc = torch.einsum("jhgd,thd->hjgt", qr, kd) / 16
    vis = (pos[None, :] < ctx) | (pos[None, :] - ctx <= torch.arange(L, device=dev)[:, None])
    o_rot = torch.einsum("hjgt,thd->jhgd", sc.masked_fill(~vis[None, :, None, :], float("-inf")).softmax(-1), vd).reshape(L, NH * G, D)
    qf = q[b * L:(b + 1) * L].float().view(L, NH, G, D)
    sc0 = torch.einsum("jhgd,thd->hjgt", qf, K0s[b].float()) / 16
    ref = torch.einsum("hjgt,thd->jhgd", sc0.masked_fill(~vis[None, :, None, :], float("-inf")).softmax(-1), V0s[b].float()).reshape(L, NH * G, D)
    ref_rot = (ref * sg) @ H
    o_k = out[b * L:(b + 1) * L].float()
    print(f"seq {b} (n={n}): torch con la KV decuantizada vs float {float((o_rot - ref_rot).norm() / ref_rot.norm()):.3%} | "
          f"SK-30 vs torch decuantizado {float((o_k - o_rot).norm() / o_rot.norm()):.3%}", flush=True)
for b, n in enumerate(LENS):
    pos = torch.arange(n, device=dev); blk = bt[b].long()[pos // BS]; sl = pos % BS
    kd = Ki[blk, sl] * Es[blk, sl][..., :1] * 2.0 ** (ek - 15); vd = Vi[blk, sl] * Es[blk, sl][..., 1:] * 2.0 ** (ev - 15)
    kr = (K0s[b].float() * sg) @ H; vr = (V0s[b].float() * sg) @ H
    ek_ = (kd - kr).norm(dim=-1) / kr.norm(dim=-1); ev_ = (vd - vr).norm(dim=-1) / vr.norm(dim=-1)
    print(f"seq {b}: error por token K mediana {ek_.median():.3%} max {ek_.max():.3%} | V mediana {ev_.median():.3%} max {ev_.max():.3%} | "
          f"tokens con V > 5%: {(ev_ > 0.05).float().mean():.2%}, primeros malos {torch.nonzero(ev_ > 0.05)[:5, 0].tolist()}")
