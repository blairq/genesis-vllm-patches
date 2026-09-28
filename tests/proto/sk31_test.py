"""SK-31 (atencion del borrador) contra float, sobre una KV int8_per_token_head con el layout de vLLM
(registro de 264 B por (slot, cabeza): K 128 | escala K fp32 | V 128 | escala V fp32; fisico NHD)."""
import os, math, torch
from vllm._genesis.kernels.ptx_lab import Kernel
dev = "cuda"; D = 128; NKV = 4; G = 4; HQ = NKV * G; BS = 880; L = 9; SW = 2048; GMAX = int(os.environ.get("GMAX", "16"))
LENS = [int(x) for x in os.environ.get("LENS", "20000").split(",")]; B = len(LENS)
torch.manual_seed(0)
pags = [(n + BS - 1) // BS for n in LENS]; nb = sum(pags) + 1
fis = torch.zeros(nb, BS, NKV, 2 * (D + 4), dtype=torch.int8, device=dev)        # NHD fisico
kv = fis.permute(0, 2, 1, 3)                                                    # logico [nb, NKV, BS, 264]
perm = torch.randperm(nb)[: sum(pags)].to(torch.int32)
bt = torch.zeros(B, 64, dtype=torch.int32)
Ks, Vs, o = [], [], 0
def cuant(x):
    a = x.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    return (x / a * 127).round().to(torch.int8), (a / 127).float()
for b, n in enumerate(LENS):
    pg = perm[o:o + pags[b]]; o += pags[b]; bt[b, :pg.numel()] = pg
    k = torch.randn(n, NKV, D, device=dev) * 2; v = torch.randn(n, NKV, D, device=dev)
    ki, ks = cuant(k); vi, vs = cuant(v)
    blk = pg.to(dev).long()[torch.arange(n, device=dev) // BS]; sl = torch.arange(n, device=dev) % BS
    fis[blk, sl, :, :D] = ki; fis[blk, sl, :, D + 4:2 * D + 4] = vi
    fis[blk, sl, :, D:D + 4] = ks.view(torch.int8).view(n, NKV, 4); fis[blk, sl, :, 2 * D + 4:] = vs.view(torch.int8).view(n, NKV, 4)
    Ks.append(ki.float() * ks); Vs.append(vi.float() * vs)
bt = bt.to(dev); sl_ = torch.tensor(LENS, dtype=torch.int32, device=dev)
q = torch.randn(B * L, HQ, D, device=dev).half()
scale = 1 / math.sqrt(D)
k31 = Kernel("sk31_borrador_attn.cu", "sk31_borrador", warps=8); k31.cargar()
ku = Kernel("sk31_borrador_attn.cu", "sk31_union", warps=4); ku.cargar()
RB = 64
Op = torch.empty(GMAX, B, NKV, RB, D, dtype=torch.float16, device=dev); Mp = torch.empty(GMAX, B, NKV, RB, device=dev); Lp = torch.empty_like(Mp)
out = torch.empty(B * L, HQ, D, dtype=torch.float16, device=dev)
sb, sh, ss = kv.stride(0), kv.stride(1), kv.stride(2)
SH = 2 * 64 * 144 + 2 * 64 * 128 + 2 * 64 * 136 * 2 + 64 * 72 * 2 + 2 * 64 * 2 * 4 + 2 * 64 * 4
import ctypes
def f():
    k31.lanzar((GMAX, B * NKV), [q, HQ * D, kv, ctypes.c_int64(sb), ctypes.c_int64(sh), ctypes.c_int64(ss), bt, bt.stride(0), sl_, NKV, G, L, BS, SW, scale * math.log2(math.e),
                                 GMAX, Op, Mp, Lp], shared=SH)
    ku.lanzar((RB, B * NKV), [Op, Mp, Lp, GMAX, B, NKV, L, G, out, HQ * D])
import ctypes
for i, a in enumerate([sb, sh, ss]): pass
f(); torch.cuda.synchronize()
errs = []
for b, n in enumerate(LENS):
    qf = q[b * L:(b + 1) * L].float().view(L, NKV, G, D)
    sc = torch.einsum("jhgd,thd->hjgt", qf, Ks[b]) * scale
    qp = n - L + torch.arange(L, device=dev)[:, None]; kp = torch.arange(n, device=dev)[None, :]
    vis = ((qp - kp) < SW) & ((kp - qp) < SW)
    sc = sc.masked_fill(~vis[None, :, None, :], float("-inf"))
    ref = torch.einsum("hjgt,thd->jhgd", sc.softmax(-1), Vs[b]).reshape(L, HQ, D)
    errs.append(float((out[b * L:(b + 1) * L].float() - ref).norm() / ref.norm()))
def t_grafo(fn, it=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it): fn()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(5): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / (5 * it)
print(f"LENS={LENS} GMAX={GMAX}: error vs float (KV decuantizada) " + " ".join(f"{e:.3%}" for e in errs) + f" | grafo {t_grafo(f):.1f} us por capa", flush=True)
print(f"   solo decode {t_grafo(lambda: k31.lanzar((GMAX, B * NKV), [q, HQ * D, kv, ctypes.c_int64(sb), ctypes.c_int64(sh), ctypes.c_int64(ss), bt, bt.stride(0), sl_, NKV, G, L, BS, SW, scale * math.log2(math.e), GMAX, Op, Mp, Lp], shared=SH)):.1f} us"
      f"  solo union {t_grafo(lambda: ku.lanzar((RB, B * NKV), [Op, Mp, Lp, GMAX, B, NKV, L, G, out, HQ * D])):.1f} us", flush=True)
# referencia de hoy: unified_attention de vLLM (Triton) sobre la MISMA KV, como la llama el backend
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode
key_cache, value_cache = kv.transpose(1, 2).split(D + 4, dim=-1)
base = torch.tensor([], dtype=torch.float32, device=dev).set_(kv.untyped_storage())
st = kv.stride()
ksc = torch.as_strided(base, (nb, BS, NKV), (st[0] // 4, st[2] // 4, st[1] // 4), D // 4)
vsc = torch.as_strided(base, (nb, BS, NKV), (st[0] // 4, st[2] // 4, st[1] // 4), (2 * D + 4) // 4)
otri = torch.empty_like(out)
cu = torch.arange(0, B * L + 1, L, dtype=torch.int32, device=dev)
unified_attention(q=q, k=key_cache, v=value_cache, out=otri, cu_seqlens_q=cu, max_seqlen_q=L, seqused_k=sl_,
                  max_seqlen_k=max(LENS), softmax_scale=scale, causal=False, window_size=(SW - 1, 0), block_table=bt,
                  softcap=0, q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=0, num_par_softmax_segments=16,
                  softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None,
                  kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, k_scale_cache=ksc, v_scale_cache=vsc)
torch.cuda.synchronize()
et = []
for b, n in enumerate(LENS):
    qf = q[b * L:(b + 1) * L].float().view(L, NKV, G, D)
    sc = torch.einsum("jhgd,thd->hjgt", qf, Ks[b]) * scale
    qp = n - L + torch.arange(L, device=dev)[:, None]; kp = torch.arange(n, device=dev)[None, :]
    vis = ((qp - kp) < SW) & ((kp - qp) < SW)
    ref = torch.einsum("hjgt,thd->jhgd", sc.masked_fill(~vis[None, :, None, :], float("-inf")).softmax(-1), Vs[b]).reshape(L, HQ, D)
    et.append(float((otri[b * L:(b + 1) * L].float() - ref).norm() / ref.norm()))
print("   Triton de hoy, error vs la misma referencia: " + " ".join(f"{e:.3%}" for e in et)
      + f" | SK-31 vs Triton {float((out.float() - otri.float()).norm() / otri.float().norm()):.3%}", flush=True)
# por el enganche de PN124 (lo que usa el servidor), dentro y fuera de un grafo
os.environ["GENESIS_PN124_SK31"] = "1"
from vllm._genesis import borrador_attn as ba
ba.ACTIVO = True; ba.GMAX = GMAX
kwt = dict(q=q, k=key_cache, v=value_cache, out=otri, cu_seqlens_q=cu, max_seqlen_q=L, seqused_k=sl_, max_seqlen_k=max(LENS),
           softmax_scale=scale, causal=False, window_size=(SW - 1, 0), block_table=bt, softcap=0, q_descale=None, k_descale=None,
           v_descale=None, seq_threshold_3D=0, num_par_softmax_segments=16, softmax_segm_output=None, softmax_segm_max=None,
           softmax_segm_expsum=None, kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, k_scale_cache=ksc, v_scale_cache=vsc)
ref_tri = otri.clone()
assert ba.intentar(kwt)
torch.cuda.synchronize()
print(f"   enganche: SK-31 vs Triton {float((otri.float() - ref_tri.float()).norm() / ref_tri.float().norm()):.3%}", flush=True)
