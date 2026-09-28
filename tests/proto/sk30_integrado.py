"""Camino integrado: P._decode_uniforme con GENESIS_PN131_SK30=0/1 (se corre dos veces), B secuencias de largos
distintos, arbol en cadena. Referencia float por secuencia; tiempo en grafo."""
import os, sys, types, torch, json
os.environ.setdefault("GENESIS_PN131_MAXTOK", "9")
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; BS = 880; L = 9; HQ = 12; G = 6
LENS = [int(x) for x in os.environ.get("LENS", "20000").split(",")]; B = len(LENS)
torch.manual_seed(0)
impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, head_size=D, num_heads=HQ,
                             alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                             scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev), layer_name="capa_int")
paginas = [(n + BS - 1) // BS for n in LENS]
nb = sum(paginas) + 2
kv = torch.zeros(nb, BS, NH, 520, dtype=torch.int8, device=dev).permute(0, 2, 1, 3)
perm = torch.randperm(nb)[: sum(paginas)].to(torch.int32)
bt = torch.zeros(B, 316, dtype=torch.int32)
K0s, V0s, o = [], [], 0
for b, n in enumerate(LENS):
    pg = perm[o:o + paginas[b]]; o += paginas[b]; bt[b, :pg.numel()] = pg
    slots = (pg[torch.arange(n) // BS].to(torch.int64) * BS + torch.arange(n) % BS).to(dev)
    K0 = (torch.randn(n, NH, D, device=dev) * 2).half(); V0 = torch.randn(n, NH, D, device=dev).half()
    for t0 in range(0, n, 8192):
        P.escribir(impl, layer, K0[t0:t0 + 8192], V0[t0:t0 + 8192], kv, slots[t0:t0 + 8192])
    K0s.append(K0); V0s.append(V0)
bt = bt.to(dev)
q = torch.randn(B * L, HQ, D, device=dev).half()
md = types.SimpleNamespace(num_actual_tokens=B * L, max_query_len=L,
    query_start_loc=torch.arange(0, B * L + 1, L, dtype=torch.int32, device=dev), max_seq_len=max(LENS),
    seq_lens=torch.tensor(LENS, dtype=torch.int32, device=dev), block_table=bt, causal=True,
    genesis_qsl_cpu=torch.arange(0, B * L + 1, L, dtype=torch.int32))
out = torch.zeros(B * L, HQ, D, dtype=torch.float16, device=dev)
f = lambda: P._decode_uniforme(impl, q, kv, md, out, B, L, False)
f(); torch.cuda.synchronize()
errs = []
for b, n in enumerate(LENS):
    ctx = n - L
    qf = q[b * L:(b + 1) * L].float().view(L, NH, G, D)
    sc = torch.einsum("jhgd,thd->hjgt", qf, K0s[b].float()) / 16
    pos = torch.arange(n, device=dev)
    vis = (pos[None, :] < ctx) | (pos[None, :] - ctx <= torch.arange(L, device=dev)[:, None])
    sc = sc.masked_fill(~vis[None, :, None, :], float("-inf"))
    ref = torch.einsum("hjgt,thd->jhgd", sc.softmax(-1), V0s[b].float()).reshape(L, HQ, D)
    o_ = out[b * L:(b + 1) * L].float()
    errs.append(float((o_ - ref).norm() / ref.norm()))
def t_grafo(f, it=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it): P._decode_uniforme(impl, q, kv, md, out, B, L, True)
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(5): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / (5 * it)
print(f"SK30={int(P.SK30)} LENS={LENS}: error por secuencia " + " ".join(f"{e:.3%}" for e in errs) + f" | grafo {t_grafo(f):.1f} us por capa", flush=True)
