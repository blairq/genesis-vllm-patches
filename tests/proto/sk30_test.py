"""SK-30 contra la atencion en float (k, v, q originales) y contra el decode actual (SK-18h), y tiempos en grafo."""
import os, sys, types, math, torch
os.environ.setdefault("GENESIS_PN131_MAXTOK", "9")
from vllm._genesis import sk18_attn as P
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; BS = 880; L = 9; HQ = 12; G = HQ // NH; GMAX = int(os.environ.get("GMAX", "128"))
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
q = torch.randn(L, HQ, D, device=dev).half()
# referencia float: q.k con los originales (la rotacion es ortonormal), mascara causal en los L nuevos
ctx = N - L
qf = q.float().view(L, NH, G, D)
sc = torch.einsum("jhgd,thd->hjgt", qf, K0.float()) * impl.scale                  # [NH, L, G, N]
pos = torch.arange(N, device=dev)
vis = (pos[None, :] < ctx) | (pos[None, :] - ctx <= torch.arange(L, device=dev)[:, None])   # [L, N]
sc = sc.masked_fill(~vis[None, :, None, :], float("-inf"))
ref = torch.einsum("hjgt,thd->jhgd", sc.softmax(-1), V0.float()).reshape(L, HQ, D)
# SK-18h (lo de hoy)
from vllm.v1.attention.backends.triton_attn import MIN_LAUNCH_GRID_SIZE_2D, NUM_PAR_SOFTMAX_SEGMENTS
thr = MIN_LAUNCH_GRID_SIZE_2D // NH
bt = torch.zeros(1, 316, dtype=torch.int32); bt[0, :perm.numel()] = perm; bt = bt.to(dev)
md = types.SimpleNamespace(num_actual_tokens=L, max_query_len=L, query_start_loc=torch.tensor([0, L], dtype=torch.int32, device=dev),
    max_seq_len=N, seq_lens=torch.tensor([N], dtype=torch.int32, device=dev), block_table=bt, causal=True,
    genesis_qsl_cpu=torch.tensor([0, L], dtype=torch.int32), seq_threshold_3D=thr, num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
    softmax_segm_output=torch.empty(thr, 12, NUM_PAR_SOFTMAX_SEGMENTS, 256, device=dev),
    softmax_segm_max=torch.empty(thr, 12, NUM_PAR_SOFTMAX_SEGMENTS, device=dev),
    softmax_segm_expsum=torch.empty(thr, 12, NUM_PAR_SOFTMAX_SEGMENTS, device=dev),
    mm_prefix_range_tensor=None, rswa_prefix_lens=None, rswa_window=None)
out18 = torch.zeros(L, HQ, D, dtype=torch.float16, device=dev)
f18 = lambda: P._decode_uniforme(impl, q, kv, md, out18, 1, L, False)
f18(); torch.cuda.synchronize()
# SK-30 con la Q de PRODUCCION: sk18h_prep2 (ARBOL + QSF) -> Qb int8, lim/abase/amask, qs float
RB = 64
ARB = os.environ.get("ARBOL_AZAR", "0") == "1"
if ARB:   # arbol al azar: cada token j tiene como ancestros un subconjunto de los anteriores (bit i -> token i+1)
    g_ = torch.Generator().manual_seed(3)
    par = [0] + [int(torch.randint(0, j, (1,), generator=g_)) for j in range(1, L)]   # padre de cada token (0 = ancla)
    anc_l = []
    for j in range(L):
        m_, x = 0, j
        while x > 0:
            x = par[x]
            if x > 0: m_ |= 1 << (x - 1)
        if j > 0: m_ |= 0                                     # (el propio token entra por lim/causal? no: bit j-1 = si mismo)
        anc_l.append(m_ | ((1 << (j - 1)) if j > 0 else 0))
    anc = torch.tensor(anc_l, dtype=torch.int32, device=dev)
else:
    anc = ((1 << torch.arange(L, device=dev, dtype=torch.int32)) - 1)
kp = Kernel("sk18h_prep2.cu", "sk18h_prep2", defs=["-DARBOL=1", "-DQSF=1"], warps=1); kp.cargar()
Qi = torch.zeros(1, NH, RB, D, dtype=torch.int8, device=dev)
lim_ = torch.empty(1, NH, RB, dtype=torch.int32, device=dev); mqb_ = torch.empty_like(lim_); dcap_ = torch.empty_like(lim_)
abase_ = torch.empty_like(lim_); amask_ = torch.empty_like(lim_); qs = torch.empty(1, NH, RB, device=dev)
sl = torch.tensor([N], dtype=torch.int32, device=dev)
q16 = q.view(L, HQ * D).view(torch.int16)
def fprep():
    kp.lanzar((L, HQ), [q16, sl, c.refs, P._signos_dev(torch.device(dev)), Qi, lim_, mqb_, dcap_, anc, abase_, amask_, qs,
                        L, NH, G, RB, P.ZSH, HQ * D])
fprep(); torch.cuda.synchronize()
BLK = kv.stride(0)                                                                 # bytes por bloque (int8)
NQ = int(os.environ.get("NQ", "2"))
k30 = Kernel("sk30_decode_1pasada.cu", "sk30_decode", defs=[f"-DNQ={NQ}", f"-DACC16={os.environ.get('ACC16', '1')}", f"-DPV8={os.environ.get('PV8', '1')}", f"-DPHL={os.environ.get('PHL', '1')}"], warps=4 * NQ); k30.cargar()
ku = Kernel("sk30_decode_1pasada.cu", "sk30_union", warps=8); ku.cargar()
Op = torch.empty(GMAX, 1, NH, RB, D, dtype=torch.float16, device=dev)
Mp = torch.empty(GMAX, 1, NH, RB, device=dev); Lp = torch.empty_like(Mp)
out30 = torch.empty(1, L, HQ, D, dtype=torch.float16, device=dev)
SH = 2 * 64 * 272 + 2 * 256 * 80 + 64 * 72 * 2 + 2 * 64 * 4 + 2 * NQ * 64 * 4
def f30():
    fprep()
    k30.lanzar((GMAX, NH), [Qi, qs, kv, bt, bt.stride(0), sl, lim_, abase_, amask_, NH, BS, BLK, c.refs, GMAX, Op, Mp, Lp], shared=SH)
    ku.lanzar((RB, NH), [Op, Mp, Lp, sl, GMAX, 1, NH, L, G, out30])
if ARB:   # referencia con la mascara del arbol
    vis2 = torch.zeros(L, N, dtype=torch.bool, device=dev)
    for j in range(L):
        vis2[j, :ctx + 1] = True
        for i in range(31):
            if (anc_l[j] >> i) & 1 and ctx + 1 + i <= ctx + j: vis2[j, ctx + 1 + i] = True
    sc2 = torch.einsum("jhgd,thd->hjgt", qf, K0.float()) * impl.scale
    sc2 = sc2.masked_fill(~vis2[None, :, None, :], float("-inf"))
    ref = torch.einsum("hjgt,thd->jhgd", sc2.softmax(-1), V0.float()).reshape(L, HQ, D)
f30(); torch.cuda.synchronize()
o30 = out30[0].float(); o18 = out18.float()
rel = lambda a: float((a - ref).norm() / ref.norm())
print("arbol al azar:", ARB, "anc:", anc.tolist())
print(f"N={N}: error relativo contra float: SK-30 {rel(o30):.3%}  SK-18h {rel(o18):.3%} | SK-30 vs SK-18h {float((o30 - o18).norm() / o18.norm()):.3%}")
det = all(torch.equal((f30(), out30.clone())[1], out30) for _ in range(3))
print("SK-30 determinista en 3 corridas:", det)
def t_grafo(f, it=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it): f()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(5): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / (5 * it)
print(f"GRAFO N={N}: SK-18h entero {t_grafo(f18):.1f} us  |  SK-30 (decode+union) {t_grafo(f30):.1f} us", flush=True)
print(f"GRAFO N={N} GMAX={GMAX} NQ={NQ}: solo decode {t_grafo(lambda: k30.lanzar((GMAX, NH), [Qi, qs, kv, bt, bt.stride(0), sl, lim_, abase_, amask_, NH, BS, BLK, c.refs, GMAX, Op, Mp, Lp], shared=SH)):.1f} us"
      f"  solo union {t_grafo(lambda: ku.lanzar((RB, NH), [Op, Mp, Lp, sl, GMAX, 1, NH, L, G, out30])):.1f} us", flush=True)
