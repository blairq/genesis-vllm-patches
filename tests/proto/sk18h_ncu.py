"""SK-18h decode (batch2 + union4 + salida) sobre una KV int8 con la geometria del servidor (paginas de 880,
2 cabezas KV por rango, 1 pedido de 9 tokens x 6 cabezas Q = 64 filas), datos al azar, para ncu y tiempos.
  GENESIS_SK18H_NW=4|8 elige el batch2. N = tokens de contexto (argv[1], 62000 por omision)."""
import os, sys, types, time, torch
os.environ.setdefault("GENESIS_PN131_MAXTOK", "9")
from vllm._genesis import sk18_attn as P
from vllm.v1.kv_cache_interface import KVQuantMode
dev = "cuda"; D = 256; NH = 2; G = 6; BS = 880; L = 9
N = int(sys.argv[1]) if len(sys.argv) > 1 else 62000
torch.manual_seed(0)
impl = types.SimpleNamespace(_kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD, head_size=D, num_heads=12,
                             alibi_slopes=None, sinks=None, sliding_window=(-1, -1), logits_soft_cap=0,
                             scale=1.0 / 16, chunk_lookback=-1, use_td=False, num_kv_heads=NH)
layer = types.SimpleNamespace(_k_scale=torch.ones(1, device=dev), _v_scale=torch.ones(1, device=dev), layer_name="capa_ncu")
nb = (N + BS - 1) // BS + 2
kv = torch.zeros(nb, BS, NH, 520, dtype=torch.int8, device=dev).permute(0, 2, 1, 3)
perm = torch.randperm(nb)[: (N + BS - 1) // BS].to(torch.int32)
slots = (perm[torch.arange(N) // BS].to(torch.int64) * BS + torch.arange(N) % BS).to(dev)
for t0 in range(0, N, 8192):
    t1 = min(N, t0 + 8192)
    k = (torch.randn(t1 - t0, NH, D, device=dev) * 2).half(); v = torch.randn(t1 - t0, NH, D, device=dev).half()
    P.escribir(impl, layer, k, v, kv, slots[t0:t1])
from vllm.v1.attention.backends.triton_attn import MIN_LAUNCH_GRID_SIZE_2D, NUM_PAR_SOFTMAX_SEGMENTS
thr = MIN_LAUNCH_GRID_SIZE_2D // NH
bt = torch.zeros(1, 316, dtype=torch.int32); bt[0, :perm.numel()] = perm
md = types.SimpleNamespace(num_actual_tokens=L, max_query_len=L, query_start_loc=torch.tensor([0, L], dtype=torch.int32, device=dev),
    max_seq_len=N, seq_lens=torch.tensor([N], dtype=torch.int32, device=dev), block_table=bt.to(dev), causal=True,
    genesis_qsl_cpu=torch.tensor([0, L], dtype=torch.int32), seq_threshold_3D=thr, num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
    softmax_segm_output=torch.empty(thr, 12, NUM_PAR_SOFTMAX_SEGMENTS, 256, device=dev),
    softmax_segm_max=torch.empty(thr, 12, NUM_PAR_SOFTMAX_SEGMENTS, device=dev),
    softmax_segm_expsum=torch.empty(thr, 12, NUM_PAR_SOFTMAX_SEGMENTS, device=dev),
    mm_prefix_range_tensor=None, rswa_prefix_lens=None, rswa_window=None)
q = torch.randn(L, 12, D, device=dev).half(); out = torch.zeros(L, 12, D, dtype=torch.float16, device=dev)
f = lambda: P._decode_uniforme(impl, q, kv, md, out, 1, L, False)
for _ in range(5): f()
torch.cuda.synchronize()
if os.environ.get("NCU") != "1":
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(50): f()
    e1.record(); torch.cuda.synchronize()
    print(f"N={N} NW={P.NW8}: decode entero {e0.elapsed_time(e1)*1000/50:.1f} us por llamada (batch2+union+salida+prep)")
else:
    torch.cuda.cudart().cudaProfilerStart(); f(); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
