"""PN124: kernel 3D (KV partida en segmentos) con varias queries por secuencia, contra el 2D de siempre.
Forma del borrador DFlash2: 9 queries por pedido, no causal, ventana (2047, 0), 16 q / 4 kv por rango,
head 128, bloques de 880 (la KV del borrador hereda la geometria del target)."""
import os, torch
os.environ["GENESIS_ENABLE_PN124_TRITON_AMPERE"] = "1"
from vllm._genesis.wiring.hybrid import patch_PN124_triton_ampere as _w   # parchear ANTES de importar el kernel
print("PN124:", _w.apply())
from vllm._genesis import triton_attn_ampere as g
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.attention.backends.triton_attn import NUM_PAR_SOFTMAX_SEGMENTS
dev = "cuda"; HQ, HKV, D, BS, Q = 16, 4, 128, 880, 9
torch.manual_seed(0)
thr = 36
so = torch.empty(thr, HQ, NUM_PAR_SOFTMAX_SEGMENTS, D, device=dev); sm = torch.empty(thr, HQ, NUM_PAR_SOFTMAX_SEGMENTS, device=dev)
se = torch.empty(thr, HQ, NUM_PAR_SOFTMAX_SEGMENTS, device=dev)


def llamada(N, L):
    nb = (L + BS - 1) // BS
    kc = torch.randn(N * nb + 1, BS, HKV, D, device=dev).half(); vc = torch.randn(N * nb + 1, BS, HKV, D, device=dev).half()
    bt = torch.arange(N * nb, dtype=torch.int32, device=dev).view(N, nb)
    q = torch.randn(N * Q, HQ, D, device=dev).half()
    return dict(q=q, k=kc, v=vc, out=torch.empty_like(q), cu_seqlens_q=torch.arange(0, N * Q + 1, Q, dtype=torch.int32, device=dev),
                max_seqlen_q=Q, seqused_k=torch.full((N,), L, dtype=torch.int32, device=dev), max_seqlen_k=L,
                softmax_scale=D ** -0.5, causal=False, window_size=(2047, 0), block_table=bt, softcap=0,
                q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=thr, num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
                softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se)


def tiempo(f):
    for _ in range(3): f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(50): f()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / 50


for N in (1, 2, 4):
    for L in (3000, 20000, 62000):
        kw = llamada(N, L)
        g._Q3D[0] = 1; unified_attention(**kw); ref = kw["out"].clone(); t2 = tiempo(lambda: unified_attention(**kw))
        g._Q3D[0] = 16
        unified_attention(**kw); o3 = kw["out"].clone(); t3 = tiempo(lambda: unified_attention(**kw))
        d = float((o3.float() - ref.float()).abs().max()); m = float(ref.float().abs().max())
        print(f"N={N} L={L:6d}: 2D {t2:6.1f} us | 3D (segmentos sobre la ventana) {t3:6.1f} us rel {d / m:.1e}  x{t2 / t3:.2f}", flush=True)

# largos mezclados en el lote (uno por debajo de la ventana, otros no multiplos del tile)
kw = llamada(4, 40000)
kw["seqused_k"] = torch.tensor([700, 2100, 17777, 40000], dtype=torch.int32, device=dev)
g._Q3D[0] = 1; unified_attention(**kw); ref = kw["out"].clone()
g._Q3D[0] = 16; unified_attention(**kw); o3 = kw["out"].clone()
print("mezclado: rel", float((o3.float() - ref.float()).abs().max()) / float(ref.float().abs().max()), flush=True)
