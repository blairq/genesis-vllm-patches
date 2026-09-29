"""SK-33 (PN161) contra el camino de siempre: RMSNorm q/k + rope neox + FWHT de PN126 + escritura KV int8 por
token-cabeza de vLLM (triton), sobre una cache con el layout de vLLM (registros K 128 | f32 | V 128 | f32)."""
import os, torch
os.environ.setdefault("GENESIS_ENABLE_PN126_ROT_QK", "1")
torch.zeros(1, device="cuda")
from vllm._genesis.kernels.ptx_lab import Kernel
from vllm._genesis import rot_qk as g126
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash_per_token_head_quant
from vllm.v1.attention.backends.triton_attn import KVQuantMode
from vllm.config import VllmConfig, set_current_vllm_config

dev = "cuda"; NHQ, NKV, D, BS, NB = 16, 4, 128, 16, 64
torch.manual_seed(0)
with set_current_vllm_config(VllmConfig()):
    re = get_rope(D, max_position=262144, is_neox_style=True, rope_parameters={"rope_theta": 10000000, "rope_type": "default"}).to(dev)
cs16 = re.cos_sin_cache.to(dev, torch.float16).contiguous()
signos = g126._signos_dev(D, torch.device(dev))
k33 = Kernel("sk33_borrador_qk_kv.cu", "sk33_qk_kv", defs=["-DROT=1"], warps=4); k33.cargar()


def caches():
    kv = torch.zeros(NB, NKV, BS, 2 * (D + 4), dtype=torch.int8, device=dev)
    kc, vc = kv.transpose(1, 2).split(D + 4, dim=-1)                    # [NB, BS, NKV, 132]
    f = torch.tensor([], dtype=torch.float32, device=dev).set_(kv.untyped_storage())
    st = [x // 4 for x in kv.stride()]
    ks = torch.as_strided(f, (NB, BS, NKV), (st[0], st[2], st[1]), D // 4)
    vs = torch.as_strided(f, (NB, BS, NKV), (st[0], st[2], st[1]), (D + 4 + D) // 4)
    return kv, kc, vc, ks, vs


def rms(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).half()


def rope_f32(pos, x):                                                     # como inductor: fp32 y un redondeo
    cos, sin = cs16.index_select(0, pos).float().chunk(2, dim=-1)
    x = x.float(); x1, x2 = x[..., :64], x[..., 64:]
    c, s = cos[:, None], sin[:, None]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), -1).half()


for T in (9, 36, 200):
    qkv = (torch.randn(T, (NHQ + 2 * NKV) * D, device=dev) * 3).half()
    qkv[:, NHQ * D + 5] *= 20                                            # canal outlier de k
    wq = (torch.rand(D, device=dev) + 0.5).half(); wk = (torch.rand(D, device=dev) + 0.5).half()
    pos = torch.randint(0, 200000, (T,), device=dev)
    slots = torch.randperm(NB * BS, device=dev)[:T].to(torch.int64)
    slots[1] = -1                                                         # relleno: no se escribe
    # referencia
    q = rms(qkv[:, :NHQ * D].reshape(T, NHQ, D), wq)
    k = rms(qkv[:, NHQ * D:(NHQ + NKV) * D].reshape(T, NKV, D), wk)
    v = qkv[:, (NHQ + NKV) * D:].reshape(T, NKV, D).contiguous()
    q, k = rope_f32(pos, q).contiguous(), rope_f32(pos, k).contiguous()
    g126.rotar_tensor(q, D); g126.rotar_tensor(k, D)
    kvr, kcr, vcr, ksr, vsr = caches()
    triton_reshape_and_cache_flash_per_token_head_quant(k, v, kcr, vcr, ksr, vsr, slots, KVQuantMode.INT8_PER_TOKEN_HEAD)
    # SK-33
    kvs, kcs, vcs, kss, vss = caches()
    qo = torch.empty(T, NHQ * D, dtype=torch.float16, device=dev)
    import ctypes
    c64 = lambda t: [t] + [ctypes.c_int64(int(x)) for x in t.stride()[:3]]
    k33.lanzar((T, -(-(NHQ + NKV) // 4)), [qkv, qkv.stride(0), pos, wq, wk, 1e-6, cs16, signos, NHQ, NKV, qo, qo.stride(0),
                                             slots, BS] + c64(kcs) + c64(vcs) + c64(kss) + c64(vss), sync=True)
    dq = (qo.float() - q.reshape(T, -1).float()).abs()
    lsb = (dq / q.reshape(T, -1).float().abs().clamp_min(1e-3) * 1024).max().item()
    distintos_q = (qo != q.reshape(T, -1)).float().mean().item()
    dk = (kvs[:, :, :, :].to(torch.int16) - kvr.to(torch.int16))
    # bytes de datos (no los de las escalas): diferencia en unidades int8
    datos = torch.ones(2 * (D + 4), dtype=torch.bool, device=dev); datos[D:D + 4] = False; datos[2 * D + 4:] = False
    dd = dk[..., datos].abs()
    ds = ((kss - ksr).abs() / ksr.abs().clamp_min(1e-9)).max().item(), ((vss - vsr).abs() / vsr.abs().clamp_min(1e-9)).max().item()
    print(f"T={T}: q distintos {distintos_q:.4f} (max {lsb:.1f} ulp rel) | KV bytes distintos {(dd > 0).float().mean().item():.5f} "
          f"max |d| {int(dd.max())} | escalas rel k {ds[0]:.1e} v {ds[1]:.1e} | cache identica {torch.equal(kvs, kvr)}")


def t_grafo(f, it=20):
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


for T in (9, 36):
    qkv = (torch.randn(T, (NHQ + 2 * NKV) * D, device=dev) * 3).half()
    pos = torch.randint(0, 200000, (T,), device=dev); slots = torch.arange(T, device=dev, dtype=torch.int64)
    kvs, kcs, vcs, kss, vss = caches(); qo = torch.empty(T, NHQ * D, dtype=torch.float16, device=dev)
    args = [qkv, qkv.stride(0), pos, wq, wk, 1e-6, cs16, signos, NHQ, NKV, qo, qo.stride(0), slots, BS] + c64(kcs) + c64(vcs) + c64(kss) + c64(vss)
    q3 = qkv[:, :NHQ * D].reshape(T, NHQ, D).contiguous(); k3 = qkv[:, NHQ * D:(NHQ + NKV) * D].reshape(T, NKV, D).contiguous()
    v3 = qkv[:, (NHQ + NKV) * D:].reshape(T, NKV, D).contiguous()
    print(f"GRAFO T={T}: SK-33 {t_grafo(lambda: k33.lanzar((T, -(-(NHQ + NKV) // 4)), args)):.1f} us | "
          f"solo fwht q+k {t_grafo(lambda: (g126.rotar_tensor(q3, D), g126.rotar_tensor(k3, D))):.1f} us | "
          f"solo escritura KV triton {t_grafo(lambda: triton_reshape_and_cache_flash_per_token_head_quant(k3, v3, kcs, vcs, kss, vss, slots, KVQuantMode.INT8_PER_TOKEN_HEAD)):.1f} us")
