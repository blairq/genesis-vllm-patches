"""SK-34 (PN162) contra la referencia COMPILADA con inductor (como corre en el servidor): finish de la conv del
borrador (_grouped_conv de vLLM, lado 1) + fused_add_rms_norm de la IR de 0.29. Prueba las variantes de redondeo
RY/RC y dice cual coincide. Ademas vuelve a chequear la norma + rope de SK-33 contra inductor."""
import os, sys, torch
import torch.nn.functional as F
torch.zeros(1, device="cuda")
from vllm._genesis.kernels.ptx_lab import Kernel

dev = "cuda"; N = 5120; GS = 16; G = N // GS; TAPS = 2; BSQ = 9; EPS = 1e-6
torch.manual_seed(0)


def grouped_conv(hs, delta, base, block_size, num_groups, group_size, taps):       # copia de qwen3_dflash2.py
    blocks = hs.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hs.shape[0], device=hs.device)
    position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


def fused_add_rms_norm(x, x_residual, weight, epsilon):                            # copia de vllm/ir/ops/layernorm.py
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + x_residual.to(torch.float32)
    x_residual = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(weight.dtype) * weight
    return x.to(orig_dtype), x_residual


def ref(h, c1, base1, res, w):
    y = grouped_conv(h, c1, base1, BSQ, G, GS, TAPS)
    return fused_add_rms_norm(y, res, w, EPS)


ref_c = torch.compile(ref, dynamic=False)
ks = {}
for ry in (0, 1):
    for rc in (0, 1):
        k = Kernel("sk34_borrador_fin_norma.cu", "sk34_fin_norma", defs=[f"-DRY={ry}", f"-DRC={rc}", "-DNT=640"], warps=20); k.cargar()
        ks[(ry, rc)] = k

for T in (9, 36, 72):
    h = (torch.randn(T, N, device=dev) * 2).half()
    proj = (torch.randn(T, 2, TAPS, G, device=dev) * 0.3).half()                 # como sale kernel_projection
    c1 = proj[:, 1]                                                                # vista, stride(0) = 4G
    base1 = (torch.randn(TAPS, N, device=dev) * 0.5).half()
    res = (torch.randn(T, N, device=dev) * 4).half()
    w = (torch.rand(N, device=dev) + 0.5).half()
    oc, rc_ = ref_c(h, c1, base1, res, w)
    oe, re_ = ref(h, c1, base1, res, w)
    print(f"T={T}: eager vs inductor: out distintos {(oe != oc).float().mean().item():.5f}, residuo {(re_ != rc_).float().mean().item():.5f}")
    for key, k in ks.items():
        out = torch.empty_like(h); ro = torch.empty_like(h)
        k.lanzar((T, 1), [h, h.stride(0), c1, c1.stride(0), base1, res, res.stride(0), w, EPS, N, BSQ, out, out.stride(0), ro, ro.stride(0)], sync=True)
        print(f"   RY={key[0]} RC={key[1]}: out distintos {(out != oc).float().mean().item():.5f} (max |d| {(out.float() - oc.float()).abs().max().item():.2e})"
              f", residuo distintos {(ro != rc_).float().mean().item():.5f}")


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
    h = (torch.randn(T, N, device=dev) * 2).half(); proj = (torch.randn(T, 2, TAPS, G, device=dev) * 0.3).half(); c1 = proj[:, 1]
    base1 = (torch.randn(TAPS, N, device=dev) * 0.5).half(); res = (torch.randn(T, N, device=dev) * 4).half(); w = (torch.rand(N, device=dev) + 0.5).half()
    out = torch.empty_like(h); ro = torch.empty_like(h); k = ks[(0, 0)]
    a = [h, h.stride(0), c1, c1.stride(0), base1, res, res.stride(0), w, EPS, N, BSQ, out, out.stride(0), ro, ro.stride(0)]
    ref_c(h, c1, base1, res, w)
    print(f"GRAFO T={T}: SK-34 {t_grafo(lambda: k.lanzar((T, 1), a)):.1f} us | inductor {t_grafo(lambda: ref_c(h, c1, base1, res, w)):.1f} us")

# --- SK-33: norma + rope contra inductor (la referencia de sk33_test.py era una formula propia) ---
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.config import VllmConfig, set_current_vllm_config
with set_current_vllm_config(VllmConfig()):
    rope = get_rope(128, max_position=262144, is_neox_style=True, rope_parameters={"rope_theta": 10000000, "rope_type": "default"}).to(dev)


def rms_ir(x, weight, epsilon):                                                    # copia de vllm/ir/ops/layernorm.py
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


def qk_ref(pos, qkv, wq, wk):
    T = qkv.shape[0]
    q, k, v = qkv.split([16 * 128, 4 * 128, 4 * 128], dim=-1)
    q = rms_ir(q.view(T, 16, 128), wq, EPS).view(T, -1)
    k = rms_ir(k.view(T, 4, 128), wk, EPS).view(T, -1)
    return rope.forward_native(pos, q, k)


qk_c = torch.compile(qk_ref, dynamic=False)
os.environ.setdefault("GENESIS_ENABLE_PN126_ROT_QK", "1")
k33 = {}
for rn in (0,):
    kk = Kernel("sk33_borrador_qk_kv.cu", "sk33_qk_kv", defs=["-DROT=0"], warps=4); kk.cargar(); k33[rn] = kk
import ctypes
cs16 = rope.cos_sin_cache.to(dev, torch.float32).contiguous()
for T in (9, 72):
    qkv = (torch.randn(T, 24 * 128, device=dev) * 3).half(); wq = (torch.rand(128, device=dev) + 0.5).half(); wk = (torch.rand(128, device=dev) + 0.5).half()
    pos = torch.randint(0, 200000, (T,), device=dev)
    qr, kr = qk_c(pos, qkv, wq, wk)
    for rn, kk in k33.items():
        qo = torch.empty(T, 16 * 128, dtype=torch.float16, device=dev)
        nulo = ctypes.c_uint64(0)
        kk.lanzar((T, 5), [qkv, qkv.stride(0), pos, wq, wk, EPS, cs16, torch.ones(128, dtype=torch.int32, device=dev), 16, 4, qo, qo.stride(0),
                          nulo, 1] + [nulo, ctypes.c_int64(0), ctypes.c_int64(0), ctypes.c_int64(0)] * 4, sync=True)
        print(f"SK-33 norma+rope T={T} RN={rn}: q distintos de inductor {(qo != qr).float().mean().item():.5f} (max |d| {(qo.float() - qr.float()).abs().max().item():.2e})")
