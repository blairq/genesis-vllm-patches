"""Inventario de kernels de un paso de decode: por nombre, us por paso, lanzamientos por paso y origen."""
import gzip, json, glob, re, sys, collections

def origen(n):
    if re.match(r"(sk\d+|gdn_arbol|arbol_|pn\d+|_k_|genesis)", n): return "Genesis PTX/CUDA"
    if "marlin" in n.lower(): return "Marlin (parcheado)"
    if n.startswith("triton_") or "triton" in n: return "Triton (inductor)"
    if "flashinfer" in n: return "flashinfer"
    if "nccl" in n.lower(): return "NCCL"
    if re.search(r"gemm|cutlass|sm80_xmma|s16816|splitK", n): return "cuBLAS/cutlass"
    if n.startswith("void at::native") or "at::native" in n: return "torch nativo"
    if n.startswith("void vllm::") or "vllm::" in n: return "vLLM C++"
    if n.startswith("_") or re.match(r"^[a-z_0-9]+$", n): return "Triton (vLLM/propio)"
    if "memcpy" in n.lower() or "Memcpy" in n or "Memset" in n: return "memcpy/memset"
    return "otro"

def cargar(d):
    f = sorted(glob.glob(d + "/rank0.*.gz"))[-1]
    t = json.load(gzip.open(f))["traceEvents"]
    return [e for e in t if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]

for d in sys.argv[1:]:
    ev = cargar(d)
    # pasos = lanzamientos del lm_head del target (1 por paso) aproximado por el kernel mas frecuente de 1 por paso
    nombres = collections.Counter(e["name"] for e in ev)
    ref = [n for n in nombres if "sk18h_salida" in n]
    pasos = nombres[ref[0]] / 16 if ref else 1
    agg = collections.defaultdict(lambda: [0.0, 0])
    for e in ev:
        n = re.sub(r"<.*", "", e["name"])[:70]
        agg[n][0] += e["dur"]; agg[n][1] += 1
    tot = sum(v[0] for v in agg.values()) / pasos
    print(f"== {d}: {pasos:.0f} pasos, {tot:.0f} us de GPU por paso, {sum(v[1] for v in agg.values()) / pasos:.0f} lanzamientos por paso")
    porig = collections.defaultdict(lambda: [0.0, 0, 0])
    for n, (us, c) in agg.items():
        o = origen(n); porig[o][0] += us / pasos; porig[o][1] += c / pasos; porig[o][2] += 1
    for o, (us, c, k) in sorted(porig.items(), key=lambda kv: -kv[1][0]):
        print(f"   {o:24} {us:8.0f} us  {100 * us / tot:5.1f}%  {c:6.0f} lanz/paso  {k:3d} kernels distintos")
    print("   -- no-Genesis por costo:")
    for n, (us, c) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        o = origen(n)
        if o in ("Genesis PTX/CUDA",): continue
        if us / pasos < 5: continue
        print(f"   {us / pasos:8.1f} us {c / pasos:6.1f}x  [{o}] {n}")
