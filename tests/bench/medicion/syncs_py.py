"""syncs_py.py <dir_traza> [pasos]: llamadas BLOQUEANTES de la CPU (cudaStreamSynchronize, cudaDeviceSynchronize,
cudaEventSynchronize, cudaMemcpy sin Async y cudaMemcpyAsync a host) en los ultimos pasos de decode puro,
atribuidas a la linea de Python (necesita PROF_STACK=true). Tambien el tiempo de CPU por paso de Python propio."""
import bisect, collections, glob, gzip, json, sys
f = sorted(glob.glob(sys.argv[1] + "/rank0*"))[0]
ev = json.load(gzip.open(f, "rt"))["traceEvents"]
ua = sorted((e for e in ev if e.get("cat") == "user_annotation"), key=lambda e: e["ts"])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
dec = [i for i, e in enumerate(ua) if "context_0(0)" in e["name"] and "generation_0(0)" not in e["name"]]
i0 = max(dec[-1] - N, dec[0]); i1 = min(i0 + N, len(ua) - 1); N = i1 - i0
t0, t1 = ua[i0]["ts"], ua[i1]["ts"]
py = sorted((e for e in ev if e.get("cat") == "python_function" and "dur" in e), key=lambda e: e["ts"])
tp = [e["ts"] for e in py]
def pila(t):
    i = bisect.bisect_right(tp, t)
    act = sorted((e for e in py[max(0, i - 20000):i] if e["ts"] <= t <= e["ts"] + e["dur"]), key=lambda e: (e["ts"], -e["dur"]))
    return [e["name"] for e in act if ("vllm/" in e["name"] or "_genesis" in e["name"]) and "torch/" not in e["name"]]
BLOQ = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy")
agg = collections.defaultdict(lambda: [0.0, 0, ""])
for e in ev:
    if e.get("cat") not in ("cuda_runtime", "cuda_driver") or not (t0 <= e["ts"] < t1):
        continue
    n = e["name"]
    if not any(n.startswith(b) for b in BLOQ):
        continue
    if n.startswith("cudaMemcpyAsync") and e.get("args", {}).get("kind", "") not in ("", "DtoH", "DeviceToHost"):
        continue
    p = pila(e["ts"])
    clave = (n, p[-1] if p else "?")
    agg[clave][0] += e["dur"]; agg[clave][1] += 1; agg[clave][2] = " > ".join(x.split("/")[-1] for x in p[-5:])
tot = sum(v[0] for v in agg.values())
paso = (t1 - t0) / N
print(f"{N} pasos de {paso:.0f} us; CPU bloqueada en syncs/copias: {tot / N:.0f} us por paso")
for (n, l), (d, c, cadena) in sorted(agg.items(), key=lambda x: -x[1][0])[:15]:
    print(f"  {d / N:7.0f} us  {c / N:4.1f}x  {n[:28]:28s} {l.split('/')[-1][:60]}\n        {cadena}")
