"""huecos_cpu.py <dir_traza>: para cada hueco de la GPU (> 15 us) en los pasos de decode, que lanzamiento
lo cierra y que operacion de CPU (la mas externa) lo contenia. Agrega por nombre de operacion."""
import gzip, json, glob, sys, collections, bisect
f = sorted(glob.glob(sys.argv[1] + "/rank0*"))[0]
ev = json.load(gzip.open(f, "rt"))["traceEvents"]
ker = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy") and "dur" in e), key=lambda e: e["ts"])
rt = {e["args"].get("correlation"): e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "args" in e}
ops = sorted((e for e in ev if e.get("cat") in ("cpu_op", "user_annotation") and "dur" in e), key=lambda e: e["ts"])
ts_ops = [e["ts"] for e in ops]
def pila(t):
    """operaciones de CPU activas en t (de la mas externa a la mas interna)."""
    i = bisect.bisect_right(ts_ops, t)
    act = [e for e in ops[max(0, i - 4000):i] if e["ts"] <= t <= e["ts"] + e["dur"]]
    return sorted(act, key=lambda e: (e["ts"], -e["dur"]))
ua = [e for e in ev if e.get("cat") == "user_annotation"]
t0, t1 = ua[5]["ts"], ua[45]["ts"]            # 40 pasos del medio
agg = collections.defaultdict(lambda: [0.0, 0])
tot_idle = 0.0
fin = None
for k in ker:
    if k["ts"] < t0 or k["ts"] > t1:
        fin = max(fin or 0, k["ts"] + k["dur"]) if k["ts"] < t0 else fin
        continue
    if fin is not None and k["ts"] - fin > 15:
        gap = k["ts"] - fin
        tot_idle += gap
        corr = k["args"].get("correlation")
        lanz = rt.get(corr)
        p = pila(lanz["ts"]) if lanz else []
        nom = " > ".join(e["name"][:38] for e in p[1:4]) if p else "?"
        agg[nom][0] += gap; agg[nom][1] += 1
    fin = max(fin or 0, k["ts"] + k["dur"])
pasos = 40
print(f"GPU ociosa en huecos > 15 us: {tot_idle / pasos:.0f} us por paso")
for n, (g, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:25]:
    print(f"{g / pasos:7.0f} us/paso  {c / pasos:5.1f} huecos  {n}")
