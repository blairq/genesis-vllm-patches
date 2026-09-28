"""huecos_py.py <dir_traza> [pasos]: los huecos de la GPU (> 8 us) de los pasos de decode, atribuidos a la
LINEA de Python (vllm/ o _genesis/) que lanzo el kernel que cierra el hueco. Necesita una traza con
torch_profiler_with_stack (PROF_STACK=true en perfil_idiotsavant.sh).

Imprime, por linea (la mas interna de vllm que aparece en la pila), cuanto hueco cierra por paso y la
cadena de llamadas de vllm hasta ahi, para saber que parte del runner se come la GPU.
"""
import bisect, collections, glob, gzip, json, sys

d = sys.argv[1]
f = sorted(glob.glob(d + "/rank0*"))[0]
ev = json.load(gzip.open(f, "rt"))["traceEvents"]
ker = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e),
             key=lambda e: e["ts"])
rt = {e["args"].get("correlation"): e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "args" in e}
py = sorted((e for e in ev if e.get("cat") == "python_function" and "dur" in e), key=lambda e: e["ts"])
ts_py = [e["ts"] for e in py]
ua = [e for e in ev if e.get("cat") == "user_annotation"]
pasos = int(sys.argv[2]) if len(sys.argv) > 2 else 20
i0 = len(ua) // 2 - pasos // 2
t0, t1 = ua[i0]["ts"], ua[i0 + pasos]["ts"]


def pila(t):
    i = bisect.bisect_right(ts_py, t)
    act = [e for e in py[max(0, i - 20000):i] if e["ts"] <= t <= e["ts"] + e["dur"]]
    act.sort(key=lambda e: (e["ts"], -e["dur"]))
    return [e["name"] for e in act]


def util(nombres):
    """frames de vllm (sin torch internos), de afuera hacia adentro."""
    out = []
    for n in nombres:
        if ("vllm/" in n or "_genesis" in n) and "site-packages/torch" not in n:
            out.append(n.split("dist-packages/")[-1].replace("vllm/", "", 1))
    return out


agg = collections.defaultdict(lambda: [0.0, 0, None])
tot = 0.0
fin = None
for k in ker:
    if k["ts"] < t0 - 5000:
        continue
    if k["ts"] > t1:
        break
    if fin is not None and k["ts"] - fin > 8 and k["ts"] >= t0:
        gap = k["ts"] - fin
        tot += gap
        l = rt.get(k["args"].get("correlation"))
        fr = util(pila(l["ts"])) if l else []
        clave = fr[-1] if fr else ("(grafo) " if l and "Graph" in l["name"] else "(sin pila) ") + k["name"][:40]
        a = agg[clave]
        a[0] += gap; a[1] += 1
        if a[2] is None:
            a[2] = " > ".join(x.split("(")[0].split("/")[-1] + x[x.find("("):x.find(")") + 1] for x in fr[-6:])
    fin = max(fin or 0, k["ts"] + k["dur"])
print(f"huecos > 8 us: {tot / pasos:.0f} us por paso ({pasos} pasos)")
for c, (g, n, cad) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:40]:
    print(f"{g / pasos:6.0f} us  {n / pasos:4.1f}x  {c}")
    if cad:
        print(f"                   {cad}")
