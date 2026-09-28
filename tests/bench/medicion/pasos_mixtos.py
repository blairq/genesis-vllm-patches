"""pasos_mixtos.py <dir_traza> [...]: los pasos con prefill dentro de un decode concurrente (fase decode4).

El analizador de pasos (analizar_perfil.py) los descarta por forma. Aca se miden aparte: por cada forward
del target cuyas filas (grilla de SK-23) no son las del decode puro, el tiempo de pared del forward y
cuanto se fue en descuantizar la KV (sk18h_decuant) y en la atencion de prefill (FlashInfer) y de decode
(sk18h_batch2). Sirve para comparar el camino de pasos mixtos (GENESIS_PN131_MIXTO) contra el completo.
"""
import collections, glob, gzip, json, statistics as st, sys

for d in sys.argv[1:]:
    f = sorted(glob.glob(d + "/rank0*"))[0]
    ev = json.load(gzip.open(f, "rt"))["traceEvents"]
    ker = sorted((e for e in ev if e.get("cat") == "kernel" and "dur" in e), key=lambda e: e["ts"])
    sk = [i for i, e in enumerate(ker) if "sk23_silu_had_q8" in e["name"]]
    fws = [sk[i:i + 64] for i in range(0, len(sk) - 63, 64)]
    filas = collections.Counter(ker[fw[0]]["args"]["grid"][0] for fw in fws)
    moda = filas.most_common(1)[0][0]
    res = []
    for fw in fws:
        rows = ker[fw[0]]["args"]["grid"][0]
        if rows == moda or rows <= 9:
            continue
        a, b = fw[0], fw[-1]
        seg = ker[a - 40:b + 5]
        t = (ker[b]["ts"] + ker[b]["dur"] - ker[a]["ts"]) / 1000
        dec = sum(e["dur"] for e in seg if "sk18h_decuant" in e["name"]) / 1000
        fi = sum(e["dur"] for e in seg if "BatchPrefill" in e["name"]) / 1000
        s18 = sum(e["dur"] for e in seg if "sk18h_batch2" in e["name"]) / 1000
        res.append((rows, t, dec, fi, s18))
    print(f"== {d}: {len(res)} pasos mixtos (moda del decode: {moda} filas)")
    if res:
        print(f"   forward mediano {st.median(r[1] for r in res):7.1f} ms | descuantizar {st.median(r[2] for r in res):6.2f} ms"
              f" | FlashInfer {st.median(r[3] for r in res):6.2f} ms | SK-18h {st.median(r[4] for r in res):6.2f} ms")
        print(f"   total de los pasos mixtos {sum(r[1] for r in res):8.1f} ms, descuantizar {sum(r[2] for r in res):7.1f} ms")
        for r in res[:12]:
            print(f"   filas {r[0]:5d}  forward {r[1]:7.1f} ms  decuant {r[2]:6.2f}  FI {r[3]:6.2f}  SK-18h {r[4]:6.2f}")
