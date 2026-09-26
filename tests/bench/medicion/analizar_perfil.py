#!/usr/bin/env python3
"""analizar_perfil.py <dir_traza> [rank]: el paso de idiotSavant kernel por kernel, capa por capa.

Lee una traza torch de perfil_idiotsavant.sh y corta cada paso del engine en regiones:
  embedding     vocab_parallel_embedding + su all-reduce (inicio del forward del target)
  capa 0..63    SK-23 corre UNA vez por capa del target (el borrador usa act_and_mul): cada capa termina
                en el all-reduce que sigue a su down_proj. GDN = capas con _k_spec_arbol, atencion = sk18h
  lm_head       norma final + lm_head W4A16 (Marlin fp16) + all-gather de los logits por vocab
  verificacion  muestreo/rejection del arbol, compactado, estado mamba
  borrador      DFlash2: fc + 5 capas + su lm_head + top-k
  arbol         armado del arbol y preparacion de entradas del paso siguiente
Por cada region: tiempo de pared, tiempo de GPU (union de kernels) y ocioso. Por cada kernel: modulo,
operacion y tipo de dato (entrada -> computo).
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import re
import statistics as st
import sys

I8 = "1125899906909952"      # vllm ScalarType de int8 en el 1er parametro de plantilla de Marlin


def corto(n):
    if n.startswith("void marlin::Marlin<"):
        return "Marlin W4A8 (IMMA s8)" if n.split("<")[1].startswith(I8) else "Marlin W4A16 (HMMA f16)"
    n = re.sub(r"^void ", "", n).replace("(anonymous namespace)::", "").replace("at::native::", "")
    m = re.match(r"([A-Za-z0-9_:]+)", n)
    b = m.group(1) if m else n[:40]
    if "elementwise" in b or "reduce_kernel" in b:
        return "torch " + b.split("::")[-1]
    if b.startswith("ampere_") or b.startswith("sm80_xmma") or b.startswith("cutlass") or b.startswith("cublasLt"):
        return "cuBLAS/cutlass " + ("splitK" if "splitK" in b else "GEMM fp16")
    if b.startswith("triton_"):
        return "triton inductor " + ("rms_norm" if "rms_norm" in b else "marlin_scale" if "marlin_gemm_s16" in b else "otros")
    if b.startswith("ncclDevKernel_"):
        return "NCCL " + b.split("_")[1]
    return b[:60]


TIPO = {
    "Marlin W4A8 (IMMA s8)": "int8 × int4 → s32",
    "Marlin W4A16 (HMMA f16)": "fp16 × int4 → f32",
    "sk23_silu_had_q8": "fp16 → f32 → int8",
    "_per_token_quant_int8": "fp16 → int8",
    "triton inductor marlin_scale": "f32",
    "triton inductor rms_norm": "fp16 → f32 → fp16",
    "cuBLAS/cutlass GEMM fp16": "fp16 HMMA → f32",
    "cuBLAS/cutlass splitK": "f32 → fp16",
    "kernel_unified_attention": "fp16 HMMA (Triton)",
    "_k_spec_arbol": "fp16 → f32 (recurrencia)",
    "_k_salidas": "f32 → fp16",
    "_k_escribir": "f32 → f32 (cinta)",
    "_reshape_cache_per_token_head": "fp16 → int8",
    "sk_fwht128_f16": "fp16",
    "NCCL AllReduce": "fp16",
    "NCCL AllGather": "fp16",
    "vllm::cross_device_reduce_1stage": "fp16 (P2P)",
}


def cargar(d, rank):
    f = sorted(glob.glob(os.path.join(d, f"rank{rank}*.json*")))[0]
    with (gzip.open if f.endswith(".gz") else open)(f, "rt") as h:
        ev = json.load(h)["traceEvents"]
    return sorted((e for e in ev if e.get("cat") == "kernel" and "dur" in e), key=lambda e: e["ts"])


def union(ks):
    tot, a, b = 0.0, None, None
    for s, e in sorted((k["ts"], k["ts"] + k["dur"]) for k in ks):
        if a is None or s > b:
            tot += (b - a) if a is not None else 0.0
            a, b = s, e
        else:
            b = max(b, e)
    return tot + ((b - a) if a is not None else 0.0)


def es_ar(n):
    return "AllReduce" in n or "cross_device" in n


def cortar(ker, nom, a, b, fw):
    reg = {}
    # embedding: hasta el all-reduce que le sigue inclusive
    j = a
    while not es_ar(nom[j]): j += 1
    reg["embedding"] = (a, j)
    prev = j
    for c, s in enumerate(fw):
        j = s
        while not es_ar(nom[j]): j += 1
        reg[f"capa {c:02d}"] = (prev + 1, j)
        prev = j
    # post: lm_head del target = hasta el 1er AllGather despues del 1er Marlin fp16
    m1 = next(x for x in range(prev + 1, b + 1) if "Marlin<" in nom[x] and not nom[x].split("<")[1].startswith(I8))
    g1 = next(x for x in range(m1, b + 1) if "AllGather" in nom[x])
    reg["lm_head + gather"] = (prev + 1, g1)
    q = next(x for x in range(g1, b + 1) if "_per_token_quant_int8" in nom[x])
    reg["verificación"] = (g1 + 1, q - 1)
    m2 = next(x for x in range(q, b + 1) if "Marlin<" in nom[x] and not nom[x].split("<")[1].startswith(I8))
    f2 = m2
    while f2 + 1 <= b and ("TopK" in nom[f2 + 1] or "AllGather" in nom[f2 + 1] or "copy" in nom[f2 + 1] or "add" in nom[f2 + 1]):
        f2 += 1
    reg["borrador: capas"] = (q, m2 - 1)
    reg["borrador: lm_head + top-k"] = (m2, f2)
    reg["árbol + preparar paso"] = (f2 + 1, b)
    return reg


def main():
    d = sys.argv[1]
    rank = sys.argv[2] if len(sys.argv) > 2 else "0"
    ker = cargar(d, rank)
    nom = [k["name"] for k in ker]
    sk = [i for i, n in enumerate(nom) if "sk23_silu_had_q8" in n]
    emb = [i for i, n in enumerate(nom) if "vocab_parallel_embedding" in n]
    # la cantidad de filas por paso (pedidos x nodos) no es visible en la traza; se agrupan las SK-23 de a 64
    fws = [sk[i:i + 64] for i in range(0, len(sk) - 63, 64)]
    # inicio del forward del target = el ultimo embedding antes de la SK-23 de la capa 0
    ini = [max(e for e in emb if e < fw[0]) for fw in fws]
    pasos, raros = [], collections.Counter()
    for k in range(len(fws) - 1):
        a, b = ini[k], ini[k + 1] - 1
        filas = ker[fws[k][0]]["args"].get("grid", [0])[0]      # SK-23: un bloque por fila (pedidos x nodos)
        try:
            reg = cortar(ker, nom, a, b, fws[k])
        except StopIteration:
            raros[filas] += 1
            continue
        reg["_filas"] = filas
        pasos.append(reg)
    pasos = pasos[1:]                        # el primero puede arrastrar el arranque del profiler
    filas = collections.Counter(p["_filas"] for p in pasos)
    moda = filas.most_common(1)[0][0]
    pasos = [p for p in pasos if p["_filas"] == moda]
    n = len(pasos)
    print(f"filas por paso (pedidos x nodos del arbol): {dict(filas)}; se analizan los {n} de {moda} filas; "
          f"descartados por forma distinta (prefill mezclado, sin borrador): {dict(raros)}")

    def med(fn):
        return st.median(fn(p) for p in pasos)

    pared_paso = med(lambda p: ker[p["árbol + preparar paso"][1]]["ts"] + ker[p["árbol + preparar paso"][1]]["dur"] - ker[p["embedding"][0]]["ts"])
    gpu_paso = med(lambda p: union(ker[p["embedding"][0]:p["árbol + preparar paso"][1] + 1]))
    print(f"{os.path.basename(d.rstrip('/'))} rank{rank}: {n} pasos | paso {pared_paso:.0f} us de pared, GPU ocupada {gpu_paso:.0f} us "
          f"({100 * gpu_paso / pared_paso:.0f}%), ociosa {pared_paso - gpu_paso:.0f} us | {med(lambda p: p['árbol + preparar paso'][1] - p['embedding'][0] + 1):.0f} kernels")

    def region(nombres):
        pared = med(lambda p: sum(ker[p[r][1]]["ts"] + ker[p[r][1]]["dur"] - ker[p[r][0]]["ts"] for r in nombres))
        gpu = med(lambda p: sum(union(ker[p[r][0]:p[r][1] + 1]) for r in nombres))
        nk = med(lambda p: sum(p[r][1] - p[r][0] + 1 for r in nombres))
        return pared, gpu, nk

    capas = [f"capa {c:02d}" for c in range(64)]
    tipo_capa = {}
    for c in capas:
        a, b = pasos[0][c]
        tipo_capa[c] = "atención" if any("sk18" in nom[x] or "unified_attention" in nom[x] or "BatchDecode" in nom[x] for x in range(a, b + 1)) else "GDN"
    gdn = [c for c in capas if tipo_capa[c] == "GDN"]
    att = [c for c in capas if tipo_capa[c] == "atención"]
    res = {"pasos": n, "paso_pared_us": pared_paso, "paso_gpu_us": gpu_paso, "regiones": {}, "capas": {}, "kernels": {}}
    print(f"\n{'región':28} {'pared us':>9} {'GPU us':>8} {'ociosa':>7} {'kernels':>8}")
    for nombre, grupo in [("embedding", ["embedding"]), (f"{len(gdn)} capas GDN", gdn), (f"{len(att)} capas atención", att),
                          ("lm_head + gather", ["lm_head + gather"]), ("verificación", ["verificación"]),
                          ("borrador: capas", ["borrador: capas"]), ("borrador: lm_head + top-k", ["borrador: lm_head + top-k"]),
                          ("árbol + preparar paso", ["árbol + preparar paso"])]:
        p, g, k = region(grupo)
        res["regiones"][nombre] = {"pared_us": p, "gpu_us": g, "kernels": k}
        print(f"{nombre:28} {p:9.0f} {g:8.0f} {p - g:7.0f} {k:8.0f}")
    for c in capas:
        p, g, k = region([c])
        res["capas"][c] = {"tipo": tipo_capa[c], "pared_us": p, "gpu_us": g}
    print("\ncapas GDN (pared us): " + " ".join(f"{int(c[5:])}:{res['capas'][c]['pared_us']:.0f}" for c in gdn))
    print("capas atención (pared us): " + " ".join(f"{int(c[5:])}:{res['capas'][c]['pared_us']:.0f}" for c in att))

    # kernel por kernel dentro de cada tipo de region: posicion en la capa -> nombre, us, tipo
    for grupo_n, grupo in [("capa GDN", gdn), ("capa atención", att), ("lm_head + gather", ["lm_head + gather"]),
                           ("verificación", ["verificación"]), ("borrador: capas", ["borrador: capas"]),
                           ("borrador: lm_head + top-k", ["borrador: lm_head + top-k"]), ("árbol + preparar paso", ["árbol + preparar paso"])]:
        acc = collections.defaultdict(lambda: [0.0, 0])
        for p in pasos:
            for r in grupo:
                a, b = p[r]
                for x in range(a, b + 1):
                    acc[corto(nom[x])][0] += ker[x]["dur"]; acc[corto(nom[x])][1] += 1
        div = n * (len(grupo) if grupo_n.startswith("capa") else 1)
        filas = sorted(acc.items(), key=lambda kv: -kv[1][0])
        res["kernels"][grupo_n] = [{"kernel": k_, "us": v[0] / div, "lanz": v[1] / div, "tipo": TIPO.get(k_, "")} for k_, v in filas]
        tot = sum(v[0] for v in acc.values()) / div
        print(f"\n== {grupo_n} (por {'capa' if grupo_n.startswith('capa') else 'paso'}; suma kernels {tot:.0f} us)")
        for k_, v in filas[:14]:
            print(f"  {v[0] / div:7.1f} us {100 * v[0] / div / tot:5.1f}% x{v[1] / div:4.1f}  {k_:42} {TIPO.get(k_, '')}")
    # secuencia de una capa de cada tipo, kernel por kernel
    for nombre, c in [("GDN", gdn[1]), ("atención", att[1])]:
        a, b = pasos[len(pasos) // 2][c]
        print(f"\n-- secuencia {nombre} ({c}): " + " | ".join(f"{corto(nom[x])} {ker[x]['dur']:.1f}" for x in range(a, b + 1)))
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analisis_perfil", f"{os.path.basename(d.rstrip('/'))}_rank{rank}.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
