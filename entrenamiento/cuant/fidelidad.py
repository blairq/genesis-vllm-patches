#!/usr/bin/env python3
"""Prueba de FIDELIDAD de un modelo servido contra la referencia BF16.

Para cada ventana pide a vLLM `prompt_logprobs=20` (max_tokens=1, cache_salt unico para que el
prefix-cache no le ahorre nada) y compara posicion por posicion con la referencia BF16:

  * KL(BF16 || modelo) sobre el soporte top-20 del BF16 (los tokens que no aparecen en el top-20
    del modelo toman su logprob numero 20, una cota: la KL sale subestimada, igual para todos);
  * acuerdo del top-1 con el BF16;
  * perplejidad del texto real (contra la del BF16).

Se reporta sobre todas las posiciones y solo sobre las de la RESPUESTA de noon.

Uso: fidelidad.py <url> <ventanas.npy> <ref_bf16.npz> <salida.json> [etiqueta]
"""
import json
import sys
import urllib.request
import uuid

import numpy as np

url, ven, ref, salida = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
etiqueta = sys.argv[5] if len(sys.argv) > 5 else ""
ids = np.load(ven)
resp = np.load(ven.replace(".npy", "_resp.npy"))
R = np.load(ref)
N, L = ids.shape
K = R["top_ids"].shape[-1]
kl = np.zeros((N, L - 1)); top1 = np.zeros((N, L - 1), bool); lp_real = np.zeros((N, L - 1))
for s in range(N):
    cuerpo = {"model": "qwen3.8", "prompt": ids[s].tolist(), "max_tokens": 1, "temperature": 0,
              "prompt_logprobs": K, "cache_salt": uuid.uuid4().hex}
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(cuerpo).encode(),
                                 headers={"Content-Type": "application/json"})
    pl = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["prompt_logprobs"]
    for p in range(L - 1):
        d = pl[p + 1]                                   # distribucion que predice el token p+1
        top = sorted(((int(t), v["logprob"]) for t, v in d.items() if v.get("rank", 99) <= K), key=lambda x: -x[1])
        mod = dict((int(t), v["logprob"]) for t, v in d.items())
        piso = top[-1][1] if top else -30.0
        lp_real[s, p] = mod.get(int(ids[s, p + 1]), piso)
        ri, rl = R["top_ids"][s, p], R["top_lp"][s, p].astype(np.float64)
        pr = np.exp(rl)
        lq = np.array([mod.get(int(t), piso) for t in ri])
        kl[s, p] = float((pr * (rl - lq)).sum())
        top1[s, p] = bool(top and top[0][0] == int(ri[0]))
    if s % 12 == 11:
        print(f"  {s + 1}/{N}", flush=True)

es_resp = np.zeros((N, L - 1), bool)
for s in range(N):
    es_resp[s, L - 1 - resp[s]:] = True
def resumen(msk):
    return {"kl": float(kl[msk].mean()), "top1_acuerdo": float(top1[msk].mean()),
            "ppl": float(np.exp(-lp_real[msk].mean())), "ppl_bf16": float(np.exp(-R["lp_real"][msk].mean())),
            "posiciones": int(msk.sum())}
out = {"etiqueta": etiqueta, "todo": resumen(np.ones_like(es_resp)), "respuesta": resumen(es_resp)}
json.dump(out, open(salida, "w"), indent=1)
for k in ("todo", "respuesta"):
    r = out[k]
    print(f"{etiqueta:8s} {k:9s} KL {r['kl']:.4f}  top-1 acuerdo {r['top1_acuerdo']:.4f}  "
          f"ppl {r['ppl']:.3f} (BF16 {r['ppl_bf16']:.3f})  [{r['posiciones']} pos]", flush=True)
