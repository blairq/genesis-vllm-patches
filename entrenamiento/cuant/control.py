#!/usr/bin/env python3
"""Control de cordura del forward de la etapa A, separando RESPUESTA de noon y resto del contexto.

El control de calibrar.py mide los ultimos 512 tokens de cada muestra, que mezclan la respuesta
generada por noon con salida de herramientas (logs, archivos), muy dificil de predecir. Aca se
retoma desde un estado guardado (por defecto el de la capa 56), se corren las capas que faltan y
se mide top-1 y perplejidad por separado:
  * sobre los tokens de RESPUESTA de noon (temperatura 0,35): un modelo sano y casi igual tiene que
    acertar la gran mayoria;
  * sobre el resto de la ventana.

El mapeo muestra -> pedido se reconstruye igual que en armar_calibracion.py (mismo orden y mismo
filtro: pedidos capturados con prompt+respuesta >= L).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrar import Pesos, estado_cargar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--banco", required=True)
    ap.add_argument("--capturas", required=True)
    ap.add_argument("--desde", type=int, default=56)
    ap.add_argument("--n", type=int, default=32, help="muestras a medir")
    a = ap.parse_args()
    dev = "cuda"
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = json.load(open(os.path.join(a.modelo, "config.json")))
    tc = Qwen3_5TextConfig(**cfg["text_config"])
    tc._attn_implementation = "sdpa"
    P = Pesos(a.modelo)
    ids = np.load(a.calib)
    N, L = ids.shape

    # largo de la respuesta de cada muestra (mismo orden que armar_calibracion.py)
    dirs = {int(os.path.basename(d).split("_")[0]): d for d in glob.glob(os.path.join(a.capturas, "*"))
            if os.path.exists(os.path.join(d, "etiquetas.npz"))}
    resp = []
    for n in range(len(open(a.banco).readlines()) if False else 10 ** 9):
        if len(resp) >= N or n > max(dirs):
            break
        if n not in dirs:
            continue
        e = np.load(os.path.join(dirs[n], "etiquetas.npz"))
        if int(e["prompt_len"]) + len(e["comp_ids"]) >= L:
            resp.append(min(len(e["comp_ids"]), L))
    # verificacion: la cola de cada muestra tiene que ser la respuesta capturada
    print(f"{len(resp)} muestras mapeadas; respuesta media {np.mean(resp[:a.n]):.0f} tokens", flush=True)

    h = estado_cargar(os.path.join(a.cache, f"estado_{a.desde:02d}.npy"), dev)[: a.n]
    rot = m.Qwen3_5TextRotaryEmbedding(tc).to(dev)
    pos = torch.arange(L, device=dev)[None].expand(1, -1)
    cos, sin = rot(h[:1], pos)
    for i in range(a.desde, tc.num_hidden_layers):
        with torch.device("meta"):
            capa = m.Qwen3_5DecoderLayer(tc, i)
        pref = f"model.language_model.layers.{i}."
        sd = {k[len(pref):]: P.get(k, dev, torch.float32 if k.endswith("A_log") else torch.bfloat16) for k in P.claves(pref)}
        capa.load_state_dict(sd, assign=True, strict=True)
        with torch.no_grad():
            for s in range(a.n):
                o = capa(h[s:s + 1], position_embeddings=(cos, sin), attention_mask=None, position_ids=pos)
                h[s:s + 1] = o[0] if isinstance(o, tuple) else o
        del capa, sd
        print(f"capa {i} lista", flush=True)
    norma = m.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(dev)
    norma.weight.data = P.get("model.language_model.norm.weight", dev, torch.float32)
    W = P.get("lm_head.weight", dev)
    r = {"resp": [0, 0, 0.0], "resto": [0, 0, 0.0], "ult512": [0, 0, 0.0]}
    with torch.no_grad():
        for s in range(a.n):
            x = norma(h[s].float()).to(torch.bfloat16)
            obj = torch.from_numpy(ids[s].astype(np.int64)).to(dev)
            for lo in range(0, L - 1, 1024):
                hi = min(L - 1, lo + 1024)
                lg = (x[lo:hi] @ W.T).float()
                t = obj[lo + 1:hi + 1]
                ok = (lg.argmax(-1) == t)
                nll = torch.nn.functional.cross_entropy(lg, t, reduction="none")
                posic = torch.arange(lo + 1, hi + 1, device=dev)
                es_resp = posic >= L - resp[s]
                for clave, msk in (("resp", es_resp), ("resto", ~es_resp), ("ult512", posic >= L - 512)):
                    r[clave][0] += int(ok[msk].sum()); r[clave][1] += int(msk.sum()); r[clave][2] += float(nll[msk].sum())
    for k, (ac, tot, nll) in r.items():
        print(f"{k:7s}: top-1 {ac / max(tot, 1):.3f}  perplejidad {np.exp(nll / max(tot, 1)):.2f}  ({tot} tokens)", flush=True)


if __name__ == "__main__":
    main()
