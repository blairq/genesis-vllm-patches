#!/usr/bin/env python3
"""Error de la activacion int8 por token (W4A8 de Marlin) POR LINEAL, y cuanto lo bajan los remedios.

Para cada capa corre la capa (pesos de una variante de la etapa B) sobre la entrada guardada en
muestra.pt de la etapa A (muestras apartadas, 4096 tokens) y engancha la entrada de cada lineal
cuantizada. Para cada una mide el error relativo de la SALIDA  ||W q(x) - W x|| / ||W x||  con:

  a8        int8 simetrico por token, como Marlin (VLLM_MARLIN_INPUT_DTYPE=int8). En las lineales
            de fila de TP=2 (o_proj, down_proj, out_proj) cada GPU ve media K y tiene su propia
            escala: se simula asi.
  had128    Hadamard por bloques de 128 sobre K antes de cuantizar (x H, H^T W^T: exacto en fp).
  had_k     Hadamard por bloques del mayor tamano potencia de 2 que divide K (por GPU).
  suave     suavizado por canal s_j = sqrt(max|x_j| / max|W_:,j|) (SmoothQuant alfa 0,5), con el
            maximo medido en la misma muestra: es el TECHO de lo que da plegar escalas.

Tambien el error de la capa entera con todas sus lineales en a8 contra su aporte al residuo.
Sale un json por capa y una tabla.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from referencia import capa_de_variante  # noqa: E402

FILA = ("o_proj", "down_proj", "out_proj")      # row-parallel en TP=2: K partida en 2


def q8(x, partes=1):
    xs = x.reshape(*x.shape[:-1], partes, -1)
    e = xs.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return ((xs / e).round().clamp(-127, 127) * e).reshape(x.shape)


def hadamard(n, dev):
    h = torch.ones(1, 1, device=dev)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5


def rot(x, b):
    """x [..., K] por bloques de b con Hadamard (ortogonal)."""
    H = hadamard(b, x.device)
    return (x.reshape(*x.shape[:-1], -1, b) @ H).reshape(x.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--A", required=True)
    ap.add_argument("--variante", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--tokens", type=int, default=2048, help="tokens por muestra (los ultimos)")
    a = ap.parse_args()
    dev, DT = "cuda", torch.float32
    torch.set_grad_enabled(False)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from safetensors import safe_open
    tc = Qwen3_5TextConfig(**json.load(open(os.path.join(a.modelo, "config.json")))["text_config"])
    tc._attn_implementation = "sdpa"
    res = {}
    for i in range(tc.num_hidden_layers):
        pref = f"model.language_model.layers.{i}."
        ruta = os.path.join(a.variante, f"capa_{i:02d}.safetensors")
        with safe_open(ruta, "pt") as f:
            bases = sorted(k[len(pref):-len(".weight_packed")] for k in f.keys() if k.endswith(".weight_packed"))
        with torch.device("meta"):
            capa = m.Qwen3_5DecoderLayer(tc, i)
        capa.load_state_dict(capa_de_variante(ruta, pref, dev, DT), assign=True, strict=True)
        mu = torch.load(os.path.join(a.A, f"capa_{i:02d}", "muestra.pt"))
        x0 = mu["entrada"][:, -a.tokens:].to(dev, DT)
        L = x0.shape[1]
        rot_emb = m.Qwen3_5TextRotaryEmbedding(tc).to(dev)
        pos = torch.arange(L, device=dev)[None]
        pe = rot_emb(x0[:1], pos)
        entradas = {}

        def gancho(nombre):
            def f(mod, args):
                entradas.setdefault(nombre, []).append(args[0].detach())
            return f
        hs = [capa.get_submodule(b).register_forward_pre_hook(gancho(b)) for b in bases]

        def correr(h):
            out = []
            for s in range(len(h)):
                o = capa(h[s:s + 1].clone(), position_embeddings=pe, attention_mask=None, position_ids=pos)
                out.append(o[0] if isinstance(o, tuple) else o)
            return torch.cat(out)
        y_fp = correr(x0)
        for hk in hs:
            hk.remove()
        fila = {}
        for b in bases:
            W = capa.get_submodule(b).weight                        # [N, K] dequantizado
            x = torch.cat(entradas[b]).reshape(-1, W.shape[1])
            partes = 2 if b.endswith(FILA) else 1
            K = W.shape[1] // partes
            y = x @ W.T
            nrm = y.norm()
            err = lambda yq: float((yq - y).norm() / nrm)  # noqa: E731
            bk = K & -K                                             # mayor potencia de 2 que divide K
            bk = min(bk, 4096)
            e_a8 = err(q8(x, partes) @ W.T)
            e_h128 = err(q8(rot(x, 128), partes) @ rot(W, 128).T)
            e_hk = err(q8(rot(x, bk), partes) @ rot(W, bk).T)
            s = (x.abs().amax(0) / W.abs().amax(0).clamp_min(1e-8)).clamp_min(1e-8).sqrt()
            s = s / s.mean()
            e_su = err(q8(x / s, partes) @ (W * s).T)
            e_suh = err(q8(rot(x / s, 128), partes) @ rot(W * s, 128).T)
            amax = x.abs().amax(1)
            cresta = float((amax / x.pow(2).mean(1).sqrt()).median())
            fila[b] = {"a8": e_a8, "had128": e_h128, f"had{bk}": e_hk, "suave": e_su, "suave_had128": e_suh,
                       "cresta_mediana": cresta, "K": W.shape[1], "partes": partes}
        # capa entera con todas las lineales en a8
        hs = [capa.get_submodule(b).register_forward_pre_hook(
            (lambda p: (lambda mod, args: (q8(args[0], p),) + tuple(args[1:])))(2 if b.endswith(FILA) else 1)) for b in bases]
        y_a8 = correr(x0)
        for hk in hs:
            hk.remove()
        fila["_capa"] = {"a8": float((y_a8 - y_fp).norm() / (y_fp - x0).norm())}
        res[i] = fila
        print(f"capa {i:2d} {tc.layer_types[i][:4]}: capa {fila['_capa']['a8']:.4f}  " +
              "  ".join(f"{b.split('.')[-1]} {v['a8']:.4f}/{v['had128']:.4f}/{v['suave']:.4f}/{v['suave_had128']:.4f}"
                        for b, v in fila.items() if b != "_capa"), flush=True)
        del capa, entradas, x0, y_fp, y_a8
        torch.cuda.empty_cache()
        json.dump(res, open(a.salida, "w"), indent=1)


if __name__ == "__main__":
    main()
