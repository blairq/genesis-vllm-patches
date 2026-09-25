#!/usr/bin/env python3
"""ETAPA C para la variante ROTADA (rotacion.py): arma el checkpoint servible por vLLM.

A diferencia de armar.py no alcanza con enlazar: en el checkpoint rotado cambian tensores que no
son las lineales cuantizadas (esas ya estan en la base rotada desde la etapa B):

  por capa   input_layernorm / post_attention_layernorm -> w = 0 (GemmaRMSNorm usa 1 + w: queda 1;
             la g original ya esta plegada en las lineales)
             linear_attn.in_proj_a / in_proj_b (bf16)    -> (W diag(g_in)) Rt
             genesis_rot.*                              -> fuera
  resto      embed_tokens                               -> E Rt
             norm (final) -> w = 0 ; lm_head           -> (W diag(g_final)) Rt
             visual.merger.linear_fc2 (escribe al residuo) -> R W, bias -> b Rt
             mtp.*                                      -> fuera (no se usa y quedaria invalido)

Servir con GENESIS_ENABLE_PN148_ROT_DOWN=1 (Hadamard por bloques antes de down_proj).

Uso: armar_rot.py --modelo <bf16> --variante <B/rot> --salida <dir> --ref_qcfg <checkpoint noon>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rotacion as ro  # noqa: E402
from armar import COPIAR, enlazar  # noqa: E402
from calibrar import Pesos  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--variante", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--ref_qcfg", required=True)
    a = ap.parse_args()
    dev = "cuda"
    torch.set_grad_enabled(False)
    os.makedirs(a.salida, exist_ok=True)
    P = Pesos(a.modelo)
    cfg = json.load(open(os.path.join(a.modelo, "config.json")))
    tc = cfg["text_config"]
    n, D = tc["num_hidden_layers"], tc["hidden_size"]
    spec = json.load(open(os.path.join(a.variante, "spec.json")))["spec"]
    Rt = ro.Rt_de(D, spec.get("bloque", 1024), dev)
    R = Rt.T.contiguous()
    mapa, total = {}, 0

    def guardar(nombre, t):
        nonlocal total
        ruta = os.path.join(a.salida, nombre)
        save_file({k: v.contiguous().cpu() for k, v in t.items()}, ruta)
        for k in t:
            mapa[k] = nombre
        total += os.path.getsize(ruta)

    for i in range(n):
        pref = f"model.language_model.layers.{i}."
        t = load_file(os.path.join(a.variante, f"capa_{i:02d}.safetensors"))
        sg = t.pop(pref + "genesis_rot.signos")
        t.pop(pref + "genesis_rot.bloques")
        assert torch.equal(sg, ro.signos(D)), "la variante se hizo con otra rotacion"
        g_in = 1 + t[pref + "input_layernorm.weight"].to(dev, torch.float32)
        for nm in ("input_layernorm", "post_attention_layernorm"):
            t[pref + nm + ".weight"] = torch.zeros_like(t[pref + nm + ".weight"])
        for nm in ("linear_attn.in_proj_a", "linear_attn.in_proj_b"):
            k = pref + nm + ".weight"
            if k in t:
                W = t[k].to(dev, torch.float32)
                t[k] = ((W * g_in[None, :]) @ Rt).to(t[k].dtype)
        guardar(f"capa_{i:02d}.safetensors", t)
        print(f"capa {i} lista", flush=True)

    resto = {}
    for k in P.mapa:
        if k.startswith("model.language_model.layers.") or k.startswith("mtp."):
            continue
        v = P.get(k, "cpu", None)
        if k == "model.language_model.embed_tokens.weight":
            v = torch.cat([(c.to(dev, torch.float32) @ Rt).to(v.dtype).cpu() for c in v.split(16384)])
        elif k == "lm_head.weight":
            g = 1 + P.get("model.language_model.norm.weight", dev, torch.float32)
            v = torch.cat([((c.to(dev, torch.float32) * g[None, :]) @ Rt).to(v.dtype).cpu() for c in v.split(16384)])
        elif k == "model.language_model.norm.weight":
            v = torch.zeros_like(v)
        elif k == "model.visual.merger.linear_fc2.weight":
            v = (R @ v.to(dev, torch.float32)).to(v.dtype).cpu()
        elif k == "model.visual.merger.linear_fc2.bias":
            v = (v.to(dev, torch.float32) @ Rt).to(v.dtype).cpu()
        resto[k] = v
    guardar("resto.safetensors", resto)

    json.dump({"metadata": {"total_size": total}, "weight_map": mapa},
              open(os.path.join(a.salida, "model.safetensors.index.json"), "w"), indent=1)
    qcfg = json.load(open(os.path.join(a.ref_qcfg, "config.json")))["quantization_config"]
    cuantizadas = {k[: -len(".weight_packed")] for k in mapa if k.endswith(".weight_packed")}
    qcfg["ignore"] = [x for x in qcfg["ignore"] if x not in cuantizadas]
    cfg["quantization_config"] = qcfg
    cfg["genesis_rotacion"] = {"semilla": ro.SEMILLA, "bloque": spec.get("bloque", 1024),
                               "bloque_down": spec.get("bloque_down", 512),
                               "requiere": "GENESIS_ENABLE_PN148_ROT_DOWN=1",
                               # la g de la norma final (plegada en el lm_head): la necesitan el borrador
                               # (PN149) y su entrenador para volver a la base original
                               "g_final": (1 + P.get("model.language_model.norm.weight", "cpu",
                                                     torch.float32)).tolist()}
    json.dump(cfg, open(os.path.join(a.salida, "config.json"), "w"), indent=2)
    for f in COPIAR:
        if os.path.exists(os.path.join(a.modelo, f)):
            enlazar(os.path.join(a.modelo, f), os.path.join(a.salida, f))
    print(f"armado {a.salida}: {len(mapa)} tensores, {total / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
