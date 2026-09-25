#!/usr/bin/env python3
"""ETAPA C: arma un checkpoint servible por vLLM a partir de capas de variantes, SIN copiar.

Manifiesto (JSON):
  {"base": "base",                        variante por defecto para todas las capas
   "capas": {"0": "g64", "63": "g64"},    sobreescrituras por capa
   "salida": "/models/propio-v1"}

El modelo es una carpeta con ENLACES DUROS a `<variantes>/<v>/capa_NN.safetensors`, un archivo
`resto.safetensors` (embed, norma final, lm_head, vision, MTP: se escribe una vez en el cache y se
enlaza), el `model.safetensors.index.json` generado, y el config con el quantization_config de
compressed-tensors. Armar una combinacion nueva tarda segundos y no ocupa disco.

Uso: armar.py --modelo <bf16> --variantes <dir> --manifiesto m.json [--ref_qcfg <checkpoint con quantization_config>]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrar import Pesos  # noqa: E402

COPIAR = ("chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
          "merges.txt", "vocab.json", "special_tokens_map.json", "LICENSE", "README.md")


def claves_de(ruta):
    with open(ruta, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return list(h)


def enlazar(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--variantes", required=True)
    ap.add_argument("--manifiesto", required=True)
    ap.add_argument("--ref_qcfg", required=True, help="checkpoint compressed-tensors del que se copia el quantization_config")
    a = ap.parse_args()
    man = json.load(open(a.manifiesto))
    sal = man["salida"]
    os.makedirs(sal, exist_ok=True)
    P = Pesos(a.modelo)
    cfg = json.load(open(os.path.join(a.modelo, "config.json")))
    n_capas = cfg["text_config"]["num_hidden_layers"]

    # resto: todo lo que no es una capa del modelo de lenguaje (se escribe UNA vez)
    resto = os.path.join(a.variantes, "resto.safetensors")
    if not os.path.exists(resto):
        ks = [k for k in P.mapa if not k.startswith("model.language_model.layers.")]
        save_file({k: P.get(k, "cpu", None).contiguous() for k in ks}, resto)
        print(f"resto: {len(ks)} tensores", flush=True)

    mapa, archivos = {}, [("resto.safetensors", resto)]
    elegidas = {}
    for i in range(n_capas):
        v = man.get("capas", {}).get(str(i), man["base"])
        src = os.path.join(a.variantes, v, f"capa_{i:02d}.safetensors")
        if not os.path.exists(src):
            raise FileNotFoundError(f"falta la capa {i} de la variante {v}: {src}")
        archivos.append((f"capa_{i:02d}.safetensors", src))
        elegidas[i] = v
    for nombre, src in archivos:
        enlazar(src, os.path.join(sal, nombre))
        for k in claves_de(src):
            mapa[k] = nombre
    total = sum(os.path.getsize(src) for _, src in archivos)
    json.dump({"metadata": {"total_size": total}, "weight_map": mapa},
              open(os.path.join(sal, "model.safetensors.index.json"), "w"), indent=1)

    # config: el del BF16 + el quantization_config de compressed-tensors. Del ignore se sacan las
    # lineales que alguna variante SI cuantizo (por ejemplo in_proj_a/b en int4).
    qcfg = json.load(open(os.path.join(a.ref_qcfg, "config.json")))["quantization_config"]
    cuantizadas = {k[: -len(".weight_packed")] for k in mapa if k.endswith(".weight_packed")}
    qcfg["ignore"] = [x for x in qcfg["ignore"] if x not in cuantizadas]
    cfg["quantization_config"] = qcfg
    json.dump(cfg, open(os.path.join(sal, "config.json"), "w"), indent=2)
    for f in COPIAR:
        if os.path.exists(os.path.join(a.modelo, f)):
            enlazar(os.path.join(a.modelo, f), os.path.join(sal, f))
    json.dump({"capas": elegidas, "manifiesto": man}, open(os.path.join(sal, "armado.json"), "w"), indent=1)
    print(f"armado {sal}: {len(mapa)} tensores, {total / 1e9:.1f} GB (enlazados), "
          f"variantes {sorted(set(elegidas.values()))}", flush=True)


if __name__ == "__main__":
    main()
