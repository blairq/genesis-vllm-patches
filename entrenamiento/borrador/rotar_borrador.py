#!/usr/bin/env python3
"""Adapta un borrador DFlash2 (bf16) a un target con el residuo ROTADO (entrenamiento/cuant/rotacion.py).

El borrador se queda en su base ORIGINAL; lo unico que toca la base del target son tres puntos:

  fc          lee los hidden states del target, ahora h Rt por trozo de 5120: se pliega W_c Rt en
              cada trozo y las features salen IDENTICAS a las de antes (la captura y el entrenador
              no cambian).
  embedding   compartido con el target, que guarda E Rt: PN149 lo desrota en linea (e = e_rot R).
  lm_head     compartido, guardado como W diag(g_final) Rt: PN149 le entrega (h / g_final) Rt.

Las dos operaciones en linea son Hadamard por bloques (sin leer pesos). Sus parametros viajan en el
config.json del borrador (clave genesis_rotacion): bloque, signos e inv_g_final.

Uso: rotar_borrador.py <borrador_bf16> <target_bf16_original> <target_rotado> <salida_bf16>
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cuant"))
import rotacion as ro  # noqa: E402
from calibrar import Pesos  # noqa: E402


def main():
    src, bf16, rot, dst = sys.argv[1:5]
    os.makedirs(dst, exist_ok=True)
    rc = json.load(open(os.path.join(rot, "config.json")))["genesis_rotacion"]
    D = json.load(open(os.path.join(bf16, "config.json")))["text_config"]["hidden_size"]
    b = rc["bloque"]
    assert rc["semilla"] == ro.SEMILLA
    Rt = ro.Rt_de(D, b, "cpu", torch.float64)
    t = load_file(os.path.join(src, "model.safetensors"))
    W = t["fc.weight"].double()
    n = W.shape[1] // D
    assert W.shape[1] == n * D, W.shape
    cfg = json.load(open(os.path.join(src, "config.json")))
    assert n == len(cfg["dflash_config"]["target_layer_ids"]), (n, cfg["dflash_config"])
    t["fc.weight"] = torch.cat([W[:, c * D:(c + 1) * D] @ Rt for c in range(n)], 1).to(t["fc.weight"].dtype)
    save_file(t, os.path.join(dst, "model.safetensors"))
    g = 1 + Pesos(bf16).get("model.language_model.norm.weight", "cpu", torch.float32)
    cfg["genesis_rotacion"] = {"semilla": ro.SEMILLA, "bloque": b, "signos": ro.signos(D).tolist(),
                               "inv_g_final": (1 / g).tolist()}
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    for f in os.listdir(src):
        if f not in ("model.safetensors", "config.json") and os.path.isfile(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print(f"borrador rotado en {dst}: fc {tuple(t['fc.weight'].shape)} ({n} trozos de {D}), bloque {b}")


if __name__ == "__main__":
    main()
