#!/usr/bin/env python3
"""Borrador DFlash2 W4A16 (compressed-tensors pack-quantized, g128) -> fp16 con los nombres del bf16 original.

Para reajustar el borrador SERVIDO: el entrenador parte de exactamente los pesos que se sirven (q * s), y
cuantizar_rtn.py --escalas <este W4A16> vuelve a los mismos int4 en todo lo que no se entreno
(round(q * s / s) = q). Se guarda en fp16: q * s tiene hasta 15 bits de mantisa y en bf16 se redondeaba.

Uso: decuantizar.py <dir_w4a16> <dir_salida>
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file


def desempacar(packed: torch.Tensor, escala: torch.Tensor, forma) -> torch.Tensor:
    out, inn = int(forma[0]), int(forma[1])
    p = packed.to(torch.int64) & 0xFFFFFFFF
    q = torch.stack([(p >> (4 * i)) & 0xF for i in range(8)], -1).view(out, inn).to(torch.float32) - 8
    g = inn // escala.shape[1]
    return (q.view(out, inn // g, g) * escala.float().unsqueeze(-1)).view(out, inn).half()


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    sd = load_file(os.path.join(src, "model.safetensors"))
    nuevo, n = {}, 0
    for k, v in sd.items():
        if k.endswith(".weight_packed"):
            base = k[: -len(".weight_packed")]
            nuevo[base + ".weight"] = desempacar(v, sd[base + ".weight_scale"], sd[base + ".weight_shape"])
            n += 1
        elif k.endswith((".weight_scale", ".weight_shape")):
            continue
        else:
            nuevo[k] = v.contiguous()
    save_file(nuevo, os.path.join(dst, "model.safetensors"))
    c = json.load(open(os.path.join(src, "config.json")))
    c.pop("quantization_config", None)
    json.dump(c, open(os.path.join(dst, "config.json"), "w"), indent=2)
    if os.path.exists(os.path.join(src, "README.md")):
        shutil.copy(os.path.join(src, "README.md"), dst)
    print(f"{n} matrices decuantizadas a fp16 en {dst}")


if __name__ == "__main__":
    main()
