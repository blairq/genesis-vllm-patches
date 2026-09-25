#!/usr/bin/env python3
"""Cuantiza un borrador DFlash2 bf16 a W4A16 compressed-tensors (pack-quantized, g128, simetrico,
escalas fp16) por redondeo al mas cercano, con el MISMO formato que syvai/Qwen3.8-27B-DFlash2-W4A16.

Sirve para comparar justo: el original de Inco y el ajustado, cuantizados igual. (El de syvai es
GPTQ con Hessianas de chat; ese queda como referencia de produccion.)

Formato (verificado contra el checkpoint de noon en tests/bench/comparar_checkpoints_int4.py):
q = round(w / s) en [-8, 7], s = max|w| / 7,5 por grupo de 128; se guarda q + 8 en nibbles, el
valor i del int32 en los bits 4i..4i+3.

Uso: cuantizar_rtn.py <dir_bf16> <dir_salida> [dir_referencia_w4a16 para copiar quantization_config]
"""
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

G = 128
OBJETIVO = re.compile(r"^(layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|fc)\.weight$")


def empaquetar(w: torch.Tensor):
    out, inn = w.shape
    wf = w.float().view(out, inn // G, G)
    s = (wf.abs().amax(-1) / 7.5).clamp_min(1e-10).half()           # escalas fp16
    q = (wf / s.float().unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32) + 8
    q = q.view(out, inn // 8, 8)
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    packed = (q << sh).sum(-1, dtype=torch.int64)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32)
    err = ((q.view(out, inn // G, G).float() - 8) * s.float().unsqueeze(-1) - wf).norm() / wf.norm()
    return packed.contiguous(), s.contiguous(), torch.tensor([out, inn], dtype=torch.int64), float(err)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ref = sys.argv[3] if len(sys.argv) > 3 else "/models/qwen3.8-27b-dflash2-w4a16"
    os.makedirs(dst, exist_ok=True)
    sd = load_file(os.path.join(src, "model.safetensors"))
    nuevo, errs = {}, []
    for k, v in sd.items():
        if OBJETIVO.match(k):
            base = k[: -len(".weight")]
            p, s, shp, e = empaquetar(v)
            nuevo[base + ".weight_packed"], nuevo[base + ".weight_scale"], nuevo[base + ".weight_shape"] = p, s, shp
            errs.append(e)
        else:
            nuevo[k] = v.contiguous()
    save_file(nuevo, os.path.join(dst, "model.safetensors"))
    c = json.load(open(os.path.join(src, "config.json")))
    c["quantization_config"] = json.load(open(os.path.join(ref, "config.json")))["quantization_config"]
    json.dump(c, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for f in ("README.md",):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), dst)
    print(f"{len(errs)} matrices cuantizadas, error relativo medio {sum(errs) / len(errs):.4f}")


if __name__ == "__main__":
    main()
