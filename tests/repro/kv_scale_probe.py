#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""¿Vale la pena calibrar k_scale / v_scale para el KV en FP8?

Este checkpoint NO trae k_scale/v_scale, asi que vLLM usa 1.0
(quantization/kv_cache.py: "If no scales were loaded ... use the default value
of 1.0"). La pregunta es si eso cuesta calidad.

POR QUE LA RESPUESTA NO ES OBVIA
--------------------------------
Para INT8 la escala lo es todo: el paso de cuantizacion es fijo, y si el rango
real es chico se desperdicia casi toda la resolucion.

Para FP8 e4m3 NO: tiene 4 bits de exponente, asi que su precision es RELATIVA
(~2 digitos decimales) en todo su rango util. Con escala 1.0 solo hay perdida
si los valores se salen de ese rango:

    |x| > 448        -> se recorta (perdida grave)
    |x| < 2^-9       -> denormales, se pierde precision
    en el medio      -> escala 1.0 es casi optima, calibrar no aporta

Asi que antes de escribir un calibrador hay que MEDIR el rango real de K y V.

COMO MIDE
---------
Registra hooks sobre k_proj/v_proj de cada capa de atencion del modelo real
(cargado en fp16, sin cuantizar el KV) y acumula max/min absolutos por capa
sobre un prompt representativo. Reporta cuanto margen queda hasta 448 y cuantos
valores caerian en denormales.

    python3 kv_scale_probe.py [--tokens 8000] [--vram-pesos 13GiB]

RESULTADO (2026-08-17, 8000 tokens, 262M valores medidos)
---------------------------------------------------------
Solo hay KV en las 16 capas de atencion plena (3, 7, 11 ... 63; las otras 48
son GDN y no usan KV cache).

    |max| global      91,0   -> queda 5x de margen hasta 448
    recortados        0 de 262.144.000 (0,0000%)
    en denormales     0,26%

Y NO es cuestion de haber medido poco: con 16x menos tokens (512) el |max| daba
90,5. La distribucion esta saturada, no tiene cola creciente.

=> escala 1.0 esta bien; calibrar k_scale/v_scale no aportaria nada.

El |max| crece con la profundidad (8 en la capa 3, 91 en la 63) y V supera a K
en la segunda mitad del modelo, pero ni el peor caso se acerca a 448.

No necesita el engine corriendo, pero SI necesita las GPUs libres.

Desempaquetar el checkpoint son ~55 GB y no entran en 2x24 GB de VRAM + 20 GB
de RAM, asi que carga CAPA POR CAPA desde disco (max_memory + offload_folder).
Desempaquetar es exacto y no altera la medicion: lo que se mide son las
ACTIVACIONES de k_proj/v_proj, identicas con los pesos empacados o no.
"""

from __future__ import annotations

import argparse
import sys

import torch

E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0**-9  # por debajo: denormales


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="orcarouter/Qwen3.8-27B-Uncensored-FP8")
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--capas", type=int, default=0,
                    help="limitar a N capas de atencion (0 = todas)")
    ap.add_argument("--offload", default="/offload",
                    help="carpeta en disco para el streaming de pesos")
    ap.add_argument("--vram-pesos", default="20GiB",
                    help="VRAM por GPU reservada a PESOS. Bajarlo deja mas aire "
                         "para activaciones (la GDN de transformers es naive y "
                         "come mucho) a costa de mas trafico de disco.")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.modelo, trust_remote_code=True)
    print(f"modelo: {args.modelo}")
    tc = getattr(cfg, "text_config", cfg)
    print(f"  capas: {tc.num_hidden_layers}  kv_heads: {tc.num_key_value_heads}  "
          f"head_dim: {tc.head_dim}")

    tok = AutoTokenizer.from_pretrained(args.modelo, trust_remote_code=True)
    # Desempaquetado a fp16 son ~55 GB y no entran en 2x24 GB + 20 GB de RAM.
    # Con max_memory + offload_folder transformers trae los pesos CAPA POR CAPA
    # desde disco. Es lento pero acotado, y no cambia un solo valor: desempacar
    # FP8 -> fp16 es exacto, y de todos modos lo que medimos son las
    # ACTIVACIONES de k_proj/v_proj, que salen desempaquetadas en ambos casos.
    #
    # Va en bf16 (el dtype nativo del checkpoint) aunque el engine corra en
    # fp16: lo que se mide son MAGNITUDES, y ahi los dos coinciden. Lo que
    # cambia entre bf16 y fp16 es la mantisa, no el rango.
    modelo = AutoModelForCausalLM.from_pretrained(
        args.modelo, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="auto", offload_folder=args.offload,
        max_memory={0: args.vram_pesos, 1: args.vram_pesos, "cpu": "8GiB"},
    )
    modelo.eval()

    stats: dict[str, dict] = {}

    def hook(nombre):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t = t.detach().float()
            a = t.abs()
            d = stats.setdefault(nombre, {"max": 0.0, "min_nz": float("inf"),
                                          "n": 0, "n_denorm": 0, "n_clip": 0})
            d["max"] = max(d["max"], a.max().item())
            nz = a[a > 0]
            if nz.numel():
                d["min_nz"] = min(d["min_nz"], nz.min().item())
            d["n"] += a.numel()
            d["n_denorm"] += int((nz < E4M3_MIN_NORMAL).sum().item())
            d["n_clip"] += int((a > E4M3_MAX).sum().item())
        return fn

    handles = []
    n_reg = 0
    for nombre, mod in modelo.named_modules():
        if nombre.endswith((".k_proj", ".v_proj")):
            handles.append(mod.register_forward_hook(hook(nombre)))
            n_reg += 1
            if args.capas and n_reg >= args.capas * 2:
                break
    print(f"  hooks registrados: {n_reg}\n")

    texto = ("def procesar(x):\n    acc = 0\n    for i in range(x):\n"
             "        acc += i * 7 - (i % 13)\n    return acc\n\n") * (args.tokens // 25)
    ids = tok(texto, return_tensors="pt").input_ids[:, : args.tokens]
    print(f"prompt de {ids.shape[1]} tokens, corriendo forward...\n")
    with torch.no_grad():
        modelo(ids.to("cuda:0"))
    for h in handles:
        h.remove()

    print(f"{'tensor':<44}{'|max|':>10}{'margen a 448':>14}{'clip':>8}{'denorm':>9}")
    peor = 0.0
    tot_clip = tot_den = tot_n = 0
    for nombre in sorted(stats):
        d = stats[nombre]
        peor = max(peor, d["max"])
        tot_clip += d["n_clip"]; tot_den += d["n_denorm"]; tot_n += d["n"]
        print(f"{nombre[-44:]:<44}{d['max']:>10.3f}{E4M3_MAX / max(d['max'], 1e-9):>13.0f}x"
              f"{d['n_clip']:>8}{100 * d['n_denorm'] / max(d['n'], 1):>8.2f}%")

    print(f"\n  |max| global: {peor:.3f}   (e4m3 llega a {E4M3_MAX})")
    print(f"  margen sin usar: {E4M3_MAX / max(peor, 1e-9):.0f}x")
    print(f"  valores recortados: {tot_clip} de {tot_n} ({100*tot_clip/max(tot_n,1):.4f}%)")
    print(f"  valores en denormales: {100 * tot_den / max(tot_n, 1):.2f}%")
    print()
    if tot_clip == 0 and tot_den < 0.01 * tot_n:
        print("  => escala 1.0 es ADECUADA: nada se recorta y casi nada cae en")
        print("     denormales. e4m3 tiene precision RELATIVA, asi que el margen")
        print("     sin usar NO cuesta nada. Calibrar k_scale/v_scale no aportaria.")
    elif tot_clip:
        print("  => HAY RECORTE: calibrar k_scale/v_scale SI vale la pena.")
    else:
        print("  => muchos denormales: conviene una escala que suba la magnitud.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
