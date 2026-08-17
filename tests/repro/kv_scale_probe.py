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
    en denormales     0,22%

Y NO es cuestion de haber medido poco: con 16x menos tokens el |max| daba 90,5.
La distribucion esta saturada, no tiene cola creciente.

=> escala 1.0 esta bien; calibrar k_scale/v_scale no aportaria nada.

K queda plana (9,2 a 22,8) porque k_norm la normaliza. La que crece con la
profundidad es V, que no pasa por ninguna norma: 6,0 en la capa 3 y 91,0 en la
63. Aun asi el peor caso queda a 5x de 448.

Las escalas que calcularia vLLM salen todas < 1, o sea que calibrar SUBIRIA las
magnitudes (alejandolas de los denormales), no las bajaria. Como el recorte ya
es cero, lo unico que tocaria es ese 0,22% de denormales: nada.

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

    def acumular(nombre, t):
        a = t.detach().float().abs()
        d = stats.setdefault(nombre, {"max": 0.0, "min_nz": float("inf"),
                                      "n": 0, "n_denorm": 0, "n_clip": 0})
        d["max"] = max(d["max"], a.max().item())
        nz = a[a > 0]
        if nz.numel():
            d["min_nz"] = min(d["min_nz"], nz.min().item())
        d["n"] += a.numel()
        d["n_denorm"] += int((nz < E4M3_MIN_NORMAL).sum().item())
        d["n_clip"] += int((a > E4M3_MAX).sum().item())

    # --- V: la salida de v_proj SI es lo que entra al cache (no pasa por
    # ninguna norma ni por RoPE), asi que alcanza con un hook.
    handles = []
    for nombre, mod in modelo.named_modules():
        if nombre.endswith(".v_proj") and ".self_attn." in nombre + ".":
            capa = nombre.split(".layers.")[1].split(".")[0]
            handles.append(mod.register_forward_hook(
                lambda _m, _i, out, c=capa: acumular(f"capa {int(c):>2}  V", out)))

    # --- K: la salida de k_proj NO es lo que entra al cache. El modelo hace
    #     key_states = k_norm(k_proj(x))  ->  apply_rotary_pos_emb(...)
    # k_norm reescala y RoPE mezcla pares, asi que hay que tomar el tensor
    # DESPUES de RoPE. Se envuelve la funcion; el contador de llamadas da el
    # indice de capa de atencion porque solo las 16 capas plenas la llaman, y
    # en orden.
    from transformers.models import qwen3_5
    mod_qwen = qwen3_5.modeling_qwen3_5
    rope_original = mod_qwen.apply_rotary_pos_emb
    capas_attn: list[int] = [
        int(n.split(".layers.")[1].split(".")[0])
        for n, _ in modelo.named_modules() if n.endswith(".self_attn.k_norm")
    ]
    contador = {"i": 0}

    def rope_instrumentado(q, k, cos, sin, *a, **kw):
        q2, k2 = rope_original(q, k, cos, sin, *a, **kw)
        i = contador["i"] % len(capas_attn)
        contador["i"] += 1
        acumular(f"capa {capas_attn[i]:>2}  K", k2)
        return q2, k2

    mod_qwen.apply_rotary_pos_emb = rope_instrumentado
    print(f"  capas con KV: {capas_attn}\n")

    texto = ("def procesar(x):\n    acc = 0\n    for i in range(x):\n"
             "        acc += i * 7 - (i % 13)\n    return acc\n\n") * (args.tokens // 25)
    ids = tok(texto, return_tensors="pt").input_ids[:, : args.tokens]
    print(f"prompt de {ids.shape[1]} tokens, corriendo forward...\n")
    with torch.no_grad():
        modelo(ids.to("cuda:0"))
    for h in handles:
        h.remove()
    mod_qwen.apply_rotary_pos_emb = rope_original

    # vLLM no divide por 448 sino por estas constantes heuristicas
    # (envs.K_SCALE_CONSTANT / V_SCALE_CONSTANT), que dejan margen de sobra:
    #     _k_scale = abs(key).max() / 200
    #     _v_scale = abs(value).max() / 100
    K_CONST, V_CONST = 200.0, 100.0

    print(f"{'tensor':<14}{'|max|':>9}{'escala vLLM':>13}{'margen a 448':>14}"
          f"{'clip':>7}{'denorm':>9}")
    peor = 0.0
    tot_clip = tot_den = tot_n = 0
    for nombre in sorted(stats):
        d = stats[nombre]
        peor = max(peor, d["max"])
        tot_clip += d["n_clip"]; tot_den += d["n_denorm"]; tot_n += d["n"]
        escala = d["max"] / (K_CONST if nombre.endswith("K") else V_CONST)
        print(f"{nombre:<14}{d['max']:>9.3f}{escala:>13.3f}"
              f"{E4M3_MAX / max(d['max'], 1e-9):>13.0f}x"
              f"{d['n_clip']:>7}{100 * d['n_denorm'] / max(d['n'], 1):>8.2f}%")

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
