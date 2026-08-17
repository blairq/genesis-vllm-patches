#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Carga MIXTA: un hilo largo + varios agentes chicos. El patron real de uso.

Las otras dos cargas prueban extremos (todo prefill, o todo generacion). Esta
prueba lo que de verdad corre en este rig: UNA conversacion de contexto largo
mas varios subagentes con prompts cortos entrando en el medio.

Lo que mide, que es la pregunta que importa:
  cuanto TARDA en arrancar un agente chico que llega mientras el hilo largo
  esta prefileando.

Es la latencia que paga --max-num-batched-tokens: un chunk de prefill del hilo
largo se come el presupuesto entero del paso, asi que el agente que llega en
ese momento espera en la cola. Se compara contra el mismo agente con el engine
ocioso, que es el piso.

    python3 carga_mixta.py <puerto> <key> [--grande 100000] [--agentes 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CUERPO = (
    "def modulo_{n}_{j}(estado):\n"
    "    total = 0\n"
    "    for i in range(estado):\n"
    "        total += i * {n} - (i % 11)\n"
    "    return total\n\n"
)

PREGUNTA_AGENTE = (
    "Escribi una funcion Python que valide un email con regex y devuelva "
    "(bool, motivo). Solo el codigo, sin explicacion."
)


def pedir(base, key, ruta, datos=None, timeout=1800):
    req = urllib.request.Request(
        base + ruta,
        data=json.dumps(datos).encode() if datos else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(base, key, texto, salida):
    t0 = time.time()
    r = pedir(base, key, "/v1/chat/completions", {
        "model": chat.modelo,
        "messages": [{"role": "user", "content": texto}],
        "max_tokens": salida,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    u = r.get("usage") or {}
    return time.time() - t0, u.get("prompt_tokens"), u.get("completion_tokens")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("puerto", type=int)
    ap.add_argument("key")
    ap.add_argument("--grande", type=int, default=100000, help="tokens del hilo largo")
    ap.add_argument("--agentes", type=int, default=5)
    ap.add_argument("--retardo", type=float, default=4.0,
                    help="segundos a esperar antes de soltar los agentes")
    args = ap.parse_args()
    # DOS CONTAMINACIONES QUE ARRUINAN LA COMPARACION ENTRE CORRIDAS:
    #
    # 1. El tier de offloading vive en /kv-offload y SOBREVIVE al reinicio del
    #    engine, asi que la segunda corrida con el mismo texto no prefilea:
    #    lo levanta del disco. Un nonce distinto por corrida rompe el prefijo
    #    y obliga a prefillear de verdad.
    # 2. Con --enable-flashinfer-autotune, los primeros requests despues del
    #    arranque pagan autotune y JIT. Medido: el mismo piso de agentes dio
    #    1,9s con el engine caliente y 48,9s recien arrancado. Por eso hay
    #    calentamiento antes de medir nada.
    nonce = f"[corrida {time.time():.0f}] "

    base = f"http://127.0.0.1:{args.puerto}"
    chat.modelo = pedir(base, args.key, "/v1/models")["data"][0]["id"]

    print("0) calentamiento (autotune + JIT), no se mide")
    for k in range(2):
        t, _, _ = chat(base, args.key, f"{nonce}calentar {k}. Deci hola.", 16)
        print(f"   {t:.1f}s")
    print()

    # ── piso: los agentes con el engine ocioso ──────────────────────────────
    print("1) piso — agentes con el engine ocioso")
    with ThreadPoolExecutor(max_workers=args.agentes) as ex:
        piso = list(ex.map(
            lambda i: chat(base, args.key, f"{nonce}[{i}] {PREGUNTA_AGENTE}", 200),
            range(args.agentes)))
    t_piso = [p[0] for p in piso]
    print(f"   {min(t_piso):.1f}s / {sum(t_piso) / len(t_piso):.1f}s / "
          f"{max(t_piso):.1f}s  (min/media/max)\n")

    # ── mixto: los mismos agentes con el hilo largo prefileando ─────────────
    print(f"2) mixto — hilo largo de ~{args.grande} tokens + {args.agentes} agentes "
          f"soltados a los {args.retardo}s")
    rep = max(1, args.grande * 3 // len(CUERPO.format(n=0, j=0)))
    texto_grande = nonce + "".join(CUERPO.format(n=99, j=j) for j in range(rep))

    res_grande: list = []

    def largo():
        res_grande.append(
            chat(base, args.key,
                 "Resumi en tres lineas que hace este codigo:\n" + texto_grande, 200))

    hilo = threading.Thread(target=largo)
    hilo.start()
    time.sleep(args.retardo)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.agentes) as ex:
        # prompts DISTINTOS del piso para que no haya hit de prefix cache
        mixto = list(ex.map(
            lambda i: chat(base, args.key, f"{nonce}[mix-{i}] {PREGUNTA_AGENTE}", 200),
            range(args.agentes)))
    t_agentes = time.time() - t0
    hilo.join()

    t_mix = [m[0] for m in mixto]
    print(f"   {min(t_mix):.1f}s / {sum(t_mix) / len(t_mix):.1f}s / "
          f"{max(t_mix):.1f}s  (min/media/max)")
    if res_grande:
        seg, pt, ct = res_grande[0]
        print(f"   hilo largo: {seg:.1f}s  prompt={pt} tok  salida={ct} tok")

    media_piso = sum(t_piso) / len(t_piso)
    media_mix = sum(t_mix) / len(t_mix)
    print(f"\n   los {args.agentes} agentes tardaron {t_agentes:.1f}s en total")
    print(f"   penalidad por estar el hilo largo prefileando: "
          f"{media_mix / media_piso:.1f}x  ({media_piso:.1f}s -> {media_mix:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
