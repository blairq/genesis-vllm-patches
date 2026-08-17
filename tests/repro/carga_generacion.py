#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Carga de GENERACION: 10 prompts que piden codigo de verdad, en paralelo.

La otra prueba (carga_paralela.py) manda prompts enormes y pide 120 tokens:
mide prefill, no generacion. Esta hace lo contrario — prompts cortos y miles
de tokens de salida — que es el patron real de un agente escribiendo codigo.

Mide:
  - tokens/s de generacion, agregados y por request
  - cuantas requests corren de verdad en paralelo (sondea /metrics, no el log)
  - aceptacion de MTP (spec decode), que es lo que decide la latencia real

    python3 carga_generacion.py <puerto> <key> [--paralelo 10] [--salida 4000]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# 10 modulos distintos de la MISMA app: prefijos distintos (no comparten
# prefix cache) pero carga comparable entre si.
MODULOS = [
    ("motor de combate",
     "el sistema de combate por turnos: iniciativa, tiradas de ataque y "
     "defensa, criticos, estados alterados (veneno, aturdido) y resolucion "
     "de la ronda"),
    ("inventario",
     "el inventario: objetos apilables, peso maximo, equipar y desequipar, "
     "consumibles con efecto y ordenamiento por categoria"),
    ("generador de mazmorras",
     "el generador procedural de mazmorras: salas, corredores que las "
     "conectan, sembrado de enemigos y cofres, y una semilla reproducible"),
    ("ficha de personaje",
     "la ficha de personaje: atributos, subida de nivel con curva de "
     "experiencia, arboles de habilidades y modificadores derivados"),
    ("guardado y carga",
     "el guardado y la carga de partida: serializacion a JSON, migracion "
     "entre versiones del formato y ranuras multiples"),
    ("dialogos",
     "el sistema de dialogos con NPCs: arbol de opciones, condiciones segun "
     "el estado del jugador y banderas que persisten"),
    ("tienda y economia",
     "la tienda y la economia: precios con regateo segun carisma, stock que "
     "se repone, compra y venta con validaciones"),
    ("render de la interfaz",
     "el render de la interfaz en la terminal: mapa con caracteres, barras "
     "de vida, log de mensajes con scroll y colores ANSI"),
    ("misiones",
     "el sistema de misiones: objetivos encadenados, seguimiento del "
     "progreso, recompensas y un diario consultable"),
    ("bucle principal",
     "el bucle principal del juego: parser de comandos, maquina de estados "
     "entre exploracion / combate / menu, y manejo de la entrada"),
]

PLANTILLA = (
    "Estas escribiendo un RPG de mazmorras jugable enteramente en la terminal, "
    "en Python 3, sin dependencias externas.\n\n"
    "Implementa {que}.\n\n"
    "Requisitos:\n"
    "- codigo completo y ejecutable, no esqueletos ni `pass`\n"
    "- type hints y dataclasses donde corresponda\n"
    "- docstrings cortos explicando las decisiones de diseño\n"
    "- al final, un bloque de ejemplo de uso\n"
)


def pedir(base, key, ruta, datos=None, timeout=1800):
    req = urllib.request.Request(
        base + ruta,
        data=json.dumps(datos).encode() if datos else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def metrica(base, key, nombre):
    try:
        req = urllib.request.Request(
            base + "/metrics", headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            for linea in r.read().decode().splitlines():
                if linea.startswith(nombre) and not linea.startswith("#"):
                    return float(linea.split()[-1])
    except Exception:
        pass
    return None


def sondear(base, key, parar: threading.Event, picos: dict):
    """/metrics cada 0.5s: el log del engine sale cada 10s y se pierde el pico."""
    while not parar.is_set():
        for clave, nombre in (("running", "vllm:num_requests_running"),
                              ("waiting", "vllm:num_requests_waiting")):
            v = metrica(base, key, nombre)
            if v is not None:
                picos[clave] = max(picos.get(clave, 0), int(v))
        parar.wait(0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("puerto", type=int)
    ap.add_argument("key")
    ap.add_argument("--paralelo", type=int, default=10)
    ap.add_argument("--salida", type=int, default=4000)
    ap.add_argument("--pensar", action="store_true",
                    help="deja el bloque de thinking activo (por defecto OFF, "
                         "para que los tokens medidos sean codigo)")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.puerto}"
    modelo = pedir(base, args.key, "/v1/models")["data"][0]["id"]
    print(f"modelo: {modelo}   {args.paralelo} en paralelo   "
          f"max {args.salida} tokens de salida   thinking={'ON' if args.pensar else 'OFF'}\n")

    def uno(i):
        nombre, que = MODULOS[i % len(MODULOS)]
        t0 = time.time()
        try:
            # En streaming para separar PREFILL de GENERACION: el primer chunk
            # con contenido marca el fin del prefill (TTFT). Sin esto los
            # segundos totales mezclan las dos fases y no se puede atribuir
            # una mejora a ninguna.
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps({
                    "model": modelo,
                    "messages": [
                        {"role": "user", "content": PLANTILLA.format(que=que)}],
                    "max_tokens": args.salida,
                    "temperature": 0.7,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": args.pensar},
                }).encode(),
                headers={"Authorization": f"Bearer {args.key}",
                         "Content-Type": "application/json"},
            )
            ttft = None
            texto = []
            razon = None
            u = {}
            with urllib.request.urlopen(req, timeout=1800) as r:
                for linea in r:
                    linea = linea.decode().strip()
                    if not linea.startswith("data: "):
                        continue
                    cuerpo = linea[6:]
                    if cuerpo == "[DONE]":
                        break
                    ch = json.loads(cuerpo)
                    if ch.get("usage"):
                        u = ch["usage"]
                    for c in ch.get("choices") or []:
                        trozo = (c.get("delta") or {}).get("content") or ""
                        if trozo:
                            if ttft is None:
                                ttft = time.time() - t0
                            texto.append(trozo)
                        if c.get("finish_reason"):
                            razon = c["finish_reason"]
            return (i, nombre, True, time.time() - t0, u.get("prompt_tokens"),
                    u.get("completion_tokens"), razon, len("".join(texto)), "",
                    ttft)
        except Exception as e:
            det = ""
            if hasattr(e, "read"):
                try:
                    det = e.read().decode()[:200]
                except Exception:
                    pass
            return (i, nombre, False, time.time() - t0, None, None, None, 0,
                    f"{type(e).__name__}: {e} {det}", None)

    picos: dict = {}
    parar = threading.Event()
    hilo = threading.Thread(target=sondear, args=(base, args.key, parar, picos),
                            daemon=True)
    hilo.start()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.paralelo) as ex:
        res = sorted(ex.map(uno, range(args.paralelo)))
    parar.set()
    hilo.join(timeout=3)
    total = time.time() - t0

    ok = gen = 0
    ttfts = []
    prompt_toks = 0
    for i, nombre, bien, seg, pt, ct, razon, chars, err, ttft in res:
        if bien:
            ok += 1
            gen += ct or 0
            prompt_toks += pt or 0
            if ttft is not None:
                ttfts.append(ttft)
            # tok/s de GENERACION pura: descuenta el prefill
            tgen = seg - (ttft or 0)
            print(f"  {nombre:<22} OK  {seg:6.1f}s  ttft {(ttft or 0):5.2f}s  "
                  f"{ct:>5} tok  {(ct or 0) / max(tgen, 1e-9):5.1f} tok/s  "
                  f"fin={razon}")
        else:
            print(f"  {nombre:<22} FALLO {seg:6.1f}s  {err}")

    print(f"\n  {ok}/{args.paralelo} en {total:.1f}s")
    if ttfts:
        ttfts.sort()
        print(f"  PREFILL (ttft): min {ttfts[0]:.2f}s  mediana "
              f"{ttfts[len(ttfts) // 2]:.2f}s  max {ttfts[-1]:.2f}s"
              f"   ({prompt_toks} tokens de prompt en total)")
    print(f"  generacion: {gen} tokens  ->  {gen / total:.0f} tok/s agregados, "
          f"{gen / total / max(ok, 1):.1f} tok/s por request")
    print(f"  concurrencia real (sondeo a /metrics cada 0.5s): "
          f"pico {picos.get('running', '?')} corriendo, "
          f"{picos.get('waiting', '?')} en cola")

    ac = metrica(base, args.key, "vllm:spec_decode_num_accepted_tokens")
    dr = metrica(base, args.key, "vllm:spec_decode_num_draft_tokens")
    if ac and dr:
        print(f"  MTP: {ac:.0f}/{dr:.0f} borradores aceptados = {100 * ac / dr:.1f}%")

    try:
        urllib.request.urlopen(urllib.request.Request(
            base + "/health", headers={"Authorization": f"Bearer {args.key}"}),
            timeout=20)
        print("  engine vivo despues de la carga: SI")
    except Exception as e:
        print(f"  engine vivo despues de la carga: NO ({e})")
        return 1
    return 0 if ok == args.paralelo else 1


if __name__ == "__main__":
    sys.exit(main())
