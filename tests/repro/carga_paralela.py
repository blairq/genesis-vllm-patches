#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Carga en paralelo contra un engine, para tunear --max-num-seqs.

Manda N prompts grandes a la vez, muestrea VRAM/RAM mientras corren y avisa
si el engine se murio. Es la prueba binaria que decide si un valor de
--max-num-seqs entra o no: pasa N/N con 0 OOMs, o no pasa.

    python3 carga_paralela.py <puerto> <key> [--paralelo 10] [--tokens 30000]

OJO con la metrica de VRAM: nvidia-smi muestrea cada pocos segundos y NO ve el
transitorio de las capas GDN, que dura microsegundos. Sirve para ver la
tendencia, no para decidir el margen. Para eso esta PN80, que lee
mem_get_info() en el punto exacto de la asignacion
(GENESIS_ENABLE_PN80_GDN_H_BUDGET_PROBE=1).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CUERPO = (
    "def paso_{n}_{j}(x):\n"
    "    acc = 0\n"
    "    for i in range(x):\n"
    "        acc += i * {n} - (i % 13)\n"
    "    return acc\n\n"
)


def pedir(base, key, ruta, datos=None, timeout=1800):
    req = urllib.request.Request(
        base + ruta,
        data=json.dumps(datos).encode() if datos else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def muestrear(parar: threading.Event, picos: dict):
    while not parar.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
            usada = sum(int(l.split(",")[0]) for l in out)
            total = sum(int(l.split(",")[1]) for l in out)
            picos["vram"] = max(picos.get("vram", 0), usada)
            picos["total"] = total
            picos["gpus"] = len(out)
        except Exception:
            pass
        parar.wait(3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("puerto", type=int)
    ap.add_argument("key")
    ap.add_argument("--paralelo", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=30000, help="aprox por prompt")
    ap.add_argument("--salida", type=int, default=120)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.puerto}"
    modelo = pedir(base, args.key, "/v1/models")["data"][0]["id"]
    # ~3 caracteres por token con este cuerpo (es codigo, tokeniza denso)
    repeticiones = max(1, args.tokens * 3 // len(CUERPO.format(n=0, j=0)))
    print(f"modelo: {modelo}   {args.paralelo} en paralelo   "
          f"~{args.tokens} tokens por prompt")

    def uno(n):
        # prefijo distinto por request para que NO compartan prefix cache:
        # asi cada una prefilea de verdad y el pico es real
        texto = "".join(CUERPO.format(n=n, j=j) for j in range(repeticiones))
        t0 = time.time()
        try:
            r = pedir(base, args.key, "/v1/chat/completions", {
                "model": modelo,
                "messages": [{"role": "user",
                              "content": f"Variante {n}. Resumi en una linea:\n{texto}"}],
                "max_tokens": args.salida,
                "temperature": 0.7,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            u = r.get("usage") or {}
            return (n, True, time.time() - t0,
                    u.get("prompt_tokens"), u.get("completion_tokens"), "")
        except Exception as e:
            det = ""
            if hasattr(e, "read"):
                try:
                    det = e.read().decode()[:200]
                except Exception:
                    pass
            return (n, False, time.time() - t0, None, None, f"{type(e).__name__}: {e} {det}")

    picos: dict = {}
    parar = threading.Event()
    hilo = threading.Thread(target=muestrear, args=(parar, picos), daemon=True)
    hilo.start()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.paralelo) as ex:
        res = sorted(ex.map(uno, range(1, args.paralelo + 1)))
    parar.set()
    hilo.join(timeout=5)

    ok = 0
    for n, bien, seg, pt, ct, err in res:
        if bien:
            ok += 1
            print(f"  prompt {n:>2}: OK    {seg:7.1f}s  prompt={pt:>7} tok  salida={ct}")
        else:
            print(f"  prompt {n:>2}: FALLO {seg:7.1f}s  {err}")

    total = time.time() - t0
    ptok = sum(r[3] or 0 for r in res)
    print(f"\n  {ok}/{args.paralelo} en {total:.1f}s   ({ptok} tokens de prompt, "
          f"{ptok / total:.0f} tok/s agregados)")
    if picos.get("vram"):
        libre = (picos["total"] - picos["vram"]) // picos.get("gpus", 1)
        print(f"  VRAM pico (nvidia-smi, subestima): {picos['vram']}/{picos['total']} MiB"
              f"   ~{libre} MiB libres por GPU")

    try:
        urllib.request.urlopen(urllib.request.Request(
            base + "/health", headers={"Authorization": f"Bearer {args.key}"}), timeout=20)
        print("  engine vivo despues de la carga: SI")
        vivo = True
    except Exception as e:
        print(f"  engine vivo despues de la carga: NO ({e})")
        vivo = False

    return 0 if (ok == args.paralelo and vivo) else 1


if __name__ == "__main__":
    sys.exit(main())
