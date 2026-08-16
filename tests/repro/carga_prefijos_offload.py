#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Carga contra un engine REAL que fuerza hit local + hit externo a la vez.

Es la validacion de PN84 sobre hardware. El banco de pruebas offline
(offload_partial_hit_harness.py) reproduce el crash en segundos, pero solo
prueba la logica del scheduler; esto prueba el engine entero.

QUE CONDICION BUSCA
-------------------
El crash (`AssertionError` en offloading/scheduler.py:612, EngineDeadError)
necesita que una request tenga, al mismo tiempo:

  - hit LOCAL: parte del prefijo sigue en el cache de prefijos de la VRAM
  - hit EXTERNO: el resto del prefijo esta en el tier de RAM/disco

Eso pasa porque vLLM libera los bloques de una request en orden INVERSO (la
cola primero), asi que bajo presion la VRAM conserva la CABEZA del prefijo y
pierde la COLA — que el tier de offloading si tiene.

COMO LO FUERZA
--------------
Por ronda:
  1. SIEMBRA: manda prompts largos con prefijos comunes y los deja terminar,
     para que se offloadeen.
  2. PRESION: manda prompts ajenos hasta desalojar la cola de los sembrados
     de la VRAM (el tier externo los conserva).
  3. REUSO: re-manda los prefijos sembrados EXTENDIDOS, con la concurrencia
     exacta del crash real: 4 en vuelo y 5 en cola.

Uso:
    python3 carga_prefijos_offload.py <puerto> <api-key> [--rondas 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ~4 caracteres por token; 6000 "parrafos" ~ 20k tokens
PARRAFO = (
    "def etapa_{i}(entrada):\n"
    "    acumulado = 0\n"
    "    for indice in range(entrada):\n"
    "        acumulado += indice * {i} - (indice % 7)\n"
    "    return acumulado\n\n"
)


class Cliente:
    def __init__(self, puerto: int, key: str):
        self.base = f"http://127.0.0.1:{puerto}"
        self.key = key
        self.modelo = self._pedir("/v1/models")["data"][0]["id"]

    def _pedir(self, ruta, datos=None, timeout=1200):
        req = urllib.request.Request(
            self.base + ruta,
            data=json.dumps(datos).encode() if datos else None,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def vivo(self) -> bool:
        try:
            req = urllib.request.Request(
                self.base + "/health",
                headers={"Authorization": f"Bearer {self.key}"},
            )
            urllib.request.urlopen(req, timeout=20)
            return True
        except Exception:
            return False

    def arrancado_en(self) -> float | None:
        """process_start_time_seconds: si cambia, el proceso se reinicio."""
        try:
            req = urllib.request.Request(
                self.base + "/metrics",
                headers={"Authorization": f"Bearer {self.key}"},
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                for linea in r.read().decode().splitlines():
                    if linea.startswith("process_start_time_seconds"):
                        return float(linea.split()[-1])
        except Exception:
            pass
        return None

    def generar(self, texto: str, etiqueta: str):
        t0 = time.time()
        try:
            r = self._pedir(
                "/v1/chat/completions",
                {
                    "model": self.modelo,
                    "messages": [{"role": "user", "content": texto}],
                    "max_tokens": 40,
                    "temperature": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            u = r.get("usage") or {}
            return (etiqueta, True, time.time() - t0, u.get("prompt_tokens"), "")
        except Exception as e:
            detalle = ""
            if hasattr(e, "read"):
                try:
                    detalle = e.read().decode()[:200]
                except Exception:
                    pass
            return (etiqueta, False, time.time() - t0, None,
                    f"{type(e).__name__}: {e} {detalle}")


def prefijo(semilla: int, parrafos: int) -> str:
    return "".join(PARRAFO.format(i=semilla * 1000 + j) for j in range(parrafos))


def en_paralelo(cli: Cliente, trabajos, hilos: int):
    with ThreadPoolExecutor(max_workers=hilos) as ex:
        return list(ex.map(lambda t: cli.generar(t[1], t[0]), trabajos))


def informe(titulo, res):
    ok = sum(1 for r in res if r[1])
    print(f"  {titulo}: {ok}/{len(res)} OK")
    for etiqueta, bien, seg, ptok, err in res:
        if not bien:
            print(f"    {etiqueta}: FALLO en {seg:.1f}s  {err}")
    return ok == len(res)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("puerto", type=int)
    ap.add_argument("key")
    ap.add_argument("--rondas", type=int, default=3)
    ap.add_argument("--parrafos", type=int, default=1400)  # ~20k tokens
    ap.add_argument("--presion", type=int, default=14)
    args = ap.parse_args()

    cli = Cliente(args.puerto, args.key)
    arranque = cli.arrancado_en()
    print(f"modelo: {cli.modelo}   proceso arrancado en {arranque}")

    todo_bien = True
    for ronda in range(1, args.rondas + 1):
        print(f"\n═══ ronda {ronda} ═══")
        base = ronda * 100

        # 1. SIEMBRA
        siembra = [
            (f"siembra{n}", prefijo(base + n, args.parrafos) + "\nResumi en una linea.")
            for n in range(3)
        ]
        todo_bien &= informe("siembra", en_paralelo(cli, siembra, 3))

        # 2. PRESION: prompts ajenos para desalojar la COLA de los sembrados
        presion = [
            (f"presion{n}",
             prefijo(base + 500 + n, args.parrafos) + "\nResumi en una linea.")
            for n in range(args.presion)
        ]
        todo_bien &= informe("presion", en_paralelo(cli, presion, 4))

        # 3. REUSO: mismo prefijo EXTENDIDO, 4 en vuelo + 5 en cola (como el crash)
        reuso = []
        for n in range(9):
            p = prefijo(base + (n % 3), args.parrafos)
            p += prefijo(base + 900 + n, args.parrafos // 4)  # extension distinta
            reuso.append((f"reuso{n}", p + "\nResumi en una linea."))
        todo_bien &= informe("reuso (4 en vuelo + 5 en cola)",
                             en_paralelo(cli, reuso, 4))

        if not cli.vivo():
            print("  ✖ el engine NO responde /health despues de esta ronda")
            return 1
        ahora = cli.arrancado_en()
        if arranque and ahora and abs(ahora - arranque) > 1:
            print(f"  ✖ el proceso se REINICIO ({arranque} -> {ahora})")
            return 1
        print("  ✔ engine vivo y sin reinicio")

    print(f"\n{'✔ TODO OK' if todo_bien else '✖ hubo requests fallidas'}")
    return 0 if todo_bien else 1


if __name__ == "__main__":
    sys.exit(main())
