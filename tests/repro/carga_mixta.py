#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Carga MIXTA: un hilo largo + varios agentes chicos. El patron real de uso.

Mide LA LATENCIA ENTRE TOKENS de un agente mientras el hilo largo prefilea.

POR QUE ESA METRICA Y NO EL WALL-CLOCK
--------------------------------------
La v1 media cuanto tardaba cada agente de punta a punta. No sirve: los agentes
generan una cantidad VARIABLE de tokens (temperatura 0.7, cortan por EOS
cuando quieren), asi que el total mezcla "que tan rapido va el engine" con
"cuanto decidio escribir el modelo". Medido: la MISMA config dio 50,6s y 77,3s
en dos corridas, mas dispersion que la diferencia entre las configs que
queriamos comparar.

La latencia inter-token no tiene ese problema: es directamente la duracion del
paso del scheduler, que es la magnitud que mueven estos flags.

    tick del scheduler = el chunk de prefill del hilo largo domina el paso
    cada request en el batch avanza 1 token (4 con MTP) por paso
    => latencia inter-token del agente == duracion del paso

EL QUANTUM ES block_size, NO UN NUMERO REDONDO
----------------------------------------------
Con mamba_cache_mode=align, todo chunk de prefill largo se trunca a multiplo
de block_size (1600):

    scheduler.py _mamba_block_aligned_split:  n = n // 1600 * 1600

Asi que 4096 no es una config: son 2,56 bloques que el scheduler recorta a 2
(3200), mientras el profiler reserva transitorio para los 4096 completos. Por
eso los valores se expresan en MULTIPLOS de block_size.

    python3 carga_mixta.py <puerto> <key> [--grande 100000] [--agentes 5]
"""

from __future__ import annotations

import argparse
import json
import statistics
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
    "(bool, motivo), despues otra que normalice un telefono argentino. "
    "Solo codigo, sin explicacion."
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
    """Sin streaming: para el hilo largo y el calentamiento."""
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


def chat_stream(base, key, texto, salida):
    """Con streaming: devuelve (ttft, [latencias entre tokens], n_tokens)."""
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps({
            "model": chat.modelo,
            "messages": [{"role": "user", "content": texto}],
            "max_tokens": salida,
            "temperature": 0.7,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    previo = None
    deltas: list[float] = []
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for linea in r:
            linea = linea.decode().strip()
            if not linea.startswith("data: ") or linea == "data: [DONE]":
                continue
            d = json.loads(linea[6:])
            trozo = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if not trozo:
                continue
            ahora = time.time()
            if ttft is None:
                ttft = ahora - t0
            else:
                deltas.append(ahora - previo)
            previo = ahora
            n += 1
    return ttft or 0.0, deltas, n


def trafico_pcie(base, key) -> dict[str, tuple[float, float]]:
    """Bytes y segundos movidos entre GPU y CPU, por sentido.

    El offloading de KV copia bloques a RAM en cada paso. Interesa saber si
    esa copia compite con el computo. (Spoiler medido en el codigo: NO —
    save_kv_layer y wait_for_save son no-op, los stores se difieren al
    start_kv_transfers del paso siguiente y corren en un stream aparte,
    v1/kv_offload/cpu/gpu_worker.py:372 con un pool de streams. Pero se mide
    igual, que para eso estan las metricas.)
    """
    out: dict[str, list] = {}
    try:
        r = urllib.request.Request(base + "/metrics",
                                   headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(r, timeout=15) as resp:
            for linea in resp.read().decode().splitlines():
                if linea.startswith("#"):
                    continue
                for clave, idx in (("kv_offload_total_bytes_total", 0),
                                   ("kv_offload_total_time_total", 1)):
                    if clave in linea and "transfer_type=" in linea:
                        tipo = linea.split('transfer_type="')[1].split('"')[0]
                        val = float(linea.split()[-1])
                        out.setdefault(tipo, [0.0, 0.0])[idx] = val
    except Exception:
        pass
    return {k: (v[0], v[1]) for k, v in out.items()}


def resumen(nombre, deltas: list[float]) -> str:
    if not deltas:
        return f"   {nombre}: sin datos"
    d = sorted(deltas)
    p50 = statistics.median(d)
    p90 = d[int(len(d) * 0.9)] if len(d) > 1 else d[0]
    return (f"   {nombre}: mediana {p50 * 1000:.0f} ms   p90 {p90 * 1000:.0f} ms   "
            f"({len(d)} intervalos)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("puerto", type=int)
    ap.add_argument("key")
    ap.add_argument("--grande", type=int, default=100000)
    ap.add_argument("--agentes", type=int, default=5)
    ap.add_argument("--retardo", type=float, default=4.0)
    ap.add_argument("--salida", type=int, default=300)
    args = ap.parse_args()

    # nonce por corrida: el tier de offloading vive en /kv-offload y SOBREVIVE
    # al reinicio del engine, asi que sin esto la segunda corrida con el mismo
    # texto lo levanta del disco en vez de prefillear.
    nonce = f"[corrida {time.time():.0f}] "

    base = f"http://127.0.0.1:{args.puerto}"
    chat.modelo = pedir(base, args.key, "/v1/models")["data"][0]["id"]

    # con --enable-flashinfer-autotune los primeros requests pagan autotune y
    # JIT: el mismo piso dio 1,9s caliente y 48,9s recien arrancado.
    for k in range(2):
        chat(base, args.key, f"{nonce}calentar {k}. Deci hola.", 16)

    # ── piso: un agente solo, engine ocioso ─────────────────────────────────
    ttft0, d0, n0 = chat_stream(base, args.key,
                                f"{nonce}[piso] {PREGUNTA_AGENTE}", args.salida)
    print(f"1) piso (engine ocioso)   ttft {ttft0 * 1000:.0f} ms")
    print(resumen("inter-token", d0))

    # ── mixto ───────────────────────────────────────────────────────────────
    print(f"\n2) mixto: hilo de ~{args.grande} tok + {args.agentes} agentes "
          f"a los {args.retardo}s")
    rep = max(1, args.grande * 3 // len(CUERPO.format(n=0, j=0)))
    texto_grande = nonce + "".join(CUERPO.format(n=99, j=j) for j in range(rep))

    antes = trafico_pcie(base, args.key)
    res_grande: list = []

    def largo():
        res_grande.append(chat(
            base, args.key,
            "Resumi en tres lineas que hace este codigo:\n" + texto_grande, 200))

    hilo = threading.Thread(target=largo)
    hilo.start()
    time.sleep(args.retardo)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.agentes) as ex:
        mixto = list(ex.map(
            lambda i: chat_stream(base, args.key,
                                  f"{nonce}[mix-{i}] {PREGUNTA_AGENTE}", args.salida),
            range(args.agentes)))
    pared = time.time() - t0
    hilo.join()
    despues = trafico_pcie(base, args.key)

    ttfts = [m[0] for m in mixto]
    todos = [d for m in mixto for d in m[1]]
    print(f"   ttft de los agentes: {min(ttfts):.1f}s / "
          f"{sum(ttfts) / len(ttfts):.1f}s / {max(ttfts):.1f}s (min/media/max)")
    print(resumen("inter-token bajo carga", todos))
    print(f"   tokens generados por agente: "
          f"{[m[2] for m in mixto]}   (por eso el wall-clock no compara)")
    if res_grande:
        seg, pt, ct = res_grande[0]
        print(f"   hilo largo: {seg:.1f}s  prompt={pt} tok")

    if d0 and todos:
        print(f"\n   degradacion del tick: "
              f"{statistics.median(todos) / statistics.median(d0):.1f}x "
              f"({statistics.median(d0) * 1000:.0f} ms -> "
              f"{statistics.median(todos) * 1000:.0f} ms)")

    print(f"\n3) trafico GPU<->CPU durante la fase mixta ({pared:.0f}s de pared)")
    if not despues:
        print("   sin metricas de kv_offload (¿offloading apagado?)")
    for tipo in sorted(set(antes) | set(despues)):
        b0, t_0 = antes.get(tipo, (0.0, 0.0))
        b1, t_1 = despues.get(tipo, (0.0, 0.0))
        gb = (b1 - b0) / (1 << 30)
        seg = t_1 - t_0
        if gb <= 0 and seg <= 0:
            continue
        print(f"   {tipo}: {gb:.2f} GiB en {seg:.2f}s "
              f"({gb / seg if seg else 0:.1f} GiB/s, {100 * seg / pared:.1f}% del "
              f"tiempo de pared)")
    print("   NOTA: la copia NO esta en el camino critico. save_kv_layer y")
    print("   wait_for_save son no-op; los stores se difieren y corren en un")
    print("   stream aparte (v1/kv_offload/cpu/gpu_worker.py:372, pool de streams).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
