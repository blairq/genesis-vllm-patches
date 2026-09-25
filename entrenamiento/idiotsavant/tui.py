#!/usr/bin/env python3
"""Visor en la terminal del avance de una corrida (OPCIONAL: todo lo que muestra sale tambien por
linea de comandos con `idiotsavant.py estado [--json]` y `dflash2.sh estado [--json]`).

Solo LEE el disco (marcas, informes, logs, archivo de estado) y nvidia-smi: abrirlo o cerrarlo no
afecta a la corrida.

    ./correr.sh tui.py modelo   --trabajo DIR --salida DIR [--capas 64]
    ./correr.sh tui.py borrador [--nombre idiotsavant_dflash2]
Salir: Ctrl-C.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

AQUI = os.path.dirname(os.path.abspath(__file__))
TRAZAS = os.path.normpath(os.path.join(AQUI, "..", "..", "tests", "bench", "medicion", "trazas"))


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        return [[x.strip() for x in l.split(",")] for l in out.strip().splitlines()]
    except (OSError, subprocess.SubprocessError):
        return []


def ram_disco(ruta):
    libre = total = 0
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"):
            libre = int(l.split()[1]) / 1048576
        if l.startswith("MemTotal:"):
            total = int(l.split()[1]) / 1048576
    st = os.statvfs(ruta)
    return libre, total, st.f_bavail * st.f_frsize / 1024 ** 3


def panel_recursos(ruta):
    libre, total, disco = ram_disco(ruta)
    t = Table.grid(padding=(0, 1))
    t.add_row("RAM", f"{total - libre:.1f}/{total:.0f} GB usados", ProgressBar(total=total, completed=total - libre, width=20))
    t.add_row("disco", f"{disco:.0f} GB libres", Text("POCO" if disco < 10 else "", style="bold red"))
    for g in gpus():
        i, util, usado, tot, pot = g
        t.add_row(f"GPU{i}", f"{util:>3}%  {float(usado) / 1024:.1f}/{float(tot) / 1024:.0f} GB  {float(pot):.0f} W",
                  ProgressBar(total=100, completed=float(util), width=20))
    return Panel(t, title="recursos")


def cola(rutas, n=12):
    lineas = []
    for r in rutas:
        if os.path.exists(r):
            with open(r, errors="replace") as f:
                lineas += [(os.path.getmtime(r), l.rstrip()) for l in f.readlines()[-n:]]
    return "\n".join(l for _, l in lineas[-n:]) or "(sin log todavia)"


# ─── modelo ──────────────────────────────────────────────────────────────────────────────────────
def vista_modelo(a):
    from idiotsavant import leer_estado
    e = leer_estado(a.trabajo, a.salida, a.capas)
    n = e["capas_total"]
    cab = Table.grid(padding=(0, 2))
    cab.add_row("calibradas", ProgressBar(total=n, completed=e["calibradas"], width=40), f"{e['calibradas']}/{n}")
    cab.add_row("cuantizadas", ProgressBar(total=n, completed=e["cuantizadas"], width=40), f"{e['cuantizadas']}/{n}")
    ritmo = f"{e['segundos_por_capa']:.0f} s/capa, faltan ~{e['eta_minutos']:.0f} min" if e["segundos_por_capa"] else "midiendo ritmo"
    med = f"error local mediano {e['error_local_mediano']:.4f}" if e["error_local_mediano"] else ""
    vivos = " ".join(f"[{'green' if v else 'red'}]{k}[/]" for k, v in e["procesos_vivos"].items()) or "sin procesos registrados"
    marcas = " ".join(k for k, v in e["marcas"].items() if v)
    cabecera = Panel(Group(cab, Text.from_markup(f"{ritmo}   {med}\nprocesos: {vivos}   {marcas}")),
                     title="qwen3.8_27b_idiotSavant_sm_86 — reconstruccion")
    grilla = Table.grid(padding=(0, 1))
    fila = []
    for c in e["capas"]:
        if c["cuantizada"]:
            err = c["error_capa"] or 0
            estilo = "black on green" if err < 0.05 else "black on yellow" if err < 0.1 else "black on dark_orange"
            txt = f"{c['capa']:02d} {err * 100:4.1f}%"
            if any((c["canales_muertos"] or {}).values()):
                estilo += " bold underline"
        elif c["calibrada"]:
            estilo, txt = "black on cyan", f"{c['capa']:02d}  cal "
        else:
            estilo, txt = "dim", f"{c['capa']:02d}   .  "
        fila.append(Text(txt, style=estilo))
        if len(fila) == 8:
            grilla.add_row(*fila)
            fila = []
    if fila:
        grilla.add_row(*fila)
    leyenda = Text.from_markup("[black on green] <5% [/] [black on yellow] <10% [/] [black on dark_orange] >=10% [/] "
                               "[black on cyan] calibrada [/] [dim]pendiente[/]  subrayado = canal con g=0")
    capas = Panel(Group(grilla, leyenda), title="capas (error local de la capa servible)")
    logs = Panel(Text(cola(sorted(glob.glob(os.path.join(a.trabajo, "logs", "*.log"))))), title="log")
    lay = Layout()
    lay.split_column(Layout(cabecera, size=7), Layout(name="medio", size=12), Layout(logs))
    lay["medio"].split_row(Layout(capas, ratio=3), Layout(panel_recursos(a.salida), ratio=2))
    return lay


# ─── borrador ────────────────────────────────────────────────────────────────────────────────────
def vista_borrador(a):
    B = os.path.join(TRAZAS, "banco")
    ruta = os.path.join(B, f"{a.nombre}.estado.json")
    d = json.load(open(ruta)) if os.path.exists(ruta) else {"paso": "-", "detalle": "sin corrida", "historia": [],
                                                              "actualizado": time.time(), "pid": 0}
    vivo = os.path.exists(f"/proc/{d.get('pid')}")
    pasos = ["1 rotar fc", "2 cuantizar base", "3 captura", "4 entrenar", "5 cuantizar", "6 A/B"]
    t = Text()
    for p in pasos:
        k = p.split()[0]
        hecho = str(d["paso"]) == "fin" or (str(d["paso"]).isdigit() and int(k) < int(d["paso"]))
        actual = str(d["paso"]) == k
        t.append(f" {p} ", style="black on green" if hecho else "black on yellow" if actual else "dim")
        t.append(" ")
    info = [t, Text(f"{d['detalle']}  ({'corriendo' if vivo else 'sin proceso'}, "
                    f"hace {(time.time() - d['actualizado']) / 60:.0f} min)")]
    cap = os.path.join(B, f"captura_{a.nombre}.log")
    if os.path.exists(cap):
        n = sum(1 for l in open(cap) if " ok:" in l)
        info.append(Group(Text(f"captura: {n} pedidos"), ProgressBar(total=2000, completed=n, width=50)))
    ft = os.path.join(B, f"ft_{a.nombre}.log")
    if os.path.exists(ft):
        ult = [l for l in open(ft) if l.startswith("ep ")]
        if ult:
            try:
                ped = ult[-1].split("ped ")[1].split()[0]
                hecho, total = (int(x) for x in ped.split("/"))
                info.append(Group(Text("entrenamiento: " + ult[-1].strip()[:100]),
                                  ProgressBar(total=total, completed=hecho, width=50)))
            except (IndexError, ValueError):
                pass
    tab = Table("replica", "largo aceptado", "tok/s decode", title="A/B (totales)")
    for f in sorted(glob.glob(os.path.join(B, f"{a.nombre}_*_r*.json"))):
        x = json.load(open(f))
        P = x["pedidos"]
        dec = sum(p["decode_s"] or 0 for p in P)
        comp = sum((p["completion_tokens"] or 1) - 1 for p in P)
        tab.add_row(os.path.basename(f)[len(a.nombre) + 1:-5], f"{x['largo_aceptacion']:.3f}", f"{comp / dec:.1f}")
    hist = "\n".join(f"{h[0]} paso {h[1]}: {h[2]}" for h in d["historia"][-8:])
    lay = Layout()
    lay.split_column(Layout(Panel(Group(*info), title=f"qwen3.8_27b_idiotSavant_sm_86_dflash2 — {a.nombre}"), size=9),
                     Layout(name="medio", size=12), Layout(Panel(Text(hist or "(sin historia)"), title="historia")))
    lay["medio"].split_row(Layout(Panel(tab)), Layout(panel_recursos(TRAZAS)))
    return lay


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="que", required=True)
    pm = sub.add_parser("modelo")
    pm.add_argument("--trabajo", required=True)
    pm.add_argument("--salida", required=True)
    pm.add_argument("--capas", type=int, default=64)
    pb = sub.add_parser("borrador")
    pb.add_argument("--nombre", default="idiotsavant_dflash2")
    ap.add_argument("--cada", type=float, default=2.0, help="segundos entre refrescos")
    a = ap.parse_args()
    vista = vista_modelo if a.que == "modelo" else vista_borrador
    try:
        with Live(vista(a), refresh_per_second=2, screen=True) as live:
            while True:
                time.sleep(a.cada)
                live.update(vista(a))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
