#!/usr/bin/env python3
"""Agrega el cache de KV en dos capas (RAM + disco) a un compose de vLLM.

Inserta SOLO lo necesario para el offloading, sin tocar modelo, sampling,
parches ni ningún otro parámetro del engine:

  1. volumen  /home/usuario/Proyectos/kv-offload -> /kv-offload  (tier 2, NVMe)
  2. --kv-transfer-config con TieringOffloadingSpec (RAM 5 GiB, ARC, tier fs)
  3. --enable-cumem-allocator
  4. env de PN81: cuota del tier de disco + purga de mmaps huérfanos

⚠️ `--gpu-memory-utilization` NO se toca: es específico de cada modelo. Pero
   OJO, cumem hace que el profiler SOBREESTIME el KV disponible, así que un
   engine que arrancaba justo puede empezar a OOMear en
   `_allocate_kv_cache_tensors`. Medido en Ektome: declaraba 14.24 GiB de KV
   cuando solo entraban 11.16 (3.08 GiB de error, 0.131 de la VRAM). Si eso
   pasa, hay que bajarle el util A ESE engine.

Uso:  python3 apply_kv_offload.py <compose.yml> [...]
      python3 apply_kv_offload.py --check <compose.yml>
"""
import json
import re
import sys

HOST_DIR = "/home/usuario/Proyectos/kv-offload"
CONT_DIR = "/kv-offload"
CPU_GIB = 5

KV_CFG = {
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": CPU_GIB * (1 << 30),
        "eviction_policy": "arc",
        "secondary_tiers": [{"type": "fs", "root_dir": CONT_DIR}],
    },
}


def ya_tiene(s: str) -> bool:
    return "kv-transfer-config" in s


def aplicar(path: str) -> str:
    s = open(path).read()
    if ya_tiene(s):
        return "ya tenía offloading, sin cambios"

    # ── 1. volumen del tier de disco ────────────────────────────────────────
    m = re.search(r"^(\s+)- [^\n]*models-cache:/root/\.cache/huggingface\n", s, re.M)
    if not m:
        return "ERROR: no encontré el volumen de models-cache donde anclar"
    i = m.group(1)
    s = s[: m.end()] + (
        f"{i}# Tier 2 del cache de KV: NVMe. Los bloques que no entran en el tier de\n"
        f"{i}# RAM bajan acá en vez de perderse y tener que re-prefillear.\n"
        f"{i}- {HOST_DIR}:{CONT_DIR}\n"
    ) + s[m.end():]

    # ── 2. env de PN81 (cuota del disco + purga de /dev/shm) ────────────────
    m = re.search(r"^(\s+)- GENESIS_[A-Z0-9_]+=\S*\n", s, re.M)
    if not m:
        return "ERROR: no encontré variables GENESIS_ donde anclar"
    i = m.group(1)
    s = s[: m.start()] + (
        f"{i}# PN81: el tier de disco de vLLM no tiene cuota NI limpieza (root_dir\n"
        f"{i}# crece sin techo: 38 GB en una sola sesión de pruebas). PN81 implementa\n"
        f"{i}# on_schedule_end() —hook que vLLM documenta para 'per-step cleanup' y\n"
        f"{i}# deja vacío— para podar por antigüedad, y purga al arrancar los mmap\n"
        f"{i}# huérfanos de /dev/shm que deja un cierre abrupto (llenarlo mata el\n"
        f"{i}# arranque siguiente con 'madvise: Bad address').\n"
        f"{i}# La cuota es POR RANK: con TP=2 el disco total es 2 x MAX_GB.\n"
        f"{i}- GENESIS_ENABLE_PN81_KV_DISK_QUOTA=1\n"
        f"{i}- GENESIS_KV_DISK_MAX_GB=30\n"
        f"{i}- GENESIS_KV_DISK_CHECK_EVERY=2000\n"
        f"{i}- GENESIS_KV_DISK_TARGET_RATIO=0.85\n"
    ) + s[m.start():]

    # ── 3. flags del engine ────────────────────────────────────────────────
    m = re.search(r"^(\s+)- --enable-prefix-caching\n", s, re.M)
    if not m:
        return "ERROR: no encontré --enable-prefix-caching donde anclar"
    i = m.group(1)
    s = s[: m.start()] + (
        f"{i}# ── Cache de KV en dos capas: RAM (5 GiB, ARC) + NVMe ───────────────\n"
        f"{i}# Evita re-prefillear un prefijo largo cuando los subagentes lo desalojan\n"
        f"{i}# de la VRAM. Medido: 51.5s de prefill en frío vs 2.8s recuperándolo.\n"
        f"{i}# ARC separa lo accedido UNA vez (subagentes efímeros -> se desalojan\n"
        f"{i}# primero) de lo accedido VARIAS (el hilo largo -> queda protegido).\n"
        f"{i}#\n"
        f"{i}# Va por --kv-transfer-config y NO por --kv-offloading-size: ese flag solo\n"
        f"{i}# setea cpu_bytes_to_use y deja el spec por defecto (RAM sola, LRU).\n"
        f"{i}#\n"
        f"{i}# --enable-cumem-allocator es OBLIGATORIO: sin él, la captura de CUDA\n"
        f"{i}# graphs falla con 'CUDA invalid argument' en _dummy_run/sm.fill_(-1).\n"
        f"{i}# ⚠️ Efecto lateral: cumem hace que el profiler SOBREESTIME el KV\n"
        f"{i}# disponible. Si el engine empieza a OOMear en\n"
        f"{i}# _allocate_kv_cache_tensors, hay que BAJARLE --gpu-memory-utilization.\n"
        f"{i}- --enable-cumem-allocator\n"
        f"{i}- --kv-transfer-config\n"
        f"{i}- '{json.dumps(KV_CFG)}'\n"
    ) + s[m.start():]

    open(path, "w").write(s)
    return "offloading agregado"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    solo_check = "--check" in sys.argv
    for p in args:
        if solo_check:
            s = open(p).read()
            print(f"  {p}: {'YA tiene' if ya_tiene(s) else 'sin offloading'}")
            continue
        print(f"  {p}: {aplicar(p)}")


if __name__ == "__main__":
    main()
