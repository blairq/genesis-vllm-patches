# SPDX-License-Identifier: Apache-2.0
"""PN166 — captura de pedidos de chat TAL COMO LLEGAN (antes de P68/P69), para reproducirlos en la instancia de
pruebas y estudiar el prefix cache con trafico real. Guarda el cuerpo del pedido (mensajes, herramientas,
parametros) y la IP de origen; NUNCA cabeceras ni claves. Solo pedidos grandes, con tope de archivos.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

log = logging.getLogger("genesis.pn166")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN166_CAPTURA_PEDIDOS", "0").strip().lower() in ("1", "true", "yes", "on")
DIR = os.environ.get("GENESIS_PN166_DIR", "/traces/pedidos")
MIN_CHARS = int(os.environ.get("GENESIS_PN166_MIN_CHARS", "20000"))
MAX_ARCHIVOS = int(os.environ.get("GENESIS_PN166_MAX", "300"))
_n = {"i": 0}
_lock = threading.Lock()


def capturar(request, raw_request) -> None:
    if not ACTIVO:
        return
    try:
        cuerpo = request.model_dump(mode="json", exclude_unset=True)
        tam = len(json.dumps(cuerpo.get("messages", []), ensure_ascii=False))
        if tam < MIN_CHARS:
            return
        os.makedirs(DIR, exist_ok=True)
        with _lock:
            if _n["i"] == 0:
                _n["i"] = len([f for f in os.listdir(DIR) if f.endswith(".json")])
            if _n["i"] >= MAX_ARCHIVOS:
                return
            _n["i"] += 1
            k = _n["i"]
        ip = getattr(getattr(raw_request, "client", None), "host", None) if raw_request is not None else None
        hd = getattr(raw_request, "headers", None) or {}
        origen = hd.get("x-forwarded-for") or hd.get("x-real-ip") or ip          # detras de Traefik: la IP real
        agente = (hd.get("user-agent") or "")[:80]
        reg = {"t": time.time(), "ip": ip, "origen": origen, "agente": agente, "chars_mensajes": tam, "n_mensajes": len(cuerpo.get("messages", [])),
               "n_herramientas": len(cuerpo.get("tools") or []), "pedido": cuerpo}
        ruta = os.path.join(DIR, f"{time.strftime('%m%d_%H%M%S')}_{k:04d}.json")
        with open(ruta, "w") as f:
            json.dump(reg, f, ensure_ascii=False)
        log.warning("PN166: capturado %s (origen %s, %d mensajes, %d herramientas, %d chars)", os.path.basename(ruta), origen,
                    reg["n_mensajes"], reg["n_herramientas"], tam)
    except Exception as e:                       # nunca rompe el pedido
        log.warning("PN166: no se pudo capturar (%s: %s)", type(e).__name__, e)
