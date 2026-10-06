# SPDX-License-Identifier: Apache-2.0
"""PN169 — conversacion de cada pedido, desde las cabeceras que pone el plugin genesis-sesion de opencode.

X-Genesis-Sesion / -Padre / -Raiz / -Agente se copian a ``kv_transfer_params`` (genesis_sesion, genesis_padre,
genesis_raiz, genesis_agente_oc): ese campo llega intacto al Request del scheduler, donde lo va a leer la politica
de desalojo por conversacion. Un pedido sin cabeceras no cambia. Solo ids opacos, nunca otras cabeceras.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.pn169")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN169_SESION", "0").strip().lower() in ("1", "true", "yes", "on")
CAMPOS = (("x-genesis-sesion", "genesis_sesion"), ("x-genesis-padre", "genesis_padre"),
          ("x-genesis-raiz", "genesis_raiz"), ("x-genesis-agente", "genesis_agente_oc"))
_vistas: set = set()


def de_cabeceras(raw_request) -> dict:
    hd = getattr(raw_request, "headers", None) or {}
    return {k: hd.get(h)[:128] for h, k in CAMPOS if hd.get(h)}


def etiquetar(request, raw_request) -> None:
    if not ACTIVO:
        return
    try:
        et = de_cabeceras(raw_request)
        if not et:
            return
        if et.get("genesis_sesion") and et["genesis_sesion"] not in _vistas:
            _vistas.add(et["genesis_sesion"])
            if len(_vistas) > 4096:
                _vistas.clear()
            log.warning("PN169 sesion nueva %s (padre %s, raiz %s, agente %s)", et["genesis_sesion"][-12:],
                        et.get("genesis_padre", "")[-12:] or "-", et.get("genesis_raiz", "")[-12:], et.get("genesis_agente_oc"))
        ktp = dict(getattr(request, "kv_transfer_params", None) or {})
        ktp.update(et)
        request.kv_transfer_params = ktp
    except Exception as e:                       # nunca rompe el pedido
        log.warning("PN169: no se pudo etiquetar (%s: %s)", type(e).__name__, e)


def de_request(req) -> dict:
    """Las etiquetas de un Request del scheduler ({} si no vino del plugin)."""
    ktp = getattr(req, "kv_transfer_params", None) or {}
    return {k: ktp[k] for _, k in CAMPOS if k in ktp}
