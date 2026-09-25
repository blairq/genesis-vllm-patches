# SPDX-License-Identifier: Apache-2.0
"""PN149: borrador DFlash en su base ORIGINAL sobre un target con el residuo rotado.

El target rotado (entrenamiento/cuant/armar_rot.py) guarda el embedding como E Rt y el lm_head como
W diag(g_final) Rt, y el borrador los comparte. El borrador rotado (entrenamiento/borrador/
rotar_borrador.py) deja su residuo en la base original y trae en el config ``genesis_rotacion``:

  desrotar(e_rot) = e_rot R            = bloques(e_rot) * s      (embedding del target -> original)
  rotar(h)        = (h / g_final) Rt   = bloques((h / g) * s)    (antes del lm_head compartido)

con Rt = diag(s) Hb y Hb la Hadamard por bloques (simetrica). Sin la clave en el config, no hace
nada: un borrador comun sobre un target comun sigue igual.
"""
from __future__ import annotations

import torch


def _hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / n ** 0.5).to(torch.get_default_dtype())


def preparar(modelo, config) -> None:
    """Desde el __init__ del modelo del borrador: buffers en el dispositivo y dtype del modelo."""
    rc = getattr(config, "genesis_rotacion", None)
    if isinstance(config, dict):
        rc = config.get("genesis_rotacion")
    modelo._g149_on = bool(rc)
    if not rc:
        return
    b = int(rc["bloque"])
    modelo._g149_b = b
    modelo.register_buffer("_g149_h", _hadamard(b), persistent=False)
    modelo.register_buffer("_g149_s", torch.tensor(rc["signos"], dtype=torch.get_default_dtype()),
                           persistent=False)
    modelo.register_buffer("_g149_ig", torch.tensor(rc["inv_g_final"], dtype=torch.float32), persistent=False)


def _bloques(m, x: torch.Tensor) -> torch.Tensor:
    return (x.view(*x.shape[:-1], -1, m._g149_b) @ m._g149_h.to(x.dtype)).view(x.shape)


def desrotar(m, e: torch.Tensor) -> torch.Tensor:
    if not getattr(m, "_g149_on", False):
        return e
    return _bloques(m, e) * m._g149_s.to(e.dtype)


def rotar(m, h: torch.Tensor) -> torch.Tensor:
    if not getattr(m, "_g149_on", False):
        return h
    x = (h.float() * m._g149_ig).to(h.dtype) * m._g149_s.to(h.dtype)
    return _bloques(m, x)
