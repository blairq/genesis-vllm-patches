# SPDX-License-Identifier: Apache-2.0
"""Estado compartido de PN99 (L2 comprimida en GPU).

El dimensionamiento lo decide `cpu/spec.py` y la transferencia lo consulta en
`cpu/gpu_worker.py`, que corren en procesos distintos (scheduler y workers).
Cada proceso deriva el estado de las MISMAS variables de entorno y del mismo
calculo, asi que no hay que comunicar nada: `activar()` solo memoriza lo ya
calculado para no recalcularlo.
"""

from __future__ import annotations

import os

_CRUDOS = 0
_COMPRIMIDOS = 0
_ACTIVO = False
_MOTIVO = ""


def habilitado() -> bool:
    """Lo que pidio el usuario por env (no implica que se pueda)."""
    v = os.environ.get("GENESIS_ENABLE_PN99_GPU_COMPRESSED_L2", "0")
    return v.strip().lower() in ("1", "true", "yes", "on")


def activar(bytes_crudos: int, bytes_comprimidos: int) -> None:
    global _CRUDOS, _COMPRIMIDOS, _ACTIVO, _MOTIVO
    _CRUDOS, _COMPRIMIDOS, _ACTIVO, _MOTIVO = (
        int(bytes_crudos),
        int(bytes_comprimidos),
        True,
        "",
    )


def desactivar(motivo: str) -> None:
    global _ACTIVO, _MOTIVO
    _ACTIVO, _MOTIVO = False, str(motivo)


def activo() -> bool:
    """True solo si ademas de pedido resulto aplicable.

    En el proceso worker el dimensionamiento tambien corre (cpu/spec.py se
    instancia en los dos lados), asi que el estado se rellena solo.
    """
    return _ACTIVO


def bytes_crudos() -> int:
    return _CRUDOS


def bytes_comprimidos() -> int:
    return _COMPRIMIDOS


def motivo() -> str:
    return _MOTIVO


def reset() -> None:
    global _CRUDOS, _COMPRIMIDOS, _ACTIVO, _MOTIVO
    _CRUDOS, _COMPRIMIDOS, _ACTIVO, _MOTIVO = 0, 0, False, ""
