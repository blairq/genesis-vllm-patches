# SPDX-License-Identifier: Apache-2.0
"""PN163 — salida del GDN sin el relleno de ceros ni la copia en los pasos de decode especulativo.

Por capa GDN (48) y paso corrian dos kernels casi vacios alrededor de la recurrencia del arbol:
  * el ``.zero_()`` de ``core_attn_out`` (el buffer persistente de P28), en el forward compilado;
  * ``memcpy32_post``: ``core_attn_out[:n] = core_attn_out_spec`` (spec_update de PN122 escribia en un tensor
    propio y vLLM lo copiaba).
En un paso solo-spec (sin prefill ni decode comun) todas las filas reales las escribe spec_update, asi que:
el cero sale del forward compilado (ahi no se sabe que paso es: se traza una vez para todos) y pasa al op, que lo
hace en todos los demas pasos; y spec_update escribe directo en core_attn_out. Las filas de relleno del grafo
quedan con lo que tenian (finito: el buffer de P28 nace en cero y solo lo escriben kernels), que es lo que
vllm#28182 queria evitar con torch.empty (memoria sin inicializar).
"""
from __future__ import annotations

import os

ACTIVO = os.environ.get("GENESIS_ENABLE_PN163_GDN_SALIDA", "0").strip().lower() in ("1", "true", "yes", "on")


def preparar_salida(attn_metadata, core_attn_out) -> bool:
    """Al comienzo de _forward_core (fuera del grafo compilado). True = paso solo-spec por PN122: spec_update
    escribe directo en core_attn_out y no hace falta el cero."""
    if not ACTIVO:
        return False
    from vllm._genesis import gdn_cinta as _g122
    directo = (attn_metadata.spec_sequence_masks is not None and attn_metadata.num_prefills == 0
               and attn_metadata.num_decodes == 0 and _g122.activo() and not (_g122.sync_bits() & 512))
    if not directo:
        core_attn_out.zero_()                 # el cero que se saco del forward compilado
    return directo
