# SPDX-License-Identifier: Apache-2.0
"""PN164 — entradas del GDN como vistas de la salida de in_proj, sin copias (con PN155).

Con PN155 la salida de in_proj_qkvz trae [q k v | z | b + relleno | a + relleno] por fila. El forward compilado
la desarmaba con _g155.separar (torch.cat de b y a) y el camino de PyTorch de PN50 (que cae ahi porque la
entrada ya no es contigua): mixed_qkv, z, b y a .contiguous(). Inductor lo junta en un kernel de copia por capa
(triton_poi_fused_0, ~1,5 us x 48). Nadie las necesita contiguas en el decode del arbol: arbol_conv lee x con su
stride de fila, SK-25 lee z con stride, y gdn_arbol / pn122_cinta leen a y b con -DSAB. En cualquier otro paso
(prefill, mixtos, spec sin arbol PTX) el op las vuelve contiguas, fuera del grafo compilado.
"""
from __future__ import annotations

import os

ACTIVO = os.environ.get("GENESIS_ENABLE_PN164_GDN_VISTAS", "0").strip().lower() in ("1", "true", "yes", "on")


def vistas(gdn, mixed):
    """(mixed_qkv, z, b, a) como vistas de la salida de in_proj_qkvz (layout de PN155)."""
    from vllm._genesis.gdn_ba_qkvz import TROZO
    base, nv = gdn._g155
    qkv = (gdn.key_dim * 2 + gdn.value_dim) // gdn.tp_size
    z = mixed[..., qkv:base].unflatten(-1, (-1, gdn.head_v_dim))
    return mixed[..., :qkv], z, mixed[..., base:base + nv], mixed[..., base + TROZO:base + TROZO + nv]


def en_op(attn_metadata, mixed_qkv, b, a):
    """En _forward_core (fuera del grafo compilado): contiguas salvo en el paso solo-spec del arbol PTX."""
    if not ACTIVO:
        return mixed_qkv, b, a
    from vllm._genesis import gdn_cinta as _g122
    directo = (attn_metadata.spec_sequence_masks is not None and attn_metadata.num_prefills == 0
               and attn_metadata.num_decodes == 0 and _g122.activo() and not (_g122.sync_bits() & 512)
               and _g122.paso_arbol_activo())
    if directo:
        return mixed_qkv, b, a
    return mixed_qkv.contiguous(), b.contiguous(), a.contiguous()
