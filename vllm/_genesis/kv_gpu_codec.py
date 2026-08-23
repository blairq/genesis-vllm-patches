# SPDX-License-Identifier: Apache-2.0
"""Codec de KV a 4 bits que corre EN LA GPU, para comprimir el offload a L2/L3.

`kv_fp4_codec.py` (PN92) es numpy: sirve para el tier de disco, donde la CPU
ya está en el camino, pero no para L2 — ahí la copia la hace el DMA y meter a
la CPU en el medio mataría el ancho de banda.

Este módulo hace la misma cuantización con ops de torch, así que corre en el
mismo device donde vive la KV. El DMA sigue siendo una copia de bytes crudos;
lo único que cambia es que el payload es 1,78x más chico.

    fp8_e4m3            8    bits/elem
    4 bits + escala fp16 por grupo de 32:  4 + 16/32 = 4,5 bits/elem
    ratio                                  8 / 4,5   = 1,78x

Layout del bloque comprimido (todo contiguo, en bytes):

    [ codigos: n/2 bytes ][ escalas: (n/grupo) * 2 bytes ]

Los códigos son enteros con signo de 4 bits guardados con offset +8
(o sea 0..15), dos por byte: el elemento par en el nibble bajo.
"""

from __future__ import annotations

GRUPO = 32
_QMAX = 7  # rango simetrico -7..7; el +8 lo lleva a 0..15


def tamano_comprimido(nbytes: int, grupo: int = GRUPO) -> int:
    """Bytes que ocupa un bloque de `nbytes` elementos fp8 una vez comprimido."""
    if nbytes % grupo:
        raise ValueError(f"{nbytes} no es multiplo del grupo {grupo}")
    return nbytes // 2 + (nbytes // grupo) * 2


def ratio(grupo: int = GRUPO) -> float:
    return 8.0 / (4.0 + 16.0 / grupo)


def _fp8_a_f16(src):
    import torch

    return src.view(torch.float8_e4m3fn).to(torch.float16)


def comprimir(src, grupo: int = GRUPO):
    """`src`: tensor uint8/int8 1-D con la KV en fp8_e4m3. Devuelve uint8 1-D."""
    import torch

    n = src.numel()
    if n % grupo:
        raise ValueError(f"{n} no es multiplo del grupo {grupo}")

    x = _fp8_a_f16(src).reshape(-1, grupo).float()
    # fp8_e4m3 no tiene infinitos pero si NaN (0x7F/0xFF): no pueden envenenar
    # la escala de todo el grupo.
    x = torch.nan_to_num(x, nan=0.0)

    escala = x.abs().amax(dim=1) / _QMAX
    escala = torch.where(escala > 0, escala, torch.ones_like(escala))

    q = torch.round(x / escala.unsqueeze(1)).clamp_(-_QMAX, _QMAX).to(torch.uint8)
    q = (q.to(torch.int16) + 8).to(torch.uint8).reshape(-1)

    bajo = q[0::2]
    alto = q[1::2]
    codigos = (bajo | (alto << 4)).contiguous()

    esc = escala.to(torch.float16).view(torch.uint8).reshape(-1)
    return torch.cat([codigos, esc])


def descomprimir(src, n: int, grupo: int = GRUPO):
    """Inversa de `comprimir`. Devuelve uint8 1-D de `n` bytes en fp8_e4m3."""
    import torch

    n_cod = n // 2
    codigos = src[:n_cod]
    esc = src[n_cod:].view(torch.float16).float()

    bajo = (codigos & 0x0F).to(torch.int16) - 8
    alto = (codigos >> 4).to(torch.int16) - 8
    q = torch.empty(n, dtype=torch.int16, device=src.device)
    q[0::2] = bajo
    q[1::2] = alto

    x = q.reshape(-1, grupo).float() * esc.unsqueeze(1)
    return x.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)


def error_relativo(original, reconstruido) -> float:
    """Error L2 relativo entre dos buffers fp8, para calibrar."""
    import torch

    a = _fp8_a_f16(original).float()
    b = _fp8_a_f16(reconstruido).float()
    a = torch.nan_to_num(a, nan=0.0)
    b = torch.nan_to_num(b, nan=0.0)
    denom = a.norm()
    if denom == 0:
        return 0.0
    return float((a - b).norm() / denom)
