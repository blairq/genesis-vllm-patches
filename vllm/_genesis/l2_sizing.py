# SPDX-License-Identifier: Apache-2.0
"""Cálculo del tamaño mínimo recomendado para el tier L2 (RAM).

La regla es la misma que en cualquier jerarquía de cachés: **cada nivel tiene
que ser más grande que el que está arriba**. Si L2 es más chica que L1, nada
de lo que L1 desaloja entra, y el tier no puede servir para lo único que
existe — retener lo que se cayó del nivel de arriba.

Acá eso se violaba por 2,5x y no había forma de darse cuenta: el engine
loguea "primary tier (arc, 222 blocks)" y ese número no se puede comparar con
nada sin hacer la cuenta a mano.
"""

from __future__ import annotations

GIB = 1 << 30


def bloques_por_request(
    tokens: int,
    tokens_por_bloque: int,
    num_grupos: int,
    grupos_recurrentes: int = 0,
    stride: int = 1,
) -> float:
    """Bloques de L2 que consume una request de `tokens` tokens.

    Los grupos recurrentes (GDN/Mamba) guardan 1 de cada `stride` fronteras
    cuando PN93 está activo, así que pesan una fracción del resto.
    """
    if tokens_por_bloque <= 0 or num_grupos <= 0:
        return 0.0
    posiciones = tokens / tokens_por_bloque
    densos = num_grupos - grupos_recurrentes
    factor = densos + (grupos_recurrentes / max(stride, 1))
    return posiciones * factor


def informe(
    *,
    num_blocks: int,
    bytes_por_bloque: int,
    tokens_por_bloque: int,
    num_grupos: int,
    max_model_len: int,
    tokens_en_l1: int = 0,
    grupos_recurrentes: int = 0,
    stride: int = 1,
) -> list[str]:
    """Devuelve el informe como lista de líneas listas para loguear."""
    if num_blocks <= 0 or bytes_por_bloque <= 0 or tokens_por_bloque <= 0:
        return []

    def gib(bloques: float) -> float:
        return bloques * bytes_por_bloque / GIB

    def blk(tokens: int) -> float:
        return bloques_por_request(
            tokens, tokens_por_bloque, num_grupos, grupos_recurrentes, stride
        )

    # Cuántos tokens entran hoy en L2 (inversa de blk()).
    por_token = blk(tokens_por_bloque) / tokens_por_bloque
    tokens_l2 = num_blocks / por_token if por_token > 0 else 0

    b_ctx = blk(max_model_len)
    b_l1 = blk(tokens_en_l1) if tokens_en_l1 else 0.0
    minimo = max(b_ctx, b_l1)

    L = [
        "[PN98] Dimensionamiento del tier L2 (RAM):",
        f"    configurado    : {num_blocks} bloques = {gib(num_blocks):.2f} GiB"
        f" = {tokens_l2:,.0f} tokens",
        f"    por bloque     : {bytes_por_bloque / 1e6:.2f} MB x {num_grupos} grupos"
        + (
            f" ({grupos_recurrentes} recurrentes, PN93 stride {stride})"
            if grupos_recurrentes and stride > 1
            else ""
        ),
    ]
    if tokens_en_l1:
        L.append(
            f"    L1 (KV en GPU) : {tokens_en_l1:,} tokens"
            f"  -> L2/L1 = {tokens_l2 / tokens_en_l1:.2f}x"
        )
    L += [
        f"    para 1 request de max_model_len ({max_model_len:,} tok):"
        f" {b_ctx:,.0f} bloques = {gib(b_ctx):.2f} GiB",
    ]
    if b_l1:
        L.append(
            f"    para retener todo lo que entra en L1:"
            f" {b_l1:,.0f} bloques = {gib(b_l1):.2f} GiB"
        )
    L.append(
        f"    MINIMO RECOMENDADO: {gib(minimo):.2f} GiB"
        f"  (cpu_bytes_to_use = {int(minimo * bytes_por_bloque):,})"
    )

    if tokens_en_l1 and tokens_l2 < tokens_en_l1:
        L += [
            "    ESTADO: JERARQUIA INVERTIDA. L2 es mas chica que L1, asi que",
            "            nada de lo que la GPU desaloja entra completo. Toda",
            "            promocion desde disco termina desalojando y el tier no",
            "            puede sostener un prefijo. Subir cpu_bytes_to_use, o",
            "            subir GENESIS_PN93_GDN_CHECKPOINT_STRIDE para que cada",
            "            request ocupe menos.",
        ]
    elif num_blocks < b_ctx:
        L.append(
            "    ESTADO: JUSTO. Una request de contexto maximo no entra entera."
        )
    else:
        L.append("    ESTADO: OK.")
    return L
