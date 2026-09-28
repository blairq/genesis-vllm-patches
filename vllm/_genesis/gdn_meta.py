# SPDX-License-Identifier: Apache-2.0
"""PN151 — metadata del GDN para el decode en arbol, en UN kernel en vez de ~35 ops de torch.

``GDNAttentionMetadataBuilder.build`` corre por cada grupo de cache GDN (3 en idiotSavant) en cada paso, en
Python: ``mamba_get_block_table_tensor`` (resta, division, clamp, arange, suma, cast, gather), ``arange``,
``empty``, indexar la tabla de bloques con una mascara de CPU, el ``expand().contiguous()`` de PN122 y una
docena de ``copy_``/``fill_`` sobre los buffers persistentes del grafo FULL. En el perfil del 27-09 (con la
pila de Python) era el mayor consumidor de GPU ociosa del paso: ~40 lanzamientos con un hueco antes de cada
uno.

En el caso que importa — decode puro con spec (todos los pedidos reales con K+1 tokens, el relleno al final,
grafo FULL) — todo eso se calcula con un solo kernel que escribe directo en los mismos buffers, con las
mismas reglas. Las condiciones se chequean con tensores de CPU (sin sincronizar). Cualquier otro caso
(prefills, mezclados, sin PN122) sigue por el builder original.

``GENESIS_PN151_VERIFICAR=1``: en las primeras llamadas corre las DOS versiones y compara todos los tensores
(la rapida primero, se clona, y la original despues sobreescribe los buffers). Loguea el resultado.
"""

from __future__ import annotations

import logging
import os

import torch

from vllm.triton_utils import tl, triton

log = logging.getLogger("genesis.pn151")
_ACTIVO = os.environ.get("GENESIS_ENABLE_PN151_GDN_META", "0").strip().lower() in ("1", "true", "yes", "on")
_VERIFICAR = int(os.environ.get("GENESIS_PN151_VERIFICAR", "0") or 0)
_verif = {"n": 0, "mal": 0}


@triton.jit
def _k_meta_spec(bt, s_bt, seq_lens, qsl, nacc, slots, st_out, s_st, mask_out, qsl_out, nacc_out, slots_out,
                 tok_out, n, B, S, NULL: tl.constexpr, BS: tl.constexpr, K1: tl.constexpr, BK1: tl.constexpr,
                 ALIGN: tl.constexpr, EXPAND: tl.constexpr):
    """Programa i (fila del lote, i < B): indices de estado, mascara, qsl, aceptados, slot y el arange de tokens."""
    i = tl.program_id(0)
    valido = i < n
    j = tl.arange(0, BK1)
    mj = j < K1
    if ALIGN:
        ini = tl.maximum((tl.load(seq_lens + i).to(tl.int32) - 1) // BS, 0)
    else:
        ini = 0
    if EXPAND:
        col = ini + j * 0          # PN122: una sola columna de estado, repetida en las K+1
    else:
        col = ini + j
    v = tl.load(bt + i * s_bt + col, mask=mj & valido, other=NULL)
    tl.store(st_out + i * s_st + j, tl.where(valido, v, NULL).to(tl.int32), mask=mj)
    tl.store(mask_out + i, valido)
    tl.store(nacc_out + i, tl.where(valido, tl.load(nacc + i, mask=valido, other=1), 1).to(tl.int32))
    tl.store(slots_out + i, tl.where(valido, tl.load(slots + i, mask=valido, other=0), 0).to(tl.int32))
    tl.store(qsl_out + i, tl.load(qsl + tl.minimum(i, n)).to(tl.int32))
    if i == B - 1:
        tl.store(qsl_out + B, tl.load(qsl + n).to(tl.int32))
    t = i * K1 + j
    tl.store(tok_out + t, t.to(tl.int32), mask=mj & (t < S))


def build_rapido(builder, m, num_accepted_tokens, num_decode_draft_tokens_cpu):
    """El camino rapido de ``build``; None si no aplica (y entonces corre el original)."""
    if not _ACTIVO or not builder.use_spec_decode or not builder.use_full_cuda_graph:
        return None
    if num_decode_draft_tokens_cpu is None or num_accepted_tokens is None:
        return None
    from vllm._genesis import gdn_cinta as g122
    if not g122.activo():
        return None
    d = num_decode_draft_tokens_cpu
    B = m.num_reqs
    if d.shape[0] < B:
        return None
    msk = d[:B] >= 0
    n = int(msk.sum())
    K1 = builder.num_spec + 1
    if n == 0 or not bool(msk[:n].all()) or int(d[:n].sum()) == 0:
        return None
    qc = m.query_start_loc_cpu[: B + 1]
    ql = qc[1:] - qc[:-1]
    if not (bool((ql[:n] == K1).all()) and bool((ql[n:] == 0).all())):
        return None                              # prefills o decodes sin spec: el original
    S = n * K1
    if n > builder.decode_cudagraph_max_bs or S > builder.decode_cudagraph_max_bs:
        return None
    spec = builder.kv_cache_spec
    modo = builder.vllm_config.cache_config.mamba_cache_mode
    # ancho de la tabla que el original le pasa al kernel: con PN122 (sin bloques especulativos) es 1 y se
    # repite en las K+1 columnas; si ya tiene K+1 columnas, van tal cual
    ancho = (1 + spec.num_speculative_blocks) if modo == "align" else min(m.block_table_tensor.shape[1], K1)
    from vllm.v1.attention.backends.gdn_attn import NULL_BLOCK_ID, GDNAttentionMetadata
    bt = m.block_table_tensor
    slots = g122.slots_gpu()
    _k_meta_spec[(B,)](
        bt, bt.stride(0), m.seq_lens, m.query_start_loc, num_accepted_tokens, slots,
        builder.spec_state_indices_tensor, builder.spec_state_indices_tensor.stride(0),
        builder.spec_sequence_masks, builder.spec_query_start_loc, builder.num_accepted_tokens,
        builder._g122_slots, builder.spec_token_indx, n, B, S,
        NULL=NULL_BLOCK_ID, BS=spec.block_size, K1=K1, BK1=triton.next_power_of_2(K1),
        ALIGN=(modo == "align"), EXPAND=(ancho != K1), num_warps=1)
    md = GDNAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=n, num_spec_decode_tokens=S, num_actual_tokens=m.num_actual_tokens,
        has_initial_state=None, chunk_indices=None, chunk_offsets=None,
        prefill_query_start_loc=None, prefill_state_indices=None, prefill_has_initial_state=None,
        spec_query_start_loc=builder.spec_query_start_loc[: B + 1],
        non_spec_query_start_loc=None,
        spec_state_indices_tensor=builder.spec_state_indices_tensor[:B],
        non_spec_state_indices_tensor=None,
        spec_sequence_masks=builder.spec_sequence_masks[:B],
        spec_token_indx=builder.spec_token_indx[:S],
        non_spec_token_indx=builder.non_spec_token_indx[:0],
        num_accepted_tokens=builder.num_accepted_tokens[:B],
        nums_dict=None, batch_ptr=None, token_chunk_offset_ptr=None,
    )
    md.g122_slots = builder._g122_slots[:B]
    return md


_CAMPOS = ("spec_query_start_loc", "spec_state_indices_tensor", "spec_sequence_masks", "spec_token_indx",
           "non_spec_token_indx", "num_accepted_tokens", "g122_slots")
_ESCALARES = ("num_prefills", "num_prefill_tokens", "num_decodes", "num_decode_tokens", "num_spec_decodes",
              "num_spec_decode_tokens", "num_actual_tokens")


def build(builder, original, common_prefix_len, m, num_accepted_tokens=None, num_decode_draft_tokens_cpu=None,
          fast_build=False):
    rap = build_rapido(builder, m, num_accepted_tokens, num_decode_draft_tokens_cpu)
    if rap is None:
        return original(builder, common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build)
    if _VERIFICAR and _verif["n"] < _VERIFICAR and not torch.cuda.is_current_stream_capturing():
        copia = {c: (getattr(rap, c).clone() if getattr(rap, c) is not None else None) for c in _CAMPOS}
        ref = original(builder, common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build)
        malos = [c for c in _ESCALARES if getattr(rap, c) != getattr(ref, c)]
        for c in _CAMPOS:
            a, b = copia[c], getattr(ref, c, None)
            if (a is None) != (b is None) or (a is not None and (a.shape != b.shape or not torch.equal(a, b.to(a.dtype)))):
                malos.append(c)
        for c in ("has_initial_state", "non_spec_query_start_loc", "non_spec_state_indices_tensor", "chunk_indices"):
            if getattr(ref, c) is not None:
                malos.append(c + " (el original no es None)")
        _verif["n"] += 1
        if malos:
            _verif["mal"] += 1
            log.warning("[PN151] DISTINTO del builder original en %s (llamada %d)", malos, _verif["n"])
        if _verif["n"] == _VERIFICAR:
            log.warning("[PN151] verificacion: %d llamadas, %d distintas", _verif["n"], _verif["mal"])
        return ref
    return rap


def instalar(cls) -> None:
    """Envuelve ``cls.build`` (lo llama la linea que PN151 agrega al final de gdn_attn.py, en cada proceso)."""
    if getattr(cls, "_genesis_pn151", False) or not _ACTIVO:
        return
    original = cls.build

    def build_pn151(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
                    num_decode_draft_tokens_cpu=None, fast_build=False):
        return build(self, original, common_prefix_len, common_attn_metadata, num_accepted_tokens,
                     num_decode_draft_tokens_cpu, fast_build)

    cls.build = build_pn151
    cls._genesis_pn151 = True
    log.info("[PN151] metadata del GDN en un kernel para el decode en arbol (verificar=%d)", _VERIFICAR)
