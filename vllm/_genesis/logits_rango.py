# SPDX-License-Identifier: Apache-2.0
"""PN153 — los logits del target sin AllGather: cada rango manda solo sus candidatos.

El paso de decode junta los logits enteros del lm_head (NCCL AllGather: 220 us con 9 filas, 844 con 36;
media fila de 124k fp16 por rango y por fila) para despues quedarse con UN token por fila (el gumbel-max
del rechazo del arbol). Por el PCIe viajan 2,2 / 8,9 MB para eso.

Aca cada rango calcula sus logits locales (``compute_logits_local``) y manda, por fila:
  * su top-C local (C = el top_k mas grande del lote + margen; 1 + margen si nadie usa top_k), y
  * su ganador del gumbel-max, con el MISMO ruido que usara el muestreo (semilla, pos y la clave =
    indice GLOBAL del token; reusa ``gumbel_noised_argmax`` de vLLM).
Se juntan (valor fp16, indice) empaquetados en int64 y se arma un tensor de logits RALO: -inf salvo
los candidatos, con sus valores exactos. El resto del muestreo de vLLM corre sin cambios sobre ese
tensor y da el MISMO token:
  * greedy: el maximo global es el maximo de un rango;
  * gumbel sin top_k: el ganador global es el mayor de los dos ganadores por rango, y en el ralo todo
    otro candidato tiene valor+ruido menor que el ganador de su rango;
  * top_k (<= C): el top_k global esta dentro de la union de los top-C; top_p despues de top_k solo
    mira a los sobrevivientes (softmax sobre el enmascarado: mismas filas ordenadas, bit a bit);
  * thinking budget: el forzado escribe 1e9 en el token forzado, sea candidato o no.
No es exacto (y va por el camino de siempre): top_p sin top_k, min_p, penalidades, logit_bias, bad
words, logprobs, gramatica, contar NaN, gumbel en fp64, RheoSampling. La unica aproximacion aceptada:
un empate de mas de ``_MARGEN`` tokens exactamente en el valor del k-esimo.

``GENESIS_PN153_VERIFICAR=N``: en las primeras N llamadas calcula tambien los logits enteros y compara
el token muestreado por los dos caminos (temperatura, top_k/top_p y gumbel, sin efectos laterales).
Los motivos del camino de siempre se cuentan y se vuelcan a /dev/shm/pn153_stats_<pid>.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax as _gna

log = logging.getLogger("genesis.pn153")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN153_LOGITS_RANGO", "0").strip().lower() in ("1", "true", "yes", "on")
_VERIFICAR = [int(os.environ.get("GENESIS_PN153_VERIFICAR", "0"))]
_C_MAX = 64
_MARGEN = 16
_ctx = [None]
_stats = {"llamadas": 0, "rango": 0}
_malos = [0, 0]      # [comparadas, distintas]


def _contar(motivo: str) -> None:
    _stats["llamadas"] += 1
    _stats[motivo] = _stats.get(motivo, 0) + 1
    if _stats["llamadas"] % 500 == 0:
        try:
            with open(f"/dev/shm/pn153_stats_{os.getpid()}", "w") as f:
                f.write(json.dumps(_stats))
        except OSError:
            pass


@triton.jit
def _k_cand_bloques(logits_ptr, stride, n_valid, off, eidx_ptr, temp_ptr, seed_ptr, pos_ptr,
                    pv_ptr, pi_ptr, gv_ptr, gi_ptr, nblk, BLOCK: tl.constexpr):
    # Por (fila, bloque de 1024): maximo simple y ganador del gumbel-max. El gumbel es copia de
    # _gumbel_sample_kernel + gumbel_block_argmax de vLLM con la clave del ruido desplazada al indice
    # GLOBAL del token (off = primer token de este rango): mismo ruido que el muestreo.
    t = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    cols = b * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_valid
    x = tl.load(logits_ptr + t * stride + cols, mask=mask, other=float("-inf")).to(tl.float32)
    r = tl.load(eidx_ptr + t).to(tl.int64)
    valido = r >= 0
    temp = tl.load(temp_ptr + r, mask=valido, other=0.0).to(tl.float32)
    seed = tl.load(seed_ptr + r, mask=valido, other=0)
    pos = tl.load(pos_ptr + t)
    pm, pj = tl.max(x, axis=0, return_indices=True)
    gm, gj = _gna(x, cols + off, mask, seed, pos, temp, IS_DRAFTING=False, USE_FP64=False,
                  APPLY_TEMPERATURE=True)
    tl.store(pv_ptr + t * nblk + b, pm)
    tl.store(pi_ptr + t * nblk + b, b * BLOCK + pj)
    tl.store(gv_ptr + t * nblk + b, gm)
    tl.store(gi_ptr + t * nblk + b, b * BLOCK + gj)


@triton.jit
def _k_cand_final(bits_ptr, stride, off, pv_ptr, pi_ptr, gv_ptr, gi_ptr, nblk, out_ptr, M,
                  NB: tl.constexpr):
    # Por fila: el mejor bloque (el primero si empatan, como argmax) de cada uno y el paquete
    # (indice global << 16) | bits fp16 del logit crudo, en las columnas 0 y 1 de out [n, M].
    t = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, NB)
    mk = j < nblk
    pv = tl.load(pv_ptr + t * nblk + j, mask=mk, other=float("-inf"))
    gv = tl.load(gv_ptr + t * nblk + j, mask=mk, other=float("-inf"))
    pb = tl.argmax(pv, axis=0)
    gb = tl.argmax(gv, axis=0)
    pi = tl.load(pi_ptr + t * nblk + pb).to(tl.int64)
    gi = tl.load(gi_ptr + t * nblk + gb).to(tl.int64)
    pbits = tl.load(bits_ptr + t * stride + pi).to(tl.int64) & 0xFFFF
    gbits = tl.load(bits_ptr + t * stride + gi).to(tl.int64) & 0xFFFF
    tl.store(out_ptr + t * M, ((pi + off) << 16) | pbits)
    tl.store(out_ptr + t * M + 1, ((gi + off) << 16) | gbits)


@triton.jit
def _k_ralo(ralo_ptr, V, todos_ptr, M, MP: tl.constexpr, BLOCK: tl.constexpr):
    # Fila del tensor ralo: -inf en todo el vocabulario y los candidatos con su valor (bits fp16).
    t = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    cols = b * BLOCK + tl.arange(0, BLOCK)
    tl.store(ralo_ptr + t * V + cols, tl.full([BLOCK], -1024, tl.int16), mask=cols < V)  # 0xFC00 = -inf
    tl.debug_barrier()
    j = tl.arange(0, MP)
    p = tl.load(todos_ptr + t * M + j, mask=j < M, other=-1)
    gid = p >> 16
    val = (p & 0xFFFF).to(tl.int16)
    en = (j < M) & (gid >= b * BLOCK) & (gid < b * BLOCK + BLOCK)
    tl.store(ralo_ptr + t * V + gid, val, mask=en)


def _candidatos(lv, local_bits, nloc, off, eidx, temp, seeds, pos, M):
    """[n, M] int64: columnas 0/1 = maximo y ganador gumbel; el resto lo llena el top-k si hace falta."""
    n = lv.shape[0]
    BLOCK = 1024
    nblk = triton.cdiv(nloc, BLOCK)
    dev = lv.device
    pv = torch.empty((n, nblk), dtype=torch.float32, device=dev)
    pi = torch.empty((n, nblk), dtype=torch.int32, device=dev)
    gv = torch.empty((n, nblk), dtype=torch.float32, device=dev)
    gi = torch.empty((n, nblk), dtype=torch.int32, device=dev)
    _k_cand_bloques[(n, nblk)](lv, lv.stride(0), nloc, off, eidx, temp, seeds, pos, pv, pi, gv, gi,
                               nblk, BLOCK=BLOCK)
    out = torch.empty((n, M), dtype=torch.int64, device=dev)
    _k_cand_final[(n,)](local_bits, local_bits.stride(0), off, pv, pi, gv, gi, nblk, out, M,
                        NB=triton.next_power_of_2(nblk))
    return out


def _motivo_no(runner, input_batch, grammar_output) -> str | None:
    """None si el lote se puede muestrear desde los candidatos (ver el docstring)."""
    from vllm._genesis import arbol_runner as ar
    s = runner.sampler
    if runner.batch_sharder is not None or grammar_output is not None:
        return "gramatica_o_sharder"
    if input_batch.num_draft_tokens == 0 or runner.rejection_sampler is None or not ar.E.listo:
        return "sin_arbol"
    if ar._RHEO and ar.E.hay_temp:
        return "rheo"
    if s.compute_nans or s.use_fp64_gumbel:
        return "nans_o_fp64"
    idx = input_batch.idx_mapping_np
    st = s.sampling_states
    if st.max_num_logprobs(idx) != -1:
        return "logprobs"
    if (np.any(s.logit_bias_state.use_logit_bias[idx]) or np.any(s.penalties_state.use_penalty[idx])
            or int(s.bad_words_state.num_bad_words.np[idx].max()) > 0):
        return "penalidades"
    if np.any(st.min_p.np[idx] != 0.0):
        return "min_p"
    tk = st.top_k.np[idx]
    usa_k = tk != st.vocab_size
    if np.any((st.top_p.np[idx] != 1.0) & ~usa_k):
        return "top_p_sin_top_k"
    if np.any(usa_k & (tk > _C_MAX)):
        return "top_k_grande"
    return None


def _logits_por_rango(model, h, runner, input_batch):
    from vllm._genesis import arbol_runner as ar
    from vllm.distributed import tensor_model_parallel_all_gather
    lm = model.lm_head if hasattr(model, "lm_head") else model.language_model.lm_head
    lp = (model.logits_processor if hasattr(model, "logits_processor")
          else model.language_model.logits_processor)
    si = lm.shard_indices
    off, nloc, V = si.org_vocab_start_index, si.org_vocab_end_index - si.org_vocab_start_index, lp.org_vocab_size
    local = model.compute_logits_local(h)
    st = runner.sampler.sampling_states
    idx = input_batch.idx_mapping_np
    tk = st.top_k.np[idx]
    usa_k = tk != st.vocab_size
    # sin top_k alcanzan el maximo y el ganador gumbel; con top_k, el top-(k + margen) local
    C = min(int(tk[usa_k].max()) + _MARGEN, nloc) if usa_k.any() else 0
    lv = local[:, :nloc]
    pos = input_batch.positions[input_batch.logits_indices]
    pos = ar.pos_del_ruido(pos, input_batch.idx_mapping, int(input_batch.num_reqs))
    eidx = input_batch.expanded_idx_mapping
    paq = _candidatos(lv, lv.view(torch.int16), nloc, off, eidx, st.temperature.gpu, st.seeds.gpu,
                      pos.contiguous(), 2 + C)
    if C:
        val, ind = torch.topk(lv, C, dim=1)
        paq[:, 2:] = ((ind + off) << 16) | (val.view(torch.int16).to(torch.int64) & 0xFFFF)
    todos = tensor_model_parallel_all_gather(paq, dim=-1)
    n, M = todos.shape
    ralo = torch.empty((n, V), dtype=local.dtype, device=h.device)
    BL = 8192
    _k_ralo[(n, triton.cdiv(V, BL))](ralo.view(torch.int16), V, todos, M,
                                     MP=triton.next_power_of_2(M), BLOCK=BL)
    if _VERIFICAR[0] > 0:
        _VERIFICAR[0] -= 1
        _verificar(model.compute_logits(h), ralo, runner, input_batch, pos, eidx)
    return ralo


def _muestra(lg, runner, input_batch, pos, eidx):
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
    st = runner.sampler.sampling_states
    x = lg.float().clone()
    apply_temperature(x, eidx, st.temperature.gpu)
    x = st.apply_top_k_top_p(x, eidx, input_batch.idx_mapping_np)
    return gumbel_sample(x, eidx, st.temperature.gpu, st.seeds.gpu, pos.contiguous(),
                         apply_temperature=False, is_drafting=False)


def _verificar(completo, ralo, runner, input_batch, pos, eidx):
    a = _muestra(completo, runner, input_batch, pos, eidx)
    b = _muestra(ralo, runner, input_batch, pos, eidx)
    d = int((a != b).sum())
    _malos[0] += a.numel()
    _malos[1] += d
    if d or _VERIFICAR[0] == 0 or _malos[0] % 2000 < a.numel():
        log.warning("[PN153] verificacion: %d filas comparadas, %d distintas (este paso %d/%d)%s",
                    _malos[0], _malos[1], d, a.numel(),
                    "" if not d else f" completo={a.tolist()[:12]} ralo={b.tolist()[:12]}")


def instalar(cls) -> None:
    """Envuelve ``GPUModelRunner.sample`` (por encima del envoltorio del arbol)."""
    if not ACTIVO or getattr(cls, "_genesis_pn153", False):
        return
    sample0 = cls.sample

    def sample(self, hidden_states, input_batch, grammar_output):
        motivo = _motivo_no(self, input_batch, grammar_output)
        _contar(motivo or "rango")
        if motivo is not None:
            return sample0(self, hidden_states, input_batch, grammar_output)
        m = self.model
        if not getattr(m, "_genesis_pn153", False):
            cl0 = m.compute_logits

            def compute_logits(h, _cl0=cl0, _m=m):
                c = _ctx[0]
                if c is None:
                    return _cl0(h)
                _ctx[0] = None
                return _logits_por_rango(_m, h, *c)
            m.compute_logits = compute_logits
            m._genesis_pn153 = True
        _ctx[0] = (self, input_batch)
        try:
            return sample0(self, hidden_states, input_batch, grammar_output)
        finally:
            _ctx[0] = None

    cls.sample = sample
    cls._genesis_pn153 = True
    log.warning("[PN153] logits por rango instalados (C max %d, margen %d, verificar %d)",
                _C_MAX, _MARGEN, _VERIFICAR[0])


__all__ = ["instalar", "ACTIVO"]
