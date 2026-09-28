# SPDX-License-Identifier: Apache-2.0
"""PN159 — los candidatos del borrador DFlash2 sobre un vocabulario recortado (FR-Spec / VocabTrim).

``compute_candidates`` del borrador saca el top-16 del lm_head COMPLETO del target (248.320 tokens):
390 us por paso con 1 pedido, en el techo de DRAM (Marlin W4A8, 318 MB por rango). La verificacion la
sigue haciendo el target con el vocabulario completo, asi que el borrador no necesita poder proponer
cualquier token: con los frecuentes de la carga real alcanza.

Fase 1 (esto): subconjunto FIJO. El orden sale de las capturas de ``dflash2.sh``
(``tests/proto/pn159_frecuencias.py`` -> ``datos/pn159_orden.npy``: tokens generados, top-4 del target y
contexto). Medido sobre pedidos apartados: 32k cubre 98,4% de los tokens generados, 64k 99,5%. De lo que
queda afuera, 57% esta en el prompt del pedido y 28% ya se genero antes (fase 2: filas dinamicas).

Por rango: la mitad del subconjunto (repartido parejo, ver ``_repartir``): las MISMAS columnas int4 y
escalas del lm_head de PN139, elegidas antes de su repack (sin recuantizar), en una matriz Marlin propia. SM86: el lm_head esta en el
layout de tiles de Marlin y no se pueden sacar columnas sueltas, por eso es una copia (42 MB por rango con
32k). El top-k por rango, el all-gather y el top-k final son los de ``get_top_k_tokens``.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger("genesis.pn159")
ACTIVO = os.environ.get("GENESIS_ENABLE_PN159_VOCAB_BORRADOR", "0").strip().lower() in ("1", "true", "yes", "on")
TAM = int(os.environ.get("GENESIS_PN159_VOCAB", "65536"))     # 32k: -3,5% de aceptacion; 64k: igual
DIN = int(os.environ.get("GENESIS_PN159_DINAMICO", "0"))           # filas del anillo dinamico por rango (0 = fase 1)
_HECHO = False
_din: dict = {}
_ORDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos", "pn159_orden.npy")


def _repartir(filas_de, ancho: int, ini: int, dev, T: int = 4096):
    """Genera (ids globales, filas) por trozos: el subconjunto de ESTE rango. ``filas_de(idx)`` devuelve
    [len(idx), ancho] int32 (el int4 y las escalas de esos tokens del tramo, empaquetados por token).

    El subconjunto se reparte PAREJO entre rangos, intercalado por posicion en el orden de frecuencia: los
    tokens frecuentes son casi todos ids bajos (con 64k, 63.808 caen en el tramo del rango 0 y 1.792 en el
    del 1) y el paso lo fija el rango lento. Cada rango tiene en memoria solo su tramo: lo que usa del otro
    llega por all-gather (NCCL), una vez, al cargar, y por trozos (la carga llega justa de memoria).
    S/tp es multiplo de 256 (con N no multiplo de 256 Marlin elige una config lenta)."""
    import numpy as np
    from vllm.distributed import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
                                  tensor_model_parallel_all_gather)
    tp, r = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    orden = torch.from_numpy(np.load(_ORDEN).astype("int64"))
    s = -(-TAM // (256 * tp)) * 256 * tp
    sub = orden[:s]
    destino = torch.arange(s) % tp                                    # rango que la usa
    if tp == 1:
        for a in range(0, s, T):
            ids = sub[a:a + T].to(dev)
            yield ids, filas_de(ids - ini)
        return
    if tp != 2:
        raise NotImplementedError("PN159 reparte solo con TP=1 o 2")
    bordes = tensor_model_parallel_all_gather(torch.tensor([ini], device=dev), dim=0).cpu()
    dueno = torch.searchsorted(bordes, sub, right=True) - 1          # rango que tiene la fila en memoria
    otro = 1 - r
    locales = sub[(dueno == r) & (destino == r)]
    manda = sub[(dueno == r) & (destino == otro)]
    recibe = sub[(dueno == otro) & (destino == r)]                    # mismo orden en que el otro lo manda
    for a in range(0, locales.numel(), T):
        ids = locales[a:a + T].to(dev)
        yield ids, filas_de(ids - ini)
    n_max = max(int(((dueno == q) & (destino == 1 - q)).sum()) for q in range(2))
    for a in range(0, n_max, T):                                      # los dos rangos, las mismas vueltas
        b = min(a + T, n_max)
        buf = torch.zeros(b - a, ancho, dtype=torch.int32, device=dev)
        m = manda[a:b]
        if m.numel():
            buf[:m.numel()] = filas_de(m.to(dev) - ini)
        todo = tensor_model_parallel_all_gather(buf.unsqueeze(0), dim=0)   # [2, b-a, B]
        n = max(0, min(b, recibe.numel()) - a)
        if n:
            yield recibe[a:a + n].to(dev), todo[otro, :n]
        del buf, todo


def preparar(layer, qp: torch.Tensor, esc: torch.Tensor) -> None:
    """Desde PN139, con el lm_head de este rango en int4 formato GPTQ, ANTES del repack de Marlin:
    ``qp`` [K/8, N] int32 y ``esc`` [K/G, N] fp16, una COLUMNA por token. Se eligen columnas: los int4 y las
    escalas son exactamente los del lm_head completo (sin recuantizar; el lm_head ya esta en la base rotada
    del residuo y la entrada llega rotada por PN149)."""
    global _HECHO
    if not ACTIVO or _HECHO:
        # Una sola vez: la primera es el lm_head del target. El borrador carga el suyo, PN139 lo procesa y
        # load_dflash_model lo BORRA y le pone el del target (dflash/utils.py): armarlo ahi seria otro
        # all-gather y memoria tirada.
        return
    _HECHO = True
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_act_int8_process_scales, marlin_permute_scales)
        from vllm import _custom_ops as ops
        from vllm._genesis import lm_head_int4 as _g139

        k, g = qp.shape[0] * 8, qp.shape[0] * 8 // esc.shape[0]
        ini = int(layer.shard_indices.org_vocab_start_index)
        # una fila por token: sus K/8 int32 de int4 + sus K/G escalas fp16 vistas como int32 (bajo demanda,
        # por trozos: el lm_head entero asi serian 328 MB de transitorio)
        def filas_de(i):
            return torch.cat([qp.t().index_select(0, i), esc.t().index_select(0, i).contiguous().view(torch.int32)], 1)

        tids, tfil = [], []
        for ids_t, f in _repartir(filas_de, k // 8 + esc.shape[0] // 2, ini, qp.device):
            tids.append(ids_t); tfil.append(f)
        ids, f = torch.cat(tids), torch.cat(tfil)
        del tids, tfil
        n = f.shape[0]
        qs = f[:, :k // 8].t().contiguous()                               # [K/8, n]
        es = f[:, k // 8:].contiguous().view(esc.dtype).t().contiguous()  # [K/G, n]
        del f
        es_a8 = _g139.a8()
        layer.pn159_w = ops.gptq_marlin_repack(qs, torch.empty(0, dtype=torch.int32, device=qp.device),
                                               k, n, 4, is_a_8bit=es_a8)
        em = marlin_permute_scales(es, size_k=k, size_n=n, group_size=g, is_a_8bit=es_a8)
        layer.pn159_igs = None
        if es_a8:
            em, layer.pn159_igs = marlin_act_int8_process_scales(em)
        layer.pn159_s = em
        layer.pn159_ids = ids                                            # int64, globales
        layer.pn159_n, layer.pn159_k = n, k
        del qs, es
        if DIN:
            _preparar_dinamico(layer, qp, esc, ini, int(layer.shard_indices.org_vocab_end_index))
        torch.cuda.empty_cache()   # lo que carga despues (el borrador) asigna en el pool de pesos de vLLM
        log.warning("PN159: candidatos del borrador: %d columnas del lm_head int4 en este rango (%.0f MB), "
                    "vocabulario %d", n, n * k / 2 / 1e6, TAM)
    except Exception as e:      # sin subconjunto el borrador usa el lm_head completo
        log.error("PN159: no se armo el vocabulario recortado (%s: %s)", type(e).__name__, e)
        for a in ("pn159_w", "pn159_ids"):
            if hasattr(layer, a):
                delattr(layer, a)


def _preparar_dinamico(layer, qp, esc, ini, fin) -> None:
    """Fase 2: copia token-major del lm_head int4 de este tramo en memoria pinned del host (la lee SK-27 por
    UVA, sin CPU en el lazo), bitmaps de pertenencia y el anillo de DIN filas fp16 en la GPU."""
    import numpy as np
    dev = qp.device
    n_tr, kw, kg = fin - ini, qp.shape[0], esc.shape[0]
    hq = torch.empty(n_tr, kw, dtype=torch.int32, pin_memory=True)
    hs = torch.empty(n_tr, kg, dtype=torch.float16, pin_memory=True)
    for a in range(0, n_tr, 8192):                                    # por trozos: la carga llega justa
        b = min(a + 8192, n_tr)
        hq[a:b].copy_(qp[:, a:b].t())
        hs[a:b].copy_(esc[:, a:b].t())
    orden = torch.from_numpy(np.load(_ORDEN).astype("int64"))       # todo el vocabulario, ordenado
    palabras = (max(int(orden.max()) + 1, fin) + 31) // 32 + 1
    s_tot = -(-TAM // (256 * layer.tp_size)) * 256 * layer.tp_size
    est = torch.zeros(palabras * 32, dtype=torch.bool)
    est[orden[:s_tot]] = True
    bits = (est.view(-1, 32).to(torch.int64) << torch.arange(32)).sum(1)
    estatico = torch.where(bits >= 2 ** 31, bits - 2 ** 32, bits).to(torch.int32).to(dev)
    k = kw * 8
    _din.update(hq=hq, hs=hs, estatico=estatico, dinbit=torch.zeros(palabras, dtype=torch.int32, device=dev),
                ids=torch.full((DIN,), -1, dtype=torch.int32, device=dev),
                filas=torch.zeros(DIN, k, dtype=torch.float16, device=dev),
                cabeza=torch.zeros(1, dtype=torch.int32, device=dev), ini=ini, fin=fin, k=k, g=k // kg)
    layer.pn159_din = _din
    log.warning("PN159 fase 2: anillo de %d filas (%.0f MB en la GPU), lm_head del tramo en el host (%.0f MB pinned)",
                DIN, DIN * k * 2 / 1e6, (hq.numel() * 4 + hs.numel() * 2) / 1e6)


def _kernel():
    if "kern" not in _din:
        from vllm._genesis.kernels.ptx_lab import Kernel
        kr = Kernel("sk27_vocab_dinamico.cu", "sk27_observar",
                    defs=[f"-DK={_din['k']}", f"-DG={_din['g']}", f"-DDMAX={max(DIN, 32)}"], warps=32)
        kr.cargar()
        _din["kern"] = kr
    return _din["kern"]


def observar_cuda(ids: torch.Tensor, cabeza: torch.Tensor) -> None:
    d = _din
    n = ids.numel()
    if n == 0:
        return
    x = ids.reshape(-1)
    if x.dtype not in (torch.int32, torch.int64):
        x = x.to(torch.int64)
    _kernel().lanzar((1, 1), [x, n, 1 if x.dtype == torch.int64 else 0, d["estatico"], d["dinbit"], d["ini"], d["fin"],
                              d["hq"], d["hs"], d["ids"], d["filas"], cabeza, DIN])


@torch.library.custom_op("genesis::pn159_observar", mutates_args=("cabeza",))
def _observar_op(ids: torch.Tensor, cabeza: torch.Tensor) -> None:
    observar_cuda(ids, cabeza)


@_observar_op.register_fake
def _observar_fake(ids, cabeza):
    return None


def observar(input_ids: torch.Tensor) -> None:
    """Desde el embedding (PN159 fase 2): los tokens del paso alimentan el anillo dinamico."""
    if "cabeza" in _din and input_ids.is_cuda:
        torch.ops.genesis.pn159_observar(input_ids, _din["cabeza"])


def _kern(nombre: str, extra=()):
    if nombre not in _din:
        from vllm._genesis.kernels.ptx_lab import Kernel
        kr = Kernel("sk27_vocab_dinamico.cu", nombre, defs=[f"-DK={_din['k']}", f"-DG={_din['g']}",
                                                             f"-DDMAX={max(DIN, 32)}", *extra], warps=8)
        kr.cargar()
        _din[nombre] = kr
    return _din[nombre]


def anillo_cuda(h: torch.Tensor, vs: torch.Tensor, is_: torch.Tensor):
    d = _din
    M, kt = vs.shape
    x = h.reshape(M, -1).to(torch.float16).contiguous()
    ld = torch.empty(M, DIN, dtype=torch.float32, device=h.device)
    _kern("sk27_logits").lanzar((-(-DIN // 8), -(-M // 16)), [x, M, d["filas"], d["ids"], ld, DIN])
    vo = torch.empty(M, kt, dtype=torch.float16, device=h.device)
    io = torch.empty(M, kt, dtype=torch.int64, device=h.device)
    _kern("sk27_fusion", (f"-DKT={kt}",)).lanzar((M, 1), [vs.to(torch.float16).contiguous(), is_.contiguous(), ld,
                                                          d["ids"], DIN, vo, io], shared=(kt + DIN) * 4)
    return vo.to(vs.dtype), io


@torch.library.custom_op("genesis::pn159_anillo", mutates_args=())
def _anillo_op(h: torch.Tensor, vs: torch.Tensor, is_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return anillo_cuda(h, vs, is_)


@_anillo_op.register_fake
def _anillo_fake(h, vs, is_):
    return torch.empty_like(vs), torch.empty_like(is_)


_SK28 = os.environ.get("GENESIS_PN159_SK28", "0") == "1"      # apagado hasta validarlo en el servidor
_k28: dict = {}


def _kern28(nombre: str, k: int, mt: int = 1):
    clave = (nombre, mt)
    if clave not in _k28:
        from vllm._genesis.kernels.ptx_lab import Kernel
        w = {"sk28_anillo": 8, "sk28_parcial": 8, "sk28_final": 1}[nombre]
        kr = Kernel("sk28_topk_borrador.cu", nombre, defs=[f"-DK={k}", "-DKT=16", f"-DMT={mt}"], warps=w)
        kr.cargar()
        _k28[clave] = kr
    return _k28[clave]


def topk_cuda(logits: torch.Tensor, h: torch.Tensor, ids_fijo: torch.Tensor):
    M, S = logits.shape
    dev = logits.device
    lg = (logits if logits.dtype == torch.float16 else logits.to(torch.float16)).contiguous()
    d = _din if "filas" in _din else None
    Dn = DIN if d is not None else 0
    vo = torch.empty(M, 16, dtype=torch.float32, device=dev)
    io = torch.empty(M, 16, dtype=torch.int64, device=dev)
    if d is not None:
        x = h.reshape(M, -1).to(torch.float16).contiguous()
        ld = torch.empty(M, Dn, dtype=torch.float32, device=dev)
        assert M <= 64, "sk28_anillo: MT=4 tiles de 16"
        _kern28("sk28_anillo", x.shape[-1], -(-M // 16)).lanzar((-(-Dn // 8), 1), [x, M, d["filas"], d["ids"], ld, Dn])
        din_ids = d["ids"]
    else:
        ld, din_ids = vo, ids_fijo                                   # no se leen con D = 0
    if "sk29" not in _k28:
        from vllm._genesis.kernels.ptx_lab import Kernel
        _k28["sk29"] = Kernel("sk29_topk_filtro.cu", "sk29_topk", warps=16)
        _k28["sk29"].cargar()
    _k28["sk29"].lanzar((M, 1), [lg, S, ids_fijo, ld, din_ids, Dn, vo, io], shared=((S + 31) // 32) * 4)
    return vo, io


@torch.library.custom_op("genesis::pn159_topk", mutates_args=())
def _topk_op(logits: torch.Tensor, h: torch.Tensor, ids_fijo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return topk_cuda(logits, h, ids_fijo)


@_topk_op.register_fake
def _topk_fake(logits, h, ids_fijo):
    M = logits.shape[0]
    return (logits.new_empty((M, 16), dtype=torch.float32), logits.new_empty((M, 16), dtype=torch.int64))


def _logits(layer, x: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear
    vacio = torch.empty(0, dtype=torch.int32, device=x.device)
    return apply_gptq_marlin_linear(
        input=x, weight=layer.pn159_w, weight_scale=layer.pn159_s, weight_zp=vacio, g_idx=vacio,
        g_idx_sort_indices=vacio, workspace=layer.workspace, wtype=layer.pn139_tipo,
        input_size_per_partition=layer.pn159_k, output_size_per_partition=layer.pn159_n, is_k_full=True,
        input_global_scale=layer.pn159_igs, input_dtype=torch.int8 if layer.pn159_igs is not None else None)


def top_k(proc, lm_head, hidden_states: torch.Tensor, k: int):
    """Reemplaza ``proc.get_top_k_tokens(lm_head, hidden_states, k)`` (mismo contrato: ids globales int64
    y valores fp32 con escala y soft cap)."""
    if getattr(lm_head, "pn159_w", None) is None:
        return proc.get_top_k_tokens(lm_head, hidden_states, k)
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.model_executor.layers.logits_processor import _topk

    logits = _logits(lm_head, hidden_states)
    if k == 16 and _SK28:
        # SK-28: top-k del fijo (+ anillo de la fase 2) con los ids ya mapeados, sin flashinfer ni casts
        values, ids = torch.ops.genesis.pn159_topk(logits, hidden_states, lm_head.pn159_ids)
    else:
        values, idx = _topk(logits, k)
        ids = lm_head.pn159_ids[idx.to(torch.int64)]
        if getattr(lm_head, "pn159_din", None) is not None:
            values, ids = torch.ops.genesis.pn159_anillo(hidden_states, values, ids)
    if lm_head.tp_size > 1:
        values = tensor_model_parallel_all_gather(values, dim=-1)
        ids = tensor_model_parallel_all_gather(ids, dim=-1)
        values, sel = _topk(values, k)
        ids = ids.gather(-1, sel)
    values = values.float()
    if proc.scale != 1.0:
        values = values * proc.scale
    if proc.soft_cap is not None:
        values = torch.tanh(values / proc.soft_cap) * proc.soft_cap
    return ids, values
