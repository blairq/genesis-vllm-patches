# SPDX-License-Identifier: Apache-2.0
"""Arbol de borrador — hito 4: la convolucion causal de GDN, por CAMINO.

La conv1d de GDN (ancho 4) mezcla cada token con los 3 anteriores. En un paso en arbol "los 3
anteriores" de un nodo no son los 3 tokens previos del lote sino sus 3 ANCESTROS mas cercanos
(y, si no alcanzan, el ancla y la historia del estado conv).

El kernel de upstream (``causal_conv1d_update``) no se toca, porque ademas de las salidas hace
algo que sigue sirviendo tal cual: escribir el estado ``[h-2, h-1, x0, x1, ..., xK]``. Entonces:

* ``salidas``: calcula aparte las salidas por camino. Se llama ANTES que upstream (que pisa
  ``x`` con sus salidas de cadena) y el resultado se copia encima despues.
* ``compactar``: despues de aceptar. El paso siguiente lee las columnas ``r, r+1, r+2`` del
  estado (``r`` = borradores aceptados) creyendo que ahi estan los 3 ultimos tokens aceptados.
  Con una cadena lo estan; con un arbol el j-esimo aceptado es el nodo ``p_j`` y vive en la
  columna ``2 + p_j``, asi que se copian a lo sumo 3 columnas. Nunca se lee una columna ya
  pisada: el origen de la columna ``r+i`` es ``2 + p_(r+i-2) >= r+i``, estrictamente creciente.

Todo en GPU, forma fija, sin sincronizar. Con la mascara de cadena ``salidas`` da lo mismo que
upstream y ``compactar`` no copia nada.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _k_salidas(x, stride_xt, out, stride_ot, cs, stride_cs_seq, stride_cs_dim, stride_cs_tok,
               w, stride_wd, stride_ww, cu, sidx, nacc, anc3, s_a3, DIM,
               BN: tl.constexpr, SILU: tl.constexpr, ESCRIBIR: tl.constexpr,
               SL: tl.constexpr, T: tl.constexpr):
    i_n, i_f = tl.program_id(0), tl.program_id(1)
    bos = tl.load(cu + i_n).to(tl.int64)
    T = tl.load(cu + i_n + 1).to(tl.int64) - bos
    s = tl.load(sidx + i_n).to(tl.int64)
    if T == 0 or s <= 0:
        return
    f = i_f * BN + tl.arange(0, BN)
    mf = f < DIM
    off = tl.load(nacc + i_n).to(tl.int64) - 1
    hist = cs + s * stride_cs_seq + f * stride_cs_dim + off * stride_cs_tok
    hA = tl.load(hist, mask=mf, other=0.0)                      # la mas vieja
    hB = tl.load(hist + stride_cs_tok, mask=mf, other=0.0)
    hC = tl.load(hist + 2 * stride_cs_tok, mask=mf, other=0.0)
    w0 = tl.load(w + f * stride_wd, mask=mf, other=0.0)
    w1 = tl.load(w + f * stride_wd + stride_ww, mask=mf, other=0.0)
    w2 = tl.load(w + f * stride_wd + 2 * stride_ww, mask=mf, other=0.0)
    w3 = tl.load(w + f * stride_wd + 3 * stride_ww, mask=mf, other=0.0)
    # Los TRES ancestros mas cercanos de cada token vienen precalculados (``anc3``, una vez por
    # paso en vez de 48 veces): indice local, o -1/-2/-3 para las columnas de historia. Antes se
    # buscaban aca con un lazo O(T^2) sobre los bits de ancestros, con una carga de BN elementos
    # por iteracion y ramas anidadas para contarlos.
    for t in range(0, T):
        xt = tl.load(x + (bos + t) * stride_xt + f, mask=mf, other=0.0)
        p3 = anc3 + (bos + t) * s_a3
        i1 = tl.load(p3 + 0).to(tl.int64)
        i2 = tl.load(p3 + 1).to(tl.int64)
        i3 = tl.load(p3 + 2).to(tl.int64)
        t1 = tl.load(x + (bos + tl.maximum(i1, 0)) * stride_xt + f, mask=mf & (i1 >= 0), other=0.0)
        t2 = tl.load(x + (bos + tl.maximum(i2, 0)) * stride_xt + f, mask=mf & (i2 >= 0), other=0.0)
        t3 = tl.load(x + (bos + tl.maximum(i3, 0)) * stride_xt + f, mask=mf & (i3 >= 0), other=0.0)
        t1 = tl.where(i1 >= 0, t1, tl.where(i1 == -1, hC, tl.where(i1 == -2, hB, hA)))
        t2 = tl.where(i2 >= 0, t2, tl.where(i2 == -1, hC, tl.where(i2 == -2, hB, hA)))
        t3 = tl.where(i3 >= 0, t3, tl.where(i3 == -1, hC, tl.where(i3 == -2, hB, hA)))
        # Igual que upstream, para que la cadena de bit a bit lo mismo: cada producto en el
        # dtype de x (fp16) y la suma en fp32, de la columna mas vieja a la mas nueva.
        acc = tl.zeros((BN,), dtype=tl.float32)
        acc += t3 * w0
        acc += t2 * w1
        acc += t1 * w2
        acc += xt * w3
        if SILU:
            acc = acc / (1 + tl.exp(-acc))
        tl.store(out + (bos + t) * stride_ot + f, acc.to(out.dtype.element_ty), mask=mf)
    if ESCRIBIR:
        # El estado conv que deja upstream es un desplazamiento: [h[off+1], h[off+2], x_0..x_{T-1}]
        # (state_len = 3 + K y seqlen = T = K+1, asi que sobreviven dos columnas de historia).
        # Escribirlo aca evita volver a llamar al kernel de upstream SOLO por el estado, que es lo
        # que hacia correr la conv DOS veces (6,7 -> 13,6 us por capa, medido).
        # Se hace al final: hB/hC ya estan en registros y no se lee mas del estado.
        dst = cs + s * stride_cs_seq + f * stride_cs_dim
        tl.store(dst + 0 * stride_cs_tok, hB.to(cs.dtype.element_ty), mask=mf)
        tl.store(dst + 1 * stride_cs_tok, hC.to(cs.dtype.element_ty), mask=mf)
        for c in range(0, T):
            tl.store(dst + (c + 2) * stride_cs_tok,
                     tl.load(x + (bos + c) * stride_xt + f, mask=mf, other=0.0), mask=mf)


@triton.jit
def _conv_token(x, stride_xt, out, stride_ot, anc3, s_a3, bos, t, vt, f, mf, hA, hB, hC,
                w0, w1, w2, w3, BN: tl.constexpr, SILU: tl.constexpr):
    """Salida de la conv para el token ``t`` (misma cuenta y mismo orden que ``_k_salidas``)."""
    m = mf & vt
    xt = tl.load(x + (bos + t) * stride_xt + f, mask=m, other=0.0)
    p3 = anc3 + (bos + t) * s_a3
    i1 = tl.load(p3 + 0, mask=vt, other=0).to(tl.int64)
    i2 = tl.load(p3 + 1, mask=vt, other=0).to(tl.int64)
    i3 = tl.load(p3 + 2, mask=vt, other=0).to(tl.int64)
    t1 = tl.load(x + (bos + tl.maximum(i1, 0)) * stride_xt + f, mask=m & (i1 >= 0), other=0.0)
    t2 = tl.load(x + (bos + tl.maximum(i2, 0)) * stride_xt + f, mask=m & (i2 >= 0), other=0.0)
    t3 = tl.load(x + (bos + tl.maximum(i3, 0)) * stride_xt + f, mask=m & (i3 >= 0), other=0.0)
    t1 = tl.where(i1 >= 0, t1, tl.where(i1 == -1, hC, tl.where(i1 == -2, hB, hA)))
    t2 = tl.where(i2 >= 0, t2, tl.where(i2 == -1, hC, tl.where(i2 == -2, hB, hA)))
    t3 = tl.where(i3 >= 0, t3, tl.where(i3 == -1, hC, tl.where(i3 == -2, hB, hA)))
    acc = tl.zeros((BN,), dtype=tl.float32)
    acc += t3 * w0
    acc += t2 * w1
    acc += t1 * w2
    acc += xt * w3
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    tl.store(out + (bos + t) * stride_ot + f, acc.to(out.dtype.element_ty), mask=m)


@triton.jit
def _k_salidas_par(x, stride_xt, out, stride_ot, cs, stride_cs_seq, stride_cs_dim, stride_cs_tok,
                   w, stride_wd, stride_ww, cu, sidx, nacc, anc3, s_a3, DIM,
                   BN: tl.constexpr, SILU: tl.constexpr, ESCRIBIR: tl.constexpr,
                   SL: tl.constexpr, T: tl.constexpr):
    """``_k_salidas`` con el lazo de tokens DESENROLLADO y tramos de features mas chicos.

    ``_k_salidas`` recorre los tokens con un lazo de largo variable: cada vuelta espera sus
    cargas antes de la siguiente, y con BN = 1024 son 5 bloques para 82 SM (9 us por capa, 48
    capas). Repartir los tokens en programas distintos no se puede: el mismo kernel reescribe al
    final las columnas del estado conv que los otros programas leen como historia. Aca los
    primeros T tokens (T = tokens por pedido del lote uniforme del arbol) van en
    ``static_range``, asi que sus cargas salen juntas, y un pedido mas largo que T (lote
    irregular) termina en un lazo dinamico de cola. BN chico da mas bloques. Mismas cuentas en
    el mismo orden: bit a bit igual.
    """
    i_n, i_f = tl.program_id(0), tl.program_id(1)
    bos = tl.load(cu + i_n).to(tl.int64)
    TT = tl.load(cu + i_n + 1).to(tl.int64) - bos
    s = tl.load(sidx + i_n).to(tl.int64)
    if TT == 0 or s <= 0:
        return
    f = i_f * BN + tl.arange(0, BN)
    mf = f < DIM
    off = tl.load(nacc + i_n).to(tl.int64) - 1
    hist = cs + s * stride_cs_seq + f * stride_cs_dim + off * stride_cs_tok
    hA = tl.load(hist, mask=mf, other=0.0)
    hB = tl.load(hist + stride_cs_tok, mask=mf, other=0.0)
    hC = tl.load(hist + 2 * stride_cs_tok, mask=mf, other=0.0)
    w0 = tl.load(w + f * stride_wd, mask=mf, other=0.0)
    w1 = tl.load(w + f * stride_wd + stride_ww, mask=mf, other=0.0)
    w2 = tl.load(w + f * stride_wd + 2 * stride_ww, mask=mf, other=0.0)
    w3 = tl.load(w + f * stride_wd + 3 * stride_ww, mask=mf, other=0.0)
    for t in tl.static_range(T):
        _conv_token(x, stride_xt, out, stride_ot, anc3, s_a3, bos, t, t < TT, f, mf, hA, hB, hC,
                    w0, w1, w2, w3, BN, SILU)
    for t in range(T, TT):
        _conv_token(x, stride_xt, out, stride_ot, anc3, s_a3, bos, t, t < TT, f, mf, hA, hB, hC,
                    w0, w1, w2, w3, BN, SILU)
    if ESCRIBIR:
        dst = cs + s * stride_cs_seq + f * stride_cs_dim
        tl.store(dst + 0 * stride_cs_tok, hB.to(cs.dtype.element_ty), mask=mf)
        tl.store(dst + 1 * stride_cs_tok, hC.to(cs.dtype.element_ty), mask=mf)
        for c in tl.static_range(T):
            mc = mf & (c < TT)
            tl.store(dst + (c + 2) * stride_cs_tok,
                     tl.load(x + (bos + c) * stride_xt + f, mask=mc, other=0.0), mask=mc)
        for c in range(T, TT):
            tl.store(dst + (c + 2) * stride_cs_tok,
                     tl.load(x + (bos + c) * stride_xt + f, mask=mf, other=0.0), mask=mf)


_SALIDAS_MODO = __import__("os").environ.get("GENESIS_ARBOL_SALIDAS_PAR", "0").strip().lower()
_SALIDAS_MODO = "par" if _SALIDAS_MODO in ("1", "true", "yes", "on") else ("ptx" if _SALIDAS_MODO == "ptx" else "serie")
_k_conv_ptx: dict = {}
_FPT, _NHILOS = 1, 32               # barrido 27-09: 3,5 us (1 pedido) y 5,2 (4) contra 5,4/7,8 de Triton


_avisado: set = set()


def _ptx_valido(x, out, conv_state, weight):
    """El PTX carga x/out de a FPT fp16; el estado y los pesos van con strides. Devuelve el motivo si no sirve."""
    al = lambda t: t.data_ptr() % (2 * _FPT) == 0
    if not (x.dtype == out.dtype == conv_state.dtype == weight.dtype == torch.float16):
        return f"dtypes x={x.dtype} out={out.dtype} estado={conv_state.dtype} w={weight.dtype}"
    if not (x.stride(1) == 1 and out.stride(1) == 1 and x.stride(0) % _FPT == 0 and out.stride(0) % _FPT == 0
            and x.shape[1] % _FPT == 0 and al(x) and al(out)):
        return f"layout x {tuple(x.stride())} out {tuple(out.stride())}"
    if weight.dim() != 2 or weight.shape[1] != 4:
        return f"pesos {tuple(weight.shape)}"
    return None


def _avisar(motivo):
    if motivo not in _avisado:
        _avisado.add(motivo)
        import logging
        logging.getLogger("genesis.arbol_conv").warning("[arbol_conv] PTX no aplica, uso Triton: %s", motivo)


def salidas(x, conv_state, weight, activation, conv_state_indices, num_accepted_tokens,
            query_start_loc, anc3, out=None, escribir_estado=False, par=None):
    # par: None = GENESIS_ARBOL_SALIDAS_PAR (0 serie | 1 Triton desenrollado | ptx), o forzado
    # (False / True / "ptx") para los tests. Los tres dan lo mismo bit a bit.
    """Mismos argumentos que ``causal_conv1d_update`` en el camino spec (``x`` [tokens, dim],
    ``conv_state`` [bloques, dim, state_len], ``weight`` [dim, 4], sin bias) mas ``anc3`` (ver ``arbol_borrador.preparar_paso_kernel``).
    Devuelve las salidas por camino en ``out`` (no toca ``x``). Con ``escribir_estado`` deja ademas
    el estado conv desplazado, igual que upstream, y entonces NO hay que llamar al kernel de
    upstream: la conv corre una sola vez."""
    assert weight.shape[1] == 4 and x.stride(1) == 1
    if out is None:
        out = torch.empty_like(x)
    N = query_start_loc.shape[0] - 1
    dim = x.shape[1]
    BN = 1024
    conv_state_indices = conv_state_indices.contiguous()      # en vLLM llega como columna de un 2D
    # T = tokens por pedido del lote uniforme del arbol (lo que se desenrolla). NO sale del largo de
    # anc3: en el servidor es un buffer fijo de n_slots*(K+1) filas, y anc3.shape[0] // N con un
    # pedido daba ~n_slots*9 -> lazo desenrollado de cientos de tokens enmascarados: 122 us por capa
    # (perfil is27_p1, 27-09). Los pedidos mas largos que T siguen en el lazo de cola.
    from vllm._genesis import gdn_cinta
    T = gdn_cinta.tokens_arbol() or 9
    modo = _SALIDAS_MODO if par is None else ("ptx" if par == "ptx" else ("par" if par else "serie"))
    motivo = _ptx_valido(x, out, conv_state, weight) if modo == "ptx" else "no pedido"
    if modo == "ptx" and motivo is not None:
        _avisar(motivo)
    if modo == "ptx" and motivo is None:
        silu = activation in ("silu", "swish", True)
        clave = (T, silu, bool(escribir_estado))
        kern = _k_conv_ptx.get(clave)
        if kern is None:
            from vllm._genesis.kernels.ptx_lab import Kernel
            kern = _k_conv_ptx[clave] = Kernel(
                "arbol_conv.cu", "arbol_conv", warps=_NHILOS // 32,
                defs=[f"-DTOK={T}", f"-DFPT={_FPT}", f"-DNHILOS={_NHILOS}", f"-DSILU={int(silu)}",
                      f"-DESCRIBIR={int(bool(escribir_estado))}"])
        import ctypes
        kern.lanzar((N, triton.cdiv(dim, _NHILOS * _FPT)),
                    [x, x.stride(0), out, out.stride(0), conv_state,
                     ctypes.c_longlong(conv_state.stride(0)), conv_state.stride(1), conv_state.stride(2),
                     weight, weight.stride(0), weight.stride(1),
                     query_start_loc, conv_state_indices, num_accepted_tokens, anc3, anc3.stride(0), dim])
        return out
    if modo in ("par", "ptx"):
        BN = 64                                   # barrido 27-09: el mejor en 3 de 4 casos (1/4 pedidos,
        _k_salidas_par[(N, triton.cdiv(dim, BN))](  # con y sin estado); 80 bloques con un pedido
            x, x.stride(0), out, out.stride(0), conv_state, conv_state.stride(0),
            conv_state.stride(1), conv_state.stride(2), weight, weight.stride(0), weight.stride(1),
            query_start_loc, conv_state_indices, num_accepted_tokens, anc3, anc3.stride(0), dim,
            BN=BN, SILU=activation in ("silu", "swish", True), ESCRIBIR=escribir_estado,
            SL=conv_state.shape[-1], T=T, num_warps=2)
        return out
    _k_salidas[(N, triton.cdiv(dim, BN))](
        x, x.stride(0), out, out.stride(0), conv_state, conv_state.stride(0),
        conv_state.stride(1), conv_state.stride(2), weight, weight.stride(0), weight.stride(1),
        query_start_loc, conv_state_indices, num_accepted_tokens, anc3, anc3.stride(0), dim,
        BN=BN, SILU=activation in ("silu", "swish", True), ESCRIBIR=escribir_estado,
        SL=conv_state.shape[-1], T=T, num_warps=4)
    return out


@triton.jit(do_not_specialize=["num_reqs"])
def _k_compactar(conv_addrs, grupos, bloques, stride_bloq_g, nacc, camino, fila_camino, num_reqs,
                 stride_cs_seq, stride_cs_dim, stride_cs_tok, DIM,
                 TM: tl.constexpr, BN: tl.constexpr):
    req, lay = tl.program_id(0), tl.program_id(1)
    if req >= num_reqs:
        return
    r = tl.load(nacc + req).to(tl.int64) - 1
    if r <= 0:
        return
    g = tl.load(grupos + lay).to(tl.int64)
    blk = tl.load(bloques + g * stride_bloq_g + req).to(tl.int64)
    if blk <= 0:
        return
    fc = tl.load(fila_camino + req).to(tl.int64)
    base = tl.load(conv_addrs + lay).to(tl.pointer_type(tl.float16)) + blk * stride_cs_seq
    for i in range(0, 3):
        sq = r + i - 3                            # indice en el camino aceptado (0 = p_1)
        if sq >= 0:
            src = 3 + tl.load(camino + fc * TM + sq).to(tl.int64)
            dst = r + i
            if src != dst:
                for c in range(0, DIM, BN):
                    f = c + tl.arange(0, BN)
                    mf = f < DIM
                    val = tl.load(base + f * stride_cs_dim + src * stride_cs_tok, mask=mf, other=0.0)
                    tl.store(base + f * stride_cs_dim + dst * stride_cs_tok, val, mask=mf)


def compactar(conv_addrs, grupos, bloques, nacc, camino, fila_camino, num_reqs, strides, dim):
    """Despues de aceptar. ``conv_addrs`` [capas] int64 (``data_ptr`` del estado conv fp16 de cada
    capa, todas con los mismos ``strides`` = (bloque, dim, columna) en ELEMENTOS), ``grupos``
    [capas] -> fila de ``bloques`` [G, reqs] (bloque del estado de cada pedido), ``nacc`` [reqs]
    = aceptados del paso (ancla incluida), ``camino`` [slots, TM] filas de cinta del camino
    aceptado (nodo - 1) y ``fila_camino`` [reqs] = que fila de ``camino`` usa cada pedido."""
    _k_compactar[(num_reqs, conv_addrs.shape[0])](
        conv_addrs, grupos, bloques, bloques.stride(0), nacc, camino, fila_camino, num_reqs,
        strides[0], strides[1], strides[2], dim, TM=camino.shape[1], BN=1024, num_warps=4)


@triton.jit(do_not_specialize=["num_reqs"])
def _k_compactar_v2(conv_addrs, grupos, bt_ptrs, bt_stride: tl.int64, state_idx, idx_map, nacc,
                    camino, num_reqs, stride_cs_seq, stride_cs_dim, stride_cs_tok, DIM,
                    TM: tl.constexpr, BN: tl.constexpr):
    """Como ``_k_compactar``, con el direccionamiento del runner v2 (el mismo de
    ``gdn_cinta._k_materializar``): el bloque del estado sale de la block table del grupo, en la
    columna ``state_idx[slot de req-state]``; la fila de ``camino`` es ``slot + 1``; ``nacc`` viene
    por fila del lote (es ``num_sampled``, todavia sin volcar a ``num_accepted_tokens``)."""
    req, lay = tl.program_id(0), tl.program_id(1)
    if req >= num_reqs:
        return
    ri = tl.load(idx_map + req)
    if ri < 0:
        return
    r = tl.load(nacc + req).to(tl.int64) - 1
    if r <= 0:
        return
    col = tl.load(state_idx + ri)
    if col < 0:
        return
    g = tl.load(grupos + lay).to(tl.int64)
    bt = tl.load(bt_ptrs + g).to(tl.pointer_type(tl.int32)) + req * bt_stride
    blk = tl.load(bt + col).to(tl.int64)
    if blk <= 0:
        return
    base = tl.load(conv_addrs + lay).to(tl.pointer_type(tl.float16)) + blk * stride_cs_seq
    fc = (ri + 1).to(tl.int64)
    for i in range(0, 3):
        sq = r + i - 3
        if sq >= 0:
            src = 3 + tl.load(camino + fc * TM + sq).to(tl.int64)
            dst = r + i
            if src != dst:
                for c in range(0, DIM, BN):
                    f = c + tl.arange(0, BN)
                    mf = f < DIM
                    val = tl.load(base + f * stride_cs_dim + src * stride_cs_tok, mask=mf, other=0.0)
                    tl.store(base + f * stride_cs_dim + dst * stride_cs_tok, val, mask=mf)


_v2 = {}


def compactar_v2(ctx, capas, grupos, state_idx, idx_mapping, nacc, camino, num_reqs) -> None:
    """``capas`` = capas GDN en el orden de ``grupos``. El estado conv es ``capa.kv_cache[0]``:
    2 bytes por elemento (se copia crudo, asi que fp16 y bf16 valen igual), con el eje de
    columnas de largo ``ancho - 1 + K``."""
    if not capas or num_reqs == 0:
        return
    if not _v2:
        c0 = capas[0].kv_cache[0]
        assert c0.element_size() == 2 and c0.dim() == 3, "estado conv inesperado"
        sl = 3 + camino.shape[1]
        eje_tok = 1 if c0.shape[1] == sl else 2
        assert c0.shape[eje_tok] == sl, f"estado conv {tuple(c0.shape)}: no hay eje de {sl} columnas"
        eje_dim = 3 - eje_tok
        _v2["strides"] = (c0.stride(0), c0.stride(eje_dim), c0.stride(eje_tok))
        _v2["dim"] = c0.shape[eje_dim]
        _v2["addrs"] = torch.tensor([c.kv_cache[0].data_ptr() for c in capas], dtype=torch.int64,
                                    device=c0.device)
    s = _v2["strides"]
    _k_compactar_v2[(num_reqs, len(capas))](
        _v2["addrs"], grupos, ctx.block_table_ptrs, ctx.block_table_stride_req, state_idx,
        idx_mapping, nacc, camino, num_reqs, s[0], s[1], s[2], _v2["dim"],
        TM=camino.shape[1], BN=1024, num_warps=4)


__all__ = ["salidas", "compactar", "compactar_v2"]
