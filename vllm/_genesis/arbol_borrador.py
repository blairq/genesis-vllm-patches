# SPDX-License-Identifier: Apache-2.0
"""Arbol de borrador para DFlash2 — hito 1: el CONSTRUCTOR (todavia sin cablear a nada).

Por que un arbol, y por que de 8 nodos
--------------------------------------
DFlash2 propone, por cada una de sus 8 posiciones, 16 candidatos y una tabla de puntajes de
transicion ``scores[paso, candidato previo, candidato]``; hoy se recorre UN camino, goloso. La
aceptacion es una cadena: el primer fallo tira todo lo que sigue, asi que las ultimas posiciones
del camino casi nunca se aprovechan (en prosa aceptan < 10%).

Medido en este rig (2026-09-21): el paso de decode cuesta ~18,8 ms + 3,8 ms por pedido extra +
0,42 ms por POSICION, y con 6 pedidos esta parado justo en el codo de ~54 tokens por lote. O sea
que no se puede verificar mas tokens ni mas profundidad sin pagarlo. Lo que si sale gratis es
gastar MEJOR los mismos 8: reemplazar la cola improbable del camino por ramas cerca de la raiz.
Simulado sobre 2.080 pasos reales: aceptacion +17% en prosa, +2% en codigo.

Que hace
--------
``construir`` elige los N nodos de mayor probabilidad de camino (best-first, como DDTree,
arXiv 2604.12989, pero con puntajes condicionados al padre, que DFlash2 ya da). Salida por nodo:
token, padre (-1 = la raiz/ancla), profundidad e indice de candidato. Los nodos salen en orden
topologico (un padre siempre antes que sus hijos), que es lo que necesitan la mascara de
ancestros, el GDN y la compactacion de la KV.

Todo en la GPU y sin sincronizar: son operaciones de tensores de forma fija, capturables en el
CUDA graph del borrador.
"""

from __future__ import annotations

import torch


def construir(cand: torch.Tensor, sc: torch.Tensor, n_nodos: int):
    """cand [R, S, K] int64, sc [R, S, K, K] float (logits de transicion; la fila 0 del paso 0
    es la del ancla). Devuelve (token, padre, prof, idx), todos [R, n_nodos].

    Best-first exacto sin heap: la frontera cabe en un tensor de forma fija. Cada nodo elegido
    agrega a la frontera a sus K hijos; se hacen ``n_nodos`` rondas de argmax.
    """
    R, S, K = cand.shape
    dev = cand.device
    lp = torch.log_softmax(sc.float(), dim=-1)                       # [R, S, K(prev), K]
    F = K * (n_nodos + 1)                                            # tope de la frontera
    f_lp = torch.full((R, F), float("-inf"), device=dev)
    f_padre = torch.full((R, F), -1, dtype=torch.int64, device=dev)  # nodo padre (-1 = ancla)
    f_prof = torch.zeros((R, F), dtype=torch.int64, device=dev)      # profundidad 0-based
    f_idx = torch.zeros((R, F), dtype=torch.int64, device=dev)       # indice de candidato
    ar = torch.arange(K, device=dev)
    f_lp[:, :K] = lp[:, 0, 0, :]
    f_idx[:, :K] = ar
    token = torch.zeros((R, n_nodos), dtype=torch.int64, device=dev)
    padre = torch.full((R, n_nodos), -1, dtype=torch.int64, device=dev)
    prof = torch.zeros((R, n_nodos), dtype=torch.int64, device=dev)
    idx = torch.zeros((R, n_nodos), dtype=torch.int64, device=dev)
    filas = torch.arange(R, device=dev)
    for n in range(n_nodos):
        j = f_lp.argmax(dim=1)                                       # [R]
        p, d, i = f_lp[filas, j], f_prof[filas, j], f_idx[filas, j]
        token[:, n] = cand[filas, d, i]
        padre[:, n] = f_padre[filas, j]
        prof[:, n] = d
        idx[:, n] = i
        # sale de la frontera (scatter_ y no f_lp[filas, j] = -inf: la asignacion indexada con un
        # escalar de Python arma un tensor en CPU y lo copia, y eso no entra en un grafo CUDA)
        f_lp.scatter_(1, j[:, None], float("-inf"))
        # hijos: solo si no es el ultimo nivel
        d1 = (d + 1).clamp(max=S - 1)
        hijos = p[:, None] + lp[filas, d1, i, :]                     # [R, K]
        hijos = torch.where((d + 1 < S)[:, None], hijos, torch.full_like(hijos, float("-inf")))
        base = K * (n + 1)
        f_lp[:, base: base + K] = hijos
        f_padre[:, base: base + K] = n
        f_prof[:, base: base + K] = (d + 1)[:, None]
        f_idx[:, base: base + K] = ar
    return token, padre, prof, idx


def ancestros(padre: torch.Tensor) -> torch.Tensor:
    """Mascara de ancestros [R, N, N] bool: m[r, a, b] = b es ancestro de a, o a == b."""
    padre = padre.long()
    R, N = padre.shape
    m = torch.eye(N, dtype=torch.bool, device=padre.device)[None].repeat(R, 1, 1)
    cur = padre.clone()
    for _ in range(N):
        ok = cur >= 0
        m |= torch.nn.functional.one_hot(cur.clamp(min=0), N).bool() & ok[..., None]
        cur = torch.where(ok, padre.gather(1, cur.clamp(min=0)), cur)
    return m


def orden_dfs(token, padre, prof, idx):
    """Reordena los nodos en preorden DFS (sigue siendo topologico), todo en GPU y forma fija.

    Para que sirve: el GDN en arbol (``gdn_cinta``, ARBOL) sigue con el estado corriente
    cuando el padre de un nodo es el nodo anterior, y solo recalcula desde la raiz en los
    saltos de rama. En preorden cada rama es una corrida contigua, asi que los saltos son
    los minimos. La atencion no depende del orden (usa la mascara de ancestros).

    La clave de orden es el camino raiz->nodo escrito en base n+1 (el prefijo de un padre,
    rellenado con ceros, es menor que el de sus hijos): 17^8 entra holgado en int64.
    """
    R, n = padre.shape
    D = int(min(n, 8))
    ar = torch.arange(n, device=padre.device)[None].expand(R, n)
    # camino[d] = ancestro de profundidad d (o -1): se sube D veces desde el nodo
    # (la columna D+1 es un descarte: ahi escriben los que ya pasaron del ancla)
    cam = torch.full((R, n, D + 2), -1, dtype=torch.long, device=padre.device)
    cur = ar.clone()
    for _ in range(D + 1):
        ok = cur >= 0
        pr = prof.gather(1, cur.clamp(min=0)).long().clamp(max=D)
        pr = torch.where(ok, pr, pr.new_full((), D + 1))
        cam.scatter_(2, pr[..., None], cur[..., None])
        cur = torch.where(ok, padre.gather(1, cur.clamp(min=0)).long(), cur)
    cam = cam[..., : D + 1]
    pot = (n + 1) ** torch.arange(D, -1, -1, device=padre.device, dtype=torch.long)
    clave = ((cam + 1) * pot).sum(-1)
    perm = clave.argsort(dim=1, stable=True)                  # posicion nueva -> nodo viejo
    inv = torch.empty_like(perm).scatter_(1, perm, ar)        # nodo viejo -> posicion nueva
    p_v = padre.gather(1, perm).long()
    p_n = torch.where(p_v >= 0, inv.gather(1, p_v.clamp(min=0)), p_v).to(padre.dtype)
    return token.gather(1, perm), p_n, prof.gather(1, perm), idx.gather(1, perm), perm


def padres_verificacion(padre: torch.Tensor) -> torch.Tensor:
    """Padres en la secuencia que se VERIFICA: [ancla] + nodos. El ancla es el token 0 (padre
    -1) y cada nodo pasa a la posicion i+1, con padre ``padre+1`` (los hijos del ancla, 0)."""
    raiz = padre.new_full((padre.shape[0], 1), -1)
    return torch.cat([raiz, padre + 1], dim=1)


def bits_ancestros(padre: torch.Tensor) -> torch.Tensor:
    """``padre`` = padres de la secuencia verificada (``padres_verificacion``).
    Mascara por token en el formato de los kernels (PN131 y GDN): int32 [R, N], con el
    bit j-1 = "el nodo j es ancestro o el mismo", para j >= 1. La raiz (nodo 0) es ancestro
    de todos y no lleva bit. La cadena da (1 << t) - 1."""
    m = ancestros(padre)[:, :, 1:]
    w = (1 << torch.arange(m.shape[-1], device=padre.device, dtype=torch.int64))
    return (m.long() * w).sum(-1).to(torch.int32)


_k_arbol = None


def construir_dfs_kernel(cand, sc, out_tok, out_padre_v, out_prof_v, n_nodos: int,
                         out_bits=None, rheo=None) -> None:
    """``construir`` + ``orden_dfs`` + ``padres_verificacion`` en UN kernel, un programa por pedido.

    Por que: en torch son ~200 kernels chicos por paso adentro del grafo del borrador, y el arbol
    solo paga si el paso no se encarece. Mismo resultado que el camino de torch (mismo desempate:
    el primer maximo; claves de preorden unicas).

    ``cand`` [R, S, K] int64, ``sc`` [R, S, K, K] float contiguos. Escribe ``out_tok[r, :n]`` (tokens
    en preorden), ``out_padre_v[r, 1:n+1]`` (padre en la secuencia verificada; 0 = el ancla) y
    ``out_prof_v[r, 1:n+1]`` (profundidad, el ancla es 0). La columna 0 no se toca.
    ``out_bits`` [R, n+1] int32 (opcional): los bits de ancestros de ``bits_ancestros``, ya listos
    para los kernels, asi el runner no los recalcula en cada paso.

    ``rheo`` (opcional) — RheoSampling (arXiv 2609.21827), para que el arbol sirva con temperatura.
    Un arbol top-K determinista propone con q one-hot: con T > 0 solo acepta p(candidato), menos
    que una cadena muestreada, que acepta el solapamiento de p y q (medido: a T=0,6 el arbol
    empataba con la cadena). Con ``rheo``, en cada grupo de hermanos de un pedido con T > 0:
      * el mejor candidato queda fijo (m = 1) y entra con su q real;
      * UN candidato Y se muestrea de la cola ``q~ = q / z`` (z = 1 - q(mejor)), y para ordenar la
        frontera entra con la probabilidad PROXY ``min(q(mejor), z)``: asi que sobreviva a la poda
        no depende de QUE token salio, que es lo que mantiene exacto al metodo;
      * el resto entra con su q real. q = softmax(puntajes / T).
    Tupla ``(temp [R] f32, semilla [R] i64, pos [R] i64, out_qres [R, n+1, K] f32, out_cand [R, n+1, K]
    i64, out_hijo_s [R, n+1] i32, out_qs [R, n+1] f32)``. Las salidas van por posicion verificada del
    PADRE: la cola ``q~`` de su grupo de hijos y sus tokens, la posicion de su hijo muestreado (-1
    si la poda lo saco, o si T = 0) y la ``q~`` de ese hijo.
    """
    global _k_arbol
    from vllm.triton_utils import tl, triton
    if _k_arbol is None:
        @triton.jit
        def _grupo(sc, fb, temp, seed, off, ak, mkk, RHEO: tl.constexpr):
            """Fila de puntajes de un grupo -> (log q por candidato, indice muestreado o -1,
            log de la proxy, indice del mejor, z)."""
            NEG = float("-inf")
            fila = tl.load(sc + fb + ak, mask=mkk, other=NEG).to(tl.float32)
            ks = tl.full((), -1, tl.int64)
            lproxy = tl.full((), 0.0, tl.float32)
            k1 = tl.argmax(fila, 0).to(tl.int64)
            z = tl.full((), 0.0, tl.float32)
            if RHEO:
                if temp > 0.0:
                    fila = fila / temp
            mx = tl.max(fila, 0)
            lse = mx + tl.log(tl.sum(tl.where(mkk, tl.exp(fila - mx), 0.0), 0))
            lq = fila - lse
            if RHEO:
                if temp > 0.0:
                    q1 = tl.exp(tl.max(lq, 0))
                    z = 1.0 - q1
                    u = tl.rand(seed, off + ak)
                    u = tl.maximum(u, 1e-20)
                    g = -tl.log(-tl.log(u) + 1e-20)
                    cola = tl.where(mkk & (ak != k1), lq + g, NEG)
                    ks = tl.argmax(cola, 0).to(tl.int64)
                    lproxy = tl.log(tl.maximum(tl.minimum(q1, z), 1e-30))
            return lq, ks, lproxy, k1, z

        @triton.jit
        def _k(cand, sc, out_tok, s_tok, out_pad, s_pad, out_prof, s_prof, out_bits, s_bits,
               temp_p, seed_p, pos_p, out_qres, out_cand, out_hs, out_qs,
               BITS: tl.constexpr, RHEO: tl.constexpr,
               S: tl.constexpr, K: tl.constexpr, N: tl.constexpr, BF: tl.constexpr,
               BN: tl.constexpr, BKK: tl.constexpr):
            r = tl.program_id(0)
            af = tl.arange(0, BF)
            an = tl.arange(0, BN)
            ak = tl.arange(0, BKK)
            mkk = ak < K
            NEG = float("-inf")
            temp = tl.full((), 0.0, tl.float32)
            seed = tl.full((), 0, tl.int64)
            off0 = tl.full((), 0, tl.int64)
            if RHEO:
                temp = tl.load(temp_p + r).to(tl.float32)
                seed = tl.load(seed_p + r).to(tl.int64)
                off0 = (tl.load(pos_p + r).to(tl.int64) % 1048576) * (N + 1) * BKK
            # frontera: los K hijos del ancla
            lq, ks0, lpx, _k1, _z = _grupo(sc, (r * S * K) * K, temp, seed, off0, ak, mkk, RHEO)
            idx0 = tl.where(af < K, af, 0)
            v0 = tl.sum(tl.where(ak[None, :] == idx0[:, None], lq[None, :], 0.0), 1)
            v0 = tl.where(idx0 == ks0, lpx, v0)
            f_lp = tl.where(af < K, v0, NEG)
            f_mu = ((af < K) & (af == ks0)).to(tl.int64)
            f_pad = tl.full((BF,), -1, tl.int64)
            f_prof = tl.zeros((BF,), tl.int64)
            f_idx = tl.where(af < K, af, 0).to(tl.int64)
            n_tok = tl.zeros((BN,), tl.int64)
            n_pad = tl.full((BN,), -1, tl.int64)
            n_prof = tl.zeros((BN,), tl.int64)
            n_idx = tl.zeros((BN,), tl.int64)
            n_clave = tl.zeros((BN,), tl.int64)
            n_hs = tl.full((BN,), -1, tl.int64)          # nodo -> su hijo muestreado (indice de nodo)
            hs0 = tl.full((), -1, tl.int64)               # idem para el ancla
            for n in range(0, N):
                j = tl.argmax(f_lp, 0)
                sel = af == j
                p = tl.sum(tl.where(sel, f_lp, 0.0), 0)
                d = tl.sum(tl.where(sel, f_prof, 0), 0)
                i = tl.sum(tl.where(sel, f_idx, 0), 0)
                pa = tl.sum(tl.where(sel, f_pad, 0), 0)
                mu = tl.sum(tl.where(sel, f_mu, 0), 0)
                tok = tl.load(cand + (r * S + d) * K + i)
                # clave de preorden: el camino en base N+1, el digito de la profundidad d
                pot = tl.full((), 1, tl.int64)
                for _ in range(0, S - 1 - d):
                    pot = pot * (N + 1)
                cpad = tl.sum(tl.where(an == pa, n_clave, 0), 0)      # pa = -1 no coincide: 0
                clave = cpad + (n + 1) * pot
                aqui = an == n
                n_tok = tl.where(aqui, tok, n_tok)
                n_pad = tl.where(aqui, pa, n_pad)
                n_prof = tl.where(aqui, d, n_prof)
                n_idx = tl.where(aqui, i, n_idx)
                n_clave = tl.where(aqui, clave, n_clave)
                if mu != 0:
                    if pa < 0:
                        hs0 = hs0 * 0 + n
                    else:
                        n_hs = tl.where(an == pa, n, n_hs)
                f_lp = tl.where(sel, NEG, f_lp)
                # los K hijos del elegido entran en [K*(n+1), K*(n+2))
                base = K * (n + 1)
                kid = af - base
                mh = (kid >= 0) & (kid < K)
                if d + 1 < S:
                    fb = ((r * S + d + 1) * K + i) * K
                    lq, ks, lpx, _k1, _z = _grupo(sc, fb, temp, seed, off0 + (n + 1) * BKK, ak, mkk, RHEO)
                    kk = tl.where(mh, kid, 0)
                    h = tl.sum(tl.where(ak[None, :] == kk[:, None], lq[None, :], 0.0), 1)
                    h = tl.where(kk == ks, lpx, h)
                    f_lp = tl.where(mh, p + h, f_lp)
                    f_mu = tl.where(mh, (kk == ks).to(tl.int64), f_mu)
                f_pad = tl.where(mh, n, f_pad)
                f_prof = tl.where(mh, d + 1, f_prof)
                f_idx = tl.where(mh, kid, f_idx).to(tl.int64)
            # preorden: la posicion de un nodo es cuantos tienen clave menor
            val = an < N
            menor = (n_clave[None, :] < n_clave[:, None]) & val[None, :]
            rango = tl.sum(menor.to(tl.int64), 1)                     # [BN]: nodo -> posicion
            rpad = tl.sum(tl.where(an[None, :] == n_pad[:, None], rango[None, :] + 1, 0), 1)  # ancla -> 0
            tl.store(out_tok + r * s_tok + rango, n_tok.to(out_tok.dtype.element_ty), mask=val)
            tl.store(out_pad + r * s_pad + 1 + rango, rpad.to(out_pad.dtype.element_ty), mask=val)
            tl.store(out_prof + r * s_prof + 1 + rango, (n_prof + 1).to(out_prof.dtype.element_ty), mask=val)
            if BITS:
                # bits[nodo] = bits[padre] | (1 << posicion): el padre siempre se eligio antes
                n_bits = tl.zeros((BN,), tl.int64)
                for n in range(0, N):
                    pa = tl.sum(tl.where(an == n, n_pad, 0), 0)
                    rg = tl.sum(tl.where(an == n, rango, 0), 0)
                    bp = tl.sum(tl.where(an == pa, n_bits, 0), 0)
                    one = tl.full((), 1, tl.int64)
                    n_bits = tl.where(an == n, bp | (one << rg), n_bits)
                tl.store(out_bits + r * s_bits + 1 + rango, n_bits.to(out_bits.dtype.element_ty), mask=val)
            if RHEO:
                # por PADRE (fila 0 = el ancla, fila rango+1 = el nodo): la cola q~ de su grupo de
                # hijos, los tokens del grupo, su hijo muestreado y la q~ de ese hijo
                T1 = N + 1
                for n in range(-1, N):
                    d = tl.full((), -1, tl.int64)
                    i = tl.full((), 0, tl.int64)
                    fila_o = tl.full((), 0, tl.int64)
                    hsn = hs0
                    if n >= 0:
                        d = tl.sum(tl.where(an == n, n_prof, 0), 0)
                        i = tl.sum(tl.where(an == n, n_idx, 0), 0)
                        fila_o = tl.sum(tl.where(an == n, rango, 0), 0) + 1
                        hsn = tl.sum(tl.where(an == n, n_hs, 0), 0)
                    qres = tl.zeros((BKK,), tl.float32)
                    qs = tl.full((), 0.0, tl.float32)
                    hpos = tl.full((), -1, tl.int64)
                    dd = tl.minimum(d + 1, S - 1)
                    if (d + 1 < S) & (temp > 0.0):
                        fb = ((r * S + d + 1) * K + i) * K
                        lq, ks, lpx, k1, z = _grupo(sc, fb, temp, seed, off0 + (n + 1) * BKK, ak, mkk, RHEO)
                        qres = tl.where(mkk & (ak != k1), tl.exp(lq) / tl.maximum(z, 1e-30), 0.0)
                        if hsn >= 0:
                            hpos = tl.sum(tl.where(an == hsn, rango, 0), 0) + 1
                            qs = tl.sum(tl.where(ak == ks, qres, 0.0), 0)
                    tl.store(out_qres + ((r * T1 + fila_o) * K) + ak, qres, mask=mkk)
                    tl.store(out_cand + ((r * T1 + fila_o) * K) + ak,
                             tl.load(cand + (r * S + dd) * K + ak, mask=mkk, other=0), mask=mkk)
                    tl.store(out_hs + r * T1 + fila_o, hpos.to(tl.int32))
                    tl.store(out_qs + r * T1 + fila_o, qs)
        _k_arbol = _k
    R, S, K = cand.shape
    assert cand.is_contiguous() and sc.is_contiguous()
    assert (n_nodos + 1) ** S < 2 ** 62, "la clave de preorden no entra en int64"
    nada = out_prof_v
    if rheo is not None:
        temp, seed, pos, o_qres, o_cand, o_hs, o_qs = rheo
        for x in (o_qres, o_cand, o_hs, o_qs):
            assert x.is_contiguous() and x.shape[1] == n_nodos + 1
    else:
        temp = seed = pos = o_qres = o_cand = o_hs = o_qs = nada
    _k_arbol[(R,)](cand, sc, out_tok, out_tok.stride(0), out_padre_v, out_padre_v.stride(0),
                   out_prof_v, out_prof_v.stride(0),
                   out_bits if out_bits is not None else nada,
                   out_bits.stride(0) if out_bits is not None else 0,
                   temp, seed, pos, o_qres, o_cand, o_hs, o_qs,
                   BITS=out_bits is not None, RHEO=rheo is not None,
                   S=S, K=K, N=n_nodos,
                   BF=triton.next_power_of_2(K * (n_nodos + 1)), BN=triton.next_power_of_2(n_nodos),
                   BKK=triton.next_power_of_2(K), num_warps=1)


def aceptar(token_v: torch.Tensor, padre_v: torch.Tensor, muestra: torch.Tensor):
    """Aceptacion por recorrido del arbol, en GPU, forma fija y sin sincronizar.

    Todo en indices de la secuencia VERIFICADA ([ancla] + nodos): ``token_v`` [R, T] (el del
    ancla no se mira), ``padre_v`` [R, T] (``padres_verificacion``), ``muestra`` [R, T] = el
    token que el target saca DESPUES de cada posicion: el argmax con decodificacion golosa, una
    muestra de su distribucion con temperatura.

    Por que es exacto tambien con temperatura: el borrador propone tokens, no distribuciones
    (q es one-hot). Muestrear y ~ p en cada nodo y bajar al hijo cuyo token coincide con y es
    el rejection sampling multi-borrador de SpecInfer en ese caso limite: se acepta algun hijo
    con probabilidad sum p(hijos), lo mismo que rechazar de a uno y renormalizar, y si no
    coincide ninguno, y YA es una muestra del residuo. Cada token emitido es una muestra
    exacta del target; el arbol solo decide cuantas se aprovechan. Las T muestras salen del
    mismo forward, en paralelo: no hay lazo sobre datos.

    Devuelve ``(camino [R, T-1], nacc [R], bono [R])``: posiciones verificadas aceptadas en
    orden (relleno -1), aceptados contando el ancla (= largo + 1, lo que vLLM llama
    ``num_accepted_tokens``) y el token que sigue al ultimo aceptado.
    """
    R, T = token_v.shape
    dev = token_v.device
    act = torch.zeros(R, dtype=torch.long, device=dev)               # posicion actual: el ancla
    vivo = torch.ones(R, dtype=torch.bool, device=dev)
    camino = torch.full((R, T - 1), -1, dtype=torch.long, device=dev)
    nacc = torch.ones(R, dtype=torch.long, device=dev)
    pv = padre_v.long()
    for d in range(T - 1):
        quiero = muestra.gather(1, act[:, None])                     # [R, 1]
        ok = (pv == act[:, None]) & (token_v == quiero)              # [R, T]
        ok[:, 0] = False
        hay = ok.any(dim=1) & vivo
        sig = ok.float().argmax(dim=1)
        camino[:, d] = torch.where(hay, sig, camino[:, d])
        act = torch.where(hay, sig, act)
        nacc += hay.long()
        vivo = hay
    return camino, nacc, muestra.gather(1, act[:, None])[:, 0]


_k_prep = None


def preparar_paso_kernel(idx_mapping, padre_v, bits_v, delta_v, anc131, anc_gdn, anc3,
                         positions, num_reqs: int, T: int):
    """UN kernel para todo lo que el paso en arbol necesita antes del forward del target.

    Antes eran ~8 lanzamientos de torch por paso (indexar por slot, aplanar, dos copias, sumar el
    delta a las posiciones). Ademas precalcula ``anc3`` [filas, 3]: los TRES ancestros mas cercanos
    de cada token, que es lo unico que mira la conv causal (ancho 4). Sin esto, el kernel de la
    conv los busca con un lazo O(T^2) sobre los bits, y eso corre 48 veces por paso, una por capa.

    ``anc3[fila, i]``: indice LOCAL del ancestro dentro del bloque (0 = el ancla), o negativo
    ``-1, -2, -3`` para las tres columnas de historia del estado conv (de la mas nueva a la mas
    vieja). El token 0 da ``[-1, -2, -3]`` y la cadena da ``[t-1, t-2, t-3]`` con el mismo
    convenio, asi que el kernel de la conv queda sin ramas.
    """
    global _k_prep
    from vllm.triton_utils import tl, triton
    if _k_prep is None:
        @triton.jit
        def _k(idx_map, pv, s_pv, bits_v, s_bv, delta_v, s_dv, anc131, anc_gdn, anc3, s_a3,
               pos_p, s_pos, ND: tl.constexpr, T: tl.constexpr, BT: tl.constexpr):
            r = tl.program_id(0)
            ri = tl.load(idx_map + r).to(tl.int64)
            at = tl.arange(0, BT)
            m = at < T
            base = r * T
            bits = tl.load(bits_v + ri * s_bv + at, mask=m, other=0)
            tl.store(anc131 + base + at, bits, mask=m)
            tl.store(anc_gdn + base + at, bits, mask=m)
            dl = tl.load(delta_v + ri * s_dv + at, mask=m, other=0).to(tl.int64)
            for d in range(0, ND):
                p = tl.load(pos_p + d * s_pos + base + at, mask=m, other=0)
                tl.store(pos_p + d * s_pos + base + at, p + dl.to(p.dtype), mask=m)
            # tres ancestros mas cercanos: se sube por los padres, con -1/-2/-3 al pasarse
            pa = tl.load(pv + ri * s_pv + at, mask=m, other=-1).to(tl.int64)
            a1 = tl.where(at > 0, pa, -1)
            a2 = tl.where(a1 > 0, tl.sum(tl.where(at[None, :] == a1[:, None],
                                                  pa[None, :], 0), 1), a1 - 1)
            a3 = tl.where(a2 > 0, tl.sum(tl.where(at[None, :] == a2[:, None],
                                                  pa[None, :], 0), 1), a2 - 1)
            tl.store(anc3 + (base + at) * s_a3 + 0, a1.to(tl.int32), mask=m)
            tl.store(anc3 + (base + at) * s_a3 + 1, a2.to(tl.int32), mask=m)
            tl.store(anc3 + (base + at) * s_a3 + 2, a3.to(tl.int32), mask=m)
        _k_prep = _k
    _k_prep[(num_reqs,)](idx_mapping, padre_v, padre_v.stride(0), bits_v, bits_v.stride(0),
                         delta_v, delta_v.stride(0), anc131, anc_gdn, anc3, anc3.stride(0),
                         positions, positions.stride(0), ND=positions.shape[0], T=T,
                         BT=triton.next_power_of_2(T), num_warps=1)


_k_rep = None


def reponer_cadena_kernel(anc131, anc_gdn, anc3, num_reqs: int, T: int):
    """Vuelve a dejar los tres buffers de mascara en CADENA, en UN kernel. Hace falta porque el
    borrador comparte los buffers de PN131 con el target y el no verifica ningun arbol: si
    quedara la mascara del arbol, sus queries verian un contexto que no les corresponde."""
    global _k_rep
    from vllm.triton_utils import tl, triton
    if _k_rep is None:
        @triton.jit
        def _k(anc131, anc_gdn, anc3, s_a3, T: tl.constexpr, BT: tl.constexpr):
            r = tl.program_id(0)
            at = tl.arange(0, BT)
            m = at < T
            base = r * T
            one = tl.full((BT,), 1, tl.int32)
            bits = (one << at.to(tl.int32)) - 1                 # (1 << t) - 1: la causalidad
            tl.store(anc131 + base + at, bits, mask=m)
            tl.store(anc_gdn + base + at, bits, mask=m)
            tl.store(anc3 + (base + at) * s_a3 + 0, (at - 1).to(tl.int32), mask=m)
            tl.store(anc3 + (base + at) * s_a3 + 1, (at - 2).to(tl.int32), mask=m)
            tl.store(anc3 + (base + at) * s_a3 + 2, (at - 3).to(tl.int32), mask=m)
        _k_rep = _k
    _k_rep[(num_reqs,)](anc131, anc_gdn, anc3, anc3.stride(0), T=T,
                        BT=triton.next_power_of_2(T), num_warps=1)


_k_comp = None
_ptrs_cache: dict = {}


def compactar_kernel(idx_mapping, camino, nacc, filas_cinta, camino_cinta, ocultos, kv_addrs,
                     slot_map, num_reqs: int, T: int, geom):
    """UN kernel para toda la compactacion posterior a aceptar: estados ocultos (el borrador los
    lee), KV de atencion y las filas de cinta del camino. Antes eran ~10 lanzamientos de torch
    (arange, where, gather, index_put) mas dos kernels.

    Los indices salen del camino DENTRO del kernel: el nodo aceptado ``camino[r, j]`` se movio al
    lugar ``j + 1``. ``src >= dst`` y ambos crecen con j, asi que copiar en orden nunca lee algo
    ya pisado. ``geom`` = (NH, BS, BLK) del KV de PN131, cuyo bloque es
    ``K [BS][NH][256] | V [NH][256][BS] | escalas [BS][NH][2]``: V esta traspuesta, o sea que un
    token NO es contiguo y hay que moverlo por partes.
    """
    global _k_comp
    import torch as _t
    from vllm.triton_utils import tl, triton
    NH, BS, BLK = geom
    n_oc = len(ocultos)
    if _k_comp is None:
        @triton.jit
        def _k(idx_map, cam, s_cam, nacc, filas, s_fil, cam_cinta, s_cc, oc_ptrs, oc_strides,
               kv_addrs, slot_map, N_OC, T: tl.constexpr, TM: tl.constexpr, D: tl.constexpr,
               NH: tl.constexpr, BS: tl.constexpr, BLK: tl.constexpr, QD: tl.constexpr,
               BD: tl.constexpr, BK: tl.constexpr, BE: tl.constexpr, FP16: tl.constexpr):
            r, lay = tl.program_id(0), tl.program_id(1)
            nj = tl.load(nacc + r).to(tl.int64) - 1
            if lay == 0:      # las filas de cinta del camino, una sola vez por pedido
                ri = tl.load(idx_map + r).to(tl.int64)
                af = tl.arange(0, TM)
                fl = tl.load(filas + r * s_fil + af)
                tl.store(cam_cinta + (ri + 1) * s_cc + af, fl)
            if nj <= 0:
                return
            if lay < N_OC:                                   # estados ocultos: fila entera
                base = tl.load(oc_ptrs + lay).to(tl.pointer_type(tl.float16))
                st = tl.load(oc_strides + lay).to(tl.int64)
                for j in range(0, nj):
                    s = tl.load(cam + r * s_cam + j).to(tl.int64)
                    d = j + 1
                    if s != d:
                        ps = base + (r * T + s) * st
                        pd = base + (r * T + d) * st
                        for c in range(0, D, BD):
                            o = c + tl.arange(0, BD)
                            mo = o < D
                            tl.store(pd + o, tl.load(ps + o, mask=mo), mask=mo)
            else:                                            # KV de atencion, por capa
                raw = tl.load(kv_addrs + (lay - N_OC)).to(tl.pointer_type(tl.int8))
                KOFF = BS * NH * QD
                ok = tl.arange(0, BK)
                mk = ok < NH * QD
                oe = tl.arange(0, BE)
                me = oe < NH * 4
                for j in range(0, nj):
                    sj = tl.load(cam + r * s_cam + j).to(tl.int64)
                    s = tl.load(slot_map + r * T + sj).to(tl.int64)
                    d = tl.load(slot_map + r * T + j + 1).to(tl.int64)
                    if (s != d) & (s >= 0) & (d >= 0):
                        bs_, os_ = raw + (s // BS) * BLK, s % BS
                        bd_, od_ = raw + (d // BS) * BLK, d % BS
                        tl.store(bd_ + od_ * NH * QD + ok,
                                 tl.load(bs_ + os_ * NH * QD + ok, mask=mk), mask=mk)
                        tl.store(bd_ + KOFF + ok * BS + od_,
                                 tl.load(bs_ + KOFF + ok * BS + os_, mask=mk), mask=mk)
                        tl.store(bd_ + 2 * KOFF + od_ * NH * 4 + oe,
                                 tl.load(bs_ + 2 * KOFF + os_ * NH * 4 + oe, mask=me), mask=me)
        _k_comp = _k
    dev = camino.device
    # Las direcciones y strides de los buffers de estados ocultos, en GPU. ANTES:
    #     _t.tensor([...], device=dev)
    # que copia desde memoria no pinned y torch lo hace SINCRONICO: cudaStreamSynchronize, o sea que la CPU
    # esperaba a que la GPU terminara todo lo encolado (el forward del target, ~20 ms), dos veces por paso.
    # Con eso el scheduling asincrono no servia y el trabajo eager posterior se lanzaba con la GPU parada
    # (perfil con pila de Python, 27-09: 19,8 ms por paso de CPU bloqueada). Ahora: una vez por juego de
    # direcciones (los buffers son persistentes), desde memoria pinned y sin bloquear.
    clave = tuple((h.data_ptr(), h.stride(0)) for h in ocultos)
    ent = _ptrs_cache.get(clave)
    if ent is None:
        host = _t.tensor([c[0] for c in clave] + [c[1] for c in clave], dtype=_t.int64).pin_memory()
        ent = _ptrs_cache[clave] = (host.to(dev, non_blocking=True), host)   # el host queda vivo: la copia es async
        if len(_ptrs_cache) == 33:
            import logging
            logging.getLogger("genesis.arbol").warning(
                "[ARBOL] los buffers de estados ocultos cambian de direccion: %d juegos cacheados", len(_ptrs_cache))
    ptrs, strs = ent[0][:n_oc], ent[0][n_oc:]
    D = ocultos[0].shape[1]
    _k_comp[(num_reqs, n_oc + kv_addrs.shape[0])](
        idx_mapping, camino, camino.stride(0), nacc, filas_cinta, filas_cinta.stride(0),
        camino_cinta, camino_cinta.stride(0), ptrs, strs, kv_addrs, slot_map, n_oc,
        T=T, TM=camino_cinta.shape[1], D=D, NH=NH, BS=BS, BLK=BLK, QD=256,
        BD=1024, BK=triton.next_power_of_2(NH * 256), BE=triton.next_power_of_2(NH * 4),
        FP16=True, num_warps=4)


_k_acep = None


def preparar_rheo(logits, tok_f, cu, fila, hs_f, qs_f, cand_f, qres_f, muestrear):
    """Lo que RheoVerification necesita por fila de logits (= por nodo, como PADRE), en torch y sin
    sincronizar. ``logits`` [n, V] ya procesados (temperatura, top-p...), ``fila`` [n] el pedido
    de cada fila, ``hs_f``/``qs_f`` [n] la posicion del hijo muestreado (-1 si no hay) y su ``q~``,
    ``cand_f``/``qres_f`` [n, K] los tokens del grupo de hijos y su cola ``q~``.

    Devuelve ``(razon [n], y_r [n])``: ``p(hijo muestreado) / q~`` (se acepta con probabilidad
    ``min(1, razon)``) y una muestra del RESIDUO ``norm(max(0, p - q~))``, que es de donde sigue la
    verificacion si ese hijo se rechaza. ``q~`` vive en K tokens, asi que el residuo es ``p`` con
    K entradas retocadas: se escriben, se muestrea con ``muestrear(logits)`` y se reponen (no se
    clona una matriz de n x vocabulario).
    """
    n, V = logits.shape
    hay = hs_f >= 0
    lse = torch.logsumexp(logits.float(), dim=-1)
    hijo = (cu[:-1].long()[fila] + hs_f.clamp(min=0).long()).clamp(max=n - 1)
    xs = tok_f[hijo].long().clamp(0, V - 1)
    lp_s = logits.gather(1, xs[:, None])[:, 0].float() - lse
    razon = torch.where(hay, torch.exp(lp_s) / qs_f.clamp(min=1e-30), torch.zeros_like(lp_s))
    c = cand_f.clamp(0, V - 1)
    viejo = logits.gather(1, c)
    val = torch.exp(viejo.float() - lse[:, None]) - qres_f
    nuevo = torch.where(val > 0, torch.log(val.clamp(min=1e-38)) + lse[:, None],
                        torch.full_like(val, float("-inf"))).to(logits.dtype)
    logits.scatter_(1, c, torch.where(hay[:, None], nuevo, viejo))
    y_r = muestrear(logits)
    logits.scatter_(1, c, viejo)
    return razon, y_r


def aceptar_kernel(tok_f, muestra_f, cu, padre_v, alcanzable, T: int, tm: int, rheo=None,
                   idx_map=None):
    """``aceptar`` + armado de la salida en UN kernel, sobre las filas PLANAS del rejection sampler
    (sin densificar). ``tok_f``/``muestra_f`` [n] (token de entrada y muestra del target por fila de
    logits), ``cu`` [R+1], ``padre_v`` [R, T] y ``alcanzable`` [R, T] bool (nodos que se pueden
    aceptar). Devuelve ``(sampled [R, T] int64, nacc [R] int32, camino [R, T-1] int64 con -1,
    filas [R, tm] int32 para la cinta)``.

    ``idx_map`` [R]: si viene, ``padre_v`` y ``alcanzable`` estan indexados por SLOT de req-state y
    el kernel los indexa adentro (asi el runner no arma dos gather por paso).

    ``rheo = (hijo_s [R, T] i32, razon [n] f32, y_r [n], semilla [R] i64, pos_f [n] i64)`` activa
    RheoVerification: en un nodo con hijo muestreado alcanzable, primero ese hijo se acepta con
    probabilidad ``min(1, razon)``; si se rechaza se sigue con ``y_r`` (muestra del residuo) en
    vez de ``muestra_f``; despues, como siempre, se baja al hijo determinista que coincida."""
    global _k_acep
    from vllm.triton_utils import tl, triton
    if _k_acep is None:
        @triton.jit
        def _k(tok_f, mu_f, cu, pv, s_pv, alc, s_alc, sampled, nacc, camino, filas,
               hs_p, s_hs, razon_f, yr_f, seed_p, pos_f, idx_map, RHEO: tl.constexpr,
               IDX: tl.constexpr, TODOS: tl.constexpr,
               T: tl.constexpr, TM: tl.constexpr, BT: tl.constexpr):
            r = tl.program_id(0)
            ri = r
            if IDX:
                ri = tl.load(idx_map + r).to(tl.int64)
            at = tl.arange(0, BT)
            ini = tl.load(cu + r).to(tl.int64)
            nf = tl.load(cu + r + 1).to(tl.int64) - ini
            mt = (at < nf) & (at < T)
            tk = tl.load(tok_f + ini + at, mask=mt, other=-1).to(tl.int64)
            mu = tl.load(mu_f + ini + at, mask=mt, other=-2).to(tl.int64)
            pa = tl.load(pv + ri * s_pv + at, mask=at < T, other=-2).to(tl.int64)
            ok0 = mt & (at > 0)
            if not TODOS:
                ok0 = ok0 & (tl.load(alc + ri * s_alc + at, mask=at < T, other=0) != 0)
            hs = tl.full((BT,), -1, tl.int64)
            rz = tl.zeros((BT,), tl.float32)
            yr = mu
            un = tl.zeros((BT,), tl.float32)
            if RHEO:
                hs = tl.load(hs_p + ri * s_hs + at, mask=at < T, other=-1).to(tl.int64)
                rz = tl.load(razon_f + ini + at, mask=mt, other=0.0).to(tl.float32)
                yr = tl.load(yr_f + ini + at, mask=mt, other=-2).to(tl.int64)
                sd = tl.load(seed_p + ri).to(tl.int64) ^ 0x5BD1E995
                ps = tl.load(pos_f + ini + at, mask=mt, other=0).to(tl.int64)
                un = tl.rand(sd, ps * 2 + 1)
            act = tl.zeros((), tl.int64)
            vivo = tl.full((), 1, tl.int64)
            n = tl.zeros((), tl.int64)
            for d in range(0, T - 1):
                en = at == act
                h = tl.sum(tl.where(en, hs, 0), 0)
                h_ok = (h > 0) & (tl.sum(tl.where(at == h, ok0.to(tl.int64), 0), 0) > 0)
                a_s = h_ok & (tl.sum(tl.where(en, un, 0.0), 0) < tl.sum(tl.where(en, rz, 0.0), 0))
                quiero = tl.sum(tl.where(en, tl.where(h_ok, yr, mu), 0), 0)
                cand = ok0 & (pa == act) & (tk == quiero) & (at != h)
                hay = (a_s | (tl.sum(cand.to(tl.int64), 0) > 0)) & (vivo != 0)
                sig = tl.where(a_s, h, tl.argmax(cand.to(tl.int64), 0).to(tl.int64))
                if hay:
                    tl.store(sampled + r * T + d, tl.sum(tl.where(at == sig, tk, 0), 0))
                    tl.store(camino + r * (T - 1) + d, sig)
                    if d < TM:
                        tl.store(filas + r * TM + d, (sig - 1).to(tl.int32))
                    act = sig
                    n += 1
                else:
                    vivo = vivo * 0
            # el bono: la muestra que corresponde al nodo donde se corto
            en = at == act
            h = tl.sum(tl.where(en, hs, 0), 0)
            h_ok = (h > 0) & (tl.sum(tl.where(at == h, ok0.to(tl.int64), 0), 0) > 0)
            tl.store(sampled + r * T + n, tl.sum(tl.where(en, tl.where(h_ok, yr, mu), 0), 0))
            tl.store(nacc + r, (n + 1).to(tl.int32))
        _k_acep = _k
    R = cu.shape[0] - 1
    dev = tok_f.device
    sampled = torch.zeros((R, T), dtype=torch.int64, device=dev)
    nacc = torch.ones(R, dtype=torch.int32, device=dev)
    camino = torch.full((R, T - 1), -1, dtype=torch.int64, device=dev)
    filas = torch.arange(tm, dtype=torch.int32, device=dev)[None].repeat(R, 1)
    if rheo is not None:
        hs, razon, y_r, seed, pos_f = rheo
    else:
        hs, razon, y_r, seed, pos_f = padre_v, muestra_f, muestra_f, cu, cu
    alc = alcanzable if alcanzable is not None else padre_v
    _k_acep[(R,)](tok_f, muestra_f, cu, padre_v, padre_v.stride(0), alc, alc.stride(0),
                  sampled, nacc, camino, filas, hs, hs.stride(0), razon, y_r, seed, pos_f,
                  idx_map if idx_map is not None else cu,
                  RHEO=rheo is not None, IDX=idx_map is not None, TODOS=alcanzable is None,
                  T=T, TM=tm, BT=triton.next_power_of_2(T), num_warps=1)
    return sampled, nacc, camino, filas


def filas_de_cinta(camino: torch.Tensor, tm: int) -> torch.Tensor:
    """``camino`` de ``aceptar`` -> filas de cinta para ``gdn_cinta.camino_gpu()`` y
    ``arbol_conv.compactar`` (fila = posicion verificada - 1). El relleno queda en la
    identidad, que nadie lee: solo se usan las primeras ``nacc - 1``."""
    ident = torch.arange(tm, device=camino.device)[None].expand(camino.shape[0], tm)
    n = min(tm, camino.shape[1])
    c = camino[:, :n]
    ident = ident.clone()
    ident[:, :n] = torch.where(c > 0, c - 1, ident[:, :n])
    return ident.to(torch.int32)


def aceptar_goloso(token, padre, objetivo_por_nodo, objetivo_raiz):
    """Recorrido del arbol con decodificacion golosa. ``objetivo_por_nodo[r, n]`` es el token que
    el target elige DESPUES del nodo n; ``objetivo_raiz[r]`` el que elige despues del ancla.
    Devuelve (camino [R, N] con -1 de relleno, largo [R]). Referencia en CPU para los tests."""
    R, N = token.shape
    caminos, largos = [], []
    for r in range(R):
        cam, act, quiero = [], -1, int(objetivo_raiz[r])
        while True:
            sig = [n for n in range(N) if int(padre[r, n]) == act and int(token[r, n]) == quiero]
            if not sig:
                break
            act = sig[0]; cam.append(act); quiero = int(objetivo_por_nodo[r, act])
        caminos.append(cam + [-1] * (N - len(cam))); largos.append(len(cam))
    return torch.tensor(caminos), torch.tensor(largos)


__all__ = ["construir", "construir_dfs_kernel", "ancestros", "aceptar", "aceptar_kernel", "preparar_rheo",
           "preparar_paso_kernel", "compactar_kernel", "reponer_cadena_kernel", "filas_de_cinta", "aceptar_goloso", "orden_dfs", "padres_verificacion",
           "bits_ancestros"]
