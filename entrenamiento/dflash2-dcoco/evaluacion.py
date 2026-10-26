"""Recorridos, arbol, aceptacion con muestreo y calibracion del arbol para entrenar_dcoco.py.

Un "puntuador" da las filas de la tabla de transiciones a medida que se recorre:

* ``Tabla``: filas fijas de una tabla [B, S, K, K] (el selector, o selector + cabeza de pares). Es lo
  que usa el servidor.
* ``recurrente.Puntuador``: filas que dependen del camino elegido (la cabeza recurrente).

Con una ``Tabla``, ``arbol`` arma exactamente el mismo arbol que ``arbol_borrador.construir`` (lo
verifica test_entrenar_dcoco.py); con ``escala`` cada fila de profundidad d se multiplica por
``escala[d]`` antes del log_softmax (la calibracion del paso 3 del README).
"""
from __future__ import annotations

import torch


class Tabla:
    """Filas de una tabla fija [B, S, K, K]: no hay estado de camino (``nodo`` devuelve None)."""

    def __init__(self, sc):
        self.sc = sc
        self.filas = torch.arange(sc.shape[0], device=sc.device)

    def raiz(self):
        return self.sc[:, 0, 0]

    def nodo(self, est_padre, d, i):
        return None

    def fila(self, est, d, i):
        """Puntajes de los hijos (profundidad d+1) del candidato i de la profundidad d; d, i [B]."""
        return self.sc[self.filas, (d + 1).clamp(max=self.sc.shape[1] - 1), i]


def cadena(punt, cand):
    """Cadena golosa prefijo-primero: tokens [B, S]."""
    B, S, K = cand.shape
    i = punt.raiz().argmax(-1)
    idx, est = [i], None
    for dd in range(S - 1):
        d = torch.full((B,), dd, dtype=torch.int64, device=cand.device)
        est = punt.nodo(est, d, i)
        i = punt.fila(est, d, i).argmax(-1)
        idx.append(i)
    return cand.gather(2, torch.stack(idx, 1)[..., None]).squeeze(-1)


def arbol(punt, cand, n, escala=None):
    """Best-first sobre la probabilidad de camino, como arbol_borrador.construir pero con las filas del
    puntuador (pueden depender del camino). Devuelve token, padre, prof [B, n]."""
    B, S, K = cand.shape
    dev = cand.device
    e = torch.ones(S, device=dev) if escala is None else torch.as_tensor(escala, device=dev, dtype=torch.float32)
    F = K * (n + 1)
    f_lp = torch.full((B, F), float("-inf"), device=dev)
    f_padre = torch.full((B, F), -1, dtype=torch.int64, device=dev)
    f_prof = torch.zeros((B, F), dtype=torch.int64, device=dev)
    f_idx = torch.zeros((B, F), dtype=torch.int64, device=dev)
    ar = torch.arange(K, device=dev)
    f_lp[:, :K] = torch.log_softmax(punt.raiz().float() * e[0], -1)
    f_idx[:, :K] = ar
    tok = torch.zeros((B, n), dtype=torch.int64, device=dev)
    padre = torch.full((B, n), -1, dtype=torch.int64, device=dev)
    prof = torch.zeros((B, n), dtype=torch.int64, device=dev)
    filas = torch.arange(B, device=dev)
    estados = None                                                     # [B, n, ...] si hay estado de camino
    for m in range(n):
        j = f_lp.argmax(1)
        lp, d, i, pa = f_lp[filas, j], f_prof[filas, j], f_idx[filas, j], f_padre[filas, j]
        tok[:, m] = cand[filas, d, i]
        padre[:, m], prof[:, m] = pa, d
        f_lp.scatter_(1, j[:, None], float("-inf"))
        est_pa = None
        if estados is not None:
            est_pa = torch.where((pa >= 0)[:, None], estados[filas, pa.clamp(min=0)], torch.zeros_like(estados[:, 0]))
        est = punt.nodo(est_pa, d, i)
        if est is not None:
            if estados is None:
                estados = torch.zeros(B, n, *est.shape[1:], device=dev, dtype=est.dtype)
            estados[:, m] = est
        fila = punt.fila(est, d, i).float() * e[(d + 1).clamp(max=S - 1)][:, None]
        hijos = lp[:, None] + torch.log_softmax(fila, -1)
        hijos = torch.where((d + 1 < S)[:, None], hijos, torch.full_like(hijos, float("-inf")))
        base = K * (m + 1)
        f_lp[:, base:base + K] = hijos
        f_padre[:, base:base + K] = m
        f_prof[:, base:base + K] = (d + 1)[:, None]
        f_idx[:, base:base + K] = ar
    return tok, padre, prof


# ───────────────────────────── aceptacion ─────────────────────────────

def aceptados_cadena(tok, lab):
    """Verificacion golosa: 1 mientras el token propuesto es el del target. [B, S]"""
    ok = (tok == lab) & (lab >= 0)
    return torch.cumprod(ok.int(), -1)


def aceptados_arbol(tok, padre, prof, lab):
    """Verificacion golosa del arbol: largo aceptado (sin el bonus) [B]."""
    B, n = tok.shape
    ok = torch.zeros(B, n, dtype=torch.bool, device=tok.device)
    for j in range(n):
        real = lab.gather(1, prof[:, j:j + 1]).squeeze(1)
        p = padre[:, j]
        previo = torch.where(p < 0, torch.ones_like(p, dtype=torch.bool),
                             ok.gather(1, p.clamp(min=0)[:, None]).squeeze(1))
        ok[:, j] = (tok[:, j] == real) & (real >= 0) & previo
    return torch.where(ok, prof + 1, torch.zeros_like(prof)).max(1).values


def dist_muestreo(top_p, T, nucleo=1.0):
    """Distribucion del target con temperatura T (y top-p), sobre su top-16 capturado. top_p [B, S, 16]
    son las probabilidades a T=1 renormalizadas sobre el top-16 (como las arma dflash2/entrenar.armar_lote)."""
    lp = torch.where(top_p > 0, top_p.clamp_min(1e-30).log() / T, torch.full_like(top_p, float("-inf")))
    p = torch.softmax(lp, -1).nan_to_num(0.0)                         # en log: sin underflow a T chica
    if nucleo < 1.0:
        orden = p.sort(-1, descending=True)
        acum = orden.values.cumsum(-1)
        dentro = (acum - orden.values) < nucleo                       # el primero siempre entra
        mascara = torch.zeros_like(p, dtype=torch.bool).scatter(-1, orden.indices, dentro)
        p = torch.where(mascara, p, torch.zeros_like(p))
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-30)
    return p


def prob_de(tok, prof, top_i, pT):
    """p_T del token tok en su profundidad: tok/prof [B, n] (o tok [B, S] con prof None)."""
    if prof is None:
        return ((top_i == tok[..., None]).float() * pT).sum(-1)
    ti = top_i.gather(1, prof[..., None].expand(-1, -1, top_i.shape[-1]))
    pt = pT.gather(1, prof[..., None].expand(-1, -1, pT.shape[-1]))
    return ((ti == tok[..., None]).float() * pt).sum(-1)


def muestreo_cadena(tok, top_i, pT):
    """Largo aceptado esperado con muestreo (propuesta determinista, se acepta con p_T): [B].

    Aproximacion: la distribucion del target en cada profundidad es la capturada con el prefijo real
    (no hay capturas para los prefijos que el target no genero)."""
    return torch.cumprod(prob_de(tok, None, top_i, pT), -1).sum(-1)


def muestreo_arbol(tok, padre, prof, top_i, pT):
    """Largo aceptado esperado con muestreo verificando el arbol: el target muestrea y se baja al hijo que
    coincide. Suma sobre los nodos de P(llegar y aceptar), con la misma aproximacion de contexto. [B]"""
    B, n = tok.shape
    q = prob_de(tok, prof, top_i, pT)
    P = torch.zeros(B, n, device=tok.device)
    for j in range(n):
        p = padre[:, j]
        previo = torch.where(p < 0, torch.ones_like(q[:, j]), P.gather(1, p.clamp(min=0)[:, None]).squeeze(1))
        P[:, j] = previo * q[:, j]
    return P.sum(-1)


# ───────────────────────────── calibracion del arbol ─────────────────────────────

GRILLA = (0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0)


def calibrar(lotes, objetivo, S, rondas=2, grilla=GRILLA, log=print):
    """Escala por profundidad que maximiza ``objetivo(lote, escala) -> suma`` sobre ``lotes`` (descenso
    por coordenadas en una grilla). Devuelve la lista de S escalas."""
    esc = [1.0] * S

    def total(e):
        return sum(objetivo(lt, e) for lt in lotes)

    mejor = total(esc)
    for r in range(rondas):
        for d in range(S):
            for g in grilla:
                prueba = esc[:d] + [g] + esc[d + 1:]
                v = total(prueba)
                if v > mejor + 1e-9:
                    mejor, esc = v, prueba
        log(f"calibracion ronda {r}: escala {['%.2f' % x for x in esc]} objetivo {mejor:.1f}")
    return esc
