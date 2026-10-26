"""Pruebas en CPU de la perdida, las metricas y el aprendizaje de la cabeza DCOCO (sin checkpoint real).

    python3 -m pytest -q entrenamiento/dflash2-dcoco/test_entrenar_dcoco.py

Necesita torch, numpy y safetensors (no vLLM): importa entrenar_dcoco, que carga dflash2/ por ruta.
"""
import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entrenar_dcoco as T  # noqa: E402

B, S, K = 4, 6, 8


def _cand(semilla=0):
    g = torch.Generator().manual_seed(semilla)
    return torch.stack([torch.stack([torch.randperm(100, generator=g)[:K] for _ in range(S)]) for _ in range(B)])


def test_perdida_baja_si_la_transicion_real_gana():
    cand = _cand()
    lab = cand[:, :, 2].clone()                                          # el real es siempre el candidato 2
    sc = torch.zeros(B, S, K, K)
    bien = sc.clone()
    bien[:, 1:, 2, 2] = 20.0
    assert T.perdida_dcoco(bien, cand, lab) < 1e-6 < T.perdida_dcoco(sc, cand, lab)
    for modo in ("fila", "conjunta"):
        x = sc.clone().requires_grad_(True)
        T.perdida_dcoco(x, cand, lab, modo=modo).backward()
        assert x.grad[:, 1:, 2].abs().sum() > 0 and x.grad[:, 0].abs().sum() == 0


def test_perdida_ignora_lo_que_no_esta_en_el_top_k():
    cand = _cand()
    lab = torch.full((B, S), 999)                                        # nunca entre los candidatos
    assert T.perdida_dcoco(torch.randn(B, S, K, K), cand, lab) == 0


def test_recorrido_y_cadena():
    cand = _cand()
    sc = torch.full((B, S, K, K), -5.0)
    sc[:, 0, :, 3] = 1.0                                                 # fila 0: el 3
    for s in range(1, S):
        sc[:, s, 3, 1] = 1.0 if s % 2 else -9.0                          # desde el 3 va al 1 (o no)
        sc[:, s, 1, 3] = 1.0
    tok = T.recorrido(sc, cand)
    esperado = torch.stack([cand[:, s, 3 if s % 2 == 0 else 1] for s in range(S)], 1)
    assert torch.equal(tok, esperado)
    lab = tok.clone()
    lab[:, 3] = -1
    assert T.aceptados_cadena(tok, lab).sum(1).tolist() == [3] * B


def _arbol_fuerza_bruta(sc, cand, lab, n):
    tok, padre, prof, _ = T.AB.construir(cand, sc, n)
    out = []
    for b in range(cand.shape[0]):
        mejor = 0
        for j in range(n):
            ok, k = True, j
            while k >= 0:
                if int(tok[b, k]) != int(lab[b, prof[b, k]]):
                    ok = False
                k = int(padre[b, k])
            if ok:
                mejor = max(mejor, int(prof[b, j]) + 1)
        out.append(mejor)
    return out


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_aceptados_arbol_contra_fuerza_bruta(semilla):
    g = torch.Generator().manual_seed(semilla)
    cand = _cand(semilla)
    sc = torch.randn(B, S, K, K, generator=g) * 3
    tok = T.recorrido(sc, cand)
    lab = torch.where(torch.rand(B, S, generator=g) < 0.8, tok, cand[:, :, 1])
    assert T.aceptados_arbol(sc, cand, lab, 8).tolist() == _arbol_fuerza_bruta(sc, cand, lab, 8)


def test_la_cabeza_aprende_una_dependencia_entre_posiciones():
    """El real en p es el candidato que 'sigue' al real en p-1 segun su embedding (dependencia que la
    tabla unaria no ve). Con la tabla del selector en cero, solo la cabeza puede acertarla."""
    torch.manual_seed(0)
    H, V, n = 32, 64, 256
    emb_tab = torch.randn(V, H)
    sucesor = torch.randperm(V)                                          # token t -> sucesor[t]
    cab = T.D.CabezaDCOCO(H, K, emb=16, state=8, mlp=32)
    opt = torch.optim.Adam(cab.parameters(), lr=3e-3)

    def lote():
        cand = torch.stack([torch.randperm(V)[:K] for _ in range(n * S)]).view(n, S, K)
        lab = torch.empty(n, S, dtype=torch.long)
        lab[:, 0] = cand[:, 0, 0]
        for s in range(1, S):
            j = torch.randint(0, K, (n,))
            cand[torch.arange(n), s, j] = sucesor[lab[:, s - 1]]
            lab[:, s] = sucesor[lab[:, s - 1]]
        return cand, lab

    def evaluar(c):
        cand, lab = lote()
        sc = torch.zeros(n, S, K, K)
        sc[:, 0, :, 0] = 5.0
        with torch.no_grad():
            sc2 = c(sc, emb_tab[cand], torch.zeros(n, S, K), torch.zeros(n, S, H))
        return float(T.aceptados_cadena(T.recorrido(sc2, cand), lab).sum(1).float().mean())

    antes = evaluar(cab)
    for _ in range(300):
        cand, lab = lote()
        sc = torch.zeros(n, S, K, K)
        sc2 = cab(sc, emb_tab[cand], torch.zeros(n, S, K), torch.zeros(n, S, H))
        loss = T.perdida_dcoco(sc2, cand, lab)
        opt.zero_grad()
        loss.backward()
        opt.step()
    despues = evaluar(cab)
    assert antes < 1.5 and despues > 4.0, (antes, despues)


# ───────────── evaluacion.py y recurrente.py (--temperaturas, --calibrar, --estado recurrente) ─────────────

EV, RC = T.EV, T.RC


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_arbol_con_tabla_igual_al_del_servidor(semilla):
    g = torch.Generator().manual_seed(semilla)
    cand = _cand(semilla)
    sc = torch.randn(B, S, K, K, generator=g) * 3
    tok, padre, prof, _ = T.AB.construir(cand, sc, 8)
    t2, p2, d2 = EV.arbol(EV.Tabla(sc), cand, 8)
    assert torch.equal(tok, t2) and torch.equal(padre, p2) and torch.equal(prof, d2)
    assert torch.equal(EV.cadena(EV.Tabla(sc), cand), T.recorrido(sc, cand))
    lab = torch.where(torch.rand(B, S, generator=g) < 0.8, T.recorrido(sc, cand), cand[:, :, 1])
    assert EV.aceptados_arbol(t2, p2, d2, lab).tolist() == _arbol_fuerza_bruta(sc, cand, lab, 8)


def test_escala_uno_no_cambia_el_arbol_y_la_escala_si():
    cand = _cand()
    sc = torch.randn(B, S, K, K) * 3
    a = EV.arbol(EV.Tabla(sc), cand, 8)
    b = EV.arbol(EV.Tabla(sc), cand, 8, [1.0] * S)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    c = EV.arbol(EV.Tabla(sc), cand, 8, [1.0] + [0.05] * (S - 1))      # filas planas: el arbol se ensancha
    assert int(c[2].max()) <= int(a[2].max())


def _top(lab, semilla=0):
    """top-16 del target con el real primero y probabilidades decrecientes."""
    g = torch.Generator().manual_seed(semilla)
    resto = torch.randint(1000, 2000, (*lab.shape, 15), generator=g)
    top_i = torch.cat((lab[..., None], resto), -1)
    top_p = torch.softmax(torch.linspace(3, -3, 16).expand(*lab.shape, 16), -1)
    return top_i, top_p


def test_muestreo_a_temperatura_baja_es_la_golosa():
    cand = _cand()
    sc = torch.randn(B, S, K, K) * 3
    tok = EV.cadena(EV.Tabla(sc), cand)
    lab = torch.where(torch.rand(B, S) < 0.7, tok, cand[:, :, 1])
    top_i, top_p = _top(lab)
    pT = EV.dist_muestreo(top_p, 1e-3)
    assert torch.allclose(EV.muestreo_cadena(tok, top_i, pT), EV.aceptados_cadena(tok, lab).sum(1).float())
    t, pa, pr = EV.arbol(EV.Tabla(sc), cand, 8)
    assert torch.allclose(EV.muestreo_arbol(t, pa, pr, top_i, pT), EV.aceptados_arbol(t, pa, pr, lab).float())
    # a T=1 el esperado es menor (el target a veces muestrea otro) y el nucleo lo acerca a la golosa
    p1 = EV.dist_muestreo(top_p, 1.0)
    pn = EV.dist_muestreo(top_p, 1.0, nucleo=0.5)
    assert EV.muestreo_cadena(tok, top_i, p1).sum() < EV.muestreo_cadena(tok, top_i, pn).sum()
    assert torch.allclose(pn.sum(-1), torch.ones(B, S))


def test_calibrar_encuentra_la_escala():
    obj = {0: 1.0, 1: 0.6, 2: 1.7}
    f = lambda _, e: -sum((e[d] - v) ** 2 for d, v in obj.items())    # noqa: E731
    assert EV.calibrar([None], f, 3, log=lambda m: None) == [1.0, 0.6, 1.7]


def _recurrente(cero=False):
    torch.manual_seed(0)
    cab = RC.CabezaRecurrente(32, K, emb=16, state=8, mlp=24)
    if not cero:
        for p in cab.parameters():
            torch.nn.init.normal_(p, std=0.2)
    return cab


def test_recurrente_en_cero_es_el_selector():
    cab = _recurrente(cero=True)
    cand, sc = _cand(), torch.randn(B, S, K, K)
    emb, lg, h = torch.randn(B, S, K, 32), torch.randn(B, S, K), torch.randn(B, S, 32)
    assert torch.allclose(cab.tabla_forzada(sc, emb, lg, h, torch.zeros(B, S, dtype=torch.long)), sc)
    a = EV.arbol(EV.Tabla(sc), cand, 8)
    b = EV.arbol(RC.Puntuador(cab, sc, emb, lg, h), cand, 8)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_recurrente_tabla_forzada_igual_al_puntuador_por_el_camino_real():
    """La tabla del entrenamiento (prefijo real) y las filas de la evaluacion (prefijo elegido) son la
    misma cuenta cuando el camino elegido es el real."""
    cab = _recurrente()
    sc = torch.randn(B, S, K, K)
    emb, lg, h = torch.randn(B, S, K, 32), torch.randn(B, S, K), torch.randn(B, S, 32)
    real = torch.randint(0, K, (B, S))
    tf = cab.tabla_forzada(sc, emb, lg, h, real)
    pu = RC.Puntuador(cab, sc, emb, lg, h)
    assert torch.allclose(pu.raiz(), tf[:, 0, 0])
    est = None
    for p in range(S - 1):
        d = torch.full((B,), p)
        est = pu.nodo(est, d, real[:, p])
        assert torch.allclose(pu.fila(est, d, real[:, p]), tf[torch.arange(B), p + 1, real[:, p]], atol=1e-5)
    pd = RC.Puntuador.desde(cab, sc, cab.pe(emb, h), lg)
    assert all(torch.equal(x, y) for x, y in zip(EV.arbol(pu, _cand(), 8), EV.arbol(pd, _cand(), 8)))


def test_recurrente_usa_el_prefijo():
    """Dos prefijos distintos con el mismo previo dan filas distintas (la de pares no puede)."""
    cab = _recurrente()
    sc = torch.zeros(1, S, K, K)
    emb, lg, h = torch.randn(1, S, K, 32), torch.randn(1, S, K), torch.randn(1, S, 32)
    a = cab.tabla_forzada(sc, emb, lg, h, torch.tensor([[0] * S]))
    b = cab.tabla_forzada(sc, emb, lg, h, torch.tensor([[1] + [0] * (S - 1)]))
    assert torch.allclose(a[:, 1], b[:, 1]) and not torch.allclose(a[:, 2, 0], b[:, 2, 0])


def test_perdida_con_prefijo_corta_en_el_primer_fuera():
    cand = _cand()
    lab = cand[:, :, 2].clone()
    lab[:, 2] = 999                                                     # fuera del top-K en la posicion 2
    sc = torch.randn(B, S, K, K)
    x = sc.clone().requires_grad_(True)
    T.perdida_dcoco(x, cand, lab, prefijo=True).backward()
    assert x.grad[:, 1].abs().sum() > 0 and x.grad[:, 3:].abs().sum() == 0
    y = sc.clone().requires_grad_(True)
    T.perdida_dcoco(y, cand, lab).backward()
    assert y.grad[:, 4:].abs().sum() > 0


def test_ce_forzada():
    cand = _cand()
    lab = cand[:, :, 2].clone()
    sc = torch.zeros(B, S, K, K)
    sc[:, 1:, 2, 2] = 50.0
    suma, n, top1 = T.ce_forzada(sc, cand, lab)
    assert n == B * (S - 1) and top1 == n and suma < 1e-6
