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
