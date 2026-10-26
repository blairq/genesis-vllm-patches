# SPDX-License-Identifier: Apache-2.0
"""PN178: cabeza DCOCO residual sobre el selector de DFlash2 (CPU, sin vLLM).

    PYTHONPATH=. python3 -m pytest -q vllm/_genesis/tests/test_pn178_dcoco.py
"""
import types

import pytest
import torch
import torch.nn.functional as F

from vllm._genesis import dcoco as D

H, K, S, R = 64, 16, 8, 3


def _cabeza(semilla=0, cero=False):
    torch.manual_seed(semilla)
    c = D.CabezaDCOCO(H, K, emb=16, state=12, mlp=24)
    if not cero:
        for p in c.parameters():
            torch.nn.init.normal_(p, std=0.2)
    return c


def _entradas(semilla=1):
    g = torch.Generator().manual_seed(semilla)
    return (torch.randn(R, S, K, H, generator=g), torch.randn(R, S, K, generator=g) * 4,
            torch.randn(R, S, H, generator=g), torch.randn(R, S, K, K, generator=g))


def _delta_concat(c, emb, lg, h):
    """La cuenta de DCOCOHead del paper, armando las K x K concatenaciones a mano."""
    e = c.emb
    base = F.layer_norm(c.cand_proj(emb), (e,))
    ctx = F.layer_norm(c.ctx_h_proj(h), (e,)).unsqueeze(-2).expand_as(base)
    pe = c.ctx_out_norm(c.cand_ctx_proj(torch.cat((base, ctx), -1)))
    st = c.pair_proj(torch.cat((h[:, :-1], h[:, 1:]), -1))                       # [R, S-1, state]
    Kk = emb.shape[2]
    x = torch.cat((st[:, :, None, None].expand(-1, -1, Kk, Kk, -1),
                   pe[:, :-1, :, None].expand(-1, -1, -1, Kk, -1),
                   pe[:, 1:, None, :].expand(-1, -1, Kk, -1, -1),
                   lg[:, :-1, :, None, None].expand(-1, -1, -1, Kk, 1),
                   lg[:, 1:, None, :, None].expand(-1, -1, Kk, -1, 1)), -1)
    return c.trans_mlp(x).squeeze(-1)


def test_sin_entrenar_no_cambia_la_tabla():
    c = _cabeza(cero=True)
    emb, lg, h, sc = _entradas()
    assert torch.equal(c(sc, emb, lg, h), sc)


def test_partida_igual_a_la_concatenacion():
    c = _cabeza()
    emb, lg, h, _ = _entradas()
    d = c.delta(emb, lg, h)
    ref = _delta_concat(c, emb, lg, h)
    assert torch.allclose(d, ref, atol=1e-4, rtol=1e-4)


def test_materializar_da_lo_mismo():
    c = _cabeza().eval()
    emb, lg, h, sc = _entradas()
    a = c(sc, emb, lg, h)
    c.materializar()
    assert torch.allclose(c(sc, emb, lg, h), a, atol=1e-6)


def test_fila_cero_intacta():
    c = _cabeza()
    emb, lg, h, sc = _entradas()
    out = c(sc, emb, lg, h)
    assert torch.equal(out[:, 0], sc[:, 0]) and not torch.equal(out[:, 1:], sc[:, 1:])


def test_nombres_como_dcocohead():
    nombres = {n for n, _ in _cabeza().named_parameters()}
    assert nombres == {"cand_proj.weight", "ctx_h_proj.weight", "cand_ctx_proj.weight", "ctx_out_norm.weight",
                       "ctx_out_norm.bias", "pair_proj.weight", "trans_mlp.0.weight", "trans_mlp.0.bias",
                       "trans_mlp.2.weight", "trans_mlp.2.bias"}


CFG = {"selector_top_k": 16, "dcoco_emb_dim": 16, "dcoco_state_dim": 12, "dcoco_mlp_dim": 24}


def test_crear_modos():
    m = torch.nn.Module()
    D.crear(m, {"selector_top_k": 16}, H, None)
    assert m.dcoco is None
    with pytest.raises(ValueError):
        D.crear(m, {"selector_top_k": 16}, H, True)
    D.crear(m, CFG, H, False)
    assert m.dcoco is None
    D.crear(m, CFG, H, None)
    assert isinstance(m.dcoco, D.CabezaDCOCO) and m.dcoco.topk == 16


def _pesos(c, prefijo="dcoco."):
    return [(prefijo + n, p.detach().clone() + 1) for n, p in c.named_parameters()]


def _cargador(modelo):
    def cargar_base(ws):
        vistos = []
        for n, w in ws:
            vistos.append(n)
            corto = n[len("model."):] if n.startswith("model.") else n
            if corto.startswith("dcoco."):
                modelo.dcoco.get_parameter(corto[len("dcoco."):]).data.copy_(w)
        return set(vistos)
    return cargar_base


def test_cargar_sin_cabeza_descarta_las_claves():
    m = torch.nn.Module()
    D.crear(m, CFG, H, False)
    pesos = [("layers.0.q_proj.weight", torch.zeros(1))] + _pesos(_cabeza())
    r = D.cargar(m, _cargador(m), pesos)
    assert r == {"layers.0.q_proj.weight"}


def test_cargar_con_cabeza_completa_y_materializa():
    m = torch.nn.Module()
    D.crear(m, CFG, H, True)
    fuente = D.CabezaDCOCO(H, 16, 16, 12, 24)
    D.cargar(m, _cargador(m), _pesos(fuente, "model.dcoco."))
    assert torch.equal(m.dcoco.cand_proj.weight, fuente.cand_proj.weight + 1)
    assert m.dcoco._partida is not None and not m.dcoco.training


def test_cargar_incompleto_falla():
    m = torch.nn.Module()
    D.crear(m, CFG, H, True)
    with pytest.raises(ValueError, match="incompleto"):
        D.cargar(m, _cargador(m), _pesos(D.CabezaDCOCO(H, 16, 16, 12, 24))[:-1])


def test_corregir_con_modelo_falso():
    c = _cabeza().eval()
    tabla = torch.randn(500, H)
    modelo = types.SimpleNamespace(dcoco=c, embed_input_ids=lambda ids: tabla[ids])
    cand = torch.randint(0, 500, (R, S, K))
    _, lg, h, sc = _entradas()
    out = D.corregir(modelo, sc, cand, lg, h)
    assert torch.allclose(out, c(sc, tabla[cand], lg, h))
    assert D.corregir(types.SimpleNamespace(dcoco=None), sc, cand, lg, h) is sc
