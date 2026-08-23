# SPDX-License-Identifier: Apache-2.0
"""Tests de PN102 — desalojo por score de bloque con envejecimiento."""

from __future__ import annotations

import os

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _S():
    from vllm._genesis import kv_block_score as S

    return S


class _Bloque:
    def __init__(self, ref_cnt: int = 0) -> None:
        self.ref_cnt = ref_cnt


class _Mgr:
    def __init__(self) -> None:
        self._policy = type("P", (), {})()


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN102_BLOCK_SCORE", "1")
    monkeypatch.delenv("GENESIS_PN102_AGING_EVERY", raising=False)
    monkeypatch.delenv("GENESIS_PN102_SCAN", raising=False)


# ─────────── la propiedad que hace innecesario el arbol ───────────


def test_el_score_es_monotono_a_lo_largo_del_prefijo():
    """El bloque N+1 nunca puede tener mas score que el N.

    Su hash INCLUYE al anterior, asi que no puede aparecer en mas requests.
    De ahi sale gratis que el desalojo se vaya a las colas.
    """
    S = _S()
    m = _Mgr()
    prefijo = [b"p0", b"p1", b"p2"]
    for i in range(5):
        # cada request comparte el prefijo y agrega una cola distinta
        for k in prefijo[: 3 if i < 3 else 2]:
            S.anotar(m, k)
        S.anotar(m, b"cola-%d" % i)
    sc = m._g102["scores"]
    valores = [sc[k] for k in prefijo]
    assert valores == sorted(valores, reverse=True), valores
    assert all(sc[k] > sc[b"cola-0"] for k in prefijo[:2])


def test_la_victima_es_una_cola_y_no_parte_el_prefijo():
    S = _S()
    m = _Mgr()
    for _ in range(6):
        for k in (b"p0", b"p1", b"p2"):
            S.anotar(m, k)
    S.anotar(m, b"cola")
    sc = m._g102["scores"]
    items = [(k, _Bloque()) for k in sc]
    elegida = S.elegir(items, set(), set(), sc)
    assert elegida[0] == b"cola"


def test_apagado_se_comporta_como_vllm(monkeypatch):
    """Sin scores devuelve la PRIMERA elegible: identico al codigo actual."""
    S = _S()
    items = [(b"a", _Bloque()), (b"b", _Bloque())]
    assert S.elegir(items, set(), set(), None)[0] == b"a"


def test_respeta_protected_y_ref_cnt():
    S = _S()
    items = [
        (b"usado", _Bloque(ref_cnt=1)),
        (b"protegido", _Bloque()),
        (b"libre", _Bloque()),
    ]
    assert S.elegir(items, {b"protegido"}, set(), {b"libre": 9})[0] == b"libre"
    assert S.elegir(items, {b"protegido"}, {b"libre"}, {}) is None


def test_no_repite_una_clave_ya_elegida():
    S = _S()
    items = [(b"a", _Bloque()), (b"b", _Bloque())]
    assert S.elegir(items, set(), {b"a"}, {b"a": 0, b"b": 5})[0] == b"b"


# ─────────── envejecimiento ───────────


def test_el_envejecimiento_halva_y_descarta_los_ceros(monkeypatch):
    monkeypatch.setenv("GENESIS_PN102_AGING_EVERY", "8")
    S = _S()
    m = _Mgr()
    for _ in range(4):
        S.anotar(m, b"caliente")
    for _ in range(4):
        S.anotar(m, b"tibio")
    sc = m._g102["scores"]
    assert m._g102["envejecidas"] == 1
    assert sc[b"caliente"] == 2 and sc[b"tibio"] == 2


def test_lo_que_deja_de_repetirse_desaparece(monkeypatch):
    """Es lo que le falta a ARC: T2 es permanente y esto no."""
    monkeypatch.setenv("GENESIS_PN102_AGING_EVERY", "4")
    S = _S()
    m = _Mgr()
    S.anotar(m, b"viejo")
    for _ in range(30):
        S.anotar(m, b"nuevo")
    sc = m._g102["scores"]
    assert b"viejo" not in sc, "un bloque que no se repite tiene que decaer a 0"
    assert sc.get(b"nuevo", 0) > 0


def test_el_dict_de_scores_tiene_techo():
    S = _S()
    m = _Mgr()
    for i in range(S._MAX_CLAVES + 100):
        S.anotar(m, b"k%d" % i)
    assert len(m._g102["scores"]) <= S._MAX_CLAVES


# ─────────── costo acotado ───────────


def test_solo_mira_las_primeras_n_candidatas(monkeypatch):
    """Ordenar 790 candidatas en el camino caliente costaria mas de lo que ahorra."""
    monkeypatch.setenv("GENESIS_PN102_SCAN", "3")
    S = _S()
    scores = {b"a": 5, b"b": 4, b"c": 3, b"lejos": 0}
    items = [(k, _Bloque()) for k in (b"a", b"b", b"c", b"lejos")]
    # 'lejos' tiene score 0 pero esta fuera de la ventana
    assert S.elegir(items, set(), set(), scores)[0] == b"c"


def test_corta_apenas_encuentra_un_score_cero():
    S = _S()
    scores = {b"a": 5, b"b": 0, b"c": 1}
    items = [(k, _Bloque()) for k in (b"a", b"b", b"c")]
    assert S.elegir(items, set(), set(), scores)[0] == b"b"


def test_anotar_nunca_levanta():
    S = _S()

    class _Roto:
        @property
        def _policy(self):
            raise RuntimeError("roto")

    S.anotar(_Roto(), b"k")  # no debe propagar


# ─────────── anclas ───────────


def _tree():
    c = []
    try:
        import vllm

        c.append(os.path.dirname(vllm.__file__))
    except Exception:
        pass
    c.append(os.path.join(_REPO, "assets", "vllm", "vllm"))
    for r in c:
        if r and os.path.exists(os.path.join(r, "v1/kv_offload/cpu/policies/arc.py")):
            return r
    return None


ROOT = _tree()


@pytest.mark.skipif(ROOT is None, reason="no hay arbol de vLLM")
def test_anclas_unicas_y_compilan():
    from vllm._genesis.wiring.hybrid import patch_N102_block_score_eviction as P

    for rel, pares in (
        ("v1/kv_offload/cpu/manager.py", [("LOOKUP_OLD", "LOOKUP_NEW")]),
        ("v1/kv_offload/cpu/policies/arc.py", [("T1_OLD", "T1_NEW"), ("T2_OLD", "T2_NEW")]),
    ):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            t = fh.read()
        if P.GENESIS_PN102_MARKER in t:
            pytest.skip(f"PN102 ya aplicado sobre {rel}")
        for o, n in pares:
            assert t.count(getattr(P, o)) == 1, f"{rel}:{o}"
            t = t.replace(getattr(P, o), getattr(P, n))
        compile(t, rel, "exec")


@pytest.mark.skipif(ROOT is None, reason="no hay arbol de vLLM")
def test_no_reusa_counts():
    """Regresion: usar `counts` como score romperia el guardado entero.

    prepare_store filtra por `counts >= store_threshold` y
    _maximal_prefix_lookup corta en el primer miss, asi que los bloques NUEVOS
    quedan en 0 y se descartarian todos.
    """
    from vllm._genesis.wiring.hybrid import patch_N102_block_score_eviction as P

    assert "self.counts" not in P.LOOKUP_NEW.split("_g102.anotar")[1]
    with open(os.path.join(ROOT, "v1/kv_offload/cpu/manager.py"), encoding="utf-8") as fh:
        assert "counts.get(k, 0) >= self.store_threshold" in fh.read()
