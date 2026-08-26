# SPDX-License-Identifier: Apache-2.0
"""TDD for PN88 — telemetría de los tiers del cache de KV.

Cubre el sink (`vllm/_genesis/kv_tier_metrics.py`), su emisión a Prometheus,
y la superficie de anclas del wiring.

El sink cruza un límite de proceso (scheduler -> logger) dentro de
`OffloadingConnectorStats.data`, así que la serializabilidad a JSON no es un
detalle: es un requisito duro y tiene su propio test.
"""
from __future__ import annotations

import json

import pytest


def _ktm():
    from vllm._genesis import kv_tier_metrics as M
    return M


def _wiring():
    from vllm._genesis.wiring.hybrid import patch_N88_kv_tier_metrics as M
    return M


@pytest.fixture
def sink(monkeypatch):
    """Sink limpio, con PN88 habilitado y la allowlist por default."""
    M = _ktm()
    monkeypatch.setenv("GENESIS_ENABLE_PN88_KV_TIER_METRICS", "1")
    monkeypatch.delenv("GENESIS_KV_AGENTS", raising=False)
    monkeypatch.setattr(M, "_AGENTS_CACHE", None, raising=False)
    s = M.TierStatsSink()
    monkeypatch.setattr(M, "_SINK", s, raising=False)
    return s


# ───────────────────────────── sink ─────────────────────────────


def test_drain_devuelve_y_resetea(sink):
    sink.inc("kv_tier_ops_total", (("tier", "disk"),), 3)
    first = sink.drain()
    assert first["counters"]["kv_tier_ops_total|tier=disk"] == 3
    assert sink.drain() is None, "drain debe vaciar el sink"


def test_drain_vacio_es_none(sink):
    assert sink.drain() is None


def test_clave_independiente_del_orden_de_labels(sink):
    sink.inc("m", (("b", "2"), ("a", "1")), 1)
    sink.inc("m", (("a", "1"), ("b", "2")), 1)
    counters = sink.drain()["counters"]
    assert len(counters) == 1, "el mismo juego de labels debe caer en una serie"
    assert counters["m|a=1,b=2"] == 2


def test_parse_key_es_inversa_de_la_clave(sink):
    M = _ktm()
    sink.inc("kv_tier_bytes_total", (("tier", "disk"), ("agent", "coder")), 1)
    key = next(iter(sink.drain()["counters"]))
    name, labels = M.parse_key(key)
    assert name == "kv_tier_bytes_total"
    assert labels == {"tier": "disk", "agent": "coder"}


def test_parse_key_sin_labels():
    assert _ktm().parse_key("solo_nombre") == ("solo_nombre", {})


def test_payload_es_serializable_a_json(sink):
    """Requisito duro: el payload viaja scheduler -> logger entre procesos."""
    sink.inc("c", (("tier", "disk"),), 1)
    sink.observe("h", (("tier", "disk"),), 0.5)
    sink.set("g", (("tier", "disk"),), 42)
    json.dumps(sink.drain())


def test_merge_suma_concatena_y_pisa():
    M = _ktm()
    a = {"counters": {"c": 10}, "hist": {"h": [1.0]}, "gauges": {"g": 100}}
    b = {"counters": {"c": 5}, "hist": {"h": [2.0, 3.0]}, "gauges": {"g": 200}}
    out = M.merge(dict(a), b)
    assert out["counters"]["c"] == 15, "los contadores suman"
    assert out["hist"]["h"] == [1.0, 2.0, 3.0], "los histogramas concatenan"
    assert out["gauges"]["g"] == 200, "el gauge lo pisa el ultimo"


def test_merge_admite_claves_nuevas():
    M = _ktm()
    out = M.merge({"counters": {"a": 1}}, {"counters": {"b": 2}})
    assert out["counters"] == {"a": 1, "b": 2}


# ─────────────────────────── allowlist ───────────────────────────


@pytest.mark.parametrize(
    "params,esperado",
    [
        ({"genesis_agent": "coder"}, "coder"),
        ({"genesis_agent": "CODER"}, "coder"),
        ({"genesis_agent": "  planner  "}, "planner"),
        ({"genesis_agent": "no_existe"}, "other"),
        ({"genesis_agent": ""}, "unknown"),
        ({}, "unknown"),
        (None, "unknown"),
        ("no soy un dict", "unknown"),
    ],
)
def test_agent_label_acota_la_cardinalidad(sink, params, esperado):
    assert _ktm().agent_label(params) == esperado


def test_allowlist_configurable_por_env(monkeypatch):
    M = _ktm()
    monkeypatch.setenv("GENESIS_KV_AGENTS", "alpha,beta")
    monkeypatch.setattr(M, "_AGENTS_CACHE", None, raising=False)
    assert M.agent_label({"genesis_agent": "alpha"}) == "alpha"
    assert M.agent_label({"genesis_agent": "coder"}) == "other"


# ──────────────────────── helpers inyectados ────────────────────────


class _FakeJobMeta:
    def __init__(self, job_id, nkeys, agent=None):
        self.job_id = job_id
        self.keys = list(range(nkeys))
        self.req_context = type(
            "C", (), {"kv_transfer_params": {"genesis_agent": agent} if agent else None}
        )()


class _FakeMgr:
    _block_size = 1000


class _FakeResult:
    def __init__(self, job_id, success=True):
        self.job_id = job_id
        self.success = success


def test_note_y_finish_registran_bytes_ops_y_latencia(sink):
    M = _ktm()
    mgr = _FakeMgr()
    M.note_job(mgr, _FakeJobMeta(1, 3, "coder"), "write")
    out = M.finish_job(mgr, _FakeResult(1))
    assert isinstance(out, _FakeResult), "finish_job debe ser transparente"
    d = sink.drain()
    k = "kv_tier_bytes_total|agent=coder,direction=write,tier=disk"
    assert d["counters"][k] == 3000
    assert d["counters"]["kv_tier_ops_total|agent=coder,direction=write,tier=disk"] == 1
    assert len(d["hist"]["kv_tier_op_seconds|direction=write,tier=disk"]) == 1


def test_job_fallido_cuenta_error_y_no_bytes(sink):
    M = _ktm()
    mgr = _FakeMgr()
    M.note_job(mgr, _FakeJobMeta(7, 2, "coder"), "read")
    M.finish_job(mgr, _FakeResult(7, success=False))
    d = sink.drain()
    assert d["counters"]["kv_tier_errors_total|direction=read,tier=disk"] == 1
    assert not any(k.startswith("kv_tier_bytes_total") for k in d["counters"])


def test_finish_sin_note_no_explota(sink):
    """El pool puede reportar un job cuyo note se perdio (p.ej. tras un clear)."""
    M = _ktm()
    assert M.finish_job(_FakeMgr(), _FakeResult(999)) is not None
    assert sink.drain() is None


def test_jobs_pendientes_tienen_techo(sink):
    """Red de seguridad: jobs que nunca terminan no pueden hacer crecer el dict."""
    M = _ktm()
    mgr = _FakeMgr()
    for i in range(5000):
        M.note_job(mgr, _FakeJobMeta(i, 1, "coder"), "write")
    assert len(mgr._genesis_pn88_jobs) <= 4096


@pytest.mark.parametrize(
    "resultado,label", [(True, "hit"), (None, "inflight"), (False, "miss")]
)
def test_note_lookup_clasifica(sink, resultado, label):
    _ktm().note_lookup("disk", resultado)
    key = f"kv_tier_lookups_total|group=unknown,result={label},tier=disk"
    assert sink.drain()["counters"][key] == 1


def test_note_lookup_etiqueta_el_grupo_de_kv_cache(sink):
    """La label `group` sale de los 4 bytes finales de la OffloadKey.

    Sin ella el hit rate del disco queda inservible con PN93 activo: los
    bloques recurrentes salteados a propósito cuentan como `miss`, y en este
    híbrido son 3 de los 4 grupos.
    """
    M = _ktm()
    attn_key = b"\xaa" * 32 + (3).to_bytes(4, "big")
    gdn_key = b"\xbb" * 32 + (0).to_bytes(4, "big")

    M.note_lookup("disk", True, attn_key)
    M.note_lookup("disk", False, gdn_key)

    counters = sink.drain()["counters"]
    assert counters["kv_tier_lookups_total|group=3,result=hit,tier=disk"] == 1
    assert counters["kv_tier_lookups_total|group=0,result=miss,tier=disk"] == 1


def test_note_lookup_tolera_una_clave_invalida(sink):
    """Una métrica nunca puede tumbar el lookup."""
    _ktm().note_lookup("disk", True, object())
    counters = sink.drain()["counters"]
    assert any("group=unknown" in k for k in counters)


# ───────── bytes reales al NVMe (la serie de desgaste del SSD) ─────────


def test_disk_write_cuenta_bytes_del_syscall(sink):
    """`kv_tier_disk_written_bytes_total` es lo único válido para desgaste.

    `kv_tier_bytes_total{direction=write}` se calcula al encolar y se desvía por
    dos motivos: store_block saltea el archivo si ya existe, y con PN92 se
    escribe el bloque comprimido.
    """
    M = _ktm()
    M.note_disk_write(4096)
    M.note_disk_write(8192)
    counters = sink.drain()["counters"]
    assert counters["kv_tier_disk_written_bytes_total|tier=disk"] == 12288
    assert counters["kv_tier_disk_write_syscalls_total|tier=disk"] == 2


def test_disk_read_cuenta_bytes_del_syscall(sink):
    M = _ktm()
    M.note_disk_read(1024)
    counters = sink.drain()["counters"]
    assert counters["kv_tier_disk_read_bytes_total|tier=disk"] == 1024
    assert counters["kv_tier_disk_read_syscalls_total|tier=disk"] == 1


@pytest.mark.parametrize("valor", [0, -1, None])
def test_disk_counters_ignoran_valores_no_positivos(sink, valor):
    M = _ktm()
    M.note_disk_write(valor)
    M.note_disk_read(valor)
    assert sink.drain() is None


def test_trim_jobs_descarta_los_viejos_no_todo(sink):
    """Regresión: antes era `jobs.clear()` y se perdían TODAS las mediciones.

    Con PN90 la democión encola un job por bloque, así que hay muchos más jobs
    vivos y un clear() se comía un pico entero de escrituras.
    """
    M = _ktm()
    mgr = _FakeMgr()
    for i in range(4200):
        M.note_demote(mgr, job_id=-i - 1, nbytes=100, agent="primary")
    jobs = mgr._genesis_pn88_jobs
    assert len(jobs) <= 4096
    # los MÁS RECIENTES sobreviven
    assert -4200 in jobs


def test_promotion_refused_tiene_contador_propio(sink):
    """No puede confundirse con un miss: el efecto es truncar todo el prefijo."""
    _ktm().note_promotion_refused("disk")
    assert sink.drain()["counters"]["kv_tier_promotion_refused_total|tier=disk"] == 1


def test_eviction_registra_conteo_y_bytes(sink):
    _ktm().note_eviction("disk", "quota", count=236, freed_bytes=6 * 1024**3)
    d = sink.drain()["counters"]
    assert d["kv_tier_evictions_total|reason=quota,tier=disk"] == 236
    assert d["kv_tier_evicted_bytes_total|reason=quota,tier=disk"] == 6 * 1024**3


def test_occupancy_es_gauge(sink):
    M = _ktm()
    M.set_occupancy("disk", 27e9, 32e9)
    M.set_occupancy("disk", 25e9, 32e9)
    g = sink.drain()["gauges"]
    assert g["kv_tier_bytes_used|tier=disk"] == 25e9, "gana la ultima medicion"
    assert g["kv_tier_capacity_bytes|tier=disk"] == 32e9


def test_helpers_son_noop_con_la_env_apagada(monkeypatch):
    M = _ktm()
    monkeypatch.delenv("GENESIS_ENABLE_PN88_KV_TIER_METRICS", raising=False)
    s = M.TierStatsSink()
    monkeypatch.setattr(M, "_SINK", s, raising=False)
    M.note_lookup("disk", True)
    M.note_promotion_refused("disk")
    M.note_eviction("disk", "quota")
    M.set_occupancy("disk", 1)
    assert s.drain() is None, "con PN88 apagado no se acumula nada"


# ───────────────────────── Prometheus ─────────────────────────


def test_observe_bucket_emite_las_tres_familias(sink):
    prom = pytest.importorskip("prometheus_client")
    import types

    M = _ktm()
    reg = prom.CollectorRegistry()

    def mk(cls):
        def f(name, documentation, labelnames, **kw):
            return cls(name.replace(":", "_"), documentation, labelnames,
                       registry=reg, **kw)
        return f

    fake = types.SimpleNamespace(
        _counter_cls=mk(prom.Counter), _gauge_cls=mk(prom.Gauge),
        _histogram_cls=mk(prom.Histogram),
        _labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["qwen3.8", "0"]},
    )

    sink.inc("kv_tier_bytes_total",
             (("tier", "disk"), ("direction", "write"), ("agent", "coder")), 100)
    sink.observe("kv_tier_op_seconds", (("tier", "disk"), ("direction", "write")), 0.04)
    sink.set("kv_tier_bytes_used", (("tier", "disk"),), 27e9)
    M.observe_bucket(fake, sink.drain(), 0)

    out = prom.generate_latest(reg).decode()
    assert 'vllm_kv_tier_bytes_total{agent="coder"' in out
    assert "vllm_kv_tier_op_seconds_count" in out
    assert "vllm_kv_tier_bytes_used" in out

    # los contadores acumulan entre drains; el gauge pisa
    sink.inc("kv_tier_bytes_total",
             (("tier", "disk"), ("direction", "write"), ("agent", "coder")), 50)
    sink.set("kv_tier_bytes_used", (("tier", "disk"),), 25e9)
    M.observe_bucket(fake, sink.drain(), 0)
    out2 = prom.generate_latest(reg).decode()
    assert "150.0" in out2, "el contador debe acumular"
    # El gauge pisa: prometheus_client >= 3 renderiza 25e9 estilo Go
    # ("2.5e+10"); versiones viejas usaban exponente zero-padded
    # ("2.5e+010") o repr Python ("25000000000.0").
    assert (
        "2.5e+10" in out2 or "2.5e+010" in out2 or "25000000000.0" in out2
    ), "el gauge debe pisar"
    # Y NO acumuló con la medicion anterior (27e9 + 25e9 = 52e9).
    assert (
        "5.2e+10" not in out2 and "5.2e+010" not in out2
        and "52000000000.0" not in out2
    ), "el gauge no debe acumular entre drains"


def test_observe_bucket_nunca_lanza():
    """Una metrica rota no puede tumbar el logger."""
    _ktm().observe_bucket(object(), {"counters": {"x": 1}}, 0)


def test_reduce_for_log_resume_histogramas():
    M = _ktm()
    out = M.reduce_for_log(
        {"counters": {"c": 5}, "gauges": {"g": 7}, "hist": {"h": [1.0, 3.0]}}
    )
    assert out["c"] == 5 and out["g"] == 7
    assert out["h#n"] == 2 and out["h#avg"] == 2.0


# ─────────────────────── superficie del wiring ───────────────────────


def test_bucket_no_puede_chocar_con_un_transfer_type():
    """Los transfer_type reales siempre son '<SRC>_to_<DST>'."""
    M = _ktm()
    assert M.GENESIS_BUCKET.startswith("_")
    assert "_to_" not in M.GENESIS_BUCKET


def test_la_constante_del_bucket_coincide_en_modulo_y_wiring():
    """D0 inyecta el literal en metrics.py; tiene que ser el mismo string."""
    assert f'_G88_BUCKET = "{_ktm().GENESIS_BUCKET}"' in _wiring().D0_NEW


def test_anclas_apuntan_a_los_sitios_correctos():
    W = _wiring()
    assert "enqueue_store" in W.A1_OLD
    assert "enqueue_load" in W.A2_OLD
    assert "JobResult(job_id=job_id, success=success)" in W.A3_OLD
    assert "primary_hit = self.primary_tier.lookup" in W.B1_OLD
    assert "_initiate_promotion" in W.B2_OLD
    assert "def get_kv_connector_stats(self) -> KVConnectorStats | None:" in W.C_OLD
    assert "other_values.items()" in W.D1_OLD
    assert "Unknown offloading stats key" in W.D2_OLD


def test_cada_reemplazo_conserva_el_codigo_original():
    """PN88 es observacion pura: agrega llamadas, no reescribe logica."""
    W = _wiring()
    for old, new in [
        (W.A1_OLD, W.A1_NEW), (W.A2_OLD, W.A2_NEW),
        (W.B1_OLD, W.B1_NEW), (W.B2_OLD, W.B2_NEW),
        (W.D1_OLD, W.D1_NEW), (W.D2_OLD, W.D2_NEW), (W.D3_OLD, W.D3_NEW),
    ]:
        for line in old.strip().splitlines():
            assert line.strip() in new, f"el reemplazo perdio: {line.strip()!r}"


def test_a3_envuelve_sin_cambiar_la_semantica():
    W = _wiring()
    assert "finish_job(self, JobResult(job_id=job_id, success=success))" in W.A3_NEW
    assert "_g88.finish_job(self, JobResult(job_id=job_id, success=success))" in W.A3_NEW


def test_c_preserva_el_camino_del_worker():
    W = _wiring()
    assert "return _g88_stats" in W.C_NEW
    assert "_g88.sink().drain()" in W.C_NEW


def test_todas_las_anclas_llevan_marker_o_lo_hereda_el_patcher():
    W = _wiring()
    assert W.GENESIS_PN88_MARKER in W.A1_NEW
    assert W.GENESIS_PN88_MARKER in W.B1_NEW
    assert W.GENESIS_PN88_MARKER in W.C_NEW
    assert W.GENESIS_PN88_MARKER in W.D0_NEW


def test_registrado_en_el_dispatcher():
    from vllm._genesis.dispatcher import PATCH_REGISTRY

    entry = PATCH_REGISTRY["PN88"]
    assert entry["env_flag"] == "GENESIS_ENABLE_PN88_KV_TIER_METRICS"
    assert entry["default_on"] is False, "observabilidad: opt-in"
    assert entry["category"] == "observability"


def test_apply_saltea_con_la_env_apagada(monkeypatch):
    monkeypatch.delenv("GENESIS_ENABLE_PN88_KV_TIER_METRICS", raising=False)
    status, reason = _wiring().apply()
    assert status == "skipped"
    assert "GENESIS_ENABLE_PN88_KV_TIER_METRICS" in reason


def test_el_tier_fs_se_normaliza_a_disk(sink):
    """El tiering manager pasa `tier.tier_type` ("fs"); PN81 dice "disk".

    Sin normalizar, el mismo tier salía en dos series que no se podían cruzar:
    kv_tier_lookups_total{tier="fs"} contra kv_tier_evictions_total{tier="disk"}.
    Visto en producción el 2026-08-21, en el primer arranque con PN88.
    """
    M = _ktm()
    assert M.tier_name("fs") == "disk"
    assert M.tier_name("ram") == "ram"
    M.note_lookup("fs", True)
    M.note_promotion_refused("fs")
    M.note_eviction("fs", "quota", 3)
    M.set_occupancy("fs", 100, 200)
    d = sink.drain()
    tiers = {k.split("tier=")[1].split(",")[0] for k in
             list(d["counters"]) + list(d["gauges"])}
    assert tiers == {"disk"}, tiers
