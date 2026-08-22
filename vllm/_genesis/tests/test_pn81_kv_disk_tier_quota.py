# SPDX-License-Identifier: Apache-2.0
"""TDD for PN81 — cuota y poda del tier de disco del cache de KV.

El foco está en la **regresión del 2026-08-21**: la purga de directorios
abandonados corría una sola vez por proceso, gateada por un booleano que se
prendía en la primera pasada y no se apagaba nunca. Un directorio que
quedaba huérfano *después* del arranque no se revisaba jamás mientras el
engine siguiera vivo.

Caso real que lo destapó: al arrancar el 08-19, el directorio
`orcarouter_..._3ef784eef730_r0` llevaba 1,6 días sin uso — por debajo de
los 3 del umbral, así que se salteó con razón. Dos días después ya
calificaba y sus 25,47 GiB seguían ahí. Su hermano sin datos (solo
`config.json`) sí se había borrado en el arranque, porque nadie le escribe
después de crearlo y su mtime ya tenía 3,2 días: se borró la cáscara y quedó
el contenido, que es lo que hacía difícil ver el patrón.

El método parcheado se compila desde `ANCHOR_NEW` y se ejerce contra un
árbol de directorios real en `tmp_path`, que es la única forma de testear
comportamiento de un parche de texto sin un vLLM instalado.
"""
from __future__ import annotations

import ast
import os
import textwrap
import time
import types

import pytest

DAY = 86400.0


def _wiring():
    from vllm._genesis.wiring.hybrid import patch_N81_kv_disk_tier_quota as M
    return M


def _compile_on_schedule_end():
    """Compila el `on_schedule_end` inyectado dentro de una clase de prueba.

    `ANCHOR_NEW` termina re-emitiendo la cabecera de `shutdown` (es el ancla
    que reemplaza), así que se corta ahí antes de compilar.
    """
    src = textwrap.dedent(_wiring().ANCHOR_NEW)
    src = src[: src.index("@override\ndef shutdown")]
    ns: dict = {"override": lambda f: f}
    exec("class T:\n" + textwrap.indent(src, "    "), ns)
    return ns["T"]


@pytest.fixture
def tier(tmp_path, monkeypatch):
    """Manager de prueba con `file_mapper` apuntando a un árbol real."""
    T = _compile_on_schedule_end()
    obj = T()
    obj.file_mapper = types.SimpleNamespace(
        base_path=str(tmp_path / "modelo_AAAA"), rank=0
    )
    monkeypatch.setenv("GENESIS_ENABLE_PN81_KV_DISK_QUOTA", "1")
    monkeypatch.setenv("GENESIS_KV_DISK_MAX_GB", "999")  # cuota fuera de juego
    monkeypatch.setenv("GENESIS_KV_DISK_CHECK_SECS", "0")
    monkeypatch.setenv("GENESIS_KV_DISK_ORPHAN_DAYS", "3")
    monkeypatch.setenv("GENESIS_KV_DISK_ORPHAN_CHECK_SECS", "3600")
    obj._root = tmp_path
    return obj


def _mkdir(root, name, age_days, size=1024):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "block.bin"
    f.write_bytes(b"\0" * size)
    t = time.time() - age_days * DAY
    os.utime(f, (t, t))
    return d


def _age(d, days):
    f = d / "block.bin"
    t = time.time() - days * DAY
    os.utime(f, (t, t))


# ─────────────────── la regresión: purga periódica ───────────────────


def test_purga_al_arranque_lo_que_ya_vencio(tier):
    root = tier._root
    _mkdir(root, "modelo_AAAA_r0", 0.01)
    _mkdir(root, "modelo_BBBB_r0", 5.0)
    tier.on_schedule_end()
    assert not (root / "modelo_BBBB_r0").exists()
    assert (root / "modelo_AAAA_r0").exists(), "nunca borrar el directorio propio"


def test_no_purga_lo_que_todavia_no_vencio(tier):
    """El caso de `_3ef784eef730_r0` al arrancar: 1,6 dias, umbral 3."""
    root = tier._root
    _mkdir(root, "modelo_AAAA_r0", 0.01)
    _mkdir(root, "modelo_CCCC_r0", 1.6)
    tier.on_schedule_end()
    assert (root / "modelo_CCCC_r0").exists()


def test_purga_lo_que_vence_DESPUES_del_arranque(tier):
    """LA REGRESION. Con el gate booleano viejo esto no se borraba nunca."""
    root = tier._root
    _mkdir(root, "modelo_AAAA_r0", 0.01)
    tarde = _mkdir(root, "modelo_CCCC_r0", 1.6)

    tier.on_schedule_end()
    assert tarde.exists(), "todavia no vencio"

    _age(tarde, 4.0)  # cruza el umbral con el engine ya corriendo
    tier._genesis_pn81_orphan_last -= 3600.0  # pasa una hora
    tier.on_schedule_end()
    assert not tarde.exists(), "una vez vencido, la purga periodica debe borrarlo"


def test_respeta_la_cadencia_entre_barridos(tier):
    """No puede recorrer todos los directorios en cada tick de 60s."""
    root = tier._root
    _mkdir(root, "modelo_AAAA_r0", 0.01)
    tarde = _mkdir(root, "modelo_CCCC_r0", 1.6)
    tier.on_schedule_end()

    _age(tarde, 4.0)
    tier._genesis_pn81_orphan_last -= 600.0  # solo 10 minutos
    tier.on_schedule_end()
    assert tarde.exists(), "antes de la hora no debe volver a barrer"


def test_el_gate_booleano_no_sobrevive_en_el_codigo():
    """`_genesis_pn81_orphans` solo puede quedar en el comentario que lo explica."""
    src = textwrap.dedent(_wiring().ANCHOR_NEW)
    code = "\n".join(
        l for l in src.splitlines()
        if l.strip() and not l.strip().startswith("#")
    )
    assert "_genesis_pn81_orphans" not in code
    assert "_genesis_pn81_orphan_last" in code


# ─────────────────────── invariantes de la purga ───────────────────────


def test_no_borra_el_directorio_de_config_del_modelo_propio(tier):
    """`{base}` (sin `_r0`) comparte prefijo con `{base}_r0` y lleva config.json."""
    root = tier._root
    _mkdir(root, "modelo_AAAA_r0", 0.01)
    propio_cfg = _mkdir(root, "modelo_AAAA", 9.0)  # viejo pero es el nuestro
    tier.on_schedule_end()
    assert propio_cfg.exists()


def test_directorio_vacio_no_se_toca(tier):
    """Sin `_peso` no hay nada que liberar; borrarlo seria ruido."""
    root = tier._root
    _mkdir(root, "modelo_AAAA_r0", 0.01)
    vacio = root / "modelo_DDDD_r0"
    vacio.mkdir()
    tier.on_schedule_end()
    assert vacio.exists()


def test_apagado_por_env_no_hace_nada(tier, monkeypatch):
    monkeypatch.delenv("GENESIS_ENABLE_PN81_KV_DISK_QUOTA", raising=False)
    root = tier._root
    _mkdir(root, "modelo_AAAA_r0", 0.01)
    viejo = _mkdir(root, "modelo_BBBB_r0", 30.0)
    tier.on_schedule_end()
    assert viejo.exists()


def test_orphan_days_cero_desactiva_solo_la_purga(tier, monkeypatch):
    monkeypatch.setenv("GENESIS_KV_DISK_ORPHAN_DAYS", "0")
    root = tier._root
    _mkdir(root, "modelo_AAAA_r0", 0.01)
    viejo = _mkdir(root, "modelo_BBBB_r0", 30.0)
    tier.on_schedule_end()
    assert viejo.exists(), "ORPHAN_DAYS=0 debe desactivar la purga de huerfanos"


# ─────────────────────────── cuota por tamaño ───────────────────────────


def test_la_cuota_poda_hasta_el_ratio_objetivo(tier, monkeypatch):
    monkeypatch.setenv("GENESIS_KV_DISK_MAX_GB", str(4 / 1024**3 * 1024**3 / 1024**3))
    # 10 bloques de 1 KiB con mtimes escalonados; limite ~4 KiB, objetivo 85%
    monkeypatch.setenv("GENESIS_KV_DISK_MAX_GB", str(4096 / 1024**3))
    monkeypatch.setenv("GENESIS_KV_DISK_TARGET_RATIO", "0.85")
    root = tier._root
    d = root / "modelo_AAAA_r0"
    d.mkdir(parents=True)
    for i in range(10):
        f = d / f"b{i}.bin"
        f.write_bytes(b"\0" * 1024)
        t = time.time() - (10 - i) * 60
        os.utime(f, (t, t))
    tier.on_schedule_end()
    quedan = sorted(p.name for p in d.glob("*.bin"))
    total = sum(p.stat().st_size for p in d.glob("*.bin"))
    assert total <= int(4096 * 0.85), f"la cuota no podo lo suficiente: {total}"
    assert "b9.bin" in quedan, "debe conservar los mas nuevos"
    assert "b0.bin" not in quedan, "debe borrar los mas viejos primero"


def test_un_fallo_de_poda_no_tumba_el_scheduler(tier):
    """El scheduler llama a esto en cada step: no puede propagar excepciones."""
    tier.file_mapper = types.SimpleNamespace(base_path=None, rank=0)
    tier.on_schedule_end()  # no debe lanzar


def test_root_inexistente_sale_limpio(tier):
    tier.file_mapper = types.SimpleNamespace(
        base_path=str(tier._root / "no_existe"), rank=0
    )
    tier.on_schedule_end()  # no debe lanzar


# ───────────────────────── superficie del wiring ─────────────────────────


def test_anchor_new_compila():
    _compile_on_schedule_end()


def test_documenta_que_la_cuota_no_es_por_rank():
    """Correccion 2026-08-21: hay UN FileSystemTierManager, en el scheduler."""
    doc = _wiring().__doc__ or ""
    assert "NO es por rank" in doc


def test_registrado_en_el_dispatcher():
    from vllm._genesis.dispatcher import PATCH_REGISTRY

    entry = PATCH_REGISTRY["PN81"]
    assert entry["env_flag"] == "GENESIS_ENABLE_PN81_KV_DISK_QUOTA"
    assert entry["default_on"] is False
