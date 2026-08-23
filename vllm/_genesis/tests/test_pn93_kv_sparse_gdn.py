# SPDX-License-Identifier: Apache-2.0
"""Tests de PN93 — checkpointing disperso del estado recurrente (GDN/Mamba)."""

from __future__ import annotations

import pytest


def _g93():
    from vllm._genesis import kv_sparse_gdn as G

    return G


class _MambaSpecFalso:
    """Duck-type de MambaSpec: lo distingue el atributo `mamba_type`."""

    mamba_type = "gated_delta_net"
    page_size_bytes = 14_483_456


class _FullAttentionSpecFalso:
    page_size_bytes = 14_483_456


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    G = _g93()
    G.reset_state()
    monkeypatch.delenv("GENESIS_PN93_GDN_CHECKPOINT_STRIDE", raising=False)
    monkeypatch.delenv("GENESIS_ENABLE_PN93_SPARSE_GDN", raising=False)
    yield
    G.reset_state()


def _registrar_layout_qwen38():
    """El layout REAL, leído de /kv-offload/.../config.json.

    g0, g1, g2 = `linear_attn` = GDN recurrente
    g3         = `self_attn.attn` + mtp = atención completa

    La primera versión de PN93 tenía esto exactamente al revés: salteaba `_g3`
    creyendo que era el grupo GDN.
    """
    G = _g93()
    for idx in (0, 1, 2):
        G.register_group_spec(idx, _MambaSpecFalso())
    G.register_group_spec(3, _FullAttentionSpecFalso())


# ─────────────── identificación de grupos ───────────────


def test_recurrent_groups_are_detected_by_spec_not_by_path():
    _registrar_layout_qwen38()
    G = _g93()
    assert G.is_recurrent_group(0) is True
    assert G.is_recurrent_group(1) is True
    assert G.is_recurrent_group(2) is True
    assert G.is_recurrent_group(3) is False, (
        "g3 es self_attn.attn (atención + MTP), no GDN"
    )


def test_attention_group_is_never_skipped():
    """El grupo de atención se guarda entero: cada bloque puede servir un hit."""
    _registrar_layout_qwen38()
    G = _g93()
    for idx in range(20):
        assert G.should_skip_recurrent_block(3, idx, 20) is False


def test_unregistered_group_is_never_skipped():
    """Sin registro no se saltea nada: fallar hacia el lado seguro."""
    assert _g93().should_skip_recurrent_block(0, 5, 20) is False


# ─────────────── lógica de stride ───────────────


def test_stride_keeps_one_of_every_n(monkeypatch):
    _registrar_layout_qwen38()
    G = _g93()
    monkeypatch.setenv("GENESIS_PN93_GDN_CHECKPOINT_STRIDE", "4")

    total = 20
    kept = [i for i in range(total) if not G.should_skip_recurrent_block(0, i, total)]
    assert kept == [0, 4, 8, 12, 16, 19]


def test_stride_efectivo_bajo_prefill_fragmentado(monkeypatch):
    """REGRESIÓN: el stride efectivo tiene que respetar al configurado.

    El bug: se pasaba el `num_blocks` del PASO, que crece con el prefill
    fragmentado, así que "conservar el último" disparaba una vez por paso.
    Con la config real (prompt 40k, bloque 832, chunk 1664 = 2 bloques/paso,
    stride 4) se conservaban 36/48 bloques -> stride efectivo 1,33, y el ahorro
    caía de 2,2x a 1,23x.

    Este test simula el bucle real de `_build_store_jobs`: varios pasos, cada
    uno procesando su lote, con el total del prompt como referencia fija.
    """
    _registrar_layout_qwen38()
    G = _g93()
    monkeypatch.setenv("GENESIS_PN93_GDN_CHECKPOINT_STRIDE", "4")

    BLOQUE, CHUNK, PROMPT = 832, 1664, 40000
    total = PROMPT // BLOQUE                 # 48 bloques
    por_paso = CHUNK // BLOQUE               # 2 bloques por paso

    kept, start = [], 0
    while start < total:
        fin = min(start + por_paso, total)
        for idx in range(start, fin):
            # tercer argumento = total del PROMPT, estable entre pasos
            if not G.should_skip_recurrent_block(0, idx, total):
                kept.append(idx)
        start = fin

    stride_efectivo = total / len(kept)
    assert stride_efectivo > 3.0, (
        f"stride efectivo {stride_efectivo:.2f} con stride configurado 4: "
        f"se conservaron {len(kept)}/{total} bloques ({kept[:12]}...). "
        f"Sintoma clasico de estar pasando el num_blocks del paso en vez del "
        f"total del prompt."
    )
    assert kept[:6] == [0, 4, 8, 12, 16, 20]


def test_bloques_de_decodificacion_no_se_conservan_todos(monkeypatch):
    """Con offload_prompt_only=False los índices pasan el final del prompt.

    Con `>=` en vez de igualdad exacta, TODOS esos bloques quedaban marcados
    como "último" y se conservaban.
    """
    _registrar_layout_qwen38()
    G = _g93()
    monkeypatch.setenv("GENESIS_PN93_GDN_CHECKPOINT_STRIDE", "4")

    total_prompt = 10
    mas_alla = [i for i in range(total_prompt, total_prompt + 8)
                if not G.should_skip_recurrent_block(0, i, total_prompt)]
    assert mas_alla == [12, 16], "pasado el prompt manda solo el stride"


def test_last_block_is_always_kept(monkeypatch):
    """El último bloque del prompt es el punto de reanudación del turno siguiente."""
    _registrar_layout_qwen38()
    G = _g93()
    monkeypatch.setenv("GENESIS_PN93_GDN_CHECKPOINT_STRIDE", "8")
    for total in (5, 9, 17, 33):
        last = total - 1
        assert G.should_skip_recurrent_block(0, last, total) is False


def test_stride_one_disables_skipping(monkeypatch):
    _registrar_layout_qwen38()
    G = _g93()
    monkeypatch.setenv("GENESIS_PN93_GDN_CHECKPOINT_STRIDE", "1")
    for idx in range(20):
        assert G.should_skip_recurrent_block(0, idx, 20) is False


def test_disabled_by_env(monkeypatch):
    _registrar_layout_qwen38()
    G = _g93()
    monkeypatch.setenv("GENESIS_ENABLE_PN93_SPARSE_GDN", "0")
    assert G.should_skip_recurrent_block(0, 5, 20) is False


def test_invalid_stride_falls_back(monkeypatch):
    monkeypatch.setenv("GENESIS_PN93_GDN_CHECKPOINT_STRIDE", "no-es-un-numero")
    assert _g93().checkpoint_stride() == 4


# ─────────── semántica de redondeo hacia abajo ───────────


def test_hit_rounds_down_to_the_previous_checkpoint(monkeypatch):
    """Modela `_sliding_window_lookup(window=1)`, que escanea desde el final.

    Si falta el bloque k, el lookup se queda con el último presente por debajo.
    Ése es el motivo por el que saltear intermedios es correcto: nunca se lee un
    estado equivocado, sólo se recomputa la cola.
    """
    _registrar_layout_qwen38()
    G = _g93()
    monkeypatch.setenv("GENESIS_PN93_GDN_CHECKPOINT_STRIDE", "4")

    num_blocks = 40
    en_disco = {
        i
        for i in range(num_blocks)
        if not G.should_skip_recurrent_block(0, i, num_blocks)
    }

    def hit_gdn(prefijo_disponible: int) -> int:
        """Último checkpoint presente dentro del prefijo (window=1, desde el final)."""
        for idx in range(prefijo_disponible - 1, -1, -1):
            if idx in en_disco:
                return idx + 1
        return 0

    # Nunca sobrepasa lo disponible, y la pérdida está acotada por el stride.
    for prefijo in range(1, num_blocks + 1):
        h = hit_gdn(prefijo)
        assert h <= prefijo, "el hit no puede exceder el prefijo disponible"
        assert prefijo - h < 4, f"pérdida {prefijo - h} ≥ stride en prefijo={prefijo}"


# ─────────────── métricas ───────────────


def test_bytes_ahorrados_usan_el_tamano_real_del_bloque(monkeypatch):
    """REGRESIÓN: el contador de bytes iba 34x corto.

    Usaba `MambaSpec.page_size_bytes` (851.968 B en este modelo), que mide el
    ESTADO recurrente, cuando lo que `store_block` deja de escribir es el bloque
    entero en disco (28.966.912 B). El conteo de BLOQUES siempre estuvo bien.
    """
    G = _g93()
    BLOQUE_EN_DISCO = 28_966_912
    for idx in (0, 1, 2):
        G.register_group_spec(idx, _MambaSpecFalso(), BLOQUE_EN_DISCO)
    G.register_group_spec(3, _FullAttentionSpecFalso(), BLOQUE_EN_DISCO)
    monkeypatch.setenv("GENESIS_PN93_GDN_CHECKPOINT_STRIDE", "4")

    total = 20
    for i in range(total):
        G.should_skip_recurrent_block(0, i, total)

    st = G.get_sparse_gdn_stats()
    assert st["block_write_bytes"] == BLOQUE_EN_DISCO
    assert st["bytes_skipped"] == st["blocks_skipped"] * BLOQUE_EN_DISCO
    assert st["bytes_skipped"] != st["blocks_skipped"] * _MambaSpecFalso.page_size_bytes


def test_stats_caen_a_page_size_si_el_spec_no_informa(monkeypatch):
    """Sin `kv_bytes_per_offloaded_block` se usa page_size_bytes, no cero."""
    _registrar_layout_qwen38()
    G = _g93()
    monkeypatch.setenv("GENESIS_PN93_GDN_CHECKPOINT_STRIDE", "4")

    num_blocks = 20
    for i in range(num_blocks):
        G.should_skip_recurrent_block(0, i, num_blocks)

    stats = G.get_sparse_gdn_stats()
    assert stats["blocks_skipped"] == 14
    assert stats["bytes_skipped"] == 14 * _MambaSpecFalso.page_size_bytes
    assert stats["stride"] == 4
    assert stats["recurrent_groups"] == 3
    assert 0 < stats["skip_ratio_pct"] < 100


def test_note_hit_truncated_publica_el_costo(monkeypatch):
    """El costo del stride tiene que ser observable, no sólo el ahorro."""
    from vllm._genesis import kv_tier_metrics as M

    _registrar_layout_qwen38()
    G = _g93()
    monkeypatch.setenv("GENESIS_ENABLE_PN88_KV_TIER_METRICS", "1")
    sink = M.TierStatsSink()
    monkeypatch.setattr(M, "_SINK", sink, raising=False)

    G.note_hit_truncated(0, 1664)   # grupo recurrente
    G.note_hit_truncated(3, 832)    # atencion: miss genuino, no culpa de PN93
    G.note_hit_truncated(0, 0)      # se ignora

    counters = sink.drain()["counters"]
    assert counters["kv_tier_hit_truncated_tokens_total|group=0,recurrent=1"] == 1664
    assert counters["kv_tier_hit_truncated_tokens_total|group=3,recurrent=0"] == 832
    assert G.get_sparse_gdn_stats()["hit_truncated_tokens"] == 2496
