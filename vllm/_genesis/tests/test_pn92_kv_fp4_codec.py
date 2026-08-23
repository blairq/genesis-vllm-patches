# SPDX-License-Identifier: Apache-2.0
"""Tests de PN92 — codec de 4 bits agrupado del tier de disco.

Estos tests ejercen el camino que REALMENTE está cableado (numpy sobre bytes de
disco). Los tests anteriores probaban `pack_fp8_to_fp4_gpu` sobre gaussianas
float32 —otra implementación, otra semántica, otro dtype— mientras la función
que corría en producción quedaba sin cobertura numérica.
"""

from __future__ import annotations

import struct

import pytest

np = pytest.importorskip("numpy")


def _codec():
    from vllm._genesis import kv_fp4_codec as C

    return C


@pytest.fixture(autouse=True)
def fp8_dtype(monkeypatch):
    monkeypatch.setenv("GENESIS_PN92_KV_DTYPE", "fp8_e4m3")
    monkeypatch.delenv("GENESIS_PN92_COMPRESS_ON_WRITE", raising=False)
    yield


def _fp8_block(nbytes: int, seed: int = 0) -> bytes:
    """Bloque sintético con la forma de un KV fp8-e4m3 real.

    Se generan valores gaussianos y se los proyecta al código E4M3 más cercano,
    en vez de sortear bytes uniformes: un byte al azar cae con la misma
    probabilidad en exponentes que jamás aparecen en un KV.
    """
    C = _codec()
    _, encode_lut = C._luts()
    rng = np.random.default_rng(seed)
    vals = rng.normal(0.0, 2.0, size=nbytes).astype(np.float16)
    return encode_lut[vals.view(np.uint16)].tobytes()


def _as_values(raw: bytes) -> np.ndarray:
    decode_lut, _ = _codec()._luts()
    return decode_lut[np.frombuffer(raw, dtype=np.uint8)]


# ─────────────────── LUTs E4M3 ───────────────────


def test_decode_lut_matches_the_e4m3_definition():
    decode_lut, _ = _codec()._luts()
    assert decode_lut[0x00] == 0.0
    assert decode_lut[0x38] == pytest.approx(1.0)  # exp=7, man=0 → 2^0
    assert decode_lut[0xB8] == pytest.approx(-1.0)
    assert decode_lut[0x3C] == pytest.approx(1.5)  # exp=7, man=4 → 1.5
    assert decode_lut[0x7E] == pytest.approx(448.0)  # máximo normal finito
    assert decode_lut[0x7F] == 0.0  # NaN neutralizado
    assert decode_lut[0x01] == pytest.approx(2.0**-9)  # subnormal mínimo


def test_encode_lut_is_the_inverse_on_representable_values():
    """Todo código E4M3 finito tiene que sobrevivir un ida y vuelta exacto."""
    decode_lut, encode_lut = _codec()._luts()
    codes = np.arange(256, dtype=np.uint8)
    finite = (codes & 0x7F) != 0x7F
    half = decode_lut[codes].astype(np.float16).view(np.uint16)
    back = encode_lut[half]
    assert np.array_equal(
        decode_lut[back[finite]], decode_lut[codes[finite]]
    ), "el ida y vuelta E4M3 no es exacto"


# ─────────────────── formato en disco ───────────────────


def test_compressed_length_is_o_direct_aligned():
    """Regresión CRÍTICA: la longitud tiene que ser múltiplo de 4096.

    La versión anterior escribía `18 + n_scales*2 + n/2` bytes. `os.write` con
    O_DIRECT exige longitud alineada, así que fallaba con EINVAL en el 100% de
    los casos y ninguna escritura llegaba al SSD.
    """
    C = _codec()
    for nbytes in (65_536, 262_144, 1_000_000):
        out = C.compress_block(_fp8_block(nbytes))
        assert out is not None
        assert len(out) % 4096 == 0, f"longitud {len(out)} no alineada a 4096"


def test_compressed_buffer_is_page_aligned():
    """El buffer también tiene que estar alineado en memoria para O_DIRECT."""
    C = _codec()
    out = C.compress_block(_fp8_block(65_536))
    addr = np.frombuffer(out, dtype=np.uint8).ctypes.data
    assert addr % 4096 == 0, f"buffer no alineado (dirección {addr})"


def test_header_roundtrips():
    C = _codec()
    raw = _fp8_block(65_536)
    out = bytes(C.compress_block(raw))
    assert out.startswith(C.FP4_MAGIC_HEADER)
    off = len(C.FP4_MAGIC_HEADER)
    codec, dtype, orig_len, group_size, num_groups = struct.unpack(
        "<BBQII", out[off : off + struct.calcsize("<BBQII")]
    )
    assert codec == 0
    assert dtype == 0  # fp8_e4m3
    assert orig_len == len(raw)
    assert num_groups == (len(raw) + group_size - 1) // group_size


def test_magic_detection_is_independent_of_the_enable_flag(monkeypatch, tmp_path):
    """Regresión: apagar la compresión no puede destruir el caché en disco.

    Antes, con la env apagada, un archivo comprimido se leía como KV crudo →
    short read → `load_block` lo BORRABA.
    """
    C = _codec()
    raw = _fp8_block(65_536)
    path = tmp_path / "bloque.bin"
    path.write_bytes(bytes(C.compress_block(raw)))

    monkeypatch.setenv("GENESIS_PN92_COMPRESS_ON_WRITE", "0")
    assert C.is_fp4_compression_enabled() is False
    assert C.probe_compressed(str(path)) is True, (
        "un archivo comprimido tiene que reconocerse aunque la compresión esté apagada"
    )

    target = memoryview(bytearray(len(raw)))
    assert C.load_compressed_block(str(path), target) == len(raw)
    assert path.exists(), "no debe borrar un archivo válido"


def test_raw_block_is_not_mistaken_for_compressed(tmp_path):
    path = tmp_path / "crudo.bin"
    path.write_bytes(_fp8_block(4096))
    assert _codec().probe_compressed(str(path)) is False


def test_missing_file_probes_false(tmp_path):
    assert _codec().probe_compressed(str(tmp_path / "no-existe.bin")) is False


def test_corrupt_compressed_block_is_removed(tmp_path):
    C = _codec()
    path = tmp_path / "roto.bin"
    path.write_bytes(C.FP4_MAGIC_HEADER + b"\x00" * 8)
    with pytest.raises((ValueError, struct.error)):
        C.load_compressed_block(str(path), memoryview(bytearray(4096)))
    assert not path.exists(), "un bloque corrupto se borra para que el hit sea un miss limpio"


def test_crc_detecta_corrupcion_silenciosa():
    """Sin CRC, un bloque corrupto se descomprime a basura PLAUSIBLE.

    El camino de lectura cruda al menos daba short read; el comprimido no tenía
    forma de notarlo y la basura entraba al KV.
    """
    C = _codec()
    raw = _fp8_block(65_536)
    out = bytearray(bytes(C.compress_block(raw)))
    out[C._HEADER_LEN + 40] ^= 0xFF          # un bit dado vuelta en el payload
    with pytest.raises(ValueError, match="CRC"):
        C.decompress_block(bytes(out), memoryview(bytearray(len(raw))))


def test_crc_cubre_todo_el_payload():
    C = _codec()
    raw = _fp8_block(65_536)
    base = bytes(C.compress_block(raw))
    payload_end = len(base)
    for pos in (C._HEADER_LEN, C._HEADER_LEN + 1, payload_end // 2):
        out = bytearray(base)
        out[pos] ^= 0x01
        with pytest.raises(ValueError):
            C.decompress_block(bytes(out), memoryview(bytearray(len(raw))))


@pytest.mark.parametrize("chunk", [64, 1024, 1 << 20])
def test_el_chunking_no_cambia_el_resultado(monkeypatch, chunk):
    """Procesar por bloques de trabajo tiene que dar bit a bit lo mismo.

    Los bordes son el riesgo: un grupo partido entre dos chunks, o un
    empaquetado de nibbles que cruza el borde.
    """
    C = _codec()
    raw = _fp8_block(100_000, seed=5)        # no es multiplo del chunk ni del grupo

    monkeypatch.setenv("GENESIS_PN92_CHUNK_ELEMS", str(1 << 30))
    ref = memoryview(bytearray(len(raw)))
    C.decompress_block(C.compress_block(raw), ref)

    monkeypatch.setenv("GENESIS_PN92_CHUNK_ELEMS", str(chunk))
    got = memoryview(bytearray(len(raw)))
    C.decompress_block(C.compress_block(raw), got)

    assert bytes(got) == bytes(ref), f"el chunking de {chunk} cambia el resultado"


def test_chunk_siempre_multiplo_del_grupo():
    """Un grupo partido entre dos chunks daría escalas mal calculadas."""
    C = _codec()
    for gs in (16, 32, 64):
        for env in ("1", "33", "1000", "1048576"):
            import os as _os

            _os.environ["GENESIS_PN92_CHUNK_ELEMS"] = env
            n = C.chunk_elems(gs)
            assert n % gs == 0 and n >= gs
            assert n % 2 == 0, "el empaquetado de nibbles necesita paridad"
    import os as _os
    _os.environ.pop("GENESIS_PN92_CHUNK_ELEMS", None)


def test_legacy_v1_is_recognised_and_rejected():
    """Los archivos del formato v1 eran irreconstruibles: no deben leerse como v2."""
    C = _codec()
    for magic in (b"GFP4\x01\x00", b"GKV4\x02\x00"):
        assert C.is_legacy_v1(magic + b"\x00" * 16) is True
        assert C.is_compressed(magic + b"\x00" * 16) is False


# ─────────────────── fidelidad numérica ───────────────────


def test_roundtrip_error_is_within_budget():
    """El camino cableado tiene que cumplir el umbral que el proyecto declara.

    La implementación anterior daba 40,4% de error L2 sobre un bloque real: ni
    de cerca. Su test no lo detectaba porque medía la otra función.
    """
    C = _codec()
    raw = _fp8_block(1 << 20, seed=7)
    out = C.compress_block(raw)
    target = memoryview(bytearray(len(raw)))
    C.decompress_block(out, target)

    orig = _as_values(raw)
    back = _as_values(bytes(target))

    l2 = float(np.linalg.norm(back - orig) / np.linalg.norm(orig))
    cos = float(
        (orig * back).sum() / (np.linalg.norm(orig) * np.linalg.norm(back))
    )
    assert l2 < 0.20, f"error L2 relativo demasiado alto: {l2:.4f}"
    assert cos > 0.98, f"similitud coseno demasiado baja: {cos:.4f}"


def test_smaller_groups_reduce_the_error():
    """Propiedad de la cuantización agrupada: menos elementos por escala, menos error."""
    C = _codec()
    raw = _fp8_block(1 << 19, seed=11)
    orig = _as_values(raw)

    errores = []
    for group_size in (16, 64, 128):
        out = C.compress_block(raw, group_size=group_size)
        target = memoryview(bytearray(len(raw)))
        C.decompress_block(out, target)
        errores.append(
            float(np.linalg.norm(_as_values(bytes(target)) - orig) / np.linalg.norm(orig))
        )
    assert errores == sorted(errores), f"el error no crece con el grupo: {errores}"


def test_zero_block_roundtrips_exactly():
    """Un bloque de ceros no puede generar escalas NaN ni ensuciarse."""
    C = _codec()
    raw = bytes(65_536)
    out = C.compress_block(raw)
    target = memoryview(bytearray(len(raw)))
    C.decompress_block(out, target)
    assert _as_values(bytes(target)).sum() == 0.0


def test_compression_ratio_beats_lossless():
    """Tiene que ganarle a zstd-3 sobre KV real (1,10-1,24×), o no vale la pena."""
    C = _codec()
    raw = _fp8_block(1 << 20)
    out = C.compress_block(raw)
    ratio = len(out) / len(raw)
    assert ratio < 0.65, f"ratio {ratio:.3f}: no supera a una compresión sin pérdida"


# ─────────────────── condiciones de borde ───────────────────


def test_unknown_dtype_disables_compression(monkeypatch):
    """Ante un dtype no reconocido, mejor no comprimir que malinterpretar bytes."""
    monkeypatch.setenv("GENESIS_PN92_KV_DTYPE", "bfloat16-inventado")
    assert _codec().compress_block(_fp8_block(4096)) is None


def test_float16_mode_roundtrips(monkeypatch):
    monkeypatch.setenv("GENESIS_PN92_KV_DTYPE", "float16")
    C = _codec()
    rng = np.random.default_rng(3)
    raw = rng.normal(0, 2, size=1 << 16).astype(np.float16).tobytes()

    out = C.compress_block(raw)
    target = memoryview(bytearray(len(raw)))
    C.decompress_block(out, target)

    orig = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    back = np.frombuffer(bytes(target), dtype=np.float16).astype(np.float32)
    l2 = float(np.linalg.norm(back - orig) / np.linalg.norm(orig))
    assert l2 < 0.20, f"error L2 en modo float16: {l2:.4f}"


def test_empty_block_is_not_compressed():
    assert _codec().compress_block(b"") is None


def test_incompressible_input_returns_none():
    """Si el payload no achica nada, se escribe el bloque crudo."""
    C = _codec()
    assert C.compress_block(_fp8_block(64)) is None


def test_decompress_rejects_a_short_target():
    C = _codec()
    raw = _fp8_block(65_536)
    out = C.compress_block(raw)
    with pytest.raises(ValueError):
        C.decompress_block(out, memoryview(bytearray(1024)))
