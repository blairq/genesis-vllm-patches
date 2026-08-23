# SPDX-License-Identifier: Apache-2.0
"""Genesis PN92: cuantización agrupada a 4 bits para el tier de disco del KV cache.

Qué cambió respecto de la primera versión (auditoría 2026-08-23)
---------------------------------------------------------------
La versión anterior trataba los bytes crudos del KV como enteros lineales::

    f_vals = (grouped - 128.0) / 16.0     # "centered approximation for FP8"

FP8-E4M3 es punto flotante: 4 bits de exponente, 3 de mantisa, y el byte 0x80
es -0.0, no 0. Cuantizar el BYTE a 4 bits mueve el campo de exponente varias
posiciones. Medido sobre un bloque real de `/kv-offload`: 7% de bytes intactos,
**40,4% de error L2 relativo** en el espacio de valores FP8. Eso no es
compresión, es corrupción.

Esta versión decodifica E4M3 de verdad (LUT exacta de 256 entradas), cuantiza
los VALORES con escala por grupo, y reconstruye vía una LUT fp16→fp8 de 65536
entradas. Las dos direcciones son gathers O(n), sin búsquedas por elemento.

Nombre
------
El esquema es **INT4 simétrico con escala fp16 por grupo**, no FP4-E2M1. Con la
misma cantidad de bits, INT4+escala reconstruye mejor que E2M1 para datos
aproximadamente gaussianos, que es lo que hay en un KV cache. El módulo y la
env var conservan el nombre `fp4` por compatibilidad con el compose ya
desplegado; los nombres de las funciones dicen lo que realmente hacen.

Formato en disco (v2)
---------------------
El v1 era incompatible y además se escribía con longitudes no alineadas, lo que
hacía fallar `os.write` con `O_DIRECT` (EINVAL, verificado). El v2 rellena hasta
múltiplo de 4096::

    magic       6B   b"GKV4\\x02\\x00"
    codec       1B   0 = int4 agrupado
    dtype       1B   0 = fp8_e4m3, 1 = float16
    orig_len    8B   <Q   bytes originales del bloque
    group_size  4B   <I
    num_groups  4B   <I
    scales           num_groups × 2B (float16)
    packed           ceil(n_elems / 2) B
    padding          hasta múltiplo de 4096
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
import threading
import zlib


def _get_logger(name: str):
    """Logger que SÍ se ve desde dentro del EngineCore.

    vLLM configura sus handlers sobre los loggers que crea `init_logger`; un
    `logging.getLogger` pelado no propaga a ninguno, así que todo lo que se
    logueara acá era invisible en `docker logs` (verificado: las líneas de
    registro de grupos de PN93 no aparecían nunca). Se cae al logger estándar
    si vLLM no está disponible, para no romper los tests.
    """
    try:
        from vllm.logger import init_logger

        return init_logger(name)
    except Exception:
        return logging.getLogger(name)


log = _get_logger("genesis.pn92.kv_codec")

FP4_MAGIC_HEADER = b"GKV4\x03\x00"
# codec, dtype, orig_len, group_size, num_groups, crc32(payload)
_HEADER_STRUCT = "<BBQIII"
_HEADER_LEN = len(FP4_MAGIC_HEADER) + struct.calcsize(_HEADER_STRUCT)

# Formatos viejos irreconstruibles. Se reconocen SÓLO para no tratarlos como KV
# crudo: si cayeran al camino de lectura directa darían short read y
# `load_block` los borraría.
_LEGACY_MAGICS = (b"GFP4\x01\x00", b"GKV4\x02\x00")

_CODEC_INT4_GROUPED = 0
_DTYPE_FP8_E4M3 = 0
_DTYPE_FLOAT16 = 1

_ALIGN = 4096  # requisito de O_DIRECT para longitud y dirección del buffer

# 32 medido como el punto de equilibrio sobre bloques reales de /kv-offload
# (grupo → error L2 en g3 / ratio):  16 → 0,104 / 0,625 · 32 → 0,124 / 0,563
#                                    64 → 0,146 / 0,531 · 128 → 0,170 / 0,516
DEFAULT_GROUP_SIZE: int = int(os.environ.get("GENESIS_KV_FP4_GROUP_SIZE", "32"))

# Bytes que alcanzan para reconocer el magic sin leer el archivo entero.
HEADER_PROBE_LEN = 16

# Elementos por bloque de trabajo. La versión anterior materializaba el bloque
# entero en float32: 28 MB de KV -> 116 MB de f32, y con 16 hilos de escritura
# eso son ~1,8 GB de transitorio compitiendo con la L2 de 6 GiB. Procesando de a
# 1M elementos el pico baja a ~4 MB por hilo.
_DEFAULT_CHUNK_ELEMS = 1 << 20


def chunk_elems(group_size: int) -> int:
    """Elementos por bloque de trabajo, redondeado a múltiplo de `group_size`.

    Tiene que ser múltiplo del grupo para que ningún grupo quede partido entre
    dos bloques, y par para que el empaquetado de 2 nibbles por byte no cruce el
    borde. Con `group_size` par (32 por default) lo primero implica lo segundo.
    """
    raw = os.environ.get("GENESIS_PN92_CHUNK_ELEMS")
    n = _DEFAULT_CHUNK_ELEMS
    if raw:
        try:
            n = max(group_size, int(raw))
        except ValueError:
            pass
    return max(1, n // group_size) * group_size

_lut_lock = threading.Lock()
_DECODE_LUT = None  # uint8  → float32  (E4M3)
_ENCODE_LUT = None  # fp16 bits → uint8 (E4M3)


# ───────────────────────────── LUTs E4M3 ─────────────────────────────


def _build_decode_lut():
    """256 valores float32, uno por patrón de bits de float8_e4m3fn.

    E4M3FN: bias 7, sin infinitos, 0x7F/0xFF son NaN.
    """
    import numpy as np

    codes = np.arange(256, dtype=np.uint16)
    sign = np.where((codes >> 7) & 1, -1.0, 1.0).astype(np.float32)
    exp = ((codes >> 3) & 0xF).astype(np.int32)
    man = (codes & 0x7).astype(np.float32)

    subnormal = (man / 8.0) * np.float32(2.0**-6)
    normal = (1.0 + man / 8.0) * np.power(2.0, (exp - 7).astype(np.float32))
    vals = np.where(exp == 0, subnormal, normal).astype(np.float32) * sign
    # NaN → 0: un NaN en el KV ya es un bloque perdido; no vale la pena
    # propagarlo por el codec.
    vals[(exp == 15) & (man == 7)] = 0.0
    return vals


def _build_encode_lut(decode_lut):
    """65536 bytes: patrón de bits float16 → código E4M3 más cercano."""
    import numpy as np

    finite = decode_lut.copy()
    order = np.argsort(finite, kind="stable")
    sorted_vals = finite[order]

    half_bits = np.arange(65536, dtype=np.uint16)
    half_vals = half_bits.view(np.float16).astype(np.float32)
    half_vals = np.nan_to_num(half_vals, nan=0.0, posinf=448.0, neginf=-448.0)

    pos = np.searchsorted(sorted_vals, half_vals)
    pos = np.clip(pos, 1, len(sorted_vals) - 1)
    left = sorted_vals[pos - 1]
    right = sorted_vals[pos]
    take_left = np.abs(half_vals - left) <= np.abs(right - half_vals)
    chosen = np.where(take_left, pos - 1, pos)
    return order[chosen].astype(np.uint8)


def _luts():
    global _DECODE_LUT, _ENCODE_LUT
    if _ENCODE_LUT is None:
        with _lut_lock:
            if _ENCODE_LUT is None:
                dec = _build_decode_lut()
                _ENCODE_LUT = _build_encode_lut(dec)
                _DECODE_LUT = dec
    return _DECODE_LUT, _ENCODE_LUT


# ───────────────────────────── config ─────────────────────────────


def is_fp4_compression_enabled() -> bool:
    """¿Comprimir bloques NUEVOS al escribirlos? (default: apagado)

    Ojo con la separación de flags, es deliberada:

      `GENESIS_ENABLE_PN92_KV_FP4_COMPRESSION`
          gate del DISPATCHER: instala los hooks en `fs/io.py`. Conviene
          dejarlo en 1 para que los archivos ya comprimidos se puedan leer.

      `GENESIS_PN92_COMPRESS_ON_WRITE`   (este)
          decide si se comprime al escribir. Default **0**.

    Apagado por default a propósito: en este híbrido el KV ya corre en fp8 y
    bajar a 4 bits degrada la reconstrucción del prefijo (12,4% de error L2
    medido en el grupo de atención). Ver `kv-fp8-degrada-gdn`.

    La DEScompresión no mira ningún flag: un archivo con magic se descomprime
    siempre, así que apagar la compresión nunca destruye el caché en disco.
    """
    return os.environ.get("GENESIS_PN92_COMPRESS_ON_WRITE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def kv_dtype_code() -> int:
    """Dtype del KV en disco. Tiene que coincidir con `--kv-cache-dtype`.

    `GENESIS_PN92_KV_DTYPE` ∈ {fp8_e4m3, float16}. Si no se reconoce, se
    devuelve -1 y la compresión se desactiva sola (mejor no comprimir que
    comprimir interpretando mal los bytes).
    """
    raw = os.environ.get("GENESIS_PN92_KV_DTYPE", "fp8_e4m3").strip().lower()
    if raw in ("fp8", "fp8_e4m3", "float8_e4m3fn", "e4m3"):
        return _DTYPE_FP8_E4M3
    if raw in ("fp16", "float16", "half", "auto"):
        return _DTYPE_FLOAT16
    log.warning("PN92: GENESIS_PN92_KV_DTYPE=%r no reconocido; sin compresión", raw)
    return -1


def _aligned_buffer(nbytes: int) -> memoryview:
    """Buffer de página alineada, apto para `O_DIRECT`.

    Un `bytes` grande de CPython suele quedar alineado porque el allocator cae
    en mmap, pero no está garantizado. Acá se garantiza.
    """
    size = (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    return memoryview(mmap.mmap(-1, size))


# ───────────────────────── compresión ─────────────────────────


def compress_block(buffer, group_size: int | None = None):
    """Comprime un bloque KV crudo. Devuelve un memoryview alineado, o None.

    None significa "no comprimir": el llamador escribe el bloque tal cual.

    Se procesa por bloques de trabajo (ver `chunk_elems`) para no materializar
    el bloque entero en float32.
    """
    import numpy as np

    dtype_code = kv_dtype_code()
    if dtype_code < 0:
        return None

    group_size = group_size or DEFAULT_GROUP_SIZE
    raw = memoryview(buffer).cast("B")
    orig_len = len(raw)
    if orig_len == 0:
        return None
    if dtype_code == _DTYPE_FLOAT16 and orig_len % 2:
        return None

    n = orig_len if dtype_code == _DTYPE_FP8_E4M3 else orig_len // 2
    num_groups = (n + group_size - 1) // group_size
    packed_len = (n + 1) // 2
    scales_len = num_groups * 2
    payload_len = _HEADER_LEN + scales_len + packed_len

    # Se compara contra el tamaño YA ALINEADO, que es lo que de verdad se
    # escribe: con bloques chicos el relleno a 4096 puede superar al original.
    if (payload_len + _ALIGN - 1) // _ALIGN * _ALIGN >= orig_len:
        return None  # no achica: se escribe el bloque crudo

    decode_lut, _ = _luts()
    out = _aligned_buffer(payload_len)
    scales_off = _HEADER_LEN
    packed_off = _HEADER_LEN + scales_len

    step = chunk_elems(group_size)
    for start in range(0, n, step):
        stop = min(start + step, n)
        count = stop - start

        if dtype_code == _DTYPE_FP8_E4M3:
            codes = np.frombuffer(raw, dtype=np.uint8, count=count, offset=start)
            values = decode_lut[codes]
        else:
            values = np.frombuffer(
                raw, dtype=np.float16, count=count, offset=start * 2
            ).astype(np.float32)

        g_first = start // group_size
        g_count = (count + group_size - 1) // group_size
        pad = g_count * group_size - count
        if pad:
            values = np.concatenate([values, np.zeros(pad, dtype=np.float32)])

        grouped = values.reshape(g_count, group_size)
        absmax = np.max(np.abs(grouped), axis=1, keepdims=True)
        scales = (absmax / 7.0).astype(np.float16)
        safe = scales.astype(np.float32)
        safe[safe == 0] = 1.0

        q = np.rint(grouped / safe).clip(-7, 7).astype(np.int8)
        q_u4 = (q & 0x0F).astype(np.uint8).reshape(-1)[:count]
        if count % 2:  # sólo puede pasar en el último bloque de trabajo
            q_u4 = np.append(q_u4, np.uint8(0))
        packed = (q_u4[0::2] | (q_u4[1::2] << 4)).astype(np.uint8)

        so = scales_off + g_first * 2
        out[so : so + g_count * 2] = scales.reshape(-1).tobytes()
        po = packed_off + start // 2
        out[po : po + packed.nbytes] = packed.tobytes()

    out[payload_len:] = b"\x00" * (len(out) - payload_len)  # relleno a 4096

    crc = zlib.crc32(bytes(out[_HEADER_LEN:payload_len])) & 0xFFFFFFFF
    header = FP4_MAGIC_HEADER + struct.pack(
        _HEADER_STRUCT,
        _CODEC_INT4_GROUPED,
        dtype_code,
        orig_len,
        group_size,
        num_groups,
        crc,
    )
    out[: len(header)] = header
    return out


def is_compressed(data) -> bool:
    return bytes(memoryview(data)[: len(FP4_MAGIC_HEADER)]) == FP4_MAGIC_HEADER


def is_legacy_v1(data) -> bool:
    head = bytes(memoryview(data)[:6])
    return head in _LEGACY_MAGICS


def decompress_block(compressed_data, target_view) -> int:
    """Descomprime dentro de `target_view`. Devuelve los bytes escritos.

    Lanza si el formato no es reconocible o el CRC no cierra: el llamador decide
    qué hacer, pero NUNCA debe caer al camino de lectura cruda (que borra el
    archivo por short read).
    """
    import numpy as np

    data = memoryview(compressed_data).cast("B")
    if not is_compressed(data):
        raise ValueError("bloque sin magic PN92 v3")

    off = len(FP4_MAGIC_HEADER)
    codec, dtype_code, orig_len, group_size, num_groups, crc = struct.unpack(
        _HEADER_STRUCT, data[off : off + struct.calcsize(_HEADER_STRUCT)]
    )
    if codec != _CODEC_INT4_GROUPED:
        raise ValueError(f"codec PN92 desconocido: {codec}")

    target = memoryview(target_view).cast("B")
    if len(target) < orig_len:
        raise ValueError(f"target de {len(target)} B para un bloque de {orig_len} B")

    n = orig_len if dtype_code == _DTYPE_FP8_E4M3 else orig_len // 2
    num_groups_calc = (n + group_size - 1) // group_size
    if num_groups_calc != num_groups:
        raise ValueError("cabecera PN92 inconsistente")

    scales_len = num_groups * 2
    packed_len = (n + 1) // 2
    payload_len = _HEADER_LEN + scales_len + packed_len
    if len(data) < payload_len:
        raise ValueError("bloque PN92 truncado")

    # Sin esto, un bloque corrupto se descomprime a basura PLAUSIBLE y entra al
    # KV sin que nada lo note. El camino crudo al menos da short read.
    actual = zlib.crc32(bytes(data[_HEADER_LEN:payload_len])) & 0xFFFFFFFF
    if actual != crc:
        raise ValueError(f"CRC PN92 no coincide: {actual:08x} != {crc:08x}")

    scales_off, packed_off = _HEADER_LEN, _HEADER_LEN + scales_len
    _, encode_lut = _luts()

    step = chunk_elems(group_size)
    for start in range(0, n, step):
        stop = min(start + step, n)
        count = stop - start
        g_first = start // group_size
        g_count = (count + group_size - 1) // group_size

        scales = np.frombuffer(
            data, dtype=np.float16, count=g_count, offset=scales_off + g_first * 2
        )
        n_packed = (count + 1) // 2
        packed = np.frombuffer(
            data, dtype=np.uint8, count=n_packed, offset=packed_off + start // 2
        )

        lo = (packed & 0x0F).astype(np.int8)
        hi = ((packed >> 4) & 0x0F).astype(np.int8)
        lo = np.where(lo >= 8, lo - 16, lo)
        hi = np.where(hi >= 8, hi - 16, hi)

        q = np.empty(n_packed * 2, dtype=np.int8)
        q[0::2] = lo
        q[1::2] = hi
        q = q[:count]

        pad = g_count * group_size - count
        if pad:
            q = np.concatenate([q, np.zeros(pad, dtype=np.int8)])

        values = (
            q.reshape(g_count, group_size).astype(np.float32)
            * scales.astype(np.float32).reshape(-1, 1)
        ).reshape(-1)[:count]

        if dtype_code == _DTYPE_FP8_E4M3:
            half_bits = values.astype(np.float16).view(np.uint16)
            target[start:stop] = encode_lut[half_bits].tobytes()
        else:
            target[start * 2 : stop * 2] = values.astype(np.float16).tobytes()

    return orig_len


# ─────────────────────── helpers de IO ───────────────────────


def probe_compressed(source_path: str) -> bool:
    """¿El archivo en disco está comprimido por PN92 v2?

    Se consulta SIEMPRE, independientemente de `is_fp4_compression_enabled()`.
    Si dependiera del flag, apagar la compresión haría que cada archivo
    comprimido se leyera como KV crudo → short read → `load_block` lo BORRA.
    Ese era el modo de fallo más caro de la versión anterior: apagar la env
    destruía el caché entero en silencio.
    """
    try:
        with open(source_path, "rb") as f:
            return is_compressed(f.read(HEADER_PROBE_LEN))
    except OSError:
        return False


def load_compressed_block(source_path: str, target_view) -> int:
    """Lee y descomprime un bloque PN92 dentro de `target_view`.

    Lectura con buffer normal (sin `O_DIRECT`): el tamaño en disco no es el del
    bloque y las restricciones de alineación no aplican.

    Si el archivo está corrupto se lo borra —igual que hace `load_block`— para
    que el siguiente lookup sea un miss limpio en vez de un error recurrente.
    """
    try:
        with open(source_path, "rb") as f:
            data = f.read()
        # El camino comprimido no pasa por el `os.readv` que instrumenta PN88,
        # así que los bytes se anotan acá o el contador de lecturas al NVMe
        # subestimaría justo cuando la compresión está activa.
        try:
            from vllm._genesis import kv_tier_metrics as _g88

            _g88.note_disk_read(len(data))
        except Exception:
            pass
        return decompress_block(data, target_view)
    except Exception:
        try:
            os.remove(source_path)
        except OSError as cleanup_exc:
            log.warning(
                "PN92: no se pudo borrar el bloque corrupto %s: %s",
                source_path,
                cleanup_exc,
            )
        raise


# ─────────── compat: nombres que usa el código inyectado ───────────


def compress_bytes_fp8_to_fp4(buffer):
    return compress_block(buffer)


def decompress_bytes_fp4_to_fp8(compressed_data, target_view) -> int:
    return decompress_block(compressed_data, target_view)
