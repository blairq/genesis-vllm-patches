# SPDX-License-Identifier: Apache-2.0
"""PN152 — all-reduce de TP=2 para el decode por P2P directo (kernel SK-24, ``kernels/cuda/sk24_ar_p2p.cu``).

NCCL RING_LL tarda 26 us por 92 KB (1 pedido, 9 filas de 5120 fp16) y 61 us por 369 KB (4 pedidos), 128
veces por paso: 12-19% del paso de decode. El enlace (PCIe 4.0 x8 con P2P, ~12 GB/s) mueve 92 KB en ~8 us:
el resto es protocolo. SK-24 escribe el parcial en la memoria de la otra placa, avisa con una bandera y
suma; doble buffer por paridad de epoca (sin barrera final) y la epoca en la GPU (sirve dentro de grafos).
Ideas de SiFAR (arXiv 2607.08973) traducidas a SM86 sin NVSwitch.

Solo TP=2 y mensajes <= ``_MAX_BYTES``; lo demas sigue por el camino de siempre. El resultado es el mismo
que el de NCCL: con 2 rangos es una suma fp16.

La memoria es ``cudaMalloc`` crudo compartido por IPC (con tensores de torch o expandable_segments un
kernel no puede tocar la memoria de la otra placa; ver p2p_buzon.py y la memoria del proyecto). La
inicializacion (reservar + intercambiar handles por el grupo CPU de TP) se hace en la primera llamada en
eager, que es la corrida de perfilado de memoria de vLLM, antes de grabar grafos.
"""

from __future__ import annotations

import ctypes
import logging
import os

import torch

log = logging.getLogger("genesis.pn152")
_ACTIVO = os.environ.get("GENESIS_ENABLE_PN152_AR_P2P", "0").strip().lower() in ("1", "true", "yes", "on")
_MAX_BYTES = int(os.environ.get("GENESIS_PN152_MAX_BYTES", str(1 << 20)))   # por ranura
_BLOQUES = int(os.environ.get("GENESIS_PN152_BLOQUES", "4"))   # barrido 27-09: 4 gana en 90 KB (14,8 us) y empata en 360/540 KB
_estado = {"listo": False, "fallo": None}


def activo() -> bool:
    return _ACTIVO and _estado["fallo"] is None


def _rt():
    from vllm._genesis.p2p_buzon import _libs, _preparar_firmas
    _preparar_firmas()
    rt, _ = _libs()
    rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    rt.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    return rt


def inicializar(rank: int, w: int, grupo_cpu) -> None:
    """Reserva [ranura0 | ranura1 | banderas] compartido por IPC y el control local (epoca, contadores)."""
    import torch.distributed as dist
    from vllm._genesis.p2p_buzon import IpcHandle, habilitar_p2p
    if w != 2:
        raise RuntimeError(f"PN152 solo TP=2 (w={w})")
    rt = _rt()
    dev = torch.cuda.current_device()
    otro = 1 - rank
    habilitar_p2p([otro] if torch.cuda.device_count() > 1 else [])
    tam = 2 * _MAX_BYTES + 256
    base = ctypes.c_void_p()
    r = rt.cudaMalloc(ctypes.byref(base), ctypes.c_size_t(tam))
    if r != 0:
        raise RuntimeError(f"cudaMalloc -> {r}")
    rt.cudaMemset(base, 0, ctypes.c_size_t(tam))
    mio = ctypes.c_void_p()                     # mi parte cuantizada (camino int8), local
    r = rt.cudaMalloc(ctypes.byref(mio), ctypes.c_size_t(_MAX_BYTES))
    if r != 0:
        raise RuntimeError(f"cudaMalloc(mio) -> {r}")
    ctl = ctypes.c_void_p()
    r = rt.cudaMalloc(ctypes.byref(ctl), ctypes.c_size_t(64))
    if r != 0:
        raise RuntimeError(f"cudaMalloc(ctl) -> {r}")
    rt.cudaMemset(ctl, 0, ctypes.c_size_t(64))
    h = IpcHandle()
    r = rt.cudaIpcGetMemHandle(ctypes.byref(h), base)
    if r != 0:
        raise RuntimeError(f"cudaIpcGetMemHandle -> {r}")
    crudo = ctypes.string_at(ctypes.byref(h), 64)
    mi_handle = torch.frombuffer(bytearray(crudo), dtype=torch.uint8).clone()
    todos = [torch.zeros(64, dtype=torch.uint8) for _ in range(w)]
    dist.all_gather(todos, mi_handle, group=grupo_cpu)
    ho = IpcHandle()
    ctypes.memmove(ctypes.byref(ho), bytes(todos[otro].numpy()), 64)
    po = ctypes.c_void_p()
    r = rt.cudaIpcOpenMemHandle(ctypes.byref(po), ho, ctypes.c_uint(1))
    if r != 0:
        raise RuntimeError(f"cudaIpcOpenMemHandle -> {r}")
    torch.cuda.synchronize()
    dist.barrier(group=grupo_cpu)           # los dos buffers en cero antes de la primera escritura remota
    _estado.update(listo=True, rx_local=base.value, rx_peer=po.value, flag_local=base.value + 2 * _MAX_BYTES,
                   flag_peer=po.value + 2 * _MAX_BYTES, ctl=ctl.value, dev=dev, kern=None, kern8=None,
                   mio=mio.value, rank=rank)
    log.warning("[PN152] all-reduce P2P listo (rank %d, %d KB por ranura, %d bloques)", rank, _MAX_BYTES >> 10, _BLOQUES)


def asegurar_inicializado() -> bool:
    """Inicializa en la primera llamada en eager (nunca dentro de una captura). True si se puede usar."""
    if not _ACTIVO or _estado["fallo"] is not None:
        return False
    if _estado["listo"]:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    try:
        from vllm.distributed.parallel_state import get_tp_group
        g = get_tp_group()
        inicializar(g.rank_in_group, g.world_size, g.cpu_group)
        return True
    except Exception as e:                      # queda desactivado para siempre, con el motivo en el log
        _estado["fallo"] = repr(e)
        log.warning("[PN152] desactivado: %s", e)
        return False


def sirve(x: torch.Tensor) -> bool:
    n = x.numel() * x.element_size()
    return (_estado["listo"] and x.dtype == torch.float16 and x.is_contiguous() and n % 16 == 0
            and n <= _MAX_BYTES and x.data_ptr() % 16 == 0)


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Suma de los parciales de los 2 rangos (x es el propio). Devuelve un tensor nuevo."""
    k = _estado["kern"]
    if k is None:
        from vllm._genesis.kernels.ptx_lab import Kernel
        k = _estado["kern"] = Kernel("sk24_ar_p2p.cu", "sk24_ar", defs=["-DHILOS=256"], warps=8)
    out = torch.empty_like(x)
    n16 = x.numel() * x.element_size() // 16
    P = ctypes.c_uint64
    k.lanzar((_BLOQUES, 1), [x, out, n16, P(_estado["rx_local"]), P(_estado["rx_peer"]), _MAX_BYTES // 16,
                             P(_estado["flag_local"]), P(_estado["flag_peer"]), P(_estado["ctl"])])
    return out


_G = 64
# bytes fp16 desde los que va en int8 (0 = nunca). Medido 27-09: int8 gana en todos los tamanos (9 filas
# 13,3 vs 14,9 us; 36 filas 26,7 vs 44,9) y la cuenta es la de PN120, que en el prefill cuesta KL 0,0001
# (0,0193 vs 0,0192 en respuestas, kl_ar_int8.sh); servido: paso -1,3% (1 pedido) / -3,5% (4). El compose usa 1.
_MIN_I8 = int(os.environ.get("GENESIS_PN152_MIN_I8", "0"))
_BLOQUES_I8 = int(os.environ.get("GENESIS_PN152_BLOQUES_I8", "8"))   # barrido 27-09: 8 gana con 36 y 54 filas


def sirve_i8(x: torch.Tensor) -> bool:
    n = x.numel()
    return (_MIN_I8 > 0 and sirve(x) and x.shape[-1] % _G == 0 and n * 2 >= _MIN_I8
            and n + (n // _G) * 2 <= _MAX_BYTES)


def all_reduce_i8(x: torch.Tensor) -> torch.Tensor:
    """Como ``all_reduce`` pero los parciales viajan en int8 por grupo de 64 (la cuenta de PN120) y las dos
    placas suman las dos partes cuantizadas en orden de rango: el resultado es identico en ambas."""
    k = _estado["kern8"]
    if k is None:
        from vllm._genesis.kernels.ptx_lab import Kernel
        k = _estado["kern8"] = Kernel("sk24_ar_p2p.cu", "sk24_ar_i8", defs=["-DHILOS=256"], warps=8)
    out = torch.empty_like(x)
    P = ctypes.c_uint64
    k.lanzar((_BLOQUES_I8, 1), [x, out, x.numel() // _G, _estado["rank"], P(_estado["rx_local"]),
                             P(_estado["rx_peer"]), _MAX_BYTES, P(_estado["mio"]),
                             P(_estado["flag_local"]), P(_estado["flag_peer"]), P(_estado["ctl"])])
    return out
