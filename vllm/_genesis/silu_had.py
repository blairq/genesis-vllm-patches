# SPDX-License-Identifier: Apache-2.0
"""PN148 fusionado: SiluAndMul + Hadamard por bloques de 512 en UN kernel (Triton).

El camino de torch (``rot_down.aplicar``) hace la Hadamard como matmul densa de 512x512 (512 MAC por
valor) y una pasada extra por memoria: 4-5% del prefill. Aca se lee la salida de gate_up una vez,
se calcula silu(g)*u y se aplica la Hadamard en el mismo programa, sin escribir el intermedio.

La Hadamard de Sylvester factoriza: H512[i, j] = H16[a, c] * H32[b, d] con i = 32a + b, j = 32c + d.
Con el bloque como matriz X [16, 32]:  Y = H16 X H32  (las dos simetricas). Son 48 MAC por valor
en vez de 512, con tl.dot en tensor cores fp16 (acumulan en fp32; el intermedio va partido
en alto + bajo, asi no se pierden bits).

Numerica igual al camino de torch: el producto silu(g)*u se redondea a fp16 (como la salida del
SiluAndMul de vLLM) antes de rotar, la rotacion acumula en fp32 y la salida es fp16.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

B = 512
A1 = tl.constexpr(16)                 # B = A1 * A2
A2 = tl.constexpr(32)


@triton.jit
def _paridad(v):
    return (v ^ (v >> 1) ^ (v >> 2) ^ (v >> 3) ^ (v >> 4)) & 1


@triton.jit
def _silu_had_kernel(x_ptr, o_ptr, T, N, sx, so, escala, M: tl.constexpr):
    """M tokens x un bloque de 512 por programa. Primero X H32 como [M*16, 32] @ [32, 32] y despues
    H16 por token sobre la traspuesta: las multiplicaciones salen de M*16 filas y no de 16."""
    t0 = tl.program_id(0) * M
    blk = tl.program_id(1)
    m = tl.arange(0, M)
    a = tl.arange(0, A1)
    b = tl.arange(0, A2)
    fila = t0 + m
    ok = (fila < T)[:, None, None]
    dentro = blk * (A1 * A2) + a[None, :, None] * A2 + b[None, None, :]
    xo = fila[:, None, None].to(tl.int64) * sx + dentro
    g = tl.load(x_ptr + xo, mask=ok, other=0.).to(tl.float32)
    u = tl.load(x_ptr + xo + N, mask=ok, other=0.).to(tl.float32)
    y = (g / (1.0 + tl.exp(-g)) * u).to(tl.float16)                      # [M, 16, 32], como SiluAndMul
    h2 = (1 - 2 * _paridad(b[:, None] & b[None, :])).to(tl.float16)      # H32 sin normalizar
    h1 = (1 - 2 * _paridad(a[:, None] & a[None, :])).to(tl.float16)      # H16
    z = tl.dot(tl.reshape(y, (M * A1, A2)), h2)                            # fp32
    z = tl.reshape(tl.permute(tl.reshape(z, (M, A1, A2)), (0, 2, 1)), (M * A2, A1))
    zh = z.to(tl.float16)                                                  # alto + bajo: sin perder bits
    zl = (z - zh.to(tl.float32)).to(tl.float16)
    w = (tl.dot(zh, h1) + tl.dot(zl, h1)) * escala                         # [M*32, 16]
    w = tl.permute(tl.reshape(w, (M, A2, A1)), (0, 2, 1))                 # [M, 16, 32]
    oo = fila[:, None, None].to(tl.int64) * so + dentro
    tl.store(o_ptr + oo, w.to(o_ptr.dtype.element_ty), mask=ok)


def silu_had_triton(x: torch.Tensor) -> torch.Tensor:
    """x [..., 2N] (gate | up) -> [..., N] = Hadamard_512(silu(gate) * up)."""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    T, N2 = x2.shape
    N = N2 // 2
    assert N % B == 0, N
    o = torch.empty((T, N), dtype=x.dtype, device=x.device)
    if T:
        # barrido 2026-09-25 (GA102, reloj fijo, grafos): 8 tokens x 2 warps en prefill (621 us a
        # 8192 tokens vs 565 del SiluAndMul solo); en decode, 1 token x 2 warps
        M = 8 if T >= 64 else 1
        _silu_had_kernel[(triton.cdiv(T, M), N // B)](x2, o, T, N, x2.stride(0), o.stride(0), B ** -0.5,
                                                     M=M, num_warps=2)
    return o.reshape(*shp[:-1], N)


@torch.library.custom_op("genesis::pn148_silu_had", mutates_args=())
def silu_had(x: torch.Tensor) -> torch.Tensor:
    """Op propia: torch.compile no la abre y queda grabada tal cual en el CUDA graph."""
    return silu_had_triton(x)


@silu_had.register_fake
def _silu_had_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


# --- SK-23: la misma cuenta, pero sale int8 por token (con la escala ya por la global) ---------
# En Triton no se puede: la escala por token necesita el maximo de TODA la fila (17 bloques de 512)
# y habria que escribir o recalcular el intermedio, que deja el trafico igual que hoy. En CUDA cada
# token es un bloque de hilos y los datos quedan en registros (ver kernels/cuda/sk23_silu_had_q8.cu).
_k23: dict = {}


def _kernel23(N: int):
    dev = torch.cuda.current_device()
    clave = (dev, N)
    if clave not in _k23:
        from vllm._genesis.kernels.ptx_lab import Kernel
        nb = N // B
        nw = min(nb, 32)
        porw = -(-nb // nw)
        k = Kernel("sk23_silu_had_q8.cu", "sk23_silu_had_q8", defs=[f"-DNW={nw}", f"-DPORW={porw}"], warps=nw)
        k.cargar()
        _k23[clave] = k
    return _k23[clave]


def silu_had_q8_cuda(x: torch.Tensor, gscale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1 or x2.stride(0) % 8:
        x2 = x2.contiguous()
    T, N2 = x2.shape
    N = N2 // 2
    assert N % B == 0 and x2.dtype == torch.float16, (N, x2.dtype)
    q = torch.empty((T, N), dtype=torch.int8, device=x.device)
    esc = torch.empty((T, 1), dtype=torch.float32, device=x.device)
    if T:
        g = gscale.reshape(-1)
        if g.dtype != torch.float32:
            g = g.float()
        _kernel23(N).lanzar((T, 1), [x2, q, esc, g, N, x2.stride(0), q.stride(0)])
    return q, esc


@torch.library.custom_op("genesis::pn148_silu_had_q8", mutates_args=())
def silu_had_q8(x: torch.Tensor, gscale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(int8 [T, N], escala fp32 [T, 1] ya multiplicada por gscale) de Hadamard_512(silu(g)*u)."""
    return silu_had_q8_cuda(x, gscale)


@silu_had_q8.register_fake
def _silu_had_q8_fake(x: torch.Tensor, gscale: torch.Tensor):
    T = x.numel() // x.shape[-1]
    return (x.new_empty((T, x.shape[-1] // 2), dtype=torch.int8),
            x.new_empty((T, 1), dtype=torch.float32))
