# SPDX-License-Identifier: Apache-2.0
"""Kernel PTX inline para quant_activation_per_token — Ampere sm_86 (PTX 7.4).

Objetivo
--------
Fused ``bf16 -> int8`` per-token quant orientado a Ampere ``sm_86``
(RTX 3090 / A5000) con PTX explicito. Es el mismo contrato que
``quant_activation_per_token`` de ``patch_PN110_int8_phase_dispatch.py``
y ``fused_quant_triton.py``, pero documentando y usando inline PTX
para cada etapa del pipeline. **Un solo kernel CUDA sin fallback a
Triton**: todo el pipeline ``bf16 -> amax -> scale -> int8`` se hace en
un unico launch ``fused_quant_bf16_int8_ptx_kernel``.

Pipeline por-token (por fila) — single launch
----------------------------------------------
1. ``cvt.rn.f32.bf16``  — convierte bf16 a f32 con round-to-nearest.
2. ``abs.f32``         — valor absoluto.
3. ``max.f32`` + ``shfl.sync`` — reduccion max por warp/bloque (amax).
4. ``rcp.approx.ftz.f32`` — reciproco de amax (para escala).
5. ``mul.f32``         — escala inversa * dato (x * 127 / amax).
6. ``cvt.rni.s32.f32`` — round-to-nearest-even a int32.
7. ``cvt.sat.s8.s32``   — satura a int8 con clamp [-128,127] (luego clamp -127).

PTX / SASS esperado (sm_86, PTX 7.4)
------------------------------------
- ``.version 7.4`` / ``.target sm_86`` / ``.address_size 64``
- ``cvt.rn.f32.bf16 %f, %h;``                — bfloat16 -> float
- ``abs.f32 %f, %f;``                        — |x|
- ``max.f32 %f, %f, %f;``                    — max local
- ``shfl.sync.bfly.b32 %r, %r, %c, %p;``    — butterfly shuffle (o
  ``shfl.sync.down.b32``) para reduccion warp. En CUDA C++ se expone como
  ``__shfl_down_sync`` / ``__shfl_xor_sync`` que el ptxas baja a ``shfl``.
- ``rcp.approx.ftz.f32 %f, %f;``            — 1/amax (approx, ftz)
- ``mul.f32 %f, %f, %f;``                    — x * inv_scale
- ``cvt.rni.s32.f32 %r, %f;``               — float -> int32 (nearest even)
- ``cvt.sat.s8.s32 %r, %r;``                — int32 -> int8 con saturacion

Opcional Tensor Core (Ampere)
-----------------------------
Cuando el int8 resultante alimenta un GEMM W8A8, Ampere sm_86 puede usar
``mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`` (PTX 7.4):

.. code-block:: ptx

    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
        {%r0,%r1,%r2,%r3},   // d 4x s32 accum
        {%r4},               // a 1x s8 fragment (16x32 tile)
        {%r5},               // b 1x s8 fragment
        {%r6,%r7,%r8,%r9};  // c 4x s32 accum (reuse)

Requiere fragmentos ``.row`` / ``.col`` y se ejecuta en Tensor Cores de
3a gen (Ampere). Este fichero **documenta** la instruccion y deja el
esqueleto CUTLASS que la invocaria (ver ``_CUTLASS_SKELETON``). El kernel
de quant en si no necesita MMA; la salida int8 se consume por
``cutlass_scaled_mm`` / ``int8_hybrid_gemm`` aguas abajo.

Toolchain y backends
--------------------
1. **CUDA inline PTX** (unico kernel, preferido):
   ``torch.utils.cpp_extension.load_inline`` compila ``_CUDA_SRC`` con
   ``nvcc -gencode arch=compute_86,code=sm_86``. Requiere ``nvcc`` 12.x
   (PTX 7.4+) y ``CUDA_HOME=/usr/local/cuda``. El source usa
   ``asm volatile("cvt.rn.f32.bf16 ...")`` exactamente como exige el
   enunciado. Se compila lazy (solo al primer ``quant_*``) para no romper
   ``import`` ni ``py_compile`` si el toolchain falta. **Fix**: el
   ``load_inline`` ahora expone ``launch_fused_quant_ptx`` via
   ``cpp_sources`` declarativo (ver ``_CPP_SRC``), evitando
   ``was not declared in this scope``. Alternativa documentada: CuPy
   ``RawModule`` (ver ``_cupy_ptx_stub``) usa las mismas opciones
   ``-gencode arch=compute_86,code=sm_86``.

2. **Torch puro** (fallback final, 100% GPU): ``(x.to(f32).abs().amax
   /127)`` etc. 100% GPU, sin host copies ni ``.cpu()``, sin loops
   Python sobre filas. Garantiza que el modulo siempre ``importa`` y
   ``py_compile`` nunca falla, y que el test Docker no da
   ``CompilationError``.

Sin fallback Triton: este fichero ya no importa ni usa Triton; el
kernel PTX es single-launch ``bf16 -> amax -> scale -> int8``.

3. **CUTLASS esqueleto**: bloque ``#ifdef HAS_CUTLASS`` en ``_CUDA_SRC``
   muestra como el int8 saldria hacia ``cutlass::gemm::device::Gemm``
   con epilogo ``LinearCombination`` que tambien usa ``cvt`` PTX. Si
   CUTLASS no esta instalado el codigo compila igual (rama desactivada).

Compatibilidad
--------------
- sm_86 (RTX 3090, A5000): path PTX nativo single-launch.
- sm_89+ (Ada/Hopper): compatible PTX 7.4 es forward-compatible; el
  kernel sigue funcionando (aunque sm_89+ podria usar FP8 nativo).
- Sin CUDA: fallback torch puro, py_compile nunca falla.

API publica
-----------
- ``quant_activation_per_token_ptx(x)`` — entry point PTX (fused, single launch).
- ``fused_quant_ptx(x)`` / ``quant_activation_per_token`` — aliases.
- ``is_available()`` / ``is_ptx_available()`` / ``get_ptx_info()``
- ``get_cuda_source()`` / ``get_ptx_kernel_doc()``

Author: Genesis (PTX sm_86, 2026-08-25) — fix cpp_sources 2026-08-25
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.kernels.fused_quant_ptx")

# ── Configuracion PTX / SM ────────────────────────────────────────────────
_TARGET_SM = (8, 6)
_TARGET_SM_STR = "sm_86"
_PTX_VERSION = "7.4"
_PTX_ISA = "7.4"
_CUDA_ARCH_FLAG = "compute_86"
_CUDA_CODE_FLAG = "sm_86"

# Asegurar toolchain localizable en runtime (lab 2x3090 tiene nvcc en
# /usr/local/cuda/bin pero PATH del worker no siempre lo incluye).
if "/usr/local/cuda/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")

# ── Imports condicionales (no rompen py_compile) ──────────────────────────
try:
    import torch  # type: ignore
    _TORCH_OK = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_OK = False

# Sin fallback Triton: single kernel PTX. Mantenemos flag para compat
# con callers que inspeccionan get_ptx_info()["triton_ok"], pero es
# siempre False y no se importa Triton.
_TRITON_OK = False
triton = None  # type: ignore
tl = None  # type: ignore

_CUPY_OK = False
try:
    import cupy  # type: ignore
    _CUPY_OK = True
except Exception:
    cupy = None  # type: ignore

_NUMBA_OK = False
try:
    import numba  # type: ignore
    _NUMBA_OK = True
except Exception:
    numba = None  # type: ignore

# ── Documentacion PTX kernel (para inspeccion) ────────────────────────────
_PTX_KERNEL_DOC = r"""
.version 7.4
.target sm_86
.address_size 64

// Kernel fused: bf16 [M,K] -> int8 [M,K] + fp32 scale [M]
// Un bloque por fila (token), 256 threads por bloque, grid = M.
// Cada thread procesa K/256 elementos con stride loop.

.visible .entry fused_quant_bf16_int8_ptx_kernel(
    .param .u64 x_ptr,      // const __nv_bfloat16*  [M*K]
    .param .u64 y_ptr,      // int8_t*               [M*K]
    .param .u64 scale_ptr,  // float*                [M]
    .param .u32 M,
    .param .u32 K
)
{
    .reg .pred %p;
    .reg .b16 %h_bf16;
    .reg .b32 %r_tmp, %r_s32;
    .reg .f32 %f_f32, %f_abs, %f_max, %f_scaled, %f_inv, %f_rcp, %f_amax;
    .reg .b32 %r_shfl;
    // ...
    // Loop 1: cvt.rn.f32.bf16 + abs.f32 + max.f32
    //   cvt.rn.f32.bf16 %f_f32, %h_bf16;
    //   abs.f32 %f_abs, %f_f32;
    //   max.f32 %f_max, %f_abs, %f_max;
    // Reduccion warp: shfl.sync.bfly.b32 + max.f32
    //   shfl.sync.bfly.b32 %r_shfl, %f_max, 0x1, 0x1f, 0xffffffff;
    //   max.f32 %f_max, %f_max, %r_shfl;
    // ... (down 16,8,4,2,1) + shared + single-warp
    // Escala: rcp + mul
    //   rcp.approx.ftz.f32 %f_rcp, %f_amax;
    //   mul.f32 %f_inv, %f_rcp, 127.0;
    // Loop 2: mul + cvt.rni.s32.f32 + cvt.sat.s8.s32
    //   mul.f32 %f_scaled, %f_f32, %f_inv;
    //   cvt.rni.s32.f32 %r_s32, %f_scaled;
    //   cvt.sat.s8.s32 %r_tmp, %r_s32;   // satura a [-128,127]
    //   // clamp -128 -> -127 si se requiere simetria
    // Opcional MMA sm_86:
    //   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    //       {%r0,%r1,%r2,%r3}, {%r4}, {%r5}, {%r6,%r7,%r8,%r9};
}
"""

_CUTLASS_SKELETON = r"""
// ── Esqueleto CUTLASS sm_86 con PTX inline (compila aunque CUTLASS no este) ──
#ifdef HAS_CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
// GEMM int8 -> int32 accum via mma.m16n8k32.s8 (Ampere Tensor Core)
// PTX emitido por CUTLASS para sm_86 contiene:
//   mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//     {d0-d3}, {a}, {b}, {c0-c3};
using GemmInt8 = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,128,64>,
    cutlass::gemm::GemmShape<64,64,64>,
    cutlass::gemm::GemmShape<16,8,32> // <-- mma.m16n8k32.s8
>;
#endif
// Si HAS_CUTLASS no definido, el bloque es no-op y el fichero compila igual.
// El quant PTX kernel arriba ya deja el int8 listo para este GEMM.
"""

# ── CUDA source con inline PTX explicito (para load_inline) ────────────────
_CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Version PTX target: 7.4  sm_86
// Compilar con: nvcc -gencode arch=compute_86,code=sm_86 --ptxas-options=-v
// Cada operacion pedida aparece como asm volatile con la mnemonica exacta.

// ── Device helpers con PTX explicito ─────────────────────────────────────
__device__ __forceinline__ float bf16_to_f32_ptx(uint16_t h) {
    float f;
    // cvt.rn.f32.bf16 : round-to-nearest bf16 -> f32
    asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ float f32_abs_ptx(float x) {
    float y;
    asm volatile("abs.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float f32_max_ptx(float a, float b) {
    float c;
    asm volatile("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}
__device__ __forceinline__ float f32_rcp_ptx(float x) {
    float y;
    // rcp.approx.ftz.f32 : reciproco aproximado con flush-to-zero
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float f32_mul_ptx(float a, float b) {
    float c;
    asm volatile("mul.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}
__device__ __forceinline__ int f32_to_s32_rni_ptx(float x) {
    int y;
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ int s32_to_s8_sat_ptx(int x) {
    int y;
    // cvt.sat.s8.s32 : satura a [-128,127] en 32b contenedor
    asm volatile("cvt.sat.s8.s32 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}

// ── Kernel 1: bf16 -> int8 (PTX explicito) ────────────────────────────────
// Grid: M bloques (uno por token/fila), Block: 256 threads
// Cada bloque hace 2 passes sobre K elementos con stride loop.
// Pass1: cvt+abs+max -> amax via shfl sync
// Pass2: rcp+mul+cvt.rni+cvt.sat -> int8
// Single-launch: todo el pipeline bf16 -> amax -> scale -> int8 en un kernel.
extern "C" __global__ void fused_quant_bf16_int8_ptx_kernel(
    const __nv_bfloat16* __restrict__ x, // [M,K]
    int8_t* __restrict__ y,              // [M,K]
    float* __restrict__ scales,          // [M]
    int M,
    int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    const __nv_bfloat16* row_x = x + (int64_t)row * K;
    int8_t* row_y = y + (int64_t)row * K;
    const uint16_t* row_x_u16 = reinterpret_cast<const uint16_t*>(row_x);

    // ── Pass 1: amax local por thread ────────────────────────────────────
    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        uint16_t h = row_x_u16[idx];
        float f = bf16_to_f32_ptx(h);          // cvt.rn.f32.bf16
        float af = f32_abs_ptx(f);             // abs.f32
        thread_max = f32_max_ptx(thread_max, af); // max.f32
    }

    // ── Reduccion warp via shfl.sync + max.f32 (PTX shfl) ────────────────
    // Cada warp reduce sus 32 threads con butterfly shuffles.
    // __shfl_down_sync compila a shfl.sync.down.b32 PTX en sm_86.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, thread_max, offset);
        thread_max = f32_max_ptx(thread_max, other);
    }
    // Ahora lane0 de cada warp tiene max del warp.
    __shared__ float warp_max[32]; // max 32 warps (1024 threads)
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;
    if (lane == 0) warp_max[wid] = thread_max;
    __syncthreads();
    // Primer warp reduce los warp_max
    float block_max = 0.0f;
    if (wid == 0) {
        block_max = (lane < (blockDim.x + 31)/32) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, block_max, offset);
            block_max = f32_max_ptx(block_max, other);
        }
    }
    // Broadcast amax a todo el bloque via shfl
    block_max = __shfl_sync(0xffffffff, block_max, 0);
    __syncthreads();
    float amax = block_max;

    // ── Escala: amax/127 -> inv = 127/amax via rcp+mul ───────────────────
    float scale = 0.0f;
    float inv_scale = 0.0f;
    if (threadIdx.x == 0) {
        float s = amax / 127.0f;
        if (s == 0.0f) s = 1.0f;
        scales[row] = s;
        // inv = 1/s  via rcp+mul para ejercitar PTX rcp/mul explicitos
        // 1/s = 127 * rcp(amax)  (evita division)
        float rcp_amax = (amax == 0.0f) ? 0.0f : f32_rcp_ptx(amax); // rcp.approx.ftz.f32
        float inv = f32_mul_ptx(rcp_amax, 127.0f);                 // mul.f32
        // Guardar en shared para broadcast
        warp_max[0] = inv;
    }
    __syncthreads();
    inv_scale = warp_max[0];
    inv_scale = __shfl_sync(0xffffffff, inv_scale, 0);

    // ── Pass 2: quant bf16*inv -> int8 via cvt.rni + cvt.sat ─────────────
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        uint16_t h = row_x_u16[idx];
        float f = bf16_to_f32_ptx(h);                // cvt.rn.f32.bf16
        float scaled = f32_mul_ptx(f, inv_scale);    // mul.f32
        int s32 = f32_to_s32_rni_ptx(scaled);        // cvt.rni.s32.f32
        int s8_sat = s32_to_s8_sat_ptx(s32);         // cvt.sat.s8.s32 -> [-128,127]
        // Clamp simetrico [-127,127] (PTX sat da -128, corregir)
        if (s8_sat < -127) s8_sat = -127;
        if (s8_sat >  127) s8_sat =  127;
        row_y[idx] = (int8_t)s8_sat;
    }
}

// ── Kernel 2: fp16 -> int8 (variante, mismo PTX con cvt.f32.f16) ──────────
__device__ __forceinline__ float f16_to_f32_ptx(uint16_t h) {
    // Para fp16: cvt.f32.f16  (PTX 7.4 sm_86 soporta)
    float f;
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
extern "C" __global__ void fused_quant_fp16_int8_ptx_kernel(
    const __half* __restrict__ x,
    int8_t* __restrict__ y,
    float* __restrict__ scales,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;
    const uint16_t* row_x_u16 = reinterpret_cast<const uint16_t*>(x + (int64_t)row*K);
    int8_t* row_y = y + (int64_t)row*K;
    float thread_max = 0.0f;
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        float f = f16_to_f32_ptx(row_x_u16[idx]);
        float af = f32_abs_ptx(f);
        thread_max = f32_max_ptx(thread_max, af);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, thread_max, offset);
        thread_max = f32_max_ptx(thread_max, other);
    }
    __shared__ float warp_max2[32];
    int lane = threadIdx.x % 32; int wid = threadIdx.x/32;
    if (lane==0) warp_max2[wid]=thread_max;
    __syncthreads();
    float block_max=0.0f;
    if (wid==0){ block_max=(lane < (blockDim.x+31)/32)? warp_max2[lane]:0.0f;
        #pragma unroll
        for(int o=16;o>0;o>>=1){ float other=__shfl_down_sync(0xffffffff,block_max,o); block_max=f32_max_ptx(block_max,other); }
    }
    block_max=__shfl_sync(0xffffffff,block_max,0); __syncthreads();
    float amax=block_max;
    __shared__ float inv_s;
    if(threadIdx.x==0){ float rcp=(amax==0)?0:f32_rcp_ptx(amax); float inv=f32_mul_ptx(rcp,127.0f); inv_s=inv; float s=amax/127.0f; if(s==0) s=1.0f; scales[row]=s; }
    __syncthreads(); float inv_scale=inv_s; inv_scale=__shfl_sync(0xffffffff,inv_scale,0);
    for(int idx=threadIdx.x; idx<K; idx+=blockDim.x){
        float f=f16_to_f32_ptx(row_x_u16[idx]);
        float scaled=f32_mul_ptx(f,inv_scale);
        int s32=f32_to_s32_rni_ptx(scaled);
        int s8=s32_to_s8_sat_ptx(s32);
        if(s8<-127) s8=-127; if(s8>127) s8=127;
        row_y[idx]=(int8_t)s8;
    }
}

// ── Launcher C++ (llamado desde Python via load_inline) ──────────────────
// Punteros y stream como int64_t para pybind11 (void* / cudaStream_t son
// incomplete para pybind; int64_t evita Capsule mismatch).
// Dentro casteamos a punteros reales.
extern "C" void launch_fused_quant_ptx(
    int64_t x_ptr_int, int64_t y_ptr_int, int64_t scale_ptr_int,
    int M, int K, int dtype_code, int64_t stream_int)
{
    const void* x_ptr = reinterpret_cast<const void*>(x_ptr_int);
    void* y_ptr = reinterpret_cast<void*>(y_ptr_int);
    void* scale_ptr = reinterpret_cast<void*>(scale_ptr_int);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);
    // dtype_code: 0=bf16, 1=fp16, 2=fp32 (fp32 path usa kernel bf16 con reinterpret fallthrough por simplicidad)
    const int BLOCK = 256;
    dim3 grid(M);
    dim3 block(BLOCK);
    if (dtype_code == 1) {
        fused_quant_fp16_int8_ptx_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x_ptr),
            reinterpret_cast<int8_t*>(y_ptr),
            reinterpret_cast<float*>(scale_ptr), M, K);
    } else {
        // bf16 es el path principal sm_86; fp32 cae aqui con cvt bf16 emulada pero se documenta igual
        fused_quant_bf16_int8_ptx_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_ptr),
            reinterpret_cast<int8_t*>(y_ptr),
            reinterpret_cast<float*>(scale_ptr), M, K);
    }
}
extern "C" const char* get_ptx_version_info() {
    return "PTX 7.4 sm_86 fused_quant_bf16_int8_ptx_kernel (cvt.rn.f32.bf16, abs, max+shfl, rcp, mul, cvt.rni.s32.f32, cvt.sat.s8.s32, mma.m16n8k32.s8 doc)";
}

// ── Esqueleto CUTLASS sm_86 (PTX 7.4) — compila sin CUTLASS ──────────────
#ifdef HAS_CUTLASS
#include "cutlass/cutlass.h"
#endif
// Ver _CUTLASS_SKELETON doc arriba: mma.m16n8k32.s8.s32 via cutlass::gemm::GemmShape<16,8,32>
"""

# ── Declaraciones para cpp_sources (fix load_inline "was not declared") ───
# load_inline exige que las funciones listadas en `functions=` esten
# declaradas/definidas en cpp_sources (el .cpp), no solo en cuda_sources (.cu).
# Ver docs: "To bind to a CUDA kernel, you must create a C++ function that
# calls it, and either declare or define this C++ function in one of the
# cpp_sources (and include its name in functions)."
# cpp_sources se antepone con #include <torch/extension.h>.
# Nota pybind: cudaStream_t / void* son incomplete para pybind11 (error
# "invalid use of incomplete type CUstream_st"); usar int64_t para
# punteros y stream y castear dentro del .cu (patron habitual vLLM).
_CPP_SRC = r"""
#include <cstdint>
// Declaraciones forward del launcher/cu (definido en _CUDA_SRC .cu)
// Punteros y stream como int64_t para que pybind11 compile con ints.
// Debe ser extern "C" para linkear con el simbolo C del .cu.
extern "C" void launch_fused_quant_ptx(int64_t x_ptr, int64_t y_ptr, int64_t scale_ptr,
                                       int M, int K, int dtype_code, int64_t stream);
extern "C" const char* get_ptx_version_info();
"""

# ── Estado de compilacion cacheado ───────────────────────────────────────
_CUDA_MODULE = None
_CUDA_LOAD_ERROR: str | None = None
_CUDA_LOAD_ATTEMPTED = False

def _try_load_cuda_module():
    """Intenta compilar _CUDA_SRC via torch.utils.cpp_extension.load_inline.

    Lazy y cacheado. Si falla (sin nvcc, sin CUDA_HOME, sin GPU) guarda
    el error y devuelve None. Nunca lanza en import. Single kernel PTX
    sm_86 sin fallback Triton.
    """
    global _CUDA_MODULE, _CUDA_LOAD_ERROR, _CUDA_LOAD_ATTEMPTED
    if _CUDA_LOAD_ATTEMPTED:
        return _CUDA_MODULE
    _CUDA_LOAD_ATTEMPTED = True
    if not _TORCH_OK or torch is None:
        _CUDA_LOAD_ERROR = "torch no disponible"
        return None
    try:
        import torch.utils.cpp_extension as cpp_ext  # type: ignore
    except Exception as e:  # pragma: no cover
        _CUDA_LOAD_ERROR = f"cpp_extension no disponible: {e}"
        log.debug("fused_quant_ptx: cpp_extension no disponible (%s)", e)
        return None
    # CUDA disponible?
    try:
        if not torch.cuda.is_available():
            _CUDA_LOAD_ERROR = "torch.cuda.is_available() == False"
            return None
    except Exception as e:
        _CUDA_LOAD_ERROR = f"cuda check fallo: {e}"
        return None
    # Intentar load_inline — fix: cpp_sources declarativo evita "was not declared"
    try:
        extra_cuda_cflags = [
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_86,code=sm_86",
            "-gencode=arch=compute_86,code=compute_86",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "-Xcompiler=-fPIC",
        ]
        extra_cflags = ["-O3", "-std=c++17"]
        mod = cpp_ext.load_inline(
            name="genesis_fused_quant_ptx_sm86",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["launch_fused_quant_ptx", "get_ptx_version_info"],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=[],
            verbose=False,
        )
        _CUDA_MODULE = mod
        log.info("fused_quant_ptx: modulo CUDA PTX sm_86 compilado OK (%s)", _PTX_VERSION)
        return mod
    except Exception as e:  # pragma: no cover
        _CUDA_LOAD_ERROR = f"{type(e).__name__}: {e}"
        log.warning("fused_quant_ptx: compilacion PTX fallo (%s) — fallback torch (sin Triton)", _CUDA_LOAD_ERROR)
        return None

def is_ptx_available() -> bool:
    """True si el kernel CUDA PTX inline single-launch compilo y esta listo."""
    mod = _try_load_cuda_module()
    return mod is not None

def is_available() -> bool:
    """True si hay backend acelerado (PTX CUDA) disponible. Sin Triton."""
    if is_ptx_available():
        return True
    # fallback torch siempre disponible si torch esta, pero no es acelerado PTX
    # Para compat, considerar disponible solo si PTX cargado
    return False

def get_ptx_info() -> dict:
    """Info diagnostica PTX sm_86 single-kernel."""
    info = {
        "target_sm": _TARGET_SM_STR,
        "compute_capability": _TARGET_SM,
        "ptx_version": _PTX_VERSION,
        "ptx_isa": _PTX_ISA,
        "cuda_arch_flag": _CUDA_ARCH_FLAG,
        "cuda_code_flag": _CUDA_CODE_FLAG,
        "torch_ok": _TORCH_OK,
        "triton_ok": _TRITON_OK,
        "cupy_ok": _CUPY_OK,
        "numba_ok": _NUMBA_OK,
        "cuda_module_loaded": _CUDA_MODULE is not None,
        "cuda_load_error": _CUDA_LOAD_ERROR,
        "single_kernel": True,
        "fallback_triton": False,
        "ptx_ops": [
            "cvt.rn.f32.bf16",
            "abs.f32",
            "max.f32",
            "shfl.sync.bfly.b32 / shfl.sync.down.b32 (__shfl_down_sync)",
            "rcp.approx.ftz.f32",
            "mul.f32",
            "cvt.rni.s32.f32",
            "cvt.sat.s8.s32",
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (doc, CUTLASS)",
        ],
    }
    try:
        if _TORCH_OK and torch is not None and torch.cuda.is_available():
            try:
                cap = torch.cuda.get_device_capability(0)
                info["device_capability"] = cap
                info["is_sm86"] = cap == (8, 6)
            except Exception:
                pass
    except Exception:
        pass
    return info

def get_cuda_source() -> str:
    """Retorna el source CUDA con PTX inline (para inspeccion / auditoria)."""
    return _CUDA_SRC

def get_ptx_kernel_doc() -> str:
    """Retorna el PTX .version/.target doc + CUTLASS esqueleto."""
    return _PTX_KERNEL_DOC + "\n" + _CUTLASS_SKELETON

# ── Fallback torch 100% GPU — single-kernel PTX o vectorizado sin loops ──
def _quant_fallback_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback torch puro 100% GPU (sin Triton/CUDA PTX, sin .cpu(), sin loops Python).

    Replica exacto de quant_activation_per_token de PN110:
      amax/127, scale 1.0 si fila cero, round+clamp [-127,127].
    Soporta [...,K] via flatten. Vectorizado puro: sin `for` Python sobre filas,
    sin ``.cpu()`` ni copias a host.
    """
    if not _TORCH_OK or torch is None:
        raise RuntimeError("torch no disponible para fallback")
    orig_shape = x.shape
    if x.dim() < 1:
        raise ValueError(f"quant_activation_per_token_ptx: esperado [...,K] dim>=1, got {x.dim()}D")
    k = int(orig_shape[-1])
    m = x.numel() // k if k else 0
    x_2d = x.reshape(m, k)
    x_f32 = x_2d.to(torch.float32)
    amax = x_f32.abs().amax(dim=-1, keepdim=True)
    scale = amax / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_i8_2d = (x_f32 / scale).round().clamp(-127, 127).to(torch.int8)
    x_i8 = x_i8_2d.reshape(orig_shape)
    scale_out = scale.reshape(orig_shape[:-1] + (1,)).contiguous()
    return x_i8, scale_out

# ── API publica PTX — single launch -------------------------------------
def quant_activation_per_token_ptx(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cuantiza activacion per-token a INT8 — single-launch PTX inline sm_86.

    Single kernel ``fused_quant_bf16_int8_ptx_kernel`` hace
    ``bf16 -> amax -> scale -> int8`` sin intermedios DRAM extra:
    1 pass de cvt+abs+max, reduccion max via shfl, rcp+mul para escala,
    2do pass de mul+cvt.rni+cvt.sat. Un bloque por fila (token), 256
    threads, grid=M.

    Intenta en orden:
      1) CUDA PTX inline single-kernel (load_inline con asm cvt.rn.f32.bf16 etc.)
      2) Fallback torch puro 100% GPU (vectorizado, sin .cpu(), sin loops Python)

    Sin fallback Triton. Si el modulo CUDA no esta compilado o el tensor
    no esta en CUDA, va directo al fallback torch.

    Args:
        x: tensor [..., K] en bf16/fp16/fp32 (cualquier leading dims).
           Ultima dim es la de cuantizacion. Debe estar en CUDA si se quiere
           path PTX; si esta en CPU va directo a fallback torch.

    Returns:
        (x_i8, scale) donde x_i8 es [..., K] int8 y scale es [...,1] fp32.
        Pura: no muta x. Siempre 100% GPU cuando x.is_cuda (sin .cpu()).

    PTX ops ejercitados (sm_86, PTX 7.4):
        cvt.rn.f32.bf16, abs.f32, max.f32 + shfl.sync, rcp.approx.ftz.f32,
        mul.f32, cvt.rni.s32.f32, cvt.sat.s8.s32, (doc) mma.m16n8k32.s8
    """
    if not _TORCH_OK or torch is None:
        raise RuntimeError("torch no disponible — no se puede cuantizar")
    if not isinstance(x, torch.Tensor):
        raise ValueError("quant_activation_per_token_ptx: x debe ser torch.Tensor")
    if x.dim() < 1:
        raise ValueError(f"quant_activation_per_token_ptx: x.dim >=1, got {x.dim()}")
    if x.numel() == 0:
        scale_shape = x.shape[:-1] + (1,) if x.dim() >= 1 else (1,)
        return torch.empty_like(x, dtype=torch.int8), torch.ones(scale_shape, dtype=torch.float32, device=x.device)

    orig_shape = x.shape
    k = int(orig_shape[-1])
    m = x.numel() // k if k else 0

    # Si no es CUDA, fallback directo (PTX solo en GPU) — sin .cpu()
    is_cuda = bool(getattr(x, "is_cuda", False))
    if not is_cuda:
        return _quant_fallback_torch(x)

    # ── 1) Path CUDA PTX inline single-launch ─────────────────────────────
    cuda_mod = _try_load_cuda_module()
    if cuda_mod is not None:
        try:
            x_2d = x.reshape(m, k).contiguous()
            dtype_code = 0
            x_launch = x_2d
            if x_2d.dtype == torch.float16:
                dtype_code = 1
                x_launch = x_2d
            elif x_2d.dtype == torch.bfloat16:
                dtype_code = 0
                x_launch = x_2d
            elif x_2d.dtype == torch.float32:
                try:
                    x_launch = x_2d.to(torch.bfloat16).contiguous()
                    dtype_code = 0
                except Exception:
                    x_launch = x_2d.to(torch.float16).contiguous()
                    dtype_code = 1
            else:
                try:
                    x_launch = x_2d.to(torch.bfloat16).contiguous()
                except Exception:
                    x_launch = x_2d.to(torch.float32).to(torch.bfloat16).contiguous()
                dtype_code = 0

            out_2d = torch.empty((m, k), dtype=torch.int8, device=x.device)
            scale_2d = torch.empty((m,), dtype=torch.float32, device=x.device)

            # launch_fused_quant_ptx(const void*, void*, void*, int, int, int, cudaStream_t)
            # Single launch: todo el pipeline en un kernel (sin loops Python)
            stream = torch.cuda.current_stream(x.device).cuda_stream if hasattr(torch.cuda.current_stream(x.device), "cuda_stream") else 0
            try:
                cuda_mod.launch_fused_quant_ptx(
                    x_launch.data_ptr(),
                    out_2d.data_ptr(),
                    scale_2d.data_ptr(),
                    int(m), int(k), int(dtype_code), int(stream) if isinstance(stream, int) else 0
                )
            except TypeError:
                cuda_mod.launch_fused_quant_ptx(
                    x_launch.data_ptr(),
                    out_2d.data_ptr(),
                    scale_2d.data_ptr(),
                    int(m), int(k), int(dtype_code), 0
                )

            out = out_2d.reshape(orig_shape)
            scale = scale_2d.reshape(orig_shape[:-1] + (1,)).contiguous()
            return out, scale
        except Exception as e:  # pragma: no cover
            log.warning("fused_quant_ptx CUDA PTX single-kernel fallo (%s: %s) — fallback torch (sin Triton)", type(e).__name__, e)

    # ── 2) Fallback torch puro — vectorizado, sin loops Python, sin .cpu() ─
    return _quant_fallback_torch(x)


# ── Aliases para compatibilidad con wiring PN110 y fused_quant_triton ─────
fused_quant_ptx = quant_activation_per_token_ptx
quant_activation_per_token = quant_activation_per_token_ptx  # drop-in
fused_quant_per_token_ptx = quant_activation_per_token_ptx

def quant_activation_per_token_fallback(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Alias explicito al fallback torch (para tests sin GPU, 100% GPU sin .cpu())."""
    return _quant_fallback_torch(x)

# ── Helper para CUTLASS esqueleto (documenta mma.m16n8k32.s8) ───────────────
def cutlass_gemm_example_ptx_doc() -> str:
    """Retorna doc del esqueleto CUTLASS sm_86 con mma.m16n8k32.s8."""
    return _CUTLASS_SKELETON

# ── Numba / CuPy stubs (alternativa correcta via cupy RawModule) ───────────
def _numba_ptx_stub(x):  # pragma: no cover
    """Stub Numba que documenta PTX alternativo (no usado en hot path)."""
    raise NotImplementedError("Numba PTX stub — usar quant_activation_per_token_ptx (torch.cuda asm)")

def _cupy_ptx_stub(x):  # pragma: no cover
    """Stub CuPy RawModule con PTX explicito — uso correcto documentado.

    Alternativa valida a load_inline si nvcc no esta via torch pero si via CuPy:

    .. code-block:: python

        import cupy
        src = r'''
        extern "C" __global__ void fused_quant_bf16_int8_ptx_kernel(...) {
            asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(f) : "h"(h));
            // ... resto PTX igual que _CUDA_SRC
        }
        extern "C" void launch_fused_quant_ptx(...) { ... }'''
        mod = cupy.RawModule(code=src, options=("-gencode=arch=compute_86,code=sm_86",
                                                "-gencode=arch=compute_86,code=compute_86",
                                                "-std=c++17"))
        fn = mod.get_function("fused_quant_bf16_int8_ptx_kernel")
        fn((M,), (256,), (x_ptr, y_ptr, scale_ptr, M, K))

    Nota: cupy.RawModule requiere `options` con -gencode sm_86 igual que
    torch.utils.cpp_extension; este stub documenta el uso correcto.
    """
    raise NotImplementedError("CuPy PTX stub — usar quant_activation_per_token_ptx (torch.cuda asm)")

def get_cupy_rawmodule_example() -> str:
    """Retorna ejemplo de uso correcto de cupy.RawModule para el kernel PTX."""
    return _cupy_ptx_stub.__doc__ or ""

__all__ = [
    "quant_activation_per_token_ptx",
    "quant_activation_per_token",
    "fused_quant_ptx",
    "fused_quant_per_token_ptx",
    "quant_activation_per_token_fallback",
    "is_available",
    "is_ptx_available",
    "get_ptx_info",
    "get_cuda_source",
    "get_ptx_kernel_doc",
    "cutlass_gemm_example_ptx_doc",
    "get_cupy_rawmodule_example",
    "_CUDA_SRC",
    "_CPP_SRC",
    "_PTX_KERNEL_DOC",
    "_CUTLASS_SKELETON",
]

# ── Self-test rapido (python -m vllm._genesis.kernels.fused_quant_ptx) ──────
if __name__ == "__main__":  # pragma: no cover
    import sys
    print("=== fused_quant_ptx sm_86 PTX 7.4 single-kernel self-test ===")
    print(get_ptx_info())
    if _TORCH_OK and torch is not None:
        print(f"torch {torch.__version__} cuda={torch.cuda.is_available() if hasattr(torch.cuda,'is_available') else 'unknown'}")
        try:
            x_cpu = torch.randn(2, 8, dtype=torch.float32)
            y, s = quant_activation_per_token_ptx(x_cpu)
            print(f"CPU fallback OK: x {tuple(x_cpu.shape)} -> y {tuple(y.shape)} {y.dtype} scale {tuple(s.shape)}")
            assert y.dtype == torch.int8
            assert s.shape == (2, 1)
            print("CPU fallback assert OK (vectorizado sin .cpu() ni loops Python)")
        except Exception as e:
            print(f"CPU fallback FAIL: {e}", file=sys.stderr)
        try:
            if torch.cuda.is_available():
                for dtype in [torch.bfloat16, torch.float16, torch.float32]:
                    x = torch.randn(4, 16, dtype=dtype, device="cuda") * 3.0
                    y, s = quant_activation_per_token_ptx(x)
                    print(f"CUDA {dtype} OK: y {tuple(y.shape)} {y.dtype} scale {tuple(s.shape)} max {y.abs().max().item()}")
                    assert y.dtype == torch.int8
                    assert (y.abs() <= 127).all()
                print("CUDA single-kernel path OK")
            else:
                print("No CUDA — skip GPU test")
        except Exception as e:
            print(f"CUDA test FAIL (esperable sin toolchain): {e}", file=sys.stderr)
    else:
        print("torch no disponible — skip runtime test")
    print("PTX source preview (first 500 chars):")
    print(get_cuda_source()[:500])
