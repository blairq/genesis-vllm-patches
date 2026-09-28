// SPDX-License-Identifier: Apache-2.0
//
// PN122 — filas 1..T-1 de la cinta de rollback del GDN, en PTX. Reemplaza a _k_escribir (Triton,
// UN programa por pedido que recorre tokens y cabezas en serie: 34 us por capa con un pedido) y
// a su version repartida _k_escribir_par (Triton, 2,1 us).
//
// Lo que hacia _k_escribir_par, leido del SASS (27-09):
//   * 128 hilos por programa y cada uno carga UN fp16 (LDG.U16) y guarda un float;
//   * la norma l2: butterfly en el warp (xor 16 con FFMA, 8/4/2/1 con FADD), los 4 parciales por
//     shared con DOS bar.sync y otro butterfly (xor 2, 1): (p0+p2)+(p1+p3);
//   * g y beta los calculaban los 128 hilos (softplus con el log de libdevice, div.full) y
//     guardaba uno solo;
//   * `slots` se cargaba recien despues del chequeo de salida: tres latencias de memoria en serie.
//
// Aca: bloques de 8 warps por fila (pedido, token): uno para k, g y beta y NVB para v. Las cuatro cargas de control salen juntas;
// cada warp normaliza una cabeza k con cargas de 8 bytes y guardas de 16, y la reduccion hace
// el MISMO arbol que Triton sobre los mismos elementos, sin shared ni barreras; la copia de v
// va en vectores; g y beta los calcula un hilo por cabeza. Toda la aritmetica en PTX explicito
// (este archivo se compila con --use_fast_math y eso cambiaria los bits): las mismas
// instrucciones que Triton, en el mismo orden. Resultado identico bit a bit.
//
// Formas (defines): H cabezas k, HV cabezas v, K = V = 128, TM filas por slot, ROW floats por fila
// = H*K + HV*V + 2*HV. Grilla (N, TM, 1 + NVB), 256 hilos.

#include <cuda_fp16.h>

#ifndef H
#define H 8
#endif
#ifndef HV
#define HV 24
#endif
#ifndef TM
#define TM 8
#endif
#define KD 128
#define VD 128
#define ROW (H * KD + HV * VD + 2 * HV)
#define NW 8
#define NVB ((HV * VD / 4 + NW * 32 - 1) / (NW * 32))   // bloques de v por fila (3 con HV = 24)

static_assert(H <= NW, "una cabeza k por warp");
static_assert(HV <= 32, "g y beta: un hilo por cabeza v dentro de un warp");

__device__ __forceinline__ float mul_rn(float a, float b) { float r; asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float add_rn(float a, float b) { float r; asm("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float sub_rn(float a, float b) { float r; asm("sub.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float fma_rn(float a, float b, float c) { float r; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float fma_ftz(float a, float b, float c) { float r; asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
__device__ __forceinline__ float ex2_ap(float a) { float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float rsqrt_ftz(float a) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float h2f(unsigned short h) { float r; asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h)); return r; }
__device__ __forceinline__ float shfl_x(float v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }

#define LOG2E 1.4426950216293334961f   // 0f3FB8AA3B

// logf de libdevice con ftz, tal cual lo inlinea Triton (__nv_logf): la misma secuencia de PTX.
__device__ __forceinline__ float logf_nv(float x) {
    float r; unsigned u;
    asm("{\n\t"
        ".reg .pred p, q;\n\t"
        ".reg .f32 s, m, c, e, f, t, l;\n\t"
        ".reg .b32 i, j, k;\n\t"
        "setp.lt.f32 p, %1, 0f00800000;\n\t"
        "mul.f32 s, %1, 0f4B000000;\n\t"
        "selp.f32 m, s, %1, p;\n\t"
        "selp.f32 c, 0fC1B80000, 0f00000000, p;\n\t"
        "mov.b32 i, m;\n\t"
        "add.s32 j, i, -1059760811;\n\t"
        "and.b32 j, j, -8388608;\n\t"
        "sub.s32 k, i, j;\n\t"
        "cvt.rn.f32.s32 e, j;\n\t"
        "fma.rn.ftz.f32 e, e, 0f34000000, c;\n\t"
        "mov.b32 f, k;\n\t"
        "add.f32 f, f, 0fBF800000;\n\t"
        "fma.rn.ftz.f32 t, 0fBE055027, f, 0f3E1039F6;\n\t"
        "fma.rn.ftz.f32 t, t, f, 0fBDF8CDCC;\n\t"
        "fma.rn.ftz.f32 t, t, f, 0f3E0F2955;\n\t"
        "fma.rn.ftz.f32 t, t, f, 0fBE2AD8B9;\n\t"
        "fma.rn.ftz.f32 t, t, f, 0f3E4CED0B;\n\t"
        "fma.rn.ftz.f32 t, t, f, 0fBE7FFF22;\n\t"
        "fma.rn.ftz.f32 t, t, f, 0f3EAAAA78;\n\t"
        "fma.rn.ftz.f32 t, t, f, 0fBF000000;\n\t"
        "mul.f32 t, f, t;\n\t"
        "fma.rn.ftz.f32 t, t, f, f;\n\t"
        "fma.rn.ftz.f32 l, e, 0f3F317218, t;\n\t"
        "setp.lt.u32 q, i, 2139095040;\n\t"
        "@!q fma.rn.ftz.f32 l, m, 0f7F800000, 0f7F800000;\n\t"
        "setp.eq.f32 q, m, 0f00000000;\n\t"
        "selp.f32 %0, 0fFF800000, l, q;\n\t"
        "}"
        : "=f"(r) : "f"(x));
    (void)u;
    return r;
}

extern "C" __global__ void __launch_bounds__(NW * 32)
pn122_cinta(const float* __restrict__ A_log, const __half* __restrict__ a, const __half* __restrict__ b,
            const float* __restrict__ dt_bias, float beta_sp, float threshold,
            const __half* __restrict__ k, const __half* __restrict__ v, long long sk, long long sv,   // stride de token
            const int* __restrict__ cu, const int* __restrict__ sidx, const int* __restrict__ slots,
            float* __restrict__ cinta, int N)
{
    const unsigned n = blockIdx.x, t = blockIdx.y + 1u, tid = threadIdx.x;
    // las cuatro cargas de control, juntas: no dependen entre si
    const int bos = __ldg(cu + n), eos = __ldg(cu + n + 1), s = __ldg(sidx + n), slot = __ldg(slots + n);
    if (s <= 0 || (int)t >= eos - bos) return;
    const unsigned src = (unsigned)bos + t;
    float* __restrict__ row = cinta + ((size_t)slot * TM + (t - 1u)) * ROW;
    const unsigned w = tid >> 5, l = tid & 31u;
    // Grilla (N, TM, 1 + NVB): z = 0 normaliza las cabezas k y calcula g y beta; z >= 1 copia un tramo
    // de v. Con un bloque por fila eran 8 bloques para 82 SM con un pedido (2,1 us, peor que Triton).
    if (blockIdx.z > 0) {
        const unsigned c = (blockIdx.z - 1u) * (NW * 32) + tid;
        if (c < HV * VD / 4) {
            const uint2 raw = __ldg(reinterpret_cast<const uint2*>(v + (size_t)src * sv) + c);
            float4 o;
            o.x = h2f((unsigned short)(raw.x & 0xffffu)); o.y = h2f((unsigned short)(raw.x >> 16));
            o.z = h2f((unsigned short)(raw.y & 0xffffu)); o.w = h2f((unsigned short)(raw.y >> 16));
            reinterpret_cast<float4*>(row + H * KD)[c] = o;
        }
        return;
    }

    // ---- cabezas k: warp w = cabeza w; el carril l tiene los elementos 4l..4l+3 ----
    // Sin rama: todos los warps hacen la cuenta (con H < NW los de mas repiten la cabeza 0 y no
    // guardan). Dentro de un `if` el compilador no puede probar que el warp esta convergente y
    // cada shfl.sync lleva una ruta de respaldo por CALL (14 en el SASS de la primera version).
    {
        const unsigned wk = w < H ? w : 0u;
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(k + (size_t)src * sk + (size_t)wk * KD) + l);
        float x[4];
        x[0] = h2f((unsigned short)(raw.x & 0xffffu)); x[1] = h2f((unsigned short)(raw.x >> 16));
        x[2] = h2f((unsigned short)(raw.y & 0xffffu)); x[3] = h2f((unsigned short)(raw.y >> 16));
        // Arbol de Triton sobre el elemento e (hilo e, warp e/32): xor 16 -> carril ^4 (FFMA sobre el
        // cuadrado redondeado del par), xor 8 -> ^2, xor 4 -> ^1, xor 2 y 1 -> dentro del hilo
        // (j^2, j^1), y entre los 4 "warps" de Triton (bits 5 y 6 de e = carril ^16 y ^8): (p0+p2)+(p1+p3).
        float q[4];
#pragma unroll
        for (int j = 0; j < 4; j++) { const float sq = mul_rn(x[j], x[j]); q[j] = fma_rn(x[j], x[j], shfl_x(sq, 4)); }
#pragma unroll
        for (int j = 0; j < 4; j++) q[j] = add_rn(q[j], shfl_x(q[j], 2));
#pragma unroll
        for (int j = 0; j < 4; j++) q[j] = add_rn(q[j], shfl_x(q[j], 1));
        const float r0 = add_rn(q[0], q[2]), r1 = add_rn(q[1], q[3]);
        float p = add_rn(r0, r1);
        p = add_rn(p, shfl_x(p, 16));
        p = add_rn(p, shfl_x(p, 8));
        const float rs = rsqrt_ftz(add_rn(p, 9.99999997e-07f));      // 0f358637BD
        float4 o;
        o.x = mul_rn(rs, x[0]); o.y = mul_rn(rs, x[1]); o.z = mul_rn(rs, x[2]); o.w = mul_rn(rs, x[3]);
        if (w < H) reinterpret_cast<float4*>(row + w * KD)[l] = o;
    }

    // ---- g (warp 0) y beta (warp 1): un hilo por cabeza v ----
    if (w == 0 && l < HV) {
        const unsigned hv = l;
        const float x = add_rn(h2f(__half_as_ushort(a[(size_t)src * HV + hv])), __ldg(dt_bias + hv));
        const float bx = mul_rn(beta_sp, x);
        const float ib = div_full(1.0f, beta_sp);
        const float lg = logf_nv(add_rn(ex2_ap(mul_rn(bx, LOG2E)), 1.0f));
        const float sp = (bx <= threshold) ? mul_rn(ib, lg) : x;
        const float eA = ex2_ap(mul_rn(__ldg(A_log + hv), LOG2E));
        row[H * KD + HV * VD + hv] = mul_rn(sub_rn(0.0f, eA), sp);
    } else if (w == 1 && l < HV) {
        const unsigned hv = l;
        const float bh = h2f(__half_as_ushort(b[(size_t)src * HV + hv]));
        const float e = ex2_ap(mul_rn(sub_rn(0.0f, bh), LOG2E));
        row[H * KD + HV * VD + HV + hv] = div_full(1.0f, add_rn(e, 1.0f));
    }
}
