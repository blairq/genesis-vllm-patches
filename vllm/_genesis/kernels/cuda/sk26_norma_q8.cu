// SK-26: residuo + RMSNorm (Gemma, peso plegado o no) + int8 por token, en UN kernel (PN156).
//
// Reemplaza tres kernels por sitio de norma (inductor rms_norm, per_token_quant_int8 y la escala por la
// global de Marlin) y la entrada llega a Marlin W4A8 ya en int8.
//
// La normalizacion se cancela en la cuantizacion: con y = r * rstd * g,
//     q = round(y * 127 / max|y|) = round(r*g * 127 / max|r*g|)
// asi que rstd NO se aplica a ningun elemento: solo entra en la escala de salida (un escalar por fila):
//     esc = max|r*g| * rstd / 127 * global
//
// Variantes para medir (SUMA):
//   0  todo en fp32: suma del residuo, g, maximo, cuantizacion y suma de cuadrados con FFMA
//   1  half2 en el camino por elemento (HADD2/HMUL2/HMNMX2, cvt.rni.sat.s8.f16) y la suma de cuadrados
//      con DP4A sobre el int8 ya cuantizado: sum r^2 ~= (max/127)^2 * sum q^2 (entero, 4 MAC por instr.)
//   2  como 1, pero la suma de cuadrados en fp32 (para aislar lo que cuesta ese bucle)
//   3  como 0 (todo fp32), con el maximo y la suma reducidos JUNTOS: una ronda de shuffles con los dos
//      valores, una escritura a shared y 2 barreras en vez de 5 (el costo del kernel son las reducciones
//      de la fila, no la aritmetica: medido y visto en el SASS, 28-09)
//
// Organizacion: un bloque por token, H/8 hilos (640 para 5120), 8 valores (16 bytes) por hilo.
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef H
#define H 5120
#endif
#ifndef SUMA
#define SUMA 3            // la que se usa (PN156); 0-2 quedan para medir
#endif
#ifndef GEMMA
#define GEMMA 1          // peso efectivo 1 + w (GemmaRMSNorm)
#endif
#define NT (H / 8)
#define NW (NT / 32)

__device__ __forceinline__ float red_max(float v, float* sm) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    if (lane == 0) sm[w] = v;
    __syncthreads();
    if (w == 0) {
        v = lane < NW ? sm[lane] : 0.f;
#pragma unroll
        for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
        if (lane == 0) sm[0] = v;
    }
    __syncthreads();
    const float r = sm[0];
    __syncthreads();
    return r;
}

template <typename T>
__device__ __forceinline__ T red_sum(T v, T* sm) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    if (lane == 0) sm[w] = v;
    __syncthreads();
    if (w == 0) {
        v = lane < NW ? sm[lane] : (T)0;
#pragma unroll
        for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
        if (lane == 0) sm[0] = v;
    }
    __syncthreads();
    return sm[0];
}

__device__ __forceinline__ int8_t s8_de_h(__half v) {     // round to nearest even, saturado
    short r;
    asm("cvt.rni.sat.s8.f16 %0, %1;" : "=h"(r) : "h"(__half_as_ushort(v)));
    return (int8_t)r;
}

// x: salida de la capa anterior [T, H]; res: residuo [T, H]; res_o: x + res (puede ser res); w: peso de la norma
// q: int8 [T, H] (paso sq); esc: [T] fp32 ya por la escala global de la lineal consumidora
extern "C" __global__ void __launch_bounds__(NT)
sk26_norma_q8(const __half* __restrict__ x, const __half* __restrict__ res, __half* __restrict__ res_o,
              const __half* __restrict__ w,
              int8_t* __restrict__ q, float* __restrict__ esc, const float* __restrict__ gscale,
              int sq, float eps)
{
    __shared__ float smf[NW];
    __shared__ int smi[NW];
    const unsigned t = blockIdx.x, i0 = threadIdx.x * 8;
    const size_t fila = (size_t)t * H;
    uint4 ux = *reinterpret_cast<const uint4*>(x + fila + i0);
    uint4 ur = *reinterpret_cast<const uint4*>(res + fila + i0);
    uint4 uw = *reinterpret_cast<const uint4*>(w + i0);
    const __half2* hx = reinterpret_cast<const __half2*>(&ux);
    const __half2* hr = reinterpret_cast<const __half2*>(&ur);
    const __half2* hw = reinterpret_cast<const __half2*>(&uw);

#if SUMA == 0 || SUMA == 3
    // ── todo en fp32 ──
    float r[8], ss = 0.f, am = 0.f;
    uint4 nr;
    __half2* hn = reinterpret_cast<__half2*>(&nr);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 a = __half22float2(hx[j]), b = __half22float2(hr[j]), g = __half22float2(hw[j]);
        const float s0 = a.x + b.x, s1 = a.y + b.y;
        hn[j] = __floats2half2_rn(s0, s1);
        ss = fmaf(s0, s0, fmaf(s1, s1, ss));
        r[2 * j] = s0 * (GEMMA ? 1.f + g.x : g.x);
        r[2 * j + 1] = s1 * (GEMMA ? 1.f + g.y : g.y);
        am = fmaxf(am, fmaxf(fabsf(r[2 * j]), fabsf(r[2 * j + 1])));
    }
    *reinterpret_cast<uint4*>(res_o + fila + i0) = nr;
#if SUMA == 3
    {   // maximo y suma en la misma ronda
        __shared__ float sms[NW];
        const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
            ss += __shfl_xor_sync(0xffffffffu, ss, m);
        }
        if (lane == 0) { smf[wp] = am; sms[wp] = ss; }
        __syncthreads();
        am = lane < NW ? smf[lane] : 0.f;
        ss = lane < NW ? sms[lane] : 0.f;
#pragma unroll
        for (int m = 16; m; m >>= 1) {          // todos los warps reducen lo mismo: sin segunda barrera
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m));
            ss += __shfl_xor_sync(0xffffffffu, ss, m);
        }
    }
#else
    am = red_max(am, smf);
    ss = red_sum(ss, smf);
#endif
    const float amax = fmaxf(am, 1e-10f), mult = 127.f / amax;
    uint2 o;
    int8_t* ob = reinterpret_cast<int8_t*>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) ob[j] = (int8_t)__float2int_rn(r[j] * mult);
    *reinterpret_cast<uint2*>(q + (size_t)t * sq + i0) = o;
    if (threadIdx.x == 0) esc[t] = amax * rsqrtf(ss * (1.f / H) + eps) / 127.f * gscale[0];
#else
    // ── half2 por elemento ──
    __half2 s[4], rg[4], am2 = __float2half2_rn(0.f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        s[j] = __hadd2(hx[j], hr[j]);
        rg[j] = GEMMA ? __hfma2(s[j], hw[j], s[j]) : __hmul2(s[j], hw[j]);   // s*(1+w) = s*w + s
        am2 = __hmax2(am2, __habs2(rg[j]));
    }
    *reinterpret_cast<uint4*>(res_o + fila + i0) = *reinterpret_cast<uint4*>(s);
    const float amax = fmaxf(red_max(fmaxf(__low2float(am2), __high2float(am2)), smf), 1e-10f);
    const __half2 mult = __float2half2_rn(127.f / amax);
    uint2 o;
    int8_t* ob = reinterpret_cast<int8_t*>(&o);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __half2 y = __hmul2(rg[j], mult);
        ob[2 * j] = s8_de_h(__low2half(y));
        ob[2 * j + 1] = s8_de_h(__high2half(y));
    }
    *reinterpret_cast<uint2*>(q + (size_t)t * sq + i0) = o;
#if SUMA == 1
    // sum r^2 desde el int8: DP4A (camino entero, 4 MAC por instruccion)
    int acc = __dp4a((int)o.x, (int)o.x, 0);
    acc = __dp4a((int)o.y, (int)o.y, acc);
    const int tot = red_sum(acc, smi);
    const float ss = (float)tot * (amax / 127.f) * (amax / 127.f);   // aprox. de sum (r*g)^2
#else
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 v = __half22float2(rg[j]);
        ss = fmaf(v.x, v.x, fmaf(v.y, v.y, ss));
    }
    ss = red_sum(ss, smf);
#endif
    // OJO: aca la suma es de (r*g)^2, no de r^2: con el peso plegado (g = 1, idiotSavant) es lo mismo
    if (threadIdx.x == 0) esc[t] = amax * rsqrtf(ss * (1.f / H) + eps) / 127.f * gscale[0];
#endif
}
