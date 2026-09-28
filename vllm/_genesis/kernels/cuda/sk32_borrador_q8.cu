// SK-32: cuantizacion int8 por token para las lineales W4A8 del borrador DFlash2 (PN160), en UN kernel.
//
// Hoy cada lineal del borrador corre per_token_quant_int8 (vLLM) + un kernel de inductor que multiplica la
// escala por input_global_scale + Marlin. Delante de down_proj, ademas, act_and_mul. Aca:
//   sk32_q8       x fp16 [T, N]          -> q int8 [T, N], esc = absmax/127 * global
//   sk32_silu_q8  gate|up fp16 [T, 2N]   -> silu(g)*u (redondeado a fp16, como SiluAndMul) -> q, esc
// Numerica de per_token_quant_int8: absmax = max(max|v|, 1e-10), q = round(v * 127/absmax) alejado del cero.
// Un bloque de 256 hilos por token; dos pasadas sobre la fila (la segunda sale de L1/L2); cargas de 16 B.
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float max_bloque(float v, float* sm) {
    const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (lane == 0) sm[w] = v;
    __syncthreads();
    float r = sm[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) r = fmaxf(r, sm[i]);
    return r;
}

__device__ __forceinline__ int8_t q_lejos(float v, float r) {        // redondeo alejado del cero, saturado
    const float a = floorf(fabsf(v) * r + 0.5f);
    const int x = (int)fminf(a, 127.f);
    return (int8_t)(v < 0.f ? -x : x);
}

__device__ __forceinline__ float silu_mul(float g, float u) {
    return __half2float(__float2half_rn(g / (1.f + __expf(-g)) * u));
}

extern "C" __global__ void __launch_bounds__(256)
sk32_q8(const __half* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ esc,
        const float* __restrict__ gscale, int N, int sx, int sq)
{
    __shared__ float sm[8];
    const int t = blockIdx.x;
    const __half* f = x + (size_t)t * sx;
    float m = 0.f;
    for (int c = threadIdx.x; c < N / 8; c += 256) {
        const uint4 u = *reinterpret_cast<const uint4*>(f + c * 8);
        const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float2 v = __half22float2(h[i]); m = fmaxf(m, fmaxf(fabsf(v.x), fabsf(v.y))); }
    }
    const float am = fmaxf(max_bloque(m, sm), 1e-10f), r = 127.f / am;
    int8_t* qf = q + (size_t)t * sq;
    for (int c = threadIdx.x; c < N / 8; c += 256) {
        const uint4 u = *reinterpret_cast<const uint4*>(f + c * 8);
        const __half2* h = reinterpret_cast<const __half2*>(&u);
        uint2 o;
        int8_t* ob = reinterpret_cast<int8_t*>(&o);
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float2 v = __half22float2(h[i]); ob[2 * i] = q_lejos(v.x, r); ob[2 * i + 1] = q_lejos(v.y, r); }
        *reinterpret_cast<uint2*>(qf + c * 8) = o;
    }
    if (threadIdx.x == 0) esc[t] = am / 127.f * gscale[0];
}

extern "C" __global__ void __launch_bounds__(256)
sk32_silu_q8(const __half* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ esc,
             const float* __restrict__ gscale, int N, int sx, int sq)   // x = [gate (N) | up (N)]
{
    __shared__ float sm[8];
    const int t = blockIdx.x;
    const __half* f = x + (size_t)t * sx;
    float m = 0.f;
    for (int c = threadIdx.x; c < N / 8; c += 256) {
        const uint4 ug = *reinterpret_cast<const uint4*>(f + c * 8);
        const uint4 uu = *reinterpret_cast<const uint4*>(f + N + c * 8);
        const __half2* g = reinterpret_cast<const __half2*>(&ug);
        const __half2* u = reinterpret_cast<const __half2*>(&uu);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 a = __half22float2(g[i]), b = __half22float2(u[i]);
            m = fmaxf(m, fmaxf(fabsf(silu_mul(a.x, b.x)), fabsf(silu_mul(a.y, b.y))));
        }
    }
    const float am = fmaxf(max_bloque(m, sm), 1e-10f), r = 127.f / am;
    int8_t* qf = q + (size_t)t * sq;
    for (int c = threadIdx.x; c < N / 8; c += 256) {
        const uint4 ug = *reinterpret_cast<const uint4*>(f + c * 8);
        const uint4 uu = *reinterpret_cast<const uint4*>(f + N + c * 8);
        const __half2* g = reinterpret_cast<const __half2*>(&ug);
        const __half2* u = reinterpret_cast<const __half2*>(&uu);
        uint2 o;
        int8_t* ob = reinterpret_cast<int8_t*>(&o);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 a = __half22float2(g[i]), b = __half22float2(u[i]);
            ob[2 * i] = q_lejos(silu_mul(a.x, b.x), r); ob[2 * i + 1] = q_lejos(silu_mul(a.y, b.y), r);
        }
        *reinterpret_cast<uint2*>(qf + c * 8) = o;
    }
    if (threadIdx.x == 0) esc[t] = am / 127.f * gscale[0];
}

// sk32_conv_q8: preparacion de la conv agrupada del borrador (DFlash2, taps = 2) + cuantizacion, en un kernel.
//   y[t, d] = (b0[d] + c[t, 0, g]) * h[t, d] + [t mod BSQ >= 1] * (b1[d] + c[t, 1, g]) * h[t-1, d],  g = d / GS
// (c = los coeficientes del lado 0 de kernel_projection, [T, 2 lados, 2 taps, G]; b = base_kernel[0], [2, N]).
// y se redondea a fp16 como la salida de inductor y se cuantiza como per_token_quant_int8.
#ifndef GS
#define GS 16
#endif
__device__ __forceinline__ float conv_y(const __half* h0, const __half* h1, const __half* b0, const __half* b1,
                                        const __half* c, int G, int d, bool prev) {
    const int g = d / GS;
    float y = (__half2float(b0[d]) + __half2float(c[g])) * __half2float(h0[d]);
    if (prev) y += (__half2float(b1[d]) + __half2float(c[G + g])) * __half2float(h1[d]);
    return __half2float(__float2half_rn(y));
}

extern "C" __global__ void __launch_bounds__(256)
sk32_conv_q8(const __half* __restrict__ h, const __half* __restrict__ coef, const __half* __restrict__ base,
             int8_t* __restrict__ q, float* __restrict__ esc, const float* __restrict__ gscale,
             int N, int G, int BSQ, int sh, int sc, int sq)
{
    __shared__ float sm[8];
    const int t = blockIdx.x;
    const bool prev = (t % BSQ) >= 1;
    const __half* h0 = h + (size_t)t * sh;
    const __half* h1 = prev ? h + (size_t)(t - 1) * sh : h0;
    const __half* c = coef + (size_t)t * sc;             // lado 0: [tap 0: G][tap 1: G]
    const __half* b0 = base;
    const __half* b1 = base + N;
    float m = 0.f;
    for (int d = threadIdx.x; d < N; d += 256) m = fmaxf(m, fabsf(conv_y(h0, h1, b0, b1, c, G, d, prev)));
    const float am = fmaxf(max_bloque(m, sm), 1e-10f), r = 127.f / am;
    int8_t* qf = q + (size_t)t * sq;
    for (int d = threadIdx.x; d < N; d += 256) qf[d] = q_lejos(conv_y(h0, h1, b0, b1, c, G, d, prev), r);
    if (threadIdx.x == 0) esc[t] = am / 127.f * gscale[0];
}
