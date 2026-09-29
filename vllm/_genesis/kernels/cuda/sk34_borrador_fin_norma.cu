// SK-34: cierre de la conv agrupada del borrador DFlash2 (lado 1) + suma del residuo + RMSNorm, en UN kernel (PN162).
//
//   y[t, d] = (b0[d] + c[t, 0, g]) * h[t, d] + [t mod BSQ >= 1] * (b1[d] + c[t, 1, g]) * h[t-1, d],  g = d / 16
//   x = y + residuo;  residuo' = fp16(x);  out = fp16(x * rsqrt(mean(x^2) + eps) * w)
// (fused_add_rms_norm de la IR de vLLM 0.29 tal como la compila inductor: sin redondeos intermedios, el
//  .to(fp16) antes del peso lo elide; b = base_kernel[1], c = los coeficientes del lado 1.)
// Hoy lo hace un kernel de inductor por norma (4,4 us con 9 tokens, 2 por capa + la final): una sola CTA chica
// por fila. Aca un bloque de 320 hilos por fila, 16 valores por hilo en registros (N = 5120), una pasada.
// RY / RC: redondear a fp16 la salida de la conv / los coeficientes (b + c) antes de seguir; los fija la prueba
// contra inductor (tests/proto/sk34_test.py).
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef RY
#define RY 0
#endif
#ifndef RC
#define RC 0
#endif
#ifndef NT
#define NT 640
#endif
#define GS 16
#define VPT (5120 / NT)                          // valores por hilo: NT * VPT = N (5120)

__device__ __forceinline__ float r16(float v) { return __half2float(__float2half_rn(v)); }

__device__ __forceinline__ void cargar8(const __half* p, float* x) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) { const float2 f = __half22float2(h[i]); x[2 * i] = f.x; x[2 * i + 1] = f.y; }
}
__device__ __forceinline__ void guardar8(__half* p, const float* x) {
    uint4 u;
    __half2* h = reinterpret_cast<__half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(x[2 * i], x[2 * i + 1]);
    *reinterpret_cast<uint4*>(p) = u;
}

extern "C" __global__ void __launch_bounds__(NT)
sk34_fin_norma(const __half* __restrict__ h, int shs,                 // salida de la atencion / MLP [T, N]
               const __half* __restrict__ coef, int sc,               // lado 1: [tap 0: G][tap 1: G] por fila
               const __half* __restrict__ base,                       // base_kernel[1]: [2, N]
               const __half* __restrict__ res, int sr,                // residuo [T, N]
               const __half* __restrict__ w, float eps, int N, int BSQ,
               __half* __restrict__ out, int so, __half* __restrict__ res_out, int sro)
{
    __shared__ float red[NT / 32];
    const int t = blockIdx.x;
    const bool prev = (t % BSQ) >= 1;
    const __half* h0 = h + (size_t)t * shs;
    const __half* h1 = h0 - shs;
    const __half* c = coef + (size_t)t * sc;
    const int G = N / GS;
    float x[VPT], wv[VPT];                                           // el peso se carga en la primera ronda
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < VPT / 8; ++k) {
        const int d0 = (k * NT + threadIdx.x) * 8;                   // 8 valores contiguos, un solo grupo de 16
        if (d0 >= N) break;
        const int g = d0 / GS;
        float a[8], b0[8], r[8];
        cargar8(h0 + d0, a);
        cargar8(base + d0, b0);
        cargar8(res + (size_t)t * sr + d0, r);
        cargar8(w + d0, wv + k * 8);
        float c0 = __half2float(c[g]);
        float y[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float k0 = b0[e] + c0;
#if RC
            k0 = r16(k0);
#endif
            y[e] = k0 * a[e];
        }
        if (prev) {
            float p[8], b1[8];
            cargar8(h1 + d0, p);
            cargar8(base + N + d0, b1);
            const float c1 = __half2float(c[G + g]);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float k1 = b1[e] + c1;
#if RC
                k1 = r16(k1);
#endif
#if RY
                y[e] = r16(y[e]);
                y[e] = r16(y[e] + r16(k1 * p[e]));
#else
                y[e] += k1 * p[e];
#endif
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
#if RY
            y[e] = r16(y[e]);
#endif
            const float v = y[e] + r[e];
            x[k * 8 + e] = v;
            ss += v * v;
        }
        guardar8(res_out + (size_t)t * sro + d0, x + k * 8);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 32; ++i) tot += red[i];
    const float rs = rsqrtf(__fdiv_rn(tot, (float)N) + eps);     // division IEEE como inductor (no la aproximada de fast-math)
#pragma unroll
    for (int k = 0; k < VPT / 8; ++k) {
        const int d0 = (k * NT + threadIdx.x) * 8;
        if (d0 >= N) break;
        float o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = x[k * 8 + e] * rs * wv[k * 8 + e];     // inductor elide el .to(fp16) de la IR: un redondeo
        guardar8(out + (size_t)t * so + d0, o);
    }
}
