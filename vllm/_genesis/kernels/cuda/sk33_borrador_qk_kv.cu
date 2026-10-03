// SK-33: de la salida de qkv del borrador DFlash2 a la atencion, en UN kernel (PN161). Por capa reemplaza:
//   triton_per_fused (q_norm + k_norm + rope neox, inductor) + sk_fwht128_f16 x2 (PN126, q y k)
//   + _reshape_cache_per_token_head (KV int8 por token-cabeza de vLLM)
// Un warp por cabeza de 128 (4 dims por lane, d = 4*lane + e: el mismo reparto que sk_fwht128):
//   cabezas q:  norma -> rope -> FWHT -> q_out (fp16)
//   cabezas kv: norma -> rope -> FWHT de k -> int8 + escala a la cache; v -> int8 + escala a la cache
// Numerica de cada paso igual a la de vLLM/PN126:
//   norma + rope, como las compila inductor (norma y rope en un kernel, los .to(fp16) intermedios elididos):
//          y = x * rsqrt(mean(x^2) + eps) * w  en fp32, sin redondear
//          o1 = fp16(y1*cos - y2*sin), o2 = fp16(y2*cos + y1*sin)  (neox; la pareja de d esta en la lane ^ 16;
//          cos/sin en fp32: la tabla del servidor es fp16 y se pasa a float, exacto)
//   FWHT   la entera de sk_fwht128_f16 (Q14, mariposas en int32, 1/sqrt(128) en Q20)
//   cache  s = max(absmax/127, 1e-6), q = clamp(redondeo alejado del cero de x/s)   (IS_INT_QUANT de vLLM)
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef ROT
#define ROT 1
#endif

#define LLENO 0xffffffffu

__device__ __forceinline__ float suma_warp(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(LLENO, v, o);
    return v;
}
__device__ __forceinline__ float max_warp(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(LLENO, v, o));
    return v;
}

__device__ __forceinline__ void cargar4(const __half* p, float* x) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const __half2* h = reinterpret_cast<const __half2*>(&u);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
    x[0] = a.x; x[1] = a.y; x[2] = b.x; x[3] = b.y;
}
__device__ __forceinline__ void guardar4(__half* p, const float* x) {
    uint2 u;
    __half2* h = reinterpret_cast<__half2*>(&u);
    h[0] = __floats2half2_rn(x[0], x[1]); h[1] = __floats2half2_rn(x[2], x[3]);
    *reinterpret_cast<uint2*>(p) = u;
}
__device__ __forceinline__ float r16(float v) { return __half2float(__float2half_rn(v)); }

// norma + rope + FWHT de una cabeza; x entra crudo (fp16 -> float) y sale redondeado a fp16
__device__ __forceinline__ void cabeza(float* x, const __half* __restrict__ w, float eps, float c[4], float s[4],
                                       const int* __restrict__ signos, unsigned lane) {
    const float ss = suma_warp(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    const float r = rsqrtf(ss * (1.f / 128.f) + eps);
    float wv[4];
    cargar4(w + lane * 4, wv);
#pragma unroll
    for (int e = 0; e < 4; ++e) x[e] = x[e] * r * wv[e];           // sin redondear: inductor funde norma y rope
    const bool alto = lane >= 16;                         // dims 64..127: x2
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const float p = __shfl_xor_sync(LLENO, x[e], 16);
        x[e] = alto ? r16(x[e] * c[e] + p * s[e]) : r16(x[e] * c[e] - p * s[e]);   // un redondeo, al final
    }
#if ROT
    int xi[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) xi[e] = (int)roundf(x[e] * 16384.0f) * signos[lane * 4 + e];
    {
        int a0 = xi[0], a1 = xi[1];
        xi[0] = a0 + a1; xi[1] = a0 - a1;
        int a2 = xi[2], a3 = xi[3];
        xi[2] = a2 + a3; xi[3] = a2 - a3;
        a0 = xi[0]; a2 = xi[2];
        xi[0] = a0 + a2; xi[2] = a0 - a2;
        a1 = xi[1]; a3 = xi[3];
        xi[1] = a1 + a3; xi[3] = a1 - a3;
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool hi = (lane & m) != 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int t = __shfl_xor_sync(LLENO, xi[e], m);
            xi[e] = hi ? (t - xi[e]) : (xi[e] + t);
        }
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int y = (int)(((long long)xi[e] * 92682LL + 524288LL) >> 20);
        x[e] = r16((float)y * (1.0f / 16384.0f));
    }
#endif
}

// cuantizacion por token-cabeza y escritura de 4 bytes + la escala (lane 0)
// los shuffles van SIEMPRE (convergentes); "escribe" solo protege los stores: con un return antes, el compilador
// no podia probar que el warp estaba convergente y llamaba a la subrutina lenta de SHFL con WARPSYNC
__device__ __forceinline__ void a_cache(const float* x, int8_t* dst, float* esc, unsigned lane, bool escribe) {
    const float am = max_warp(fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3]))));
    const float sc = fmaxf(am / 127.f, 1e-6f), inv = 1.f / sc;
    char4 q;
    int8_t* qb = reinterpret_cast<int8_t*>(&q);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        float v = x[e] * inv;
        v = v >= 0.f ? v + 0.5f : v - 0.5f;
        v = fminf(fmaxf(v, -128.f), 127.f);
        qb[e] = (int8_t)(int)v;                              // trunca hacia cero, como el store de triton
    }
    if (escribe) {
        *reinterpret_cast<char4*>(dst + lane * 4) = q;
        if (lane == 0) *esc = sc;
    }
}

extern "C" __global__ void __launch_bounds__(128)
sk33_qk_kv(const __half* __restrict__ qkv, int sq,              // [T, NHQ*128 + 2*NKV*128]
           const int64_t* __restrict__ pos,                     // [T]
           const __half* __restrict__ wq, const __half* __restrict__ wk, float eps,
           const float* __restrict__ cs,                        // cos_sin_cache [P, 128] en float: cos 0..63 | sin 64..127
           const int* __restrict__ signos,
           int NHQ, int NKV,
           __half* __restrict__ qo, int so,                     // [T, NHQ*128]
           const int64_t* __restrict__ slots, int BS,           // slot_mapping (nullptr: sin escritura)
           int8_t* __restrict__ kc, int64_t kcb, int64_t kcs, int64_t kch,
           int8_t* __restrict__ vc, int64_t vcb, int64_t vcs, int64_t vch,
           float* __restrict__ ks, int64_t ksb, int64_t kss, int64_t ksh,
           float* __restrict__ vs, int64_t vsb, int64_t vss, int64_t vsh)
{
    const unsigned lane = threadIdx.x & 31;
    const int t = blockIdx.x;
    const int u0 = blockIdx.y * 4 + (threadIdx.x >> 5);
    // SIN ramas alrededor de los shuffles (norma, rope, FWHT, maximo de la cuantizacion): cada warp elige sus punteros
    // con selects y hace siempre la misma secuencia; solo los stores van condicionados. Con return / if por tipo de
    // cabeza el compilador no podia probar convergencia y cada SHFL era una llamada con WARPSYNC (68 en el SASS).
    const bool valido = u0 < NHQ + NKV;
    const int u = valido ? u0 : 0;
    const bool esq = u < NHQ;
    const int h = esq ? 0 : u - NHQ;                         // cabeza KV (las q usan 0: se calcula y no se guarda)
    const __half* fila = qkv + (size_t)t * sq;

    // cos/sin de las dims de esta lane: indice d mod 64
    const float* cp = cs + (size_t)pos[t] * 128;
    const float4 c4 = *reinterpret_cast<const float4*>(cp + (lane & 15) * 4);
    const float4 s4 = *reinterpret_cast<const float4*>(cp + 64 + (lane & 15) * 4);
    float c[4] = {c4.x, c4.y, c4.z, c4.w}, s[4] = {s4.x, s4.y, s4.z, s4.w};

    float x[4], v[4];
    cargar4(fila + (esq ? u * 128 : NHQ * 128 + h * 128) + lane * 4, x);
    cargar4(fila + (NHQ + NKV) * 128 + h * 128 + lane * 4, v);
    cabeza(x, esq ? wq : wk, eps, c, s, signos, lane);
    if (valido && esq) guardar4(qo + (size_t)t * so + u * 128 + lane * 4, x);
    // slot en 32 bits (bloques * BS << 2^31): la division de 64 bits era una subrutina emulada
    const int sl = slots != nullptr ? (int)slots[t] : -1;
    const bool escribe = valido && !esq && sl >= 0;          // uniforme en el warp (un token y una cabeza por warp)
    const int b = sl >= 0 ? sl / BS : 0, i = sl >= 0 ? sl - b * BS : 0;
    a_cache(x, kc + (int64_t)b * kcb + (int64_t)i * kcs + h * kch, ks + (int64_t)b * ksb + (int64_t)i * kss + h * ksh, lane, escribe);
    a_cache(v, vc + (int64_t)b * vcb + (int64_t)i * vcs + h * vch, vs + (int64_t)b * vsb + (int64_t)i * vss + h * vsh, lane, escribe);
}
