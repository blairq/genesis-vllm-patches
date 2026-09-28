// SK-25: la entrada de o_proj / out_proj de idiotSavant v2 en UN kernel (PN154): la operacion por cabeza
// que la produce + Hadamard de una cabeza + cuantizacion int8 por token para Marlin W4A8.
//
//   MODO 0 (GDN, out_proj):  y = RMSNorm(x) * w * silu(z)   por cabeza de 128  (RMSNormGated, norm antes)
//   MODO 1 (atencion, o_proj): y = x * sigmoid(g)           por cabeza de 256  (compuerta de salida)
//
// Antes eran 4 kernels (la operacion, per_token_quant_int8, la escala por la global, y la Hadamard en
// torch): ahora se lee x (y z/g) una vez y se escribe el int8 y la escala.
//
// Organizacion: un bloque de hilos por token, un warp por cabeza (PORW cabezas por warp si no alcanzan
// los warps). Cada hilo tiene E = D/32 valores consecutivos de la cabeza: i = lane*E + r. Hadamard de
// Sylvester = mariposas sobre los bits de i: los bajos (r) dentro del hilo y los 5 altos (lane) con
// shfl_xor. Todo en fp32; 1/sqrt(D) al final.
//
// Numerica: y se redondea a fp16 (como la salida de la norma / del producto en vLLM) antes de rotar; la
// cuantizacion es la de per_token_quant_int8 (absmax = max(max|v|, 1e-10), q = round(v*127/absmax)),
// y la escala sale ya multiplicada por input_global_scale.
//
// Defines: MODO, D (64..256, potencia de 2), NW (warps por bloque), PORW (cabezas por warp).

#include <cuda_fp16.h>
#include <stdint.h>

#ifndef MODO
#define MODO 0
#endif
#ifndef D
#define D 128
#endif
#ifndef NW
#define NW 24
#endif
#ifndef PORW
#define PORW 1
#endif
#ifndef W32
#define W32 0
#endif
#define E (D / 32)
#if W32
typedef float tpeso;
__device__ __forceinline__ float peso(const float* w, int i) { return w[i]; }
#else
typedef __half tpeso;
__device__ __forceinline__ float peso(const __half* w, int i) { return __half2float(w[i]); }
#endif

__device__ __forceinline__ void cargar(const __half* p, float* v) {
#if E == 8
    uint4 u = *reinterpret_cast<const uint4*>(p);
    const __half* h = reinterpret_cast<const __half*>(&u);
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] = __half2float(h[r]);
#elif E == 4
    uint2 u = *reinterpret_cast<const uint2*>(p);
    const __half* h = reinterpret_cast<const __half*>(&u);
#pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = __half2float(h[r]);
#else
#pragma unroll
    for (int r = 0; r < E; ++r) v[r] = __half2float(p[r]);
#endif
}

// x: [T, NH*D] con paso de fila sx; a: z (MODO 0) o g (MODO 1), paso sa; w: peso de la norma [D]
// (fp16, o fp32 con W32=1; en MODO 1 no se lee)
extern "C" __global__ void __launch_bounds__(NW * 32)
sk25_cabeza_had_q8(const __half* __restrict__ x, const __half* __restrict__ a, const tpeso* __restrict__ w,
                   int8_t* __restrict__ q, float* __restrict__ esc, const float* __restrict__ gscale,
                   int NH, int sx, int sa, int sq, float eps)
{
    const unsigned t = blockIdx.x;
    const unsigned wp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float v[PORW][E];
    float amax = 0.f;

#pragma unroll
    for (unsigned p = 0; p < PORW; ++p) {
        const unsigned h = wp + p * NW;
        if (h < (unsigned)NH) {
            const unsigned col = h * D + lane * E;
            float xv[E], av[E];
            cargar(x + (size_t)t * (unsigned)sx + col, xv);
            cargar(a + (size_t)t * (unsigned)sa + col, av);
#if MODO == 0
            float ss = 0.f;
#pragma unroll
            for (int r = 0; r < E; ++r) ss += xv[r] * xv[r];
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
            const float rs = rsqrtf(ss * (1.f / D) + eps);
#pragma unroll
            for (int r = 0; r < E; ++r) {
                const float zf = av[r];
                const float y = xv[r] * rs * peso(w, lane * E + r) * (zf / (1.f + __expf(-zf)));
                v[p][r] = __half2float(__float2half_rn(y));
            }
#else
#pragma unroll
            for (int r = 0; r < E; ++r) {
                const float y = xv[r] / (1.f + __expf(-av[r]));
                v[p][r] = __half2float(__float2half_rn(y));
            }
#endif
            // mariposas dentro del hilo (bits bajos del indice)
#pragma unroll
            for (int hh = 1; hh < E; hh <<= 1)
#pragma unroll
                for (int r = 0; r < E; ++r)
                    if (!(r & hh)) {
                        float c0 = v[p][r], c1 = v[p][r + hh];
                        v[p][r] = c0 + c1;
                        v[p][r + hh] = c0 - c1;
                    }
            // mariposas entre hilos (bits del lane)
#pragma unroll
            for (int m = 1; m < 32; m <<= 1)
#pragma unroll
                for (int r = 0; r < E; ++r) {
                    float o = __shfl_xor_sync(0xffffffffu, v[p][r], m);
                    v[p][r] = (lane & m) ? (o - v[p][r]) : (v[p][r] + o);
                }
            const float nrm = rsqrtf((float)D);
#pragma unroll
            for (int r = 0; r < E; ++r) {
                v[p][r] *= nrm;
                amax = fmaxf(amax, fabsf(v[p][r]));
            }
        }
    }

#pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    __shared__ float sm[NW];
    if (lane == 0) sm[wp] = amax;
    __syncthreads();
    if (wp == 0) {
        float b = lane < NW ? sm[lane] : 0.f;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1)
            b = fmaxf(b, __shfl_xor_sync(0xffffffffu, b, m));
        if (lane == 0) sm[0] = b;
    }
    __syncthreads();
    const float absmax = fmaxf(sm[0], 1e-10f);
    const float mult = 127.f / absmax;
    if (threadIdx.x == 0) esc[t] = absmax / 127.f * gscale[0];

    int8_t* qf = q + (size_t)t * (unsigned)sq;
#pragma unroll
    for (unsigned p = 0; p < PORW; ++p) {
        const unsigned h = wp + p * NW;
        if (h < (unsigned)NH) {
            int8_t ob[E];
#pragma unroll
            for (int r = 0; r < E; ++r) ob[r] = (int8_t)roundf(v[p][r] * mult);
#if E == 8
            *reinterpret_cast<uint2*>(qf + h * D + lane * E) = *reinterpret_cast<uint2*>(ob);
#elif E == 4
            *reinterpret_cast<uint32_t*>(qf + h * D + lane * E) = *reinterpret_cast<uint32_t*>(ob);
#else
#pragma unroll
            for (int r = 0; r < E; ++r) qf[h * D + lane * E + r] = ob[r];
#endif
        }
    }
}
