// SK-23: SiluAndMul + Hadamard por bloques de 512 + cuantizacion int8 por token, en UN kernel.
//
// Es la entrada de down_proj en el checkpoint rotado (PN148). Hoy son tres pasadas por memoria
// (SiluAndMul/Hadamard escribe fp16, per_token_quant_int8 lo relee y escribe int8, y un kernel mas
// multiplica la escala por la global): ~9N bytes por token. Aca los datos nunca salen de los
// registros: se lee gate|up una vez (4N bytes) y se escribe el int8 (N) y la escala.
//
// Organizacion: un bloque de hilos por token, un warp por bloque de 512 (PORW bloques por warp si
// no alcanzan los warps). Cada hilo tiene 16 valores consecutivos: indice i = lane*16 + r.
// Hadamard de Sylvester = mariposas sobre los 9 bits de i: los 4 bajos (r) dentro del hilo y los
// 5 altos (lane) con shfl_xor. Todo en fp32; 1/sqrt(512) al final.
//
// Numerica: silu(g)*u se redondea a fp16 como la salida del SiluAndMul de vLLM; la cuantizacion es
// la de per_token_quant_int8 de vLLM: absmax = max(max|v|, 1e-10), q = round(v * 127/absmax) con
// redondeo alejado del cero, escala = absmax/127 (y aca ya multiplicada por input_global_scale).
//
// Defines: NW (warps por bloque de hilos), PORW (bloques de 512 por warp).

#include <cuda_fp16.h>
#include <stdint.h>

#ifndef NW
#define NW 17
#endif
#ifndef PORW
#define PORW 1
#endif

extern "C" __global__ void __launch_bounds__(NW * 32)
sk23_silu_had_q8(const __half* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ esc,
                 const float* __restrict__ gscale, int N, int sx, int sq)
{
    const unsigned t = blockIdx.x;
    const unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned nb = (unsigned)N >> 9;
    const __half* fila = x + (size_t)t * (unsigned)sx;
    float v[PORW][16];
    float amax = 0.f;

#pragma unroll
    for (unsigned p = 0; p < PORW; ++p) {
        const unsigned b = w + p * NW;
        if (b < nb) {
            const unsigned base = (b << 9) + (lane << 4);
            uint4 g4[2], u4[2];
            g4[0] = *reinterpret_cast<const uint4*>(fila + base);
            g4[1] = *reinterpret_cast<const uint4*>(fila + base + 8);
            u4[0] = *reinterpret_cast<const uint4*>(fila + N + base);
            u4[1] = *reinterpret_cast<const uint4*>(fila + N + base + 8);
            const __half* g = reinterpret_cast<const __half*>(g4);
            const __half* u = reinterpret_cast<const __half*>(u4);
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                float gf = __half2float(g[r]);
                float y = gf / (1.f + __expf(-gf)) * __half2float(u[r]);
                v[p][r] = __half2float(__float2half_rn(y));
            }
            // mariposas dentro del hilo (bits 0-3 del indice)
#pragma unroll
            for (int h = 1; h < 16; h <<= 1)
#pragma unroll
                for (int r = 0; r < 16; ++r)
                    if (!(r & h)) {
                        float a = v[p][r], c = v[p][r + h];
                        v[p][r] = a + c;
                        v[p][r + h] = a - c;
                    }
            // mariposas entre hilos (bits 4-8 = bits del lane)
#pragma unroll
            for (int m = 1; m < 32; m <<= 1)
#pragma unroll
                for (int r = 0; r < 16; ++r) {
                    float o = __shfl_xor_sync(0xffffffffu, v[p][r], m);
                    v[p][r] = (lane & m) ? (o - v[p][r]) : (v[p][r] + o);
                }
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                v[p][r] *= 0.044194173824159216f;          // 1/sqrt(512)
                amax = fmaxf(amax, fabsf(v[p][r]));
            }
        }
    }

    // maximo del token: warp y despues bloque de hilos
#pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    __shared__ float sm[NW];
    if (lane == 0) sm[w] = amax;
    __syncthreads();
    if (w == 0) {
        float a = lane < NW ? sm[lane] : 0.f;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1)
            a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
        if (lane == 0) sm[0] = a;
    }
    __syncthreads();
    const float absmax = fmaxf(sm[0], 1e-10f);
    const float mult = 127.f / absmax;
    if (threadIdx.x == 0) esc[t] = absmax / 127.f * gscale[0];

    int8_t* qf = q + (size_t)t * (unsigned)sq;
#pragma unroll
    for (unsigned p = 0; p < PORW; ++p) {
        const unsigned b = w + p * NW;
        if (b < nb) {
            uint4 o;
            int8_t* ob = reinterpret_cast<int8_t*>(&o);
#pragma unroll
            for (int r = 0; r < 16; ++r)
                ob[r] = (int8_t)roundf(v[p][r] * mult);
            *reinterpret_cast<uint4*>(qf + (b << 9) + (lane << 4)) = o;
        }
    }
}
