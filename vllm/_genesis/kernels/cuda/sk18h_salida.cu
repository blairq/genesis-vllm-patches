// SPDX-License-Identifier: Apache-2.0
//
// SK-18h/salida — de los acumuladores de la union (grupos de tramos, int64) a la salida
// fp16 de la atencion, todo entero:
//   valor = O * 2^(ev + VSH) / (32767 * S),  O = sum_g Og,  S = sum_g Sg
//   r = (|O| << 20) / S ;  u = r * 32769 (~ r * 2^30 / 32767) ;  valor = u * 2^(ev+VSH-50)
// y u -> bits fp16 con clz. Grilla: x = token, y = cabeza Q; 32 lanes = grupos de 8 dims.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256

extern "C" __global__ void __launch_bounds__(32)
sk18h_salida(
    const long long* __restrict__ Og,       // [NG, R, 256]
    const long long* __restrict__ Sg,       // [NG, R]
    const int* __restrict__ refs,           // [2] ek, ev
    unsigned short* __restrict__ out,       // [B*L, NH*G, 256] bits fp16
    int NG, int R, int L, int NH, int G, int MB, int VSH)
{
    const int t = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x;
    const int b = t / L, j = t % L;
    const int kvh = qh / G, g = qh % G;
    const size_t row = ((size_t)b * NH + kvh) * MB + j * G + g;
    const size_t dst = ((size_t)t * NH * G + qh) * QD;
    const int E0 = refs[1] + VSH - 50;

    long long S = 0;
    for (int k = 0; k < NG; ++k) S += Sg[(size_t)k * R + row];
    if (S < 1) S = 1;
    // (a << 20) / S sin division de 64 bits (emulada: cientos de instrucciones, 8 por hilo): cociente
    // estimado con el reciproco en double (error relativo ~2^-52, y el cociente entra holgado en 52
    // bits) y corregido a la baja o al alza: da el MISMO piso entero, bit a bit (28-09).
    const unsigned long long Su = (unsigned long long)S;
    const double inv = 1.0 / (double)Su;
    long long Oe[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) Oe[e] = 0;
    for (int k = 0; k < NG; ++k) {
        const long long* p = Og + ((size_t)k * R + row) * QD + lane * 8;
#pragma unroll
        for (int e = 0; e < 8; ++e) Oe[e] += p[e];
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = lane * 8 + e;
        const long long O = Oe[e];
        const int neg = O < 0;
        const unsigned long long a = (unsigned long long)(neg ? -O : O);
        const unsigned long long x = a << 20;
        unsigned long long r = (unsigned long long)((double)x * inv);
        unsigned long long pr = r * Su;
        if (pr > x) { r -= 1; pr -= Su; }
        if (x - pr >= Su) r += 1;
        out[dst + d] = (unsigned short)q_a_fp16(r * 32769ull, E0, neg);
    }
}
