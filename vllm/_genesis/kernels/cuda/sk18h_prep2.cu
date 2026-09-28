// SPDX-License-Identifier: Apache-2.0
//
// SK-18h/prep2 — como prep + rotacion Hadamard ENTERA de q (antes PN126 en fp16):
//   q~ = H (s * q) / 16 con los signos s de PN126, Walsh-Hadamard en lanes + mariposa.
// SK-18h/prep — queries del decode en UN lanzamiento, todo entero: q fp16 (bits) ->
// Q int8 por token-cabeza en las filas del kernel, lim (causal), mqb y dcap.
//   mqb = kref * sq * 2^ZSH * 2^24 / (32767 * 16 * ln2),  sq = max|q| / 127, kref = 2^ek
//       = (maxQ16 * MQ) >> (56 - (ZSH + 8) - ek),        MQ = round(2^56 / C)
//   dcap = ceil(2^28 / mqb)
// Filas: (b, kvh, j*G + g). Filas >= L*G de cada (b, kvh) quedan con lim = -1.
// Grilla: x = token (b*L + j), y = cabeza Q; 32 lanes = 32 grupos de 8 dimensiones.
#include <cuda_pipeline.h>
#include "sk18h_comun.cuh"

#define QD 256
#ifndef ARBOL
#define ARBOL 0
#endif
#ifndef MQ
#define MQ 1561327149LL
#endif
// QSF=1: ademas, la escala de fila en float para SK-30 (softmax en fp32, log2):
//   qs = (mx / 256 / 127) * 2^(ek-15) * (1/16) * log2(e)     (mx en Q8, 1/16 = 1/sqrt(256))
#ifndef QSF
#define QSF 0
#endif

extern "C" __global__ void __launch_bounds__(32)
sk18h_prep2(
    const unsigned short* __restrict__ q,   // [B*L, NH*G, 256] bits fp16
    const int* __restrict__ seq,            // [B]
    const int* __restrict__ refs,           // [2] ek, ev
    const int* __restrict__ signos,         // [256] +-1
    signed char* __restrict__ Qb,           // [B, NH, MB, 256]
    int* __restrict__ lim,                  // [B, NH, MB]
    int* __restrict__ mqb,
    int* __restrict__ dcap,
#if ARBOL
    const int* __restrict__ anc,            // [B*L] bits de ancestros de cada token (0 para el ancla)
    int* __restrict__ abase,                // [B, NH, MB]
    int* __restrict__ amask,
#endif
#if QSF
    float* __restrict__ qsf,                // [B, NH, MB]
#endif
    int L, int NH, int G, int MB, int ZSH, int QS)   // QS = paso de fila de q (elementos)
{
    const int t = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x;
    const int b = t / L, j = t % L;
    const int kvh = qh / G, g = qh % G;
    const size_t base_bh = ((size_t)b * NH + kvh) * MB;
    const size_t row = base_bh + j * G + g;
    const size_t src = (size_t)t * QS + (size_t)qh * QD;

    // q en Q8 con signos, Walsh-Hadamard entero, / 16
    int x[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = lane * 8 + e;
        x[e] = fp16_a_q8s(q[src + d]) * signos[d];
    }
    fwht_lanes(x, lane);
    unsigned mx = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        x[e] = (x[e] + 8) >> 4;                             // / sqrt(256), redondeado (el truncado sesga q: b.k cambia con la key)
        const unsigned a = (unsigned)(x[e] < 0 ? -x[e] : x[e]);
        mx = mx > a ? mx : a;
    }
    mx = max_lanes(mx);                                     // Q8
    const long long rec = mx ? ((127LL << 24) / (long long)mx) : 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int d = lane * 8 + e;
        const long long a = x[e] < 0 ? -(long long)x[e] : (long long)x[e];
        long long v = (a * rec + (1LL << 23)) >> 24;
        v = v < 127 ? v : 127;
        Qb[row * QD + d] = (signed char)(x[e] < 0 ? -v : v);
    }
    if (lane == 0) {
        const int ek = refs[0];
        const int sh = 56 - (ZSH + 8) - ek - 8;             // mx en Q8 (antes Q16)
        long long m = sh > 0 ? (((long long)mx * MQ) >> sh) : (((long long)mx * MQ) << (-sh));
        m = m < 1 ? 1 : (m > 2147483647LL ? 2147483647LL : m);
        long long dc = ((1LL << 28) + m - 1) / m;
        dc = dc < (1LL << 30) ? dc : (1LL << 30);
        lim[row] = seq[b] - L + j;
#if QSF
        qsf[row] = (float)mx * (1.f / (256.f * 127.f)) * exp2f((float)(ek - 15)) * (1.f / 16.f) * 1.4426950408889634f;
#endif
        mqb[row] = (int)m;
        dcap[row] = (int)dc;
#if ARBOL
        abase[row] = seq[b] - L;
        amask[row] = anc[t];
#endif
        if (j == L - 1 && g == G - 1)
            for (int r = L * G; r < MB; ++r) {
                lim[base_bh + r] = -1; mqb[base_bh + r] = 1; dcap[base_bh + r] = 0;
#if ARBOL
                abase[base_bh + r] = -1; amask[base_bh + r] = 0;
#endif
#if QSF
                qsf[base_bh + r] = 0.f;
#endif
            }
    }
}
