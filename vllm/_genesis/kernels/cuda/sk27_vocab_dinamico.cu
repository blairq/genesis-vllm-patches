// SK-27: vocabulario dinamico del borrador (PN159 fase 2, estilo NanoSpec arXiv 2605.26444, traducido a SM86).
//
// Mira los tokens que entran al modelo en el paso (prompt y lo aceptado) y agrega a un anillo de D filas
// las de los tokens que NO estan en el subconjunto fijo de PN159 y son del tramo de vocabulario de este
// rango. La fila (int4 GPTQ + escalas del lm_head de PN139, una por token) se lee de memoria pinned del
// host por UVA (PCIe) y se guarda decuantizada en fp16. En decode entran 0-2 filas por paso.
//
// UN bloque: los candidatos se deduplican con un bitmap (atomicOr) y se les asigna ranura en orden de
// llegada; con un solo bloque no hay carreras entre bloques por la misma ranura del anillo.
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef K
#define K 5120
#endif
#ifndef G
#define G 128
#endif
#ifndef DMAX
#define DMAX 2048
#endif
#define KW (K / 8)          // int32 por fila (8 nibbles cada uno)

extern "C" __global__ void __launch_bounds__(1024)
sk27_observar(const void* __restrict__ ids, int n, int ids64,
              const uint32_t* __restrict__ estatico,   // bitmap global: token en el subconjunto fijo
              uint32_t* __restrict__ dinbit,           // bitmap global: token en el anillo
              int ini, int fin,
              const int32_t* __restrict__ hq,          // host pinned [fin-ini, KW]
              const __half* __restrict__ hs,           // host pinned [fin-ini, K/G]
              int32_t* __restrict__ din_ids,           // [D], -1 = vacia
              __half* __restrict__ din_filas,          // [D, K]
              int32_t* __restrict__ cabeza, int D)
{
    __shared__ int cand[DMAX];
    __shared__ int cnt;
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    if (tid == 0) cnt = 0;
    __syncthreads();
    for (unsigned i = tid; i < (unsigned)n; i += nt) {
        const int t = ids64 ? (int)((const int64_t*)ids)[i] : ((const int32_t*)ids)[i];
        if (t < ini || t >= fin) continue;
        const uint32_t b = 1u << (t & 31);
        if (estatico[t >> 5] & b) continue;
        if (atomicOr(&dinbit[t >> 5], b) & b) continue;          // ya estaba (o lo tomo otro hilo)
        const int p = atomicAdd(&cnt, 1);
        if (p < D) cand[p] = t;
        else atomicAnd(&dinbit[t >> 5], ~b);                     // no entra este paso
    }
    __syncthreads();
    const int m = min(cnt, D), h = *cabeza;
    for (int i = tid; i < m; i += nt) {                          // desalojar y asignar ranura
        const int r = (h + i) % D;
        const int prev = din_ids[r];
        if (prev >= 0) atomicAnd(&dinbit[prev >> 5], ~(1u << (prev & 31)));
        din_ids[r] = cand[i];
    }
    __syncthreads();
    const unsigned lane = tid & 31, w = tid >> 5, nw = nt >> 5;
    for (int i = w; i < m; i += nw) {                            // un warp por fila
        const int t = cand[i], r = (h + i) % D;
        const int32_t* q = hq + (size_t)(t - ini) * KW;
        const __half* s = hs + (size_t)(t - ini) * (K / G);
        uint4* o = reinterpret_cast<uint4*>(din_filas + (size_t)r * K);
        for (unsigned j = lane; j < KW; j += 32) {
            const uint32_t v = (uint32_t)q[j];
            const float e = __half2float(s[(j * 8) / G]);
            __half2 x[4];
#pragma unroll
            for (int b = 0; b < 4; ++b)
                x[b] = __floats2half2_rn((float)((int)((v >> (8 * b)) & 15) - 8) * e,
                                         (float)((int)((v >> (8 * b + 4)) & 15) - 8) * e);
            o[j] = *reinterpret_cast<uint4*>(x);
        }
    }
    if (tid == 0) *cabeza = (h + m) % D;
}

// ── SK-27b: logits del anillo. Bloque de 8 warps = 8 ranuras; h en shared por trozos de 256 columnas y
//    hasta MB filas por bloque (grid.y recorre las filas). Cada fila del anillo se lee UNA vez por grid.y.
#ifndef MB
#define MB 16
#endif
extern "C" __global__ void __launch_bounds__(256)
sk27_logits(const __half* __restrict__ h, int M, const __half* __restrict__ filas,
            const int32_t* __restrict__ din_ids, float* __restrict__ out, int D)
{
    __shared__ __align__(16) __half hs[MB][256];
    const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int r = blockIdx.x * 8 + w, m0 = blockIdx.y * MB;
    const int mm = min(MB, M - m0);
    float acc[MB];
#pragma unroll
    for (int i = 0; i < MB; ++i) acc[i] = 0.f;
    for (int k0 = 0; k0 < K; k0 += 256) {
        __syncthreads();
        for (int e = threadIdx.x; e < MB * 32; e += 256) {         // MB filas x 32 uint4 (256 halves)
            const int i = e >> 5, c = e & 31;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (i < mm) v = *reinterpret_cast<const uint4*>(h + (size_t)(m0 + i) * K + k0 + c * 8);
            *reinterpret_cast<uint4*>(&hs[i][c * 8]) = v;
        }
        __syncthreads();
        if (r < D) {
            const uint4 fv = *reinterpret_cast<const uint4*>(filas + (size_t)r * K + k0 + lane * 8);
            const __half2* f2 = reinterpret_cast<const __half2*>(&fv);
#pragma unroll
            for (int i = 0; i < MB; ++i) {
                const __half2* x2 = reinterpret_cast<const __half2*>(&hs[i][lane * 8]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 a = __half22float2(f2[j]), b = __half22float2(x2[j]);
                    acc[i] = fmaf(a.x, b.x, fmaf(a.y, b.y, acc[i]));
                }
            }
        }
    }
    if (r >= D) return;
    const bool vale = din_ids[r] >= 0;
#pragma unroll
    for (int i = 0; i < MB; ++i) {
        float v = acc[i];
#pragma unroll
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0 && i < mm) out[(size_t)(m0 + i) * D + r] = vale ? v : -INFINITY;
    }
}

// ── SK-27c: por fila, top-KT de [KT del fijo | D del anillo], ordenado de mayor a menor. KT rondas de
//    arg-max de bloque (KT = 16: 16 barreras, nada que ordenar entero).
#ifndef KT
#define KT 16
#endif
extern "C" __global__ void __launch_bounds__(256)
sk27_fusion(const __half* __restrict__ vs, const int64_t* __restrict__ is,     // [M, KT] del fijo
            const float* __restrict__ ld, const int32_t* __restrict__ din_ids, int D,
            __half* __restrict__ vo, int64_t* __restrict__ io)                 // [M, KT]
{
    extern __shared__ float cv[];                                               // KT + D
    __shared__ float wv[8];
    __shared__ int wi[8];
    const int m = blockIdx.x, n = KT + D;
    const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int e = threadIdx.x; e < n; e += 256)
        cv[e] = e < KT ? __half2float(vs[(size_t)m * KT + e]) : ld[(size_t)m * D + e - KT];
    __syncthreads();
    for (int t = 0; t < KT; ++t) {
        float bv = -INFINITY; int bi = 0x7fffffff;
        for (int e = threadIdx.x; e < n; e += 256)
            if (cv[e] > bv || (cv[e] == bv && e < bi)) { bv = cv[e]; bi = e; }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { wv[w] = bv; wi[w] = bi; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int j = 1; j < 8; ++j)
                if (wv[j] > bv || (wv[j] == bv && wi[j] < bi)) { bv = wv[j]; bi = wi[j]; }
            vo[(size_t)m * KT + t] = __float2half(bv);
            io[(size_t)m * KT + t] = bi < KT ? is[(size_t)m * KT + bi] : (int64_t)din_ids[bi - KT];
            cv[bi] = -INFINITY;
        }
        __syncthreads();
    }
}
