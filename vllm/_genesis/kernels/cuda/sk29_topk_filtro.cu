// SK-29: top-16 por fila de los logits del borrador (PN159), un bloque por fila, sin insercion serial.
//
// Los trozos de 32 que contienen algun elemento del top-16 tienen su maximo >= e16, y hay a lo sumo 16 trozos
// asi: son los 16 de maximo mas alto. Entonces:
//   1. maximo de cada trozo (la fila se lee una vez, uint4)
//   2. top-16 de los maximos -> 16 trozos
//   3. top-16 de los 16*32 candidatos (+ los D logits del anillo de la fase 2)
// Los top-16 son sort bitonico de 32 entre lanes (shfl_xor, sin barreras) y union con una lista de 16:
// max(A[i], B[15-i]) es bitonica y la ordenan 4 etapas mas.
#include <cuda_fp16.h>
#include <stdint.h>

#define LLENO 0xffffffffu
#define NW 16

__device__ __forceinline__ bool mayor(float a, int ia, float b, int ib) {   // orden total: valor, despues indice
    return a > b || (a == b && ia < ib);
}

__device__ __forceinline__ void ordenar32(float& v, int& i) {              // descendente en las 32 lanes
    const unsigned lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
        for (int j = k >> 1; j; j >>= 1) {
            const float pv = __shfl_xor_sync(LLENO, v, j);
            const int pi = __shfl_xor_sync(LLENO, i, j);
            const bool abajo = (lane & j) == 0, desc = (lane & k) == 0 || k == 32;
            const bool quiero_mayor = abajo == desc;
            if (mayor(pv, pi, v, i) == quiero_mayor) { v = pv; i = pi; }
        }
}

// L (lanes 0..15, descendente) <- top-16 de L U B (B descendente en 32 lanes)
__device__ __forceinline__ void unir16(float& lv, int& li, float bv, int bi) {
    const unsigned lane = threadIdx.x & 31;
    const float cv = __shfl_sync(LLENO, bv, (15 - lane) & 31);
    const int ci = __shfl_sync(LLENO, bi, (15 - lane) & 31);
    if (lane < 16 && mayor(cv, ci, lv, li)) { lv = cv; li = ci; }
    if (lane >= 16) { lv = -INFINITY; li = 0x7fffffff; }
#pragma unroll
    for (int j = 8; j; j >>= 1) {
        const float pv = __shfl_xor_sync(LLENO, lv, j);
        const int pi = __shfl_xor_sync(LLENO, li, j);
        if (mayor(pv, pi, lv, li) == ((lane & j) == 0)) { lv = pv; li = pi; }
    }
}

// union en arbol de las NW listas (una por warp, en lv/li): al final el warp 0 tiene el top-16 del bloque.
// log2(NW) niveles de unir16 (5 etapas) en paralelo, en vez de NW uniones seriales en un warp.
__device__ __forceinline__ void arbol(float& lv, int& li, float (*sv)[16], int (*si)[16]) {
    const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int s = 1; s < NW; s <<= 1) {
        if ((w & (2 * s - 1)) == (unsigned)s && lane < 16) { sv[w][lane] = lv; si[w][lane] = li; }
        __syncthreads();
        if ((w & (2 * s - 1)) == 0) {
            const float v = lane < 16 ? sv[w + s][lane] : -INFINITY;
            const int i = lane < 16 ? si[w + s][lane] : 0x7fffffff;
            unir16(lv, li, v, i);
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(NW * 32)
sk29_topk(const __half* __restrict__ lg, int S,              // logits del fijo [M, S] (fp16, de Marlin)
          const int64_t* __restrict__ ids_fijo,              // [S] ids globales
          const float* __restrict__ ld, const int32_t* __restrict__ din_ids, int D,   // anillo [M, D] (D = 0: no)
          float* __restrict__ vo, int64_t* __restrict__ io)  // [M, 16]
{
    extern __shared__ float cmax[];                          // S/32 maximos
    __shared__ float lv_s[NW][16];
    __shared__ int li_s[NW][16];
    __shared__ int trozos[16];
    const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int m = blockIdx.x, C = (S + 31) >> 5;
    const __half* fila = lg + (size_t)m * S;

    // 1. maximo de cada trozo de 32
    for (int c = threadIdx.x; c < C; c += NW * 32) {
        float mx = -INFINITY;
        if ((c + 1) * 32 <= S) {
            const uint4* p = reinterpret_cast<const uint4*>(fila + c * 32);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 u = p[q];
                const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float2 f = __half22float2(h2[r]);
                    mx = fmaxf(mx, fmaxf(f.x, f.y));
                }
            }
        } else {
            for (int e = c * 32; e < S; ++e) mx = fmaxf(mx, __half2float(fila[e]));
        }
        cmax[c] = mx;
    }
    __syncthreads();

    // 2. top-16 de los maximos -> trozos
    float lv = -INFINITY; int li = 0x7fffffff;
    for (int c0 = (int)w * 32; c0 < C; c0 += NW * 32) {
        const int c = c0 + (int)lane;
        float v = c < C ? cmax[c] : -INFINITY; int i = c;
        ordenar32(v, i);
        unir16(lv, li, v, i);
    }
    arbol(lv, li, lv_s, li_s);
    if (w == 0 && lane < 16) trozos[lane] = li;
    __syncthreads();

    // 3. top-16 de los 16*32 candidatos + el anillo (indices < 0: -(ranura + 1))
    const int ncand = 16 * 32 + D;
    lv = -INFINITY; li = 0x7fffffff;
    for (int q0 = (int)w * 32; q0 < ncand; q0 += NW * 32) {
        const int q = q0 + (int)lane;
        float v = -INFINITY; int i = 0x7fffffff;
        if (q < 16 * 32) {
            const int c = trozos[q >> 5];
            const int e = c * 32 + (q & 31);
            if (c >= 0 && c < C && e < S) { v = __half2float(fila[e]); i = e; }
        } else if (q < ncand) {
            const int r = q - 16 * 32;
            v = ld[(size_t)m * D + r]; i = -(r + 1);
        }
        ordenar32(v, i);
        unir16(lv, li, v, i);
    }
    arbol(lv, li, lv_s, li_s);
    if (w) return;
    if (lane < 16) {
        vo[(size_t)m * 16 + lane] = lv;
        io[(size_t)m * 16 + lane] = li >= 0 ? ids_fijo[li] : (int64_t)din_ids[-li - 1];
    }
}
