// SK-28: top-k de los candidatos del borrador (PN159) sin flashinfer ni pegamento de torch.
//
// Lo que corria (profiler, TP=1, 32k): Marlin 105 us + RadixTopK 20 + StableSort 3 + index 3,7 + casts 5,6
// (+ en la fase 2, logits del anillo 32 y fusion 14). Aca, despues de Marlin:
//   sk28_anillo   logits del anillo [M, D] con mma.sync m16n8k16 (tensor cores), K repartido en 4 warps
//   sk28_parcial  top-KT por trozo de los logits del subconjunto fijo: lista de KT en registros por warp
//                 (lanes 0..KT-1, ordenada), insercion por ballot + shfl_up; casi nada supera el umbral
//   sk28_final    por fila: parciales + anillo -> top-KT con los ids globales ya mapeados (int64) y fp32
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef K
#define K 5120
#endif
#ifndef KT
#define KT 16
#endif
#define LLENO 0xffffffffu

// lista top-KT del warp: lanes 0..KT-1 guardan (valor, indice) en orden descendente
struct Lista { float v; int i; };

__device__ __forceinline__ void insertar(Lista& L, float nv, int ni, float& umbral) {
    const unsigned lane = threadIdx.x & 31;
    unsigned pide = __ballot_sync(LLENO, nv > umbral);
    while (pide) {
        const int src = __ffs(pide) - 1;
        pide &= pide - 1;
        const float x = __shfl_sync(LLENO, nv, src);
        const int xi = __shfl_sync(LLENO, ni, src);
        if (!(x > umbral)) continue;                                 // el umbral subio con otra insercion
        const int pos = __popc(__ballot_sync(LLENO, lane < KT && L.v >= x));
        const float av = __shfl_up_sync(LLENO, L.v, 1);
        const int ai = __shfl_up_sync(LLENO, L.i, 1);
        if (lane < KT && (int)lane > pos) { L.v = av; L.i = ai; }
        if ((int)lane == pos) { L.v = x; L.i = xi; }
        umbral = __shfl_sync(LLENO, L.v, KT - 1);
    }
}

// ── logits del anillo: grid (ceil(D/8), ceil(M/16)), 8 warps que se reparten K; mma f16 -> f32.
//    k PERMUTADO: el producto no depende del orden de k, asi que A y B usan la misma permutacion y cada lane
//    carga un uint4 contiguo (8 halves) de su fila de A (g y g+8) y de B (g): 3 cargas de 16 B = 2 mma.
//    Logico k (2t,2t+1 | 2t+8,2t+9) del paso 0 <-> fisico 8t+(0,1 | 2,3); paso 1 <-> 8t+(4,5 | 6,7).
#define WA 8
#ifndef MT
#define MT 1                                         // tiles de 16 filas de M: una variante por ceil(M/16)
#endif
extern "C" __global__ void __launch_bounds__(WA * 32)
sk28_anillo(const __half* __restrict__ h, int M, const __half* __restrict__ filas,
            const int32_t* __restrict__ din_ids, float* __restrict__ out, int D)
{
    __shared__ float red[WA][32][4 * MT];
    const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int n0 = blockIdx.x * 8, n = n0 + g, nmt = (M + 15) >> 4;
    const uint4* fb = reinterpret_cast<const uint4*>(filas + (size_t)min(n, D - 1) * K);
    const uint32_t zn = n < D ? LLENO : 0u;
    const uint4* ha[MT]; const uint4* hb[MT]; uint32_t za[MT], zb[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int ma = mt * 16 + g, mb = ma + 8;
        ha[mt] = reinterpret_cast<const uint4*>(h + (size_t)min(ma, M - 1) * K);
        hb[mt] = reinterpret_cast<const uint4*>(h + (size_t)min(mb, M - 1) * K);
        za[mt] = ma < M ? LLENO : 0u; zb[mt] = mb < M ? LLENO : 0u;
    }
    float c[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) c[mt][0] = c[mt][1] = c[mt][2] = c[mt][3] = 0.f;
    constexpr int V = K / 8 / 4;
    constexpr int VW = V / WA;
#pragma unroll 2
    for (int v = (int)w * VW; v < ((int)w + 1) * VW; ++v) {
        const int q = v * 4 + (int)t;
        const uint4 F = fb[q];                           // la fila del anillo se lee UNA vez para todas las M
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {                // MT = ceil(M/16) fijo al compilar: sin ramas en el lazo
            const uint4 A = ha[mt][q], B = hb[mt][q];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c[mt][0]), "+f"(c[mt][1]), "+f"(c[mt][2]), "+f"(c[mt][3])
                         : "r"(A.x & za[mt]), "r"(B.x & zb[mt]), "r"(A.y & za[mt]), "r"(B.y & zb[mt]),
                           "r"(F.x & zn), "r"(F.y & zn));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(c[mt][0]), "+f"(c[mt][1]), "+f"(c[mt][2]), "+f"(c[mt][3])
                         : "r"(A.z & za[mt]), "r"(B.z & zb[mt]), "r"(A.w & za[mt]), "r"(B.w & zb[mt]),
                           "r"(F.z & zn), "r"(F.w & zn));
        }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int j = 0; j < 4; ++j) red[w][lane][mt * 4 + j] = c[mt][j];
    __syncthreads();
    if (w) return;
    for (int mt = 0; mt < nmt; ++mt)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float s = 0.f;
#pragma unroll
            for (int x = 0; x < WA; ++x) s += red[x][lane][mt * 4 + j];
            const int m = mt * 16 + g + (j >= 2 ? 8 : 0), nn = n0 + t * 2 + (j & 1);
            if (m < M && nn < D) out[(size_t)m * D + nn] = din_ids[nn] >= 0 ? s : -INFINITY;
        }
}

// ── top-KT parcial del fijo: grid (M, P), 8 warps; el bloque (m, p) mira [p*T, (p+1)*T) de la fila m
extern "C" __global__ void __launch_bounds__(256)
sk28_parcial(const __half* __restrict__ lg, int S, int T, float* __restrict__ pv, int* __restrict__ pi)
{
    __shared__ float sv[8][KT];
    __shared__ int si[8][KT];
    const unsigned lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int m = blockIdx.x, p = blockIdx.y, P = gridDim.y;
    const int a = p * T, b = min(a + T, S);
    const __half* fila = lg + (size_t)m * S;
    Lista L{-INFINITY, 0};
    float umbral = -INFINITY;
    for (int e0 = a + (int)w * 32; e0 < b; e0 += 256) {
        const int e = e0 + (int)lane;
        insertar(L, e < b ? __half2float(fila[e]) : -INFINITY, e, umbral);
    }
    if (lane < KT) { sv[w][lane] = L.v; si[w][lane] = L.i; }
    __syncthreads();
    if (w) return;
    Lista F{-INFINITY, 0};
    umbral = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8 * KT; j += 32) {
        const int r = (j + (int)lane) / KT, c = (j + (int)lane) % KT;
        insertar(F, sv[r][c], si[r][c], umbral);
    }
    if (lane < KT) {
        pv[((size_t)m * P + p) * KT + lane] = F.v;
        pi[((size_t)m * P + p) * KT + lane] = F.i;
    }
}

// ── final por fila (un warp): P*KT parciales del fijo + D del anillo -> top-KT, ids globales int64, fp32
extern "C" __global__ void __launch_bounds__(32)
sk28_final(const float* __restrict__ pv, const int* __restrict__ pi, int P,
           const int64_t* __restrict__ ids_fijo,
           const float* __restrict__ ld, const int32_t* __restrict__ din_ids, int D,
           float* __restrict__ vo, int64_t* __restrict__ io)
{
    const unsigned lane = threadIdx.x;
    const int m = blockIdx.x;
    Lista L{-INFINITY, 0};
    float umbral = -INFINITY;
    for (int e0 = 0; e0 < P * KT; e0 += 32) {                        // indices >= 0: columnas del fijo
        const int e = e0 + (int)lane;
        const bool ok = e < P * KT;
        insertar(L, ok ? pv[(size_t)m * P * KT + e] : -INFINITY, ok ? pi[(size_t)m * P * KT + e] : 0, umbral);
    }
    for (int e0 = 0; e0 < D; e0 += 32) {                             // indices < 0: -(ranura + 1) del anillo
        const int e = e0 + (int)lane;
        insertar(L, e < D ? ld[(size_t)m * D + e] : -INFINITY, -(e + 1), umbral);
    }
    if (lane < KT) {
        vo[(size_t)m * KT + lane] = L.v;
        io[(size_t)m * KT + lane] = L.i >= 0 ? ids_fijo[L.i] : (int64_t)din_ids[-L.i - 1];
    }
}
