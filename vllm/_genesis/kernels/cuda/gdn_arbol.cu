// SPDX-License-Identifier: Apache-2.0
//
// Recurrencia del GDN para un paso en ARBOL (PN122), en PTX. Reemplaza a _k_spec_arbol (Triton).
//
// Lo que hacia _k_spec_arbol (TTGIR, 27-09): el tile de estado [BV=32, K=128] repartido POR COLUMNAS
// (sizePerThread [1,1], threadsPerWarp [1,32], warpsPerCTA [1,4]): cada hilo tenia UNA columna y las 32
// filas, asi que cada h.k y h.q era una reduccion ENTRE LOS 4 WARPS (shared + barrera), dos por token,
// ~17 pasos en serie por programa. 19,7 us por capa con un pedido, 45 con cuatro.
//
// Aca, por programa (tile de 32 filas de v de una cabeza v de un pedido), 4 warps:
//   * cada warp es dueno de 8 FILAS enteras; el carril l tiene las columnas 4l..4l+3. Las reducciones son
//     dentro del warp: sin shared ni barreras en el lazo;
//   * UNA reduccion por token: o = h'.q + d (k.q), con h' = e^g h. h'.k y h'.q se reducen juntas y k.q
//     sale del prologo. Reduce-scatter (8+4+2+1+1 shuffles para 16 valores) en vez de butterfly completo;
//   * prologo: lo que no depende del estado (normas l2 de q y k, g, beta, mascaras del arbol, filas de la
//     cinta a reproducir) se calcula antes, repartido entre warps, y va a shared con UNA barrera;
//   * q, k y v se leen con su stride de token: no hacen falta las copias .contiguous().
//
// Numerica: fp32 como Triton, pero en otro ORDEN de sumas (y el h.q algebraico), asi que no es identico
// bit a bit a Triton. Se valida contra una referencia fp64 (error <= el de Triton) y por la aceptacion
// del borrador en el servidor.
//
// Semantica (igual que _k_spec_arbol): reproducir el camino aceptado del paso anterior desde la cinta;
// despues los T tokens del paso en preorden, cada uno sobre el estado de su padre (en un salto de rama
// se vuelve al estado tras el token 0 y se rehacen los ancestros); el estado tras el token 0 es la unica
// escritura del estado completo.

#include <cuda_fp16.h>

#ifndef H
#define H 8            // cabezas k
#endif
#ifndef HV
#define HV 24          // cabezas v
#endif
#ifndef TM
#define TM 8           // filas de cinta por slot
#endif
#ifndef TMAX
#define TMAX 9         // tokens por pedido en el arbol (K+1)
#endif
#ifndef HDT
#define HDT 0          // estado: 0 = fp16, 1 = fp32
#endif
#ifndef RPW
#define RPW 8          // filas de v por warp; el bloque (4 warps) cubre BV = 4*RPW filas
#endif
#define KD 128
#define VD 128
#define BV (4 * RPW)
#define ROW (H * KD + HV * VD + 2 * HV)
#define NPASO (TM + TMAX)
#ifndef DIAG
#define DIAG 0                // 1: marcas de clock64 del bloque 0 en diag[] (diagnostico de tiempos)
#endif   // filas en shared: cinta a reproducir + tokens

#if HDT == 0
typedef __half hdt_t;
#else
typedef float hdt_t;
#endif

__device__ __forceinline__ float h2f(unsigned short h) { float r; asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h)); return r; }
__device__ __forceinline__ float ex2_ap(float a) { float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float rsqrt_ap(float a) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float expf_tr(float x) { return ex2_ap(x * 1.4426950216293334961f); }   // tl.exp
__device__ __forceinline__ float shx(float v, int m) { return __shfl_xor_sync(0xffffffffu, v, m); }
__device__ __forceinline__ float shs(float v, int src) { return __shfl_sync(0xffffffffu, v, src); }

__device__ __forceinline__ void carga4(const __half* p, float (&x)[4]) {
    const uint2 r = __ldg(reinterpret_cast<const uint2*>(p));
    x[0] = h2f(r.x & 0xffffu); x[1] = h2f(r.x >> 16); x[2] = h2f(r.y & 0xffffu); x[3] = h2f(r.y >> 16);
}

// Suma de un valor por carril sobre los 32 carriles del warp (todos quedan con el total).
__device__ __forceinline__ float suma_warp(float v) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) v += shx(v, m);
    return v;
}

// softplus como Triton: log(1+exp(x)) si x <= umbral, si no x (beta_sp = 1, umbral = 20).
__device__ __forceinline__ float softplus(float x) { return x <= 20.0f ? __logf(1.0f + expf_tr(x)) : x; }
__device__ __forceinline__ float sigmoide(float x) { return 1.0f / (1.0f + expf_tr(-x)); }

// Reduce-scatter de NV valores (potencia de 2 <= 32) sobre el warp: en cada nivel (xor 16, 8, ...) cada
// carril se queda con la mitad de los valores (segun su bit) y manda la otra mitad; despues, los xor que
// quedan suman el unico valor. Al final el carril l tiene el TOTAL del indice (l >> (5 - log2 NV)) & (NV-1)
// (bit 4 del carril = bit mas alto del indice), repetido en los carriles que difieren en los bits bajos.
// Shuffles: NV/2 + NV/4 + ... + 1 + (5 - log2 NV), contra 5*NV de un butterfly completo por valor.
template <int NV>
__device__ __forceinline__ float reducir(float (&v)[NV], unsigned l) {
    float buf[NV];
#pragma unroll
    for (int i = 0; i < NV; i++) buf[i] = v[i];
    int m = 16;
#pragma unroll
    for (int tam = NV; tam > 1; tam >>= 1, m >>= 1) {
        const bool bit = (l & (unsigned)m) != 0u;
        const int mitad = tam >> 1;
#pragma unroll
        for (int i = 0; i < mitad; i++) {
            const float mando = bit ? buf[i] : buf[mitad + i];
            const float quedo = bit ? buf[mitad + i] : buf[i];
            buf[i] = quedo + shx(mando, m);
        }
    }
#pragma unroll
    for (; m >= 1; m >>= 1) buf[0] += shx(buf[0], m);
    return buf[0];
}

template <int NV> struct Log2 { static constexpr int v = 1 + Log2<NV / 2>::v; };
template <> struct Log2<1> { static constexpr int v = 0; };

struct __align__(16) Paso {            // en shared, por paso (cinta a reproducir o token)
    float k[KD];         // k normalizada
    float q[KD];         // q normalizada y escalada (solo tokens)
    float e, beta, kq;   // decaimiento e = exp(g), beta, k.q
    float v[BV];         // v de las 32 filas del tile
};

#ifndef SAB
#define SAB HV           // stride de fila de a y b (PN164: vistas de la salida de in_proj, sin copia)
#endif
extern "C" __global__ void __launch_bounds__(128)
gdn_arbol(const float* __restrict__ A_log, const __half* __restrict__ a, const __half* __restrict__ b,
          const float* __restrict__ dt_bias,
          const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
          long long sq, long long sk, long long sv,           // stride de token (elementos) de q, k, v
          __half* __restrict__ o, hdt_t* __restrict__ h, long long stride_h,
          const int* __restrict__ cu, const int* __restrict__ sidx, const int* __restrict__ nacc,
          const int* __restrict__ slots, const float* __restrict__ cinta, const int* __restrict__ camino,
          const int* __restrict__ anc, float scale, int N
#if DIAG
          , long long* __restrict__ diag
#endif
          )
{
#if DIAG
    long long c0 = clock64();
#define MARCA(i) do { if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) diag[i] = clock64() - c0; } while (0)
#else
#define MARCA(i) do {} while (0)
#endif
    __shared__ Paso P[NPASO];
    __shared__ int msk[TMAX];     // mascaras de ancestros: leidas en direccion UNIFORME (ver abajo)
    const unsigned i_v = blockIdx.x, i_n = blockIdx.y / HV, i_hv = blockIdx.y % HV;
    const unsigned i_h = i_hv / (HV / H);
    const unsigned tid = threadIdx.x, w = tid >> 5, l = tid & 31u;
    const int bos = __ldg(cu + i_n), eos = __ldg(cu + i_n + 1), s = __ldg(sidx + i_n);
    const int r = __ldg(nacc + i_n) - 1, slot = __ldg(slots + i_n);
    const int T = eos - bos;
    if (T <= 0 || s <= 0) return;
    const int R = r < 0 ? 0 : (r > TM ? TM : r);
    const float nexpAl = -expf_tr(__ldg(A_log + i_hv));
    const float db = __ldg(dt_bias + i_hv);
    const unsigned f0 = 4u * l;                       // mis 4 columnas
    const unsigned fil0 = i_v * BV + w * RPW;          // mis RPW filas (en el tile de la cabeza)

    // estado primero: mis 8 filas x 4 columnas (cargas independientes, quedan en vuelo durante el prologo)
    float S[RPW][4];
    hdt_t* hp = h + (size_t)s * stride_h + (size_t)i_hv * VD * KD;
#if HDT == 0
    uint2 rawS[RPW];
#pragma unroll
    for (int i = 0; i < RPW; i++) rawS[i] = *reinterpret_cast<const uint2*>(hp + (size_t)(fil0 + i) * KD + f0);
#else
    float4 rawS[RPW];
#pragma unroll
    for (int i = 0; i < RPW; i++) rawS[i] = *reinterpret_cast<const float4*>(hp + (size_t)(fil0 + i) * KD + f0);
#endif

    // ---------------- prologo: todo lo que no depende del estado -> shared ----------------
    // Todas las cargas globales primero y las cuentas despues: la version que recorria los tokens del warp
    // en serie esperaba una latencia de memoria por vuelta (6.400 ciclos, el 40% del bloque).
    // pasos 0..R-1: filas de la cinta (k ya normalizada, v, g, beta guardados); pasos TM..TM+T-1: tokens.
    const int TT = T < TMAX ? T : TMAX;
    constexpr int NCW = (TM + 3) / 4, NTW = (TMAX + 3) / 4;       // pasos por warp
    float4 ck[NCW]; float cv[NCW], cg[NCW], cb[NCW];
    // Sin condiciones alrededor de cargas ni shuffles: las condiciones dependen de w (= tid >> 5), que el
    // compilador ve como divergente, y cada shfl.sync dentro de una rama asi lleva una ruta de respaldo por
    // CALL (71 en el SASS de la version anterior). Indices acotados y solo las escrituras con guarda.
#pragma unroll
    for (int i = 0; i < NCW; i++) {
        const int p = min(w + 4 * i, TM - 1);
        const int jj = __ldg(camino + slot * TM + p);
        const float* row = cinta + ((size_t)slot * TM + jj) * ROW;
        ck[i] = *reinterpret_cast<const float4*>(row + i_h * KD + f0);
        cv[i] = row[H * KD + i_hv * VD + i_v * BV + (l % BV)];
        cg[i] = row[H * KD + HV * VD + i_hv];
        cb[i] = row[H * KD + HV * VD + HV + i_hv];
    }
    float kx[NTW][4], qx[NTW][4], tv[NTW], ta[NTW], tb[NTW];
#pragma unroll
    for (int i = 0; i < NTW; i++) {
        const int src = bos + min(w + 4 * i, TT - 1);
        carga4(k + (size_t)src * sk + i_h * KD + f0, kx[i]);
        carga4(q + (size_t)src * sq + i_h * KD + f0, qx[i]);
        tv[i] = h2f(__half_as_ushort(v[(size_t)src * sv + i_hv * VD + i_v * BV + (l % BV)]));
        ta[i] = h2f(__half_as_ushort(a[(size_t)src * SAB + i_hv]));
        tb[i] = h2f(__half_as_ushort(b[(size_t)src * SAB + i_hv]));
    }
    if (w == 0 && l < (unsigned)TT) msk[l] = __ldg(anc + bos + l);
#pragma unroll
    for (int i = 0; i < NCW; i++) {
        const int p = w + 4 * i;
        if (p < R) {
            P[p].k[f0] = ck[i].x; P[p].k[f0 + 1] = ck[i].y; P[p].k[f0 + 2] = ck[i].z; P[p].k[f0 + 3] = ck[i].w;
            if (l < BV) P[p].v[l] = cv[i];
            if (l == 0) { P[p].e = expf_tr(cg[i]); P[p].beta = cb[i]; }
        }
    }
#pragma unroll
    for (int i = 0; i < NTW; i++) {
        const int t = w + 4 * i;
        {
            Paso& Pt = P[TM + min(t, TMAX - 1)];
            const bool vale = t < TT;
            const float nk = rsqrt_ap(suma_warp(kx[i][0] * kx[i][0] + kx[i][1] * kx[i][1] + kx[i][2] * kx[i][2] + kx[i][3] * kx[i][3]) + 1e-6f);
            const float nq = rsqrt_ap(suma_warp(qx[i][0] * qx[i][0] + qx[i][1] * qx[i][1] + qx[i][2] * qx[i][2] + qx[i][3] * qx[i][3]) + 1e-6f) * scale;
            float kq = 0.f;
#pragma unroll
            for (int j = 0; j < 4; j++) { kx[i][j] *= nk; qx[i][j] *= nq; kq += kx[i][j] * qx[i][j]; }
            kq = suma_warp(kq);
            if (vale) {
                *reinterpret_cast<float4*>(&Pt.k[f0]) = make_float4(kx[i][0], kx[i][1], kx[i][2], kx[i][3]);
                *reinterpret_cast<float4*>(&Pt.q[f0]) = make_float4(qx[i][0], qx[i][1], qx[i][2], qx[i][3]);
            }
            if (vale && l < BV) Pt.v[l] = tv[i];
            if (vale && l == 0) {
                Pt.e = expf_tr(nexpAl * softplus(ta[i] + db));
                Pt.beta = sigmoide(tb[i]);
                Pt.kq = kq;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < RPW; i++) {
#if HDT == 0
        S[i][0] = h2f(rawS[i].x & 0xffffu); S[i][1] = h2f(rawS[i].x >> 16); S[i][2] = h2f(rawS[i].y & 0xffffu); S[i][3] = h2f(rawS[i].y >> 16);
#else
        S[i][0] = rawS[i].x; S[i][1] = rawS[i].y; S[i][2] = rawS[i].z; S[i][3] = rawS[i].w;
#endif
    }
    MARCA(1);
    __syncthreads();
    MARCA(2);

    // un paso de la regla delta; con salida (tokens) escribe o
    // Un paso de la regla delta: h' = e h;  d = (v - h'.k) beta;  h = h' + d k^T;  o = h.q = h'.q + d (k.q).
    // El decaimiento sale del camino critico: se reduce sobre el estado VIEJO y se escala el resultado
    // (h'.k = e (h.k)), y la actualizacion es S = fma(S, e, d k). Con salida, h.k y h.q van en UNA reduccion.
    auto paso = [&](const Paso& Pp, bool salida, int src) {
        const float4 k4 = *reinterpret_cast<const float4*>(&Pp.k[f0]);
        const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
        const float e = Pp.e, beta = Pp.beta;
        float d_mio;
        if (salida) {
            const float4 q4 = *reinterpret_cast<const float4*>(&Pp.q[f0]);
            const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
            float pv[2 * RPW];
#pragma unroll
            for (int i = 0; i < RPW; i++) {
                pv[i] = S[i][0] * kk[0] + S[i][1] * kk[1] + S[i][2] * kk[2] + S[i][3] * kk[3];
                pv[RPW + i] = S[i][0] * qq[0] + S[i][1] * qq[1] + S[i][2] * qq[2] + S[i][3] * qq[3];
            }
            constexpr int SH = 5 - Log2<2 * RPW>::v;        // indice = l >> SH; bit 4 del carril = k o q
            const unsigned fila = (l >> SH) & (RPW - 1u);
            const float vf = Pp.v[w * RPW + fila];
            const float tot = reducir<2 * RPW>(pv, l);
            const float hq = e * shx(tot, 16);               // el carril l^16 tiene h.q de la misma fila
            d_mio = (vf - e * tot) * beta;                   // valido en carriles con bit 4 = 0
            if ((l & (16u | ((1u << SH) - 1u))) == 0u)
                o[(size_t)src * HV * VD + (size_t)i_hv * VD + fil0 + fila] = __float2half_rn(hq + d_mio * Pp.kq);
#pragma unroll
            for (int i = 0; i < RPW; i++) {
                const float di = shs(d_mio, i << SH);        // la fila i esta en el carril i << SH
#pragma unroll
                for (int j = 0; j < 4; j++) S[i][j] = fmaf(S[i][j], e, di * kk[j]);
            }
        } else {
            float pv[RPW];
#pragma unroll
            for (int i = 0; i < RPW; i++)
                pv[i] = S[i][0] * kk[0] + S[i][1] * kk[1] + S[i][2] * kk[2] + S[i][3] * kk[3];
            constexpr int SH = 5 - Log2<RPW>::v;
            const unsigned fila = (l >> SH) & (RPW - 1u);
            const float vf = Pp.v[w * RPW + fila];
            const float tot = reducir<RPW>(pv, l);
            d_mio = (vf - e * tot) * beta;
#pragma unroll
            for (int i = 0; i < RPW; i++) {
                const float di = shs(d_mio, i << SH);
#pragma unroll
                for (int j = 0; j < 4; j++) S[i][j] = fmaf(S[i][j], e, di * kk[j]);
            }
        }
    };

    // 1) reproducir el camino aceptado del paso anterior
    for (int p = 0; p < R; p++) paso(P[p], false, 0);

    MARCA(3);
    // 2) tokens del paso, cada uno sobre el estado de su padre
    float Raiz[RPW][4];
    int m_prev = 0;
    for (int t = 0; t < TT; t++) {
        const int m_t = msk[t];                         // direccion uniforme: el compilador sabe que es igual en el warp
        if (t >= 2 && m_t != (m_prev | (1 << (t - 1)))) {
#pragma unroll
            for (int i = 0; i < RPW; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) S[i][j] = Raiz[i][j];
            for (int j = 1; j < t; j++)
                if ((m_t >> (j - 1)) & 1) paso(P[TM + j], false, 0);
        }
        m_prev = m_t;
        paso(P[TM + t], true, bos + t);
        if (t == 0) {
#pragma unroll
            for (int i = 0; i < RPW; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) Raiz[i][j] = S[i][j];
#if HDT == 0
                const __half2 lo = __floats2half2_rn(S[i][0], S[i][1]), hi = __floats2half2_rn(S[i][2], S[i][3]);
                uint2 raw; raw.x = *reinterpret_cast<const unsigned*>(&lo); raw.y = *reinterpret_cast<const unsigned*>(&hi);
                *reinterpret_cast<uint2*>(hp + (size_t)(fil0 + i) * KD + f0) = raw;
#else
                *reinterpret_cast<float4*>(hp + (size_t)(fil0 + i) * KD + f0) = make_float4(S[i][0], S[i][1], S[i][2], S[i][3]);
#endif
            }
        }
    }
    MARCA(4);
}
