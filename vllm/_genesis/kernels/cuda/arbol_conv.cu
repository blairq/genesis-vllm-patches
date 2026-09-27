// SPDX-License-Identifier: Apache-2.0
//
// Conv causal del GDN por CAMINO del arbol (salidas + estado desplazado), en PTX. Reemplaza a
// _k_salidas (Triton: lazo de tokens de largo variable, BN = 1024 -> 5 bloques, ~14 us por capa
// con el estado) y a _k_salidas_par (Triton desenrollado, BN = 64: ~5 us, pero 1248
// instrucciones SASS, casi todas indices en 64 bits con strides de runtime y cargas de 2 bytes).
//
// Cada hilo lleva FPT features contiguas de un pedido, TODOS los tokens: asi el hilo que lee la
// historia del estado conv (columnas off..off+2) es el mismo que despues la reescribe, y no hay
// carrera entre programas. x y out van en cargas/guardas de FPT*2 bytes; los 4 pesos de una
// feature son 8 bytes contiguos. Los TOK tokens del lote uniforme del arbol se desenrollan (-DTOK) y
// un pedido mas largo (lote irregular) sigue en un lazo de cola.
//
// Numerica identica a Triton, en PTX explicito (el archivo se compila con --use_fast_math): cada
// producto en fp16 (mul.f16), cvt a f32, acc = p3 + 0 y suma de la columna mas vieja a la mas
// nueva; SiLU = acc / (1 + ex2(-acc*log2e)) con div.full; cvt.rn.f16.f32 al guardar.
//
// Requisitos (los chequea el lanzador): x y out fp16 con stride(1) == 1 y stride(0) multiplo de FPT.
// El estado conv y los pesos van con strides cualquiera (el servidor usa el estado con la dim contigua).

#include <cuda_fp16.h>

#ifndef TOK
#define TOK 9                 // tokens por pedido del lote uniforme del arbol (K+1)
#endif
#ifndef FPT
#define FPT 4
#endif
#ifndef NHILOS
#define NHILOS 64
#endif
#ifndef SILU
#define SILU 1
#endif
#ifndef ESCRIBIR
#define ESCRIBIR 1
#endif

#if FPT == 4
typedef uint2 vec_t;
#elif FPT == 2
typedef unsigned int vec_t;
#elif FPT == 1
typedef unsigned short vec_t;
#else
#error "FPT 1, 2 o 4"
#endif

__device__ __forceinline__ unsigned short hmul(unsigned short a, unsigned short b) {
    unsigned short r; asm("mul.f16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(b)); return r; }
__device__ __forceinline__ float h2f(unsigned short h) { float r; asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h)); return r; }
__device__ __forceinline__ unsigned short f2h(float f) { unsigned short r; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
__device__ __forceinline__ float add_rn(float a, float b) { float r; asm("add.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float sub_rn(float a, float b) { float r; asm("sub.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float mul_rn(float a, float b) { float r; asm("mul.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float ex2_ap(float a) { float r; asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }

__device__ __forceinline__ void desempacar(vec_t v, unsigned short (&h)[FPT]) {
#if FPT == 4
    h[0] = v.x & 0xffffu; h[1] = v.x >> 16; h[2] = v.y & 0xffffu; h[3] = v.y >> 16;
#elif FPT == 2
    h[0] = v & 0xffffu; h[1] = v >> 16;
#else
    h[0] = v;
#endif
}
__device__ __forceinline__ vec_t empacar(const unsigned short (&h)[FPT]) {
#if FPT == 4
    return make_uint2((unsigned)h[0] | ((unsigned)h[1] << 16), (unsigned)h[2] | ((unsigned)h[3] << 16));
#elif FPT == 2
    return (unsigned)h[0] | ((unsigned)h[1] << 16);
#else
    return h[0];
#endif
}

struct Hist { unsigned short a[FPT], b[FPT], c[FPT]; };

// Fila del token (bos + i) o columna de historia (i = -1, -2, -3 -> hC, hB, hA). SIN RAMAS: la carga
// va siempre (indice acotado a >= 0, igual que la mascara de Triton) y despues se elige. La primera
// version tenia `if (i >= 0)` y cada token esperaba a su rama: 76 BRA en el SASS y mas lento que
// Triton (8,1 contra 5,4 us).
__device__ __forceinline__ void fila(const __half* __restrict__ x, int sxt, int bos, int i, int f,
                                     const Hist& h, unsigned short (&o)[FPT]) {
    unsigned short ld[FPT];
    desempacar(__ldg(reinterpret_cast<const vec_t*>(x + (size_t)(bos + (i < 0 ? 0 : i)) * sxt + f)), ld);
#pragma unroll
    for (int j = 0; j < FPT; j++) o[j] = i >= 0 ? ld[j] : (i == -1 ? h.c[j] : (i == -2 ? h.b[j] : h.a[j]));
}

// Salida del token t con sus tres ancestros ya elegidos (xt, t1..t3 en registros).
__device__ __forceinline__ void cuenta(__half* __restrict__ out, int sot, int bos, int t, int f,
                                       const unsigned short (&xt)[FPT], const unsigned short (&t1)[FPT],
                                       const unsigned short (&t2)[FPT], const unsigned short (&t3)[FPT],
                                       const unsigned short (&w)[4][FPT]) {
    unsigned short r[FPT];
#pragma unroll
    for (int j = 0; j < FPT; j++) {
        float acc = add_rn(h2f(hmul(t3[j], w[0][j])), 0.0f);
        acc = add_rn(acc, h2f(hmul(t2[j], w[1][j])));
        acc = add_rn(acc, h2f(hmul(t1[j], w[2][j])));
        acc = add_rn(acc, h2f(hmul(xt[j], w[3][j])));
#if SILU
        acc = div_full(acc, add_rn(ex2_ap(mul_rn(sub_rn(0.0f, acc), 1.4426950216293334961f)), 1.0f));
#endif
        r[j] = f2h(acc);
    }
    *reinterpret_cast<vec_t*>(out + (size_t)(bos + t) * sot + f) = empacar(r);
}

// Ancestro i de un token de los primeros TOK: i < t < TOK, asi que su fila YA esta en registros
// (filas[i]); i < 0 es historia. Cadena de SEL, sin cargas ni ramas.
__device__ __forceinline__ void elegir(int i, const unsigned short (&filas)[TOK][FPT], const Hist& h,
                                       unsigned short (&o)[FPT]) {
#pragma unroll
    for (int j = 0; j < FPT; j++) o[j] = i == -1 ? h.c[j] : (i == -2 ? h.b[j] : h.a[j]);
#pragma unroll
    for (int q = 0; q < TOK; q++)
#pragma unroll
        for (int j = 0; j < FPT; j++) o[j] = i == q ? filas[q][j] : o[j];
}

extern "C" __global__ void __launch_bounds__(NHILOS)
arbol_conv(const __half* __restrict__ x, int sxt, __half* __restrict__ out, int sot,
           __half* __restrict__ cs, long long scs_seq, int scs_dim, int scs_tok,
           const __half* __restrict__ wt, int swd, int sww, const int* __restrict__ cu, const int* __restrict__ sidx,
           const int* __restrict__ nacc, const int* __restrict__ anc3, int sa3, int DIM)
{
    const int n = blockIdx.x;
    const int f = (blockIdx.y * NHILOS + threadIdx.x) * FPT;
    const int bos = __ldg(cu + n), eos = __ldg(cu + n + 1), s = __ldg(sidx + n), na = __ldg(nacc + n);
    const int TT = eos - bos;
    if (TT == 0 || s <= 0 || f >= DIM) return;
    const int off = na - 1;
    __half* __restrict__ col = cs + (size_t)s * scs_seq;           // columna de estado de la feature f: col + f*scs_dim
    Hist h;
    unsigned short w[4][FPT];
#pragma unroll
    for (int j = 0; j < FPT; j++) {
        const unsigned short* hp = reinterpret_cast<const unsigned short*>(col + (size_t)(f + j) * scs_dim + (size_t)off * scs_tok);
        h.a[j] = hp[0]; h.b[j] = hp[scs_tok]; h.c[j] = hp[2 * scs_tok];
        const unsigned short* wp = reinterpret_cast<const unsigned short*>(wt) + (size_t)(f + j) * swd;
        w[0][j] = __ldg(wp); w[1][j] = __ldg(wp + sww); w[2][j] = __ldg(wp + 2 * sww); w[3][j] = __ldg(wp + 3 * sww);
    }
    // Dos niveles de latencia, no tres: los ancestros (anc3) y las filas x de los TOK tokens se cargan
    // JUNTOS (solo dependen de bos); los ancestros de un token del arbol son tokens anteriores del
    // mismo pedido, asi que se eligen de registros en vez de cargarse despues de leer anc3.
    int anc[TOK][3];
    unsigned short filas[TOK][FPT];
#pragma unroll
    for (int t = 0; t < TOK; t++) {
        const int tt = t < TT ? t : 0;
        const int* p3 = anc3 + (size_t)(bos + tt) * sa3;
        anc[t][0] = __ldg(p3); anc[t][1] = __ldg(p3 + 1); anc[t][2] = __ldg(p3 + 2);
        desempacar(__ldg(reinterpret_cast<const vec_t*>(x + (size_t)(bos + tt) * sxt + f)), filas[t]);
    }
#pragma unroll
    for (int t = 0; t < TOK; t++) {
        if (t < TT) {
            unsigned short t1[FPT], t2[FPT], t3[FPT];
            elegir(anc[t][0], filas, h, t1);
            elegir(anc[t][1], filas, h, t2);
            elegir(anc[t][2], filas, h, t3);
            cuenta(out, sot, bos, t, f, filas[t], t1, t2, t3, w);
        }
    }
    // Cola (pedido mas largo que el arbol, lote irregular): cargas comunes.
    for (int t = TOK; t < TT; t++) {
        const int* p3 = anc3 + (size_t)(bos + t) * sa3;
        const int ia[3] = {__ldg(p3), __ldg(p3 + 1), __ldg(p3 + 2)};
        unsigned short xt[FPT], tr[3][FPT];
        desempacar(__ldg(reinterpret_cast<const vec_t*>(x + (size_t)(bos + t) * sxt + f)), xt);
#pragma unroll
        for (int a = 0; a < 3; a++) {
            unsigned short ld[FPT];
            desempacar(__ldg(reinterpret_cast<const vec_t*>(x + (size_t)(bos + (ia[a] < 0 ? 0 : ia[a])) * sxt + f)), ld);
#pragma unroll
            for (int j = 0; j < FPT; j++)
                tr[a][j] = ia[a] >= 0 ? ld[j] : (ia[a] == -1 ? h.c[j] : (ia[a] == -2 ? h.b[j] : h.a[j]));
        }
        cuenta(out, sot, bos, t, f, xt, tr[0], tr[1], tr[2], w);
    }
#if ESCRIBIR
    // Estado desplazado que deja upstream: [h[off+1], h[off+2], x_0 .. x_{TT-1}]. Lo escribe el mismo
    // hilo que leyo la historia, despues de leerla.
#pragma unroll
    for (int j = 0; j < FPT; j++) {
        unsigned short* d = reinterpret_cast<unsigned short*>(col + (size_t)(f + j) * scs_dim);
        d[0] = h.b[j]; d[scs_tok] = h.c[j];
    }
#pragma unroll
    for (int c = 0; c < TOK; c++) {
        if (c < TT) {
#pragma unroll
            for (int j = 0; j < FPT; j++)
                reinterpret_cast<unsigned short*>(col + (size_t)(f + j) * scs_dim)[(size_t)(c + 2) * scs_tok] = filas[c][j];
        }
    }
    for (int c = TOK; c < TT; c++) {
        unsigned short xc[FPT];
        desempacar(__ldg(reinterpret_cast<const vec_t*>(x + (size_t)(bos + c) * sxt + f)), xc);
#pragma unroll
        for (int j = 0; j < FPT; j++)
            reinterpret_cast<unsigned short*>(col + (size_t)(f + j) * scs_dim)[(size_t)(c + 2) * scs_tok] = xc[j];
    }
#endif
}
