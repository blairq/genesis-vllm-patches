// SPDX-License-Identifier: Apache-2.0
//
// SK-24 — all-reduce de TP=2 para el DECODE, por P2P directo (PN152). Reemplaza a NCCL RING_LL, que tarda
// 26 us por 92 KB (1 pedido, 9 filas) y 61 us por 369 KB (4 pedidos), 128 veces por paso.
//
// Ideas de SiFAR (arXiv 2607.08973, medido en H200 con NVSwitch) traducidas a 2 x 3090 por PCIe 4.0 x8 con
// P2P: la reduccion en el switch (multimem.ld_reduce) no existe aca, pero si:
//   * ESCRIBIR y no leer: cada rango escribe su parcial en el buffer de recepcion de la OTRA placa (las
//     escrituras P2P son "posted"; una lectura remota paga la ida y vuelta del PCIe);
//   * DOBLE BUFFER por paridad de epoca: la otra placa escribe el all-reduce siguiente en el otro buffer
//     mientras este suma el actual, asi que no hace falta barrera al final;
//   * BANDERA con el numero de epoca, escrita despues de los datos (con __threadfence_system entre medio):
//     el que la ve ya tiene los datos. La epoca vive en la GPU y la avanza el kernel: el mismo lanzamiento
//     grabado en un grafo CUDA sirve para todos los pasos.
//
// Con 2 rangos la suma es UNA suma fp16 (x + recibido, conmutativa): mismo resultado que NCCL.
//
// Memoria: rx_local/rx_peer y flag_local/flag_peer salen de cudaMalloc + IPC (sobre tensores de torch
// compartidos un kernel da acceso ilegal; ver la memoria p2p-anda-bien-con-memoria-cruda). No se usa
// ld.global.nc sobre memoria de la otra placa.
//
// La grilla tiene que caber entera a la vez (los bloques esperan la bandera): BLOQUES chico.

#include <cuda_fp16.h>

#ifndef HILOS
#define HILOS 256
#endif

__device__ __forceinline__ int ld_vol(const int* p) { int v; asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ void st_vol(int* p, int v) { asm volatile("st.volatile.global.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }

__device__ __forceinline__ uint4 sumar8(uint4 a, uint4 b) {
    uint4 r;
    const __half2* pa = reinterpret_cast<const __half2*>(&a);
    const __half2* pb = reinterpret_cast<const __half2*>(&b);
    __half2* pr = reinterpret_cast<__half2*>(&r);
#pragma unroll
    for (int i = 0; i < 4; i++) pr[i] = __hadd2(pa[i], pb[i]);
    return r;
}

// ctl (local): [0] epoca, [1] bloques que terminaron de escribir, [2] bloques que terminaron
extern "C" __global__ void __launch_bounds__(HILOS)
sk24_ar(const uint4* __restrict__ x, uint4* __restrict__ out, int n16,
        const uint4* rx_local, uint4* rx_peer, int maxn16,
        const int* flag_local, int* flag_peer, int* ctl)
{
    __shared__ int e_s;
    if (threadIdx.x == 0) e_s = ld_vol(ctl);
    __syncthreads();
    const int e = e_s, slot = e & 1;
    const int paso = gridDim.x * HILOS;
    const int i0 = blockIdx.x * HILOS + threadIdx.x;

    // 1) mi parcial -> buffer de recepcion de la otra placa (escrituras P2P de 16 bytes)
    uint4* dst = rx_peer + (size_t)slot * maxn16;
    for (int i = i0; i < n16; i += paso) dst[i] = x[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (atomicAdd(ctl + 1, 1) == (int)gridDim.x - 1) {    // el ultimo bloque en escribir avisa
            ctl[1] = 0;
            __threadfence_system();
            st_vol(flag_peer + slot, e + 1);
        }
        // 2) esperar la bandera de la otra placa (escrita en MI memoria)
        while (ld_vol(flag_local + slot) != e + 1) { }
    }
    __syncthreads();
    __threadfence();

    // 3) sumar: lo mio + lo recibido
    const uint4* src = rx_local + (size_t)slot * maxn16;
    for (int i = i0; i < n16; i += paso) out[i] = sumar8(x[i], src[i]);

    // 4) el ultimo bloque en terminar avanza la epoca (la lee el lanzamiento siguiente)
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(ctl + 2, 1) == (int)gridDim.x - 1) {
        ctl[2] = 0;
        st_vol(ctl, e + 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────────────
// SK-24/i8 — la misma idea con los parciales en int8 por grupo de 64 (la mitad de bytes por el PCIe), para
// los mensajes de varios pedidos. Misma cuenta que PN120: escala = amax/127 (clamp 1e-6) guardada en fp16,
// q = redondeo(v / escala) en [-127, 127].
//
// En TP el residuo tiene que quedar IDENTICO en las dos placas: por eso cada una suma las DOS partes
// cuantizadas (tambien la propia), en orden de rango: out = fp16(q0*s0 + q1*s1), con la misma expresion.
//
// Buffers (por ranura): [q int8: n bytes][escalas fp16: n/64 * 2 bytes]. "mio" (local, sin ranura) guarda
// mi propia parte cuantizada para la suma. Un hilo = un grupo de 64 (128 bytes de x).
#define G 64

// v2 (27-09): la v1 hacia UN HILO POR GRUPO (128 bytes por hilo, mal coalescido, 64 divisiones IEEE) y
// escribia las escalas de a 2 bytes sueltos por PCIe: salia MAS LENTA que el fp16 (64,6 contra 44,8 us con
// 36 filas). Ahora: 8 hilos por grupo (16 bytes cada uno, carga coalescida), maximo con 3 shuffles,
// reciproco de la escala en vez de division, y las 4 escalas de un warp en UNA escritura de 8 bytes.
// Las dos placas siguen dando identico: cada una recibe los BYTES ya cuantizados de la otra.
__device__ __forceinline__ float shx8(float v, int m) { return __shfl_xor_sync(0xffffffffu, v, m, 8); }

extern "C" __global__ void __launch_bounds__(HILOS)
sk24_ar_i8(const uint4* __restrict__ x, uint4* __restrict__ out, int ngrupos, int rank,
           const unsigned char* rx_local, unsigned char* rx_peer, int maxbytes,
           unsigned char* mio, const int* flag_local, int* flag_peer, int* ctl)
{
    __shared__ int e_s;
    if (threadIdx.x == 0) e_s = ld_vol(ctl);
    __syncthreads();
    const int e = e_s, slot = e & 1;
    const unsigned lane = threadIdx.x & 31u, sub = lane & 7u;       // sub = hilo dentro del grupo
    const int gw = (blockIdx.x * HILOS + threadIdx.x) >> 3;          // grupo de este hilo (primera vuelta)
    const int pasog = (gridDim.x * HILOS) >> 3;
    const size_t nq = (size_t)ngrupos * G;
    unsigned char* dq = rx_peer + (size_t)slot * maxbytes;
    __half* ds = reinterpret_cast<__half*>(dq + nq);
    __half* ms = reinterpret_cast<__half*>(mio + nq);
    const int ngr4 = (ngrupos + 3) & ~3;                              // multiplo de 4: los warps van enteros

    // 1) cuantizar (8 hilos por grupo) y mandar a la otra placa; copia propia en "mio"
    for (int gi = gw; gi < ngr4; gi += pasog) {
        const bool vale = gi < ngrupos;
        const int gc = vale ? gi : 0;
        const uint4 u = x[(size_t)gc * 8 + sub];
        const __half2* h = reinterpret_cast<const __half2*>(&u);
        float v[8];
#pragma unroll
        for (int j = 0; j < 4; j++) { const float2 f = __half22float2(h[j]); v[2 * j] = f.x; v[2 * j + 1] = f.y; }
        float am = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) am = fmaxf(am, fabsf(v[j]));
        am = fmaxf(am, shx8(am, 4)); am = fmaxf(am, shx8(am, 2)); am = fmaxf(am, shx8(am, 1));
        const __half s16 = __float2half_rn(fmaxf(am, 1e-6f) / 127.0f);
        const float inv = 1.0f / __half2float(s16);
        uint2 q;
        unsigned char* qb = reinterpret_cast<unsigned char*>(&q);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            float r = rintf(v[j] * inv);
            r = fminf(fmaxf(r, -127.f), 127.f);
            qb[j] = (unsigned char)(signed char)(int)r;
        }
        if (vale) {
            reinterpret_cast<uint2*>(dq + (size_t)gi * G)[sub] = q;
            reinterpret_cast<uint2*>(mio + (size_t)gi * G)[sub] = q;
        }
        // las 4 escalas del warp (grupos gi-sub..): el carril 0 las junta y escribe 8 bytes
        const unsigned short sb = __half_as_ushort(s16);
        const unsigned s0 = __shfl_sync(0xffffffffu, sb, 0), s1 = __shfl_sync(0xffffffffu, sb, 8);
        const unsigned s2 = __shfl_sync(0xffffffffu, sb, 16), s3 = __shfl_sync(0xffffffffu, sb, 24);
        if (lane == 0) {
            const int g0 = gi;                                          // gi del carril 0 = primer grupo del warp
            if (g0 + 3 < ngrupos) {
                const uint2 pk = make_uint2(s0 | (s1 << 16), s2 | (s3 << 16));
                *reinterpret_cast<uint2*>(ds + g0) = pk;
                *reinterpret_cast<uint2*>(ms + g0) = pk;
            } else {
                const unsigned sv[4] = {s0, s1, s2, s3};
                for (int k = 0; k < 4 && g0 + k < ngrupos; k++) {
                    ds[g0 + k] = __ushort_as_half((unsigned short)sv[k]);
                    ms[g0 + k] = __ushort_as_half((unsigned short)sv[k]);
                }
            }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        if (atomicAdd(ctl + 1, 1) == (int)gridDim.x - 1) {
            ctl[1] = 0;
            __threadfence_system();
            st_vol(flag_peer + slot, e + 1);
        }
        while (ld_vol(flag_local + slot) != e + 1) { }
    }
    __syncthreads();
    __threadfence();

    // 3) out = q0*s0 + q1*s1 en orden de RANGO (igual en las dos placas); 8 hilos por grupo
    const unsigned char* rq = rx_local + (size_t)slot * maxbytes;
    const __half* rs = reinterpret_cast<const __half*>(rq + nq);
    for (int gi = gw; gi < ngrupos; gi += pasog) {
        const unsigned char* q0 = (rank == 0 ? mio : rq) + (size_t)gi * G;
        const unsigned char* q1 = (rank == 0 ? rq : mio) + (size_t)gi * G;
        const float f0 = __half2float(rank == 0 ? ms[gi] : rs[gi]);
        const float f1 = __half2float(rank == 0 ? rs[gi] : ms[gi]);
        const uint2 a = reinterpret_cast<const uint2*>(q0)[sub];
        const uint2 b = reinterpret_cast<const uint2*>(q1)[sub];
        const signed char* pa = reinterpret_cast<const signed char*>(&a);
        const signed char* pb = reinterpret_cast<const signed char*>(&b);
        uint4 o;
        __half* po = reinterpret_cast<__half*>(&o);
#pragma unroll
        for (int j = 0; j < 8; j++) po[j] = __float2half_rn(__fmaf_rn((float)pb[j], f1, __fmul_rn((float)pa[j], f0)));
        out[(size_t)gi * 8 + sub] = o;
    }
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(ctl + 2, 1) == (int)gridDim.x - 1) {
        ctl[2] = 0;
        st_vol(ctl, e + 1);
    }
}
