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
