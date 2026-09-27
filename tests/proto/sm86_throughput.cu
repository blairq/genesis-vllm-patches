// sm86_throughput.cu — throughput por instruccion en SM86 (resultados por ciclo y SM, con clock64). nvcc -O3 -arch=sm_86 -o tp sm86_throughput.cu && ./tp
// Un bloque de 1024 hilos por SM (SM86 admite 1536: dos bloques no entran juntos). Operandos en registros: con constantes ptxas fusiona las cadenas.
#include <cstdio>
#include <cuda_fp16.h>
#define ILP 8
#define ITER 2048
template<int OP> __global__ void k(float* out, long long* cyc, int seed) {
  float f[ILP]; unsigned u[ILP]; __half2 h[ILP];
  #pragma unroll
  for (int i = 0; i < ILP; i++) { f[i] = threadIdx.x * 1e-3f + i + seed; u[i] = threadIdx.x + i * 7 + seed; h[i] = __floats2half2_rn(f[i], f[i]); }
  __syncthreads();
  long long t0 = clock64();
  for (int it = 0; it < ITER; it++) {
    #pragma unroll
    for (int i = 0; i < ILP; i++) {
      if (OP == 0) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[i]) : "f"(f[(i+1)%ILP]), "f"(f[(i+3)%ILP]));
      if (OP == 1) asm volatile("fma.rn.f16x2 %0, %0, %0, %0;" : "+r"(*(unsigned*)&h[i]));
      if (OP == 2) asm volatile("add.u32 %0, %0, %1;" : "+r"(u[i]) : "r"(u[(i+1)%ILP]));
      if (OP == 3) asm volatile("mad.lo.u32 %0, %0, 0x9E3779B1, 7;" : "+r"(u[i]));
      if (OP == 4) asm volatile("lop3.b32 %0, %0, 0x5555, 0x3333, 0x96;" : "+r"(u[i]));
      if (OP == 5) asm volatile("shf.l.wrap.b32 %0, %0, %0, %1;" : "+r"(u[i]) : "r"(u[(i+1)%ILP]));
      if (OP == 6) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f[i]));
      if (OP == 7) { asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f[i]) : "r"(u[i])); asm volatile("mov.b32 %0, %1;" : "=r"(u[i]) : "f"(f[i])); }
      if (OP == 8) { unsigned p; asm volatile("cvt.rn.f16x2.f32 %0, %1, %1;" : "=r"(p) : "f"(f[i])); asm volatile("mov.b32 %0, %1;" : "=f"(f[i]) : "r"(p)); }
      if (OP == 9) asm volatile("max.f32 %0, %0, %1;" : "+f"(f[i]) : "f"(f[(i+1)%ILP]));
      if (OP == 10) { // int32 -> float por numero magico: IADD + FADD (valido |i| < 2^22)
        float g; asm volatile("add.u32 %0, %0, 0x4B400000;" : "+r"(u[i]));
        asm volatile("mov.b32 %0, %1;" : "=f"(g) : "r"(u[i]));
        asm volatile("sub.f32 %0, %1, 0f4B400000;" : "=f"(f[i]) : "f"(g));
        asm volatile("mov.b32 %0, %1;" : "=r"(u[i]) : "f"(f[i])); }
      if (OP == 11) { // mezcla 1 FFMA + 1 IADD: ¿comparten camino?
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[i]) : "f"(f[(i+1)%ILP]), "f"(f[(i+3)%ILP]));
        asm volatile("add.u32 %0, %0, %1;" : "+r"(u[i]) : "r"(u[(i+1)%ILP])); }
      if (OP == 12) { // mezcla 1 FFMA + 1 EX2: ¿MUFU corre en paralelo?
        asm volatile("fma.rn.f32 %0, %0, 0f3F7FFFFF, 0f00000000;" : "+f"(f[i]));
        float e; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(e) : "f"(f[i])); f[i] += 0.f * e; u[i] ^= __float_as_uint(e); }
      if (OP == 13) asm volatile("prmt.b32 %0, %0, 0x64646464, 0x4150;" : "+r"(u[i]));
    }
  }
  long long t1 = clock64();
  float s = 0; for (int i = 0; i < ILP; i++) s += f[i] + u[i] + __low2float(h[i]);
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
template<int OP> void run(const char* nom, int nops) {
  int blocks = 82, th = 1024; float* o; long long* c; cudaMalloc(&o, blocks * th * 4); cudaMalloc(&c, blocks * 8);
  k<OP><<<blocks, th>>>(o, c, 1); cudaDeviceSynchronize();
  k<OP><<<blocks, th>>>(o, c, 1); cudaDeviceSynchronize();
  long long h[82]; cudaMemcpy(h, c, blocks * 8, cudaMemcpyDeviceToHost);
  double cy = 0; for (int i = 0; i < blocks; i++) cy += h[i]; cy /= blocks;
  // 2 bloques por SM corren a la vez: ops por SM = 2 * th * ITER * ILP * nops
  double ops = 1.0 * th * ITER * ILP * nops;
  printf("%-44s %7.1f por ciclo y SM\n", nom, ops / cy);
  cudaFree(o); cudaFree(c);
}
int main() {
  run<0>("FFMA fp32", 1); run<1>("HFMA2 (2 resultados fp16)", 2); run<9>("FMNMX fp32 (max)", 1);
  run<2>("IADD u32", 1); run<3>("IMAD u32", 1); run<4>("LOP3", 1); run<5>("SHF (shift)", 1); run<13>("PRMT", 1);
  run<6>("MUFU.EX2", 1); run<7>("I2F (cvt f32<-s32)", 1); run<8>("F2F pack (cvt f16x2<-f32)", 1);
  run<10>("int->float magico (IADD+FADD) [pares]", 1);
  run<11>("FFMA + IADD mezclados [pares]", 1); run<12>("FFMA + EX2 mezclados [pares]", 1);
}
