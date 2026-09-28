// SK-30: atencion de decode en UNA pasada sobre la KV int8_per_token_head de PN131 (prototipo).
//
// SK-18h (batch2) lee K dos veces (pasada A: maximo exacto; pasada B: pesos y w.v) y tiene 1,7 olas en 82
// SM (una pagina de 880 por bloque). A 62k: ~119 MB leidos contra 64 MB de K+V, 55% del ancho de banda.
// Aca: softmax en linea en fp32 (flash-decoding): cada tile de 64 keys se lee una vez.
//   S = Q.K   mma m16n8k32 s8 -> s32 (Q int8 por fila en registros, K int8 en shared)
//   s = S * qs[fila] * ks[key]                     (en log2: qs ya trae scale*log2(e) y 2^(ek-15))
//   m' = max(m, max_k s); a = 2^(m-m'); p = 2^(s-m')
//   O = a*O + (p*vs[key]) . V   mma m16n8k16 f16 -> f32 (V int8 -> fp16 exacto al armar el fragmento)
// Reparto: cada (secuencia, cabeza KV) se parte en GMAX grupos de keys; el tamano del grupo sale del
// seq_len REAL en la GPU (vale dentro del grafo). Salida por grupo: O/l en fp16, m y l en fp32.
//
// Layout del bloque de KV (el de SK-18h): K [BS][NH][256] | V [NH][256][BS] | escalas int16 [BS][NH][2].
// Filas: r = j * G + g (token nuevo j, cabeza Q g del grupo de la cabeza KV); 64 filas (L*G <= 64).
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <stdint.h>

#define QD 256
#define RB 64            // filas por (secuencia, cabeza KV)
#define TK 64            // keys por tile
#define KST 272          // paso de fila de K en shared (256 + 16: sin conflictos de banco)
#define VST 80           // paso de fila de V^T en shared (64 + 16)
#define PST 72           // paso de fila de P (halves)
#define NEG (-1e30f)
#ifndef NQ
#define NQ 4             // cuartos: warps = 4 tiles de filas x NQ (keys en S, dims en O)
#endif
#define NWK (4 * NQ)
#ifndef PV8
#define PV8 0            // P.V en int8: P'=p*vs cuantizado por fila y tile, V int8 directo (m16n8k32 s8)
#endif
#ifndef PHL
#define PHL 1            // con PV8: P' en 16 bits = hi*256 + lo (dos planos int8, dos mma): el error de P desaparece
#endif
#ifndef ACC16
#define ACC16 1          // P.V con acumulador fp16 por tile (Sage)
#endif
#define KPQ (TK / NQ)    // keys por warp en S
#define DPQ (QD / NQ)    // dims por warp en O

__device__ __forceinline__ unsigned sdir(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

// int8 x2 (16 bits) -> half2 exacto: 0x6400|(x+128) es 1024+x+128 en fp16; restar 1152
__device__ __forceinline__ unsigned i8x2_a_h2(unsigned short u) {
    const unsigned x = (unsigned)u ^ 0x8080u;                                  // x + 128 por byte
    unsigned r;
    asm("prmt.b32 %0, %1, %2, 0x7170;" : "=r"(r) : "r"(x), "r"(0x64646464u));  // [0x64,b1,0x64,b0]
    const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&r), __float2half2_rn(1152.f));
    return *reinterpret_cast<const unsigned*>(&h);
}

extern "C" __global__ void __launch_bounds__(NWK * 32)
sk30_decode(const signed char* __restrict__ Qi,     // [B, NH, RB, 256] int8 (q rotada)
            const float* __restrict__ qs,           // [B, NH, RB] escala de fila (log2, con scale y 2^(ek-15))
            const signed char* __restrict__ pool,   // bloques de KV
            const int* __restrict__ tabla, int tstride,   // [B, tstride] bloque fisico por pagina
            const int* __restrict__ seqlen,         // [B]
            const int* __restrict__ lim,            // [B, NH, RB] ultima key visible (causal); -1 = fila vacia
            const int* __restrict__ abase,          // [B, NH, RB] keys <= abase se ven siempre (arbol)
            const int* __restrict__ amask,          // [B, NH, RB] bit (key-abase-1): ancestro visible
            int NH, int BS, long long BLK, const int* __restrict__ refs,   // [ek, ev]: vs = svf * 2^(ev-15)
            int GMAX,                               // grupos de keys por (secuencia, cabeza)
            __half* __restrict__ Op, float* __restrict__ Mp, float* __restrict__ Lp)  // [GMAX, B, NH, RB, (256)]
{
    extern __shared__ __align__(16) unsigned char sm[];
    unsigned char* sK = sm;                                  // [2][TK][KST]
    unsigned char* sV = sK + 2 * TK * KST;                   // [2][256][VST]
    __half* sP = reinterpret_cast<__half*>(sV + 2 * QD * VST);   // [RB][PST]
    short* sE = reinterpret_cast<short*>(sP + RB * PST);     // [2][TK][2]
    float* sM = reinterpret_cast<float*>(sE + 2 * TK * 2);   // [NQ][RB] maximos / sumas parciales por cuarto

    const int grp = blockIdx.x, bh = blockIdx.y, b = bh / NH, h = bh % NH;
    const int B = gridDim.y / NH;
    const unsigned tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int rt = w & 3, hh = w >> 2;                       // tile de 16 filas, cuarto hh (keys en S, dims en O)
    const unsigned NT = NWK * 32;
    const int n = seqlen[b];
    int paso = (n + GMAX - 1) / GMAX; paso = (paso + TK - 1) / TK * TK;
    const int k_ini = grp * paso, k_fin = min(n, k_ini + paso);
    const size_t so = (((size_t)grp * B + b) * NH + h) * RB;
    const float vsc = exp2f((float)(refs[1] - 15));
    if (k_ini >= k_fin) {                                    // grupo vacio: m = -inf, l = 0
        if (tid < RB) { Mp[so + tid] = NEG; Lp[so + tid] = 0.f; }
        return;
    }

    // Q de este warp en registros: filas rt*16 + gid (+8), 8 pasos de k=32 -> 4 regs cada uno
    unsigned qa[8][4];
    const signed char* qb = Qi + ((size_t)(b * NH + h) * RB + rt * 16) * QD;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        qa[s][0] = *reinterpret_cast<const unsigned*>(qb + (size_t)gid * QD + s * 32 + tig * 4);
        qa[s][1] = *reinterpret_cast<const unsigned*>(qb + (size_t)(gid + 8) * QD + s * 32 + tig * 4);
        qa[s][2] = *reinterpret_cast<const unsigned*>(qb + (size_t)gid * QD + s * 32 + 16 + tig * 4);
        qa[s][3] = *reinterpret_cast<const unsigned*>(qb + (size_t)(gid + 8) * QD + s * 32 + 16 + tig * 4);
    }
    const int f0 = rt * 16 + gid, f1 = f0 + 8;
    const float qs0 = qs[(size_t)(b * NH + h) * RB + f0], qs1 = qs[(size_t)(b * NH + h) * RB + f1];
    const size_t rb0 = (size_t)(b * NH + h) * RB;
    const int lm0 = lim[rb0 + f0], lm1 = lim[rb0 + f1];         // mismas reglas que SK-18h (SK18H_ANCEXP)
    const int lb0 = abase[rb0 + f0], lb1 = abase[rb0 + f1];
    const unsigned am0 = (unsigned)amask[rb0 + f0], am1 = (unsigned)amask[rb0 + f1];

    float o[DPQ / 8][4];                                     // 16 filas x DPQ dims (cuarto hh)
#pragma unroll
    for (int i = 0; i < DPQ / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;
    const size_t KOFF = (size_t)BS * NH * QD, EOFF = 2 * KOFF;

    // carga de un tile: 64 keys = 4 trozos de 16 (880 = 55*16: un trozo nunca cruza pagina)
    auto cargar = [&](int st, int k0) {
        // K: 64 keys x 256 B = 1024 copias de 16 B
#pragma unroll
        for (int c = 0; c < 1024 / (NWK * 32); ++c) {
            const int e = tid + c * NT, key = e >> 4, seg = e & 15, kk = k0 + key;
            const int ok = kk < k_fin;
            const int kc = ok ? kk : k0;
            const long long blk = tabla[(size_t)b * tstride + kc / BS];
            const int off = kc % BS;
            const signed char* src = pool + blk * BLK + ((size_t)off * NH + h) * QD + seg * 16;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                         :: "r"(sdir(sK + (st * TK + key) * KST + seg * 16)), "l"(src), "r"(ok ? 16 : 0));
        }
        // V^T: 256 dims x 64 keys = 256 filas x 4 trozos de 16 keys -> 1024 copias
#pragma unroll
        for (int c = 0; c < 1024 / (NWK * 32); ++c) {
            const int e = tid + c * NT, d = e >> 2, tr = e & 3, kk = k0 + tr * 16;
            const int ok = kk < k_fin;
            const int kc = ok ? kk : k0;
            const long long blk = tabla[(size_t)b * tstride + kc / BS];
            const int off = kc % BS;
            const signed char* src = pool + blk * BLK + KOFF + ((size_t)h * QD + d) * BS + off;
            const int bytes = ok ? min(16, k_fin - kk) : 0;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                         :: "r"(sdir(sV + (st * QD + d) * VST + tr * 16)), "l"(src), "r"(bytes));
        }
        // escalas: 64 keys x (skf, svf) de la cabeza h
        if (tid < TK) {
            const int kk = k0 + tid, ok = kk < k_fin, kc = ok ? kk : k0;
            const long long blk = tabla[(size_t)b * tstride + kc / BS];
            const int off = kc % BS;
            const signed char* src = pool + blk * BLK + EOFF + ((size_t)off * NH + h) * 4;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                         :: "r"(sdir(sE + (st * TK + tid) * 2)), "l"(src), "r"(ok ? 4 : 0));
        }
        __pipeline_commit();
    };

    int st = 0;
    cargar(0, k_ini);
    for (int k0 = k_ini; k0 < k_fin; k0 += TK) {
        if (k0 + TK < k_fin) cargar(st ^ 1, k0 + TK); else __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        const unsigned char* K_ = sK + st * TK * KST;
        const short* E_ = sE + st * TK * 2;
        // ── S = Q.K para 16 filas x KPQ keys (cuarto hh): KPQ/8 tiles de n=8
        int sacc[KPQ / 8][4];
#pragma unroll
        for (int t = 0; t < KPQ / 8; ++t) sacc[t][0] = sacc[t][1] = sacc[t][2] = sacc[t][3] = 0;
#pragma unroll
        for (int t = 0; t < KPQ / 8; ++t) {
            const unsigned char* kr = K_ + (hh * KPQ + t * 8 + gid) * KST + tig * 4;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const unsigned b0 = *reinterpret_cast<const unsigned*>(kr + s * 32);
                const unsigned b1 = *reinterpret_cast<const unsigned*>(kr + s * 32 + 16);
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+r"(sacc[t][0]), "+r"(sacc[t][1]), "+r"(sacc[t][2]), "+r"(sacc[t][3])
                             : "r"(qa[s][0]), "r"(qa[s][1]), "r"(qa[s][2]), "r"(qa[s][3]), "r"(b0), "r"(b1));
            }
        }
        // ── escala, mascara y maximo de la mitad
        float sf[KPQ / 8][4];
        float mx0 = NEG, mx1 = NEG;
#pragma unroll
        for (int t = 0; t < KPQ / 8; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = hh * KPQ + t * 8 + tig * 2 + (e & 1), kk = k0 + col;
                const float ks = (float)E_[col * 2];
                const bool fila1 = e >= 2;
                const int lm = fila1 ? lm1 : lm0, lb = fila1 ? lb1 : lb0;
                const unsigned am = fila1 ? am1 : am0;
                const bool vis = kk < k_fin && kk <= lm &&
                                 (kk <= lb || ((unsigned)(kk - lb - 1) < 31u && ((am >> ((kk - lb - 1) & 31)) & 1u)));
                const float v = vis ? (float)sacc[t][e] * (fila1 ? qs1 : qs0) * ks : NEG;
                sf[t][e] = v;
                if (fila1) mx1 = fmaxf(mx1, v); else mx0 = fmaxf(mx0, v);
            }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        if (tig == 0) { sM[hh * RB + f0] = mx0; sM[hh * RB + f1] = mx1; }
        __syncthreads();
        float mn0 = m0, mn1 = m1;
#pragma unroll
        for (int x = 0; x < NQ; ++x) { mn0 = fmaxf(mn0, sM[x * RB + f0]); mn1 = fmaxf(mn1, sM[x * RB + f1]); }
        const float a0 = exp2f(m0 - mn0), a1 = exp2f(m1 - mn1);
        m0 = mn0; m1 = mn1;
        // ── p = 2^(s - m), l, y P' = p * vs[key] a shared (fp16, o int8 por fila con PV8)
        float ps0 = 0.f, ps1 = 0.f;
#if PV8
        float pv[KPQ / 8][4];
        float px0 = 0.f, px1 = 0.f;
#endif
#pragma unroll
        for (int t = 0; t < KPQ / 8; ++t) {
            const int col = hh * KPQ + t * 8 + tig * 2;
            const float vs0 = (float)E_[col * 2 + 1] * vsc, vs1 = (float)E_[(col + 1) * 2 + 1] * vsc;
            const float p00 = sf[t][0] > 0.5f * NEG ? exp2f(sf[t][0] - m0) : 0.f;
            const float p01 = sf[t][1] > 0.5f * NEG ? exp2f(sf[t][1] - m0) : 0.f;
            const float p10 = sf[t][2] > 0.5f * NEG ? exp2f(sf[t][2] - m1) : 0.f;
            const float p11 = sf[t][3] > 0.5f * NEG ? exp2f(sf[t][3] - m1) : 0.f;
            ps0 += p00 + p01; ps1 += p10 + p11;
#if PV8
            pv[t][0] = p00 * vs0; pv[t][1] = p01 * vs1; pv[t][2] = p10 * vs0; pv[t][3] = p11 * vs1;
            px0 = fmaxf(px0, fmaxf(pv[t][0], pv[t][1])); px1 = fmaxf(px1, fmaxf(pv[t][2], pv[t][3]));
#else
            *reinterpret_cast<__half2*>(sP + f0 * PST + col) = __floats2half2_rn(p00 * vs0, p01 * vs1);
            *reinterpret_cast<__half2*>(sP + f1 * PST + col) = __floats2half2_rn(p10 * vs0, p11 * vs1);
#endif
        }
#if PV8
        px0 = fmaxf(px0, __shfl_xor_sync(0xffffffffu, px0, 1)); px0 = fmaxf(px0, __shfl_xor_sync(0xffffffffu, px0, 2));
        px1 = fmaxf(px1, __shfl_xor_sync(0xffffffffu, px1, 1)); px1 = fmaxf(px1, __shfl_xor_sync(0xffffffffu, px1, 2));
        float* sX = sM + NQ * RB;                                // [NQ][RB] maximos de P' por cuarto
        if (tig == 0) { sX[hh * RB + f0] = px0; sX[hh * RB + f1] = px1; }
        __syncthreads();
        float qx0 = 0.f, qx1 = 0.f;
#pragma unroll
        for (int x = 0; x < NQ; ++x) { qx0 = fmaxf(qx0, sX[x * RB + f0]); qx1 = fmaxf(qx1, sX[x * RB + f1]); }
#if PHL
        const float NIV = 32639.f;                               // hi en [-127,127], lo en [-128,127]
#else
        const float NIV = 127.f;
#endif
        const float r0 = qx0 > 0.f ? NIV / qx0 : 0.f, r1 = qx1 > 0.f ? NIV / qx1 : 0.f;
        const float e0 = qx0 / NIV, e1 = qx1 / NIV;              // escala de fila de este tile
        signed char* sP8 = reinterpret_cast<signed char*>(sP);   // [RB][PST*2 bytes]: plano hi, y lo a +64
#pragma unroll
        for (int t = 0; t < KPQ / 8; ++t) {
            const int col = hh * KPQ + t * 8 + tig * 2;
            const int a = __float2int_rn(pv[t][0] * r0), b = __float2int_rn(pv[t][1] * r0);
            const int c = __float2int_rn(pv[t][2] * r1), d = __float2int_rn(pv[t][3] * r1);
#if PHL
            const int ah = (a + 128) >> 8, bh_ = (b + 128) >> 8, ch = (c + 128) >> 8, dh = (d + 128) >> 8;
            const int al = a - (ah << 8), bl = b - (bh_ << 8), cl = c - (ch << 8), dl = d - (dh << 8);
            *reinterpret_cast<unsigned short*>(sP8 + f0 * PST * 2 + col) = (unsigned short)((ah & 0xff) | ((bh_ & 0xff) << 8));
            *reinterpret_cast<unsigned short*>(sP8 + f1 * PST * 2 + col) = (unsigned short)((ch & 0xff) | ((dh & 0xff) << 8));
            *reinterpret_cast<unsigned short*>(sP8 + f0 * PST * 2 + 64 + col) = (unsigned short)((al & 0xff) | ((bl & 0xff) << 8));
            *reinterpret_cast<unsigned short*>(sP8 + f1 * PST * 2 + 64 + col) = (unsigned short)((cl & 0xff) | ((dl & 0xff) << 8));
#else
            *reinterpret_cast<unsigned short*>(sP8 + f0 * PST * 2 + col) = (unsigned short)((a & 0xff) | ((b & 0xff) << 8));
            *reinterpret_cast<unsigned short*>(sP8 + f1 * PST * 2 + col) = (unsigned short)((c & 0xff) | ((d & 0xff) << 8));
#endif
        }
#endif
        l0 = l0 * a0 + ps0; l1 = l1 * a1 + ps1;              // parcial de esta mitad de keys
        // O *= a (filas f0 / f1)
#pragma unroll
        for (int i = 0; i < DPQ / 8; ++i) { o[i][0] *= a0; o[i][1] *= a0; o[i][2] *= a1; o[i][3] *= a1; }
        __syncthreads();
        // ── O += P'.V : 16 filas x DPQ dims (cuarto hh) x 64 keys
        const unsigned char* V_ = sV + st * QD * VST;
#if PV8
        {
            const signed char* sP8 = reinterpret_cast<const signed char*>(sP);
            int t32[DPQ / 8][4];
#pragma unroll
            for (int i = 0; i < DPQ / 8; ++i) t32[i][0] = t32[i][1] = t32[i][2] = t32[i][3] = 0;
#if PHL
            int u32[DPQ / 8][4];
#pragma unroll
            for (int i = 0; i < DPQ / 8; ++i) u32[i][0] = u32[i][1] = u32[i][2] = u32[i][3] = 0;
#endif
#pragma unroll
            for (int kk = 0; kk < TK; kk += 32) {
                unsigned pa[4];
                pa[0] = *reinterpret_cast<const unsigned*>(sP8 + f0 * PST * 2 + kk + tig * 4);
                pa[1] = *reinterpret_cast<const unsigned*>(sP8 + f1 * PST * 2 + kk + tig * 4);
                pa[2] = *reinterpret_cast<const unsigned*>(sP8 + f0 * PST * 2 + kk + 16 + tig * 4);
                pa[3] = *reinterpret_cast<const unsigned*>(sP8 + f1 * PST * 2 + kk + 16 + tig * 4);
#if PHL
                unsigned pl[4];
                pl[0] = *reinterpret_cast<const unsigned*>(sP8 + f0 * PST * 2 + 64 + kk + tig * 4);
                pl[1] = *reinterpret_cast<const unsigned*>(sP8 + f1 * PST * 2 + 64 + kk + tig * 4);
                pl[2] = *reinterpret_cast<const unsigned*>(sP8 + f0 * PST * 2 + 64 + kk + 16 + tig * 4);
                pl[3] = *reinterpret_cast<const unsigned*>(sP8 + f1 * PST * 2 + 64 + kk + 16 + tig * 4);
#endif
#pragma unroll
                for (int i = 0; i < DPQ / 8; ++i) {
                    const unsigned char* vr = V_ + (hh * DPQ + i * 8 + gid) * VST + kk + tig * 4;
                    const unsigned vb0 = *reinterpret_cast<const unsigned*>(vr);
                    const unsigned vb1 = *reinterpret_cast<const unsigned*>(vr + 16);
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+r"(t32[i][0]), "+r"(t32[i][1]), "+r"(t32[i][2]), "+r"(t32[i][3])
                                 : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb0), "r"(vb1));
#if PHL
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                                 : "+r"(u32[i][0]), "+r"(u32[i][1]), "+r"(u32[i][2]), "+r"(u32[i][3])
                                 : "r"(pl[0]), "r"(pl[1]), "r"(pl[2]), "r"(pl[3]), "r"(vb0), "r"(vb1));
#endif
                }
            }
#pragma unroll
            for (int i = 0; i < DPQ / 8; ++i) {
#if PHL
                o[i][0] += (float)(t32[i][0] * 256 + u32[i][0]) * e0; o[i][1] += (float)(t32[i][1] * 256 + u32[i][1]) * e0;
                o[i][2] += (float)(t32[i][2] * 256 + u32[i][2]) * e1; o[i][3] += (float)(t32[i][3] * 256 + u32[i][3]) * e1;
#else
                o[i][0] += (float)t32[i][0] * e0; o[i][1] += (float)t32[i][1] * e0;
                o[i][2] += (float)t32[i][2] * e1; o[i][3] += (float)t32[i][3] * e1;
#endif
            }
        }
#else
#if ACC16
        // acumulador fp16 por tile (tasa doble en GA102: math_pipe_throttle era el HMMA f32); se suma al O fp32
        unsigned t16[DPQ / 8][2];
#pragma unroll
        for (int i = 0; i < DPQ / 8; ++i) t16[i][0] = t16[i][1] = 0u;
#endif
#pragma unroll
        for (int kk = 0; kk < TK; kk += 16) {
            unsigned pa[4];
            pa[0] = *reinterpret_cast<const unsigned*>(sP + f0 * PST + kk + tig * 2);
            pa[1] = *reinterpret_cast<const unsigned*>(sP + f1 * PST + kk + tig * 2);
            pa[2] = *reinterpret_cast<const unsigned*>(sP + f0 * PST + kk + 8 + tig * 2);
            pa[3] = *reinterpret_cast<const unsigned*>(sP + f1 * PST + kk + 8 + tig * 2);
#pragma unroll
            for (int i = 0; i < DPQ / 8; ++i) {
                const unsigned char* vr = V_ + (hh * DPQ + i * 8 + gid) * VST + kk + tig * 2;
                const unsigned vb0 = i8x2_a_h2(*reinterpret_cast<const unsigned short*>(vr));
                const unsigned vb1 = i8x2_a_h2(*reinterpret_cast<const unsigned short*>(vr + 8));
#if ACC16
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                             "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(t16[i][0]), "+r"(t16[i][1])
                             : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb0), "r"(vb1));
#else
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(o[i][0]), "+f"(o[i][1]), "+f"(o[i][2]), "+f"(o[i][3])
                             : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(vb0), "r"(vb1));
#endif
            }
        }
#if ACC16
#pragma unroll
        for (int i = 0; i < DPQ / 8; ++i) {
            const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&t16[i][0]));
            const float2 y = __half22float2(*reinterpret_cast<const __half2*>(&t16[i][1]));
            o[i][0] += x.x; o[i][1] += x.y; o[i][2] += y.x; o[i][3] += y.y;
        }
#endif
#endif
        __syncthreads();
        st ^= 1;
    }
    __pipeline_wait_prior(0);
    // l total por fila = suma de las 4 lanes (tig) y de las dos mitades
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    if (tig == 0) { sM[hh * RB + f0] = l0; sM[hh * RB + f1] = l1; }
    __syncthreads();
    float lt0 = 0.f, lt1 = 0.f;
#pragma unroll
    for (int x = 0; x < NQ; ++x) { lt0 += sM[x * RB + f0]; lt1 += sM[x * RB + f1]; }
    const float i0 = lt0 > 0.f ? 1.f / lt0 : 0.f, i1 = lt1 > 0.f ? 1.f / lt1 : 0.f;
    __half* ob = Op + so * QD;
#pragma unroll
    for (int i = 0; i < DPQ / 8; ++i) {
        const int d = hh * DPQ + i * 8 + tig * 2;
        *reinterpret_cast<__half2*>(ob + (size_t)f0 * QD + d) = __floats2half2_rn(o[i][0] * i0, o[i][1] * i0);
        *reinterpret_cast<__half2*>(ob + (size_t)f1 * QD + d) = __floats2half2_rn(o[i][2] * i1, o[i][3] * i1);
    }
    if (hh == 0 && tig == 0) {
        Mp[so + f0] = m0; Mp[so + f1] = m1; Lp[so + f0] = lt0; Lp[so + f1] = lt1;
    }
}

// Union de los grupos usados: out[b, j, h*G+g, :] = sum_g O_g l_g 2^(m_g - M) / sum_g l_g 2^(m_g - M).
// 8 warps: cada warp recorre grupos g = w, w+8, ...; cada lane 8 dims (uint4 de halves); suma entre warps en shared.
// ROTV=1: la V del pool esta ROTADA en d (Hadamard/16 con signos, como K): la salida sale en esa base y se
// des-rota aca: O = (O' Hn) * s, con una FWHT de 256 en shared (8 etapas).
#ifndef ROTV
#define ROTV 0
#endif
extern "C" __global__ void __launch_bounds__(256)
sk30_union(const __half* __restrict__ Op, const float* __restrict__ Mp, const float* __restrict__ Lp,
           const int* __restrict__ seqlen, int GMAX, int B, int NH, int L, int G, __half* __restrict__ out,
           const int* __restrict__ signos)
{
    __shared__ float cg[256];
    __shared__ float acc[8][QD];
    __shared__ float red[8];
    const int f = blockIdx.x, bh = blockIdx.y, b = bh / NH, h = bh % NH;
    if (f >= L * G) return;
    const unsigned tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int n = seqlen[b];
    int paso = (n + GMAX - 1) / GMAX; paso = (paso + TK - 1) / TK * TK;
    const int ng = min(GMAX, (n + paso - 1) / paso);
    // maximo y pesos de cada grupo
    float mg = NEG;
    if ((int)tid < ng) mg = Mp[(((size_t)tid * B + b) * NH + h) * RB + f];
    float M = mg;
#pragma unroll
    for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    if (lane == 0) red[w] = M;
    __syncthreads();
    M = red[0];
#pragma unroll
    for (int x = 1; x < 8; ++x) M = fmaxf(M, red[x]);
    if ((int)tid < ng) {
        const float l = Lp[(((size_t)tid * B + b) * NH + h) * RB + f];
        cg[tid] = l > 0.f ? l * exp2f(mg - M) : 0.f;
    }
    __syncthreads();
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int g = w; g < ng; g += 8) {
        const float c = cg[g];
        if (c == 0.f) continue;
        const uint4 u = *reinterpret_cast<const uint4*>(Op + ((((size_t)g * B + b) * NH + h) * RB + f) * QD + lane * 8);
        const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float2 x = __half22float2(h2[r]);
            s[2 * r] += c * x.x; s[2 * r + 1] += c * x.y;
        }
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[w][lane * 8 + r] = s[r];
    float den = 0.f;
    for (int g = 0; g < ng; ++g) den += cg[g];                 // igual en todos los hilos (shared)
    __syncthreads();
    float num = 0.f;
#pragma unroll
    for (int x = 0; x < 8; ++x) num += acc[x][tid];
    const int j = f / G, gq = f % G;
    float y = den > 0.f ? num / den : 0.f;
#if ROTV
    __shared__ float xr[QD];
    xr[tid] = y;
#pragma unroll
    for (int s = 1; s < QD; s <<= 1) {
        __syncthreads();
        const float a = xr[tid], c = xr[tid ^ s];
        __syncthreads();
        xr[tid] = (tid & s) ? c - a : a + c;
    }
    __syncthreads();
    y = xr[tid] * (1.f / 16.f) * (float)signos[tid];
#endif
    out[(((size_t)b * L + j) * (NH * G) + h * G + gq) * QD + tid] = __float2half(y);
}
