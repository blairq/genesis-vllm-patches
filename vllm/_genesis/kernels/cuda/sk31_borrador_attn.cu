// SK-31: atencion del borrador DFlash2 en una pasada (diseno de SK-30) sobre la KV int8_per_token_head de vLLM.
//
// Borrador: head 128, G = HQ/NKV cabezas Q por cabeza KV, 9 queries por pedido NO causales con ventana
// simetrica (|q - k| < SW, SW = 2048). Hoy va por kernel_unified_attention (Triton, PN124 3D): 185 us por paso
// con 1 pedido y 530 con 4, para ~2k keys por capa.
//
// KV de vLLM (logico [bloques, NKV, BS, 2*(128+4)] int8, strides fisicos por argumento):
//   [K 128 int8 | escala K fp32 | V 128 int8 | escala V fp32] por (bloque, cabeza, slot).
//   S = Q.K   mma m16n8k32 s8 (Q int8 por fila, cuantizada aca; K fila por token = fragmento B directo)
//   p = 2^(s - m) en fp32 en linea; P' = p * vs en fp16
//   O += P'.V  V int8 -> fp16 en shared [key][d], fragmento B con ldmatrix.trans; mma f16, acumulador fp16
// Filas: r = j * G + g (query j del bloque, cabeza Q g del grupo); RB = 64 (L*G <= 64).
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <stdint.h>

#define QD 128
#define RB 64
#define TK 64
#define KST 144          // paso de fila de K int8 en shared (128 + 16)
#define VST 136          // paso de fila de V fp16 en shared (halves: 128 + 8)
#define PST 72           // paso de fila de P (halves)
#define NEG (-1e30f)
#define NW 8             // 4 tiles de 16 filas x 2 mitades (keys en S, dims en O)

__device__ __forceinline__ unsigned sdir(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ unsigned i8x2_a_h2(unsigned short u) {
    const unsigned x = (unsigned)u ^ 0x8080u;
    unsigned r;
    asm("prmt.b32 %0, %1, %2, 0x7170;" : "=r"(r) : "r"(x), "r"(0x64646464u));
    const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&r), __float2half2_rn(1152.f));
    return *reinterpret_cast<const unsigned*>(&h);
}

extern "C" __global__ void __launch_bounds__(NW * 32)
sk31_borrador(const __half* __restrict__ q,          // [B*L, NKV*G, 128] fp16 (rotada como K)
              int qs0,                              // paso de fila de q (elementos)
              const signed char* __restrict__ kv,   // cache de la capa
              long long sb, long long sh, long long ss,   // strides en BYTES: bloque, cabeza, slot
              const int* __restrict__ tabla, int tstride,
              const int* __restrict__ seqlen,       // [B]
              int NKV, int G, int L, int BS, int SW, float escala,   // escala = softmax_scale * log2(e)
              int GMAX,
              __half* __restrict__ Op, float* __restrict__ Mp, float* __restrict__ Lp)   // [GMAX, B, NKV, RB, (128)]
{
    extern __shared__ __align__(16) unsigned char sm[];
    unsigned char* sK = sm;                                  // [2][TK][KST] int8
    signed char* sV8 = reinterpret_cast<signed char*>(sK + 2 * TK * KST);   // [2][TK][128] int8
    __half* sV = reinterpret_cast<__half*>(sV8 + 2 * TK * QD);             // [TK][VST] fp16
    __half* sK16 = sV + TK * VST;                            // [TK][VST] K en fp16 (Q.K en fp16, sin cuantizar Q)
    __half* sP = sK16 + TK * VST;                            // [RB][PST]
    float* sEs = reinterpret_cast<float*>(sP + RB * PST);    // [2][TK][2] escalas K, V
    float* sM = sEs + 2 * TK * 2;                            // [2][RB]

    const int grp = blockIdx.x, bh = blockIdx.y, b = bh / NKV, h = bh % NKV;
    const int B = gridDim.y / NKV;
    const unsigned tid = threadIdx.x, lane = tid & 31, w = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int rt = w & 3, hh = w >> 2;
    const int n = seqlen[b], ctx = n - L;
    // keys posibles: ventana de la PRIMERA query hasta el final: [max(0, ctx - SW + 1), n)
    const int kbase = max(0, ctx - SW + 1), ntot = n - kbase;
    int paso = (ntot + GMAX - 1) / GMAX; paso = (paso + TK - 1) / TK * TK;
    const int k_ini = kbase + grp * paso, k_fin = min(n, k_ini + paso);
    const size_t so = (((size_t)grp * B + b) * NKV + h) * RB;
    if (k_ini >= k_fin) {
        if (tid < RB) { Mp[so + tid] = NEG; Lp[so + tid] = 0.f; }
        return;
    }
    // ── Q de este warp: 16 filas (rt) x 128 dims, cuantizada a int8 por fila; qa[4 pasos de k=32][4]
    const int f0 = rt * 16 + gid, f1 = f0 + 8;
    const bool v0 = f0 < L * G, v1 = f1 < L * G;
    const int j0 = v0 ? f0 / G : 0, j1 = v1 ? f1 / G : 0;
    const __half* q0 = q + (size_t)(b * L + j0) * qs0 + (size_t)(h * G + (v0 ? f0 % G : 0)) * QD;
    const __half* q1 = q + (size_t)(b * L + j1) * qs0 + (size_t)(h * G + (v1 ? f1 % G : 0)) * QD;
    // Q fp16 en registros: fragmentos A de m16n8k16 (8 pasos de k=16): filas f0/f1, k = 2tig (+8)
    unsigned qa[8][4];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        qa[s][0] = v0 ? *reinterpret_cast<const unsigned*>(q0 + s * 16 + tig * 2) : 0u;
        qa[s][1] = v1 ? *reinterpret_cast<const unsigned*>(q1 + s * 16 + tig * 2) : 0u;
        qa[s][2] = v0 ? *reinterpret_cast<const unsigned*>(q0 + s * 16 + 8 + tig * 2) : 0u;
        qa[s][3] = v1 ? *reinterpret_cast<const unsigned*>(q1 + s * 16 + 8 + tig * 2) : 0u;
    }
    const int qp0 = ctx + j0, qp1 = ctx + j1;                // posicion absoluta de cada query

    float o[8][4];                                           // 16 filas x 64 dims (mitad hh)
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;

    auto cargar = [&](int st, int k0) {
        // K y V de 64 keys: registro de 264 B por (slot, cabeza) -> V en el byte 132: solo alineado a 4.
        // Hilo = (key = tid/4, cuarto = tid%4): una direccion por hilo, 8 copias de 4 B por lado.
        {
            const int key = tid >> 2, part = tid & 3, kk = k0 + key, ok = kk < k_fin, kc = ok ? kk : k0;
            const long long blk = tabla[(size_t)b * tstride + kc / BS];
            const signed char* src = kv + blk * sb + (long long)h * sh + (long long)(kc % BS) * ss + part * 32;
            unsigned char* dk = sK + (st * TK + key) * KST + part * 32;
            unsigned char* dv = reinterpret_cast<unsigned char*>(sV8 + (st * TK + key) * QD + part * 32);
            const int nb = ok ? 4 : 0;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(sdir(dk + c * 4)), "l"(src + c * 4), "r"(nb));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(sdir(dv + c * 4)), "l"(src + QD + 4 + c * 4), "r"(nb));
            }
        }
        if (tid < 2 * TK) {
            const int lado = tid >> 6, key = tid & 63, kk = k0 + key, ok = kk < k_fin, kc = ok ? kk : k0;
            const long long blk = tabla[(size_t)b * tstride + kc / BS];
            const signed char* src = kv + blk * sb + (long long)h * sh + (long long)(kc % BS) * ss + lado * (QD + 4) + QD;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                         :: "r"(sdir(sEs + (st * TK + key) * 2 + lado)), "l"(src), "r"(ok ? 4 : 0));
        }
        __pipeline_commit();
    };

    int st = 0;
    cargar(0, k_ini);
    for (int k0 = k_ini; k0 < k_fin; k0 += TK) {
        if (k0 + TK < k_fin) cargar(st ^ 1, k0 + TK); else __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
        // K y V int8 -> fp16 [key][d] (una vez por tile, entre todos)
        {
            const signed char* v8 = sV8 + st * TK * QD;
            const unsigned char* k8 = sK + st * TK * KST;
#pragma unroll
            for (int c = 0; c < TK * QD / 2 / 256; ++c) {
                const int e = tid + c * 256, key = e >> 6, d2 = e & 63;
                const unsigned short u = *reinterpret_cast<const unsigned short*>(v8 + key * QD + d2 * 2);
                *reinterpret_cast<unsigned*>(sV + key * VST + d2 * 2) = i8x2_a_h2(u);
                const unsigned short uk = *reinterpret_cast<const unsigned short*>(k8 + key * KST + d2 * 2);
                *reinterpret_cast<unsigned*>(sK16 + key * VST + d2 * 2) = i8x2_a_h2(uk);
            }
        }
        __syncthreads();
        const float* E_ = sEs + st * TK * 2;
        // S: 16 filas x 32 keys (mitad hh), mma m16n8k16 f16 -> f32; B = K fp16 [key][d] (fila por key)
        float sacc[4][4];
#pragma unroll
        for (int t = 0; t < 4; ++t) sacc[t][0] = sacc[t][1] = sacc[t][2] = sacc[t][3] = 0.f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const __half* kr = sK16 + (hh * 32 + t * 8 + gid) * VST + tig * 2;
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const unsigned b0 = *reinterpret_cast<const unsigned*>(kr + s * 16);
                const unsigned b1 = *reinterpret_cast<const unsigned*>(kr + s * 16 + 8);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(sacc[t][0]), "+f"(sacc[t][1]), "+f"(sacc[t][2]), "+f"(sacc[t][3])
                             : "r"(qa[s][0]), "r"(qa[s][1]), "r"(qa[s][2]), "r"(qa[s][3]), "r"(b0), "r"(b1));
            }
        }
        float sf[4][4];
        float mx0t = NEG, mx1t = NEG;
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = hh * 32 + t * 8 + tig * 2 + (e & 1), kk = k0 + col;
                const bool fila1 = e >= 2;
                const int qp = fila1 ? qp1 : qp0;
                const bool vis = kk < k_fin && (fila1 ? v1 : v0) && (qp - kk < SW) && (kk - qp < SW);
                const float v = vis ? sacc[t][e] * escala * E_[col * 2] : NEG;
                sf[t][e] = v;
                if (fila1) mx1t = fmaxf(mx1t, v); else mx0t = fmaxf(mx0t, v);
            }
        mx0t = fmaxf(mx0t, __shfl_xor_sync(0xffffffffu, mx0t, 1)); mx0t = fmaxf(mx0t, __shfl_xor_sync(0xffffffffu, mx0t, 2));
        mx1t = fmaxf(mx1t, __shfl_xor_sync(0xffffffffu, mx1t, 1)); mx1t = fmaxf(mx1t, __shfl_xor_sync(0xffffffffu, mx1t, 2));
        if (tig == 0) { sM[hh * RB + f0] = mx0t; sM[hh * RB + f1] = mx1t; }
        __syncthreads();
        const float mn0 = fmaxf(m0, fmaxf(sM[f0], sM[RB + f0])), mn1 = fmaxf(m1, fmaxf(sM[f1], sM[RB + f1]));
        const float a0 = exp2f(m0 - mn0), a1 = exp2f(m1 - mn1);
        m0 = mn0; m1 = mn1;
        float ps0 = 0.f, ps1 = 0.f;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int col = hh * 32 + t * 8 + tig * 2;
            const float vs0 = E_[col * 2 + 1], vs1 = E_[(col + 1) * 2 + 1];
            const float p00 = sf[t][0] > 0.5f * NEG ? exp2f(sf[t][0] - m0) : 0.f;
            const float p01 = sf[t][1] > 0.5f * NEG ? exp2f(sf[t][1] - m0) : 0.f;
            const float p10 = sf[t][2] > 0.5f * NEG ? exp2f(sf[t][2] - m1) : 0.f;
            const float p11 = sf[t][3] > 0.5f * NEG ? exp2f(sf[t][3] - m1) : 0.f;
            ps0 += p00 + p01; ps1 += p10 + p11;
            *reinterpret_cast<__half2*>(sP + f0 * PST + col) = __floats2half2_rn(p00 * vs0, p01 * vs1);
            *reinterpret_cast<__half2*>(sP + f1 * PST + col) = __floats2half2_rn(p10 * vs0, p11 * vs1);
        }
        l0 = l0 * a0 + ps0; l1 = l1 * a1 + ps1;
#pragma unroll
        for (int i = 0; i < 8; ++i) { o[i][0] *= a0; o[i][1] *= a0; o[i][2] *= a1; o[i][3] *= a1; }
        __syncthreads();
        // O += P'.V: 16 filas x 64 dims (mitad hh) x 64 keys; B de V [key][d] con ldmatrix.trans
        unsigned t16[8][2];
#pragma unroll
        for (int i = 0; i < 8; ++i) t16[i][0] = t16[i][1] = 0u;
#pragma unroll
        for (int kk = 0; kk < TK; kk += 16) {
            unsigned pa[4];
            pa[0] = *reinterpret_cast<const unsigned*>(sP + f0 * PST + kk + tig * 2);
            pa[1] = *reinterpret_cast<const unsigned*>(sP + f1 * PST + kk + tig * 2);
            pa[2] = *reinterpret_cast<const unsigned*>(sP + f0 * PST + kk + 8 + tig * 2);
            pa[3] = *reinterpret_cast<const unsigned*>(sP + f1 * PST + kk + 8 + tig * 2);
#pragma unroll
            for (int i = 0; i < 8; i += 2) {                   // dos tiles de n=8 por ldmatrix.x4.trans
                const int dn = hh * 64 + i * 8;
                // matrices: (k 0-7, n dn..+7), (k 8-15, n dn..), (k 0-7, n dn+8..), (k 8-15, n dn+8..)
                const int mtx = lane >> 3, r = lane & 7;
                const __half* pr = sV + (kk + (mtx & 1) * 8 + r) * VST + dn + (mtx >> 1) * 8;
                unsigned b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(sdir(pr)));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(t16[i][0]), "+r"(t16[i][1]) : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(b0), "r"(b1));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                             : "+r"(t16[i + 1][0]), "+r"(t16[i + 1][1]) : "r"(pa[0]), "r"(pa[1]), "r"(pa[2]), "r"(pa[3]), "r"(b2), "r"(b3));
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 x = __half22float2(*reinterpret_cast<const __half2*>(&t16[i][0]));
            const float2 y = __half22float2(*reinterpret_cast<const __half2*>(&t16[i][1]));
            o[i][0] += x.x; o[i][1] += x.y; o[i][2] += y.x; o[i][3] += y.y;
        }
        __syncthreads();
        st ^= 1;
    }
    __pipeline_wait_prior(0);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    if (tig == 0) { sM[hh * RB + f0] = l0; sM[hh * RB + f1] = l1; }
    __syncthreads();
    const float lt0 = sM[f0] + sM[RB + f0], lt1 = sM[f1] + sM[RB + f1];
    const float i0 = lt0 > 0.f ? 1.f / lt0 : 0.f, i1 = lt1 > 0.f ? 1.f / lt1 : 0.f;
    __half* ob = Op + so * QD;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = hh * 64 + i * 8 + tig * 2;
        *reinterpret_cast<__half2*>(ob + (size_t)f0 * QD + d) = __floats2half2_rn(o[i][0] * i0, o[i][1] * i0);
        *reinterpret_cast<__half2*>(ob + (size_t)f1 * QD + d) = __floats2half2_rn(o[i][2] * i1, o[i][3] * i1);
    }
    if (hh == 0 && tig == 0) { Mp[so + f0] = m0; Mp[so + f1] = m1; Lp[so + f0] = lt0; Lp[so + f1] = lt1; }
}

// Union (como la de SK-30, head 128): out[(b*L + j), h*G + g, :] con paso de fila os0
extern "C" __global__ void __launch_bounds__(128)
sk31_union(const __half* __restrict__ Op, const float* __restrict__ Mp, const float* __restrict__ Lp,
           int GMAX, int B, int NKV, int L, int G, __half* __restrict__ out, int os0)
{
    __shared__ float cg[256];
    __shared__ float acc[4][QD];
    __shared__ float red[4];
    const int f = blockIdx.x, bh = blockIdx.y, b = bh / NKV, h = bh % NKV;
    if (f >= L * G) return;
    const unsigned tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    float mg = NEG;
    for (int g = tid; g < GMAX; g += 128) mg = fmaxf(mg, Mp[(((size_t)g * B + b) * NKV + h) * RB + f]);
    float M = mg;
#pragma unroll
    for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    if (lane == 0) red[w] = M;
    __syncthreads();
    M = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    for (int g = tid; g < GMAX; g += 128) {
        const size_t i = (((size_t)g * B + b) * NKV + h) * RB + f;
        const float l = Lp[i];
        cg[g] = l > 0.f ? l * exp2f(Mp[i] - M) : 0.f;
    }
    __syncthreads();
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    for (int g = w; g < GMAX; g += 4) {
        const float c = cg[g];
        if (c == 0.f) continue;
        const uint2 u = *reinterpret_cast<const uint2*>(Op + ((((size_t)g * B + b) * NKV + h) * RB + f) * QD + lane * 4);
        const __half2* h2 = reinterpret_cast<const __half2*>(&u);
        const float2 x = __half22float2(h2[0]), y = __half22float2(h2[1]);
        s[0] += c * x.x; s[1] += c * x.y; s[2] += c * y.x; s[3] += c * y.y;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[w][lane * 4 + r] = s[r];
    float den = 0.f;
    for (int g = 0; g < GMAX; ++g) den += cg[g];
    __syncthreads();
    const float num = acc[0][tid] + acc[1][tid] + acc[2][tid] + acc[3][tid];
    const int j = f / G, gq = f % G;
    out[(size_t)(b * L + j) * os0 + (size_t)(h * G + gq) * QD + tid] = __float2half(den > 0.f ? num / den : 0.f);
}
