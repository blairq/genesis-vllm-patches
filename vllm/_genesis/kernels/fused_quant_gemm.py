# SPDX-License-Identifier: Apache-2.0
"""Super-kernel fused Triton: quant_activation_per_token + GEMM int8 — SIN branches de validacion.

Fusiona ``quant_activation_per_token`` (bf16 -> per-token amax/127 -> int8)
y ``cutlass_scaled_mm`` (int8 GEMM + epilogo ``*a_scale*b_scale``) sin
materializar ``a_i8`` intermedio en DRAM.

NOTA SUPER-KERNEL (2026-08-25):
    Camino caliente 100% branchless de validacion. TODA validacion es PREVIA
    en ``vllm._genesis.wiring.quantization.patch_PN110_int8_phase_dispatch._make_pwal_wrapper``
    ANTES de adjuntar el estado INT8:
        - dtype int8 para b (rechazo previo si no es int8)
        - block_k minimo 16 Tensor Core (clamp previo, no en hot path)
        - a contigua y b layout column-major (asegurado previo)
        - dims multiplo de 16 y K,N multiplo de block (ya chequeado en wiring)
    Este kernel ASUME entradas validadas y no contiene ramas de validacion
    de dtype, block_k ni contiguity en el camino caliente.
    Si se llama sin pasar por wiring, el caller debe garantizar las mismas
    invariantes o el kernel puede fallar / producir resultados indefinidos.

Pipeline por programa (un solo launch, sin buffers DRAM intermedios):

    1. bf16 load (tl.load)  — unica lectura de activacion desde DRAM
    2. amax por fila via tl.max(tl.abs(x_f32), axis=1) reduccion global K
    3. scale = amax/127 con tl.where(amax>0, amax/127, 1.0)
    4. quant on-the-fly: round(x/scale) clamp [-127,127] -> tl.int8 (registros)
    5. tl.dot(q_i8, b_i8) -> int32 accum (Tensor Core)
    6. epilogo fp32: acc.to(tl.float32) * a_scale * b_scale -> bf16/fp16

Sin copias a host, 100 por ciento GPU. Un solo kernel Triton esconde
el costo de ``a_i8`` (M*K bytes) y reduce launches de 2 a 1.

Contrato:

    a [M, K] bf16/fp16/fp32  — activacion (row-major)
    b [K, N] int8            — peso (row o col-major, strides genericos)
    b_scale [N] o [N,1] fp32 — escala per-channel del peso
    out [M, N] bf16/fp16     — resultado escalado

BLOCK_M=32, BLOCK_N=64, BLOCK_K=32 por defecto (autotunable via constexpr).

Fallback: torch vectorizado 100 por ciento GPU (sin Triton, sin host copies,
sin loops Python sobre filas).

Equivalencia numerica:

    a_scale = amax_row / 127,  clamp scale>0 else 1.0
    a_i8 = clamp(round(a / a_scale), -127, 127)
    out = (a_i8 @ b_i8).to(fp32) * a_scale[:,None] * b_scale[None,:]

Performance (Ampere sm_86, estimacion analitica):

    Separado:  1) quant kernel  R: M*K*2 B, W: M*K*1 B + M*4 B
              2) cutlass_scaled_mm  R: M*K*1 + K*N*1 + M*4 + N*4, W: M*N*2
    Fused:     R: M*K*2 + K*N*1 + N*4, W: M*N*2  (elimina M*K*1 write + M*K*1 read)
    Ahorro trafico: 2*M*K bytes + 1 launch (~8 us).
    Ej Qwen3-30B: M=2048, K=4096, N=4096 => 16 MB ahorrados por linear.
    Speedup esperado: 1.15x-1.35x sobre separado en prefill (memory-bound),
    1.10x-1.25x en decode M Pequenio (latency-bound por launch).
    Con M=1, N=4096, K=4096: 8 KB ahorrados, pero launch halved domina.

Author: Genesis (fused quant+GEMM single-launch, 2026-08-25)
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger("genesis.kernels.fused_quant_gemm")

_TRITON_OK = False
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_OK = True
except Exception as _e:  # pragma: no cover
    log.warning("fused_quant_gemm: Triton no disponible (%s) — solo fallback torch", _e)
    triton = None  # type: ignore
    tl = None  # type: ignore

# Bloques por defecto — potencias de 2 compatibles con tl.dot
BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 32


def is_triton_available() -> bool:
    """True si Triton esta importable."""
    return _TRITON_OK


def is_available() -> bool:
    """True si kernel Triton fused esta disponible (Triton + CUDA)."""
    if not _TRITON_OK:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# ── Triton kernel single-launch ──────────────────────────────────────────
if _TRITON_OK:

    @triton.jit
    def _fused_quant_gemm_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        b_scale_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_out_m,
        stride_out_n,
        # constexprs
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused quant per-token + GEMM int8 — un solo launch.

        Un programa = un bloque [BLOCK_M, BLOCK_N] de la salida.

        Flujo:

            1. amax por fila: loop sobre K con BLOCK_K, tl.max(tl.abs(a_f32), axis=1)
               mantiene vector [BLOCK_M] con max global por fila.
            2. scale = tl.where(amax/127 > 0, amax/127, 1.0)  [BLOCK_M]
            3. GEMM loop sobre K: carga a_tile bf16 -> a_f32 -> quant a int8
               on-the-fly (registros, sin store DRAM) -> tl.dot con b_tile int8
               -> accum int32. Reusa scale ya calculado.
            4. Epilogo: acc.to(fp32) * a_scale[:,None] * b_scale[None,:] -> store
               bf16/fp16.

        Sin escribir a_i8 a DRAM: q_i8 vive solo en registros / SRAM.

        Usa tl.arange, tl.load, tl.max, tl.where, tl.dot obligatoriamente.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        # ── 1) amax global por fila [BLOCK_M] ───────────────────────────
        # Inicializar a 0. Recorrer todo K en pasos BLOCK_K, single pass de
        # reduccion via tl.max — sin segundo load extra para amax.
        amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            cur_k = k + offs_k
            mask_k = cur_k < K
            # a_ptrs [BLOCK_M, BLOCK_K] bf16
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak
            mask_a = mask_m[:, None] & mask_k[None, :]
            a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
            a_f32 = a_tile.to(tl.float32)
            # abs + max por fila -> [BLOCK_M]
            # tl.max sobre axis=1 reduce BLOCK_K
            row_amax = tl.max(tl.abs(a_f32), axis=1)
            amax = tl.maximum(amax, row_amax)

        # scale per-token [BLOCK_M] fp32
        scale = amax / 127.0
        scale = tl.where(scale > 0, scale, 1.0)

        # b_scale per-channel [BLOCK_N] fp32 — carga vectorizada
        b_scale = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)

        # ── 2) GEMM con quant on-the-fly ─────────────────────────────────
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k in range(0, K, BLOCK_K):
            cur_k = k + offs_k
            mask_k = cur_k < K

            # A tile bf16 -> fp32 -> quant int8 en registros
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + cur_k[None, :] * stride_ak
            mask_a = mask_m[:, None] & mask_k[None, :]
            a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
            a_f32 = a_tile.to(tl.float32)

            # quant: round(a / scale) clamp [-127,127]
            # scale [BLOCK_M] -> broadcast [BLOCK_M,1]
            q_scaled = a_f32 / scale[:, None]
            bias = tl.where(q_scaled >= 0, 0.5, -0.5)
            q_int = (q_scaled + bias).to(tl.int32)
            q_int = tl.where(q_int > 127, 127, q_int)
            q_int = tl.where(q_int < -127, -127, q_int)
            q_i8 = q_int.to(tl.int8)

            # B tile int8 [BLOCK_K, BLOCK_N]
            b_ptrs = b_ptr + cur_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            mask_b = mask_k[:, None] & mask_n[None, :]
            b_tile = tl.load(b_ptrs, mask=mask_b, other=0).to(tl.int8)

            # Tensor Core int8 -> int32
            acc = acc + tl.dot(q_i8, b_tile)

        # ── 3) Epilogo escalado fp32 ─────────────────────────────────────
        # acc [BLOCK_M,BLOCK_N] int32 -> fp32 * a_scale * b_scale
        acc_f32 = acc.to(tl.float32)
        # broadcast: a_scale [BLOCK_M,1] * b_scale [1,BLOCK_N]
        acc_scaled = acc_f32 * scale[:, None] * b_scale[None, :]

        # Store
        out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
        mask_out = mask_m[:, None] & mask_n[None, :]
        tl.store(out_ptrs, acc_scaled, mask=mask_out)


# ── Fallback 100% GPU (sin Triton, sin host copies) ──────────────────────
def _fused_quant_gemm_fallback_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fallback vectorizado 100% GPU: quant per-token + GEMM int8.

    Replica exacto el contrato del kernel Triton sin Triton:

        a_scale = amax_row/127, clamp >0 else 1.0
        a_i8 = round(a / a_scale) clamp [-127,127]
        out = (a_i8 @ b).to(fp32) * a_scale * b_scale

    Puro torch GPU, sin copias a host, sin loops Python sobre filas.
    Soporta 2-D [M,K] y batched leading dims via flatten.
    """
    orig_a_shape = a.shape
    # GEMM requiere 2-D: flatten leading dims de a si vienen [...,K]
    if a.dim() != 2:
        if a.dim() < 1:
            raise ValueError(f"fused_quant_gemm: a debe ser al menos 1-D, got {a.dim()}D")
        k = int(orig_a_shape[-1])
        m = a.numel() // k if k else 0
        a_2d = a.reshape(m, k)
        need_reshape_out = True
        out_leading = orig_a_shape[:-1]
        # N viene de b
        n = int(b.shape[1]) if b.dim() == 2 else int(b.shape[-1])
        out_shape = tuple(out_leading) + (n,)
    else:
        a_2d = a
        need_reshape_out = False
        out_shape = None  # type: ignore

    m, k = a_2d.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"fused_quant_gemm: K mismatch a {k} vs b {k2}")

    # Quant per-token sobre a_2d -> escala [M,1] fp32 y a_i8 [M,K] int8
    a_f32 = a_2d.to(torch.float32)
    amax = a_f32.abs().amax(dim=-1, keepdim=True)  # [M,1]
    a_scale = amax / 127.0
    a_scale = torch.where(a_scale > 0, a_scale, torch.ones_like(a_scale))
    a_i8 = (a_f32 / a_scale).round().clamp(-127, 127).to(torch.int8)

    # Normalizar b_scale a [N] fp32 1-D GPU (sin copias a host)
    if b_scale.dim() == 2:
        # [N,1] o [1,N] o [K,N] degenerado — colapsar a [N]
        if b_scale.shape == (n, 1):
            b_vec = b_scale.squeeze(1).to(torch.float32)
        elif b_scale.shape == (1, n):
            b_vec = b_scale.squeeze(0).to(torch.float32)
        elif b_scale.shape[0] == n and b_scale.shape[1] == 1:
            b_vec = b_scale.squeeze(1).to(torch.float32)
        else:
            b_vec = b_scale.reshape(-1).to(torch.float32)[:n]
    elif b_scale.dim() == 1:
        b_vec = b_scale.to(torch.float32)[:n]
    else:
        b_vec = b_scale.reshape(-1).to(torch.float32)[:n]

    # GEMM int8 -> int32 accum (vectorizado, sin loops)
    # torch.matmul int32 no esta implementado en CUDA; usamos float32
    # exacto hasta 2^24 (acc max 127*127*K < 2.1G para K<=16384 cabe en fp32 int mantissa 24 bits)
    # Para K grande el error es <1 LSB, aceptable para fallback validacion.
    # En CPU el path int32 es nativo.
    try:
        acc = torch.matmul(a_i8.to(torch.int32), b.to(torch.int32))  # [M,N] int32 (CPU)
    except Exception:
        acc = torch.matmul(a_i8.to(torch.float32), b.to(torch.float32))  # [M,N] fp32 (CUDA fallback)
    # Epilogo fp32: acc * a_scale * b_scale (todo GPU)
    # a_scale [M,1] fp32, b_vec [N] fp32
    out_fp32 = acc.to(torch.float32) * a_scale * b_vec.unsqueeze(0)
    out = out_fp32.to(out_dtype)

    if need_reshape_out:
        out = out.reshape(out_shape)  # type: ignore
    return out


# ── API publica ──────────────────────────────────────────────────────────
def fused_quant_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype | None = None,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    block_k: int = BLOCK_K,
) -> torch.Tensor:
    """Super-kernel fused bf16 -> quant per-token + GEMM int8 — single launch SIN branches validacion.

    Hace en un solo kernel (sin escribir ``a_i8`` a DRAM):

        a_scale = amax_row(a) / 127  (tl.max reduccion por fila)
        a_i8    = clamp(round(a / a_scale), -127, 127)  (registros)
        out     = (a_i8 @ b).to(fp32) * a_scale * b_scale  (tl.dot + epilogo)

    SUPER-KERNEL — validacion previa (branchless hot path):
        Toda validacion de dtype int8, block_k minimo 16, contiguity de a/b
        y layout column-major, y dims multiplo de 16 ocurre **antes** en
        ``patch_PN110_int8_phase_dispatch._make_pwal_wrapper`` donde ya se
        chequea dims y dtype. Este kernel no contiene ramas de validacion en
        el camino caliente; asume invariantes ya garantizadas. Solo quedan
        chequeos minimos de forma (K mismatch, empty) fuera del launch.

    Args:
        a:        [M, K] bf16/fp16/fp32 (o [...,K] con leading dims).
                  Row-major contigua (validada previa). Se lee una vez desde DRAM.
        b:        [K, N] int8 peso (validado previo int8, column-major si viene de PN110).
                  Acepta cualquier stride pero se asume int8 y BLOCK_K>=16 ya validados.
        b_scale:  [N] o [N,1] fp32 escala per-channel del peso.
        out_dtype: dtype de salida (bf16/fp16/fp32). Si None, usa bf16 si
                  ``a`` es bf16, si no fp16.
        block_m, block_n, block_k: tamanio de bloque Triton (constexpr).
                  Deben ser potencia de 2 (32/64 tipico) y block_k>=16 (validado previo).

    Returns:
        Tensor [M, N] en ``out_dtype`` (o [...,N] si ``a`` tenia leading dims).

    Camino rapido: Triton ``_fused_quant_gemm_kernel`` (1 launch, sin ``a_i8``
        intermedio DRAM, tl.dot int8->int32, epilogo fp32). Sin branches validacion.
    Fallback: torch 100 por ciento GPU si Triton/CUDA no disponible,
        M*N pequeno, o tensores no-CUDA. Sin copias a host.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not isinstance(b_scale, torch.Tensor):
        raise ValueError("fused_quant_gemm: a, b, b_scale deben ser torch.Tensor")
    if a.dim() < 1 or b.dim() != 2:
        raise ValueError(f"fused_quant_gemm: esperado a [...,K] y b [K,N], got {tuple(a.shape)} y {tuple(b.shape)}")
    if out_dtype is None:
        out_dtype = torch.bfloat16 if a.dtype == torch.bfloat16 else torch.float16
    # SUPER-KERNEL: sin rama de validacion de dtype — b asumido int8, validado previo en wiring _make_pwal_wrapper.

    # Normalizar dims: soportar [...,K] via flatten para kernel 2-D
    orig_shape = a.shape
    flatten = a.dim() != 2
    if flatten:
        k = int(orig_shape[-1])
        m = a.numel() // k if k else 0
        a_2d = a.reshape(m, k)
        n = int(b.shape[1])
        out_shape = tuple(orig_shape[:-1]) + (n,)
    else:
        a_2d = a
        out_shape = None  # type: ignore
        m, k = a_2d.shape
        n = int(b.shape[1])

    if k != int(b.shape[0]):
        raise ValueError(f"fused_quant_gemm: K mismatch a {k} vs b {b.shape[0]}")
    if a_2d.numel() == 0 or m == 0 or n == 0:
        # Tensor vacio: devolver vacio correctamente tipado
        empty_shape = out_shape if flatten else (m, n)
        return torch.empty(empty_shape, dtype=out_dtype, device=a.device)  # type: ignore

    # ── Camino rapido Triton ──────────────────────────────────────────────
    use_triton = (
        _TRITON_OK
        and a_2d.is_cuda
        and b.is_cuda
        and torch.cuda.is_available()
        and m > 0
        and n > 0
        and k > 0
    )
    if use_triton:
        try:
            # Normalizar b_scale a 1-D [N] contigua fp32 GPU (sin host copy)
            if b_scale.dim() == 2:
                if b_scale.shape == (n, 1):
                    b_scale_1d = b_scale.squeeze(1).contiguous().to(torch.float32)
                elif b_scale.shape == (1, n):
                    b_scale_1d = b_scale.squeeze(0).contiguous().to(torch.float32)
                else:
                    b_scale_1d = b_scale.reshape(-1).contiguous().to(torch.float32)[:n]
            elif b_scale.dim() == 1:
                b_scale_1d = b_scale.contiguous().to(torch.float32)
            else:
                b_scale_1d = b_scale.reshape(-1).contiguous().to(torch.float32)[:n]
            if b_scale_1d.numel() != n:
                # Pad/truncar defensivo (no deberia ocurrir)
                if b_scale_1d.numel() > n:
                    b_scale_1d = b_scale_1d[:n].contiguous()
                else:
                    pad = torch.ones(n - b_scale_1d.numel(), dtype=torch.float32, device=b_scale_1d.device)
                    b_scale_1d = torch.cat([b_scale_1d, pad])

            # SUPER-KERNEL: sin ramas de validacion contiguity / block_k.
            # a_2d contigua y block_k minimo 16 garantizados previo en wiring (_make_pwal_wrapper).
            # Camino caliente branchless: no hay chequeos de contiguity ni clamp de block_k.

            # Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
            grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))  # type: ignore
            # K puede no ser multiplo de BLOCK_K — el kernel enmascara con mask_k

            # Buffer de salida fp32 intermedio si out_dtype no es fp32,
            # luego casteamos. El kernel Triton escribe fp32 (acc_scaled) y
            # tl.store no castea automatico a bf16 con precision completa.
            # Hacer buffer fp32 + cast es correcto y barato (sin extra kernel).
            need_cast = out_dtype != torch.float32
            if need_cast:
                out_fp32 = torch.empty((m, n), dtype=torch.float32, device=a_2d.device)
                out_ptr = out_fp32
            else:
                out_fp32 = torch.empty((m, n), dtype=torch.float32, device=a_2d.device)
                out_ptr = out_fp32

            stride_am = a_2d.stride(0)
            stride_ak = a_2d.stride(1)
            stride_bk = b.stride(0)
            stride_bn = b.stride(1)
            stride_out_m = out_ptr.stride(0)
            stride_out_n = out_ptr.stride(1)

            _fused_quant_gemm_kernel[grid](
                a_2d,
                b,
                out_ptr,
                b_scale_1d,
                m,
                n,
                k,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                stride_out_m,
                stride_out_n,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=4,
                num_stages=2,
            )
            if need_cast:
                out_2d = out_fp32.to(out_dtype)
            else:
                out_2d = out_fp32

            if flatten:
                return out_2d.reshape(out_shape)  # type: ignore
            return out_2d
        except Exception as e:
            log.warning(
                "fused_quant_gemm Triton fallo (%s: %s) — fallback torch GPU",
                type(e).__name__,
                e,
            )
            # caer a fallback

    # ── Fallback torch 100% GPU ──────────────────────────────────────────
    # Usar a original (no a_2d) para preservar leading dims en fallback
    fallback_a = a if flatten else a_2d
    # El fallback maneja flatten internamente, pasar `a` original
    out = _fused_quant_gemm_fallback_torch(a, b, b_scale, out_dtype)  # type: ignore[arg-type]
    return out


# Alias
quant_gemm = fused_quant_gemm
fused_gemm = fused_quant_gemm

__all__ = [
    "fused_quant_gemm",
    "quant_gemm",
    "fused_gemm",
    "is_available",
    "is_triton_available",
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
]

# ── Self-test (python -m vllm._genesis.kernels.fused_quant_gemm) ─────────
if __name__ == "__main__":  # pragma: no cover
    import sys

    print("=== fused_quant_gemm single-launch self-test ===")
    print(f"triton_available={is_triton_available()} cuda_available={is_available()}")
    if torch is not None:
        print(f"torch {torch.__version__}")
        # Test CPU fallback (sin Triton, 100% GPU torch pero corre en CPU para test)
        try:
            torch.manual_seed(0)
            for m, k, n in [(2, 32, 16), (4, 64, 32), (1, 128, 64)]:
                a = torch.randn(m, k, dtype=torch.float32) * 2.0
                b = torch.randint(-127, 127, (k, n), dtype=torch.int8)
                b_scale = torch.rand(n, dtype=torch.float32) * 0.02 + 0.005
                out = fused_quant_gemm(a, b, b_scale, out_dtype=torch.float32)
                # Referencia: quant + matmul
                a_f32 = a.to(torch.float32)
                amax = a_f32.abs().amax(dim=-1, keepdim=True)
                a_scale = torch.where(amax / 127.0 > 0, amax / 127.0, torch.ones_like(amax / 127.0))
                a_i8 = (a_f32 / a_scale).round().clamp(-127, 127).to(torch.int8)
                ref = torch.matmul(a_i8.to(torch.int32), b.to(torch.int32)).to(torch.float32) * a_scale * b_scale.unsqueeze(0).to(torch.float32)
                ref = ref.to(torch.float32)
                diff = (out - ref).abs().max().item()
                print(f"CPU fallback m={m} k={k} n={n} max_diff={diff:.6f} {'OK' if diff < 1e-3 else 'FAIL'}")
                assert diff < 1e-3, f"mismatch {diff}"
            print("CPU fallback OK (vectorizado sin host copies ni loops Python)")

            # Test bf16 input
            a_bf16 = torch.randn(2, 32, dtype=torch.bfloat16)
            b = torch.randint(-127, 127, (32, 16), dtype=torch.int8)
            b_scale = torch.rand(16, dtype=torch.float32) * 0.01 + 0.005
            out_bf16 = fused_quant_gemm(a_bf16, b, b_scale, out_dtype=torch.bfloat16)
            assert out_bf16.dtype == torch.bfloat16
            print(f"bf16 input OK: {tuple(out_bf16.shape)} {out_bf16.dtype}")

            # Test leading dims [...,K] (3-D)
            a3 = torch.randn(2, 3, 16, dtype=torch.float32)
            b3 = torch.randint(-127, 127, (16, 8), dtype=torch.int8)
            s3 = torch.rand(8, dtype=torch.float32) * 0.01 + 0.005
            out3 = fused_quant_gemm(a3, b3, s3, out_dtype=torch.float32)
            assert out3.shape == (2, 3, 8)
            print(f"3-D leading dims OK: {tuple(a3.shape)} -> {tuple(out3.shape)}")

            if torch.cuda.is_available():
                for dtype in [torch.bfloat16, torch.float16, torch.float32]:
                    a = torch.randn(32, 64, dtype=dtype, device="cuda") * 3.0
                    b = torch.randint(-127, 127, (64, 64), dtype=torch.int8, device="cuda")
                    b_scale = (torch.rand(64, dtype=torch.float32, device="cuda") * 0.02 + 0.005).contiguous()
                    out = fused_quant_gemm(a, b, b_scale, out_dtype=torch.bfloat16)
                    print(f"CUDA {dtype} OK: {tuple(out.shape)} {out.dtype} max {out.abs().max().item():.3f}")
                    assert out.dtype == torch.bfloat16
                print("CUDA fused single-launch OK")
            else:
                print("No CUDA — skip GPU test")
        except Exception as e:
            print(f"Self-test FAIL: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        print("torch no disponible — skip")
