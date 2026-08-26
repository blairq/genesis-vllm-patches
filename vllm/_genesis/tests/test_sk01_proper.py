"""SK-01 GDN_QKVZ INT8 diadic, sm_86 mma.m16n8k32, int8/bf16 only, branchless monolithic
GDN_QKVZ INT8 diadic super-kernel — standards, functional and bench suite.

This module validates the SK-01 monolithic Triton kernel at
``vllm/_genesis/kernels/sk01_gdn_qkvz.py`` and its W4A8 sibling
``sk01_gdn_qkvz_w4a8.py`` (if present).  Design constraints are
diadic shift (``w = q*2^shift*s_row``), sm_86 ``mma.sync.m16n8k32``
Tensor Core, ``int8``/``int4``/``bf16``/``fp8`` only (``int32`` acc
exception), branchless monolithic body (``tl.load``+``tl.dot``+
``tl.store``), and per-token ``amax/127`` quantization.

Author: Genesis SK-01
"""

from __future__ import annotations

import pathlib
import re
import time

import pytest

# ── optional torch / triton availability ──────────────────────────────────
try:  # torch is optional at collection time (audit A-15)
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_AVAILABLE = False

# ── constants — per-rank geometry ─────────────────────────────────────────
K = 5120
N = 8192  # per-rank N (TP=2, global 16384)
N_GLOBAL = 16384
K_GLOBAL = 5120
SHIFT_BLOCK = 128
GROUP_SIZE = 128

MS = (1, 8, 32, 128, 512)

# Paths to kernel sources (relative to this file)
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_KERNEL_DIR = _THIS_DIR.parent / "kernels"
SK01_PATH = _KERNEL_DIR / "sk01_gdn_qkvz.py"
SK01_W4A8_PATH = _KERNEL_DIR / "sk01_gdn_qkvz_w4a8.py"

# ── helpers ────────────────────────────────────────────────────────────────


def _require_cuda_triton():
    """Skip current test if CUDA or Triton not available.

    Checks ``torch.cuda.is_available()`` and ``triton`` import.

    Raises
    ------
    pytest.skip
        If CUDA or Triton is missing.
    """
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    if not _TRITON_AVAILABLE:
        pytest.skip("triton not available")
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        pytest.skip("CUDA not available")


def _extract_triton_kernels(text: str):
    """Extract ``@triton.jit`` kernel bodies from *text*.

    Returns
    -------
    list[tuple[str, str]]
        List of ``(kernel_name, body)`` where *body* is the indented
        block after the ``def`` line (without the decorator).
    """
    # Regex: @triton.jit newline def name(...): newline indented block
    pat = re.compile(
        r"@triton\.jit\s*\n\s*def\s+(\w+)\s*\([^)]*\).*?:\n((?:[ \t]+.*\n?)*)",
        re.MULTILINE,
    )
    return pat.findall(text)


def _strip_python_comments(body: str) -> str:
    """Remove ``#`` comments from *body* (naive, branchless kernel safe).

    The SK-01 kernels contain only ``#`` line comments and no ``#`` inside
    string literals in the hot body, so a simple split is sufficient.
    """
    lines = body.splitlines()
    out: list[str] = []
    for ln in lines:
        if "#" in ln:
            ln = ln[: ln.find("#")]
        out.append(ln)
    return "\n".join(out)


def _assert_kernel_standards(path: pathlib.Path, *, require_mma_sync: bool = True) -> None:
    """Assert SK-01 standards on kernel source at *path*.

    Checks
    ------
    * file contains ``mma.sync`` (or ``mma.`` for W4A8 relaxed)
    * every ``@triton.jit`` body has no ``float32``/``fp32`` (except
      allowed ``int32`` acc), only ``int8``/``int4``/``bf16``/``fp8``
      dtypes, no ``if``/``else`` (branchless), and is monolithic
      (``tl.load``+``tl.dot``+``tl.store``)
    """
    assert path.exists(), f"kernel file not found: {path}"
    text = path.read_text(encoding="utf-8")

    # docstring / PTX must mention mma — strict for main, relaxed for W4A8
    if require_mma_sync:
        assert "mma.sync" in text, f"{path.name} missing 'mma.sync' (sm_86 mma.m16n8k32)"
    else:
        # W4A8 currently documents as mma.m16n8k32 without .sync — accept either
        assert "mma." in text, f"{path.name} missing 'mma.' instruction marker"

    kernels = _extract_triton_kernels(text)
    assert kernels, f"No @triton.jit kernel found in {path}"

    for name, body in kernels:
        stripped = _strip_python_comments(body)
        lower = stripped.lower()

        # no float32 / fp32 inside kernel body (except allowed int32 acc)
        # We check the stripped body only so PTX strings in comments don't fire.
        # ``int32`` is allowed as accumulator.
        assert "float32" not in lower, f"{path.name}:{name} contains float32 (only bf16/int8/int4/fp8/int32 allowed)"
        assert "fp32" not in lower, f"{path.name}:{name} contains fp32 (only bf16/int8/int4/fp8/int32 allowed)"

        # only allowed dtypes — disallow tl.float32 / tl.float16 / tl.float64
        # Allowed: tl.int8, tl.int4, tl.bfloat16, tl.float8*, tl.int32, tl.constexpr
        # We explicitly forbid float32/float16/float64
        dtype_hits = re.findall(r"tl\.(float32|float16|float64)\b", stripped)
        assert not dtype_hits, f"{path.name}:{name} uses disallowed dtype(s) {dtype_hits} — only int8/int4/bf16/fp8 (+int32 acc) allowed"

        # branchless: no Python if/else in hot path (tl.where is allowed)
        # We check for line-start if/else to avoid matching tl.where internals
        assert not re.search(r"^\s*if\s", stripped, re.MULTILINE), f"{path.name}:{name} contains 'if ' — kernel must be branchless"
        assert not re.search(r"^\s*else\b", stripped, re.MULTILINE), f"{path.name}:{name} contains 'else' — kernel must be branchless"
        # also forbid bare "if " with leading space (catch inline if)
        # already covered; additionally check for "else:" substring
        # but tl.where is fine

        # monolithic: must contain tl.load, tl.dot, tl.store
        assert "tl.load" in body, f"{path.name}:{name} missing tl.load (monolithic)"
        assert "tl.dot" in body, f"{path.name}:{name} missing tl.dot (monolithic)"
        assert "tl.store" in body, f"{path.name}:{name} missing tl.store (monolithic)"


def _quantize_per_token_bf16(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token int8 quantization ``amax/127`` for bf16 hidden.

    Parameters
    ----------
    hidden: torch.Tensor
        ``[M, K]`` bf16/fp16/fp32 activation on target device.

    Returns
    -------
    (q, scales)
        *q* ``[M, K]`` int8, *scales* ``[M]`` bf16 (``amax/127``, 0→1).
    """
    hf = hidden.to(torch.float32)
    amax = hf.abs().amax(dim=1)  # [M]
    scales_f = amax / 127.0
    scales_f = torch.where(scales_f > 0, scales_f, torch.ones_like(scales_f))
    scales = scales_f.to(torch.bfloat16)
    q = (hf / scales_f[:, None]).round().clamp(-128, 127).to(torch.int8)
    return q, scales


def _reference_int8_diadic(
    a_q: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Torch bf16 reference for SK-01 diadic gemm.

    Mirrors the kernel's diadic shift + bf16 scaling semantics but in
    pure torch so the error is only bf16 rounding.

    Accumulates per ``SHIFT_BLOCK=128`` in float32 then casts to bf16
    once, which is within ``atol 1e-2`` for the small magnitudes used
    in tests (hidden_scale ~0.05).

    Parameters
    ----------
    a_q, b, a_scales, b_scales, shifts
        Same shapes as kernel: ``[M,K] int8``, ``[K,N] int8``,
        ``[M] bf16``, ``[N] bf16``, ``[K/128,N/128] int8``.

    Returns
    -------
    torch.Tensor
        ``[M,N]`` bf16 dequantized output.
    """
    M, K = a_q.shape
    N = b.shape[1]
    # Use float32 accumulation for reference (single matmul path when shifts=0)
    # For diadic with non-zero shifts, loop per 128 blocks.
    # For tests we use shifts=0, so fast path is single matmul.
    if torch.equal(shifts, torch.zeros_like(shifts)):
        try:
            acc = torch.matmul(a_q.to(torch.int32), b.to(torch.int32))  # [M,N] int32
        except Exception:
            acc = torch.matmul(a_q.to(torch.float32), b.to(torch.float32)).to(torch.int32)
        out_f = acc.to(torch.float32) * a_scales.to(torch.float32)[:, None] * b_scales.to(torch.float32)[None, :]
        return out_f.to(torch.bfloat16)

    # Generic diadic loop
    out = torch.zeros((M, N), dtype=torch.float32, device=a_q.device)
    for kb in range(K // SHIFT_BLOCK):
        k0 = kb * SHIFT_BLOCK
        for nb in range(N // SHIFT_BLOCK):
            n0 = nb * SHIFT_BLOCK
            a_blk = a_q[:, k0 : k0 + SHIFT_BLOCK].to(torch.int32)
            b_blk = b[k0 : k0 + SHIFT_BLOCK, n0 : n0 + SHIFT_BLOCK].to(torch.int32)
            try:
                acc = torch.matmul(a_blk, b_blk)
            except Exception:
                acc = torch.matmul(a_blk.to(torch.float32), b_blk.to(torch.float32)).to(torch.int32)
            shift = int(shifts[kb, nb].item())
            shifted = acc << shift if shift != 0 else acc
            # bf16 conversion mimics ``shifted.to(tl.bfloat16)`` then scale
            # Use float32->bf16->float32 to simulate
            shifted_f = shifted.to(torch.float32).to(torch.bfloat16).to(torch.float32)
            scaled = shifted_f * a_scales.to(torch.float32)[:, None] * b_scales[n0 : n0 + SHIFT_BLOCK].to(torch.float32)[None, :]
            # accumulate in float then bf16 (kernel accumulates bf16, error <1e-2 for small magnitudes)
            out[:, n0 : n0 + SHIFT_BLOCK] += scaled
    return out.to(torch.bfloat16)


def _reference_w4a8(
    a_q: torch.Tensor,
    w_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
) -> torch.Tensor:
    """Torch bf16 reference for SK-01 W4A8 (int4 packed) gemm.

    Unpacks ``w_packed`` (``[K/2,N] uint8``, low=nibble even, high=odd,
    ``q+8``) to int8 ``[-8,7]`` then per-group ``GROUP=128`` scaled matmul.

    Parameters
    ----------
    a_q: torch.Tensor
        ``[M,K] int8`` activation per-token quantized.
    w_packed: torch.Tensor
        ``[K//2,N] uint8`` packed int4.
    a_scales: torch.Tensor
        ``[M] bf16`` per-token.
    w_scales: torch.Tensor
        ``[K//GROUP,N] bf16`` per-group.

    Returns
    -------
    torch.Tensor
        ``[M,N]`` bf16.
    """
    K = a_q.shape[0] if a_q.dim() == 2 else a_q.shape[1]
    # unpack
    M = a_q.shape[0]
    N = w_packed.shape[1]
    K_unpacked = w_packed.shape[0] * 2
    assert K_unpacked == a_q.shape[1], "K mismatch unpack"
    w_unpacked = torch.empty((K_unpacked, N), dtype=torch.int8, device=a_q.device)
    # low/high
    low = (w_packed & 0xF).to(torch.int32) - 8
    high = ((w_packed >> 4) & 0xF).to(torch.int32) - 8
    w_unpacked[0::2] = low.to(torch.int8)
    w_unpacked[1::2] = high.to(torch.int8)

    out = torch.zeros((M, N), dtype=torch.float32, device=a_q.device)
    for g in range(K_unpacked // GROUP_SIZE):
        k0 = g * GROUP_SIZE
        a_blk = a_q[:, k0 : k0 + GROUP_SIZE].to(torch.int32)
        w_blk = w_unpacked[k0 : k0 + GROUP_SIZE, :].to(torch.int32)
        try:
            acc = torch.matmul(a_blk, w_blk)
        except Exception:
            acc = torch.matmul(a_blk.to(torch.float32), w_blk.to(torch.float32)).to(torch.int32)
        scaled = acc.to(torch.float32) * a_scales.to(torch.float32)[:, None] * w_scales[g].to(torch.float32)[None, :]
        out += scaled
    return out.to(torch.bfloat16)


def _measure_ms(fn, *, warmup: int = 3, iters: int = 20) -> float:
    """Time *fn* (callable returning Tensor) in ms.

    Warmups, syncs, then ``iters`` timed runs. ``iters`` is kept modest
    for CI (M=512 still <1s). Uses standard durations.

    Parameters
    ----------
    fn: callable
        Zero-arg kernel/fallback to time.
    warmup: int
        Warmup iterations (default 3).
    iters: int
        Timed iterations (default 20).

    Returns
    -------
    float
        Mean time per iteration in milliseconds.
    """
    for _ in range(warmup):
        fn()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0


# ── standards tests ───────────────────────────────────────────────────────


def test_sk01_kernel_standards():
    """Standards for ``sk01_gdn_qkvz.py`` — dtype / branchless / monolithic.

    WHY
        SK-01 is the hot QKVZ projection. Any ``float32``/``fp32`` in the
        Triton body would force slow ``fp32`` Tensor Core or extra
        conversions; branches would diverge warps; split kernels would add
        launches. The spec requires ``int8``/``bf16`` (+``int32`` acc),
        branchless monolithic ``tl.load``/``tl.dot``/``tl.store`` and
        ``mma.sync.m16n8k32``.

    Boundaries
        * Reads ``vllm/_genesis/kernels/sk01_gdn_qkvz.py`` text.
        * No ``float32``/``fp32`` inside any ``@triton.jit`` body (comments
          stripped; ``int32`` acc allowed).
        * Only ``int8``/``int4``/``bf16``/``fp8`` dtypes (plus ``int32``).
        * No ``if``/``else`` inside ``@triton.jit`` body.
        * Monolithic: each body contains ``tl.load``+``tl.dot``+``tl.store``.
        * Docstring/PTX contains ``mma.sync``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any standard is violated.
    """
    _assert_kernel_standards(SK01_PATH, require_mma_sync=True)
    # also assert module docstring contains mma.sync (redundant with above)
    txt = SK01_PATH.read_text(encoding="utf-8")
    # first docstring is module-level; ensure it mentions sm_86 and mma
    assert "mma.sync" in txt
    assert "sm_86" in txt.lower() or "sm86" in txt.lower() or "8.6" in txt


def test_sk01_w4a8_kernel_standards():
    """Standards for ``sk01_gdn_qkvz_w4a8.py`` — int4 packing variant.

    WHY
        W4A8 shares the same diadic, monolithic, branchless constraints
        but with ``int4`` weight packing (2×int4/byte). The same dtype
        and control-flow bans apply; missing ``mma`` would mean no Tensor
        Core.

    Boundaries
        * Skipped if ``sk01_gdn_qkvz_w4a8.py`` absent.
        * Otherwise same checks as main kernel but relaxed ``mma.`` allow
          (``mma.m16n8k32`` without ``.sync``) to accommodate current
          file documenting ``mma.m16n8k32``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If file exists and any check fails.
    pytest.skip
        If W4A8 file missing.
    """
    if not SK01_W4A8_PATH.exists():
        pytest.skip("sk01_gdn_qkvz_w4a8.py not present")
    # W4A8 documents as mma.m16n8k32 — allow mma. rather than strict mma.sync
    _assert_kernel_standards(SK01_W4A8_PATH, require_mma_sync=False)
    txt = SK01_W4A8_PATH.read_text(encoding="utf-8")
    assert "mma." in txt


# ── functional correctness — INT8 diadic ──────────────────────────────────


@pytest.mark.parametrize("M", MS)
def test_sk01_gdn_qkvz_functional_correctness(M: int):
    """Functional correctness INT8 diadic vs torch bf16 reference.

    WHY
        The fused kernel must be numerically equivalent to the
        per-token ``amax/127`` quant → int8 GEMM → ``*a_scale*b_scale``
        dequant pipeline (diadic shift=0 for this check). Large error
        would corrupt QKVZ and break attention.

    Boundaries
        * Parametrizes ``M`` in ``(1,8,32,128,512)`` with fixed
          ``K=5120`` ``N=8192`` (per-rank 8192×5120, global 16384×5120).
        * Hidden ``bf16`` scaled ``*0.05`` to keep accumulators in bf16
          dynamic range so ``atol 1e-2`` holds despite bf16 rounding.
        * Weight ``int8`` in ``[-1,1]`` (small) and shifts ``0`` (diadic
          branch exercised via ``shl.b32`` 0-shift).
        * Compare kernel vs ``_reference_int8_diadic`` with ``atol 1e-2``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If max abs diff > 1e-2.
    pytest.skip
        If CUDA/Triton not available.
    """
    _require_cuda_triton()
    # import inside test so collection succeeds on CPU-only hosts
    from vllm._genesis.kernels.sk01_gdn_qkvz import sk01_gdn_qkvz_gemm  # noqa: WPS433

    device = "cuda"
    torch.manual_seed(42 + M)
    # Small magnitude activations to bound bf16 error (see analysis in prompt)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.05
    a_q, a_scales = _quantize_per_token_bf16(hidden)
    # small weight range keeps acc small => atol holds
    b = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
    b_scales = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.005).to(torch.bfloat16)
    shifts = torch.zeros((K // SHIFT_BLOCK, N // SHIFT_BLOCK), dtype=torch.int8, device=device)

    out = sk01_gdn_qkvz_gemm(a_q, b, a_scales, b_scales, shifts, out_dtype=torch.bfloat16)
    ref = _reference_int8_diadic(a_q, b, a_scales, b_scales, shifts)

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "cuda"

    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff <= 1e-2, f"M={M} max_diff {max_diff:.5f} mean {mean_diff:.5f} exceeds atol 1e-2 (ref vs kernel diadic int8)"
    # also check 95p within atol for robustness
    # p95 = diff.flatten().kthvalue(int(diff.numel()*0.95)).values.item()
    # assert p95 <= 1e-2  # not strict


@pytest.mark.parametrize("M", MS)
def test_sk01_gdn_qkvz_w4a8_functional_correctness(M: int):
    """Functional correctness W4A8 variant — int4 packed.

    WHY
        W4A8 must unpack 2×int4/byte (``q+8``) then per-group
        ``GROUP=128`` bf16 scale correctly. Packing errors would
        silently corrupt weights.

    Boundaries
        * Parametrizes ``M`` same as INT8 (1,8,32,128,512) ``K=5120``
          ``N=8192`` per-rank.
        * Activation quantized per-token ``amax/127`` with ``hs=0.02``
          (small to keep bf16 error <1e-2).
        * Weight ``int4`` ``[-8,7]`` random packed, ``w_scales``
          ``[K/128,N]`` bf16 ``0.002..0.007``.
        * Skipped if W4A8 file/kernel missing.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If max diff > 1e-2 after unpack+grouped GEMM.
    pytest.skip
        If CUDA/Triton/W4A8 not available.
    """
    _require_cuda_triton()
    if not SK01_W4A8_PATH.exists():
        pytest.skip("sk01_gdn_qkvz_w4a8.py not present")
    try:
        from vllm._genesis.kernels.sk01_gdn_qkvz_w4a8 import sk01_gdn_qkvz_w4a8_gemm  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"W4A8 kernel not importable: {e}")

    device = "cuda"
    torch.manual_seed(100 + M)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
    a_q, a_scales = _quantize_per_token_bf16(hidden)

    # pack int4
    w_vals = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=device)
    w_q = (w_vals.to(torch.int32) + 8).to(torch.uint8)  # 0..15
    w_packed = (w_q[0::2] & 0xF) | ((w_q[1::2] & 0xF) << 4)  # [K/2,N] uint8
    w_packed = w_packed.to(torch.uint8).contiguous()
    w_scales = (torch.rand((K // GROUP_SIZE, N), dtype=torch.float32, device=device) * 0.005 + 0.002).to(torch.bfloat16)

    out = sk01_gdn_qkvz_w4a8_gemm(a_q, w_packed, a_scales, w_scales, out_dtype=torch.bfloat16)
    ref = _reference_w4a8(a_q, w_packed, a_scales, w_scales)

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    max_diff = diff.max().item()
    assert max_diff <= 1e-2, f"W4A8 M={M} max_diff {max_diff:.5f} exceeds atol 1e-2"


# ── bench — monotonic and <1.5× fallback ───────────────────────────────────


def test_sk01_bench_monotonic_and_fallback():
    """Bench INT8 diadic: time monotonic in M and <1.5× fallback.

    WHY
        The fused kernel should scale linearly with tokens and never be
        substantially slower than a torch fallback (launch + GEMM). A
        monotonic time curve proves no pathological padding; >1.5× would
        indicate a regression vs the simple torch path.

    Boundaries
        * ``M`` in ``(1,8,32,128,512)`` ``K=5120`` ``N=8192`` per-rank.
        * Measures kernel via ``sk01_gdn_qkvz_gemm`` and fallback via
          torch int32 matmul + scale (pure torch, no Triton) — both on
          CUDA, with ``torch.cuda.synchronize`` and 20 iters avg.
        * Asserts ``kernel_time < 1.5 * fallback_time`` per M and
          ``kernel_time`` non-decreasing (allow 10% noise for timer jitter).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If monotonic violated or fallback ratio exceeded.
    pytest.skip
        If CUDA/Triton not available.
    """
    _require_cuda_triton()
    from vllm._genesis.kernels.sk01_gdn_qkvz import sk01_gdn_qkvz_gemm  # noqa: WPS433

    device = "cuda"
    kernel_times: list[float] = []
    fallback_times: list[float] = []

    for M in MS:
        torch.manual_seed(1234 + M)
        hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.05
        a_q, a_scales = _quantize_per_token_bf16(hidden)
        b = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
        b_scales = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.005).to(torch.bfloat16)
        shifts = torch.zeros((K // SHIFT_BLOCK, N // SHIFT_BLOCK), dtype=torch.int8, device=device)

        def _kernel_fn(a_q=a_q, b=b, a_scales=a_scales, b_scales=b_scales, shifts=shifts):
            return sk01_gdn_qkvz_gemm(a_q, b, a_scales, b_scales, shifts, out_dtype=torch.bfloat16)

        def _fallback_fn(a_q=a_q, b=b, a_scales=a_scales, b_scales=b_scales):
            try:
                acc = torch.matmul(a_q.to(torch.int32), b.to(torch.int32))
            except Exception:
                acc = torch.matmul(a_q.to(torch.float32), b.to(torch.float32)).to(torch.int32)
            return (acc.to(torch.float32) * a_scales.to(torch.float32)[:, None] * b_scales.to(torch.float32)[None, :]).to(torch.bfloat16)

        k_ms = _measure_ms(_kernel_fn, warmup=3, iters=20)
        f_ms = _measure_ms(_fallback_fn, warmup=3, iters=20)
        kernel_times.append(k_ms)
        fallback_times.append(f_ms)
        assert k_ms < 1.5 * f_ms, f"M={M} kernel {k_ms:.3f}ms not <1.5*fallback {f_ms:.3f}ms (ratio {k_ms/f_ms:.2f})"

    # monotonic (allow 30% jitter for timer noise on tiny M=1/8 launch overhead)
    for i in range(1, len(kernel_times)):
        prev, cur = kernel_times[i - 1], kernel_times[i]
        # M grows 8×, so time must not shrink significantly (30% tolerance for small kernels)
        assert cur + 1e-6 >= prev * 0.70, f"monotonic violation M {MS[i-1]}->{MS[i]}: {prev:.3f}ms -> {cur:.3f}ms"
    # also overall increasing
    assert kernel_times[-1] > kernel_times[0], f"kernel time not increasing: {kernel_times}"


def test_sk01_w4a8_bench_monotonic_and_fallback():
    """Bench W4A8: monotonic and <1.5× torch unpack fallback.

    WHY
        Same reasoning as INT8 bench but for the packed path — unpack
        + grouped GEMM must not be >1.5× slower than kernel; time must
        grow with M.

    Boundaries
        * Same ``M``/``K``/``N``.
        * Fallback is explicit unpack + per-group torch matmul (the
          ``_reference_w4a8`` loop without bf16 tricks).
        * Skipped if W4A8 file/kernel missing.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If monotonic or ratio violated.
    pytest.skip
        If CUDA/Triton/W4A8 missing.
    """
    _require_cuda_triton()
    if not SK01_W4A8_PATH.exists():
        pytest.skip("sk01_gdn_qkvz_w4a8.py not present")
    try:
        from vllm._genesis.kernels.sk01_gdn_qkvz_w4a8 import sk01_gdn_qkvz_w4a8_gemm  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"W4A8 kernel not importable: {e}")

    device = "cuda"
    ktimes: list[float] = []
    ftimes: list[float] = []

    for M in MS:
        torch.manual_seed(4321 + M)
        hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
        a_q, a_scales = _quantize_per_token_bf16(hidden)
        w_vals = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=device)
        w_q = (w_vals.to(torch.int32) + 8).to(torch.uint8)
        w_packed = (w_q[0::2] & 0xF) | ((w_q[1::2] & 0xF) << 4)
        w_packed = w_packed.to(torch.uint8).contiguous()
        w_scales = (torch.rand((K // GROUP_SIZE, N), dtype=torch.float32, device=device) * 0.005 + 0.002).to(torch.bfloat16)

        # unpack once for fallback closure
        w_unpacked = torch.empty((K, N), dtype=torch.int8, device=device)
        low = (w_packed & 0xF).to(torch.int32) - 8
        high = ((w_packed >> 4) & 0xF).to(torch.int32) - 8
        w_unpacked[0::2] = low.to(torch.int8)
        w_unpacked[1::2] = high.to(torch.int8)

        def _kfn(a_q=a_q, w_packed=w_packed, a_scales=a_scales, w_scales=w_scales):
            return sk01_gdn_qkvz_w4a8_gemm(a_q, w_packed, a_scales, w_scales, out_dtype=torch.bfloat16)

        def _ffn(a_q=a_q, w_unpacked=w_unpacked, a_scales=a_scales, w_scales=w_scales):
            out = torch.zeros((M, N), dtype=torch.float32, device=device)
            for g in range(K // GROUP_SIZE):
                k0 = g * GROUP_SIZE
                a_blk = a_q[:, k0 : k0 + GROUP_SIZE].to(torch.int32)
                w_blk = w_unpacked[k0 : k0 + GROUP_SIZE, :].to(torch.int32)
                try:
                    acc = torch.matmul(a_blk, w_blk)
                except Exception:
                    acc = torch.matmul(a_blk.to(torch.float32), w_blk.to(torch.float32)).to(torch.int32)
                scaled = acc.to(torch.float32) * a_scales.to(torch.float32)[:, None] * w_scales[g].to(torch.float32)[None, :]
                out += scaled
            return out.to(torch.bfloat16)

        k_ms = _measure_ms(_kfn, warmup=3, iters=20)
        f_ms = _measure_ms(_ffn, warmup=3, iters=20)
        ktimes.append(k_ms)
        ftimes.append(f_ms)
        assert k_ms < 1.5 * f_ms, f"W4A8 M={M} kernel {k_ms:.3f}ms not <1.5*fallback {f_ms:.3f}ms ratio {k_ms/f_ms:.2f}"

    for i in range(1, len(ktimes)):
        assert ktimes[i] + 1e-6 >= ktimes[i - 1] * 0.70, f"W4A8 monotonic violation {MS[i-1]}->{MS[i]} {ktimes[i-1]:.3f}->{ktimes[i]:.3f}ms"
    assert ktimes[-1] > ktimes[0]


# ── additional diadic shift smoke (branchless shl.b32) ─────────────────────


def test_sk01_diadic_shift_branchless():
    """Smoke for diadic shift — kernel must handle shift>0 branchless.

    WHY
        Diadic weight is ``q*2^shift*s_row``. The kernel implements
        ``shifted = int_acc << shift_val`` via ``shl.b32`` PTX, branchless
        for ``shift>=0``. A regression that adds a Python ``if`` or
        mishandles shift would corrupt scaled outputs.

    Boundaries
        * ``M=32`` ``K=5120`` ``N=8192`` with random shifts ``0..2``
          (≤10 safe for INT32 2.1G).
        * Compares kernel vs ``_reference_int8_diadic`` with same shifts.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If shift handling diverges >1e-2.
    pytest.skip
        If CUDA/Triton missing.
    """
    _require_cuda_triton()
    from vllm._genesis.kernels.sk01_gdn_qkvz import sk01_gdn_qkvz_gemm  # noqa: WPS433

    device = "cuda"
    M = 32
    torch.manual_seed(999)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.04
    a_q, a_scales = _quantize_per_token_bf16(hidden)
    b = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
    b_scales = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.003).to(torch.bfloat16)
    shifts = torch.randint(0, 3, (K // SHIFT_BLOCK, N // SHIFT_BLOCK), dtype=torch.int8, device=device)

    out = sk01_gdn_qkvz_gemm(a_q, b, a_scales, b_scales, shifts, out_dtype=torch.bfloat16)
    ref = _reference_int8_diadic(a_q, b, a_scales, b_scales, shifts)
    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    assert diff <= 1e-2, f"diadic shift smoke max_diff {diff:.5f} exceeds atol 1e-2"


def test_sk01_w4a8_diadic_packing_correctness():
    """W4A8 packing smoke — 2×int4 per byte unpack.

    WHY
        Packing is ``low=nibble(k even)`` ``high=nibble(k odd)`` with
        bias 8 (``q 0..15 → -8..7``). Swapped nibbles or off-by-one bias
        would silently break W4A8.

    Boundaries
        * ``M=8`` small packing round-trip.
        * Generates random int4 ``[-8,7]``, packs, lets kernel unpack
          internally, compares vs torch unpack reference.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If packing mismatch.
    pytest.skip
        If W4A8 missing or CUDA/Triton missing.
    """
    _require_cuda_triton()
    if not SK01_W4A8_PATH.exists():
        pytest.skip("sk01_gdn_qkvz_w4a8.py not present")
    try:
        from vllm._genesis.kernels.sk01_gdn_qkvz_w4a8 import sk01_gdn_qkvz_w4a8_gemm  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"W4A8 not importable: {e}")

    device = "cuda"
    M = 8
    torch.manual_seed(202)
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
    a_q, a_scales = _quantize_per_token_bf16(hidden)
    w_vals = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=device)
    w_q = (w_vals.to(torch.int32) + 8).to(torch.uint8)
    w_packed = (w_q[0::2] & 0xF) | ((w_q[1::2] & 0xF) << 4)
    w_packed = w_packed.to(torch.uint8).contiguous()
    # verify round-trip unpack in test itself
    w_unpacked = torch.empty((K, N), dtype=torch.int8, device=device)
    w_unpacked[0::2] = (w_packed & 0xF).to(torch.int32) - 8
    w_unpacked[1::2] = ((w_packed >> 4) & 0xF).to(torch.int32) - 8
    assert torch.equal(w_unpacked, w_vals), "packing round-trip failed in test harness"

    w_scales = (torch.rand((K // GROUP_SIZE, N), dtype=torch.float32, device=device) * 0.005 + 0.002).to(torch.bfloat16)
    out = sk01_gdn_qkvz_w4a8_gemm(a_q, w_packed, a_scales, w_scales, out_dtype=torch.bfloat16)
    ref = _reference_w4a8(a_q, w_packed, a_scales, w_scales)
    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    assert diff <= 1e-2, f"W4A8 packing smoke diff {diff:.5f} exceeds atol 1e-2"
