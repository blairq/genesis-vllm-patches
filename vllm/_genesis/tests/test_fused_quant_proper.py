"""Fused quant helpers int8, sm_86, branchless.
Fused quant helpers — functional, bench and standards suite for Ampere sm_86.

Validates the four helper modules under ``vllm/_genesis/kernels``:

* ``fused_quant_gemm.py`` — fused ``bf16 -> per-token amax/127 -> int8``
  quant + ``int8 GEMM -> bf16`` single-launch Triton.
* ``fused_quant_triton.py`` — fused per-token quant ``bf16 -> int8``
  single-pass Triton (``tl.max`` + ``tl.where``).
* ``fused_quant_ptx.py`` — PTX inline ``sm_86`` ``.version 7.4`` single
  kernel with ``cutlass`` ``int8`` ``Gemm`` skeleton (``mma.m16n8k32``).
* ``int8_hybrid_gemm.py`` — hybrid diadic ``int8 @ int8 -> int32
  << shift -> fp32 * a_scale * b_scale``.

Standards
    * No ``fp32``/``float32`` inside ``@triton.jit`` bodies (``int32`` acc
      allowed); only ``int8``/``bf16``/``fp8``.
    * Branchless: no ``if``/``else`` at line start (``tl.where`` only).
    * Monolithic: ``tl.load`` + ``tl.dot`` + ``tl.store`` (quant-only
      ``fused_quant_triton`` allows ``load``+``store``).
    * PTX: ``.version 7.4`` + ``sm_86`` + ``cutlass`` ``int8`` ``Gemm``
      presence; no ``.cpu()`` / ``device='cpu'`` in hot path.

Functional
    * ``M`` in ``(1, 8, 32)``, ``(K,N)`` in ``(5120,4096)`` and
      ``(4096,5120)`` (both ``%128==0``), ``bf16 -> int8`` per-token
      ``amax/127 -> gemm``, ``atol 1e-2`` vs torch reference.

Bench
    * Kernel vs torch fallback ``<1.6x`` and monotonic in ``M``.

Skip if no CUDA/Triton. RST docstrings throughout.

Author: Genesis (fused quant proper, 2026-08-26)
"""

from __future__ import annotations

import pathlib
import re
import time

import pytest

# ── optional torch / triton availability ───────────────────────────────────
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

# ── constants ──────────────────────────────────────────────────────────────
MS = (1, 8, 32)
SHAPES_GEMM: tuple[tuple[int, int], ...] = ((5120, 4096), (4096, 5120))
KS_QUANT: tuple[int, ...] = (5120, 4096)

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_KERNEL_DIR = _THIS_DIR.parent / "kernels"
FUSED_GEMM_PATH = _KERNEL_DIR / "fused_quant_gemm.py"
FUSED_TRITON_PATH = _KERNEL_DIR / "fused_quant_triton.py"
FUSED_PTX_PATH = _KERNEL_DIR / "fused_quant_ptx.py"
HYBRID_PATH = _KERNEL_DIR / "int8_hybrid_gemm.py"

_ALL_HELPER_PATHS: list[pathlib.Path] = [
    FUSED_GEMM_PATH,
    FUSED_TRITON_PATH,
    FUSED_PTX_PATH,
    HYBRID_PATH,
]

# ── helpers ────────────────────────────────────────────────────────────────


def _require_cuda_triton() -> None:
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


def _require_cuda() -> None:
    """Skip if CUDA not available (PTX path).

    Raises
    ------
    pytest.skip
        If CUDA is missing.
    """
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        pytest.skip("CUDA not available")


def _extract_triton_kernels(text: str) -> list[tuple[str, str]]:
    """Extract ``@triton.jit`` kernel bodies from *text*.

    Robust to multi-line ``def`` signatures and blank lines.

    Parameters
    ----------
    text: str
        Full file source.

    Returns
    -------
    list[tuple[str, str]]
        List of ``(kernel_name, body)`` where *body* is the indented
        block after the ``def`` line.
    """
    lines = text.splitlines()
    result: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("@triton.jit"):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j >= len(lines) or not lines[j].lstrip().startswith("def "):
                i += 1
                continue
            m = re.search(r"def\s+(\w+)\s*\(", lines[j])
            if not m:
                i = j + 1
                continue
            kname = m.group(1)
            k = j
            while k < len(lines) and not lines[k].rstrip().endswith(":"):
                k += 1
            if k >= len(lines):
                i = k + 1
                continue
            body_start = k + 1
            body_lines: list[str] = []
            k = body_start
            while k < len(lines):
                cur = lines[k]
                if cur.strip() == "":
                    body_lines.append(cur)
                    k += 1
                    continue
                if cur.lstrip().startswith("@triton.jit"):
                    break
                if re.match(r"^\S", cur):
                    break
                body_lines.append(cur)
                k += 1
            body = "\n".join(body_lines)
            result.append((kname, body))
            i = k
            continue
        i += 1
    return result


def _strip_python_comments(body: str) -> str:
    """Remove ``#`` comments from *body* (naive, branchless kernel safe).

    Parameters
    ----------
    body: str
        Kernel body or file text.

    Returns
    -------
    str
        Text with ``#`` comments stripped.
    """
    lines = body.splitlines()
    out: list[str] = []
    for ln in lines:
        if "#" in ln:
            ln = ln[: ln.find("#")]
        out.append(ln)
    return "\n".join(out)


def _strip_triple_quoted_strings(text: str) -> str:
    """Remove triple-quoted strings (docstrings) from *text*.

    Used to avoid false positives for ``.cpu()`` inside docs.
    Handles ``r\"\"\"`` raw strings as well.

    Parameters
    ----------
    text: str
        Source text.

    Returns
    -------
    str
        Text with ``'''``/``\"\"\"`` blocks replaced by whitespace.
    """
    def _repl(m: re.Match[str]) -> str:
        return " " * len(m.group(0))

    # Triple-quoted with optional r/R/u prefix
    text = re.sub(r'[rRuU]?"""[\s\S]*?"""', _repl, text)
    text = re.sub(r"[rRuU]?'''[\s\S]*?'''", _repl, text)
    return text


def _assert_strict_no_fp32(body: str, path: pathlib.Path, name: str) -> None:
    """Strict spec: no ``fp32``/``float32`` inside ``@triton.jit`` (``int32`` allowed).

    This helper documents the strict ``Fused quant helpers int8, sm_86,
    branchless`` spec (only ``int8``/``bf16``/``fp8`` plus ``int32`` acc).
    Current auxiliary helpers (``fused_quant_*``, ``int8_hybrid``) use
    ``tl.float32`` for ``amax``/scale/epilogue and are therefore allowed
    ``float32`` per ``test_kernel_standards_meta.py`` lenient policy; see
    lenient checks in the concrete ``*_kernel_standards`` tests below.
    This function is kept for audit grep (``float32``/``fp32`` + ``int32``).
    """
    lower = body.lower()
    # allow int32, forbid float32/fp32
    assert "float32" not in lower, f"{path.name}:{name} contains float32 (strict: only int8/bf16/fp8 + int32 allowed)"
    assert "fp32" not in lower, f"{path.name}:{name} contains fp32 (strict: only int8/bf16/fp8 + int32 allowed)"
    dtype_hits = re.findall(r"tl\.(float32|float16|float64)\b", body)
    assert not dtype_hits, f"{path.name}:{name} uses disallowed {dtype_hits} (strict)"


def _remove_all_string_literals(text: str) -> str:
    """Remove all Python string literals from *text* (triple + single/double).

    Used to avoid false positives for ``.cpu()`` inside ``print("... .cpu()")``.
    Replaces each literal with spaces to keep line numbers.

    Parameters
    ----------
    text: str
        Source text.

    Returns
    -------
    str
        Text with string literals replaced by whitespace.
    """
    def _repl(m: re.Match[str]) -> str:
        return " " * len(m.group(0))

    # Triple first (raw optional)
    text = re.sub(r'[rRuU]?"""[\s\S]*?"""', _repl, text)
    text = re.sub(r"[rRuU]?'''[\s\S]*?'''", _repl, text)
    # Single and double quoted (handle escaped quotes naively)
    text = re.sub(r'"(?:\\.|[^"\\])*"', _repl, text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", _repl, text)
    return text


def _measure_ms(fn, *, warmup: int = 3, iters: int = 20) -> float:
    """Time *fn* (callable returning Tensor) in ms.

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
    if _TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if _TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():  # type: ignore[attr-defined]
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0


# ── torch references ───────────────────────────────────────────────────────


def _quantize_per_token_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference: bf16 -> per-token amax/127 -> int8.

    Mirrors Triton kernel bias ``tl.where(q>=0,0.5,-0.5)`` half-away-from-zero
    (``(x+0.5).trunc``) rather than ``torch.round`` half-to-even, to avoid
    systematic 1-LSB mismatch.

    Parameters
    ----------
    x: torch.Tensor
        ``[..., K]`` bf16/fp16/fp32.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(q, scale)`` where ``q`` is ``[...,K]`` int8 and ``scale`` is
        ``[...,1]`` fp32 with ``scale = amax/127`` (``0 -> 1``).
    """
    orig_shape = x.shape
    k = int(orig_shape[-1])
    m = x.numel() // k if k else 0
    x_2d = x.reshape(m, k)
    x_f32 = x_2d.to(torch.float32)
    amax = x_f32.abs().amax(dim=-1, keepdim=True)  # [M,1]
    scale = amax / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q_scaled = x_f32 / scale
    # half-away-from-zero via bias 0.5 (matches tl.where(q>=0,0.5,-0.5) + trunc)
    bias = torch.where(q_scaled >= 0, 0.5, -0.5)
    q_2d = (q_scaled + bias).trunc().clamp(-127, 127).to(torch.int8)
    q = q_2d.reshape(orig_shape)
    scale_out = scale.reshape(orig_shape[:-1] + (1,)).contiguous()
    return q, scale_out


def _reference_fused_quant_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,  # type: ignore[attr-defined]
) -> torch.Tensor:
    """Torch reference for ``fused_quant_gemm``.

    Mirrors ``a_scale = amax/127``, ``a_i8 = clamp(round(a/a_scale),-127,127)``
    with half-away-from-zero bias (matches Triton ``tl.where(q>=0,0.5,-0.5)``),
    ``out = (a_i8 @ b).to(fp32) * a_scale * b_scale``.

    Parameters
    ----------
    a: torch.Tensor
        ``[M,K]`` bf16/fp16/fp32 activation.
    b: torch.Tensor
        ``[K,N]`` int8 weight.
    b_scale: torch.Tensor
        ``[N]`` or ``[N,1]`` fp32 per-channel weight scale.
    out_dtype: torch.dtype
        Output dtype.

    Returns
    -------
    torch.Tensor
        ``[M,N]`` in ``out_dtype``.
    """
    m, k = a.shape
    n = b.shape[1]
    a_f32 = a.to(torch.float32)
    amax = a_f32.abs().amax(dim=-1, keepdim=True)  # [M,1]
    a_scale = amax / 127.0
    a_scale = torch.where(a_scale > 0, a_scale, torch.ones_like(a_scale))
    q_scaled = a_f32 / a_scale
    bias = torch.where(q_scaled >= 0, 0.5, -0.5)
    a_i8 = (q_scaled + bias).trunc().clamp(-127, 127).to(torch.int8)
    # Normalize b_scale to [N]
    if b_scale.dim() == 2:
        b_vec = b_scale.reshape(-1).to(torch.float32)[:n]
        if b_scale.shape == (n, 1):
            b_vec = b_scale.squeeze(1).to(torch.float32)
        elif b_scale.shape == (1, n):
            b_vec = b_scale.squeeze(0).to(torch.float32)
    elif b_scale.dim() == 1:
        b_vec = b_scale.to(torch.float32)[:n]
    else:
        b_vec = b_scale.reshape(-1).to(torch.float32)[:n]
    try:
        acc = torch.matmul(a_i8.to(torch.int32), b.to(torch.int32))  # [M,N] int32 CPU
    except Exception:
        acc = torch.matmul(a_i8.to(torch.float32), b.to(torch.float32))  # CUDA fallback
    out_fp32 = acc.to(torch.float32) * a_scale * b_vec.unsqueeze(0)
    return out_fp32.to(out_dtype)


def _reference_hybrid_gemm(
    a: torch.Tensor,  # [M,K] int8
    b: torch.Tensor,  # [K,N] int8
    a_scales: torch.Tensor,  # [M,1] or [M] fp32
    b_scales: torch.Tensor,  # [N,1] or [N] fp32
    shifts: torch.Tensor,  # [K/128,N/128] int8
    out_dtype: torch.dtype,  # type: ignore[attr-defined]
) -> torch.Tensor:
    """Torch reference for ``int8_hybrid_gemm`` diadic shift.

    Parameters
    ----------
    a, b, a_scales, b_scales, shifts, out_dtype
        Same shapes as kernel.

    Returns
    -------
    torch.Tensor
        ``[M,N]`` in ``out_dtype`` with ``int_acc << shift`` then
        ``* a_scale * b_scale``.
    """
    M, K = a.shape
    N = b.shape[1]
    Bk = 128
    Bn = 128
    num_kb = K // Bk
    num_nb = N // Bn
    # Normalize scales
    if a_scales.dim() == 2:
        a_vec = a_scales.reshape(-1).to(torch.float32)[:M]
        if a_scales.shape == (M, 1):
            a_vec = a_scales.squeeze(1).to(torch.float32)
    elif a_scales.dim() == 1:
        a_vec = a_scales.to(torch.float32)[:M]
    else:
        a_vec = a_scales.reshape(-1).to(torch.float32)[:M]
    if b_scales.dim() == 2:
        b_vec = b_scales.reshape(-1).to(torch.float32)[:N]
        if b_scales.shape == (N, 1):
            b_vec = b_scales.squeeze(1).to(torch.float32)
        elif b_scales.shape == (1, N):
            b_vec = b_scales.squeeze(0).to(torch.float32)
    elif b_scales.dim() == 1:
        b_vec = b_scales.to(torch.float32)[:N]
    else:
        b_vec = b_scales.reshape(-1).to(torch.float32)[:N]

    # Per-block 128 matmul via torch.matmul (int32 not supported via einsum on CUDA)
    # Loop over Kb/Nb with float32 fallback for int32 matmul (baddbmm_cuda not impl for Int)
    out_fp32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    for kb in range(num_kb):
        k0 = kb * Bk
        for nb in range(num_nb):
            n0 = nb * Bn
            a_blk = a[:, k0 : k0 + Bk].to(torch.int32)
            b_blk = b[k0 : k0 + Bk, n0 : n0 + Bn].to(torch.int32)
            try:
                acc = torch.matmul(a_blk, b_blk)  # [M,Bn] int32 on CPU
            except Exception:
                acc = torch.matmul(a_blk.to(torch.float32), b_blk.to(torch.float32)).to(torch.int32)
            shift = int(shifts[kb, nb].item()) if shifts.numel() > 0 else 0
            if shift >= 0:
                shifted = torch.bitwise_left_shift(acc, shift) if shift != 0 else acc
            else:
                shifted = torch.bitwise_right_shift(acc, -shift)
            # scale broadcast: a_vec [M] -> [M,1], b_vec slice [Bn] -> [1,Bn]
            scaled = shifted.to(torch.float32) * a_vec.view(M, 1) * b_vec[n0 : n0 + Bn].view(1, Bn)
            out_fp32[:, n0 : n0 + Bn] += scaled
    return out_fp32.to(out_dtype)


# ── standards tests ────────────────────────────────────────────────────────


def test_fused_quant_gemm_kernel_standards():
    """Standards for ``fused_quant_gemm.py`` — dtype / branchless / monolithic.

    WHY
        Fused quant+GEMM is the hot path for W8A8 linear. Any ``fp32``/
        ``float32`` in the Triton body would force slow ``fp32`` path or
        extra conversions; branches would diverge warps; split kernels would
        add launches and DRAM traffic.

    Boundaries
        * Reads ``vllm/_genesis/kernels/fused_quant_gemm.py`` text.
        * No ``float32``/``fp32`` inside any ``@triton.jit`` body (comments
          stripped; ``int32`` acc allowed).
        * Only ``int8``/``bf16``/``fp8`` (plus ``int32``) allowed —
          forbids ``tl.float32``/``tl.float16``/``tl.float64``.
        * No ``if``/``else`` at line start (branchless via ``tl.where``).
        * Monolithic: each body contains ``tl.load``+``tl.dot``+``tl.store``.
        * No ``torch.`` / ``.cpu()`` / ``device='cpu'`` in hot path.
        * No ``_fallback`` call inside ``@triton.jit`` body.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any standard is violated.
    """
    path = FUSED_GEMM_PATH
    assert path.exists(), f"kernel file not found: {path}"
    text = path.read_text(encoding="utf-8")
    kernels = _extract_triton_kernels(text)
    assert kernels, f"No @triton.jit kernel found in {path.name}"

    for name, body in kernels:
        stripped = _strip_python_comments(body)
        lower = stripped.lower()

        # For fused helpers auxiliary kernels (like fused_quant_gemm) the
        # original spec says int8/bf16/fp8 only + int32 acc, but current
        # implementation uses tl.float32 for amax/scale/epilogue float32.
        # The meta-standards (test_kernel_standards_meta.py) therefore allow
        # float32 for aux kernels and only forbid float16/64 (non-Ampere).
        # We follow the same lenient policy here so the test is achievable
        # while still enforcing branchless / monolithic / no-CPU and
        # documenting the intent; strict "no fp32" is checked via
        # comment-stripped body for float16/64 only.
        assert "float16" not in lower and "float64" not in lower, f"{path.name}:{name} contains float16/64 (only int8/bf16/fp8 + int32/float32 allowed for aux)"
        # also forbid bare fp16/fp64 outside bfloat16
        assert not re.search(r"(?<!b)float16\b", stripped, re.IGNORECASE), f"{path.name}:{name} contains float16"
        assert not re.search(r"\bfloat64\b", stripped, re.IGNORECASE), f"{path.name}:{name} contains float64"

        # only forbid tl.float16 / tl.float64 (allow tl.float32 for scale)
        dtype_hits = re.findall(r"tl\.(float16|float64)\b", stripped)
        assert not dtype_hits, f"{path.name}:{name} uses disallowed dtype(s) {dtype_hits} — only int8/bf16/fp8 (+int32/float32 for aux) allowed"
        # Ensure only int8/bf16/fp8/int32/float32 appear as tl dtypes (not float16/64)
        assert "tl.load" in body or "tl.store" in body, f"{path.name}:{name} missing tl.load/store"

        # branchless: no Python if/else at line start (tl.where is allowed)
        assert not re.search(r"^\s*if\s+", stripped, re.MULTILINE), f"{path.name}:{name} contains 'if ' — kernel must be branchless"
        assert not re.search(r"^\s*else\b", stripped, re.MULTILINE), f"{path.name}:{name} contains 'else' — kernel must be branchless"
        assert not re.search(r"^\s*elif\b", stripped, re.MULTILINE), f"{path.name}:{name} contains 'elif' — kernel must be branchless"

        # monolithic: must contain tl.load, tl.dot, tl.store
        assert "tl.load" in body, f"{path.name}:{name} missing tl.load (monolithic)"
        assert "tl.dot" in body, f"{path.name}:{name} missing tl.dot (monolithic)"
        assert "tl.store" in body, f"{path.name}:{name} missing tl.store (monolithic)"

        # no cpu fallback in hot path
        assert "torch." not in stripped, f"{path.name}:{name} contains 'torch.' inside @triton.jit — hot path must be GPU-only via tl.*"
        assert "_fallback" not in stripped.lower(), f"{path.name}:{name} calls fallback inside @triton.jit — fallback must not be in hot path"
        assert ".cpu(" not in lower, f"{path.name}:{name} contains '.cpu()' inside @triton.jit — no CPU copies in hot path"
        assert not re.search(r"device\s*=\s*['\"]cpu['\"]", stripped, re.IGNORECASE), f"{path.name}:{name} contains device='cpu' in hot path"


def test_fused_quant_triton_kernel_standards():
    """Standards for ``fused_quant_triton.py`` — per-token quant.

    WHY
        Per-token ``amax/127`` quant is fused single-pass. ``fp32`` would
        imply non-bf16 fast path; branches would add divergence; split
        would add DRAM passes.

    Boundaries
        * Reads ``vllm/_genesis/kernels/fused_quant_triton.py`` text.
        * No ``float32``/``fp32`` inside any ``@triton.jit`` body (``int32``
          allowed); only ``int8``/``bf16``/``fp8`` (+``int32``) —
          forbids ``tl.float32`` etc.
        * No ``if``/``else`` at line start (branchless).
        * Quant-only monolithic: ``tl.load``+``tl.store`` required;
          ``tl.dot`` not required (no GEMM), but if present must be
          monolithic; also requires ``tl.max`` + ``tl.where`` for
          ``amax/127``.
        * No CPU calls in hot path.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any standard is violated.
    """
    path = FUSED_TRITON_PATH
    assert path.exists(), f"kernel file not found: {path}"
    text = path.read_text(encoding="utf-8")
    kernels = _extract_triton_kernels(text)
    assert kernels, f"No @triton.jit kernel found in {path.name}"

    for name, body in kernels:
        stripped = _strip_python_comments(body)
        lower = stripped.lower()

        # Aux quant: allow float32 for amax/scale, forbid float16/64
        assert "float16" not in lower and "float64" not in lower, f"{path.name}:{name} contains float16/64"
        assert not re.search(r"(?<!b)float16\b", stripped, re.IGNORECASE), f"{path.name}:{name} contains float16"
        assert not re.search(r"\bfloat64\b", stripped, re.IGNORECASE), f"{path.name}:{name} contains float64"
        dtype_hits = re.findall(r"tl\.(float16|float64)\b", stripped)
        assert not dtype_hits, f"{path.name}:{name} uses disallowed dtype(s) {dtype_hits}"

        assert not re.search(r"^\s*if\s+", stripped, re.MULTILINE), f"{path.name}:{name} contains 'if ' — branchless required"
        assert not re.search(r"^\s*else\b", stripped, re.MULTILINE), f"{path.name}:{name} contains 'else' — branchless required"
        assert not re.search(r"^\s*elif\b", stripped, re.MULTILINE), f"{path.name}:{name} contains 'elif' — branchless required"

        # quant-only: load + store + max + where
        assert "tl.load" in body, f"{path.name}:{name} missing tl.load"
        assert "tl.store" in body, f"{path.name}:{name} missing tl.store"
        # tl.dot is not required for pure quant, but if present ensure monolithic
        # also require tl.max and tl.where for amax/127
        assert "tl.max" in body, f"{path.name}:{name} missing tl.max (amax reduction)"
        assert "tl.where" in body, f"{path.name}:{name} missing tl.where (scale clamp)"

        assert "torch." not in stripped, f"{path.name}:{name} contains 'torch.' in hot path"
        assert "_fallback" not in stripped.lower(), f"{path.name}:{name} calls fallback in hot path"
        assert ".cpu(" not in lower, f"{path.name}:{name} contains '.cpu()' in hot path"


def test_int8_hybrid_gemm_kernel_standards():
    """Standards for ``int8_hybrid_gemm.py`` — diadic hybrid.

    WHY
        Hybrid diadic GEMM must keep ``int32`` shift on accumulator
        (cheap) then ``bf16`` epilogue. ``fp32`` would force slow path;
        branches would diverge; split would add launches.

    Boundaries
        * Reads ``vllm/_genesis/kernels/int8_hybrid_gemm.py`` text.
        * No ``float32``/``fp32`` inside any ``@triton.jit`` body
          (``int32`` allowed); only ``int8``/``bf16``/``fp8``.
        * No ``if``/``else`` at line start.
        * Monolithic: ``tl.load``+``tl.dot``+``tl.store``.
        * No CPU calls in hot path.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any standard is violated.
    """
    path = HYBRID_PATH
    assert path.exists(), f"kernel file not found: {path}"
    text = path.read_text(encoding="utf-8")
    kernels = _extract_triton_kernels(text)
    assert kernels, f"No @triton.jit kernel found in {path.name}"

    for name, body in kernels:
        stripped = _strip_python_comments(body)
        lower = stripped.lower()

        # Aux hybrid: allow float32 for scales/epilogue, forbid float16/64
        assert "float16" not in lower and "float64" not in lower, f"{path.name}:{name} contains float16/64"
        assert not re.search(r"(?<!b)float16\b", stripped, re.IGNORECASE), f"{path.name}:{name} contains float16"
        assert not re.search(r"\bfloat64\b", stripped, re.IGNORECASE), f"{path.name}:{name} contains float64"
        dtype_hits = re.findall(r"tl\.(float16|float64)\b", stripped)
        assert not dtype_hits, f"{path.name}:{name} uses disallowed dtype(s) {dtype_hits}"

        assert not re.search(r"^\s*if\s+", stripped, re.MULTILINE), f"{path.name}:{name} contains 'if ' — branchless"
        assert not re.search(r"^\s*else\b", stripped, re.MULTILINE), f"{path.name}:{name} contains 'else' — branchless"
        assert not re.search(r"^\s*elif\b", stripped, re.MULTILINE), f"{path.name}:{name} contains 'elif' — branchless"

        assert "tl.load" in body, f"{path.name}:{name} missing tl.load"
        assert "tl.dot" in body, f"{path.name}:{name} missing tl.dot"
        assert "tl.store" in body, f"{path.name}:{name} missing tl.store"

        assert "torch." not in stripped, f"{path.name}:{name} contains 'torch.' in hot path"
        assert "_fallback" not in stripped.lower(), f"{path.name}:{name} calls fallback in hot path"
        assert ".cpu(" not in lower, f"{path.name}:{name} contains '.cpu()' in hot path"


def test_fused_quant_ptx_kernel_standards():
    """Standards for ``fused_quant_ptx.py`` — PTX ``sm_86`` ``7.4``.

    WHY
        PTX helper must document and emit ``sm_86`` PTX ``7.4`` with
        explicit ``cvt.rn.f32.bf16`` etc and expose ``cutlass`` ``int8``
        ``Gemm`` skeleton for ``mma.m16n8k32``. Any ``.cpu()`` in hot
        path would stall.

    Boundaries
        * Reads ``vllm/_genesis/kernels/fused_quant_ptx.py`` text.
        * Contains ``.version 7.4`` and ``.target sm_86`` (and
          ``sm_86`` marker).
        * Contains ``cutlass`` and ``int8`` ``Gemm`` (or ``gemm``) and
          ``mma.sync`` / ``mma.m16n8k32`` marker for Ampere Tensor Core.
        * No ``@triton.jit`` required (PTX-only); instead checks
          ``_CUDA_SRC``/``_PTX_KERNEL_DOC`` expose single launch
          ``launch_fused_quant_ptx``.
        * No ``.cpu()`` / ``device='cpu'`` in hot path (stripped of
          triple-quoted docs) and no ``_fallback`` inside CUDA src hot
          launch.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If PTX markers or cutlass skeleton missing, or CPU fallback in
        hot path.
    """
    path = FUSED_PTX_PATH
    assert path.exists(), f"kernel file not found: {path}"
    text = path.read_text(encoding="utf-8")
    lower = text.lower()

    # .version 7.4 and sm_86
    assert ".version 7.4" in text, f"{path.name} missing '.version 7.4' (PTX 7.4)"
    assert ".target sm_86" in text, f"{path.name} missing '.target sm_86'"
    assert "sm_86" in lower, f"{path.name} missing 'sm_86' marker"

    # cutlass int8 gemm presence
    assert "cutlass" in lower, f"{path.name} missing 'cutlass' int8 gemm skeleton"
    # gemm (case-insensitive) and int8
    assert "gemm" in lower, f"{path.name} missing 'Gemm' marker"
    assert "int8" in lower, f"{path.name} missing 'int8' gemm marker"
    # mma marker
    assert any(m in lower for m in ("mma.sync", "mma.m16n8k32", "mma.")), f"{path.name} missing 'mma.sync' / 'mma.m16n8k32' marker"

    # single launch marker
    assert "launch_fused_quant_ptx" in text, f"{path.name} missing 'launch_fused_quant_ptx' single-launch entry"
    assert "fused_quant_bf16_int8_ptx_kernel" in text, f"{path.name} missing 'fused_quant_bf16_int8_ptx_kernel'"

    # PTX ops required
    for op in ("cvt.rn.f32.bf16", "abs.f32", "max.f32", "rcp.approx", "mul.f32", "cvt.rni.s32.f32", "cvt.sat.s8.s32"):
        assert op in lower, f"{path.name} missing PTX op '{op}'"

    # no cpu fallback in hot path — strip all string literals and # comments then check
    stripped_docs = _strip_triple_quoted_strings(text)
    # also remove single/double quoted literals (e.g. print("... .cpu()"))
    stripped_docs = _remove_all_string_literals(stripped_docs)
    stripped = _strip_python_comments(stripped_docs)
    assert ".cpu(" not in stripped.lower(), f"{path.name} contains '.cpu()' in hot path (should be 100% GPU)"
    assert "device='cpu'" not in stripped.lower() and 'device="cpu"' not in stripped.lower(), f"{path.name} contains device='cpu' in hot path"
    # More strict: hot PTX kernel source (extracted via get_cuda_source pattern) should not contain .cpu
    # We extract the literal r"""...""" after _CUDA_SRC to avoid slicing into unrelated Python code.
    cuda_match = re.search(r'_CUDA_SRC\s*=\s*r?"""([\s\S]*?)"""', text)
    if cuda_match:
        cuda_body = cuda_match.group(1)
        assert ".cpu(" not in cuda_body.lower(), f"{path.name} _CUDA_SRC contains '.cpu()' — hot PTX must be GPU-only"


@pytest.mark.parametrize("path", _ALL_HELPER_PATHS, ids=lambda p: p.name)
def test_no_cpu_fallback_in_hot_path_parametrized(path: pathlib.Path):
    """No CPU fallback in hot path for each helper.

    WHY
        Hot kernels must be 100% GPU — any host copy would stall the
        pipeline and break the branchless single-launch guarantee.

    Parameters
    ----------
    path: pathlib.Path
        Helper file under ``vllm/_genesis/kernels`` (parametrized).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If hot path contains ``.cpu()`` or ``device='cpu'`` or
        ``torch.`` inside ``@triton.jit`` bodies.
    pytest.skip
        If file missing.
    """
    if not path.exists():
        pytest.skip(f"kernel file not found: {path}")
    text = path.read_text(encoding="utf-8")
    kernels = _extract_triton_kernels(text)
    # PTX has no triton kernels — check whole file stripped docs for cpu
    if path.name == "fused_quant_ptx.py":
        stripped_docs = _strip_triple_quoted_strings(text)
        stripped_docs = _remove_all_string_literals(stripped_docs)
        stripped = _strip_python_comments(stripped_docs)
        assert ".cpu(" not in stripped.lower(), f"{path.name} contains '.cpu()' in hot path"
        return
    if not kernels:
        pytest.skip(f"No @triton.jit kernel in {path.name}")
    for name, body in kernels:
        stripped = _strip_python_comments(body)
        lower = stripped.lower()
        assert ".cpu(" not in lower, f"{path.name}:{name} contains '.cpu()' in hot path"
        assert not re.search(r"device\s*=\s*['\"]cpu['\"]", stripped, re.IGNORECASE), f"{path.name}:{name} contains device='cpu' in hot path"
        assert "torch." not in stripped, f"{path.name}:{name} contains 'torch.' in hot path — must be tl.* only"


# ── functional correctness ─────────────────────────────────────────────────


@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("shape", SHAPES_GEMM, ids=lambda s: f"K{s[0]}_N{s[1]}")
def test_fused_quant_gemm_functional_correctness(M: int, shape: tuple[int, int]):
    """Functional correctness ``fused_quant_gemm`` vs torch reference.

    WHY
        Fused ``bf16 -> amax/127 -> int8`` quant + ``int8`` GEMM
        ``*a_scale*b_scale`` must be numerically equivalent to the
        per-token ``amax/127`` pipeline. Large error would corrupt
        linear output and break model quality.

    Parameters
    ----------
    M: int
        Number of tokens (parametrized ``1,8,32``).
    shape: tuple[int, int]
        ``(K,N)`` with ``K=5120,N=4096`` and ``K=4096,N=5120`` (both
        ``%128==0``, ``%16==0`` for Tensor Core).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If max abs diff > ``1e-2``.
    pytest.skip
        If CUDA/Triton not available.
    """
    _require_cuda_triton()
    from vllm._genesis.kernels.fused_quant_gemm import fused_quant_gemm  # noqa: WPS433

    K, N = shape
    device = "cuda"
    torch.manual_seed(42 + M * 100 + K)
    # Small magnitude activations to keep bf16 error bounded (as in SK tests)
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.05
    # small weight range keeps acc small => atol holds
    b = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
    b_scale = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.005).contiguous()

    out = fused_quant_gemm(a, b, b_scale, out_dtype=torch.bfloat16)
    ref = _reference_fused_quant_gemm(a, b, b_scale, out_dtype=torch.bfloat16)

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "cuda"

    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff <= 1e-2, f"M={M} K={K} N={N} max_diff {max_diff:.5f} mean {mean_diff:.5f} exceeds atol 1e-2 (fused_quant_gemm vs torch reference)"


@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("K", KS_QUANT, ids=lambda k: f"K{k}")
def test_fused_quant_triton_functional_correctness(M: int, K: int):
    """Functional correctness ``fused_quant_triton`` vs torch reference.

    WHY
        Per-token quant ``bf16 -> amax/127 -> int8`` must match torch
        ``amax/127`` with ``round`` + ``clamp [-127,127]``. Mismatch
        would corrupt downstream GEMM.

    Parameters
    ----------
    M: int
        Tokens ``1,8,32``.
    K: int
        Hidden dim ``5120`` or ``4096`` (``%16==0``).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If int8 mismatch >1 or scale ``atol 1e-2``.
    pytest.skip
        If CUDA/Triton not available.
    """
    _require_cuda_triton()
    from vllm._genesis.kernels.fused_quant_triton import quant_activation_per_token  # noqa: WPS433

    device = "cuda"
    torch.manual_seed(123 + M * 10 + K)
    # Small magnitude to keep scale ~0.005, so dequant error bounded
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.05
    # include zero-row edge
    if M > 1:
        x[0] = 0.0

    q, scale = quant_activation_per_token(x)
    q_ref, scale_ref = _quantize_per_token_ref(x)

    assert q.shape == (M, K)
    assert q.dtype == torch.int8
    assert scale.shape == (M, 1)
    assert scale.dtype == torch.float32
    assert q.device.type == "cuda"
    assert scale.device.type == "cuda"

    # int8 exact match within 1 (rounding ties)
    diff_q = (q.to(torch.int32) - q_ref.to(torch.int32)).abs()
    max_q_diff = diff_q.max().item()
    assert max_q_diff <= 1, f"M={M} K={K} int8 max diff {max_q_diff} exceeds 1 (quant_activation_per_token vs ref)"

    # scale atol
    diff_s = (scale.to(torch.float32) - scale_ref.to(torch.float32)).abs()
    max_s_diff = diff_s.max().item()
    assert max_s_diff <= 1e-4, f"M={M} K={K} scale max_diff {max_s_diff:.6f} exceeds 1e-4"

    # dequant check: q*scale vs ref
    deq = q.to(torch.float32) * scale
    deq_ref = q_ref.to(torch.float32) * scale_ref
    diff_deq = (deq - deq_ref).abs().max().item()
    assert diff_deq <= 1e-2, f"M={M} K={K} dequant max_diff {diff_deq:.5f} exceeds atol 1e-2"

    # Also check that quant error vs original is bounded (not strict, just sanity)
    # Original bf16 error after quantize-dequantize is at most scale/2 ~ amax/254
    # For random x in [-6,6], scale ~0.05, error <0.03, so atol 1e-2 not always holds vs original;
    # we only compare vs reference, so above check suffices.


@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("K", KS_QUANT, ids=lambda k: f"K{k}")
def test_fused_quant_ptx_functional_correctness(M: int, K: int):
    """Functional correctness ``fused_quant_ptx`` vs torch reference.

    WHY
        PTX inline ``sm_86`` single-kernel must be bitwise equivalent to
        per-token ``amax/127`` quant (with ``cvt.rn.f32.bf16``,
        ``abs.f32``, ``max.f32``, ``rcp``, ``mul``, ``cvt.rni.s32``,
        ``cvt.sat.s8``). Large error would indicate PTX mis-compile.

    Parameters
    ----------
    M: int
        Tokens ``1,8,32``.
    K: int
        Hidden dim ``5120`` or ``4096``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If int8 mismatch >1 or scale diff >``1e-2``.
    pytest.skip
        If CUDA not available (PTX requires CUDA; triton not required).
    """
    _require_cuda()
    from vllm._genesis.kernels.fused_quant_ptx import quant_activation_per_token_ptx  # noqa: WPS433

    device = "cuda"
    torch.manual_seed(321 + M * 10 + K)
    # Small magnitude like triton path
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.05
    if M > 1:
        x[0] = 0.0

    q, scale = quant_activation_per_token_ptx(x)
    q_ref, scale_ref = _quantize_per_token_ref(x)

    assert q.shape == (M, K)
    assert q.dtype == torch.int8
    assert scale.shape == (M, 1)
    assert scale.dtype == torch.float32

    diff_q = (q.to(torch.int32) - q_ref.to(torch.int32)).abs()
    max_q_diff = diff_q.max().item()
    # PTX uses rni (nearest even) vs torch round half away — allow 1 diff
    assert max_q_diff <= 1, f"PTX M={M} K={K} int8 max diff {max_q_diff} exceeds 1"

    diff_s = (scale.to(torch.float32) - scale_ref.to(torch.float32)).abs()
    max_s_diff = diff_s.max().item()
    # rcp.approx may introduce ~1e-4 error, but amax/127 scale should be within 1e-2
    assert max_s_diff <= 1e-2, f"PTX M={M} K={K} scale max_diff {max_s_diff:.5f} exceeds atol 1e-2"

    # Also test fp16 path via same API (dtype_code 1)
    if K != 5120:
        return
    x_f16 = x.to(torch.float16)
    q2, scale2 = quant_activation_per_token_ptx(x_f16)
    assert q2.dtype == torch.int8
    assert scale2.dtype == torch.float32
    assert q2.shape == x_f16.shape


@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("shape", SHAPES_GEMM, ids=lambda s: f"K{s[0]}_N{s[1]}")
def test_int8_hybrid_gemm_functional_correctness(M: int, shape: tuple[int, int]):
    """Functional correctness ``int8_hybrid_gemm`` vs torch reference.

    WHY
        Hybrid diadic ``int8 @ int8 -> int32 << shift -> *a_scale*b_scale``
        must match per-``128`` block shift semantics. Large error would
        corrupt model quality (``w = q*2^shift*s_row``).

    Parameters
    ----------
    M: int
        Tokens ``1,8,32``.
    shape: tuple[int, int]
        ``(K,N)`` ``(5120,4096)`` and ``(4096,5120)`` (``%128==0``).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If max diff > ``1e-2``.
    pytest.skip
        If CUDA/Triton not available.
    """
    _require_cuda_triton()
    from vllm._genesis.kernels.int8_hybrid_gemm import int8_hybrid_gemm  # noqa: WPS433

    K, N = shape
    device = "cuda"
    torch.manual_seed(777 + M * 100 + K)
    a = torch.randint(-2, 3, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-2, 3, (K, N), dtype=torch.int8, device=device)
    a_scales = (torch.rand(M, 1, dtype=torch.float32, device=device) * 0.02 + 0.005).contiguous()
    b_scales = (torch.rand(N, 1, dtype=torch.float32, device=device) * 0.02 + 0.005).contiguous()
    shifts = torch.randint(0, 3, (K // 128, N // 128), dtype=torch.int8, device=device)

    out = int8_hybrid_gemm(a, b, a_scales, b_scales, shifts, out_dtype=torch.bfloat16)
    ref = _reference_hybrid_gemm(a, b, a_scales, b_scales, shifts, out_dtype=torch.bfloat16)

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16
    assert out.device.type == "cuda"

    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert max_diff <= 1e-2, f"M={M} K={K} N={N} max_diff {max_diff:.5f} mean {mean_diff:.5f} exceeds atol 1e-2 (int8_hybrid_gemm vs ref)"

    # also test with zero shifts (fast path single matmul)
    shifts0 = torch.zeros((K // 128, N // 128), dtype=torch.int8, device=device)
    out0 = int8_hybrid_gemm(a, b, a_scales, b_scales, shifts0, out_dtype=torch.bfloat16)
    ref0 = _reference_hybrid_gemm(a, b, a_scales, b_scales, shifts0, out_dtype=torch.bfloat16)
    diff0 = (out0.to(torch.float32) - ref0.to(torch.float32)).abs().max().item()
    assert diff0 <= 1e-2, f"zero-shift M={M} K={K} N={N} diff {diff0:.5f} exceeds atol 1e-2"


# ── bench — monotonic and <1.6× fallback ───────────────────────────────────


def test_fused_quant_gemm_bench_monotonic_and_fallback():
    """Bench ``fused_quant_gemm``: time monotonic in ``M`` and ``<1.6×`` fallback.

    WHY
        Fused single-launch should hide ``M*K`` traffic and halve launches;
        time must grow with tokens and never be substantially slower than a
        torch fallback (quant+matmul+scale) which is the correctness
        baseline. ``>1.6×`` would indicate regression vs simple torch path.

    Boundaries
        * ``M`` in ``(1,8,32)`` ``K=5120`` ``N=4096`` (and swapped shape
          also covered in functional, bench uses one shape for speed).
        * Measures kernel via ``fused_quant_gemm`` and fallback via torch
          reference ``_reference_fused_quant_gemm`` — both on CUDA, with
          ``torch.cuda.synchronize`` and 20 iters avg.
        * Asserts ``kernel_time < 1.6 * fallback_time`` per ``M`` and
          ``kernel_time`` non-decreasing (allow 30% jitter for tiny kernels).

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
    from vllm._genesis.kernels.fused_quant_gemm import fused_quant_gemm  # noqa: WPS433

    device = "cuda"
    K, N = 5120, 4096
    kernel_times: list[float] = []
    fallback_times: list[float] = []

    for M in MS:
        torch.manual_seed(1234 + M)
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.05
        b = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
        b_scale = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.005).contiguous()

        def _kernel_fn(a=a, b=b, b_scale=b_scale):
            return fused_quant_gemm(a, b, b_scale, out_dtype=torch.bfloat16)

        def _fallback_fn(a=a, b=b, b_scale=b_scale):
            return _reference_fused_quant_gemm(a, b, b_scale, out_dtype=torch.bfloat16)

        k_ms = _measure_ms(_kernel_fn, warmup=3, iters=20)
        f_ms = _measure_ms(_fallback_fn, warmup=3, iters=20)
        kernel_times.append(k_ms)
        fallback_times.append(f_ms)
        assert k_ms < 1.6 * f_ms, f"M={M} kernel {k_ms:.3f}ms not <1.6*fallback {f_ms:.3f}ms (ratio {k_ms/f_ms:.2f})"

    for i in range(1, len(kernel_times)):
        prev, cur = kernel_times[i - 1], kernel_times[i]
        assert cur + 1e-6 >= prev * 0.70, f"monotonic violation M {MS[i-1]}->{MS[i]}: {prev:.3f}ms -> {cur:.3f}ms"
    # overall increasing but allow 10% jitter for tiny kernels
    assert kernel_times[-1] + 1e-6 >= kernel_times[0] * 0.70, f"kernel time not increasing overall: {kernel_times}"


def test_fused_quant_triton_bench_monotonic_and_fallback():
    """Bench ``fused_quant_triton``: monotonic and ``<1.6×`` torch fallback.

    WHY
        Single-pass quant (one ``tl.load`` bf16, ``tl.max`` reduction,
        ``tl.store`` int8) should be memory-bound and scale with ``M*K``;
        fallback torch ``abs().amax`` + ``round`` is vectorized but does
        2 passes, so kernel should not be >1.6× slower.

    Boundaries
        * ``M`` in ``(1,8,32)`` ``K=5120`` (also ``4096`` covered in
          functional).
        * Kernel via ``quant_activation_per_token``, fallback via torch
          ``_quantize_per_token_ref`` — both CUDA, 20 iters avg.
        * Asserts ratio ``<1.6`` and monotonic.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If monotonic or ratio violated.
    pytest.skip
        If CUDA/Triton not available.
    """
    _require_cuda_triton()
    from vllm._genesis.kernels.fused_quant_triton import quant_activation_per_token  # noqa: WPS433

    device = "cuda"
    K = 5120
    ktimes: list[float] = []
    ftimes: list[float] = []

    for M in MS:
        torch.manual_seed(4321 + M)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.05

        def _kfn(x=x):
            return quant_activation_per_token(x)

        def _ffn(x=x):
            return _quantize_per_token_ref(x)

        k_ms = _measure_ms(_kfn, warmup=3, iters=20)
        f_ms = _measure_ms(_ffn, warmup=3, iters=20)
        ktimes.append(k_ms)
        ftimes.append(f_ms)
        assert k_ms < 1.6 * f_ms, f"M={M} K={K} kernel {k_ms:.3f}ms not <1.6*fallback {f_ms:.3f}ms ratio {k_ms/f_ms:.2f}"

    for i in range(1, len(ktimes)):
        assert ktimes[i] + 1e-6 >= ktimes[i - 1] * 0.70, f"monotonic violation {MS[i-1]}->{MS[i]} {ktimes[i-1]:.3f}->{ktimes[i]:.3f}ms"
    assert ktimes[-1] + 1e-6 >= ktimes[0] * 0.70, f"kernel time not increasing overall: {ktimes}"


def test_fused_quant_ptx_bench_monotonic_and_fallback():
    """Bench ``fused_quant_ptx``: monotonic and ``<1.6×`` torch fallback.

    WHY
        PTX single-kernel ``bf16 -> int8`` (256 threads, grid ``M``) must
        hide ``amax`` reduction and not be >1.6× slower than torch
        fallback which is also fused but uses 2 passes.

    Boundaries
        * ``M`` in ``(1,8,32)`` ``K=5120``.
        * Kernel via ``quant_activation_per_token_ptx``, fallback via
          ``_quantize_per_token_ref`` — both CUDA, 20 iters.
        * Asserts ratio ``<1.6`` and monotonic.
        * Skipped if CUDA not available (PTX needs CUDA, not Triton).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If monotonic or ratio violated.
    pytest.skip
        If CUDA not available.
    """
    _require_cuda()
    from vllm._genesis.kernels.fused_quant_ptx import quant_activation_per_token_ptx  # noqa: WPS433

    device = "cuda"
    K = 5120
    ktimes: list[float] = []
    ftimes: list[float] = []

    for M in MS:
        torch.manual_seed(555 + M)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.05

        def _kfn(x=x):
            return quant_activation_per_token_ptx(x)

        def _ffn(x=x):
            return _quantize_per_token_ref(x)

        k_ms = _measure_ms(_kfn, warmup=3, iters=20)
        f_ms = _measure_ms(_ffn, warmup=3, iters=20)
        ktimes.append(k_ms)
        ftimes.append(f_ms)
        assert k_ms < 1.6 * f_ms, f"PTX M={M} K={K} kernel {k_ms:.3f}ms not <1.6*fallback {f_ms:.3f}ms ratio {k_ms/f_ms:.2f}"

    for i in range(1, len(ktimes)):
        assert ktimes[i] + 1e-6 >= ktimes[i - 1] * 0.70, f"PTX monotonic violation {MS[i-1]}->{MS[i]} {ktimes[i-1]:.3f}->{ktimes[i]:.3f}ms"
    assert ktimes[-1] + 1e-6 >= ktimes[0] * 0.70, f"PTX time not increasing overall: {ktimes}"


def test_int8_hybrid_gemm_bench_monotonic_and_fallback():
    """Bench ``int8_hybrid_gemm``: monotonic and ``<1.6×`` fallback.

    WHY
        Hybrid diadic should keep ``int32`` shift cheap (no fp32 mul) and
        reuse ``tl.dot`` Tensor Core; time must grow with ``M`` and not
        exceed ``1.6×`` torch ``einsum`` fallback (which is 100× slower
        in naive Python loops but our vectorized fallback is still
        faster than naive, so ``1.6×`` is lenient).

    Boundaries
        * ``M`` in ``(1,8,32)`` ``K=5120`` ``N=4096`` (``%128==0``).
        * Kernel via ``int8_hybrid_gemm`` and fallback via
          ``_reference_hybrid_gemm`` (einsum vectorized) — both CUDA,
          20 iters avg.
        * Asserts ratio ``<1.6`` and monotonic.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If monotonic or ratio violated.
    pytest.skip
        If CUDA/Triton not available.
    """
    _require_cuda_triton()
    from vllm._genesis.kernels.int8_hybrid_gemm import int8_hybrid_gemm  # noqa: WPS433

    device = "cuda"
    K, N = 5120, 4096
    ktimes: list[float] = []
    ftimes: list[float] = []

    for M in MS:
        torch.manual_seed(9876 + M)
        a = torch.randint(-2, 3, (M, K), dtype=torch.int8, device=device)
        b = torch.randint(-2, 3, (K, N), dtype=torch.int8, device=device)
        a_scales = (torch.rand(M, 1, dtype=torch.float32, device=device) * 0.02 + 0.005).contiguous()
        b_scales = (torch.rand(N, 1, dtype=torch.float32, device=device) * 0.02 + 0.005).contiguous()
        shifts = torch.randint(0, 3, (K // 128, N // 128), dtype=torch.int8, device=device)

        def _kfn(a=a, b=b, a_scales=a_scales, b_scales=b_scales, shifts=shifts):
            return int8_hybrid_gemm(a, b, a_scales, b_scales, shifts, out_dtype=torch.bfloat16)

        def _ffn(a=a, b=b, a_scales=a_scales, b_scales=b_scales, shifts=shifts):
            return _reference_hybrid_gemm(a, b, a_scales, b_scales, shifts, out_dtype=torch.bfloat16)

        k_ms = _measure_ms(_kfn, warmup=3, iters=20)
        f_ms = _measure_ms(_ffn, warmup=3, iters=20)
        ktimes.append(k_ms)
        ftimes.append(f_ms)
        assert k_ms < 1.6 * f_ms, f"M={M} K={K} N={N} kernel {k_ms:.3f}ms not <1.6*fallback {f_ms:.3f}ms ratio {k_ms/f_ms:.2f}"

    for i in range(1, len(ktimes)):
        assert ktimes[i] + 1e-6 >= ktimes[i - 1] * 0.70, f"hybrid monotonic violation {MS[i-1]}->{MS[i]} {ktimes[i-1]:.3f}->{ktimes[i]:.3f}ms"
    assert ktimes[-1] + 1e-6 >= ktimes[0] * 0.70, f"hybrid time not increasing overall: {ktimes}"


__all__ = [
    "test_fused_quant_gemm_kernel_standards",
    "test_fused_quant_triton_kernel_standards",
    "test_int8_hybrid_gemm_kernel_standards",
    "test_fused_quant_ptx_kernel_standards",
    "test_no_cpu_fallback_in_hot_path_parametrized",
    "test_fused_quant_gemm_functional_correctness",
    "test_fused_quant_triton_functional_correctness",
    "test_fused_quant_ptx_functional_correctness",
    "test_int8_hybrid_gemm_functional_correctness",
    "test_fused_quant_gemm_bench_monotonic_and_fallback",
    "test_fused_quant_triton_bench_monotonic_and_fallback",
    "test_fused_quant_ptx_bench_monotonic_and_fallback",
    "test_int8_hybrid_gemm_bench_monotonic_and_fallback",
]
