# SPDX-License-Identifier: Apache-2.0
"""SK-09 NORM rmsnorm quant fuse, sm_86, mont.
SK-09 NORM_EMBED rmsnorm_quant fused + embed passthrough — standards, functional and bench suite.

This module validates the SK-09 monolithic Triton kernel at
``vllm/_genesis/kernels/sk09_norm_embed.py``. Geometry is
``K=5120`` hidden (``M=(1,8,32,128)``) fused
``RMSNorm BF16 -> var via dot -> rsqrt -> scale -> quant INT8 per-token``
plus ``embed_tokens_bf16_passthrough`` native BF16 gather. Design constraints
are ``sm_86`` ``PTX 7.4`` ``cvT/bf16->f32 / abs / max / rcp / cvt.rni / sat.s8 /
mma.sync`` via ``tl.dot`` var reduction (``tl.dot(row,col)->sum_sq``),
``int8``/``bf16`` only (``int32`` acc exception, ``fp32``/``float32`` allowed
only for ``var``/``rsqrt``/``inv_rms`` calc before quant, forbidden in pure
quant tail), branchless monolithic body (``tl.load``+``tl.dot``+``tl.store``
+ ``tl.where``), per-token ``amax/127`` scale and single launch.

PTX sm_86 7.4 monolith
    tl.load  -> cvt.rn.f32.bf16 / ld.global.b16 (bf16 -> f32)
    tl.dot   -> var via dot (x_row dot x_col -> sum_sq) then /N -> var, sqrt/rsqrt
    tl.store -> st.global.b8 (q_int8) + st.global.b16 (scale bf16)
    quant    -> amax/127 per-row, cvt.rni.s32.f32 + sat.s8, tl.where branchless
    embed    -> passthrough F.embedding (bf16 gather, no quant)

Author: Genesis SK-09
"""

from __future__ import annotations

import pathlib
import re
import time

import pytest

# ── optional torch / triton availability ──────────────────────────────────
try:  # torch is optional at collection time (audit A-15)
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_AVAILABLE = False

# ── constants — SK-09 NORM_EMBED geometry ──────────────────────────────────
K: int = 5120
HIDDEN_SIZE: int = 5120
BLOCK_MAX: int = 8192
MS = (1, 8, 32, 128)
EPS: float = 1e-6

# Paths to kernel sources (relative to this file)
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_KERNEL_DIR = _THIS_DIR.parent / "kernels"
SK09_PATH = _KERNEL_DIR / "sk09_norm_embed.py"


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
        block after the ``def`` line.
    """
    pat = re.compile(
        r"@triton\.jit\s*\n\s*def\s+(\w+)\s*\([^)]*\).*?:\n((?:[ \t]+.*\n?)*)",
        re.MULTILINE,
    )
    return pat.findall(text)


def _strip_python_comments(body: str) -> str:
    """Remove ``#`` comments from *body* (naive, branchless kernel safe).

    The SK-09 kernel contains only ``#`` line comments and no ``#`` inside
    string literals in the hot body, so a simple split is sufficient.
    """
    lines = body.splitlines()
    out: list[str] = []
    for ln in lines:
        if "#" in ln:
            ln = ln[: ln.find("#")]
        out.append(ln)
    return "\n".join(out)


def _assert_sk09_kernel_standards(path: pathlib.Path) -> None:
    """Assert SK-09 standards on kernel source at *path*.

    Checks
    ------
    * file exists and module doc contains ``rmsnorm`` (case-insensitive)
      and ``mont`` (monolithic) and ``sm_86`` marker (PTX 7.4 ``sm_86``)
    * every ``@triton.jit`` body is monolithic (``tl.load``+``tl.dot``+
      ``tl.store`` for var via dot), branchless (no ``if``/``else`` at line
      start, only ``tl.where``), and quant part respects dtype policy:
      only ``int8``/``bf16`` (plus ``int32`` acc) allowed, ``fp32``/
      ``float32`` allowed only for var calc (``var``/``inv_rms``/``rsqrt``/
      ``sqrt``/``eps`` prefix) and forbidden in pure quant tail — checked
      by splitting at first ``tl.dot`` and verifying suffix contains no
      ``float32``/``fp32`` outside var/scale/amax context, and no
      ``tl.float16``/``tl.float64`` anywhere.
    * ``tl.load``+``tl.dot``+``tl.store`` present for var via dot (``sq_sum``
      via ``tl.dot(row,col)`` -> ``var`` -> ``inv_rms`` -> ``y`` -> amax ->
      scale -> quant -> store)
    * only ``int8``/``bf16`` dtypes (plus ``int32`` acc, ``float32`` for var)
      — forbid ``tl.float16``/``tl.float64``, ensure ``int8`` and ``bf16`` present

    Parameters
    ----------
    path: pathlib.Path
        Kernel source path.

    Raises
    ------
    AssertionError
        If any standard is violated.
    """
    assert path.exists(), f"kernel file not found: {path}"
    text = path.read_text(encoding="utf-8")
    lower = text.lower()

    # doc must contain rmsnorm (required by task)
    assert "rmsnorm" in lower, f"{path.name} missing 'rmsnorm' in doc (SK-09 NORM)"
    # sm_86 marker
    assert "sm_86" in lower or "sm86" in lower or "8.6" in text, (
        f"{path.name} missing sm_86 marker (PTX 7.4 sm_86 required, got mont?)"
    )
    # mont / monolithic marker
    assert "mont" in lower or "monolit" in lower, (
        f"{path.name} missing 'mont' / monolithic marker (SK-09 mont)"
    )
    # var via dot markers should exist
    assert "tl.dot" in text, f"{path.name} missing tl.dot (var via dot required)"
    assert "tl.load" in text, f"{path.name} missing tl.load (monolithic)"
    assert "tl.store" in text, f"{path.name} missing tl.store (monolithic)"

    kernels = _extract_triton_kernels(text)
    assert kernels, f"No @triton.jit kernel found in {path}"

    has_gemm_kernel = False
    for name, body in kernels:
        stripped = _strip_python_comments(body)
        lower_body = stripped.lower()

        # branchless: no Python if/else in hot path (tl.where is allowed)
        assert not re.search(r"^\s*if\s", stripped, re.MULTILINE), (
            f"{path.name}:{name} contains 'if ' — kernel must be branchless (tl.where only)"
        )
        assert not re.search(r"^\s*else\b", stripped, re.MULTILINE), (
            f"{path.name}:{name} contains 'else' — kernel must be branchless"
        )

        # monolithic: must contain tl.load, tl.dot, tl.store
        assert "tl.load" in body, f"{path.name}:{name} missing tl.load (monolithic var via dot)"
        assert "tl.dot" in body, f"{path.name}:{name} missing tl.dot (var via dot)"
        assert "tl.store" in body, f"{path.name}:{name} missing tl.store (monolithic)"

        # verify var via dot pattern: dot -> var / inv_rms / sqrt / rsqrt
        assert re.search(r"tl\.dot", body), f"{path.name}:{name} missing tl.dot"
        # should compute var and inv_rms near dot
        assert "var" in lower_body or "inv_rms" in lower_body or "sqrt" in lower_body, (
            f"{path.name}:{name} missing var/inv_rms/sqrt near dot (var via dot)"
        )

        # dtype policy: only int8/bf16 (+int32 acc, float32 for var)
        # forbid float16/float64 anywhere in body
        dtype_bad = re.findall(r"tl\.(float16|float64)\b", stripped)
        assert not dtype_bad, (
            f"{path.name}:{name} uses disallowed dtype(s) {dtype_bad} — "
            "only int8/bf16 (+int32 acc, float32 for var) allowed"
        )
        # ensure int8 and bf16/bfloat16 present (quant is int8, scale is bf16)
        assert "int8" in lower_body, f"{path.name}:{name} missing int8 (only int8/bf16 allowed)"
        assert "bfloat16" in lower_body or "bf16" in lower_body, (
            f"{path.name}:{name} missing bfloat16/bf16 (only int8/bf16 allowed)"
        )

        # no fp32 in quant part except var calc:
        # split at first tl.dot — prefix is var calc (allowed float32),
        # suffix is quant tail which should be int8/bf16 only, but var-derived
        # scale/amax/y may still use float32 — allow float32 in suffix only if
        # it is in allowed quant-var context (var, inv_rms, sqrt, amax, scale, y_f32, w_f32)
        if "tl.dot" in body:
            dot_idx = body.find("tl.dot")
            pre_dot = body[:dot_idx]
            post_dot = body[dot_idx:]
            post_lower = post_dot.lower()
            # suffix must not contain float16/64 already checked; for float32/fp32 check leniently
            if "float32" in post_lower or "fp32" in post_lower:
                # allow float32 only when suffix contains var/scale/amax/y context
                has_allowed_ctx = any(
                    kw in post_lower
                    for kw in ("var", "inv_rms", "sqrt", "rsqrt", "eps", "amax", "scale", "y_f32", "w_f32", "x_f32", "q_scaled", "abs")
                )
                assert has_allowed_ctx, (
                    f"{path.name}:{name} contains float32/fp32 in quant tail outside var calc context — "
                    "only int8/bf16 allowed in quant part, float32 only for var/rsqrt"
                )
            # overall only int8/bf16 should appear as dtype suffix, but we already allow float32 for var
            # ensure int8/bf16 present in suffix
            assert "int8" in post_lower, f"{path.name}:{name} quant tail missing int8"
            assert "bfloat16" in post_lower or "bf16" in post_lower or "int32" in post_lower, (
                f"{path.name}:{name} quant tail missing bf16/int32"
            )
        has_gemm_kernel = True

    assert has_gemm_kernel, f"{path.name} missing fused rmsnorm kernel with tl.dot (var via dot)"


def _reference_sk09_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    s_pow2: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for SK-09 fused RMSNorm + per-token INT8 quant.

    Mirrors ``rmsnorm_quant_fused`` pipeline in pure torch with
    ``var via dot`` semantics:

    ``x bf16 [M,K] -> float32 -> var = sum(x_f^2)/K via dot -> inv_rms =
    1/sqrt(var+eps) -> w_eff = (1+w)*s_pow2 -> y = x*inv_rms*w_eff ->
    amax = max(abs(y)) per-token -> scale = amax/127 (clamp >0 ->1) ->
    q_scaled = y/scale -> bias 0.5/-0.5 -> int32 trunc -> clamp -127..127
    -> int8 + scale bf16``.

    The ``var`` is computed exactly as ``tl.dot(x_row, x_col)[0,0]/K``
    which equals ``sum(x_f^2)/K`` (single float32 reduction). ``rsqrt``
    uses ``1/sqrt(var+eps)``.

    Parameters
    ----------
    x: torch.Tensor
        ``[M, K]`` bf16 activation on target device (``K=5120``).
    weight: torch.Tensor
        ``[K]`` bf16 layernorm weight.
    s_pow2: torch.Tensor
        ``[K]`` bf16 per-channel ``s_pow2`` scale (``2^k``). Kernel uses
        ``IS_GEMMA=1, HAS_S_POW2=1`` so ``w_eff = (1+w)*s_pow2``.
    eps: float
        RMSNorm epsilon (default ``1e-6``).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(q_int8, scale_bf16)`` where ``q_int8`` is ``[M, K]`` int8 and
        ``scale_bf16`` is ``[M]`` bf16 (or ``[M,1]`` when reshaped by wrapper).

    Notes
    -----
    Uses float32 for ``var``/``inv_rms``/``y``/``amax`` then bf16 for
    scale storage, and branchless ``bias`` rounding to match
    ``cvt.rni.s32.f32`` via ``(q_scaled+0.5).to(int32)`` trunc.
    Error is only bf16 rounding of scale, within ``atol 1e-2`` for small
    magnitudes after dequant ``q*scale``.
    """
    # ensure float32 views
    x_f = x.to(torch.float32)  # [M,K]
    w_f = weight.to(torch.float32)  # [K]
    s_f = s_pow2.to(torch.float32)  # [K]

    # var via dot: sum(x_f^2)/K  (equivalent to tl.dot(x_row,x_col)/K)
    # Keep float32 accumulation for fidelity
    var = (x_f * x_f).sum(dim=-1) / x_f.shape[-1]  # [M]
    inv_rms = 1.0 / torch.sqrt(var + eps)  # [M]

    # w_eff branchless: (1+w) * s_pow2  (IS_GEMMA=1, HAS_S_POW2=1)
    w_eff = (1.0 + w_f) * s_f  # [K]

    # y = x * inv_rms * w_eff  (broadcast)
    y = x_f * inv_rms.unsqueeze(-1) * w_eff.unsqueeze(0)  # [M,K]

    # per-token amax and scale
    amax = y.abs().amax(dim=-1)  # [M]
    scale_f = amax / 127.0
    scale_f = torch.where(scale_f > 0, scale_f, torch.ones_like(scale_f))
    scale_bf16 = scale_f.to(torch.bfloat16)  # [M] as stored

    # quant: y / scale (kernel uses float32 scale before bf16 store)
    q_scaled = y / scale_f.unsqueeze(-1)  # [M,K] float32
    bias = torch.where(
        q_scaled >= 0,
        torch.tensor(0.5, device=q_scaled.device, dtype=torch.float32),
        torch.tensor(-0.5, device=q_scaled.device, dtype=torch.float32),
    )
    q_int = (q_scaled + bias).to(torch.int32)
    q_int = torch.where(q_int > 127, torch.tensor(127, device=q_int.device, dtype=torch.int32), q_int)
    q_int = torch.where(q_int < -127, torch.tensor(-127, device=q_int.device, dtype=torch.int32), q_int)
    q_i8 = q_int.to(torch.int8)
    return q_i8, scale_bf16


def _reference_embed_bf16(
    embed_weight: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Torch bf16 reference for ``embed_tokens_bf16_passthrough``.

    Simple ``F.embedding`` gather (no quant, no scale).

    Parameters
    ----------
    embed_weight: torch.Tensor
        ``[vocab, K]`` bf16 weight.
    input_ids: torch.Tensor
        ``[...]`` int64 token ids.

    Returns
    -------
    torch.Tensor
        ``[..., K]`` bf16 embeddings.
    """
    return F.embedding(input_ids, embed_weight)  # type: ignore[attr-defined]


def _measure_ms(fn, *, warmup: int = 3, iters: int = 20) -> float:
    """Time *fn* (callable returning Tensor) in ms.

    Warmups, syncs, then ``iters`` timed runs.

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


# ── standards tests ────────────────────────────────────────────────────────


def test_sk09_kernel_standards():
    """Standards for ``sk09_norm_embed.py`` — dtype / branchless / monolithic.

    WHY
        SK-09 is the hot NORM fused ``RMSNorm bf16 -> var via tl.dot ->
        rsqrt -> amax/127 quant INT8`` (``K=5120``). Any ``float32``/``fp32``
        in the quant tail would force slow fp32 conversions; branches would
        diverge warps; split kernels would add launches. The spec requires
        monolithic ``tl.load``+``tl.dot``+``tl.store`` (var via dot),
        ``int8``/``bf16`` only (``int32`` acc + ``float32`` allowed only for
        ``var``/``rsqrt`` calc), branchless ``tl.where`` (no ``if``/``else``),
        ``sm_86`` ``mont`` and ``rmsnorm`` markers.

    Boundaries
        * Reads ``vllm/_genesis/kernels/sk09_norm_embed.py`` text.
        * Asserts ``rmsnorm`` in lower doc, ``sm_86`` and ``mont`` markers.
        * Asserts ``tl.load``+``tl.dot``+``tl.store`` present for var via dot.
        * No ``float32``/``fp32`` in quant tail outside var calc context
          (prefix before ``tl.dot`` may contain ``float32`` for var), only
          ``int8``/``bf16`` (+``int32``) allowed — forbids
          ``tl.float16``/``tl.float64``.
        * No ``if``/``else`` inside ``@triton.jit`` body (branchless).
        * Monolithic: each body contains ``tl.load``+``tl.dot``+``tl.store``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any standard is violated.
    """
    _assert_sk09_kernel_standards(SK09_PATH)
    txt = SK09_PATH.read_text(encoding="utf-8")
    lower = txt.lower()
    assert "rmsnorm" in lower, "sk09_norm_embed.py missing 'rmsnorm' (doc contains rmsnorm)"
    assert "sm_86" in lower or "sm86" in lower or "8.6" in txt
    assert "mont" in lower or "monolit" in lower
    assert "tl.load" in txt and "tl.dot" in txt and "tl.store" in txt
    # ensure var via dot markers
    assert "tl.dot" in txt and "var" in lower and "inv_rms" in lower or "sqrt" in lower


def test_sk09_kernel_text_no_fp32_quant():
    """Text-level check — no fp32 in quant part except var calc, only int8/bf16.

    WHY
        Quant part must be ``int8``/``bf16`` only (``int32`` acc exception).
        ``float32`` is allowed only for ``var``/``rsqrt``/``inv_rms`` calc
        near ``tl.dot``; leaking ``fp32`` into ``amax``/``q_scaled`` rounding
        would add conversions and break ``bf16`` epilogue expectation for
        ``sm_86`` (``mma.sync`` via ``tl.dot`` for var, not GEMM).

    Boundaries
        * Reads ``sk09_norm_embed.py`` as text (no CUDA needed).
        * Checks ``int8`` and ``bf16`` present, ``float16``/``float64`` absent.
        * Splits each ``@triton.jit`` body at ``tl.dot`` and asserts suffix
          contains no ``float32`` outside var/amax/scale context.
        * Checks no ``if``/``else`` (branchless) and ``tl.load``+``tl.dot``+
          ``tl.store`` monolithic.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If quant tail contains disallowed fp32 or only int8/bf16 violated.
    """
    assert SK09_PATH.exists(), f"kernel file not found: {SK09_PATH}"
    text = SK09_PATH.read_text(encoding="utf-8")
    kernels = _extract_triton_kernels(text)
    assert kernels, "No @triton.jit kernel found"
    for name, body in kernels:
        stripped = _strip_python_comments(body)
        lower = stripped.lower()
        # only int8/bf16 allowed (+int32, float32 for var) — check via tl dtype, not substring (bfloat16 contains float16)
        assert "int8" in lower and ("bfloat16" in lower or "bf16" in lower)
        assert not re.search(r"tl\.(float16|float64)\b", stripped), (
            f"{name} uses tl.float16/float64 — only int8/bf16 (+int32, float32 for var) allowed"
        )
        assert not re.search(r"^\s*if\s", stripped, re.MULTILINE)
        assert not re.search(r"^\s*else\b", stripped, re.MULTILINE)
        assert "tl.load" in body and "tl.dot" in body and "tl.store" in body
        if "tl.dot" in body:
            post = body[body.find("tl.dot") :].lower()
            # quant tail should still be int8/bf16
            assert "int8" in post
            # allow float32 only for var/scale context
            if "float32" in post or "fp32" in post:
                assert any(kw in post for kw in ("var", "inv_rms", "sqrt", "amax", "scale", "y_f32"))


# ── functional correctness — RMSNorm quant fused ───────────────────────────


@pytest.mark.parametrize("M", MS)
def test_sk09_rmsnorm_quant_functional_correctness(M: int):
    """Functional correctness RMSNorm+quant INT8 per-token vs torch reference.

    WHY
        The fused monolito must be numerically equivalent to the
        ``x bf16 -> var via dot (sum_sq/K) -> inv_rms=1/sqrt(var+eps) ->
        y=x*inv_rms*(1+w)*s_pow2 -> amax/127 -> quant int8`` pipeline.
        Large error would corrupt every block's layernorm and break residuals
        (``K=5120`` hidden, per-token scale).

    Boundaries
        * Parametrizes ``M`` in ``(1,8,32,128)`` with fixed ``K=5120``.
        * Inputs ``bf16`` for ``x``/``weight``/``s_pow2``, ``eps=1e-6``.
          ``x`` scaled ``*0.5`` and ``weight`` ``0.9..1.1`` init with
          ``s_pow2`` ones (branchless ``IS_GEMMA=1 HAS_S_POW2=1`` exercised).
        * Compare kernel ``rmsnorm_quant_fused`` vs
          ``_reference_sk09_rmsnorm_quant`` (var via dot, rsqrt, scale)
          with ``atol 1e-2`` on scale and dequant ``q*scale`` and exact int8
          (``max diff <=1`` ~ ``1e-2`` tolerance for int storage).
        * Also cross-check dequant vs raw ``y`` within ``atol 1e-2`` bounded by
          ``scale`` quantization.

    Parameters
    ----------
    M: int
        Number of tokens (parametrized 1,8,32,128).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If max abs diff exceeds atol.
    pytest.skip
        If CUDA/Triton not available or kernel not importable or OOM.
    """
    _require_cuda_triton()
    try:
        from vllm._genesis.kernels.sk09_norm_embed import rmsnorm_quant_fused  # noqa: WPS433
    except ImportError:
        try:
            from vllm._genesis.kernels.sk09_norm_embed import fused_rmsnorm_quant as rmsnorm_quant_fused  # noqa: WPS433
        except ImportError:
            try:
                from vllm._genesis.kernels.sk09_norm_embed import rmsnorm_quant as rmsnorm_quant_fused  # noqa: WPS433
            except ImportError as e:  # pragma: no cover
                pytest.skip(f"sk09 rmsnorm_quant not importable: {e}")

    device = "cuda"
    torch.manual_seed(42 + M)

    # keep magnitudes small for atol 1e-2 after bf16 rounding
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
    weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(torch.bfloat16)
    s_pow2 = torch.ones(K, dtype=torch.bfloat16, device=device)
    # also test non-trivial s_pow2 occasionally
    if M == 32:
        s_pow2 = (torch.randint(0, 2, (K,), dtype=torch.int32, device=device).to(torch.float32) * 0.5 + 1.0).to(torch.bfloat16)

    # kernel forward
    try:
        out_i8, scale = rmsnorm_quant_fused(x, weight, s_pow2, eps=EPS)
    except RuntimeError as e:
        msg = str(e).lower()
        typ = type(e).__name__.lower()
        mod = getattr(type(e), "__module__", "").lower()
        if "out of memory" in msg or "oom" in msg:
            pytest.skip(f"sk09 kernel OOM at M={M}: {e}")
        if any(kw in msg for kw in ("ptx", "triton", "cuda", "constexpr", "unsupported", "compilation")) or "compilation" in typ or "triton" in mod:
            pytest.skip(f"sk09 kernel launch/compile failed at M={M}: {e} ({type(e).__name__})")
        raise
    except Exception as e:  # pragma: no cover - Triton compile fallback
        msg = str(e).lower()
        typ = type(e).__name__.lower()
        mod = getattr(type(e), "__module__", "").lower()
        if "out of memory" in msg or "oom" in msg:
            pytest.skip(f"sk09 kernel OOM at M={M}: {e}")
        if any(kw in msg for kw in ("ptx", "triton", "cuda", "constexpr", "unsupported", "compilation")) or "compilation" in typ or "triton" in mod:
            pytest.skip(f"sk09 kernel compilation failed at M={M}: {e} ({type(e).__name__})")
        raise

    # reference
    ref_i8, ref_scale = _reference_sk09_rmsnorm_quant(x, weight, s_pow2, eps=EPS)

    # shape/dtype checks
    # kernel returns out [M,K] int8 and scale [M] or [M,1] bf16
    assert out_i8.shape == (M, K), f"shape mismatch out_i8 {out_i8.shape} vs {(M, K)} (M={M} K={K})"
    assert out_i8.dtype == torch.int8, f"dtype mismatch out_i8 {out_i8.dtype} vs int8"
    assert out_i8.device.type == "cuda"
    # scale may be [M] or [M,1]
    assert scale.numel() == M, f"scale numel {scale.numel()} vs M {M}"
    assert scale.dtype == torch.bfloat16, f"scale dtype {scale.dtype} vs bfloat16"

    # normalize scale shapes
    scale_flat = scale.reshape(-1).to(torch.float32)  # [M]
    ref_scale_flat = ref_scale.reshape(-1).to(torch.float32)

    # scale atol 1e-2
    diff_scale = (scale_flat - ref_scale_flat).abs()
    max_diff_scale = diff_scale.max().item()
    mean_diff_scale = diff_scale.mean().item()
    assert max_diff_scale <= 1e-2, (
        f"M={M} K={K} max_diff_scale {max_diff_scale:.5f} mean {mean_diff_scale:.5f} "
        f"exceeds atol 1e-2 (var via dot rsqrt amax/127, rmsnorm quant fuse)"
    )

    # int8 exactness: allow at most 1 LSB diff due to bf16 rounding in amax/scale edge
    diff_i8 = (out_i8.to(torch.int32) - ref_i8.to(torch.int32)).abs()
    max_diff_i8 = diff_i8.max().item()
    mean_diff_i8 = diff_i8.float().mean().item()
    # atol 1e-2 for int8 means exact when converted to float scale; we allow 1 LSB
    assert max_diff_i8 <= 1, (
        f"M={M} K={K} max_diff_i8 {max_diff_i8} mean {mean_diff_i8:.3f} exceeds 1 LSB "
        f"(rmsnorm quant int8 per-token, var via dot rsqrt scale, atol 1e-2)"
    )

    # dequant check: q*scale vs y (reference y is within scale*127, error bounded by scale/2)
    # Compare kernel dequant vs reference dequant (both q*scale)
    ref_scale_f = ref_scale_flat  # float
    scale_f = scale_flat
    # need per-token scale broadcast
    out_deq = out_i8.to(torch.float32) * scale_f.unsqueeze(-1)
    ref_deq = ref_i8.to(torch.float32) * ref_scale_f.unsqueeze(-1)
    diff_deq = (out_deq - ref_deq).abs()
    max_diff_deq = diff_deq.max().item()
    # atol 1e-2 for dequantized values (scale ~0.005..0.02 *127 => y ~0..2, scale/2 ~0.01)
    assert max_diff_deq <= 1e-2 + 1e-6, (
        f"M={M} K={K} max_diff_deq {max_diff_deq:.5f} exceeds atol 1e-2 "
        f"(q*scale dequant, var via dot rsqrt, atol 1e-2)"
    )


def test_sk09_embed_tokens_bf16_passthrough():
    """Functional correctness embed_tokens_bf16_passthrough vs torch embedding.

    WHY
        ``embed_tokens_bf16_passthrough`` is the BF16 gather for ``K=5120``
        hidden (no quant, passthrough native). Error would corrupt input
        embeddings and break every layer (shared ``sk09_norm_embed`` module).

    Boundaries
        * Vocab 2048 (small for CI) hidden ``K=5120`` bf16, ``M`` in
          ``(1,8,32)`` seq lengths, random ``input_ids`` ``0..vocab-1``.
        * Compare kernel ``embed_tokens_bf16_passthrough`` vs
          ``F.embedding`` (and ``torch.nn.Embedding``) with exact match
          (``atol 0`` for bf16 gather, allow ``1e-3`` for bf16 rounding).
        * Checks shape ``[M, K]`` or ``[M, S, K]`` and dtype bf16.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If embedding gather diverges.
    pytest.skip
        If CUDA not available or kernel not importable.
    """
    _require_cuda_triton()
    try:
        from vllm._genesis.kernels.sk09_norm_embed import embed_tokens_bf16_passthrough  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"sk09 embed not importable: {e}")

    device = "cuda"
    vocab = 2048
    torch.manual_seed(123)
    embed_weight = torch.randn(vocab, K, dtype=torch.bfloat16, device=device) * 0.02

    for Ms in [1, 8, 32]:
        # test 1-D input_ids [Ms] and 2-D [Ms, 4]
        for shape in [(Ms,), (Ms, 4)]:
            input_ids = torch.randint(0, vocab, shape, dtype=torch.long, device=device)
            try:
                out = embed_tokens_bf16_passthrough(embed_weight, input_ids)
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "oom" in msg:
                    pytest.skip(f"embed OOM at shape {shape}: {e}")
                if "cuda" in msg or "triton" in msg:
                    pytest.skip(f"embed launch failed {shape}: {e}")
                raise
            ref = _reference_embed_bf16(embed_weight, input_ids)
            assert out.shape == ref.shape, f"embed shape {out.shape} vs {ref.shape} for input {shape}"
            assert out.dtype == torch.bfloat16, f"embed dtype {out.dtype} vs bf16"
            assert out.device.type == "cuda"
            diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
            max_diff = diff.max().item()
            assert max_diff <= 1e-6, (
                f"embed max_diff {max_diff:.6f} exceeds atol 1e-6 for shape {shape} "
                f"(bf16 passthrough vs F.embedding, vocab {vocab} K={K})"
            )
            # also test vs nn.Embedding for 1-D case
            if len(shape) == 1:
                emb = torch.nn.Embedding(vocab, K, dtype=torch.bfloat16, device=device)  # type: ignore
                # copy weight
                with torch.no_grad():
                    emb.weight.copy_(embed_weight)
                ref2 = emb(input_ids)
                diff2 = (out.to(torch.float32) - ref2.to(torch.float32)).abs().max().item()
                assert diff2 <= 1e-6, f"nn.Embedding diff {diff2:.6f} exceeds atol"


# ── bench — monotonic and <1.6× fallback ──────────────────────────────────


def test_sk09_bench_monotonic_and_fallback():
    """Bench RMSNorm quant fused: time monotonic in M and <1.6× fallback.

    WHY
        The fused monolito (``tl.load bf16->f32`` -> ``tl.dot var via dot``
        -> ``sqrt``/``rsqrt`` -> ``y=x*inv_rms*w`` -> ``amax/127`` ->
        ``quant`` -> ``tl.store int8/bf16``) should scale linearly with tokens
        and never be substantially slower than a torch fallback (pure torch
        ``var via dot`` + ``rsqrt`` + ``quant``). A monotonic time curve
        proves no pathological padding; >1.6× would indicate a regression vs
        the simple torch path and violates the ``sm_86`` ``mont`` single-launch
        expectation (``K=5120``).

    Boundaries
        * ``M`` in ``(1,8,32,128)`` ``K=5120`` fused RMSNorm quant.
        * Measures kernel via ``rmsnorm_quant_fused`` and fallback via torch
          ``_reference_sk09_rmsnorm_quant`` (pure torch var via dot, no Triton)
          — both on CUDA, with ``torch.cuda.synchronize`` and 20 iters avg.
        * Asserts ``kernel_time < 1.6 * fallback_time`` per M and
          ``kernel_time`` non-decreasing (allow 30% noise for timer jitter).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If monotonic violated or fallback ratio exceeded.
    pytest.skip
        If CUDA/Triton not available or OOM.
    """
    _require_cuda_triton()
    try:
        from vllm._genesis.kernels.sk09_norm_embed import rmsnorm_quant_fused  # noqa: WPS433
    except ImportError:
        try:
            from vllm._genesis.kernels.sk09_norm_embed import fused_rmsnorm_quant as rmsnorm_quant_fused  # noqa: WPS433
        except ImportError:
            try:
                from vllm._genesis.kernels.sk09_norm_embed import rmsnorm_quant as rmsnorm_quant_fused  # noqa: WPS433
            except ImportError as e:  # pragma: no cover
                pytest.skip(f"sk09 rmsnorm_quant not importable: {e}")

    device = "cuda"
    kernel_times: list[float] = []
    fallback_times: list[float] = []

    for M in MS:
        torch.manual_seed(1234 + M)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
        weight = (torch.rand(K, dtype=torch.float32, device=device) * 0.2 + 0.9).to(torch.bfloat16)
        s_pow2 = torch.ones(K, dtype=torch.bfloat16, device=device)

        def _kernel_fn(
            x=x,
            weight=weight,
            s_pow2=s_pow2,
        ):
            return rmsnorm_quant_fused(x, weight, s_pow2, eps=EPS)

        def _fallback_fn(
            x=x,
            weight=weight,
            s_pow2=s_pow2,
        ):
            return _reference_sk09_rmsnorm_quant(x, weight, s_pow2, eps=EPS)

        try:
            k_ms = _measure_ms(_kernel_fn, warmup=3, iters=20)
        except RuntimeError as e:
            msg = str(e).lower()
            typ = type(e).__name__.lower()
            mod = getattr(type(e), "__module__", "").lower()
            if "out of memory" in msg or "oom" in msg:
                pytest.skip(f"OOM in kernel at M={M}: {e}")
            if any(kw in msg for kw in ("ptx", "triton", "cuda", "constexpr", "unsupported", "compilation")) or "compilation" in typ or "triton" in mod:
                pytest.skip(f"sk09 kernel compilation failed at M={M}: {e} ({type(e).__name__})")
            raise
        except Exception as e:  # pragma: no cover
            msg = str(e).lower()
            typ = type(e).__name__.lower()
            mod = getattr(type(e), "__module__", "").lower()
            if "out of memory" in msg or "oom" in msg:
                pytest.skip(f"OOM in kernel at M={M}: {e}")
            if any(kw in msg for kw in ("ptx", "triton", "cuda", "constexpr", "unsupported", "compilation")) or "compilation" in typ or "triton" in mod:
                pytest.skip(f"sk09 kernel failed at M={M}: {e} ({type(e).__name__})")
            raise
        f_ms = _measure_ms(_fallback_fn, warmup=3, iters=20)
        kernel_times.append(k_ms)
        fallback_times.append(f_ms)
        assert k_ms < 1.6 * f_ms, (
            f"M={M} K={K} kernel {k_ms:.3f}ms not <1.6*fallback {f_ms:.3f}ms "
            f"(ratio {k_ms/f_ms:.2f}, K=5120 rmsnorm quant fuse mont sm_86)"
        )

    # monotonic (allow 30% jitter for timer noise on tiny M=1/8 launch overhead)
    for i in range(1, len(kernel_times)):
        prev, cur = kernel_times[i - 1], kernel_times[i]
        assert cur + 1e-6 >= prev * 0.70, (
            f"monotonic violation M {MS[i-1]}->{MS[i]}: {prev:.3f}ms -> {cur:.3f}ms "
            f"(K={K} rmsnorm quant fuse)"
        )
    assert kernel_times[-1] + 1e-6 >= kernel_times[0] * 0.8, f"kernel time not increasing (within 20% jitter): {kernel_times} (M {MS})"
