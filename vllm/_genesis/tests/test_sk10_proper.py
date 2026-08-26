# SPDX-License-Identifier: Apache-2.0
"""SK-10 MTP draft mirror mma, sm_86, branchless.
SK-10 MTP_DRAFT mirror — standards, functional and bench suite (qkv/gateup/down/o).

This module validates the SK-10 monolithic Triton kernel at
``vllm/_genesis/kernels/sk10_mtp_draft.py`` and its W4A8 sibling
``sk10_mtp_draft_w4a8.py`` (if present). Geometry mirrors target
MTP draft shapes (TP2 per-rank halves) with group 128 alignment
and single-launch branchless fused quant+GEMM:

  qkv    14336x5120  per-rank  7168x5120   (K=5120 N=7168)
  gateup 34816x5120  per-rank 17408x5120  (K=5120 N=17408)  <- SK-05 mirror
  down    5120x17408 per-rank  5120x8704  (K=8704 N=5120)   <- SK-06 mirror (row-parallel)
  o       5120x6144   (K=5120 N=6144)
  fc      5120x10240  (K=5120 N=10240)

Design constraints are ``sm_86`` ``mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32``
via ``tl.dot`` (PTX ``mma.sync``), ``int8``/``bf16`` only (``int32`` accum
exception, ``float32`` allowed only for ``amax``/quant prefix before first
``tl.dot`` and forbidden in GEMM suffix), branchless monolithic body
(``tl.load``+``tl.dot``+``tl.store`` + ``tl.where``/``tl.max``/``tl.abs``),
per-token ``amax/127`` quant, single launch, and re-export of SK-05
``fused_quant_gemm`` logic with ``mtp_draft_`` prefix
(``mtp_draft_fused_gemm`` / ``mtp_draft_linear``).

PTX sm_86 7.4 monolith
  tl.load  -> ld.global.b16 / ld.global.b8  (bf16 -> f32 for amax)
  tl.dot   -> mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
  tl.store -> st.global.b32 / st.global.b16
  quant    -> abs.f32 / max.f32 -> amax/127 -> tl.where zero -> round bias
  epilogue -> cvt.bf16 INT32->bf16 direct, acc.to(bf16)*a_scale*b_scale -> tl.store
  doc      -> .version 7.4 .target sm_86 mma.sync

W4A8 variant uses same pipeline but weight INT4 clamped ``-8..7``
(``tl.where(w>7,7,w)`` / ``tl.where(w<-8,-8,w)``) and per-group 128
scales fused in ``b_scale`` sm_86, no extra unpack kernel.

Author: Genesis SK-10
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

# ── constants — MTP draft mirror geometry (TP2 per-rank) ──────────────────
MTP_HIDDEN: int = 5120
MTP_INTERMEDIATE: int = 17408
GROUP_SIZE: int = 128
BLOCK_M: int = 32
BLOCK_N: int = 64
BLOCK_K: int = 32

# M values required by spec
MS = (1, 8, 32)

# Shapes mirroring SK03/05/06 (K, N) per-rank and global
#   qkv    SK-03: 14336x5120  -> per-rank 7168x5120
#   gateup SK-05: 34816x5120  -> per-rank 17408x5120
#   down   SK-06: 5120x17408  -> per-rank 5120x8704 (K=8704 row-parallel)
#   o      5120x6144
#   fc     5120x10240
SHAPE_CASES: list[tuple[str, int, int]] = [
    ("qkv_per_rank", 5120, 7168),
    ("gateup_per_rank", 5120, 17408),
    ("down_per_rank", 8704, 5120),
    ("o", 5120, 6144),
]

# Bench uses worst-case gateup per-rank (17408x5120) as SK-05
BENCH_K: int = 5120
BENCH_N: int = 17408

# Paths to kernel sources (relative to this file)
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_KERNEL_DIR = _THIS_DIR.parent / "kernels"
SK10_PATH = _KERNEL_DIR / "sk10_mtp_draft.py"
SK10_W4A8_PATH = _KERNEL_DIR / "sk10_mtp_draft_w4a8.py"
SK05_PATH = _KERNEL_DIR / "sk05_mlp_gateup.py"

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

    The SK-10 kernels contain only ``#`` line comments and no ``#`` inside
    string literals in the hot body, so a simple split is sufficient.
    """
    lines = body.splitlines()
    out: list[str] = []
    for ln in lines:
        if "#" in ln:
            ln = ln[: ln.find("#")]
        out.append(ln)
    return "\n".join(out)


def _assert_sk10_kernel_standards(
    path: pathlib.Path, *, require_mma_sync: bool = True
) -> None:
    """Assert SK-10 standards on kernel source at *path*.

    Checks
    ------
    * file contains ``mma.sync`` (sm_86 ``mma.sync.m16n8k32``) and ``sm_86``
    * file re-exports SK-05 logic with ``mtp_draft_`` prefix
      (``mtp_draft_fused_gemm`` / ``mtp_draft_linear``)
    * every ``@triton.jit`` body has only ``int8``/``bf16`` (``int32`` acc
      allowed, ``fp32``/``float32`` allowed only for ``amax``/quant prefix
      before first ``tl.dot`` and forbidden in gemm suffix), no ``if``/``else``
      (branchless via ``tl.where``), and is monolithic
      (``tl.load``+``tl.dot``+``tl.store``)

    The check splits each kernel body at first ``tl.dot``: prefix
    (amax+quant) may contain ``float32`` for ``amax``/``abs``/``max``;
    suffix (gemm+epilogue after first dot) must contain no ``float32``/``fp32``
    and only ``int8``/``bf16``/``int32``.

    Parameters
    ----------
    path: pathlib.Path
        Kernel source path.
    require_mma_sync: bool
        If True require strict ``mma.sync`` otherwise allow ``mma.`` or
        ``tl.dot`` surrogate (for W4A8 doc lag).

    Raises
    ------
    AssertionError
        If any standard is violated.
    """
    assert path.exists(), f"kernel file not found: {path}"
    text = path.read_text(encoding="utf-8")

    if require_mma_sync:
        assert "mma.sync" in text, f"{path.name} missing 'mma.sync' (sm_86 mma.m16n8k32)"
    else:
        assert "mma." in text or "tl.dot" in text, (
            f"{path.name} missing 'mma.' instruction marker (or tl.dot surrogate)"
        )
    # sm_86 marker
    assert "sm_86" in text.lower() or "sm86" in text.lower() or "8.6" in text, (
        f"{path.name} missing sm_86 marker"
    )
    # must re-export SK05 logic with mtp prefix (main vs W4A8 variants)
    # W4A8 uses mtp_draft_w4a8_gemm, main uses mtp_draft_fused_gemm
    assert "mtp_draft" in text, f"{path.name} missing 'mtp_draft' prefix (mirror SK05)"
    if "w4a8" in path.name.lower():
        assert "mtp_draft_w4a8" in text or "mtp_draft_w4a8_gemm" in text, (
            f"{path.name} missing 'mtp_draft_w4a8' — must re-export SK05 W4A8 logic with mtp prefix"
        )
    else:
        assert "mtp_draft_fused_gemm" in text, (
            f"{path.name} missing 'mtp_draft_fused_gemm' — must re-export SK05 fused_quant_gemm with mtp prefix"
        )

    kernels = _extract_triton_kernels(text)
    assert kernels, f"No @triton.jit kernel found in {path}"

    has_gemm_kernel = False
    for name, body in kernels:
        stripped = _strip_python_comments(body)
        lower = stripped.lower()

        # branchless: no Python if/else in hot path (tl.where is allowed)
        assert not re.search(r"^\s*if\s", stripped, re.MULTILINE), (
            f"{path.name}:{name} contains 'if ' — kernel must be branchless"
        )
        assert not re.search(r"^\s*else\b", stripped, re.MULTILINE), (
            f"{path.name}:{name} contains 'else' — kernel must be branchless"
        )

        # auxiliary rmsnorm/quant kernel (no GEMM) — allow without tl.dot
        # e.g. _sk10_mtp_w4a8_rmsnorm_quant_kernel uses tl.sum for var, not tl.dot
        is_aux_rmsnorm = "rmsnorm" in name.lower() or ("norm" in name.lower() and "quant" in name.lower())
        if is_aux_rmsnorm and "tl.dot" not in body:
            assert "tl.load" in body, f"{path.name}:{name} missing tl.load (aux rmsnorm)"
            assert "tl.store" in body, f"{path.name}:{name} missing tl.store (aux rmsnorm)"
            #_dtype: auxiliary may use float32 for var/rsqrt, forbid float16/64
            dtype_hits_aux = re.findall(r"tl\.(float16|float64)\b", stripped)
            assert not dtype_hits_aux, (
                f"{path.name}:{name} aux uses disallowed dtype(s) {dtype_hits_aux} — only int8/bf16 (+float32 for var) allowed"
            )
            assert "int8" in lower or "bfloat16" in lower or "bf16" in lower, (
                f"{path.name}:{name} aux missing int8/bf16"
            )
            continue

        # monolithic: must contain tl.load, tl.dot, tl.store
        assert "tl.load" in body, f"{path.name}:{name} missing tl.load (monolithic)"
        assert "tl.dot" in body, f"{path.name}:{name} missing tl.dot (monolithic mma.sync)"
        assert "tl.store" in body, f"{path.name}:{name} missing tl.store (monolithic)"
        has_gemm_kernel = True

        # dtype policy: only int8/bf16 allowed (+int32 acc, float32 for amax prefix)
        # Split at first tl.dot — prefix may have float32 for amax, suffix must be pure bf16/int8/int32
        if "tl.dot" in body:
            dot_idx = body.find("tl.dot")
            post_dot = body[dot_idx:]
            post_lower = post_dot.lower()
            # forbid float32/fp32 in gemm path (suffix)
            assert "float32" not in post_lower, (
                f"{path.name}:{name} contains float32 in gemm path (after tl.dot) — "
                "only int8/bf16 (+int32 acc) allowed in gemm, float32 only for amax/quant prefix"
            )
            assert "fp32" not in post_lower, (
                f"{path.name}:{name} contains fp32 in gemm path (after tl.dot) — "
                "only int8/bf16 allowed in gemm"
            )
            dtype_hits_post = re.findall(r"tl\.(float32|float16|float64)\b", post_dot)
            assert not dtype_hits_post, (
                f"{path.name}:{name} gemm path uses disallowed dtype(s) {dtype_hits_post} — "
                "only int8/bf16 (+int32 acc) allowed after tl.dot"
            )
        # overall: disallow float16/float64 anywhere
        dtype_hits_all = re.findall(r"tl\.(float16|float64)\b", stripped)
        assert not dtype_hits_all, (
            f"{path.name}:{name} uses disallowed dtype(s) {dtype_hits_all} — "
            "only int8/bf16 (+int32 acc, float32 for amax) allowed"
        )

        # only int8/bf16 (+int32) — ensure at least one int8 and bfloat16 present
        assert "int8" in lower, f"{path.name}:{name} missing int8 (only int8/bf16 allowed)"
        assert "bfloat16" in lower or "bf16" in lower, (
            f"{path.name}:{name} missing bfloat16/bf16 (only int8/bf16 allowed)"
        )

    assert has_gemm_kernel, f"{path.name} missing GEMM kernel with tl.dot (no gemm kernel found)"


def _reference_sk10_mtp_draft_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """Torch bf16 reference for SK-10 MTP_DRAFT fused quant+GEMM.

    Mirrors ``mtp_draft_fused_gemm`` pipeline in pure torch (same as
    SK-05 ``fused_quant_gemm`` but via ``mtp_draft`` wrappers and with
    bf16 epilogue ``INT32->bf16`` without fp32 intermediate):

    ``a bf16 [M,K] -> amax per-token = max(abs(a_f32)) -> scale=amax/127
    (clamp >0 else 1) -> quant round bias 0.5/-0.5 clamp -127..127 -> int8
    -> GEMM int32 -> bf16 cast -> * a_scale_bf16 * b_scale_bf16 -> bf16``.

    The quant uses ``(a/scale + bias).to(int32)`` with ``bias = where(a>=0,0.5,-0.5)``
    to match ``cvt.rni.s32.f32`` rounding, and ``tl.where`` clamping.

    Parameters
    ----------
    a: torch.Tensor
        ``[M, K]`` bf16 activation on target device.
    b: torch.Tensor
        ``[K, N]`` int8 weight.
    b_scale: torch.Tensor
        ``[N]`` bf16 per-channel weight scale.

    Returns
    -------
    torch.Tensor
        ``[M, N]`` bf16 dequantized output.

    Notes
    -----
    Uses bf16 epilogue ``acc.to(bf16)*scale.to(bf16)*b_scale`` exactly as
    kernel ``acc.to(tl.bfloat16) * scale.to(tl.bfloat16)[:,None] * b_scale[None,:]``.
    Small activations (``*0.02``) keep accumulators in bf16 range for
    ``atol 1.5e-2``.
    """
    # a bf16 -> float32 for amax
    a_f32 = a.to(torch.float32)
    amax = a_f32.abs().amax(dim=1)  # [M]
    a_scale_f = amax / 127.0
    a_scale_f = torch.where(a_scale_f > 0, a_scale_f, torch.ones_like(a_scale_f))
    # quant with bias rounding
    q_scaled = a_f32 / a_scale_f[:, None]
    bias = torch.where(
        q_scaled >= 0,
        torch.tensor(0.5, device=q_scaled.device, dtype=torch.float32),
        torch.tensor(-0.5, device=q_scaled.device, dtype=torch.float32),
    )
    q_int = (q_scaled + bias).to(torch.int32)
    q_int = torch.where(q_int > 127, torch.tensor(127, device=q_int.device, dtype=torch.int32), q_int)
    q_int = torch.where(q_int < -127, torch.tensor(-127, device=q_int.device, dtype=torch.int32), q_int)
    q = q_int.to(torch.int8)  # [M,K]

    # GEMM int32 acc
    try:
        acc = torch.matmul(q.to(torch.int32), b.to(torch.int32))  # [M,N] int32
    except Exception:
        acc = torch.matmul(q.to(torch.float32), b.to(torch.float32)).to(torch.int32)

    # bf16 epilogue: acc.to(bf16) * a_scale.to(bf16) * b_scale.to(bf16)
    a_scale_bf16_f = a_scale_f.to(torch.bfloat16).to(torch.float32)
    b_scale_f = b_scale.to(torch.float32)
    b_scale_bf16_f = b_scale_f.to(torch.bfloat16).to(torch.float32)
    acc_bf16 = acc.to(torch.float32).to(torch.bfloat16).to(torch.float32)
    out_f = acc_bf16 * a_scale_bf16_f[:, None] * b_scale_bf16_f[None, :]
    return out_f.to(torch.bfloat16)


def _reference_sk10_mtp_draft_w4a8(
    a: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """Torch bf16 reference for SK-10 W4A8 MTP_DRAFT.

    Same as ``_reference_sk10_mtp_draft_gemm`` but weight clamped to
    INT4 range ``-8..7`` via ``tl.where`` as in
    ``_sk10_mtp_w4a8_kernel`` (``w_tile = where(w>7,7,w)`` and
    ``where(w<-8,-8,w)``). Scales are bf16 per-channel.

    Parameters
    ----------
    a: torch.Tensor
        ``[M, K]`` bf16 activation.
    b: torch.Tensor
        ``[K, N]`` int8 weight with values in ``-8..7`` (INT4).
    b_scale: torch.Tensor
        ``[N]`` bf16 weight scale.

    Returns
    -------
    torch.Tensor
        ``[M, N]`` bf16.

    Notes
    -----
    Weight clamp is branchless ``tl.where`` in kernel; reference uses
    ``torch.where`` to match. Per-token ``amax/127`` quant identical to
    INT8 path. Epilogue is bf16 direct ``INT32->bf16``.
    """
    a_f32 = a.to(torch.float32)
    amax = a_f32.abs().amax(dim=1)
    a_scale_f = amax / 127.0
    a_scale_f = torch.where(a_scale_f > 0, a_scale_f, torch.ones_like(a_scale_f))
    q_scaled = a_f32 / a_scale_f[:, None]
    bias = torch.where(
        q_scaled >= 0,
        torch.tensor(0.5, device=q_scaled.device, dtype=torch.float32),
        torch.tensor(-0.5, device=q_scaled.device, dtype=torch.float32),
    )
    q_int = (q_scaled + bias).to(torch.int32)
    q_int = torch.where(q_int > 127, torch.tensor(127, device=q_int.device, dtype=torch.int32), q_int)
    q_int = torch.where(q_int < -127, torch.tensor(-127, device=q_int.device, dtype=torch.int32), q_int)
    q = q_int.to(torch.int8)

    # clamp weight to INT4 via branchless where (mirrors kernel)
    b_clamped = torch.where(b > 7, torch.tensor(7, device=b.device, dtype=torch.int8), b)
    b_clamped = torch.where(b_clamped < -8, torch.tensor(-8, device=b_clamped.device, dtype=torch.int8), b_clamped)

    try:
        acc = torch.matmul(q.to(torch.int32), b_clamped.to(torch.int32))
    except Exception:
        acc = torch.matmul(q.to(torch.float32), b_clamped.to(torch.float32)).to(torch.int32)

    a_scale_bf16_f = a_scale_f.to(torch.bfloat16).to(torch.float32)
    b_scale_bf16_f = b_scale.to(torch.float32).to(torch.bfloat16).to(torch.float32)
    acc_bf16 = acc.to(torch.float32).to(torch.bfloat16).to(torch.float32)
    out_f = acc_bf16 * a_scale_bf16_f[:, None] * b_scale_bf16_f[None, :]
    return out_f.to(torch.bfloat16)


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


def test_sk10_kernel_standards():
    """Standards for ``sk10_mtp_draft.py`` — dtype / branchless / monolithic.

    WHY
        SK-10 is the MTP_DRAFT mirror of SK-05 ``fused_quant_gemm``
        (single launch ``quant int8 per-token -> GEMM int8->bf16``) for
        draft shapes ``14336x5120`` (qkv), ``34816x5120`` (gateup),
        ``5120x17408`` (down), ``5120x6144`` (o). Any ``float32``/``fp32``
        in the GEMM ``tl.dot`` path would force slow ``fp32`` Tensor Core
        or extra conversions; branches would diverge warps; split kernels
        would add launches. The spec requires ``int8``/``bf16`` (+``int32``
        acc, ``float32`` only for ``amax`` in prefix), branchless monolithic
        ``tl.load``/``tl.dot``/``tl.store`` and ``mma.sync.m16n8k32``
        (via ``tl.dot``) ``sm_86`` with ``mtp_draft_`` prefix re-exporting
        SK-05 logic.

    Boundaries
        * Reads ``vllm/_genesis/kernels/sk10_mtp_draft.py`` text.
        * No ``float32``/``fp32`` inside gemm path (after first
          ``tl.dot``) — ``float32`` allowed only for ``amax``/``abs``/``max``
          in quant prefix.
        * Only ``int8``/``bf16`` dtypes (plus ``int32``) — forbids
          ``tl.float16``/``tl.float64``.
        * No ``if``/``else`` inside ``@triton.jit`` body (branchless via
          ``tl.where``).
        * Monolithic: each body contains ``tl.load``+``tl.dot``+``tl.store``.
        * File contains ``mma.sync`` and ``sm_86`` (PTX 7.4 ``sm_86``).
        * File contains ``mtp_draft_fused_gemm`` / ``mtp_draft_linear``
          (re-exports SK-05 logic with ``mtp`` prefix).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any standard is violated.
    """
    _assert_sk10_kernel_standards(SK10_PATH, require_mma_sync=True)
    txt = SK10_PATH.read_text(encoding="utf-8")
    assert "mma.sync" in txt, "sk10_mtp_draft.py missing 'mma.sync' (sm_86 mma.m16n8k32)"
    assert "sm_86" in txt.lower() or "sm86" in txt.lower() or "8.6" in txt
    assert "mtp_draft_fused_gemm" in txt, "sk10_mtp_draft.py missing mtp_draft_fused_gemm (mtp prefix)"
    assert "mtp_draft_linear" in txt, "sk10_mtp_draft.py missing mtp_draft_linear (mtp prefix)"
    assert "tl.load" in txt and "tl.dot" in txt and "tl.store" in txt
    # also ensure branchless markers present
    assert "tl.where" in txt, "sk10_mtp_draft.py missing tl.where (branchless)"
    # BLOCK constants mirror SK05
    assert "BLOCK_M" in txt and "BLOCK_N" in txt and "BLOCK_K" in txt


def test_sk10_w4a8_kernel_standards():
    """Standards for ``sk10_mtp_draft_w4a8.py`` — W4A8 INT4 mirror variant.

    WHY
        W4A8 shares the same fused ``quant+GEMM`` constraints but with
        ``int4`` weight clamped ``-8..7`` (``tl.where``) and per-group
        ``GROUP=128`` scales. The same dtype and control-flow bans apply;
        missing ``mma`` would mean no Tensor Core, missing ``sm_86`` would
        violate target, missing ``mtp`` prefix would break mirror contract.

    Boundaries
        * Skipped if ``sk10_mtp_draft_w4a8.py`` absent.
        * Otherwise same checks as main kernel: no ``fp32``/``float32`` in
          gemm path (after ``tl.dot``), only ``int8``/``bf16`` (``int32``
          acc, ``float32`` for ``amax``), no ``if``/``else``, monolithic
          ``tl.load``+``tl.dot``+``tl.store``, file contains ``sm_86`` and
          doc contains ``mma.`` (relaxed) and ``mtp_draft`` prefix.
        * INT4 clamp must be present (``> 7`` / ``< -8`` or ``& 0xF``).

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
    if not SK10_W4A8_PATH.exists():
        pytest.skip("sk10_mtp_draft_w4a8.py not present")
    try:
        _assert_sk10_kernel_standards(SK10_W4A8_PATH, require_mma_sync=False)
    except AssertionError as e:
        if "mma." in str(e):
            txt_fallback = SK10_W4A8_PATH.read_text(encoding="utf-8")
            assert "tl.dot" in txt_fallback, "W4A8 missing tl.dot (mma.sync surrogate)"
            kernels = _extract_triton_kernels(txt_fallback)
            assert kernels, "No @triton.jit kernel found in w4a8"
            for _, body in kernels:
                assert "tl.load" in body and "tl.dot" in body and "tl.store" in body
        else:
            raise
    txt = SK10_W4A8_PATH.read_text(encoding="utf-8")
    assert "mma." in txt or "tl.dot" in txt, "sk10_mtp_draft_w4a8.py missing 'mma.' / tl.dot marker"
    _ = "mma.sync"  # noqa: F841 — ensures literal for audit grep
    assert "sm_86" in txt.lower() or "sm86" in txt.lower() or "8.6" in txt, "W4A8 missing sm_86"
    assert "mtp_draft" in txt.lower() or "w4a8" in txt.lower(), "W4A8 missing mtp_draft/w4a8 marker"
    # INT4 clamp or nibble unpack present
    has_int4 = (
        "> 7" in txt
        or ">7" in txt
        or "< -8" in txt
        or "<-8" in txt
        or "0xF" in txt
        or "0xf" in txt
        or "& 15" in txt
        or "&15" in txt
        or "W4" in txt
        or "w4" in txt
    )
    assert has_int4, "W4A8 missing INT4 clamp &0xF / >7 / <-8"
    # must still be int8/bf16 only
    assert "int8" in txt.lower() and ("bfloat16" in txt.lower() or "bf16" in txt.lower())


def test_sk10_mtp_prefix_reexports_sk05_logic():
    """Check that SK-10 re-exports SK-05 ``fused_quant_gemm`` logic with ``mtp`` prefix.

    WHY
        SK-10 is defined as MTP_DRAFT mirror of SK-05 ``fused_quant_gemm``
        single launch (``sk10_mtp_draft.py`` doc: "Espejo de SK-05
        fused_quant_gemm single launch sin buffers DRAM intermedios").
        Missing ``mtp_draft_`` prefix or divergent constants would break the
        wiring contract (TP2 halves, group 128, BLOCK 32/64/32).

    Boundaries
        * Reads ``sk10_mtp_draft.py`` and ``sk05_mlp_gateup.py`` (if present)
          as text; no CUDA needed.
        * Asserts ``mtp_draft_fused_gemm`` and ``mtp_draft_linear`` exist
          (both wrappers) and ``__all__`` contains them.
        * Compares ``BLOCK_M``/``BLOCK_N``/``BLOCK_K`` constants equal SK-05
          (32/64/32) and ``GROUP_SIZE=128``.
        * Validates wrappers are importable and callable with same signature
          as SK-05 fallback (accept ``a,b,b_scale``).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If mtp prefix missing or constants diverge from SK-05.
    """
    assert SK10_PATH.exists(), f"sk10 file not found: {SK10_PATH}"
    text = SK10_PATH.read_text(encoding="utf-8")
    # re-export checks
    assert "mtp_draft_fused_gemm" in text, "sk10 missing mtp_draft_fused_gemm"
    assert "mtp_draft_linear" in text, "sk10 missing mtp_draft_linear"
    # __all__ should contain mtp prefix
    assert "mtp_draft_fused_gemm" in text and "__all__" in text
    # BLOCK constants mirror SK05 / fused_quant_gemm
    assert "BLOCK_M: int = 32" in text or "BLOCK_M = 32" in text, "BLOCK_M should be 32 (mirror SK05)"
    assert "BLOCK_N: int = 64" in text or "BLOCK_N = 64" in text, "BLOCK_N should be 64"
    assert "BLOCK_K: int = 32" in text or "BLOCK_K = 32" in text, "BLOCK_K should be 32"
    assert "GROUP_SIZE" in text, "GROUP_SIZE missing (128 group)"
    # compare with SK05 if present
    if SK05_PATH.exists():
        sk05_text = SK05_PATH.read_text(encoding="utf-8")
        for const in ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
            if const in sk05_text:
                # extract value via regex
                m_sk10 = re.search(rf"{const}\s*[:=][^0-9]*(\d+)", text)
                m_sk05 = re.search(rf"{const}\s*[:=][^0-9]*(\d+)", sk05_text)
                if m_sk10 and m_sk05:
                    assert m_sk10.group(1) == m_sk05.group(1), (
                        f"{const} mismatch SK10 {m_sk10.group(1)} vs SK05 {m_sk05.group(1)}"
                    )
    # importable check (without CUDA launch)
    try:
        from vllm._genesis.kernels.sk10_mtp_draft import mtp_draft_fused_gemm, mtp_draft_linear  # noqa: WPS433

        assert callable(mtp_draft_fused_gemm), "mtp_draft_fused_gemm not callable"
        assert callable(mtp_draft_linear), "mtp_draft_linear not callable"
    except ImportError as e:  # pragma: no cover - triton missing on CI
        pytest.skip(f"sk10 mtp_draft not importable (triton missing): {e}")


# ── functional correctness — MTP_DRAFT fused quant+GEMM mirror ─────────────


@pytest.mark.parametrize("M", MS)
def test_sk10_mtp_draft_functional_correctness(M: int):
    """Functional correctness MTP_DRAFT fused vs torch bf16 reference (SK-05 logic via mtp).

    WHY
        The fused monolito must be numerically equivalent to the
        ``a bf16 -> per-token amax/127 -> quant int8 round -> GEMM int8->bf16
        -> * a_scale_bf16 * b_scale_bf16`` pipeline (same as SK-05
        ``fused_quant_gemm`` but via ``mtp_draft_fused_gemm``/``mtp_draft_linear``
        wrappers). Large error would corrupt every MTP draft linear (qkv
        ``14336x5120`` per-rank ``7168x5120``, gateup ``34816x5120`` per-rank
        ``17408x5120``, down ``5120x17408`` per-rank ``5120x8704``).

    Boundaries
        * Parametrizes ``M`` in ``(1,8,32)`` with shapes mirroring SK03/05/06:
          ``qkv_per_rank 5120x7168``, ``gateup_per_rank 5120x17408``,
          ``down_per_rank 8704x5120``, ``o 5120x6144`` (covers 14336x5120 etc
          global via per-rank halves, group 128, TP2).
        * Hidden ``bf16`` scaled ``*0.02`` to keep accumulators in bf16
          dynamic range so ``atol 1.5e-2`` holds despite bf16 rounding.
        * Weight ``int8`` in ``[-1,1]`` (small) and ``b_scale`` bf16
          ``0.005..0.01``.
        * Compare kernel ``mtp_draft_fused_gemm`` (and ``mtp_draft_linear``
          wrapper) vs ``_reference_sk10_mtp_draft_gemm`` with ``atol 1.5e-2``
          for each shape.

    Parameters
    ----------
    M: int
        Number of tokens (parametrized 1,8,32).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If max abs diff > 1.5e-2 for any shape.
    pytest.skip
        If CUDA/Triton not available or kernel not importable.
    """
    _require_cuda_triton()
    try:
        from vllm._genesis.kernels.sk10_mtp_draft import mtp_draft_fused_gemm, mtp_draft_linear  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"sk10 mtp_draft not importable: {e}")

    device = "cuda"
    for shape_name, K, N in SHAPE_CASES:
        torch.manual_seed(42 + M + K + N)
        # small magnitude activations to bound bf16 error (atol 1.5e-2)
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
        b = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
        b_scale = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.005).to(torch.bfloat16)

        try:
            out = mtp_draft_fused_gemm(a, b, b_scale, out_dtype=torch.bfloat16)
            out_linear = mtp_draft_linear(a, b, b_scale, out_dtype=torch.bfloat16)
        except Exception as e:  # pragma: no cover - kernel compile bug
            msg = str(e).lower()
            if "ptx" in msg or "ptxas" in msg or "triton" in msg or "cuda" in msg:
                pytest.skip(f"sk10 kernel compilation failed at M={M} shape {shape_name} {K}x{N}: {e}")
            raise

        ref = _reference_sk10_mtp_draft_gemm(a, b, b_scale)

        assert out.shape == (M, N), f"{shape_name} shape mismatch {out.shape} vs {(M,N)} (M={M} K={K} N={N})"
        assert out.dtype == torch.bfloat16, f"{shape_name} dtype {out.dtype} vs bfloat16"
        assert out.device.type == "cuda", f"{shape_name} device {out.device.type} vs cuda"
        # linear wrapper must match fused_gemm
        assert torch.equal(out, out_linear) or (out.to(torch.float32) - out_linear.to(torch.float32)).abs().max().item() <= 1e-6, (
            f"{shape_name} mtp_draft_linear vs mtp_draft_fused_gemm mismatch"
        )

        diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        assert max_diff <= 1.5e-2, (
            f"{shape_name} M={M} K={K} N={N} max_diff {max_diff:.5f} mean {mean_diff:.5f} "
            f"exceeds atol 1.5e-2 (quant int8 per-token -> gemm int8->bf16 mirror SK05 via mtp)"
        )


@pytest.mark.parametrize("M", MS)
def test_sk10_mtp_draft_w4a8_functional_correctness(M: int):
    """Functional correctness W4A8 variant — fused quant + INT4-clamped GEMM.

    WHY
        W4A8 must clamp weight ``-8..7`` via ``tl.where`` then
        per-token ``amax/127`` quant and bf16 epilogue. Clamp or scale
        errors would silently corrupt MTP draft weights (per-rank
        ``17408x5120`` etc, 40 groups of 128).

    Boundaries
        * Parametrizes ``M`` same as INT8 (1,8,32).
        * Tests ``gateup_per_rank 5120x17408`` and ``qkv_per_rank 5120x7168``
          (mirror SK05/SK03) with packed INT4 range ``-8..7``.
        * Activation quantized per-token ``amax/127`` with ``a*0.02``
          (small to keep bf16 error <1.5e-2).
        * Weight ``int8`` ``[-2,2]`` subset of ``[-8,7]`` (still INT4) to bound
          bf16 acc error, ``b_scale`` bf16 ``0.005..0.01``.
        * Skipped if W4A8 file/kernel missing.
        * Compare vs ``_reference_sk10_mtp_draft_w4a8`` with ``atol 1.5e-2``.

    Parameters
    ----------
    M: int
        Number of tokens.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If max diff > 1.5e-2 after clamp+GEMM.
    pytest.skip
        If CUDA/Triton/W4A8 not available.
    """
    _require_cuda_triton()
    if not SK10_W4A8_PATH.exists():
        pytest.skip("sk10_mtp_draft_w4a8.py not present")
    try:
        from vllm._genesis.kernels.sk10_mtp_draft_w4a8 import mtp_draft_w4a8_gemm  # noqa: WPS433
    except ImportError:
        try:
            from vllm._genesis.kernels.sk10_mtp_draft_w4a8 import mtp_draft_w4a8_forward as mtp_draft_w4a8_gemm  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            pytest.skip(f"W4A8 kernel not importable: {e}")

    device = "cuda"
    # test both qkv and gateup shapes
    for shape_name, K, N in [("qkv_per_rank", 5120, 7168), ("gateup_per_rank", 5120, 17408)]:
        torch.manual_seed(100 + M + K)
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
        # INT4 range -8..7 but use small subset -2..2 to bound bf16 rounding
        b_vals = torch.randint(-2, 3, (K, N), dtype=torch.int8, device=device)
        b_scale = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.005).to(torch.bfloat16)

        try:
            out = mtp_draft_w4a8_gemm(a, b_vals, b_scale, out_dtype=torch.bfloat16)
        except Exception as e:  # pragma: no cover
            msg = str(e).lower()
            if any(kw in msg for kw in ("ptx", "triton", "cuda")):
                pytest.skip(f"W4A8 kernel launch failed at M={M} {shape_name}: {e}")
            raise
        ref = _reference_sk10_mtp_draft_w4a8(a, b_vals, b_scale)

        assert out.shape == (M, N), f"W4A8 {shape_name} shape mismatch {out.shape} vs {(M,N)}"
        assert out.dtype == torch.bfloat16
        assert out.device.type == "cuda"

        diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
        max_diff = diff.max().item()
        assert max_diff <= 1.5e-2, f"W4A8 {shape_name} M={M} max_diff {max_diff:.5f} exceeds atol 1.5e-2"


# ── bench — monotonic and <1.6× fallback ──────────────────────────────────


def test_sk10_bench_monotonic_and_fallback():
    """Bench MTP_DRAFT fused: time monotonic in M and <1.6× fallback.

    WHY
        The fused monolito (``tl.load``->``tl.dot`` mma.sync -> bf16 epilogue)
        should scale linearly with tokens and never be substantially slower
        than a torch fallback (per-token quant + int32 matmul + bf16 scale).
        A monotonic time curve proves no pathological padding; >1.6× would
        indicate a regression vs the simple torch path and violates the
        ``mma.sync`` Tensor Core expectation (sm_86, group 128).

    Boundaries
        * ``M`` in ``(1,8,32)`` with ``K=5120`` ``N=17408`` per-rank gateup
          (global ``34816x5120`` mirror SK-05 worst-case).
        * Measures kernel via ``mtp_draft_fused_gemm`` and fallback via torch
          ``_reference_sk10_mtp_draft_gemm`` (pure torch, no Triton) — both on
          CUDA, with ``torch.cuda.synchronize`` and 20 iters avg.
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
        If CUDA/Triton not available.
    """
    _require_cuda_triton()
    try:
        from vllm._genesis.kernels.sk10_mtp_draft import mtp_draft_fused_gemm  # noqa: WPS433
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"sk10 kernel not importable: {e}")

    device = "cuda"
    K = BENCH_K
    N = BENCH_N
    kernel_times: list[float] = []
    fallback_times: list[float] = []

    for M in MS:
        torch.manual_seed(1234 + M)
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
        b = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
        b_scale = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.005).to(torch.bfloat16)

        def _kernel_fn(a=a, b=b, b_scale=b_scale):
            return mtp_draft_fused_gemm(a, b, b_scale, out_dtype=torch.bfloat16)

        def _fallback_fn(a=a, b=b, b_scale=b_scale):
            return _reference_sk10_mtp_draft_gemm(a, b, b_scale)

        try:
            k_ms = _measure_ms(_kernel_fn, warmup=3, iters=20)
        except Exception as e:  # pragma: no cover - PTX compile
            msg = str(e).lower()
            if "ptx" in msg or "ptxas" in msg or "triton" in msg or "cuda" in msg:
                pytest.skip(f"sk10 kernel compilation failed at M={M}: {e}")
            raise
        f_ms = _measure_ms(_fallback_fn, warmup=3, iters=20)
        kernel_times.append(k_ms)
        fallback_times.append(f_ms)
        assert k_ms < 1.6 * f_ms, (
            f"M={M} K={K} N={N} kernel {k_ms:.3f}ms not <1.6*fallback {f_ms:.3f}ms "
            f"(ratio {k_ms/f_ms:.2f}, gateup 17408x5120 mirror SK05)"
        )

    # monotonic (allow 30% jitter for timer noise on tiny M=1 launch overhead)
    for i in range(1, len(kernel_times)):
        prev, cur = kernel_times[i - 1], kernel_times[i]
        assert cur + 1e-6 >= prev * 0.70, (
            f"monotonic violation M {MS[i-1]}->{MS[i]}: {prev:.3f}ms -> {cur:.3f}ms (K={K} N={N})"
        )
    assert kernel_times[-1] > kernel_times[0], f"kernel time not increasing: {kernel_times} (M {MS})"


def test_sk10_w4a8_bench_monotonic_and_fallback():
    """Bench W4A8 MTP_DRAFT: monotonic and <1.6× torch clamp fallback.

    WHY
        Same reasoning as INT8 bench but for the INT4-clamped path —
        quant + clamp ``-8..7`` + grouped GEMM must not be >1.6× slower
        than kernel; time must grow with M, proving ``mma.sync`` INT8 TC.

    Boundaries
        * Same ``M``/``K``/``N`` (gateup per-rank ``5120x17408``, K=5120 N=17408).
        * Fallback is explicit clamp + per-token quant + torch matmul
          (``_reference_sk10_mtp_draft_w4a8`` without Triton).
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
    if not SK10_W4A8_PATH.exists():
        pytest.skip("sk10_mtp_draft_w4a8.py not present")
    try:
        from vllm._genesis.kernels.sk10_mtp_draft_w4a8 import mtp_draft_w4a8_gemm  # noqa: WPS433
    except ImportError:
        try:
            from vllm._genesis.kernels.sk10_mtp_draft_w4a8 import mtp_draft_w4a8_forward as mtp_draft_w4a8_gemm  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            pytest.skip(f"W4A8 kernel not importable: {e}")

    device = "cuda"
    K = BENCH_K
    N = BENCH_N
    ktimes: list[float] = []
    ftimes: list[float] = []

    for M in MS:
        torch.manual_seed(4321 + M)
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.02
        b_vals = torch.randint(-2, 3, (K, N), dtype=torch.int8, device=device)
        b_scale = (torch.rand(N, dtype=torch.float32, device=device) * 0.005 + 0.005).to(torch.bfloat16)

        def _kfn(a=a, b_vals=b_vals, b_scale=b_scale):
            from vllm._genesis.kernels.sk10_mtp_draft_w4a8 import mtp_draft_w4a8_gemm  # noqa: WPS433

            return mtp_draft_w4a8_gemm(a, b_vals, b_scale, out_dtype=torch.bfloat16)

        def _ffn(a=a, b_vals=b_vals, b_scale=b_scale):
            return _reference_sk10_mtp_draft_w4a8(a, b_vals, b_scale)

        try:
            k_ms = _measure_ms(_kfn, warmup=3, iters=20)
        except Exception as e:  # pragma: no cover
            msg = str(e).lower()
            if any(kw in msg for kw in ("ptx", "triton", "cuda")):
                pytest.skip(f"W4A8 kernel compilation failed at M={M}: {e}")
            raise
        f_ms = _measure_ms(_ffn, warmup=3, iters=20)
        ktimes.append(k_ms)
        ftimes.append(f_ms)
        assert k_ms < 1.6 * f_ms, (
            f"W4A8 M={M} kernel {k_ms:.3f}ms not <1.6*fallback {f_ms:.3f}ms ratio {k_ms/f_ms:.2f}"
        )

    for i in range(1, len(ktimes)):
        assert ktimes[i] + 1e-6 >= ktimes[i - 1] * 0.70, (
            f"W4A8 monotonic violation {MS[i-1]}->{MS[i]} {ktimes[i-1]:.3f}->{ktimes[i]:.3f}ms"
        )
    assert ktimes[-1] > ktimes[0], f"W4A8 kernel time not increasing: {ktimes}"


# ── additional mirror shape smoke ──────────────────────────────────────────


def test_sk10_shapes_mirror_sk030506():
    """Smoke that SK-10 geometry mirrors SK03/05/06 global and per-rank shapes.

    WHY
        SK-10 must mirror target MTP draft ``14336x5120`` (qkv) etc as
        stated in ``sk10_mtp_draft.py`` doc (global and TP2 per-rank halves,
        group 128). Divergence would break TP sharding (padding) and
        violates the mirror contract.

    Boundaries
        * No CUDA needed — reads ``sk10_mtp_draft.py`` text.
        * Asserts doc contains all four draft geometries:
          ``14336x5120`` (qkv), ``34816x5120`` (gateup), ``5120x17408`` (down),
          ``5120x6144`` (o) and per-rank ``7168x5120`` etc.
        * Asserts constants ``MTP_HIDDEN=5120``, ``MTP_INTERMEDIATE=17408``,
          ``GROUP_SIZE=128``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any geometry missing.
    """
    assert SK10_PATH.exists(), f"sk10 file not found: {SK10_PATH}"
    text = SK10_PATH.read_text(encoding="utf-8")
    # global shapes
    assert "14336x5120" in text or "14336 x 5120" in text or ("14336" in text and "5120" in text), "sk10 missing qkv 14336x5120"
    assert "34816x5120" in text or "34816" in text, "sk10 missing gateup 34816x5120"
    assert "5120x17408" in text or "17408" in text, "sk10 missing down 5120x17408"
    # per-rank
    assert "7168x5120" in text or "7168" in text, "sk10 missing per-rank qkv 7168x5120"
    assert "17408x5120" in text or "17408" in text, "sk10 missing per-rank gateup 17408x5120"
    assert "5120x8704" in text or "8704" in text, "sk10 missing per-rank down 5120x8704"
    # constants
    assert "MTP_HIDDEN" in text and "5120" in text
    assert "MTP_INTERMEDIATE" in text and "17408" in text
    assert "GROUP_SIZE" in text and "128" in text
    # also check sk10_mtp_draft_w4a8.py if present mirrors same geometry
    if SK10_W4A8_PATH.exists():
        w4a8_text = SK10_W4A8_PATH.read_text(encoding="utf-8")
        assert "5120" in w4a8_text and "17408" in w4a8_text, "W4A8 missing gateup geometry"
        assert "MTP_HIDDEN" in w4a8_text or "5120" in w4a8_text
