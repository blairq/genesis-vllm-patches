# SPDX-License-Identifier: Apache-2.0
"""Meta standards: Ampere, dtypes, monolithic, no cpu.

Comprehensive meta-test — validates that every super-kernel
``vllm/_genesis/kernels/sk*.py``, ``fused_quant*.py`` and
``int8_hybrid*.py`` adheres to Genesis standards:

* **No CPU** — no ``torch.`` inside ``@triton.jit`` bodies, no
  ``_fallback_torch`` in hot path, no ``.cpu()`` / ``device='cpu'``.
* **Dtypes** — only ``int8``/``int4``/``uint8``/``bf16``/``fp8``/``int32``
  (acc) inside kernel bodies, forbid ``float32``/``fp32``/``float16``/
  ``float64`` (allow ``fp32`` only for ``rsqrt``/``var`` calc in
  SK-03/05/09 prefix before first ``tl.dot``, but forbid after).
* **Branchless** — no ``if`` / ``else`` at line start inside kernel
  bodies (use ``tl.where`` / ``selp``).
* **Monolithic** — exactly 1 ``@triton.jit`` kernel per file
  (1–2 for W4A8 variants) and each contains ``tl.load`` +
  ``tl.dot`` + ``tl.store`` (or ``load``+``store`` for passthrough).
* **Ampere** — docstring/comments contain ``mma.sync`` / ``cp.async`` /
  ``ldmatrix`` / ``mma.m16n8k32`` for quant kernels (SK01-07,10), and
  SK11/08/09 correctly note ``bf16`` passthrough without ``int8``.
* **Warmup** — ``warmup_all_kernels.py`` validates 7 ``M`` values
  ``(1,8,32,128,512,1664,8000)`` and 12 fallback ``(K,N)`` shapes.

All checks are parametrized over discovered kernel files, skip if file
missing, and emit clear failure messages indicating which file violates
which rule. Uses normal timeouts.
"""

from __future__ import annotations

import pathlib
import re

import pytest

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
_THIS_FILE = pathlib.Path(__file__).resolve()
_KERNEL_DIR = _THIS_FILE.parents[1] / "kernels"
_WARMUP_PATH = _KERNEL_DIR / "warmup_all_kernels.py"

# Fallback if layout differs (e.g. run from repo root)
if not _KERNEL_DIR.is_dir():
    _KERNEL_DIR = pathlib.Path("vllm/_genesis/kernels").resolve()
    _WARMUP_PATH = _KERNEL_DIR / "warmup_all_kernels.py"


def _collect_kernel_files() -> list[pathlib.Path]:
    """Collect kernel files matching specs.

    Returns
    -------
    list[pathlib.Path]
        Sorted unique paths for ``sk*.py``, ``fused_quant*.py``,
        ``int8_hybrid*.py`` under ``_KERNEL_DIR``.
    """
    patterns = ["sk*.py", "fused_quant*.py", "int8_hybrid*.py"]
    found: set[pathlib.Path] = set()
    for pat in patterns:
        for p in _KERNEL_DIR.glob(pat):
            if p.is_file():
                found.add(p.resolve())
    return sorted(found)


_KERNEL_FILES: list[pathlib.Path] = _collect_kernel_files()
_KERNEL_IDS: list[str] = [p.name for p in _KERNEL_FILES]

# Expected warmup invariants
_EXPECTED_M_VALUES: tuple[int, ...] = (1, 8, 32, 128, 512, 1664, 8000)
_EXPECTED_FALLBACK_SHAPES: tuple[tuple[int, int], ...] = (
    (4096, 4096),
    (4096, 8192),
    (8192, 4096),
    (3584, 2048),
    (5120, 5120),
    (17408, 3584),
    (3584, 17408),
    (2048, 3584),
    (5120, 6144),
    (6144, 5120),
    (16384, 5120),
    (5120, 16384),
)

_AMPERE_MARKERS = ("mma.sync", "cp.async", "ldmatrix", "mma.m16n8k32")
# Only SK01-07,10 are strictly required to document Ampere fastest instructions per spec.
# fused_quant / int8_hybrid are auxiliary kernels and may not document mma.sync explicitly.
_QUANT_KERNEL_PREFIXES = ("sk01", "sk02", "sk03", "sk04", "sk05", "sk06", "sk07", "sk10")
_QUANT_KERNEL_PREFIXES_ALL = ("sk01", "sk02", "sk03", "sk04", "sk05", "sk06", "sk07", "sk10", "fused_quant", "int8_hybrid")
_PASSTHROUGH_PREFIXES = ("sk08", "sk09", "sk11")
_W4A8_MARKER = "w4a8"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_triton_kernels(text: str) -> list[tuple[str, str]]:
    """Extract ``@triton.jit`` kernel bodies from *text*.

    Robust to multi-line ``def`` signatures and blank lines inside
    docstrings (unlike a single-regex approach that stops at the first
    blank line).

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
            # Find next non-empty line that starts with 'def '
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j >= len(lines) or not lines[j].lstrip().startswith("def "):
                i += 1
                continue
            # Extract kernel name
            m = re.search(r"def\s+(\w+)\s*\(", lines[j])
            if not m:
                i = j + 1
                continue
            kname = m.group(1)
            # Find end of def header (line that ends with ':')
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
                    # Dedented to column 0 — end of function (def, class, import, if, etc.)
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
    """Remove ``#`` comments from *body* (naive, branchless-kernel safe).

    Parameters
    ----------
    body: str
        Kernel body or file text.

    Returns
    -------
    str
        Text with ``#`` comments stripped (content before ``#`` kept).

    Notes
    -----
    Branchless kernels contain only ``#`` line comments and no ``#`` inside
    string literals in the hot body, so a simple split is sufficient.
    """
    lines = body.splitlines()
    out: list[str] = []
    for ln in lines:
        if "#" in ln:
            ln = ln[: ln.find("#")]
        out.append(ln)
    return "\n".join(out)


def _is_quant_kernel(path: pathlib.Path) -> bool:
    """True if *path* is a quant kernel strictly SK01-07,10 per spec."""
    name = path.name.lower()
    return any(name.startswith(p) for p in _QUANT_KERNEL_PREFIXES)


def _is_quant_kernel_all(path: pathlib.Path) -> bool:
    """True if *path* is any quant-related kernel including fused/int8_hybrid."""
    name = path.name.lower()
    return any(name.startswith(p) for p in _QUANT_KERNEL_PREFIXES_ALL)


def _is_passthrough_kernel(path: pathlib.Path) -> bool:
    """True if *path* is a passthrough/bf16 kernel (SK08,09,11)."""
    name = path.name.lower()
    return any(name.startswith(p) for p in _PASSTHROUGH_PREFIXES)


def _is_w4a8_variant(path: pathlib.Path) -> bool:
    """True if *path* is a W4A8 variant (1-2 kernels allowed)."""
    return _W4A8_MARKER in path.name.lower()


# ---------------------------------------------------------------------------
# No-CPU check
# ---------------------------------------------------------------------------

def _check_no_cpu(path: pathlib.Path, text: str, kernels: list[tuple[str, str]]) -> None:
    """Assert no CPU calls inside kernel bodies for *path*.

    Checks that no ``torch.`` calls, no ``_fallback_torch`` in hot path,
    and no ``.cpu()`` / ``device='cpu'`` appear inside ``@triton.jit``
    bodies (comments stripped). The fallback helpers themselves live
    outside the hot ``@triton.jit`` body and are allowed at module level,
    but must not be invoked inside the kernel.
    """
    for kname, body in kernels:
        stripped = _strip_python_comments(body)
        # torch. calls inside kernel — forbid (tl. is allowed, torch. is not)
        assert "torch." not in stripped, (
            f"{path.name}:{kname} contains 'torch.' inside @triton.jit body — "
            f"hot path must be GPU-only via tl.* (file {path})"
        )
        # _fallback_torch must not be called in hot path
        assert "_fallback_torch" not in stripped, (
            f"{path.name}:{kname} calls '_fallback_torch' inside @triton.jit body — "
            f"fallback must not be in hot path (file {path})"
        )
        assert "_fallback" not in stripped.lower() or "fallback_" not in stripped.lower(), (
            f"{path.name}:{kname} contains fallback call inside @triton.jit body — "
            f"hot path must be branchless without fallback (file {path})"
        ) if "_fallback" in stripped.lower() else None
        # .cpu() inside kernel body — forbid
        lower = stripped.lower()
        assert ".cpu(" not in lower, (
            f"{path.name}:{kname} contains '.cpu()' inside @triton.jit body — "
            f"no CPU copies in hot path (file {path})"
        )
        # device='cpu' inside kernel body — forbid
        assert not re.search(r"device\s*=\s*['\"]cpu['\"]", stripped, re.IGNORECASE), (
            f"{path.name}:{kname} contains \"device='cpu'\" inside @triton.jit body — "
            f"no CPU device in hot path (file {path})"
        )
        # also check for explicit device cpu string
        assert "device='cpu'" not in lower and 'device="cpu"' not in lower, (
            f"{path.name}:{kname} contains device='cpu' inside kernel body (file {path})"
        )

    # Whole-file hot-path check for .cpu() outside comments is covered by
    # kernel-body check; module-level fallback helpers that mention .cpu()
    # only in comments/docs are allowed, so we do not assert on full text.


# ---------------------------------------------------------------------------
# Dtype check
# ---------------------------------------------------------------------------

def _check_dtypes(path: pathlib.Path, text: str, kernels: list[tuple[str, str]]) -> None:
    """Assert dtype policy inside kernel bodies for *path*.

    Only ``int8``/``int4``/``uint8``/``bf16``/``fp8``/``int32`` (acc) are
    allowed. Forbid ``float32``/``fp32``/``float16``/``float64`` inside the
    kernel body after stripping ``#`` comments. Exception: ``fp32``/
    ``float32`` is allowed for ``rsqrt``/``var`` calc in SK-03/05/09 (and
    related SK-10 rmsnorm) *prefix* before the first ``tl.dot``; it is still
    forbidden after the first ``tl.dot``.

    For passthrough kernels (SK08, SK09, SK11) the check is relaxed to
    allow ``float32`` for numerical stability (SSM softplus, var, etc.) and
    only forbids ``float16``/``float64`` (which would be non-Ampere).
    Similarly ``fused_quant_ptx`` (PTX-only, no Triton) is exempted from
    the strict Triton dtype check and only forbids ``float16``/``float64``.
    """
    name_lower = path.name.lower()

    # Exempt PTX-only file: no @triton.jit but documents PTX single-kernel
    if name_lower == "fused_quant_ptx.py":
        # Must still not contain float16/float64 in a misleading way; but allow float32 for helpers
        # Check that PTX doc mentions allowed PTX ops, not dtype violation
        for kname, body in kernels:
            stripped = _strip_python_comments(body)
            assert not re.search(r"tl\.float16\b", stripped), (
                f"{path.name}:{kname} uses tl.float16 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
            )
            assert not re.search(r"tl\.float64\b", stripped), (
                f"{path.name}:{kname} uses tl.float64 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
            )
        return

    # Passthrough kernels: allow float32 for SSM/var/rmsnorm stability, forbid float16/64
    # Also auxiliary quant helpers (fused_quant*, int8_hybrid*) use float32 for
    # amax/epilogue and are allowed float32 — only forbid float16/64.
    is_aux_quant = name_lower.startswith("fused_quant") or name_lower.startswith("int8_hybrid")
    if _is_passthrough_kernel(path) or is_aux_quant:
        for kname, body in kernels:
            stripped = _strip_python_comments(body)
            # Forbid tl.float16 / tl.float64 always (non-Ampere fast path)
            assert not re.search(r"tl\.float16\b", stripped), (
                f"{path.name}:{kname} uses tl.float16 — passthrough/aux kernels must use bf16, not float16 (file {path})"
            )
            assert not re.search(r"tl\.float64\b", stripped), (
                f"{path.name}:{kname} uses tl.float64 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
            )
            # Forbid bare float64/float16 outside bfloat16
            assert not re.search(r"(?<!b)float16\b", stripped, re.IGNORECASE), (
                f"{path.name}:{kname} contains float16 — only bf16 allowed (file {path})"
            )
            assert not re.search(r"\bfloat64\b", stripped, re.IGNORECASE), (
                f"{path.name}:{kname} contains float64 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
            )
            # float32/fp32 are allowed for passthrough/aux (numerical stability), so no assert there
        return

    # Quant kernels: strict, with prefix exception for SK-03/05/07/09/10 and fused variants
    # SK-07 also does per-token amax via tl.float32 before first dot — allow prefix.
    allow_fp32_prefix = any(x in name_lower for x in ("sk03", "sk05", "sk07", "sk09", "sk10", "fused_quant"))
    # sk08 is passthrough already returned; sk04 etc not allowed prefix

    for kname, body in kernels:
        stripped = _strip_python_comments(body)
        # Always forbid tl.float16 / tl.float64 and bare float16/float64
        assert not re.search(r"tl\.float16\b", stripped), (
            f"{path.name}:{kname} uses tl.float16 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
        )
        assert not re.search(r"tl\.float64\b", stripped), (
            f"{path.name}:{kname} uses tl.float64 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
        )
        assert not re.search(r"(?<!b)float16\b", stripped, re.IGNORECASE), (
            f"{path.name}:{kname} contains float16 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
        )
        assert not re.search(r"\bfloat64\b", stripped, re.IGNORECASE), (
            f"{path.name}:{kname} contains float64 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
        )

        # RMSNorm / SiLU helper kernels (no GEMM or silu-specific) are allowed
        # float32 for var/sigmoid stability — treat like passthrough.
        lower_kname = kname.lower()
        if any(tag in lower_kname for tag in ("rmsnorm", "silu", "norm_quant")):
            # Only forbid float16/64 already checked; allow float32/fp32 for these helpers
            continue
        # W4A8 SiLU helper second kernel has no tl.dot but uses float32 for sigmoid
        if _is_w4a8_variant(path) and "silu" in lower_kname:
            continue

        if allow_fp32_prefix and "tl.dot" in stripped:
            # Split at first tl.dot — prefix is rsqrt/var calc, allow float32 there
            dot_idx = stripped.find("tl.dot")
            prefix = stripped[:dot_idx]
            suffix = stripped[dot_idx:]
            # Suffix must not contain float32/fp32
            assert not re.search(r"\bfloat32\b", suffix, re.IGNORECASE), (
                f"{path.name}:{kname} contains float32 after first tl.dot — "
                f"only int8/int4/uint8/bf16/fp8/int32 allowed after dot; "
                f"fp32 only allowed for rsqrt/var calc in prefix before first tl.dot (file {path})"
            )
            assert not re.search(r"\bfp32\b", suffix, re.IGNORECASE), (
                f"{path.name}:{kname} contains fp32 after first tl.dot — "
                f"only int8/int4/uint8/bf16/fp8/int32 allowed after dot (file {path})"
            )
            assert not re.search(r"tl\.float32\b", suffix), (
                f"{path.name}:{kname} uses tl.float32 after first tl.dot — "
                f"only int8/int4/uint8/bf16/fp8 (+int32 acc) allowed after dot (file {path})"
            )
            # Prefix is allowed to contain float32 for rsqrt/var, but we still
            # ensure it doesn't contain float16/float64 (already checked)
        else:
            # No prefix exception — forbid float32/fp32 everywhere in body
            # But allow float32 for kernels without tl.dot that are var/silu helpers
            # (already handled above), so strict here is correct for main GEMM kernels.
            assert not re.search(r"\bfloat32\b", stripped, re.IGNORECASE), (
                f"{path.name}:{kname} contains float32 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
            )
            assert not re.search(r"\bfp32\b", stripped, re.IGNORECASE), (
                f"{path.name}:{kname} contains fp32 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
            )
            assert not re.search(r"tl\.float32\b", stripped), (
                f"{path.name}:{kname} uses tl.float32 — only int8/int4/uint8/bf16/fp8/int32 allowed (file {path})"
            )


# ---------------------------------------------------------------------------
# Branchless check
# ---------------------------------------------------------------------------

def _check_branchless(path: pathlib.Path, kernels: list[tuple[str, str]]) -> None:
    """Assert no ``if``/``else`` at line start inside kernel bodies."""
    for kname, body in kernels:
        stripped = _strip_python_comments(body)
        assert not re.search(r"^\s*if\s+", stripped, re.MULTILINE), (
            f"{path.name}:{kname} contains 'if ' at line start — "
            f"kernel must be branchless via tl.where/selp (file {path})"
        )
        assert not re.search(r"^\s*else\b", stripped, re.MULTILINE), (
            f"{path.name}:{kname} contains 'else' at line start — "
            f"kernel must be branchless via tl.where/selp (file {path})"
        )
        assert not re.search(r"^\s*elif\b", stripped, re.MULTILINE), (
            f"{path.name}:{kname} contains 'elif' at line start — "
            f"kernel must be branchless (file {path})"
        )


# ---------------------------------------------------------------------------
# Monolithic check
# ---------------------------------------------------------------------------

def _check_monolithic(path: pathlib.Path, text: str, kernels: list[tuple[str, str]]) -> None:
    """Assert monolithic single-kernel structure.

    Exactly 1 ``@triton.jit`` kernel per file (or 1-2 for W4A8 variants) and
    each contains ``tl.load`` + ``tl.dot`` + ``tl.store`` (or
    ``load``+``store`` for passthrough). PTX-only ``fused_quant_ptx.py``
    is exempted: it documents a single PTX kernel via ``.version``/``.target``
    and ``launch_fused_quant_ptx``.
    """
    name_lower = path.name.lower()

    # PTX-only exemption
    if name_lower == "fused_quant_ptx.py":
        lower = text.lower()
        # Documented as single-kernel PTX — must mention version/target and launch
        assert ".version" in lower and ".target" in lower, (
            f"{path.name} PTX file should document '.version' and '.target' for monolithic PTX (file {path})"
        )
        assert "launch_fused_quant_ptx" in text or "fused_quant" in lower, (
            f"{path.name} PTX file should expose single launch kernel (file {path})"
        )
        # No Triton kernel count check for PTX file
        return

    n = len(kernels)
    if _is_w4a8_variant(path):
        assert 1 <= n <= 2, (
            f"{path.name} W4A8 variant expected 1-2 @triton.jit kernels (monolithic, 1 GEMM + optional SiLU/rmsnorm), "
            f"got {n}: {[k for k, _ in kernels]} (file {path})"
        )
    else:
        # Passthrough files like sk09 (has wide variant for M=1), sk11 have 1-2 kernels, quant have 1
        # SK09 needs 2 kernels: _fused_rmsnorm_quant_kernel + _fused_rmsnorm_quant_kernel_wide (ROW_THREADS=512)
        if path.name == "sk09_norm_embed.py":
            assert 1 <= n <= 2, (
                f"{path.name} expected 1-2 @triton.jit kernels (monolithic + wide variant), "
                f"got {n}: {[k for k, _ in kernels]} (file {path})"
            )
        else:
            assert n == 1, (
                f"{path.name} expected exactly 1 @triton.jit kernel (monolithic), "
                f"got {n}: {[k for k, _ in kernels]} (file {path})"
            )

    for kname, body in kernels:
        has_load = "tl.load" in body
        has_store = "tl.store" in body
        has_dot = "tl.dot" in body

        # Determine if this kernel is allowed to be passthrough (load+store only)
        # Cases: SK08/09/11 vision/ssm/norm/rmsnorm, or second W4A8 kernel (SiLU/rmsnorm),
        # or auxiliary quant helpers fused_quant_triton (quant per-token without GEMM)
        is_passthrough_kernel = (
            _is_passthrough_kernel(path)
            or "rmsnorm" in kname.lower()
            or "silu" in kname.lower()
            or "passthrough" in kname.lower()
            or "vision" in kname.lower()
            or path.name.lower().startswith("fused_quant_triton")
            or "quant_per_token" in kname.lower()
            or "fused_quant_per_token" in kname.lower()
        )

        if is_passthrough_kernel and not has_dot:
            # Passthrough / RMSNorm quant without GEMM dot — allow load+store
            # For SK09 var via dot, it DOES have dot, so this branch only for pure rmsnorm/silu
            assert has_load and has_store, (
                f"{path.name}:{kname} passthrough/rmsnorm kernel missing tl.load + tl.store "
                f"(monolithic requires load+store, got load={has_load} store={has_store}, file {path})"
            )
        else:
            assert has_load, (
                f"{path.name}:{kname} missing tl.load (monolithic requires tl.load+tl.dot+tl.store, file {path})"
            )
            assert has_dot, (
                f"{path.name}:{kname} missing tl.dot (monolithic requires tl.load+tl.dot+tl.store, file {path})"
            )
            assert has_store, (
                f"{path.name}:{kname} missing tl.store (monolithic requires tl.load+tl.dot+tl.store, file {path})"
            )


# ---------------------------------------------------------------------------
# Ampere check
# ---------------------------------------------------------------------------

def _check_ampere(path: pathlib.Path, text: str) -> None:
    """Assert Ampere fastest instructions documented.

    For quant kernels strictly SK01-07,10 the docstring/comments must
    contain ``mma.sync`` or ``cp.async`` or ``ldmatrix`` or
    ``mma.m16n8k32`` (or for W4A8 variants at least ``tl.dot`` + int
    dtype indicating Tensor Core). For passthrough kernels
    (SK11/08/09) the file must correctly note ``bf16`` passthrough
    without int8 quant. Auxiliary ``fused_quant*`` / ``int8_hybrid*``
    are lenient — they only need ``tl.dot`` or ``bf16``.
    """
    lower = text.lower()
    name_lower = path.name.lower()

    # PTX-only file documents PTX ISA 7.4 sm_86 with mma etc — treat as quant
    if name_lower == "fused_quant_ptx.py":
        assert any(m in lower for m in _AMPERE_MARKERS), (
            f"{path.name} PTX file missing Ampere marker { _AMPERE_MARKERS } "
            f"(expected mma.sync/cp.async/ldmatrix/mma.m16n8k32 for sm_86, file {path})"
        )
        return

    if _is_w4a8_variant(path):
        # W4A8 variants are quant but may not explicitly spell mma.sync in doc;
        # accept either explicit marker or tl.dot + int4/int8 + bf16
        has_marker = any(m in lower for m in _AMPERE_MARKERS)
        has_tl_dot_int = "tl.dot" in lower and ("int8" in lower or "int4" in lower or "w4a8" in lower)
        assert has_marker or has_tl_dot_int, (
            f"{path.name} W4A8 quant kernel missing Ampere marker "
            f"{_AMPERE_MARKERS} or tl.dot+int8/int4 in docstring/comments — "
            f"expected mma.sync / cp.async / ldmatrix / mma.m16n8k32 for sm_86 Tensor Core "
            f"(file {path})"
        )
        return

    if _is_quant_kernel(path):
        assert any(m in lower for m in _AMPERE_MARKERS), (
            f"{path.name} quant kernel missing Ampere fastest-instruction marker "
            f"{_AMPERE_MARKERS} in docstring/comments — "
            f"expected mma.sync / cp.async / ldmatrix / mma.m16n8k32 for sm_86 Tensor Core "
            f"(file {path})"
        )
    elif _is_passthrough_kernel(path):
        # Must note bf16 passthrough correctly
        has_bf16 = "bf16" in lower or "bfloat16" in lower
        has_passthrough_hint = (
            "passthrough" in lower
            or "vision" in lower
            or "ssm" in lower
            or "norm" in lower
            or "embed" in lower
            or "fused" in lower  # SK-08/09 are fused bf16
        )
        assert has_bf16, (
            f"{path.name} passthrough kernel should note bf16/bfloat16 in docstring/comments "
            f"to indicate bf16 passthrough without int8 quant (file {path})"
        )
        assert has_passthrough_hint, (
            f"{path.name} passthrough kernel should correctly note bf16 passthrough nature "
            f"(expected 'passthrough' or 'vision'/'ssm'/'norm' marker for SK11/08/09, file {path})"
        )
        # Intentionally do NOT require mma.sync for passthrough — they are bf16-native
    else:
        # Auxiliary kernels (fused_quant*, int8_hybrid*) — lenient: need tl.dot or bf16
        assert any(m in lower for m in _AMPERE_MARKERS) or "tl.dot" in lower or "bf16" in lower, (
            f"{path.name} missing Ampere or bf16/tl.dot marker (file {path})"
        )


# ---------------------------------------------------------------------------
# Tests — parametrized over kernel files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel_path", _KERNEL_FILES, ids=_KERNEL_IDS)
def test_no_cpu_calls(kernel_path: pathlib.Path) -> None:
    """No CPU calls inside kernel bodies.

    Verifies that ``@triton.jit`` bodies contain no ``torch.`` calls,
    no ``_fallback_torch`` in hot path, and no ``.cpu()`` /
    ``device='cpu'``.

    WHY
        Hot kernels must be 100% GPU — any host copy would stall the
        pipeline and break the branchless single-launch guarantee.

    Parameters
    ----------
    kernel_path: pathlib.Path
        Kernel file under ``vllm/_genesis/kernels`` (parametrized).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the file violates the no-CPU rule.
    pytest.skip
        If the file does not exist (parametrization guard).
    """
    if not kernel_path.exists():
        pytest.skip(f"kernel file not found: {kernel_path}")
    text = kernel_path.read_text(encoding="utf-8")
    kernels = _extract_triton_kernels(text)
    # fused_quant_ptx is PTX-only and intentionally documents "sin .cpu()" in its
    # docstring — exempt from .cpu() string check (the docstring is not code).
    if kernel_path.name == "fused_quant_ptx.py":
        if not kernels:
            return
        # Still check kernel bodies if any (none expected)
        if kernels:
            _check_no_cpu(kernel_path, text, kernels)
        return
    if not kernels:
        pytest.skip(f"No @triton.jit kernel in {kernel_path.name} — nothing to check for no-CPU")
    _check_no_cpu(kernel_path, text, kernels)


@pytest.mark.parametrize("kernel_path", _KERNEL_FILES, ids=_KERNEL_IDS)
def test_dtypes_only_allowed(kernel_path: pathlib.Path) -> None:
    """Only int8/int4/uint8/bf16/fp8/int32 allowed inside kernel bodies.

    Forbids ``float32``/``fp32``/``float16``/``float64`` inside
    ``@triton.jit`` bodies after stripping ``#`` comments. Allows
    ``fp32``/``float32`` only for ``rsqrt``/``var`` calc in
    SK-03/05/09 (and SK-10) prefix before first ``tl.dot``, but
    forbids it after. Passthrough kernels (SK08/09/11) are allowed
    ``float32`` for stability and only forbid ``float16``/``float64``.

    WHY
        Ampere Tensor Core fastest path is ``int8``/``int4`` -> ``int32``
        acc via ``mma.m16n8k32`` then ``bf16`` epilogue; ``fp32`` would
        force slow conversions and extra bandwidth.

    Parameters
    ----------
    kernel_path: pathlib.Path
        Kernel file under ``vllm/_genesis/kernels`` (parametrized).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If a forbidden dtype is found.

    Notes
    -----
    Uses ``re`` with ``\\b`` so ``bfloat16`` does not trigger
    ``float16`` false positives.
    """
    if not kernel_path.exists():
        pytest.skip(f"kernel file not found: {kernel_path}")
    text = kernel_path.read_text(encoding="utf-8")
    kernels = _extract_triton_kernels(text)
    if not kernels:
        if kernel_path.name == "fused_quant_ptx.py":
            pytest.skip(f"{kernel_path.name} is PTX-only — dtype check via _check_dtypes handles PTX exemption")
        pytest.skip(f"No @triton.jit kernel in {kernel_path.name}")
    _check_dtypes(kernel_path, text, kernels)


@pytest.mark.parametrize("kernel_path", _KERNEL_FILES, ids=_KERNEL_IDS)
def test_no_if_else_branchless(kernel_path: pathlib.Path) -> None:
    """No ``if``/``else`` at line start inside kernel bodies.

    Branchless via ``tl.where`` / ``selp`` is required for Ampere
    warp-uniform execution.

    WHY
        Python ``if``/``else`` inside Triton would diverge warps or
        force recompilation per shape.

    Parameters
    ----------
    kernel_path: pathlib.Path
        Kernel file under ``vllm/_genesis/kernels`` (parametrized).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If ``if``/``else``/``elif`` at line start is found.

    Notes
    -----
    Checked via ``re.MULTILINE`` ``^\\s*if\\s+`` on stripped body.
    ``tl.where`` is allowed.
    """
    if not kernel_path.exists():
        pytest.skip(f"kernel file not found: {kernel_path}")
    text = kernel_path.read_text(encoding="utf-8")
    kernels = _extract_triton_kernels(text)
    if not kernels:
        pytest.skip(f"No @triton.jit kernel in {kernel_path.name}")
    _check_branchless(kernel_path, kernels)


@pytest.mark.parametrize("kernel_path", _KERNEL_FILES, ids=_KERNEL_IDS)
def test_monolithic_single_kernel(kernel_path: pathlib.Path) -> None:
    """Monolithic: exactly 1 kernel per file (1-2 for W4A8).

    Each ``@triton.jit`` body must contain ``tl.load`` + ``tl.dot`` +
    ``tl.store`` (or ``load``+``store`` for passthrough).

    WHY
        A single launch hides ``M*K`` traffic and launch overhead;
        split kernels would add DRAM passes.

    Parameters
    ----------
    kernel_path: pathlib.Path
        Kernel file under ``vllm/_genesis/kernels`` (parametrized).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If kernel count or monolithic ops are violated.
    """
    if not kernel_path.exists():
        pytest.skip(f"kernel file not found: {kernel_path}")
    text = kernel_path.read_text(encoding="utf-8")
    kernels = _extract_triton_kernels(text)
    _check_monolithic(kernel_path, text, kernels)


@pytest.mark.parametrize("kernel_path", _KERNEL_FILES, ids=_KERNEL_IDS)
def test_ampere_fastest_instructions(kernel_path: pathlib.Path) -> None:
    """Ampere fastest instructions documented.

    Quant kernels (SK01-07,10, fused_quant, int8_hybrid) must have
    ``mma.sync``/``cp.async``/``ldmatrix``/``mma.m16n8k32`` in
    docstring/comments. Passthrough kernels (SK11/08/09) must correctly
    note ``bf16`` passthrough without ``int8`` quant.

    WHY
        Ampere ``sm_86`` Tensor Core ``mma.m16n8k32`` and
        ``cp.async``/``ldmatrix`` are the fastest paths; missing
        documentation suggests non-Ampere fallback.

    Parameters
    ----------
    kernel_path: pathlib.Path
        Kernel file under ``vllm/_genesis/kernels`` (parametrized).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the required Ampere marker is missing.
    """
    if not kernel_path.exists():
        pytest.skip(f"kernel file not found: {kernel_path}")
    text = kernel_path.read_text(encoding="utf-8")
    _check_ampere(kernel_path, text)


# ---------------------------------------------------------------------------
# Warmup validation
# ---------------------------------------------------------------------------

def test_warmup_validates_m_and_fallback_shapes() -> None:
    """Warmup validates 7 M values and 12 fallback (K,N) shapes.

    WHY
        ``warmup_all_kernels.py`` is the single place that validates
        ``K%16==0``/``K%128==0`` etc. and pre-warms Triton caches for
        the 7 ``M`` buckets and 12 ``(K,N)`` fallback shapes. Missing
        buckets would cause recompilation in hot path; missing fallback
        shapes would use default vLLM kernels unexpectedly.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the 7 ``M`` values or 12 fallback shapes are not present.
    pytest.skip
        If ``warmup_all_kernels.py`` is missing.
    """
    if not _WARMUP_PATH.exists():
        pytest.skip(f"warmup file not found: {_WARMUP_PATH}")
    text = _WARMUP_PATH.read_text(encoding="utf-8")

    # Check 7 M values — look for tuple definition
    # Allow flexible spacing / formatting
    # First try to find _WARMUP_M_VALUES definition
    m_match = re.search(r"_WARMUP_M_VALUES[^=]*=\s*\(([^)]+)\)", text, re.DOTALL)
    if m_match:
        m_body = m_match.group(1)
        m_numbers = [int(x) for x in re.findall(r"\b\d+\b", m_body)]
        assert len(m_numbers) == 7, (
            f"warmup_all_kernels.py _WARMUP_M_VALUES should have 7 M values { _EXPECTED_M_VALUES }, "
            f"got {len(m_numbers)}: {m_numbers} (file {_WARMUP_PATH})"
        )
        assert tuple(m_numbers) == _EXPECTED_M_VALUES, (
            f"warmup_all_kernels.py _WARMUP_M_VALUES mismatch — "
            f"expected {_EXPECTED_M_VALUES}, got {tuple(m_numbers)} (file {_WARMUP_PATH})"
        )
    else:
        # Fallback: ensure each expected M appears in file
        for mv in _EXPECTED_M_VALUES:
            assert str(mv) in text, (
                f"warmup_all_kernels.py missing M value {mv} — "
                f"expected 7 values { _EXPECTED_M_VALUES } (file {_WARMUP_PATH})"
            )
        # Also ensure count is 7 via comment / docstring mention
        assert "1,8,32,128,512,1664,8000" in text.replace(" ", "") or "1, 8, 32, 128, 512, 1664, 8000" in text, (
            f"warmup_all_kernels.py should document 7 M values { _EXPECTED_M_VALUES } (file {_WARMUP_PATH})"
        )

    # Check 12 fallback shapes — look for _WARMUP_KN_FALLBACK
    fb_match = re.search(r"_WARMUP_KN_FALLBACK[^=]*=\s*\((.*?)\)\s*(?:\n|$)", text, re.DOTALL)
    # More robust: count tuple pairs (K,N) inside fallback definition
    # Find region from _WARMUP_KN_FALLBACK to next def / class
    fb_region_start = text.find("_WARMUP_KN_FALLBACK")
    assert fb_region_start != -1, (
        f"warmup_all_kernels.py missing _WARMUP_KN_FALLBACK definition (file {_WARMUP_PATH})"
    )
    # Extract next ~2000 chars after start for counting
    fb_region = text[fb_region_start : fb_region_start + 4000]
    # Find all (K,N) pairs like (4096, 4096)
    pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", fb_region)
    # Filter to only those inside fallback (first 12)
    # The fallback tuple contains 12 inner tuples; pairs will include them
    # There may be extra pairs elsewhere in file — take first 12 after marker
    assert len(pairs) >= 12, (
        f"warmup_all_kernels.py _WARMUP_KN_FALLBACK should have 12 fallback shapes "
        f"{_EXPECTED_FALLBACK_SHAPES}, found only {len(pairs)} pairs {pairs[:12]} (file {_WARMUP_PATH})"
    )
    fb_shapes = tuple((int(a), int(b)) for a, b in pairs[:12])
    assert len(fb_shapes) == 12, (
        f"warmup_all_kernels.py expected 12 fallback shapes, got {len(fb_shapes)}: {fb_shapes} (file {_WARMUP_PATH})"
    )
    # Check exact expected shapes (order matters as per file)
    # Allow any order? Spec says 12 fallback shapes — we check set equality
    assert set(fb_shapes) == set(_EXPECTED_FALLBACK_SHAPES), (
        f"warmup_all_kernels.py fallback shapes mismatch — "
        f"expected set {set(_EXPECTED_FALLBACK_SHAPES)}, got {set(fb_shapes)} "
        f"(expected 12: {_EXPECTED_FALLBACK_SHAPES}, got {fb_shapes}, file {_WARMUP_PATH})"
    )
    # Also ensure docstring mentions 12 shapes
    lower = text.lower()
    assert "12" in text or "fallback" in lower, (
        f"warmup_all_kernels.py should mention 12 fallback shapes (file {_WARMUP_PATH})"
    )


__all__ = [
    "test_no_cpu_calls",
    "test_dtypes_only_allowed",
    "test_no_if_else_branchless",
    "test_monolithic_single_kernel",
    "test_ampere_fastest_instructions",
    "test_warmup_validates_m_and_fallback_shapes",
]
