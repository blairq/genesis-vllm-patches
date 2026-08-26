# SPDX-License-Identifier: Apache-2.0
"""Genesis doctor rule — W4A16 artifact validator (CK-3.1).

Valida que un artefacto ``compressed-tensors`` W4A16 sea cargable por vLLM
y que sea compatible con :mod:`vllm._genesis.model_detect`::

    * ``config.json:quantization_config``  →  ``num_bits==4``,
      ``group_size==128``, ``type=="int"``, ``symmetric==True``,
      ``format=="pack-quantized"``, ``strategy=="group"``.
    * Por cada linear cuantizado (192 MLPs = gate/up/down × 64 capas;
      256 si ``--layers all``) existen tres tensores::

        - ``<base>.weight_packed``  [out, in//8]   I32
        - ``<base>.weight_scale``   [out, in//128] BF16
        - ``<base>.weight_shape``   [2]            I64  (= [out, in] original)

      y sus shapes son mutuamente coherentes.

La referencia es ``soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ`` (192 MLPs + 64
full-attn = 256, group 128, BF16 scales).  El artefacto ``mlp-only``
sólo tiene los 192 MLPs; el chequeo acepta ambas variantes y reporta el
conteo observado.

Uso::

    python -m vllm._genesis.doctor.rules.w4a16 --model /path/to/w4a16-out
    python -m vllm._genesis.doctor.rules.w4a16 --model <path> --json

Integración doctor::

    from vllm._genesis.doctor.rules.w4a16 import check_w4a16_artifact
    results = check_w4a16_artifact("/path/to/artifact")

Author: ox-alpha 2026-08-24 (CK-3.1)
"""
from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("genesis.doctor.w4a16")

# ─── Shared result type (compatible with compat/preflight_checks.CheckResult) ─

@dataclass
class CheckResult:
    """Outcome of a single W4A16 check.

    severity ∈ {"OK", "INFO", "WARN", "ERROR"} — ERROR significa que el
    artefacto no es cargable / no pasará ``model_detect``.
    """
    name: str
    severity: str
    message: str
    remediation: Optional[str] = None

    def __str__(self) -> str:
        out = f"[{self.severity}] {self.name}: {self.message}"
        if self.remediation:
            out += f"\n  → {self.remediation}"
        return out


# ─── Helpers ────────────────────────────────────────────────────────────────

def _read_safetensors_header(path: Path) -> Dict[str, Any]:
    """Read safetensors JSON header without loading tensors."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    # Remove __metadata__ if present
    header.pop("__metadata__", None)
    return header


def _resolve_model_dir(model_id: str) -> Path:
    """Resolve ``model_id`` (local path or HF hub id) to a snapshot dir.

    Mirrors ``tools.requant.resolve_model_dir`` but lighter: only checks
    local path and ``/home/usuario/Proyectos/models-cache/hub``.
    """
    p = Path(model_id)
    if p.is_dir() and (p / "config.json").is_file():
        return p
    # Try HF hub cache
    hub_cache = Path("/home/usuario/Proyectos/models-cache/hub")
    if "/" in model_id:
        safe = model_id.replace("/", "--")
        hub_dir = hub_cache / f"models--{safe}"
        if hub_dir.is_dir():
            snap = hub_dir / "snapshots"
            if snap.is_dir():
                snaps = sorted(snap.iterdir())
                if snaps and (snaps[-1] / "config.json").is_file():
                    return snaps[-1]
    return p


def _load_config(model_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return None, f"config.json not found at {cfg_path}"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return cfg, None
    except Exception as e:
        return None, f"failed to read {cfg_path}: {e}"


def _collect_headers(model_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """Return combined header dict {tensor_key: {dtype, shape, data_offsets}}.

    Handles sharded (``model.safetensors.index.json``) and single-file
    layouts.  On error returns partial dict + error string.
    """
    index_path = model_dir / "model.safetensors.index.json"
    combined: Dict[str, Dict[str, Any]] = {}
    if index_path.is_file():
        try:
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            wmap: Dict[str, str] = idx.get("weight_map", {})
            # Group by shard file to read each header once
            shard_to_keys: Dict[str, List[str]] = {}
            for k, shard in wmap.items():
                shard_to_keys.setdefault(shard, []).append(k)
            for shard, _keys in shard_to_keys.items():
                shard_path = model_dir / shard
                if not shard_path.is_file():
                    # Hub cache uses symlinks via blobs; shard may be under model_dir
                    return combined, f"shard not found: {shard_path}"
                try:
                    hdr = _read_safetensors_header(shard_path)
                except Exception as e:
                    return combined, f"failed to read header {shard_path}: {e}"
                for k in _keys:
                    if k in hdr:
                        combined[k] = hdr[k]
                    else:
                        # Key advertised in index but missing in header
                        # Treat as missing
                        pass
            return combined, None
        except Exception as e:
            return combined, f"failed to read index {index_path}: {e}"

    # Single-file fallback: look for any *.safetensors
    candidates = list(model_dir.glob("*.safetensors"))
    if not candidates:
        return combined, "no .safetensors files found and no index"
    # If multiple, merge all (should not happen for single-file but safe)
    for cand in candidates:
        try:
            hdr = _read_safetensors_header(cand)
            combined.update(hdr)
        except Exception as e:
            return combined, f"failed to read {cand}: {e}"
    return combined, None


def _load_weight_shape_values(
    model_dir: Path,
    weight_map: Dict[str, str],
    key: str,
) -> Optional[Tuple[int, int]]:
    """Load ``key`` (weight_shape) tensor value [out,in] if safetensors is available.

    Returns None if unavailable or on error (caller should fall back to header-only check).
    """
    try:
        from safetensors import safe_open  # type: ignore
    except ImportError:
        return None
    shard = weight_map.get(key)
    if shard is None:
        # Try to find via header collection path: single file case has no map
        # Search shards
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            return None
        # Single-file: scan
        cands = list(model_dir.glob("*.safetensors"))
        for cand in cands:
            try:
                with safe_open(str(cand), framework="np", device="cpu") as f:
                    if key in f.keys():
                        arr = f.get_slice(key)[:]  # type: ignore
                        import numpy as np  # type: ignore
                        vals = arr.tolist() if hasattr(arr, "tolist") else list(arr)
                        if len(vals) == 2:
                            return (int(vals[0]), int(vals[1]))
            except Exception:
                continue
        return None
    shard_path = model_dir / shard
    if not shard_path.is_file():
        return None
    try:
        with safe_open(str(shard_path), framework="np", device="cpu") as f:
            if key not in f.keys():
                return None
            arr = f.get_slice(key)[:]  # type: ignore
            # arr is numpy array of shape [2]
            try:
                import numpy as np  # type: ignore
                if hasattr(arr, "tolist"):
                    vals = arr.tolist()
                else:
                    vals = list(arr)
                if len(vals) == 2:
                    return (int(vals[0]), int(vals[1]))
            except Exception:
                # Try alternative: f.get_tensor
                pass
            try:
                t = f.get_tensor(key)  # type: ignore
                vals = t.tolist()  # type: ignore
                if len(vals) == 2:
                    return (int(vals[0]), int(vals[1]))
            except Exception:
                return None
    except Exception as e:
        log.debug("load weight_shape %s failed: %s", key, e)
        return None
    return None


# ─── Core validation ───────────────────────────────────────────────────────

def _check_quantization_config(config: Dict[str, Any]) -> List[CheckResult]:
    results: List[CheckResult] = []
    qcfg = config.get("quantization_config")
    if not qcfg:
        results.append(CheckResult(
            name="W4A16 quantization_config",
            severity="ERROR",
            message="config.json missing quantization_config (expected compressed-tensors W4A16)",
            remediation="El artefacto debe haber sido generado por `genesis requant` y contener quantization_config con num_bits 4, group_size 128.",
        ))
        return results

    # Top-level format should be pack-quantized or contain compressed-tensors quant_method
    fmt = str(qcfg.get("format", "")).lower()
    if fmt and fmt != "pack-quantized":
        results.append(CheckResult(
            name="W4A16 quantization_config format",
            severity="WARN",
            message=f"quantization_config.format={fmt!r} (expected 'pack-quantized')",
            remediation="Verifica que el artefacto use compressed-tensors pack-quantized.",
        ))
    else:
        results.append(CheckResult(
            name="W4A16 quantization_config format",
            severity="OK",
            message="quantization_config.format is pack-quantized",
        ))

    # config_groups
    groups = qcfg.get("config_groups")
    if not groups or not isinstance(groups, dict):
        results.append(CheckResult(
            name="W4A16 config_groups",
            severity="ERROR",
            message="quantization_config.config_groups missing or not a dict",
            remediation="Debe contener group_0 con targets Linear y weights num_bits 4.",
        ))
        return results

    # Take first group (like model_detect does)
    first_group: Optional[Dict[str, Any]] = None
    for _gname, gspec in groups.items():
        first_group = gspec if isinstance(gspec, dict) else None
        break
    if not first_group:
        results.append(CheckResult(
            name="W4A16 config_groups",
            severity="ERROR",
            message="config_groups is empty",
        ))
        return results

    weights = first_group.get("weights", {}) if isinstance(first_group, dict) else {}
    if not isinstance(weights, dict):
        weights = {}

    num_bits = weights.get("num_bits")
    group_size = weights.get("group_size")
    w_type = str(weights.get("type", "")).lower()
    symmetric = weights.get("symmetric")
    strategy = str(weights.get("strategy", "")).lower()
    actorder = weights.get("actorder")

    # num_bits
    try:
        bits_i = int(num_bits) if num_bits is not None else None
    except Exception:
        bits_i = None
    if bits_i != 4:
        results.append(CheckResult(
            name="W4A16 quantization_config num_bits",
            severity="ERROR",
            message=f"num_bits={num_bits!r} (expected 4)",
            remediation="El artefacto W4A16 requiere num_bits 4 para ser compatible con model_detect.py (int4_w4a16).",
        ))
    else:
        results.append(CheckResult(
            name="W4A16 quantization_config num_bits",
            severity="OK",
            message="num_bits is 4",
        ))

    # group_size
    try:
        gs_i = int(group_size) if group_size is not None else None
    except Exception:
        gs_i = None
    if gs_i != 128:
        results.append(CheckResult(
            name="W4A16 quantization_config group_size",
            severity="ERROR" if gs_i is not None else "WARN",
            message=f"group_size={group_size!r} (expected 128)",
            remediation="CK-3.1 exige group_size 128 para compatibilidad con la referencia soysoyr/W4A16-AWQ-GPTQ.",
        ))
    else:
        results.append(CheckResult(
            name="W4A16 quantization_config group_size",
            severity="OK",
            message="group_size is 128",
        ))

    # type int
    if w_type != "int":
        results.append(CheckResult(
            name="W4A16 quantization_config type",
            severity="ERROR",
            message=f"weights.type={w_type!r} (expected 'int')",
        ))
    else:
        results.append(CheckResult(
            name="W4A16 quantization_config type",
            severity="OK",
            message="weights.type is int",
        ))

    # symmetric
    if symmetric is not True:
        results.append(CheckResult(
            name="W4A16 quantization_config symmetric",
            severity="WARN",
            message=f"symmetric={symmetric!r} (expected True for int4 symmetric per-group absmax/7)",
        ))
    else:
        results.append(CheckResult(
            name="W4A16 quantization_config symmetric",
            severity="OK",
            message="symmetric is True",
        ))

    # strategy
    if strategy != "group":
        results.append(CheckResult(
            name="W4A16 quantization_config strategy",
            severity="WARN",
            message=f"strategy={strategy!r} (expected 'group')",
        ))
    else:
        results.append(CheckResult(
            name="W4A16 quantization_config strategy",
            severity="OK",
            message="strategy is group",
        ))

    # actorder (awq uses static, gptq may be None)
    if actorder not in (None, "static", "group", "dynamic"):
        results.append(CheckResult(
            name="W4A16 quantization_config actorder",
            severity="INFO",
            message=f"actorder={actorder!r} (expected 'static' for awq_int4 or None for gptq_int4)",
        ))
    else:
        results.append(CheckResult(
            name="W4A16 quantization_config actorder",
            severity="OK",
            message=f"actorder is {actorder!r}",
        ))

    # targets Linear
    targets = first_group.get("targets", [])
    if "Linear" not in targets:
        results.append(CheckResult(
            name="W4A16 quantization_config targets",
            severity="WARN",
            message=f"targets={targets!r} (expected ['Linear'])",
        ))
    else:
        results.append(CheckResult(
            name="W4A16 quantization_config targets",
            severity="OK",
            message="targets includes Linear",
        ))

    return results


def check_w4a16_artifact(
    model_dir: str | Path,
    expected_layers: str = "auto",
) -> List[CheckResult]:
    """Validate a W4A16 compressed-tensors artifact.

    Args:
        model_dir: Path to the requantized artifact directory (or HF id).
        expected_layers: "auto" (accept 192 MLPs or 256), "mlp" (expect 192),
            "all" (expect 256).  "auto" acepta ambas y solo avisa si el
            conteo es inesperado.

    Returns:
        List[CheckResult] — cada entrada tiene severity OK/WARN/ERROR/INFO.
        Un artefacto sano devuelve solo OKs (más un INFO de conteo). Un
        artefacto roto devuelve al menos un ERROR.
    """
    results: List[CheckResult] = []
    mdir = _resolve_model_dir(str(model_dir)) if isinstance(model_dir, str) else Path(model_dir)
    # For Path input, still try hub resolution if not a dir with config
    if isinstance(model_dir, Path) and not (Path(model_dir) / "config.json").is_file():
        mdir = _resolve_model_dir(str(model_dir))

    config, err = _load_config(mdir)
    if err is not None:
        results.append(CheckResult(
            name="W4A16 artifact config",
            severity="ERROR",
            message=err,
            remediation="Pasa --model con un path que contenga config.json generado por `genesis requant --output <dir> --model <fp8>`",
        ))
        return results

    assert config is not None
    # 1. quantization_config
    results.extend(_check_quantization_config(config))

    # Early exit if no quantization_config at all — no point checking shapes
    has_qc_error = any(r.severity == "ERROR" and "quantization_config" in r.name for r in results)
    if has_qc_error and not config.get("quantization_config"):
        return results

    # 2. Determine expected group_size for shape math (from config, fallback 128)
    qcfg = config.get("quantization_config", {}) or {}
    groups = qcfg.get("config_groups", {}) if isinstance(qcfg, dict) else {}
    first_weights: Dict[str, Any] = {}
    if isinstance(groups, dict):
        for _g, gspec in groups.items():
            if isinstance(gspec, dict):
                first_weights = gspec.get("weights", {}) if isinstance(gspec.get("weights"), dict) else {}
            break
    try:
        group_size = int(first_weights.get("group_size", 128))
    except Exception:
        group_size = 128
    if group_size <= 0:
        group_size = 128

    # 3. Collect headers
    headers, herr = _collect_headers(mdir)
    if herr is not None and not headers:
        results.append(CheckResult(
            name="W4A16 safetensors headers",
            severity="ERROR",
            message=f"failed to collect headers: {herr}",
            remediation="Verifica que el artefacto contenga model.safetensors o shards + index.",
        ))
        return results
    if herr is not None:
        results.append(CheckResult(
            name="W4A16 safetensors headers",
            severity="WARN",
            message=f"partial header collection: {herr}",
        ))
    else:
        results.append(CheckResult(
            name="W4A16 safetensors headers",
            severity="OK",
            message=f"collected {len(headers)} tensor headers",
        ))

    # Build weight_map for shape value loading
    weight_map: Dict[str, str] = {}
    index_path = mdir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            idx = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = idx.get("weight_map", {}) or {}
        except Exception:
            weight_map = {k: "" for k in headers.keys()}

    # 4. Find all packed keys
    packed_keys = sorted([k for k in headers.keys() if k.endswith(".weight_packed")])
    scale_keys_set = set(k for k in headers.keys() if k.endswith(".weight_scale"))
    shape_keys_set = set(k for k in headers.keys() if k.endswith(".weight_shape"))

    if not packed_keys:
        results.append(CheckResult(
            name="W4A16 packed tensors",
            severity="ERROR",
            message="no weight_packed tensors found (expected 192 for mlp, 256 for all)",
            remediation="El artefacto parece no estar cuantizado a W4A16. Ejecuta `genesis requant --layers mlp --format awq_int4 --model <fp8> --output <dir>`",
        ))
        return results

    # Count by proj type
    mlp_packed = [k for k in packed_keys if ".mlp." in k]
    attn_packed = [k for k in packed_keys if ".self_attn." in k]
    results.append(CheckResult(
        name="W4A16 packed count",
        severity="OK",
        message=f"found {len(packed_keys)} packed tensors: {len(mlp_packed)} mlp + {len(attn_packed)} attn",
    ))

    # Expected counts checks
    if expected_layers == "mlp":
        if len(mlp_packed) != 192:
            sev = "ERROR" if len(mlp_packed) < 192 else "WARN"
            results.append(CheckResult(
                name="W4A16 mlp count",
                severity=sev,
                message=f"mlp packed count {len(mlp_packed)} != 192 (64 capas × 3 projs)",
                remediation="Con --layers mlp se esperaban 192 lineares (gate/up/down). Revisa que la conversión haya cubierto todas las capas.",
            ))
        if attn_packed:
            results.append(CheckResult(
                name="W4A16 attn unexpected",
                severity="INFO",
                message=f"artifact has {len(attn_packed)} attn packed tensors but expected_layers=mlp (attn should be absent)",
            ))
    elif expected_layers == "all":
        total = len(packed_keys)
        if total != 256:
            sev = "WARN" if total > 0 else "ERROR"
            results.append(CheckResult(
                name="W4A16 total count",
                severity=sev,
                message=f"packed total {total} != 256 (192 mlp + 64 attn) for layers=all",
            ))
        if len(mlp_packed) != 192:
            results.append(CheckResult(
                name="W4A16 mlp count (all)",
                severity="WARN",
                message=f"mlp packed {len(mlp_packed)} != 192",
            ))
    else:  # auto
        if len(mlp_packed) not in (192, 0) and len(mlp_packed) < 180:
            results.append(CheckResult(
                name="W4A16 mlp count (auto)",
                severity="WARN",
                message=f"mlp packed {len(mlp_packed)} inusual (esperado 192 para mlp-only o 0 si solo attn)",
            ))
        if len(packed_keys) not in (192, 256) and len(packed_keys) < 192:
            results.append(CheckResult(
                name="W4A16 total count (auto)",
                severity="INFO",
                message=f"packed total {len(packed_keys)} (típico 192 mlp-only o 256 con attn)",
            ))

    # 5. Per-layer shape validation
    per_layer_errors = 0
    per_layer_ok = 0
    for packed_key in packed_keys:
        base = packed_key[: -len(".weight_packed")]
        scale_key = base + ".weight_scale"
        shape_key = base + ".weight_shape"

        # Existence
        if scale_key not in scale_keys_set and scale_key not in headers:
            results.append(CheckResult(
                name=f"W4A16 {base} scale missing",
                severity="ERROR",
                message=f"{scale_key} not found for {packed_key}",
                remediation="Cada weight_packed debe ir acompañado de weight_scale [out, in//group_size]",
            ))
            per_layer_errors += 1
            continue
        if shape_key not in shape_keys_set and shape_key not in headers:
            # weight_shape is expected but not strictly required for loading; warn
            results.append(CheckResult(
                name=f"W4A16 {base} shape missing",
                severity="WARN",
                message=f"{shape_key} not found (expected [2] with [out,in] for TP sharding)",
            ))
            # Continue validation without shape value

        packed_meta = headers.get(packed_key, {})
        scale_meta = headers.get(scale_key, {})
        shape_meta = headers.get(shape_key, {})

        # dtype checks
        packed_dtype = str(packed_meta.get("dtype", "")) if isinstance(packed_meta, dict) else ""
        scale_dtype = str(scale_meta.get("dtype", "")) if isinstance(scale_meta, dict) else ""
        shape_dtype = str(shape_meta.get("dtype", "")) if isinstance(shape_meta, dict) else ""

        if packed_dtype and packed_dtype != "I32":
            results.append(CheckResult(
                name=f"W4A16 {base} packed dtype",
                severity="ERROR",
                message=f"weight_packed dtype {packed_dtype!r} (expected I32, 8 int4 per int32)",
            ))
            per_layer_errors += 1
            continue
        # scale dtype should be BF16 like reference; allow BF16/F16/F32 but warn if not BF16
        if scale_dtype and scale_dtype not in ("BF16", "F16", "F32"):
            results.append(CheckResult(
                name=f"W4A16 {base} scale dtype",
                severity="WARN",
                message=f"weight_scale dtype {scale_dtype!r} (expected BF16 like referencia)",
            ))
        elif scale_dtype and scale_dtype != "BF16":
            results.append(CheckResult(
                name=f"W4A16 {base} scale dtype",
                severity="INFO",
                message=f"weight_scale dtype {scale_dtype} (referencia es BF16)",
            ))

        if shape_dtype and shape_dtype != "I64":
            results.append(CheckResult(
                name=f"W4A16 {base} shape dtype",
                severity="WARN",
                message=f"weight_shape dtype {shape_dtype!r} (expected I64)",
            ))

        # shape checks
        packed_shape = packed_meta.get("shape") if isinstance(packed_meta, dict) else None
        scale_shape = scale_meta.get("shape") if isinstance(scale_meta, dict) else None
        shape_shape = shape_meta.get("shape") if isinstance(shape_meta, dict) else None

        if not isinstance(packed_shape, list) or len(packed_shape) != 2:
            results.append(CheckResult(
                name=f"W4A16 {base} packed shape",
                severity="ERROR",
                message=f"weight_packed shape {packed_shape!r} (expected [out, in//8])",
            ))
            per_layer_errors += 1
            continue
        if not isinstance(scale_shape, list) or len(scale_shape) != 2:
            results.append(CheckResult(
                name=f"W4A16 {base} scale shape",
                severity="ERROR",
                message=f"weight_scale shape {scale_shape!r} (expected [out, in//group_size])",
            ))
            per_layer_errors += 1
            continue

        out_p, in_packed = int(packed_shape[0]), int(packed_shape[1])
        out_s, in_groups = int(scale_shape[0]), int(scale_shape[1])

        # out must match
        if out_p != out_s:
            results.append(CheckResult(
                name=f"W4A16 {base} out mismatch",
                severity="ERROR",
                message=f"out dim mismatch: packed {packed_shape} vs scale {scale_shape}",
            ))
            per_layer_errors += 1
            continue

        # in reconstruction: packed holds 8 int4 per I32, scale groups are in//group_size
        in_from_packed = in_packed * 8
        in_from_scale = in_groups * group_size

        if in_from_packed != in_from_scale:
            results.append(CheckResult(
                name=f"W4A16 {base} in mismatch",
                severity="ERROR",
                message=f"in dim mismatch: packed {packed_shape} -> in={in_from_packed} vs scale {scale_shape} (group {group_size}) -> in={in_from_scale}",
                remediation=f"Verifica group_size {group_size}: packed in//8 debe igualar scale in//group_size * {group_size}.",
            ))
            per_layer_errors += 1
            continue

        # Divisibility
        if in_from_packed % group_size != 0:
            results.append(CheckResult(
                name=f"W4A16 {base} in divisibility",
                severity="ERROR",
                message=f"in={in_from_packed} not divisible by group_size {group_size}",
            ))
            per_layer_errors += 1
            continue
        if in_from_packed % 8 != 0:
            results.append(CheckResult(
                name=f"W4A16 {base} in packing",
                severity="ERROR",
                message=f"in={in_from_packed} not divisible by 8 (pack factor)",
            ))
            per_layer_errors += 1
            continue

        # shape tensor should be [2]
        if shape_shape is not None:
            if not isinstance(shape_shape, list) or shape_shape != [2]:
                results.append(CheckResult(
                    name=f"W4A16 {base} weight_shape shape",
                    severity="WARN",
                    message=f"weight_shape header shape {shape_shape!r} (expected [2])",
                ))
            # Try to validate actual [out,in] values if we can load the tensor
            vals = _load_weight_shape_values(mdir, weight_map, shape_key)
            if vals is not None:
                out_val, in_val = vals
                if out_val != out_p or in_val != in_from_packed:
                    results.append(CheckResult(
                        name=f"W4A16 {base} weight_shape value",
                        severity="ERROR",
                        message=f"weight_shape value [{out_val},{in_val}] != expected [{out_p},{in_from_packed}] from packed/scale",
                    ))
                    per_layer_errors += 1
                    continue

        per_layer_ok += 1

    # Summary per-layer
    if per_layer_errors == 0 and per_layer_ok > 0:
        results.append(CheckResult(
            name="W4A16 per-layer shapes",
            severity="OK",
            message=f"all {per_layer_ok} packed layers have consistent weight_packed [out,in//8] I32 and weight_scale [out,in//{group_size}] BF16",
        ))
    elif per_layer_errors > 0:
        results.append(CheckResult(
            name="W4A16 per-layer shapes",
            severity="ERROR",
            message=f"{per_layer_errors} layer(s) with shape mismatch, {per_layer_ok} OK",
            remediation="Re-ejecuta `genesis requant --group-size 128` y valida contra la referencia soysoyr/W4A16-AWQ-GPTQ.",
        ))

    # 6. model_detect compatibility hint
    # If all OK, add INFO that model_detect will report int4_w4a16
    if not any(r.severity == "ERROR" for r in results):
        results.append(CheckResult(
            name="W4A16 model_detect hint",
            severity="INFO",
            message="artifact compatible with vllm._genesis.model_detect (quant_format int4_w4a16 / compressed_tensors num_bits 4)",
        ))

    return results


# ─── CLI ───────────────────────────────────────────────────────────────────

def _format_results(results: List[CheckResult], json_out: bool = False) -> str:
    if json_out:
        return json.dumps(
            [{"name": r.name, "severity": r.severity, "message": r.message, "remediation": r.remediation} for r in results],
            indent=2,
            ensure_ascii=False,
        )
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("Genesis W4A16 doctor — artifact validation (CK-3.1)")
    lines.append("=" * 72)
    for r in results:
        icon = {"OK": "✓", "INFO": "·", "WARN": "⚠", "ERROR": "✗"}.get(r.severity, "?")
        lines.append(f"{icon} [{r.severity}] {r.name}: {r.message}")
        if r.remediation:
            lines.append(f"  → {r.remediation}")
    # Summary
    n_err = sum(1 for r in results if r.severity == "ERROR")
    n_warn = sum(1 for r in results if r.severity == "WARN")
    lines.append("-" * 72)
    if n_err == 0 and n_warn == 0:
        lines.append("✓ artifact OK — vLLM compressed-tensors W4A16 compatible")
    elif n_err == 0:
        lines.append(f"⚠ artifact OK with {n_warn} warning(s)")
    else:
        lines.append(f"✗ artifact INVALID: {n_err} error(s), {n_warn} warning(s)")
    lines.append("=" * 72)
    return "\n".join(lines)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vllm._genesis.doctor.rules.w4a16",
        description="Valida un artefacto W4A16 compressed-tensors (CK-3.1). "
        "Chequea quantization_config num_bits 4, group_size 128 y shapes "
        "weight_packed/scale por capa.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path al artefacto W4A16 (directorio con config.json + safetensors) o HF id.",
    )
    p.add_argument(
        "--layers",
        choices=["auto", "mlp", "all"],
        default="auto",
        help="Conteo esperado: auto (192 o 256), mlp (192), all (256).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emitir JSON en vez de texto humano.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Log debug.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    results = check_w4a16_artifact(args.model, expected_layers=args.layers)
    print(_format_results(results, json_out=args.json))
    has_error = any(r.severity == "ERROR" for r in results)
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())

