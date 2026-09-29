# SPDX-License-Identifier: Apache-2.0
"""PN161 — atencion del borrador DFlash2: norma de q/k + rope + FWHT de PN126 + escritura de la KV int8 en un
kernel (SK-33); ver ``vllm._genesis.borrador_qkkv``. Toca DFlashQwen3Attention (qwen3_dflash.py): la decision se
toma en __init__ (queda como atributo constante para dynamo) y el forward vuelve antes del camino de siempre.
Compatible con PN160 (o_proj va por borrador_fusion.lineal, que cae al camino de siempre si PN160 esta apagado).
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN161: qk y KV del borrador]"
_IMP_OLD = "from vllm.logger import init_logger\n"
_IMP = "from vllm._genesis import borrador_qkkv as _g161  # " + MARKER + "\n"
_INIT_OLD = "        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)\n"
_INIT_NEW = _INIT_OLD + "        _g161.preparar(self)  # " + MARKER + "\n"
_FWD_OLD = "        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)\n"
_FWD_NEW = (
    "        if self._pn161:  # " + MARKER + "\n"
    "            return _g161.o_proj(self, _g161.atencion(self, _g161.qk_kv(self, qkv, positions)))\n"
    + _FWD_OLD
)


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN161")
    log_decision("PN161", decision, reason)
    if not decision:
        return "skipped", reason
    fa = resolve_vllm_file("model_executor/models/qwen3_dflash.py")
    if fa is None:
        return "skipped", "falta qwen3_dflash.py"
    p = TextPatcher(patch_name="PN161 qk y KV del borrador", target_file=str(fa), marker=MARKER, sub_patches=[
        TextPatch(name="pn161_imp", anchor=_IMP_OLD, replacement=_IMP_OLD + _IMP, required=True),
        TextPatch(name="pn161_init", anchor=_INIT_OLD, replacement=_INIT_NEW, required=True),
        TextPatch(name="pn161_fwd", anchor=_FWD_OLD, replacement=_FWD_NEW, required=True)],
        upstream_drift_markers=["_g161"])
    r, fl = p.apply()
    e = result_to_wiring_status(r, fl, applied_message=p.patch_name, patch_name=p.patch_name)
    if e[0] != "applied":
        return e[0], f"{p.patch_name}: {e[1]}"
    return "applied", "norma q/k + rope + FWHT + KV int8 del borrador en un kernel (SK-33)"
