# SPDX-License-Identifier: Apache-2.0
"""PN158 — el all-reduce del embedding de vocabulario partido por P2P (SK-24) en vez de NCCL.

PN120/PN152 solo cubren los all-reduce de las lineales de fila; el del VocabParallelEmbedding seguia por
NCCL: ~22 us en el target y ~24 en el borrador por paso de decode. Cada token lo aporta un solo rango, asi
que la suma fp16 es exacta: va SIEMPRE por la ruta fp16 (nunca la int8 de PN152). Si no entra en la
ranura P2P (prefill grande), NCCL como siempre.
"""
from __future__ import annotations

from vllm._genesis.guards import resolve_vllm_file
from vllm._genesis.wiring.text_patch import TextPatch, TextPatcher, result_to_wiring_status

MARKER = "[Genesis PN158: embedding por P2P]"
_OLD = (
    "        # Reduce across all the model parallel GPUs.\n"
    "        return tensor_model_parallel_all_reduce(output_parallel)\n"
)
_NEW = (
    "        # Reduce across all the model parallel GPUs.  " + MARKER + "\n"
    "        return _g158.all_reduce_exacto(output_parallel)\n"
)
_IMP_OLD = "from vllm.model_executor.utils import set_weight_attrs\n"
_IMP_NEW = _IMP_OLD + "from vllm._genesis import ar_int8 as _g158  # " + MARKER + " (registra el op al importar)\n"


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN158")
    log_decision("PN158", decision, reason)
    if not decision:
        return "skipped", reason
    f = resolve_vllm_file("model_executor/layers/vocab_parallel_embedding.py")
    if f is None:
        return "skipped", "falta vocab_parallel_embedding.py"
    p = TextPatcher(patch_name="PN158 embedding por P2P", target_file=str(f), marker=MARKER, sub_patches=[
        TextPatch(name="pn158_imp", anchor=_IMP_OLD, replacement=_IMP_NEW, required=True),
        TextPatch(name="pn158_ar", anchor=_OLD, replacement=_NEW, required=True)], upstream_drift_markers=["_g158"])
    r, fl = p.apply()
    return result_to_wiring_status(r, fl, applied_message="all-reduce del embedding por P2P", patch_name=p.patch_name)
