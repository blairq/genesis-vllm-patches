# SPDX-License-Identifier: Apache-2.0
"""PN178: las anclas del text-patch sobre fragmentos de qwen3_dflash2.py y dflash2/speculator.py de vLLM
v0.29.0, idempotencia y que el resultado compila.

    PYTHONPATH=. python3 -m pytest -q vllm/_genesis/tests/test_pn178_wiring.py
"""
import py_compile

import pytest

MODELO = '''from .utils import maybe_prefix


class DFlash2Qwen3Model(DFlashQwen3Model):
    def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
        draft_config = self.config.dflash_config
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids):
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
'''

SPECULATOR = '''from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


class DFlash2Speculator(DFlashSpeculator):
    def _generate_draft(self, num_reqs):
        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        self._sample_path(candidate_ids, scores, num_reqs)
'''


@pytest.fixture
def archivos(tmp_path, monkeypatch):
    import vllm._genesis.dispatcher as d
    import vllm._genesis.wiring.spec_decode.patch_PN178_dcoco as w
    m = tmp_path / "model_executor/models/qwen3_dflash2.py"
    s = tmp_path / "v1/worker/gpu/spec_decode/dflash2/speculator.py"
    for f, t in ((m, MODELO), (s, SPECULATOR)):
        f.parent.mkdir(parents=True)
        f.write_text(t)
    monkeypatch.setattr(d, "should_apply", lambda pid: (True, "test"))
    monkeypatch.setattr(d, "log_decision", lambda *a, **k: None)
    monkeypatch.setattr(w, "resolve_vllm_file", lambda rel: tmp_path / rel)
    return w, m, s


def test_aplica_compila_y_es_idempotente(archivos):
    w, m, s = archivos
    estado, _ = w.apply()
    assert estado == "applied"
    tm, ts = m.read_text(), s.read_text()
    assert "_g178.crear(self, draft_config" in tm and "_g178.cargar(self.model, super().load_weights" in tm
    assert ts.index("_g178.corregir(") < ts.index("self._sample_path(candidate_ids, scores")
    for f in (m, s):
        py_compile.compile(str(f), doraise=True)
    assert w.apply()[0] != "applied"
    assert m.read_text() == tm and s.read_text() == ts


def test_sin_ancla_no_toca_nada(archivos):
    w, m, s = archivos
    s.write_text(SPECULATOR.replace("self._sample_path(candidate_ids, scores, num_reqs)", "pass"))
    antes = s.read_text()
    assert w.apply()[0] != "applied"
    assert s.read_text() == antes
