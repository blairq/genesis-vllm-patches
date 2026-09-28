"""PN159 con TP=2 real: columnas int4 del lm_head de PN139 (antes del repack) repartidas parejo entre rangos.
Verifica los logits del subconjunto contra las MISMAS columnas del lm_head completo (juntando los dos rangos).
  torchrun --nproc-per-node 2 pn159_tp2.py"""
import os, types, torch
os.environ["GENESIS_ENABLE_PN159_VOCAB_BORRADOR"] = "1"; os.environ.setdefault("GENESIS_PN139_A8", "1")
from vllm.distributed import init_distributed_environment, initialize_model_parallel, tensor_model_parallel_all_gather
from vllm.config import VllmConfig, set_current_vllm_config
r = int(os.environ["RANK"]); torch.cuda.set_device(r)
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(2, r, "env://", r, "nccl"); initialize_model_parallel(2)
from vllm._genesis import lm_head_int4 as g139, vocab_borrador as g159
N, K = 124160, 5120; dev = "cuda"


def tiempo(f):
    for _ in range(3): f()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): f()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(5): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / 50


for tam in (32768, 65536):
    g159.TAM = tam; g159._HECHO = False
    capa = torch.nn.Module()
    gen = torch.Generator(device=dev).manual_seed(100 + r)
    capa.register_parameter("weight", torch.nn.Parameter((torch.randn(N, K, device=dev, generator=gen) * 0.02).half(),
                                                         requires_grad=False))
    capa.shard_indices = types.SimpleNamespace(org_vocab_start_index=r * N, org_vocab_end_index=(r + 1) * N)
    capa.tp_size = 2
    torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    assert g139.preparar(capa, "lm_head")
    torch.cuda.synchronize(); pico = (torch.cuda.max_memory_allocated() - m0) / 2**20
    x = torch.randn(9, K, device=dev, generator=torch.Generator(device=dev).manual_seed(7)).half()
    completo = tensor_model_parallel_all_gather(g139.aplicar(capa, x), dim=-1)      # [9, 2N] global
    ref = completo[:, capa.pn159_ids]
    sub = g159._logits(capa, x)
    d = float((sub.float() - ref.float()).abs().max() / ref.float().abs().max())
    ts = [tiempo(lambda: g159._logits(capa, torch.randn(M, K, device=dev).half())) for M in (9, 36)]
    tf = [tiempo(lambda: g139.aplicar(capa, torch.randn(M, K, device=dev).half())) for M in (9, 36)]
    print(f"rango {r} vocab {tam}: {capa.pn159_n} columnas ({int((capa.pn159_ids < N).sum())} del tramo 0) | "
          f"vs lm_head completo rel {d:.1e} | pico PN139+PN159 +{pico:.0f} MiB | M=9 {tf[0]:.0f}->{ts[0]:.0f} us, "
          f"M=36 {tf[1]:.0f}->{ts[1]:.0f} us", flush=True)
    del capa; torch.cuda.empty_cache()
