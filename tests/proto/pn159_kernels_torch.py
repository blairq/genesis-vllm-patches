"""Que kernels lanza realmente el top_k del borrador (fase 1 y fase 2), con el profiler de torch."""
import os, types, torch
os.environ.update(GENESIS_ENABLE_PN159_VOCAB_BORRADOR="1", GENESIS_PN139_A8="1", GENESIS_PN159_DINAMICO="512",
                  GENESIS_PN159_VOCAB="32768", RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29512", LOCAL_RANK="0")
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.config import VllmConfig, set_current_vllm_config
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(1, 0, "env://", 0, "nccl"); initialize_model_parallel(1)
from vllm._genesis import lm_head_int4 as g139, vocab_borrador as g159
N, K = 248320, 5120; dev = "cuda"
capa = torch.nn.Module()
capa.register_parameter("weight", torch.nn.Parameter((torch.randn(N, K, device=dev) * 0.02).half(), requires_grad=False))
capa.shard_indices = types.SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=N); capa.tp_size = 1
g139.preparar(capa, "lm_head")
g159.observar(torch.arange(60000, 60400, device=dev))
proc = types.SimpleNamespace(scale=1.0, soft_cap=None)
x = torch.randn(9, K, device=dev).half()
for nombre in ("fase2", "fase1"):
    if nombre == "fase1":
        del capa.pn159_din
    for _ in range(5): g159.top_k(proc, capa, x, 16)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(20): g159.top_k(proc, capa, x, 16)
        torch.cuda.synchronize()
    print(f"== {nombre}: kernels por llamada (us promedio)")
    ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    tot = {}
    for e in ev:
        tot.setdefault(e.name[:90], []).append(e.device_time)
    for k, v in sorted(tot.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {sum(v) / 20:7.1f} us  x{len(v) // 20}  {k}")
