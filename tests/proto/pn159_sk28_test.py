"""SK-28 contra la referencia de torch (topk del fijo + anillo), y tiempos dentro de un grafo CUDA."""
import os, types, torch
D = os.environ.get("D", "512")
os.environ.update(GENESIS_ENABLE_PN159_VOCAB_BORRADOR="1", GENESIS_PN139_A8="1", GENESIS_PN159_DINAMICO=D,
                  GENESIS_PN159_VOCAB=os.environ.get("VOC", "32768"), GENESIS_PN159_SK28="1",
                  RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29513", LOCAL_RANK="0")
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.config import VllmConfig, set_current_vllm_config
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(1, 0, "env://", 0, "nccl"); initialize_model_parallel(1)
from vllm._genesis import lm_head_int4 as g139, vocab_borrador as g159
from vllm.model_executor.layers.logits_processor import _topk
N, Kd = 248320, 5120; dev = "cuda"
capa = torch.nn.Module()
capa.register_parameter("weight", torch.nn.Parameter((torch.randn(N, Kd, device=dev) * 0.02).half(), requires_grad=False))
capa.shard_indices = types.SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=N); capa.tp_size = 1
g139.preparar(capa, "lm_head")
g159.observar(torch.arange(60000, 60000 + int(D) // 2, device=dev))          # anillo a medio llenar
d = g159._din
proc = types.SimpleNamespace(scale=1.0, soft_cap=None)


def ref(x):
    ls = g159._logits(capa, x); vs, is_ = torch.topk(ls.float(), 16); is_ = capa.pn159_ids[is_]
    ld = torch.nn.functional.linear(x, d["filas"]).float().masked_fill(d["ids"] < 0, float("-inf"))
    tv = torch.cat([vs, ld], -1); ti = torch.cat([is_, d["ids"].long().expand(x.shape[0], -1)], -1)
    rv, sel = torch.topk(tv, 16); return rv, ti.gather(-1, sel)


def t_grafo(f, it=20):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it): f()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(5): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / (5 * it)


for M in (9, 36, 54):
    x = torch.randn(M, Kd, device=dev).half()
    x[0] = d["filas"][3] * 30                                    # que gane un token del anillo en la fila 0
    rv, ri = ref(x)
    ii, iv = g159.top_k(proc, capa, x, 16)
    igual = (ii == ri).float().mean().item()
    mismo_conj = sum(set(a.tolist()) == set(b.tolist()) for a, b in zip(ii, ri)) / M
    print(f"M={M}: ids iguales {igual:.3f}, mismo conjunto {mismo_conj:.3f}, valores rel {float((iv - rv).abs().max() / rv.abs().max()):.1e}, "
          f"fila 0 top-1 del anillo {int(ii[0, 0]) == int(d['ids'][3])}", flush=True)
    ls = g159._logits(capa, x)
    t_sk = t_grafo(lambda: torch.ops.genesis.pn159_topk(ls, x, capa.pn159_ids))
    g159._SK28 = False
    t_viejo = t_grafo(lambda: g159.top_k(proc, capa, x, 16)) - t_grafo(lambda: g159._logits(capa, x))
    g159._SK28 = True
    print(f"   GRAFO despues de Marlin: torch/flashinfer + anillo {t_viejo:.1f} us  ->  SK-28 {t_sk:.1f} us", flush=True)

for M in (9, 36):
    x = torch.randn(M, Kd, device=dev).half(); ls = g159._logits(capa, x)
    for _ in range(3): torch.ops.genesis.pn159_topk(ls, x, capa.pn159_ids)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(20): torch.ops.genesis.pn159_topk(ls, x, capa.pn159_ids)
        torch.cuda.synchronize()
    tot = {}
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            tot.setdefault(e.name[:40], []).append(e.device_time)
    print(f"PERFIL M={M}: " + ", ".join(f"{k} {sum(v) / 20:.1f}us" for k, v in sorted(tot.items(), key=lambda kv: -sum(kv[1]))))
