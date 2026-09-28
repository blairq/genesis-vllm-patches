"""PN159 fase 2 (TP=1, una GPU): SK-27 lee filas del host por UVA; verifica filas, dedup/anillo y que el top-k
propone un token de afuera del subconjunto fijo cuando esta en el anillo; tiempos de SK-27."""
import os, types, torch
os.environ.update(GENESIS_ENABLE_PN159_VOCAB_BORRADOR="1", GENESIS_PN139_A8="1", GENESIS_PN159_DINAMICO="1024",
                  GENESIS_PN159_VOCAB="32768")
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.config import VllmConfig, set_current_vllm_config
os.environ.update(RANK="0", WORLD_SIZE="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29511", LOCAL_RANK="0")
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(1, 0, "env://", 0, "nccl"); initialize_model_parallel(1)
from vllm._genesis import lm_head_int4 as g139, vocab_borrador as g159
import numpy as np
N, K = 248320, 5120; dev = "cuda"
capa = torch.nn.Module()
capa.register_parameter("weight", torch.nn.Parameter((torch.randn(N, K, device=dev) * 0.02).half(), requires_grad=False))
w_ref = capa.weight.data.clone().cpu()
capa.shard_indices = types.SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=N)
capa.tp_size = 1
assert g139.preparar(capa, "lm_head")
d = capa.pn159_din
orden = np.load(g159._ORDEN)
fuera = [int(t) for t in orden[40000:40010]]                        # fuera de los 32k fijos
ids = torch.tensor([5, 7] + fuera + fuera[:3] + [9], device=dev)    # repetidos y estaticos
g159.observar(ids); torch.cuda.synchronize()
din_ids = d["ids"].cpu().tolist()
ok_ids = sorted(t for t in din_ids if t >= 0) == sorted(fuera)
# filas: contra el int4 de PN139 decuantizado (misma cuantizacion)
from vllm._genesis.lm_head_int4 import cuantizar_int4_grupo
qp, esc = cuantizar_int4_grupo(w_ref[fuera].to(dev))
q = torch.stack([(qp >> (4 * i)) & 15 for i in range(8)], 1).reshape(K, len(fuera)).t().float() - 8
ref = (q.view(len(fuera), K // 128, 128) * esc.t().float()[:, :, None]).view(len(fuera), K)
pos = {t: i for i, t in enumerate(din_ids)}
got = torch.stack([d["filas"][pos[t]].float() for t in fuera])
print("ids del anillo", "OK" if ok_ids else f"MAL {din_ids[:12]}", "| filas rel", float((got - ref).abs().max() / ref.abs().max()))
# top-k: un h alineado con la fila de un token de afuera -> tiene que salir primero
proc = types.SimpleNamespace(scale=1.0, soft_cap=None)
h = ref[3:4].half() * 50
ids_k, vals = g159.top_k(proc, capa, h, 16)
print("top-1 con h alineado a", fuera[3], "->", int(ids_k[0, 0]), "(OK)" if int(ids_k[0, 0]) == fuera[3] else "(MAL)")
# anillo: meter 1100 tokens de afuera distintos -> se queda con los ultimos 1024, sin duplicados
muchos = torch.tensor([int(t) for t in orden[50000:51100]], device=dev)
g159.observar(muchos); torch.cuda.synchronize()
v = [t for t in d["ids"].cpu().tolist() if t >= 0]
bits = int(sum(bin(x & 0xFFFFFFFF).count("1") for x in d["dinbit"].cpu().tolist()))
print("anillo", len(v), "distintos", len(set(v)), "bits prendidos", bits)
# tiempos
def t_us(f, it=200):
    for _ in range(5): f()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(it): f()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / it
dec = torch.tensor([5, 7, 9, 11, 13, 15, 17, 19, 21], device=dev)
print("SK-27 decode 9 tokens sin nuevos: %.1f us" % t_us(lambda: g159.observar(dec)))
pre = torch.tensor(orden[:8192].astype("int64"), device=dev)
print("SK-27 prefill 8192 tokens (todos estaticos): %.1f us" % t_us(lambda: g159.observar(pre)))
base = 60000
def nuevos():
    global base
    g159.observar(torch.arange(base, base + 64, device=dev)); base += 64
print("SK-27 con 64 filas nuevas del host: %.1f us" % t_us(nuevos, 50))
x = torch.randn(9, K, device=dev).half()
print("top_k con anillo: %.1f us" % t_us(lambda: g159.top_k(proc, capa, x, 16)))


# referencia en torch de la fusion (SK-27b + SK-27c)
from vllm.model_executor.layers.logits_processor import _topk
for M in (9, 36):
    xm = torch.randn(M, K, device=dev).half()
    xm[0] = ref[3].half() * 50
    ls = g159._logits(capa, xm); vs, is_ = _topk(ls, 16); is_ = capa.pn159_ids[is_.long()]
    ld = torch.nn.functional.linear(xm, d["filas"]).float().masked_fill(d["ids"] < 0, float("-inf"))
    todos_v = torch.cat([vs.float(), ld], -1); todos_i = torch.cat([is_, d["ids"].long().expand(M, -1)], -1)
    rv, sel = torch.topk(todos_v, 16); ri = todos_i.gather(-1, sel)
    ii, iv = g159.top_k(proc, capa, xm, 16)
    igual = (ii == ri).float().mean().item()
    print(f"fusion M={M}: ids iguales {igual:.3f}, valores rel {float((iv.float() - rv).abs().max() / rv.abs().max()):.1e}", flush=True)


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


print("GRAFO SK-27 decode 9 tokens: %.1f us" % t_grafo(lambda: g159.observar(dec)))
print("GRAFO top_k fijo+anillo: %.1f us" % t_grafo(lambda: g159.top_k(proc, capa, x, 16)))
din = capa.pn159_din; del capa.pn159_din
print("GRAFO top_k solo fijo: %.1f us" % t_grafo(lambda: g159.top_k(proc, capa, x, 16)))
capa.pn159_din = din

vs9, is9 = _topk(g159._logits(capa, x), 16); is9 = capa.pn159_ids[is9.long()]
print("GRAFO solo SK-27b+c: %.1f us" % t_grafo(lambda: g159.anillo_cuda(x, vs9, is9)))
xx = x.to(torch.float16).contiguous(); ldd = torch.empty(9, g159.DIN, dtype=torch.float32, device=dev)
print("GRAFO solo SK-27b: %.1f us" % t_grafo(lambda: g159._kern("sk27_logits").lanzar((-(-g159.DIN // 8), 1), [xx, 9, d["filas"], d["ids"], ldd, g159.DIN])))
vo = torch.empty(9, 16, dtype=torch.float16, device=dev); io = torch.empty(9, 16, dtype=torch.int64, device=dev)
print("GRAFO solo SK-27c: %.1f us" % t_grafo(lambda: g159._kern("sk27_fusion", ("-DKT=16",)).lanzar((9, 1), [vs9, is9, ldd, d["ids"], g159.DIN, vo, io], shared=(16 + g159.DIN) * 4)))
print("GRAFO solo lm_head fijo (Marlin): %.1f us" % t_grafo(lambda: g159._logits(capa, x)))
print("GRAFO solo _topk fijo: %.1f us" % t_grafo(lambda: _topk(g159._logits(capa, x), 16)))
