"""PN159 fase 2 + SK-28/29 con TP=2 real (K=1024 para que entre al lado del entrenamiento): top-k contra una
referencia en torch sobre el lm_head completo de los dos rangos, con tokens del anillo en ambos tramos."""
import os, types, torch
os.environ.update(GENESIS_ENABLE_PN159_VOCAB_BORRADOR="1", GENESIS_PN139_A8="1", GENESIS_PN159_DINAMICO="512",
                  GENESIS_PN159_VOCAB="32768", GENESIS_PN159_SK28="1")
from vllm.distributed import init_distributed_environment, initialize_model_parallel, tensor_model_parallel_all_gather
from vllm.config import VllmConfig, set_current_vllm_config
r = int(os.environ["RANK"]); torch.cuda.set_device(r)
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(2, r, "env://", r, "nccl"); initialize_model_parallel(2)
from vllm._genesis import lm_head_int4 as g139, vocab_borrador as g159
import numpy as np
N, K = 124160, 1024; dev = "cuda"
gen = torch.Generator(device=dev).manual_seed(100 + r)
capa = torch.nn.Module()
capa.register_parameter("weight", torch.nn.Parameter((torch.randn(N, K, device=dev, generator=gen) * 0.02).half(), requires_grad=False))
capa.shard_indices = types.SimpleNamespace(org_vocab_start_index=r * N, org_vocab_end_index=(r + 1) * N); capa.tp_size = 2
assert g139.preparar(capa, "lm_head")
orden = np.load(g159._ORDEN)
fuera = [int(t) for t in orden[40000:40600]]                        # fuera de 32k, de los dos tramos
g159.observar(torch.tensor(fuera, device=dev)); torch.cuda.synchronize()
en_anillo = sorted(t for t in g159._din["ids"].cpu().tolist() if t >= 0)
mios = sorted(t for t in fuera if r * N <= t < (r + 1) * N)
print(f"rango {r}: anillo {len(en_anillo)} filas, las de su tramo: {'OK' if en_anillo == mios[-512:] or en_anillo == sorted(mios)[:512] or set(en_anillo) <= set(mios) else 'MAL'}", flush=True)
proc = types.SimpleNamespace(scale=1.0, soft_cap=None)
torch.manual_seed(7)
x = torch.randn(9, K, device=dev).half()
# que en la fila 0 gane un token del anillo del rango 1 y en la fila 1 uno del rango 0
completo = tensor_model_parallel_all_gather(g139.aplicar(capa, x), dim=-1).float()        # [9, 2N]
ii, iv = g159.top_k(proc, capa, x, 16)
# referencia: top-16 sobre (subconjunto fijo U anillo de ambos rangos) del lm_head completo
est = torch.zeros(2 * N, dtype=torch.bool, device=dev); est[torch.from_numpy(orden[:32768].astype("int64")).to(dev)] = True
anillo_todos = tensor_model_parallel_all_gather(g159._din["ids"].unsqueeze(0), dim=0).flatten()
anillo_todos = anillo_todos[anillo_todos >= 0].long()
perm = est.clone(); perm[anillo_todos] = True
rv, ri = completo.masked_fill(~perm, float("-inf")).topk(16)
mismo = sum(set(a.tolist()) == set(b.tolist()) for a, b in zip(ii, ri)) / 9
print(f"rango {r}: mismo conjunto que la referencia {mismo:.3f}, valores rel {float((iv - rv).abs().max() / rv.abs().max()):.1e}, "
      f"ids del anillo en el top-16: {int(sum(t in set(anillo_todos.tolist()) for t in ii.flatten().tolist()))}", flush=True)
