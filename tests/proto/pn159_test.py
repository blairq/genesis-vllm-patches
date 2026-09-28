"""PN159: logits del subconjunto contra las mismas columnas del lm_head completo de PN139 (A8), y tiempos."""
import os, types, torch
os.environ["GENESIS_ENABLE_PN159_VOCAB_BORRADOR"] = "1"; os.environ.setdefault("GENESIS_PN139_A8", "1")
from vllm._genesis import lm_head_int4 as g139, vocab_borrador as g159
dev = "cuda"; torch.manual_seed(0)
N, K = 124160, 5120


def tiempo(f):
    for _ in range(3): f()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): f()
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(5): g.replay()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) * 1000 / 50


for rango in (0, 1):
    for tam in (32768, 65536):
        g159.TAM = tam
        capa = torch.nn.Module()
        capa.register_parameter("weight", torch.nn.Parameter((torch.randn(N, K, device=dev) * 0.02).half(), requires_grad=False))
        capa.shard_indices = types.SimpleNamespace(org_vocab_start_index=rango * N, org_vocab_end_index=(rango + 1) * N)
        capa.tp_size = 1
        assert g139.preparar(capa, "lm_head")
        for M in (9, 36):
            x = torch.randn(M, K, device=dev).half()
            full = g139.aplicar(capa, x); sub = g159._logits(capa, x)
            ref = full[:, capa.pn159_ids - rango * N]
            d = float((sub.float() - ref.float()).abs().max() / ref.float().abs().max())
            t1 = tiempo(lambda: g139.aplicar(capa, x)); t2 = tiempo(lambda: g159._logits(capa, x))
            print(f"rango {rango} vocab {tam} ({capa.pn159_n} filas) M={M}: rel {d:.1e} | completo {t1:6.1f} us  recortado {t2:6.1f} us", flush=True)
        del capa; torch.cuda.empty_cache()
