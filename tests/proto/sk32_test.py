"""SK-32 contra vLLM: per_token_quant_int8 (+ escala global) y SiluAndMul + eso. Bit a bit en q; esc exacta."""
import torch
from vllm._genesis import borrador_fusion as bf
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_quant_input
from vllm.model_executor.layers.activation import SiluAndMul
torch.manual_seed(0); dev = "cuda"
from vllm.config import VllmConfig, set_current_vllm_config
_ctx = set_current_vllm_config(VllmConfig()); _ctx.__enter__()
g = torch.tensor([0.37], device=dev)
for T, N in ((9, 5120), (36, 5120), (9, 2048), (9, 8704), (300, 5120)):
    x = (torch.randn(T, N, device=dev) * 3).half()
    q, esc = bf._q8_cuda(x, g, False)
    qr, er = marlin_quant_input(x, torch.int8); er = er * g
    dq = int((q.int() - qr.int()).abs().max()); de = float((esc.view(-1) - er.view(-1)).abs().max() / er.abs().max())
    xs = (torch.randn(T, 2 * N, device=dev) * 2).half()
    qs, es = bf._q8_cuda(xs, g, True)
    y = SiluAndMul().forward_native(xs).half()
    qr2, er2 = marlin_quant_input(y, torch.int8); er2 = er2 * g
    dq2 = int((qs.int() - qr2.int()).abs().max()); n2 = int((qs != qr2).sum())
    print(f"T={T} N={N}: q8 max|dq|={dq} esc rel {de:.1e} | silu_q8 max|dq|={dq2} ({n2} de {qs.numel()} distintos)", flush=True)
# conv_q8 contra _grouped_conv de vLLM + per_token_quant_int8
from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv
H, G, GS, TAPS, BSQ = 5120, 320, 16, 2, 9
for T in (9, 36, 54):
    h = torch.randn(T, H, device=dev).half()
    coef = (torch.randn(T, 2 * TAPS * G, device=dev) * 0.3).half()
    base = (torch.randn(2, TAPS, H, device=dev) * 0.5 + 1).half()
    q, esc = bf._conv_q8_cuda(h, coef, base[0], g, BSQ)
    y = _grouped_conv(h, coef.view(T, 2, TAPS, G)[:, 0], base[0], BSQ, G, GS, TAPS)
    qr, er = marlin_quant_input(y.half(), torch.int8); er = er * g
    print(f"conv_q8 T={T}: max|dq|={int((q.int() - qr.int()).abs().max())} ({int((q != qr).sum())} de {q.numel()} distintos), "
          f"esc rel {float((esc.view(-1) - er.view(-1)).abs().max() / er.abs().max()):.1e}", flush=True)
