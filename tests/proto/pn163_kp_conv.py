"""Error que llega a la conv: kernel = base + c (c = kernel_projection(x)), con c en fp16 vs int8 por canal / g64."""
import torch
from safetensors import safe_open
P = "/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86_dflash2/model.safetensors"
f = safe_open(P, "pt", device="cuda")
torch.manual_seed(0)


def rtn8(w, g):
    n, k = w.shape; g = k if g <= 0 else g
    wf = w.view(n, k // g, g); s = wf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return (torch.round(wf / s).clamp(-128, 127) * s).view(n, k)


tot = {}
for capa in range(5):
    for lado in ("attention_conv", "mlp_conv"):
        w = f.get_tensor(f"layers.{capa}.{lado}.kernel_projection.weight").float()
        b = f.get_tensor(f"layers.{capa}.{lado}.base_kernel").float()          # [2 lados, 2 taps, N]
        norma = "input_layernorm" if lado == "attention_conv" else "post_attention_layernorm"
        nw = f.get_tensor(f"layers.{capa}.{norma}.weight").float()
        x = torch.randn(64, w.shape[1], device="cuda") * nw
        G = w.shape[0] // 4
        def kern(wq):
            c = (x @ wq.t()).view(64, 2, 2, G)                                 # lado, tap, grupo
            return b[None] + c.repeat_interleave(16, dim=-1)                   # [64, 2, 2, N]
        kr = kern(w)
        for nom, wq in (("int8 canal", rtn8(w, -1)), ("int8 g64", rtn8(w, 64))):
            er = float((kern(wq) - kr).norm() / kr.norm())
            tot.setdefault(nom, []).append(er)
        tot.setdefault("|c| / |b+c|", []).append(float((kr - b[None]).norm() / kr.norm()))
for n, v in tot.items():
    print(f"{n:12s} error relativo del kernel de la conv (b + c): media {sum(v) / len(v):.2e} max {max(v):.2e}")
