"""kernel_projection del borrador: espectro, outliers y error de variantes (rango bajo, rotacion Hadamard + int8)."""
import torch
from safetensors import safe_open
P = "/root/.cache/huggingface/qwen3.8_27b_idiotSavant_sm_86_dflash2/model.safetensors"
f = safe_open(P, "pt", device="cuda")
kps = sorted(k for k in f.keys() if "kernel_projection" in k)
torch.manual_seed(0)


def had(n):
    H = torch.ones(1, 1, device="cuda")
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / H.shape[0] ** 0.5


def rtn8(w, g):
    n, k = w.shape; g = k if g <= 0 else g
    wf = w.view(n, k // g, g); s = wf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return (torch.round(wf / s).clamp(-128, 127) * s).view(n, k)


H = had(1024)
for k in kps[:4] + kps[-2:]:
    w = f.get_tensor(k).float()
    x = torch.randn(64, w.shape[1], device="cuda")
    ref = x @ w.t()
    e = lambda wq: float((x @ wq.t() - ref).norm() / ref.norm())
    S = torch.linalg.svdvals(w)
    en = (S ** 2).cumsum(0) / (S ** 2).sum()
    rs = [int((en < t).sum()) + 1 for t in (0.99, 0.999, 0.9999)]
    U, Sv, V = torch.linalg.svd(w, full_matrices=False)
    lr = {r: e((U[:, :r] * Sv[:r]) @ V[:r]) for r in (256, 512)}
    col = w.abs().amax(0); fila = w.abs().amax(1)
    # rotacion por bloques de 1024 en la entrada: W' = W H (la entrada se rotaria con H^T)
    wr = (w.view(w.shape[0], -1, 1024) @ H).view_as(w)
    er_rot = float(((x.view(64, -1, 1024) @ H).view_as(x) @ rtn8(wr, -1).t() - ref).norm() / ref.norm())
    print(f"{k.split('.kernel')[0]}: rango 99/99,9/99,99% {rs} | rango 256 {lr[256]:.1e} 512 {lr[512]:.1e} | "
          f"cresta col {float(col.max() / col.median()):.1f} fila {float(fila.max() / fila.median()):.1f} | "
          f"int8 canal {e(rtn8(w, -1)):.2e} g64 {e(rtn8(w, 64)):.2e} rotado+canal {er_rot:.2e}")
