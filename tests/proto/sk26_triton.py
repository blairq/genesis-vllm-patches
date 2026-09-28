"""SK-26 en Triton (para comparar con el .cu): T0 = suma de cuadrados en fp32, T1 = en int32 sobre el int8.
Correccion contra la referencia, tiempo con grafos, y el SASS de cada uno (cuobjdump) para contar
instrucciones: FFMA/FMUL (camino float), IMAD/IDP (camino entero), HADD2/HFMA2, cargas y barreras."""
import os, re, subprocess, glob, torch, triton, triton.language as tl
dev = "cuda"; H = 5120; eps = 1e-6
torch.manual_seed(0); torch.zeros(1, device=dev)


@triton.jit
def _k(x_ptr, r_ptr, w_ptr, q_ptr, e_ptr, g_ptr, eps, H: tl.constexpr, B: tl.constexpr, ENTERO: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, B); m = c < H
    x = tl.load(x_ptr + t * H + c, mask=m, other=0.).to(tl.float32)
    r = tl.load(r_ptr + t * H + c, mask=m, other=0.).to(tl.float32)
    w = tl.load(w_ptr + c, mask=m, other=0.).to(tl.float32)
    s = x + r
    tl.store(r_ptr + t * H + c, s.to(tl.float16), mask=m)
    y = s * (1.0 + w)
    am = tl.maximum(tl.max(tl.abs(y), 0), 1e-10)
    qf = y * (127.0 / am)
    q = tl.extra.cuda.libdevice.rint(qf).to(tl.int8)
    tl.store(q_ptr + t * H + c, q, mask=m)
    if ENTERO:
        qi = q.to(tl.int32)
        ss = tl.sum(qi * qi, 0).to(tl.float32) * (am / 127.0) * (am / 127.0)
    else:
        ss = tl.sum(s * s, 0)
    g = tl.load(g_ptr)
    tl.store(e_ptr + t, am * tl.rsqrt(ss / H + eps) / 127.0 * g)


def ref(x, r, w, g):
    s = x.float() + r.float()
    y = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * (1 + w.float())).half().float()
    am = y.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    return s.half(), torch.round(y * (127 / am)).to(torch.int8), (am / 127 * g).squeeze(-1)


def lanzar(ent, x, r, w, q, e, g):
    return _k[(x.shape[0],)](x, r, w, q, e, g, eps, H=H, B=8192, ENTERO=ent, num_warps=8)


for T in (9, 8192):
    x = (torch.randn(T, H, device=dev) * 2).half(); r0 = (torch.randn(T, H, device=dev) * 20).half()
    w = torch.zeros(H, device=dev).half(); g = torch.tensor([0.37], device=dev)
    rn, qr, er = ref(x, r0, w, g)
    for ent in (0, 1):
        r = r0.clone(); q = torch.empty(T, H, dtype=torch.int8, device=dev); e = torch.empty(T, device=dev)
        lanzar(ent, x, r, w, q, e, g); torch.cuda.synchronize()
        dq = (q.int() - qr.int()).abs()
        print(f"T={T} T{ent}: res {'ok' if torch.equal(r, rn) else 'DIF'} q!= {float((dq>0).float().mean())*100:.2f}% max {int(dq.max())} esc {float(((e-er).abs()/er).max())*100:.3f}%")

for T in (9, 36, 512, 8192):
    x = torch.randn(T, H, device=dev).half(); r = torch.randn(T, H, device=dev).half(); w = torch.zeros(H, device=dev).half()
    g = torch.tensor([1.0], device=dev); q = torch.empty(T, H, dtype=torch.int8, device=dev); e = torch.empty(T, device=dev)
    out = []
    for ent in (0, 1):
        f = lambda: lanzar(ent, x, r, w, q, e, g)
        for _ in range(3): f()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s): f()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(100): f()
        for _ in range(3): gr.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
        for _ in range(20): gr.replay()
        e1.record(); torch.cuda.synchronize()
        out.append(f"T{ent} {e0.elapsed_time(e1)*1000/2000:7.2f}")
    print(f"tiempo T={T:5d}: " + " | ".join(out), flush=True)

# SASS: el cubin que dejo Triton en su cache
def contar(sass):
    ins = re.findall(r"^\s+/\*[0-9a-f]+\*/\s+([A-Z0-9_.]+)", sass, re.M)
    base = [i.split(".")[0] for i in ins]
    fam = {k: sum(1 for b in base if b == k) for k in sorted(set(base))}
    return len(ins), fam
for ent in (0, 1):
    k = lanzar(ent, torch.zeros(1, H, device=dev).half(), torch.zeros(1, H, device=dev).half(), torch.zeros(H, device=dev).half(),
               torch.empty(1, H, dtype=torch.int8, device=dev), torch.empty(1, device=dev), torch.ones(1, device=dev))
    cub = k.asm["cubin"]
    open(f"/t/sass_sk26/triton_T{ent}.cubin", "wb").write(cub); open(f"/t/sass_sk26/triton_T{ent}.ptx", "w").write(k.asm["ptx"])
    open("/tmp/k.cubin", "wb").write(cub)
    sass = subprocess.run(["cuobjdump", "-sass", "/tmp/k.cubin"], capture_output=True, text=True).stdout
    n, fam = contar(sass)
    sel = {a: fam.get(a, 0) for a in ("FFMA", "FMUL", "FADD", "FMNMX", "IMAD", "IDP", "IADD3", "HADD2", "HFMA2", "LDG", "STG", "SHFL", "BAR", "F2I", "F2F", "I2F", "MUFU")}
    print(f"SASS Triton T{ent}: {n} instrucciones, regs {k.n_regs}: {sel}")
