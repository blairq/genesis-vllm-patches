"""SK-30 actual contra la version del commit anterior (sk30_prev_tmp.cu), mismas entradas: salida bit a bit."""
import os
src = open("/t/sk30_test.py").read().split("rel = lambda")[0]
exec(compile(src, "sk30", "exec"))
import torch
from vllm._genesis.kernels.ptx_lab import Kernel
defs = [f"-DNQ={NQ}", "-DPV8=1", "-DPHL=1"]
ka = Kernel("sk30_prev_tmp.cu", "sk30_decode", defs=defs, warps=4 * NQ); ka.cargar()
kb = Kernel("sk30_decode_1pasada.cu", "sk30_decode", defs=defs + [f"-DPING={os.environ.get('PING', '0')}"], warps=4 * NQ); kb.cargar()
a = [Qi, qs, kv, bt, bt.stride(0), sl, lim_, abase_, amask_, NH, BS, BLK, c.refs, GMAX, Op, Mp, Lp]
fprep()
Op.zero_(); ka.lanzar((GMAX, NH), a, shared=SH); torch.cuda.synchronize(); oa, ma, la = Op.clone(), Mp.clone(), Lp.clone()
Op.zero_(); kb.lanzar((GMAX, NH), a, shared=SH); torch.cuda.synchronize()
print(f"N={N} arbol={os.environ.get('ARBOL_AZAR')}: O igual {torch.equal(oa, Op)} M igual {torch.equal(ma, Mp)} L igual {torch.equal(la, Lp)}")
