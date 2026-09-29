import os, sys
src = open("/t/sk30_test.py").read().split("rel = lambda")[0]
exec(compile(src, "sk30", "exec"))
import torch
from vllm._genesis.kernels.ptx_lab import Kernel
defs = [f"-DNQ={NQ}", "-DPV8=1", "-DPHL=1"]
k1 = Kernel("sk30_v1_ref.cu", "sk30_decode", defs=defs, warps=4 * NQ); k1.cargar()
k2 = Kernel("sk30_decode_1pasada.cu", "sk30_decode", defs=defs + [f"-DPFIJA={os.environ.get('PFIJA', '2')}"], warps=4 * NQ); k2.cargar()
a = [Qi, qs, kv, bt, bt.stride(0), sl, lim_, abase_, amask_, NH, BS, BLK, c.refs, GMAX, Op, Mp, Lp]
fprep(); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
k1.lanzar((GMAX, NH), a, shared=SH); k2.lanzar((GMAX, NH), a, shared=SH)
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
