import os, sys, runpy
sys.argv = ["sk30_test.py", sys.argv[1] if len(sys.argv) > 1 else "62000"]
os.environ["SK30_SOLO"] = "1"
src = open("/t/sk30_test.py").read().split("rel = lambda")[0]
exec(compile(src, "sk30", "exec"))
import torch
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart()
k30.lanzar((GMAX, NH), [Qi, qs, kv, bt, bt.stride(0), sl, NH, BS, BLK, L, G, 2.0 ** (ev - 15), 1.0, GMAX, Op, Mp, Lp], shared=SH)
torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
