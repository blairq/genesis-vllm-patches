"""Una invocacion de cada kernel (Triton y PTX) para ncu: cadena r=5, 1 pedido."""
import torch
exec(open("/t/gdn_arbol_ptx.py").read().split("ok = True")[0])
c = caso(1, torch.float16, True, 5)
correr_triton(c); correr_ptx(c); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
correr_triton(c); correr_ptx(c); torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
