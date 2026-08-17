# P2P entre las dos RTX 3090: cómo se logró y qué dio

**El all-reduce de TP=2 era el 31,8% del tiempo de GPU. Con P2P funcionando el
prefill de un hilo de 126k tokens bajó de 133,9 s a 103,6 s (−22,6%).**

Hicieron falta **cuatro** cosas y ninguna alcanzaba sola. Este documento existe
porque tres de ellas parecían suficientes y no lo eran.

---

## 1. El resultado

Carga mixta (`tests/repro/carga_mixta.py`): un hilo de 126k tokens + 5 agentes.

| | 580.178.04 sin P2P | **610.57.04 + fork, con P2P** |
|---|---|---|
| hilo largo (126k tok) | 133,9 s | **103,6 s** (−22,6%) |
| TTFT de los agentes | 127,4 s | **97,7 s** (−23,3%) |
| inter-token bajo carga | 51 ms | **42 ms** (−17,6%) |
| inter-token en vacío | 31 ms | 28 ms |
| KV en GPU | 361.729 tok | **374.909 tok** (+13.180) |

A nivel de la colectiva (`tests/repro/nccl_allreduce_bench.py`), con
verificación numérica en cada tamaño:

| tamaño | con P2P | sin P2P (`NCCL_P2P_DISABLE=1`) | ganancia |
|---|---|---|---|
| 4 KiB | 0,15 GiB/s | 0,14 GiB/s | — (manda la latencia) |
| 1 MiB | 8,53 GiB/s | 5,32 GiB/s | 1,60× |
| **15,6 MiB** (chunk de prefill) | **10,66 GiB/s** | **6,34 GiB/s** | **1,68×** |
| 64 MiB | 10,87 GiB/s | 6,36 GiB/s | 1,71× |

**Sin tocar una línea del compose**: el driver nuevo lo aprovecha solo.

Nota: los 6,34 GiB/s del control coinciden con lo que medía el offloading de KV
(6,3-6,6 GiB/s). Ese siempre fue el número real del camino por host.

---

## 2. Las cuatro piezas

| pieza | por sí sola |
|---|---|
| **vBIOS con ReBAR** en la Zotac (BAR1 256 MiB → 32 GiB) | no habilita P2P |
| **`iommu=pt`** en GRUB | no habilita P2P |
| `ForceP2P=529` sobre el driver **stock** | P2P "OK" pero **CORROMPE DATOS** |
| **módulos de [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules)** sobre las dos primeras | **funciona** |

### Por qué el `ForceP2P` sobre el driver stock no sirve

`kern_bus_gp100.c` tiene tres caminos de P2P:

```c
if (..._CONNECTION_TYPE, _NVLINK, ...)     return ...ForNvlink_HAL(...);
if (..._CONNECTION_TYPE, _PCIE_BAR1, ...)  return ...ForBar1P2P_HAL(...);  // solo en el FORK
if (..._CONNECTION_TYPE, _PCIE, ...)       return ...ForMailbox_HAL(...);  // en el stock
```

`ForceP2P=529` sólo cambia `p2pOverride`: hace que el driver **declare** P2P.
Sin la rama `_PCIE_BAR1` cae al **mailbox**, que en estas placas produce
**corrupción silenciosa**:

```
BAR 32 GB + ForceP2P:   directo 12,56 GiB/s | host 12,27 GiB/s | *** CORRUPCION ***
BAR 32 GB sin ForceP2P: directo  5,31 GiB/s | host 12,27 GiB/s | integridad OK
```

Por eso NCCL y `CustomAllreduce` giraban al 99% de CPU: esperaban flags de
sincronización que nunca llegaban correctos. **No usar `ForceP2P` con el driver
stock en este hardware.**

El fork usa `_PCIE_BAR1`, que es mapeo directo — y **por eso necesita el BAR1
grande**, que es lo que dio el flasheo del vBIOS.

---

## 3. El procedimiento

Estado de partida: driver 580.178.04 por paquetes de Ubuntu, vBIOS de la Zotac
ya flasheado (ver `vbios-backup/LEEME.md`), `iommu=pt` en GRUB.

```bash
# 1. compilar el fork ANTES de tocar nada (si no compila, no se sigue)
mkdir -p ~/Proyectos/p2p-build && cd ~/Proyectos/p2p-build
git clone --depth 1 https://github.com/aikitoria/open-gpu-kernel-modules.git
cd open-gpu-kernel-modules && make modules -j$(nproc)
cat version.mk | head -2     # tiene que decir 610.57.04

# 2. bajar el userspace de esa MISMA version
cd .. && curl -LO https://us.download.nvidia.com/XFree86/Linux-x86_64/610.57.04/NVIDIA-Linux-x86_64-610.57.04.run

# 3. liberar las GPUs
docker compose -f ~/Proyectos/genesis-vllm-patches/compose/docker-compose.qwen38-27b-fp8.yml down
docker stop spark-dashboard
sudo systemctl stop nvidia-persistenced
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia

# 4. sacar el driver de Ubuntu (17 paquetes; guardar la lista para revertir)
sudo apt-get remove --dry-run 'nvidia-driver-580*' 'libnvidia-*-580' \
  'nvidia-dkms-580*' 'nvidia-kernel-*-580*' 'xserver-xorg-video-nvidia-580' \
  | grep '^Remv' | awk '{print $2}' > paquetes-removidos.txt
sudo apt-get remove -y 'nvidia-driver-580*' 'libnvidia-*-580' \
  'nvidia-dkms-580*' 'nvidia-kernel-*-580*' 'xserver-xorg-video-nvidia-580'

# 5. userspace SIN sus modulos (usamos los del fork)
sudo sh ./NVIDIA-Linux-x86_64-610.57.04.run --silent --no-questions \
  --ui=none --no-x-check --no-kernel-modules

# 6. modulos del fork
cd open-gpu-kernel-modules && sudo make modules_install && sudo depmod -a
sudo modprobe nvidia && sudo modprobe nvidia_uvm
sudo systemctl start nvidia-persistenced
```

### Detalles que cuestan tiempo si no se saben

- **El `.run` se niega** si están los paquetes de Ubuntu:
  `ERROR: The installation was canceled due to the availability or presence of
  an alternate driver installation`. Hay que sacarlos primero (paso 4).
- **`libnvidia-container` / `nvidia-container-toolkit` NO se tocan** en el paso
  4 — verificado con `--dry-run`. Docker conserva el acceso a GPU.
- `nvidia-persistenced` **sí** se remueve; lo repone el `.run`. Y tiene que
  estar arriba **antes** que los contenedores, o fallan con
  `open /run/nvidia-persistenced/socket: no such file or directory`.
- **`spark-dashboard` retiene `/dev/nvidia-uvm`** aunque sólo lea NVML, y
  bloquea el `rmmod`. Bajarle las capabilities a `[utility]` **no alcanza**
  (probado). Hay que pararlo.
- `make modules_install` avisa `missing 'System.map' file. Skipping depmod` —
  por eso el `depmod -a` explícito.
- Recargar módulos puede dejar una placa con `WPR2 already up` y `nvidia-smi -L`
  mostrando una sola. Se arregla sin reiniciar:
  `sudo bash -c 'echo 1 > /sys/bus/pci/devices/0000:0f:00.0/reset'`

### Revertir

```bash
sudo apt install -y $(cat ~/Proyectos/p2p-build/paquetes-removidos.txt)
```

---

## 4. Cómo se verifica (integridad ANTES que velocidad)

Este orden no es una formalidad: el `ForceP2P` medía **12,56 GiB/s con datos
corruptos**. Un test de ancho de banda sin verificación de correccion habría
dado "éxito".

```bash
cd ~/Proyectos/genesis-vllm-patches

# 1. el driver declara P2P
nvidia-smi topo -p2p r          # OK en ambos sentidos, SIN ForceP2P
nvidia-smi -q | grep -A2 "BAR1 Memory" | grep Total   # 32768 MiB en las dos

# 2. integridad + control contra el camino por host
docker run --rm --gpus all -v $PWD/tests:/tests --entrypoint python3 \
  vllm/vllm-openai:v0.23.0 /tests/repro/p2p_bandwidth.py --mib 128

# 3. la colectiva real, con y sin P2P para aislar
docker run --rm --gpus all --ipc=host --shm-size=2gb -v $PWD/tests:/tests \
  --entrypoint python3 vllm/vllm-openai:v0.23.0 \
  /tests/repro/nccl_allreduce_bench.py
# y de nuevo con -e NCCL_P2P_DISABLE=1: la diferencia es la prueba
```

⚠️ **`p2p_bandwidth.py` da razón 1,00× aunque el P2P funcione.** `torch.copy_`
no ejercita el camino P2P: hace su propio staging. Sirve para el chequeo de
**integridad**, no para decidir si hay P2P. Para eso está el benchmark de NCCL.

---

## 5. Lo que NO funcionó

**`--disable-custom-all-reduce` sacado del compose.** El all-reduce propio de
vLLM tiene sentido con P2P (suele ganarle a NCCL con 2 GPUs), pero el engine no
arranca:

```
Using ['CUSTOM','PYNCCL'] all-reduce backends for group 'tp:0'
Failed: Cuda error csrc/custom_all_reduce.cuh:455 'invalid argument'
-> RuntimeError: Engine core initialization failed
```

⚠️ **`tests/repro/p2p_ipc_allreduce.py` da OK** (all-reduce de 15,6 MiB por IPC
entre procesos, resultado numérico correcto) **y sin embargo el engine falla**.
El test **no ejercita la captura de CUDA graphs**, y `CustomAllreduce` tiene un
camino aparte para eso (`capture()` / `register_graph_buffers`). Ahí rompe.

Si algún día se resuelve, además hay que subirle el `max_size`: son 8 MiB por
defecto y los all-reduce de prefill son de 15,6 MiB, así que caerían a NCCL
igual. El test verificó hasta 32 MiB.

---

## 6. Pendiente

- **ACS del switch PLX.** Los dos puertos (`0e:00.0`, `0e:10.0`) tienen
  `ACSCtl: SrcValid+ ReqRedir+ CmpltRedir+ UpstreamFwd+`, que redirige el
  tráfico GPU-GPU al root complex. La eficiencia medida es 10,87 de 15,75 GB/s
  teóricos (**74%**), así que puede quedar margen. Se apaga con
  `pcie_acs_override=downstream,multifunction`, pero **el kernel de Ubuntu no
  trae ese parche**: haría falta kernel custom, o la opción en el BIOS.
- **Re-perfilar con `torch.profiler`** para ver el nuevo reparto. El 31,8% del
  all-reduce se midió con el driver viejo y con `max-num-batched-tokens 4096`;
  hoy corre en 1600.
- **`CustomAllreduce` bajo CUDA graphs** (ver §5).
