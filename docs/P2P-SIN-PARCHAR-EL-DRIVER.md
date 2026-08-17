# P2P entre dos RTX 3090 sin parchar el driver — qué funcionó y qué no

**Veredicto: el P2P se habilita con una sola clave de registry del driver stock,
y para `cudaMemcpy` dentro de un proceso funciona (12,58 GiB/s medidos). Pero
NO sirve para nada de lo que necesita vLLM: todo camino que use CUDA IPC entre
procesos se cuelga — NCCL y el all-reduce propio de vLLM por igual. Revertido.**

**El bloqueo real y unico es el BAR1 de 256 MB de la Zotac.** No es el driver,
no es el ACS, no es el IOMMU, no es NCCL. Sin ese firmware no hay camino por
software: esta comprobado en el codigo Y medido en las dos implementaciones.

---

## 1. Por qué se buscó esto

Profile con `torch.profiler` sobre un prefill de 42k tokens (rank0):

| familia | ms | % |
|---|---|---|
| GEMM Marlin (pesos FP8) | 16.751 | 50,4% |
| **all-reduce TP (NCCL)** | **10.569** | **31,8%** |
| atención (FlashInfer) | 4.461 | 13,4% |
| kernels GDN (FLA) | 322 | 1,0% |

Casi un tercio del tiempo la GPU espera la PCIe: 128 all-reduce de 16 MB por
forward (2 por capa × 64 capas), y sin P2P cada byte cruza PCIe **dos veces**,
rebotando por la RAM del host.

Todas las palancas de vLLM estaban cerradas: sequence parallelism y la fusión
allreduce+RMSNorm sólo tienen tablas para capability 90 y 100
(`SP_MIN_HIDDEN_SIZE`, `FI_ALLREDUCE_FUSION_MAX_SIZE_MB` — Ampere no figura);
pipeline parallel no existe para `Qwen3_5ForConditionalGeneration` (no
implementa `SupportsPP`); y `NCCL_PROTO=Simple` no movió nada (133,9 → 133,1 s).

---

## 2. El hallazgo: no hace falta el fork

Punto de partida: <https://github.com/aikitoria/open-gpu-kernel-modules>. Su
README pide driver 610.57.04, `iommu=pt`, ACS off y —por el mecanismo— BAR1
grande en las dos placas.

**Pero el P2P de NVIDIA no es uno solo.** El diff del fork, en
`kern_bus_gp100.c`:

```c
kbusCreateP2PMapping_GP100(...)
{
    if (..._CONNECTION_TYPE, _NVLINK, ...)  return ...ForNvlink_HAL(...);
+   if (..._CONNECTION_TYPE, _PCIE_BAR1, ...) return ...ForBar1P2P_HAL(...);  // lo AGREGA el fork
    if (..._CONNECTION_TYPE, _PCIE, ...)    return ...ForMailbox_HAL(...);    // YA existe en stock
}
```

Y en `kernel_bif.c`, estas líneas son **contexto del diff, no cambios** — ya
están en el driver stock:

```c
  if (osReadRegistryDword(pGpu, NV_REG_STR_CL_FORCE_P2P, &data32) == NV_OK)
      pKernelBif->p2pOverride = data32;
```

El fork sólo cambia el **default**. Ese default se reproduce desde afuera:

```c
// src/nvidia/interface/nvrm_registry.h:1082-1094
#define NV_REG_STR_CL_FORCE_P2P          "ForceP2P"
#define NV_REG_STR_CL_FORCE_P2P_READ     1:0   // ENABLE=1
#define NV_REG_STR_CL_FORCE_P2P_WRITE    5:4   // ENABLE=1
#define NV_REG_STR_CL_FORCE_P2P_ATOMICS  9:8   // DEFAULT=2

READ=1<<0 | WRITE=1<<4 | ATOMICS=2<<8  =  0x211  =  529
```

```
options nvidia NVreg_RegistryDwords="ForceP2P=529"
```

**Resultado inmediato, con el driver 580.178.04 stock, sin compilar nada:**

```
nvidia-smi topo -p2p r:   CNS  →  OK   (lectura y escritura)
can_device_access_peer:   True en ambos sentidos
copia gpu0→gpu1:          12,58 GiB/s   (~86% del techo de Gen4 x8)
```

Ese 12,58 GiB/s es la prueba de que el tráfico cruza el switch directamente:
rebotando por el host cada byte cruza dos veces y el efectivo queda en ~6-7
GiB/s, que es justo lo que medía el offloading de KV (6,3-6,6 GiB/s).

---

## 3. Y sin embargo, vLLM no arranca

Con `ForceP2P=529` activo, el engine **se cuelga en la inicializacion de NCCL**:

```
(Worker pid=150) INFO [pynccl.py:113] vLLM is using nccl==2.28.9
   <- y nada mas. +10 minutos. Procesos vivos, VRAM en 454 MiB, sin error.
```

### Se probo el camino sin NCCL, y tambien se cuelga

vLLM tiene un all-reduce propio que **no usa NCCL**: `CustomAllreduce`, el que
apaga `--disable-custom-all-reduce`. Asigna UN buffer fijo (`max_size`, 8 MiB
por defecto), lo comparte por handle IPC y escribe directo en el del vecino con
su propio kernel CUDA.

La hipotesis era que ese camino si entraria: NCCL registra buffers ARBITRARIOS
del usuario, mientras que el custom mapea uno CHICO Y FIJO, que en 256 MB de
BAR1 deberia caber.

`tests/repro/p2p_ipc_allreduce.py` instancia la clase real de vLLM en dos
procesos (uno por GPU, como el engine) y le pide un all-reduce de 1600 tokens
x 5120 = 16,4 MiB con `max_size` subido a 32 MiB:

```
sin ForceP2P:  CustomAllreduce quedo DESHABILITADO      <- el test es valido
con ForceP2P:  *** SE COLGO *** (>150s)
```

**Se cuelga igual.** Y eso es lo concluyente: NCCL y `CustomAllreduce` son
implementaciones distintas que comparten **un solo mecanismo** — CUDA IPC entre
procesos (`cudaIpcOpenMemHandle` + escrituras peer).

### La linea exacta donde esta el limite

| mecanismo | quien lo usa | ¿anda con mailbox P2P? |
|---|---|---|
| `cudaMemcpyPeer` dentro de UN proceso | el test de ancho de banda | **si** — 12,58 GiB/s |
| **CUDA IPC entre procesos** | **NCCL y CustomAllreduce** | **no** — se cuelga |

El P2P por mailbox mapea una ventana con traduccion de direcciones, suficiente
para que el runtime copie de una GPU a otra dentro de un proceso. Pero exportar
memoria de GPU por IPC para que **otro proceso** la escriba directo necesita que
esa memoria sea alcanzable por PCIe de verdad, que es lo que da el BAR1 grande.

Y vLLM con TP=2 corre **un proceso por rank** (`multiproc_executor`), asi que
IPC no es opcional: es como estan hechos los dos caminos.

**Conclusion: el BAR1 era el bloqueo real desde el principio.** El razonamiento
inicial era correcto; lo unico que estaba mal era creer que `cudaMemcpy` y el
IPC comparten requisitos.

---

## 4. Daño colateral: la GPU se traba al recargar módulos

Descargar y recargar el driver dejó una de las placas sin inicializar:

```
NVRM: _kgspBootGspRm: unexpected WPR2 already up, cannot proceed with booting GSP
NVRM: (the GPU is likely in a bad state and may need to be reset)
NVRM: GPU 0000:0f:00.0: RmInitAdapter failed! (0x62:0x40:2028)
```

`nvidia-smi -L` mostraba **una sola** GPU, aunque `lspci` seguía viendo las dos.
El GSP (el procesador de firmware de la GPU) quedó con su región WPR2 marcada
como activa del load anterior.

**No es daño y no hace falta reiniciar.** Se resuelve con un reset PCI:

```bash
sudo systemctl stop nvidia-persistenced
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia
sudo bash -c 'echo 1 > /sys/bus/pci/devices/0000:0f:00.0/reset'
sudo modprobe nvidia && sudo modprobe nvidia_uvm
sudo systemctl start nvidia-persistenced
nvidia-smi -L    # tienen que verse las DOS
```

⚠️ Esto **no** tiene nada que ver con el P2P: es el `rmmod`/`modprobe` en sí
sobre un driver con GSP. Vale para cualquier recarga de módulos en este equipo.

---

## 5. Las dos trampas de operación

**Encontrar qué tiene las GPUs tomadas.** `fuser -v /dev/nvidia*` no mostraba
nada y `nvidia_uvm` seguía con refcount 4. Lo que sirvió:

```bash
sudo bash -c 'for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q nvidia \
  && echo "PID $(basename $p) $(tr -d "\0" < $p/comm)"; done'
```

El culpable era **`spark-dashboard`**, que abre `/dev/nvidia-uvm` aunque sólo
lea métricas por NVML. Se probó bajarle las capabilities a `[utility]` en su
compose: **no alcanza**, el toolkit monta el nodo igual y el proceso lo abre
(verificado: `Capabilities:[["utility"]]` y el fd seguía ahí). **Hay que parar
ese contenedor antes de recargar módulos.**

**`nvidia-persistenced` tiene que estar arriba antes que los contenedores.** Si
no, fallan con:

```
docker: Error response from daemon: ... open /run/nvidia-persistenced/socket:
no such file or directory
```

---

## 6. Estado actual: revertido

- `ForceP2P` **desactivado**. El archivo quedó en `docs/99-nvidia-p2p.conf.disabled`
  como referencia; para reactivarlo va a `/etc/modprobe.d/`.
- Verificado tras revertir: `RegistryDwords: ""`, las dos GPUs enumeradas,
  engine levantando normal.
- `spark-dashboard` con su compose original.

---

## 7. Qué haría falta para que esto sirva de verdad

En orden de viabilidad:

1. **Puente NVLink.** Es el camino que el propio fork lista como preferido para
   3090 (*"Pairwise NVLink where available, PCIe BAR1 otherwise"*), no necesita
   BAR1 grande, no pasa por el switch PLX (el ACS deja de importar) y da
   ~112 GB/s contra los ~15 de PCIe Gen4 x8. Hoy `nvidia-smi nvlink -s` dice
   *"all links are inActive"*: no hay puente puesto. La incógnita es física —
   las placas están en un expansor con switch Broadcom PEX880xx, no en dos
   slots x16 de la madre, así que hay que medir la separación entre conectores.
2. **vBIOS con ReBAR en la Zotac.** Destraba el camino BAR1 que NCCL necesita.
   Es el único bloqueo que queda del lado del firmware.
3. ~~Evitar NCCL usando el all-reduce propio de vLLM~~ — **probado y descartado**:
   `CustomAllreduce` se cuelga igual, porque comparte con NCCL el mecanismo de
   CUDA IPC entre procesos. Ver §3.

---

## 8. Lo que quedó descartado, medido y no supuesto

| hipótesis | veredicto |
|---|---|
| hace falta el driver 610.57.04 | **no**: la clave existe en el 580.178.04 stock |
| hay que compilar los módulos del fork | **no**: sólo cambia un default configurable |
| el ACS bloquea el P2P | **no fue un problema medible**: 12,58 GiB/s con ACS activo (`ReqRedir+ UpstreamFwd+` en los puertos del PEX880xx) |
| BAR1 grande en las dos placas | **sí, y es EL bloqueo**. Vale para NCCL y para CustomAllreduce: los dos usan CUDA IPC entre procesos. Sólo `cudaMemcpy` dentro de un proceso se salva con el mailbox |
| `NCCL_PROTO=Simple` ayuda | **no**: 133,9 → 133,1 s, ruido |
| el `pcie.link.gen 1` es un problema | **no**: es ahorro de energía en reposo; bajo carga entrena a Gen4 x8 |

Pendiente y sin relación con esto: `iommu=pt` ya quedó armado en GRUB
(`/etc/default/grub`, backup en `grub.bak.20260817-025421`), entra en el
próximo arranque.

---

## Referencias

- Fork: <https://github.com/aikitoria/open-gpu-kernel-modules> (commit `6c244435`)
- `src/nvidia/interface/nvrm_registry.h:1082-1094` — definición de `ForceP2P`
- `src/nvidia/src/kernel/gpu/bus/arch/pascal/kern_bus_gp100.c` — dispatch de P2P
- `src/nvidia/src/kernel/gpu/bif/kernel_bif.c` — `_kbifInitRegistryOverrides`
- `tests/repro/p2p_bandwidth.py` — mide el camino directo contra el control por host
- Costo de la comunicación: comentario de `--tensor-parallel-size` en
  `compose/docker-compose.qwen38-27b-fp8.yml`
