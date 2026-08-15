# Cache de KV en dos capas (RAM + NVMe)

Cómo evitar que un prefijo largo se recalcule cuando los subagentes lo
desalojan de la VRAM.

**Medido en este rig** (2× RTX 3090, TP=2):

```
prefill FRÍO de un hilo de 92k tokens ............ 51,5 s
  ... 8 agentes de ~45k barren la VRAM ...
vuelta al hilo, recuperado del cache ............   2,8 s   (5,4% del frío)
```

Extrapolado a un hilo de 220k, es la diferencia entre esperar minutos y
esperar segundos.

---

## 1. Qué hace

vLLM mantiene un *prefix cache* en VRAM. Cuando se llena, los bloques viejos
se **descartan** y hay que recalcularlos. Con el offloading, en vez de
descartarse bajan a RAM y de ahí al NVMe, y vuelven por PCIe cuando se los
necesita.

Dos detalles del diseño que importan:

- **El guardado es proactivo, no al desalojar.** `_build_store_jobs()` corre
  en cada paso del scheduler, así que los bloques se copian a RAM *mientras
  se computan*. Cuando los agentes llenan la VRAM, la copia ya existe.
- **La recuperación es automática.** En cada request, el scheduler llama
  `get_num_new_matched_tokens()`, que consulta el tier de CPU por *más*
  tokens allá de los que quedaron en VRAM, y los carga en vez de recalcular.

No hace falta marcar qué hilo proteger: con `eviction_policy: arc` la
política lo hace sola (ver §5).

---

## 2. Cómo se activa

⚠️ **No sirve `--kv-offloading-size`.** Ese flag solo setea `cpu_bytes_to_use`
y deja el spec por defecto (`CPUOffloadingSpec`: RAM sola, LRU). El tiering y
ARC requieren el JSON completo.

### Compose

```yaml
volumes:
  - /home/usuario/Proyectos/kv-offload:/kv-offload      # tier 2, en NVMe

environment:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
  - GENESIS_ENABLE_PN81_KV_DISK_QUOTA=1                 # cuota + purga
  - GENESIS_KV_DISK_MAX_GB=30
  - GENESIS_KV_DISK_CHECK_EVERY=2000
  - GENESIS_KV_DISK_TARGET_RATIO=0.85

command:
  - --enable-cumem-allocator
  - --kv-transfer-config
  - '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
      "kv_connector_extra_config":{
        "spec_name":"TieringOffloadingSpec",
        "cpu_bytes_to_use":5368709120,
        "eviction_policy":"arc",
        "secondary_tiers":[{"type":"fs","root_dir":"/kv-offload"}]}}'
```

Para aplicarlo a un compose existente sin tocar nada más:

```bash
python3 compose/apply_kv_offload.py docker-compose.<engine>.yml
```

---

## 3. Las tres reglas (cada una costó un fallo)

### 3.1 Bajar `--gpu-memory-utilization` unos **0,135**

`OffloadingConnector` exige `--enable-cumem-allocator`, y **cumem hace que el
profiler de memoria SOBREESTIME el KV disponible en ~3,2 GiB**. El engine
arranca, calcula un KV que no entra, y muere en
`_allocate_kv_cache_tensors`.

El error es **constante, no proporcional al util** — por eso bajarlo de a
poco no sirve. Medido en w8a16-mtp:

```
util 0.965 -> faltaban 419 MiB
util 0.93  -> faltaban 309 MiB
util 0.88  -> faltaban 209 MiB      (cada 0,05 recupera solo ~110 MiB)
```

La corrección es restar `3,2 / 23,56 = 0,135` al valor que funcionaba sin
offloading. Valores en uso:

| engine | sin offload | con offload |
|---|---|---|
| w8a16-mtp | 0,965 | **0,82** |
| w4a16 | 0,94 | **0,805** |
| ektome | 0,94 | **0,78** |

### 3.2 `max_split_size_mb:512` es OBLIGATORIO

Junto con `expandable_segments:True`. Sin él, y **solo cuando cumem está
activo**, la captura de CUDA graphs falla:

```
_dummy_run -> sm.fill_(-1) -> torch.AcceleratorError: CUDA error: invalid argument
```

y el engine queda colgado repitiendo *"No available shared memory broadcast
block found in 60 seconds"*. Fue la única variable que diferenciaba al engine
que fallaba de los dos que andaban.

### 3.3 PN82 es obligatorio con TP>1 (bug de vLLM)

⚠️ **Si el engine muere en `sm.fill_(-1)` de forma intermitente, NO es la
VRAM.** Es un bug de vLLM y bajar `--gpu-memory-utilization` no lo arregla.

`pin_mmap_region()` (`v1/kv_offload/cpu/gpu_worker.py`) pinea el mmap del
tier de RAM con `cudaHostRegister`. Chequea el retorno, loguea un warning y
sigue — **sin consumir el error del contexto de CUDA**. El runtime lo deja
latcheado; PyTorch consulta ese estado en cada op, así que la primera op del
rank afectado explota lejos de la causa:

```
20:13:49  TP1  Created mmap file /dev/shm/vllm_offload_... (5.35 GB)
20:13:49  TP0  Opened existing mmap file (el mismo)
20:13:51  TP1  WARNING cudaHostRegister failed for rank=1 (code=1)
20:13:53  TP1  ERROR   sm.fill_(-1) -> CUDA error: invalid argument
```

Muere **el mismo rank** que falló el registro, y solo ese.

Con TP>1 el registro fallido es el caso *normal*, no una rareza: los dos
ranks mapean el mismo archivo de `/dev/shm` y ambos lo registran, así que el
segundo falla sobre las mismas páginas físicas. Que a veces arranque depende
de qué llamada consuma el error latcheado — de ahí la intermitencia que hace
imposible tunear el engine.

Medido, mismo compose sin tocar un solo parámetro (KV idéntico, 379.303, en
las tres corridas):

| corrida | `cudaHostRegister` | arranque | chat |
|---|---|---|---|
| 1 | falló | **FALLO** | — |
| 2 | OK | OK | HTTP 200 |
| 3 | falló | **FALLO** | — |

Repro determinístico (sin esperar al azar):

```python
torch.cuda.cudart().cudaHostRegister(0xdeadbeef, 4096, 0)  # -> code 1
torch.zeros(8, device="cuda").fill_(-1)
# AcceleratorError: CUDA error: invalid argument
```

Con `cudaGetLastError()` en el medio, el `fill_` funciona. **PN82 hace
exactamente eso** y es default ON (kill switch `GENESIS_DISABLE_PN82=1`).

⚠️ `torch.cuda.cudart()` **no expone** `cudaGetLastError` (torch 2.11.0+cu130
solo trae `cudaError` y `cudaGetErrorString`), así que PN82 lo llama por
`ctypes` sobre `libcudart`. Verificado post-fix: corrida con
`cudaHostRegister failed` → PN82 limpia → arranca y responde HTTP 200.

### 3.4 Dejar ~400 MiB para el workspace lazy de FlashInfer

Con MTP, `flashinfer.py:_get_workspace_buffer()` aloca **394 MiB** para el
wrapper de spec-decode prefill, y lo hace **lazy: en el primer request**, no
en el arranque. El profiler ya repartió toda la memoria al KV para entonces.

Síntoma: el engine arranca perfecto y muere en el primer request con

```
torch.OutOfMemoryError: Tried to allocate 394.00 MiB.
GPU 1 ... 89.00 MiB is free
  File flashinfer.py, line 781, in _get_workspace_buffer
```

Este sí es un problema de memoria real. Cualquier flag que libere VRAM y se
la ceda al KV (§8) tiene que dejar ese margen.

### 3.5 Sacar `restart: unless-stopped`

Un fallo de arranque reintenta en loop (medido: 9 reinicios) y parece que
"tarda en levantar" en vez de mostrarse como roto. Para detectar un reinicio
del proceso sin mirar el contenedor:

```bash
curl -s :8320/metrics -H "Authorization: Bearer $VLLM_API_KEY" | grep process_start_time_seconds
```

---

## 4. PN81 — cuota de disco y purga de `/dev/shm`

vLLM **no tiene ninguna de las dos**. Sin PN81:

- `root_dir` crece sin techo: **38 GB en una sola sesión de pruebas**. Los
  únicos `os.remove` del tier `fs` son manejo de errores, y la interfaz de
  secondary tier ni declara un método de desalojo.
- Los mmap de `/dev/shm` sobreviven al contenedor. vLLM los borra en
  `cleanup()`, pero eso solo corre en cierre ordenado; un `docker rm -f` los
  deja. Con `ipc: host` quedan en el `/dev/shm` del **host**. Medido: 4
  huérfanos de 5,3 GB llenaron los 16 GB y el arranque siguiente murió con
  `madvise: Bad address`.

PN81 implementa `on_schedule_end()` —hook que vLLM documenta para *"per-step
cleanup"* y deja vacío— para podar por antigüedad, y purga los huérfanos al
arrancar (saltea los que algún proceso tenga mapeados).

⚠️ **La cuota es POR RANK**: `FileMapper` escribe en `{base_path}_r{rank}`, así
que con TP=2 el disco total es `2 × GENESIS_KV_DISK_MAX_GB`.

---

## 5. Por qué ARC y no LRU

`eviction_policy: "arc"` separa lo accedido **una** vez de lo accedido
**varias**:

```
T1: bloques accedidos una vez      -> los subagentes efímeros caen acá
T2: bloques accedidos varias veces -> el hilo largo sube acá al volver
B1/B2: listas fantasma de lo recién desalojado
```

Con LRU, cuatro coders recientes valen más que tu prefijo de hace diez
minutos y te lo empujan. Con ARC, no. Y si igual se desaloja una vez, el hit
en la lista fantasma hace que ARC **ajuste la partición** para que no se
repita.

**No se puede darle pistas por request.** La política recibe solo
`OffloadKey = hash(bloque) + group_idx`; ni `priority`, ni
`kv_transfer_params`, ni `cache_salt` llegan hasta ahí (`cache_salt` entra en
el hash: *aísla* prefijos, no los protege). La única palanca real es que
**acceder promueve**: mandar un request barato con el mismo prefijo lo sube a
T2.

> El tier **secundario** sí recibe `ReqContext` (con `kv_transfer_params`) en
> `on_new_request()`, y devuelve el `RequestOffloadingContext` con su
> `OffloadPolicy`. O sea que un tier propio *podría* decidir por request. El
> `fs` que viene lo ignora: `return RequestOffloadingContext()`.

---

## 6. Troubleshooting

| Síntoma | Causa | Fix |
|---|---|---|
| OOM en `_allocate_kv_cache_tensors` | cumem sobreestima el KV | bajar `util` (§3.1) |
| `sm.fill_(-1)` CUDA invalid argument, **intermitente** con TP>1 | `cudaHostRegister` fallido deja el error latcheado (bug de vLLM) — NO es la VRAM | PN82, §3.3 |
| OOM de 394 MiB en el **primer request**, `_get_workspace_buffer` | workspace de FlashInfer alocado lazy tras el profiling | dejar margen, §3.4 |
| Colgado en *"No available shared memory broadcast block"* + `sm.fill_(-1)` CUDA invalid argument | falta `max_split_size_mb` | §3.2 |
| Colgado en lo mismo **sin** error CUDA | está compilando; puede tardar >10 min | esperar; mirar CPU/GPU |
| `madvise: Bad address` al arrancar | `/dev/shm` lleno de mmaps huérfanos | PN81 los purga; si no, borrarlos |
| `Please specify kv_role` | falta en el JSON | `"kv_role":"kv_both"` |
| `incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | falta cumem | `--enable-cumem-allocator` |
| El disco crece sin parar | PN81 apagado | `GENESIS_ENABLE_PN81_KV_DISK_QUOTA=1` |

---

## 7. Estado de los engines

| engine | pesos | util | KV en GPU |
|---|---|---|---|
| ektome (GPTQ 4-bit) | 8,8 GiB | 0,78 | **665.096** |
| w4a16 (AWQ-GPTQ) | 12,6 GiB | 0,805 | 459.850 |
| w8a16-mtp (INT8) | 14,5 GiB | 0,82 | 379.303 |

Ektome rinde mucho más KV porque **cuantiza el doble de tensores**: 1600
contra 789 del AutoRound (borrado), que dejaba 936 en bf16 al excluir todo
`linear_attn` (las 48 capas DeltaNet).

---

## 8. Flags que liberan VRAM para el KV (medido en w8a16-mtp)

Medido 2026-08-15, TP=2, `util 0.82`. Los tres números salen del profiler y
son reproducibles; los crashes que aparecieron durante el barrido eran PN82
(§3.3), **no** los flags.

| config | KV en GPU | delta |
|---|---|---|
| baseline | 379.303 | — |
| `+ --mm-processor-kwargs` | 399.806 | **+20.503** (+5,4%) |
| `+ --max-num-batched-tokens 4096` | 430.560 | **+51.257** (+13,5%) |

### `--mm-processor-kwargs '{"max_pixels":2000000,"min_pixels":65536}'`

El profiler corre un dummy multimodal con el ítem **más grande permitido**.
Sin este flag usa el `max_pixels` del `preprocessor_config.json` del modelo y
reserva activaciones para una imagen enorme que nunca se manda. 2.000.000 px
≈ 1414×1414, de sobra para el contrato de una imagen por request.

Complementa a `--limit-mm-per-prompt '{"image":1,"video":0}'`: ese acota
*cuántos* ítems, este acota *qué tan grande* es cada uno.

### `--max-num-batched-tokens 4096`

Achica el pico transitorio de las capas GDN, que escala lineal con los
tokens del batch (12,05 KiB/token/GPU, ver PN80). **Costo**: parte el
prefill en chunks la mitad de grandes, así que un prompt de 200k tarda más
en procesarse. Es un cambio de KV por throughput de prefill, no gratis.

⚠️ Al subir el KV hay que verificar el margen de §3.4: el workspace de
FlashInfer (394 MiB) se aloca en el **primer request**, no en el arranque.
