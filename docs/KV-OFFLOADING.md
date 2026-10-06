> **Note (2026-10-06): this write-up predates two fixes and is partly superseded.** On this hybrid
> model with speculative decoding the external (L2/L3) lookup never hit: the GDN groups were treated
> as EAGLE groups and the reuse state sat in the wrong block. PN168 and PN170 fix both. The rig now
> runs **L2 in RAM only (12 GiB), no L3**. The current numbers and reasoning are in the
> [README, *Tiered KV cache*](../README.md#tiered-kv-cache-l2-in-ram); the sizing rule below assumed
> the pre-fix lookup.

# Tiered KV cache (RAM + NVMe)

Keeping a long prefix from being recomputed after subagents evict it from VRAM.

**Status, measured 2026-09-19** on 2× RTX 3090, TP=2, Qwen3.8-27B + DFlash2:

| | |
|---|---|
| Synthetic rescue (same 20K prompt, evicted from GPU, requested again) | **6.88 s → 1.21 s (−82.5%)**, 612 MiB returned over `CPU_to_GPU`, reproducible across 3 runs |
| Real traffic since the last counter reset | **0 external hits** out of 97 623 queried tokens |
| What actually carries the load | L1, the VRAM prefix cache: **76.4%** hit rate |

Read that table before trusting this feature. The write path works, the read
path works, and the end-to-end win is real *when the whole prefix hits* — but
under the live workload that condition has not yet been met. §4 explains why,
and it is a design constraint, not a bug we can patch away.

---

## 1. What it does

vLLM keeps a prefix cache in VRAM. When it fills up, old blocks are
**discarded** and must be recomputed. With offloading they instead sink to RAM
(L2) and from there to NVMe (L3), and come back over PCIe when needed.

Two design details that matter:

- **Stores are proactive, not on eviction.** `_build_store_jobs()` runs on
  every scheduler step, so blocks are copied to RAM *while they are being
  computed*. By the time agents fill VRAM, the copy already exists.
- **Retrieval is automatic.** On each request the scheduler calls
  `get_num_new_matched_tokens()`, which asks the CPU tier for *more* tokens
  than remain in VRAM and loads them instead of recomputing.

No need to mark which thread to protect: `eviction_policy: arc` handles that
(§7).

---

## 2. Production configuration

From `compose/docker-compose.qwen38-27b-noon-dflash2-v029.yml`, the container
serving today. These are the values in use, not a template.

```yaml
volumes:
  - /home/usuario/Proyectos/kv-offload:/kv-offload    # L3, on NVMe

environment:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
  - GENESIS_ENABLE_PN81_KV_DISK_QUOTA=1
  - GENESIS_KV_DISK_MAX_GB=64        # must exceed L1 — see §6
  - GENESIS_DISABLE_PN97=1           # see §3.1
  - GENESIS_PN91_MAX_DEFER_SECONDS=15.0   # see §3.2
  - GENESIS_PN91_MAX_DEFER_STEPS=400
  - GENESIS_PN100_RING_BLOCKS=32     # do not disable — see §7

command:
  - --enable-cumem-allocator
  - --gpu-memory-utilization
  - "0.92"
  - --kv-transfer-config
  - '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
      "kv_connector_extra_config":{
        "spec_name":"TieringOffloadingSpec",
        "cpu_bytes_to_use":2147483648,
        "eviction_policy":"arc",
        "store_threshold":1,
        "secondary_tiers":[{"type":"fs","root_dir":"/kv-offload",
                            "n_read_threads":4,"n_write_threads":4}]}}'
```

⚠️ **`--kv-offloading-size` does not work.** That flag only sets
`cpu_bytes_to_use` and leaves the default spec (`CPUOffloadingSpec`: RAM only,
LRU). Tiering and ARC require the full JSON.

To apply it to an existing compose without touching anything else:

```bash
python3 compose/apply_kv_offload.py docker-compose.<engine>.yml
```

---

## 3. The three patches that had it deadlocked

Until 2026-09-19 this subsystem wrote **50 GB per run and had never returned a
single hit**. The cause was not vLLM. It was three of this project's own
patches, each correct when written and broken by a later change to another
patch. Full diagnosis in `vllm/_genesis/diag_offload.py`.

### 3.1 PN97 — the deadlock

PN97 says *"an L3→L2 promotion may only use genuinely free slots"*. It was
written when L2 held 222 slots and a prompt cost ~36, so six prompts fit and
the rule was sound.

PN145/PN146 then raised the GPU block to 880 tokens. The offload slot became
29.8 MB and L2 dropped to **72 slots**, with ~28 consumed per request. L2 now
sits permanently at 100% (`bytes_used == capacity_bytes`), so "only free slots"
means **never**: `_initiate_promotion` always returns False and the disk tier
is never consulted.

```
with PN97 on : 7 347 queries to the primary tier vs 32 to disk, 1 hit
with PN97 off:                                       73 to disk, 42 hits
```

**Lesson:** when a patch states a numeric precondition in its docstring, treat
it as a pending assertion. Re-measure it whenever anything that feeds it
changes.

### 3.2 PN91 — a budget shorter than the disk

This was the principal cause. When PN91's deferral budget runs out it switches
to strict mode, and in strict mode an **in-flight** block (`HIT_PENDING`)
counts as a MISS — which breaks the sliding-window run in
`_sliding_window_lookup` and makes the *entire* lookup return 0.

L3→L2 promotions are asynchronous and resolve **a batch per scheduler pass**,
so the budget needs real slack:

| budget | result |
|---|---|
| 0.2 s / 4 steps | resolved none; the offload never read a byte |
| 3 s / 60 steps | resolved some, and varied per run: the attention prefix hit 0, 8, 12 or 16 of 20 chunks |
| **15 s / 400 steps** | all nine groups hit, 20/20 and 19/19, reproducibly |

It costs nothing in the bad case: with nothing cached the backend returns MISS
immediately, not `HIT_PENDING`. Cold TTFT stayed at 6.76–6.88 s across the
change.

The 15 s / 400 figure is generous by judgement, not by measurement — the
minimum budget that still hits all nine groups has not been bisected.

### 3.3 PN81 — L3 smaller than L1, pruned backwards

See §6.

---

## 4. The binding constraint: hits must be complete

`_lookup_complete_chunks` requires **all nine KV-cache groups** to hit. The
final hit is the minimum across them, and any group returning 0 makes the whole
lookup return 0 for everyone.

This is why the synthetic test succeeds and live traffic does not. The test
replays the *same* prompt, so the prefix hits 100%. Real traffic sends prefixes
that are similar but not identical, the attention prefix hits partially, and a
sliding-window group cannot serve that truncated boundary — so 227 chunks found
in L2 get discarded wholesale.

```
chunks found in L2 (RAM)    227 hits / 881 queries
chunks found in L3 (disk)     0 hits /  45 queries
external prefix hits          0 tokens / 97 623 queried
```

There is no partial-credit path: the connector either loads a whole
chunk-aligned prefix for every group or recomputes everything.

---

## 5. Boot-time rules

Each of these cost a failure to learn.

### 5.1 `max_split_size_mb:512` is MANDATORY

Together with `expandable_segments:True`. Without it, and **only when cumem is
active**, CUDA graph capture fails:

```
_dummy_run -> sm.fill_(-1) -> torch.AcceleratorError: CUDA error: invalid argument
```

and the engine hangs repeating *"No available shared memory broadcast block
found in 60 seconds"*. It was the only variable separating the engine that
failed from the two that worked.

### 5.2 PN82 is mandatory with TP>1 (vLLM bug)

⚠️ **If the engine dies intermittently in `sm.fill_(-1)`, it is NOT VRAM.**
Lowering `--gpu-memory-utilization` will not fix it.

`pin_mmap_region()` (`v1/kv_offload/cpu/gpu_worker.py`) pins the RAM tier's
mmap with `cudaHostRegister`. It checks the return, logs a warning and carries
on — **without consuming the error from the CUDA context**. The runtime leaves
it latched; PyTorch queries that state on every op, so the first op on the
affected rank explodes far from the cause:

```
20:13:49  TP1  Created mmap file /dev/shm/vllm_offload_... (5.35 GB)
20:13:49  TP0  Opened existing mmap file (the same one)
20:13:51  TP1  WARNING cudaHostRegister failed for rank=1 (code=1)
20:13:53  TP1  ERROR   sm.fill_(-1) -> CUDA error: invalid argument
```

The rank that dies is exactly the one whose registration failed, and only that
one.

With TP>1 a failed registration is the *normal* case, not a rarity: both ranks
map the same `/dev/shm` file and both register it, so the second fails over the
same physical pages. Whether it boots at all depends on which call consumes the
latched error — hence the intermittency that makes the engine impossible to
tune.

Measured, same compose with no parameter changed (identical KV, 379 303, in all
three runs):

| run | `cudaHostRegister` | boot | chat |
|---|---|---|---|
| 1 | failed | **FAILED** | — |
| 2 | OK | OK | HTTP 200 |
| 3 | failed | **FAILED** | — |

Deterministic repro, no waiting on luck:

```python
torch.cuda.cudart().cudaHostRegister(0xdeadbeef, 4096, 0)  # -> code 1
torch.zeros(8, device="cuda").fill_(-1)
# AcceleratorError: CUDA error: invalid argument
```

With `cudaGetLastError()` in between, the `fill_` works. **PN82 does exactly
that**, default ON (kill switch `GENESIS_DISABLE_PN82=1`).

⚠️ `torch.cuda.cudart()` **does not expose** `cudaGetLastError` (torch
2.11.0+cu130 ships only `cudaError` and `cudaGetErrorString`), so PN82 calls it
through `ctypes` against `libcudart`.

### 5.3 Leave ~400 MiB for FlashInfer's lazy workspace

With MTP, `flashinfer.py:_get_workspace_buffer()` allocates **394 MiB** for the
spec-decode prefill wrapper, and does so **lazily, on the first request**, not
at boot. By then the profiler has already handed all memory to the KV cache.

Symptom: the engine boots perfectly and dies on the first request with

```
torch.OutOfMemoryError: Tried to allocate 394.00 MiB.
GPU 1 ... 89.00 MiB is free
  File flashinfer.py, line 781, in _get_workspace_buffer
```

This one *is* a real memory problem. Any flag that frees VRAM and hands it to
the KV cache (§10) must leave that margin.

### 5.4 Drop `restart: unless-stopped`

A boot failure retries in a loop (measured: 9 restarts) and looks like "slow to
start" rather than broken. To detect a process restart without inspecting the
container:

```bash
curl -s :8320/metrics -H "Authorization: Bearer $VLLM_API_KEY" | grep process_start_time_seconds
```

### 5.5 Historical: lowering `--gpu-memory-utilization` by 0.135

**This no longer reproduces on vLLM 0.29.0.** Production runs cumem *and*
`--gpu-memory-utilization 0.92` and boots healthy with 566 314 KV tokens.

On 0.27.1 the rule was real: `OffloadingConnector` requires
`--enable-cumem-allocator`, and cumem made the memory profiler **overestimate
available KV by ~3.2 GiB**. The engine would boot, compute a KV cache that did
not fit, and die in `_allocate_kv_cache_tensors`. The error was **constant, not
proportional to util**, so shaving the value gradually did not help:

```
util 0.965 -> 419 MiB short
util 0.93  -> 309 MiB short
util 0.88  -> 209 MiB short      (each 0.05 recovers only ~110 MiB)
```

The correction was to subtract `3.2 / 23.56 = 0.135`. Kept here for anyone
still on 0.27.1; whether upstream fixed it or the accounting merely changed has
not been verified.

---

## 6. PN81 — disk quota and `/dev/shm` sweeping

vLLM has **neither**. Without PN81:

- `root_dir` grows unbounded: **38 GB in a single test session**. The only
  `os.remove` calls in the `fs` tier are error handling, and the secondary-tier
  interface does not even declare an eviction method.
- `/dev/shm` mmaps outlive the container. vLLM deletes them in `cleanup()`, but
  that only runs on an orderly shutdown; `docker rm -f` leaves them. With
  `ipc: host` they land in the **host's** `/dev/shm`. Measured: 4 orphans of
  5.3 GB filled all 16 GB and the next boot died with `madvise: Bad address`.

PN81 implements `on_schedule_end()` — a hook vLLM documents for *"per-step
cleanup"* and leaves empty — to prune by age, and sweeps orphans at boot
(skipping any still mapped by a live process).

**Two things learned on 2026-09-19:**

- **L3 must be larger than L1.** The quota was 30 GB. The GPU KV cache holds
  566 314 tokens and offloading writes 62.9 KB per token, so mirroring L1 needs
  35.6 GB. A cache tier smaller than the tier it backs is useless by
  construction — everything evicted from GPU immediately overflowed the disk,
  541 disk evictions in a single 30-prompt run. At 64 GB they disappear.
- **PN81 prunes by mtime**, which is age-of-*write*, not age-of-use. That is
  the wrong policy for a cache: the first victim is exactly the long thread's
  prefix, the one thing worth keeping. A generous quota makes it matter less,
  but it remains wrong.

⚠️ **The quota is PER RANK**: `FileMapper` writes to `{base_path}_r{rank}`, so
with TP=2 total disk is `2 × GENESIS_KV_DISK_MAX_GB`.

⚠️ **The offload root is namespaced by `engine_id`, a fresh UUID per boot.**
Dead boots leave directories behind that PN81 counts against its own quota. A
stale tree of 12 boots was measured at 183 GB.

---

## 7. Why ARC, and why the staging ring stays

`eviction_policy: "arc"` separates blocks seen **once** from blocks seen
**several times**:

```
T1: accessed once       -> ephemeral subagents land here
T2: accessed repeatedly -> the long thread climbs here on return
B1/B2: ghost lists of what was just evicted
```

Under LRU, four recent coders outweigh your ten-minute-old prefix and push it
out. Under ARC they do not. And if it is evicted anyway, the ghost-list hit
makes ARC **repartition** so it does not happen again.

**You cannot hint per request.** The policy receives only
`OffloadKey = hash(block) + group_idx`; neither `priority`, nor
`kv_transfer_params`, nor `cache_salt` reach it (`cache_salt` goes into the
hash: it *isolates* prefixes, it does not protect them). The only real lever is
that **access promotes**: sending a cheap request with the same prefix lifts it
to T2.

> The **secondary** tier does receive `ReqContext` (with `kv_transfer_params`)
> in `on_new_request()`, and returns the `RequestOffloadingContext` with its
> `OffloadPolicy`. So a custom tier *could* decide per request. The bundled
> `fs` tier ignores it: `return RequestOffloadingContext()`.

**PN100's staging ring must stay on.** It caps ephemeral traffic at
`GENESIS_PN100_RING_BLOCKS` (32) of L2's 72 slots, which looks like 44% wasted.
It is not: disabling it made the attention prefix drop from 12/20 to **0/20**
hits. Bounding ephemeral traffic is exactly what protects the long thread's
prefix — which is what its own docstring says.

---

## 8. Dead ends, recorded so nobody repeats them

**"The draft group vetoes everyone else's hit, remove its veto."** FALSE. With
PN91's budget at 3 s the lookup returned `('window4', N, 0)` for the DFlash2
draft group, which reads as "the drafter has nothing stored, and cannot have
anything, because its old chunks carry `block_id 0` and `_build_store_jobs`
skips them". A patch was written (PN147, three anchors), applied, and measured
with a successful rescue. That was **n=1 and did not reproduce**. With the
budget at 15 s and PN147 *off*, the same group answers `('window4', 19, 19)` —
the data was always there; the 0 was PN91's strict mode scoring in-flight
blocks as misses. PN147 was deleted: its text patch touched three upstream
anchors unconditionally for no benefit.

**"PN100's ring takes 32 of L2's 72 slots, freeing it gains 44%."** WORSE, see
§7.

**Method note.** Both of the above came from single runs, and both had to be
retracted. This subsystem has genuine run-to-run variance because it depends on
what L2/L3 happened to retain. Every result here should be taken with a clean
disk, a restart, and at least two repetitions.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Writes GBs, never reads a byte | PN97 deadlock, PN91 budget too short, or L3 quota below L1 | §3 |
| `external_prefix_cache_hits` stays 0 while `chunk_hits` is nonzero | partial hit vetoed by a sliding-window group | §4 — no fix today |
| OOM in `_allocate_kv_cache_tensors` | cumem overestimates KV (0.27.1 only) | §5.5 |
| `sm.fill_(-1)` CUDA invalid argument, **intermittent**, TP>1 | failed `cudaHostRegister` leaves a latched error (vLLM bug) — NOT VRAM | PN82, §5.2 |
| 394 MiB OOM on the **first request**, `_get_workspace_buffer` | FlashInfer workspace allocated lazily, after profiling | leave margin, §5.3 |
| Hang on *"No available shared memory broadcast block"* + CUDA invalid argument | missing `max_split_size_mb` | §5.1 |
| Same hang **without** a CUDA error | still compiling; can exceed 10 min | wait; watch CPU/GPU |
| `madvise: Bad address` at boot | `/dev/shm` full of orphaned mmaps | PN81 sweeps them; otherwise delete by hand |
| `Please specify kv_role` | missing from the JSON | `"kv_role":"kv_both"` |
| `incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | cumem missing | `--enable-cumem-allocator` |
| Disk grows without bound | PN81 off | `GENESIS_ENABLE_PN81_KV_DISK_QUOTA=1` |
| Bare `AssertionError` at `offloading/scheduler.py:612` → `EngineDeadError`, after hours of normal use | inconsistent prefix hit across groups (hybrid + MTP) — vLLM bug | PN84, §11 |
| Bare `AssertionError` at `scheduler.py:771` (`_build_store_jobs`) | prompt longer than `max_model_len` reaching the scheduler (cannot happen via the API: returns 400) | §11.4 |

### Checking it from outside

```bash
# the only witness that does not lie: return traffic
curl -s :8320/metrics -H "Authorization: Bearer $VLLM_API_KEY" \
  | grep -E 'kv_offload_total_bytes_total|external_prefix_cache'

# per-tier detail
curl -s :8320/metrics -H "Authorization: Bearer $VLLM_API_KEY" \
  | grep -E 'kv_tier_|tiering_chunk_'
```

A low TTFT alone proves nothing: if eviction failed to push the prompt out of
VRAM, the "rescue" hit L1 and the timing says nothing about offloading. Only
`CPU_to_GPU > 0` distinguishes them. The functional test in
`tests/` enforces this.

### Per-lookup tracing

```bash
GENESIS_DIAG_OFFLOAD=1 GENESIS_DIAG_OFFLOAD_LOOKUPS=3000
```

Dumps, for every lookup, each group's spec type, window, chunks requested and
chunks the backend answered, plus which early return fired. This is what found
all three causes in §3; the guesswork before it was wrong three times.

---

## 10. Flags that free VRAM for the KV cache

Measured 2026-08-15 on w8a16-mtp, TP=2, `util 0.82` — a different engine from
today's production, kept for the method. All three numbers come from the
profiler and are reproducible; the crashes seen during the sweep were PN82
(§5.2), **not** the flags.

| config | KV in GPU | delta |
|---|---|---|
| baseline | 379 303 | — |
| `+ --mm-processor-kwargs` | 399 806 | **+20 503** (+5.4%) |
| `+ --max-num-batched-tokens 4096` | 430 560 | **+51 257** (+13.5%) |

**`--mm-processor-kwargs '{"max_pixels":2000000,"min_pixels":65536}'`** — the
profiler runs a dummy multimodal pass with the **largest permitted** item.
Without this flag it uses `max_pixels` from the model's
`preprocessor_config.json` and reserves activations for a huge image that is
never sent. 2 000 000 px ≈ 1414×1414, ample for a one-image-per-request
contract. Complements `--limit-mm-per-prompt`: that one bounds *how many*
items, this one bounds *how large* each is.

**`--max-num-batched-tokens 4096`** — shrinks the transient peak of the GDN
layers, which scales linearly with batch tokens (12.05 KiB/token/GPU, see
PN80). **Cost:** prefill is split into half-size chunks, so a 200K prompt takes
longer to process. This trades prefill throughput for KV capacity; it is not
free.

⚠️ When raising KV, re-check the margin in §5.3: FlashInfer's 394 MiB workspace
is allocated on the **first request**, not at boot.

---

## 11. PN84 — the engine used to die on its own after hours

### 11.1 What was seen

`genesis-27b-qwen38-fp8`, 2026-08-16 20:03:34. Fourteen hours up, ~60 requests
served without a single error, 4 jobs in flight and 5 queued, KV at 65%. Then:

```
File ".../kv_connector/v1/offloading/scheduler.py", line 612,
  in update_state_after_alloc
    num_locally_computed_tokens
AssertionError
vllm.v1.engine.exceptions.EngineDeadError
```

A bare `assert`, no message. Nobody catches it: it takes down the EngineCore
and every in-flight request returns 500.

### 11.2 The cause

The assert says: *"tokens vLLM marks as already computed in VRAM must be
covered by hashed blocks"*. The harness dump at the moment of failure:

```
--- r19: L=1600 E=6400 ---
  get_computed_blocks returned: (1600, [[], [block 10]])
  g0 (attention) bs=1600 nblk=5 edge=0 pattern=nnnnn   <<< BREAKS
  g1 (GDN)       bs=1600 nblk=5 edge=4 pattern=....n
```

`get_computed_blocks` reports **1600 tokens already computed** and returns
**zero blocks** for the attention group. And the block did exist: the probe
confirms block 0's hash was cached for both groups.

The path, in `HybridKVCacheCoordinator.find_longest_cache_hit`:

1. The attention group is an *eagle* group because MTP is on, so it enters with
   `drop_eagle_block=True`: it matches 1 block and **discards it** (eagle
   matches one extra and drops the last). Left with `[]`, candidate = 0.
2. The GDN group enters with `_max_length = min(0 + block_size, max)` = 1600.
   Here is the bug: **`MambaManager.find_longest_cache_hit` ignores
   `drop_eagle_block`** — it discards nothing. It scans right to left, finds its
   state block, returns 1 block.
3. `curr_hit_length` goes from 0 to 1600: a group **raised** the candidate. The
   algorithm assumes the opposite; its own comment says *"Each attention type
   either accepts the current candidate length or reduces it"*.
4. `is_simple_hybrid` breaks the `while` right there without re-querying the
   attention group, "because one iteration suffices" — true only while nobody
   raises.
5. The final truncation trims the attention group's list to 1 block… but that
   list is empty, so it trims nothing.

Out comes `(1600, ([], [gdn_state]))`.

### 11.3 Why it matters more than it looks

**The crash is the lucky part.** The offloading connector's assert is the only
thing that notices. `num_computed_tokens = 1600` means the scheduler skips
prefill for those 1600 tokens, but the attention group does not have those
blocks: `allocate_slots` hands it fresh, unwritten ones. The dump shows it
directly — group 0 ends with ids `[39, 17, 46, 4, 13]`, no trace of the cached
block 12.

**Without the offloading connector there is no assert and the engine does not
crash:** it answers by reading garbage attention KV for the prompt's first
block.

### 11.4 How it was reproduced

`tests/repro/offload_partial_hit_harness.py` runs the **real Scheduler**, the
**real KVCacheManager** and the **real OffloadingConnector** with the model's
real config (taken from the cached `config.json`), with no GPU and without
loading a single weight. It only simulates the model forward and the connector
worker.

Reproducing the crash by booting the 27B costs ~6 minutes per attempt and
depends on the eviction lottery landing just right. Here thousands of scenarios
run per second and every failure keeps its seed.

Measured matrix, 60 seeds per cell:

| | spec decode YES | spec decode NO |
|---|---|---|
| **hybrid YES** | **3 failures** at `:612` | 0 |
| **hybrid NO** | 0 | 0 |

**Both** are required: a hybrid model (attention + GDN) and speculative
decoding, which is what makes the attention group an eagle group. That is
exactly the configuration of every qwen38 engine on this rig. It also needs the
request to have a **local** (VRAM) hit and an **external** (RAM/disk) hit at the
same time — that is the rare part, and why it took 14 hours to appear. When
that combination occurs it fires in ~1 of 3.

The failure still appears with `--sin-async`, so it is not an artifact of async
scheduling.

#### The `scheduler.py:771` false positive

An earlier harness version also tripped
`assert len(offload_keys) == len(offload_block_ids)` in `_build_store_jobs`, in
6 of 240 scenarios, **without** depending on hybrid or spec decode. That was a
**harness artifact**, not a reachable bug:

```
r24: computed=16000 sched=3522 num_tokens=19522  max_model_len=16384
  g0 keys=12  block_ids_connector=11  real_blocks=11
  allocate_slots: new=3522 newly_computed=16000 -> blocks=[11]
```

The prompt (19 522 tokens) is **longer than `max_model_len`** (16 384).
`allocate_slots` clamps with `min(..., self.max_model_len)` and reserves 11
blocks; `_build_store_jobs` computes `min(computed + scheduled,
req.num_tokens)` **without** clamping and asks for 12. The numbers disagree and
the assert fires.

That prompt never reaches the scheduler in production: the engine rejects it
earlier with a 400 (*"maximum context length"*). The harness built `Request`
objects by hand and skipped that validation. With the cap applied
(`prompt[:max_model_len-1]`) the 6 failures **disappear** and the 3 at `:612`
remain — so the cap does not mask the real bug, it only removes noise.

Recorded as a vLLM robustness note, not something to patch: if a request longer
than `max_model_len` ever did reach the scheduler with offloading active, it
would kill the EngineCore.

### 11.5 The fix

A group must not ask for a length beyond what the candidate permits if its
manager does not implement eagle dropping. `MambaManager` does not, so for
`MambaSpec` groups PN84 neither inflates `_max_length` nor requests the drop.
With that, the GDN group can only **accept or lower** the candidate, which is
the invariant the algorithm already assumed.

No real hits are lost: in the normal case the attention group matches N+1 and
drops 1, the candidate settles at N blocks, and the GDN group finds its state in
block N−1 scanning right to left. The only thing lost is the 1-block "hit" that
is outright false today.

Result with PN84 (same matrix, same seeds):

| | before | after |
|---|---|---|
| failures at `:612` (hybrid + spec) | 3 | **0** |
| local+external hit events exercised | 9 | **57** |

Events went **up 6×**: the patch did not close the path, it made it consistent
— so it happens far more often, and survives.

Kill switch: `GENESIS_DISABLE_PN84=1`.

---

## 12. Copy-path audit (2026-08-17)

Triggered by a finding from the P2P work: on this rig the card's **copy engine
(DMA) delivers 5.28 GiB/s against 12.10 for SM-issued writes** over the same
link (see `docs/P2P-ENTRE-LAS-3090.md` §8). The question was whether vLLM uses
the slow engine anywhere on the hot path.

| candidate | verdict |
|---|---|
| `_select_swap_blocks_fn` wires DMA for GPU→CPU (`gpu_worker.py:43`) | **correct, do not touch.** The comment says "the dedicated copy engine beats Triton" and it holds here: DMA to host gives 12.27 GiB/s, i.e. link speed. DMA's problem is GPU→GPU, not GPU→CPU. |
| `THRESHOLD_BYTES = 28 KiB` for Triton on CPU→GPU | does not apply: our `page_size` is ~1.6 MiB, well above, so it takes the DMA path. Correct. |
| GPU↔GPU copies via `cudaMemcpyPeer` | **absent from the hot path.** One `DeviceToDevice` hit in all of vLLM and it belongs to another model (minimax). Collectives go through NCCL, which already uses the SMs. |
| **the offload mmap is left half-pinned** | **real finding, below** |

### The offload mmap runs half-pinned

`pin_mmap_region()` (`gpu_worker.py:139`) registers the whole region with
`cudaHostRegister`. With TP=2 **both ranks register the same physical pages**
and the driver rejects the second:

```
cudaHostRegister failed for rank=0 (code=1) — transfers will still work
but may be slower (unpinned DMA)
```

We already knew this from PN82, but treated it purely as a sticky-error problem
in the CUDA context. The other half is performance.

Measured (`tests/repro/pinned_vs_mmap.py`, GPU→CPU of 512 MiB):

| destination | GiB/s |
|---|---|
| torch pinned | 12.27 |
| `/dev/shm` mmap + `cudaHostRegister` | 12.27 |
| **mmap, unregistered** | **9.24** |

**Not pinning costs 25%.** Since one rank of two fails, the effect on aggregate
offloading is ~12%.

**It was not `memlock`.** Reasonable suspicion, and false: the container starts
with `memlock` = 8 MB while 5.36 GB is being registered. Tested with
`ulimits: memlock: -1` — **still fails with the same `code=1`**. And a 512 MiB
torch pinned buffer works at 12.27 GiB/s with the 8 MB limit in place: NVIDIA's
driver pins outside `RLIMIT_MEMLOCK`. Do not add the ulimit, it does nothing.

**How it would be fixed.** The layout is **already partitioned by rank**
(`_worker_offset = rank * cpu_page_size`), and the `MADV_POPULATE_WRITE`
routine already walks only its own worker's pages
(`shared_offload_region.py:90-102`). But the slices are **interleaved within
each block row**, not contiguous, so registering one range is not enough: it
needs one `cudaHostRegister` per row, mirroring the existing `madvise` loop.
Precondition: `cpu_page_size % mmap.PAGESIZE == 0`, or page rounding overruns
the other rank's area and fails again. It holds in this config
(1 703 936 = 416 pages), but the patch must verify it and fall back otherwise.

**Low priority.** Offloading copies are not on the critical path (`save_kv_layer`
and `wait_for_save` are no-ops, stores are deferred to separate streams) and
are 2.4% of wall time today. 12% of that is end-to-end noise. It rises in
priority if offload volume grows or the disk tier becomes genuinely active.
Pleasant side effect: if registration stops failing, PN82 loses its reason to
exist.

---

## 13. PN83: the engine explains itself at boot

Everything in this document is emitted by the engine itself at the end of boot,
with that run's actual numbers:

```bash
docker logs <container> | sed -n '/GENESIS · ANALISIS DE ARRANQUE/,/Re-ejecutar/p'
```

or without restarting anything:

```bash
docker exec <container> python3 -m vllm._genesis.analisis_arranque
```

Sections: VRAM breakdown, how many threads and agents fit in the cache, tier
state, MTP, **risks with a ✔/✖ verdict** (PN82, FlashInfer's lazy workspace,
the disk quota) and which flags would yield more KV and at what cost.

Default ON, kill switch `GENESIS_DISABLE_PN83=1`. The usage pattern it uses to
translate capacity is tuned with `GENESIS_ANALISIS_HILO_PRINCIPAL` (220000) and
`GENESIS_ANALISIS_AGENTE` (40000).

⚠️ On the VRAM breakdown: vLLM reports `non_kv_cache_memory` measured **inside**
the requested budget (`total × util`), not against the whole card, and it comes
out *smaller* than the weights. Its arithmetic is
`requested − non_kv_cache − cudagraphs = KV`. PN83 does not mix that with total
VRAM: it breaks down only what is measured in absolute terms (weights, KV,
graphs) and calls the rest "free / activations".
