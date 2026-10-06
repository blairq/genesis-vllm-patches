<p align="center">
  <img src="assets/logo.png" alt="Genesis vLLM Patches" width="780">
</p>

# Genesis vLLM Patches

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-0.29.0-orange.svg)](https://github.com/vllm-project/vllm)
[![status](https://img.shields.io/badge/status-in%20development-yellow.svg)](#status-and-open-problems)
[![GPU](https://img.shields.io/badge/GPU-2%C3%97%20RTX%203090%20(sm__86)-purple.svg)](docs/HARDWARE.md)

**A personal, work-in-progress fork of runtime patches for
[vLLM](https://github.com/vllm-project/vllm), built around one idea: make
`int8` the *compute* standard through every stage of the pipeline — on hardware
that has no `fp8` tensor cores. Next target: `int4` in the KV cache.**

## Read this first

This is **not a product**. It is a working notebook with code attached:

- **It targets one machine.** 2× RTX 3090 (sm_86), TP=2, PCIe gen4 ×8, no
  NVLink, 30 GB of host RAM, one specific quantized checkpoint. Numbers,
  thresholds and several patches are tuned to exactly that.
- **Every number here is measured with the cards power-capped** — 229 W and 243 W, against
  350 / 370 W stock. Uncapped, this same stack does about **+20% prefill and +7% decode**
  ([measured](#what-the-power-cap-costs)). Read the tables as a floor, not a ceiling. On different
  hardware, expect some patches to be useless and others to be wrong.
- **It is in active development.** Patches land, get measured, and sometimes get
  deleted when the measurement says the idea was wrong. The git history contains
  retractions on purpose — see [Dead ends](#dead-ends).
- **It pins to vLLM 0.29.0.** Patching is by text anchors, so a different vLLM
  version will silently skip patches rather than fail loudly. There is a
  preflight tool for that (below), and it should be run before any bump.
- **Documentation is bilingual.** Code comments and commit messages are in
  Spanish; the docs under `docs/` are being moved to English.

> Forked from [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches),
> re-targeted from Qwen3.6 / RTX A5000 to **Qwen3.8-27B on 2× RTX 3090**, and
> moved from vLLM 0.27.1 to **0.29.0**. Upstream's patch framework, dispatcher
> and most patches below PN120 are theirs; the PTX kernels and everything from
> **PN120 up** are this fork's.

---

## The thesis: int8 end to end

Ampere (sm_86) has **no `fp8` tensor cores**. Everything modern inference
assumes — fp8 KV caches, fp8 GEMMs, fp8 collectives — either falls back to
emulation or simply is not there. What the 3090 *does* have is a full-rate
`int8` tensor path (`mma.m16n8k32.s8`) and an `int4` one (`mma.m16n8k64.s4`,
measured at 2.0× the int8 TOPS).

So the bet of this fork is to stop treating int8 as a storage format and use it
as the **compute** format, stage by stage:

| stage | stock vLLM here | this fork | how | measured |
|---|---|---|---|---|
| Weights | int4 | int4 | AutoRound / GPTQ checkpoint | — |
| GEMM activations | fp16 | **int8** | `VLLM_MARLIN_INPUT_DTYPE=int8` + **PN130** (own Marlin, signed int16 scales) | **+57%** prefill |
| KV cache | fp16 / fp8 | **int8 per token-head** | `--kv-cache-dtype int8_per_token_head` | **+30%** decode @50K vs fp8, and **4× less error** |
| Attention decode | fp16/fp8 kernel | **integer, hand-written PTX** | **PN131** — SK-18h reads the int8 KV directly, no dequant | parity with FlashInfer fp8 on quality, decode and PP |
| TP all-reduce | fp16 | **int8** | **PN120** — half the bytes over a PCIe link already at 96% | **+6.5%** prefill |
| Drafter KV | inherits fp16 | **int8** | compose | part of the **+226%** KV capacity |
| `lm_head` (target + drafter) | bf16 → fp8 | **int4 g128** | **PN139** — half the bytes of fp8, a format Ampere has no silicon for | **+3%** steps/s, **+3%** KV |

The point is the *composition*. Any one of these is a known trick; doing all of
them means a tensor never has to leave the integer domain between the RMSNorm
that produces it and the attention that consumes it — which is also why the
KV cache can be int8 without a dequant step in front of the attention kernel.

### What each KV format actually costs, measured

The least intuitive claims here are about the KV cache, so: numbers first. This
is the error of the attention **output** against float — what reaches the next
layer, not the error of the stored tensor. Measured on real dumps from this
model (`PN123`, layers 3 and 35, 8 204 tokens of code, 192 queries seeing
almost the whole context). Reproduce with
[`tests/proto/kv_escalas_grupo_eval.py`](tests/proto/kv_escalas_grupo_eval.py).

| KV scheme | bits/value | layer 3 | layer 35 |
|---|---|---|---|
| `fp8_e4m3`, per-tensor scale | 8.00 | 1.27% | 2.30% |
| `int8` per token-head | 8.10 | 0.31% | 0.79% |
| **`int8` per token-head + Hadamard** ← in production | 8.10 | **0.22%** | **0.39%** |
| `int4` per token-head | 4.10 | 6.49% | 12.61% |
| `int4` per token-head + Hadamard | 4.10 | 4.08% | 6.68% |
| `int4` group 64 + Hadamard | 4.25 | 3.54% | 5.46% |
| `int4` group 32 + Hadamard | 4.50 | 3.15% | 4.96% |
| `int4` group 16 + Hadamard | 5.00 | 2.71% | 4.33% |

Three things fall out of that table.

**int8 beats fp8 by ~4× at the same width.** The reason is where the bits go.
`e4m3` spends 4 of its 8 bits on an exponent, buying dynamic range this tensor
does not use: post-RoPE vectors with qk-norm are nearly isotropic and each
(token, head) slice occupies a narrow range. int8 with a scale fitted *per
token-head* spends all 8 bits on mantissa inside exactly that range. The 0.1 bit
is that scale, amortized over the head.

**Hadamard rotation helps int8 — roughly halving the error at layer 35** (0.79%
→ 0.39%). This corrected an earlier claim in this README that said rotation did
nothing for int8; that claim came from misreading a study which had compared
Hadamard against WUSH, *both* rotated, never against no rotation at all.
`PN126` is now **on** in production.

**Per-group scales rescue a lot of int4, but not enough.** Going from
per-token-head to group-32 with rotation takes layer 35 from 12.61% to 4.96%.
That is a 2.5× improvement for 0.4 extra bits — real, and the direction the
quantization work should take. It is still an order of magnitude worse than
int8, which is why int4 is not in production.

### What that costs in throughput

`bench.sh`, same harness and depths as above, one full run per arm:

| | PN126 off | PN126 on | |
|---|---|---|---|
| prefill @ 10K | 2829 ±5 | 2791 ±38 | **−1.4%** |
| prefill @ 90K | 1850 ±11 | 1859 ±5 | +0.5% |
| decode, narrative | 131 ±6 | 135 ±6 | +3.2% |
| decode, code | 235 ±22 | 258 ±12 | +9.5% |

Read that honestly: only the 10K prefill difference is outside the noise (that
arm's CV was 0.2%). The decode numbers move in the right direction but sit
inside a 4–9% run-to-run spread, and this is **one run per arm** — not enough to
claim a decode gain. The defensible statement is that rotation costs at most
~1.4% of prefill and buys a measurable accuracy improvement.

And on Ampere, int8 KV is also simply **faster than fp8**, which was not the
goal — the switch was made looking for quality:

| | fp8 | int8 |
|---|---|---|
| decode @1K | 114.7 | **157.8** tok/s |
| decode @50K | 91.8 | **119.7** tok/s (+30%) |
| prefill @50K | 2251 | 2206 tok/s |

Two reasons: Triton **does not accept fp8 on SM86** at all, so that path pays
conversions the hardware cannot do natively; and the integer kernel consumes the
int8 KV directly, with no dequant in front of it. The honest cost is ~7% of KV
capacity, because the integer kernel reserves its own accumulators.

### Next: int4 in the KV cache

int8 is the standard the pipeline runs on today. **int4 in the KV cache is the
target**, and the reason it is not in production yet is worth stating precisely,
because it is not what you would guess.

The kernel is not the problem. Ampere has a native `int4` tensor path
(`mma.m16n8k64.s4`, measured at **2.0×** the int8 TOPS, same register layout as
`s8`), the integer attention kernel has an int4 variant, and a full working
compose exists in this repo's history (`9305c90`). **Quality is the problem.**

The numeric side is in the table above: the best int4 scheme measured so far
(group-32 + Hadamard, 4.5 bits) sits at 3.15% / 4.96%, against 0.22% / 0.39%
for the int8 that runs in production. An order of magnitude. It also costs
**−27% prefill and −21% long decode** against int8.

But the blocking symptom is not numeric at all — it is behavioural: **the model
runs on and never closes.**
On a long coding task (a Tetris with SRS, 7-bag, hold and T-spin), three runs
each: fp8 produced 6 111 / 6 594 / 7 780 tokens and passed 6/6 execution checks;
int4 produced 11 153 / 32 000 / 32 000 — the last two hitting the `max_tokens`
ceiling — and passed 4/6, 2/6, or emitted no code at all.

**Where it stands.** Per-group scales along the head dimension were the
outstanding lead, and they have now been measured (the table above): they take
layer 35 from 12.61% to 4.96%, a 2.5× improvement for 0.4 extra bits. Real, and
the right direction — but not enough on its own. What is still untried: group
smoothing folded into the group scales and the RMSNorm (2–4.8× more precise than
dynamic, at no cost), per-layer decisions about whether rotating helps, and
learned rather than fixed rotations. Prototypes under `tests/proto/sk18_a0*`.

So the open work is **an advanced quantization process aimed at int4 quality**,
not another kernel. And its oracle cannot be per-layer numeric error or a short
answer — it has to be a long generation task where you check whether the model
*terminates*, because that is the failure mode.

### Where int8 really is the floor

Not everything below 8 bits is worth chasing. These are negative results from
this rig, and they are settled:

- **All-reduce in int4** is **7× worse than int8** at equal traffic. The path
  there is overlapping the transfer, not compressing it harder.
- **W4A4 in the MLP** composes its errors: `gate_up` alone ≈ fp16, but both
  projections together lose 0.1 logprob.
- **Dictionary / VQ methods on the KV** (PQ, RVQ, Lexico-style sparse coding)
  all lose to plain int4 at the same width. Post-RoPE vectors with qk-norm are
  nearly isotropic — there is no structure left for a codebook to exploit.
- **The decode GEMMs are already at the DRAM roof** — 98–103% of achievable
  bandwidth. No amount of PTX buys anything there; the remaining 2× is in
  unpacking nibbles, not in arithmetic.

---

## What is running right now

```
compose/docker-compose.qwen38-27b-idiotsavant-sm86.yml   →  genesis-27b-idiotsavant
```

**This is the compose to copy.** It serves this project's own checkpoint,
[**idiotSavant**](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86), with its
[DFlash2 drafter](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2), and every
setting in it carries a comment with the measurement behind it.

| | |
|---|---|
| Model | [`BlairQ/qwen3.8_27b_idiotSavant_sm_86`](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86): Qwen3.8-27B uncensored, int4 with the **residual stream rotated** (Hadamard), served as **W4A8**. KL to BF16 **0.0178**, vs 0.0385 for the popular AutoRound int4. Weights of 2026-09-28 |
| Drafter | [`…_dflash2`](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2): DFlash2 re-tuned against the served target, **8-node tree** speculation, `num_speculative_tokens=8` |
| Mandatory patches | **PN148** + **PN154** (online Hadamards before `down_proj` / `o_proj` / `out_proj`), **PN155** (int4 GDN gates), **PN149** (drafter on the rotated target). Without them the model outputs garbage |
| Engine | vLLM 0.29.0 + Genesis, 2× RTX 3090 (sm_86), TP=2, PCIe, no NVLink |
| KV cache | **`int8_per_token_head`**, read by this project's integer PTX decode kernel (SK-30) |
| Context | 262 144 tokens · `max-num-seqs 11` · `gpu-memory-utilization 0.95` |
| **KV capacity** | **653 695 tokens** (2.49× concurrency at full context) |
| Prefix cache | sparse GDN checkpoints every 14 080 tokens + the reuse boundary (**PN168**) |
| KV offload | **L2 in RAM, 12 GiB, no disk** (**PN170**/**PN171**): what the GPU evicts comes back over PCIe in seconds |
| Scheduling | one prefill at a time with priority bypasses (**PN115**), dynamic prefill chunk (**PN173**) |
| Address on this host | `172.20.0.228:8320` (alias `vllm-server`), published on `:8360` |

What that buys on a captured real session (two opencode threads of ~100K and ~125K plus a
sub-agent, 13 requests replayed with their original timing): **70% of prompt tokens served from
cache, every follow-up turn 93–97% cached**, against 49% before the 2026-10-05/06 prefix-cache
work. With the GPU KV deliberately cut to 0.82, L2 keeps it at 70% (22% without L2). Details in
[Tiered KV cache](#tiered-kv-cache-l2-in-ram).

The two `noon` composes (`docker-compose.qwen38-27b-noon-*.yml`) are kept for the record only: that
checkpoint is no longer on this host. Unless a section says otherwise, the measurements from here
to *What this fork adds* were taken on it (2026-09); idiotSavant's own numbers are on its
[model card](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86).

### Measured performance

**All of it under the power cap (229 W / 243 W).** Measured 2026-09-20 with the `bench.sh` harness from
[club-3090](https://github.com/noonghunna/club-3090), run **directly against the
container** (no reverse proxy in the path). Decode: 3 warm-ups + 5 measured
runs. Prefill: 1 warm-up + 3 measured runs, cache-busted with a fresh haystack
per run. Raw output in [`tests/bench/resultados/`](tests/bench/resultados/).

| | mean | CV |
|---|---|---|
| prefill @ 10K | **2808** tok/s | 1.3% |
| prefill @ 90K | **1838** tok/s | 0.7% |
| decode, narrative | **134** tok/s | 5.3% |
| decode, code | **257** tok/s | 8.7% |

That is the stack as it stands at the end of 2026-09-20: PN139 on, PN122 with its
model-runner-v2 hooks, PN92/PN108 off. The run from that morning, before those changes, gave
2829 / 1850 / 131 / 235. Prefill is unchanged within its CV; the decode means moved +2% and
+9%, which is **inside** this harness's 5–9% decode spread — one run per stack does not
resolve it. The A/B that does resolve it is [below](#ab-on-a-greedy-bench-2026-09-20).

Prefill is very stable. Decode spread is real, not noise: throughput tracks
DFlash2's acceptance rate, which depends heavily on the content being generated
— code accepts far more draft tokens than prose, which is the whole 131 → 235
gap.

The harness reports `INTEGRITY: OK` and `swap check: PASS`. (Its measured requests go through
`urllib`, not `curl`, so `tests/bench/medicion/club_bench.sh` now injects the API key with a
temporary `sitecustomize`, scoped to the server under test — the `.curlrc` alone got 401s.) Its draft-acceptance
and engine-timing captures came back empty (the engine does not log acceptance
at this verbosity), so the acceptance rate behind that gap is inferred from
throughput, not measured here.

### What the power cap costs

The cap is deliberate — heat, noise and the power bill on a machine that runs all day — and it
is *asymmetric* on purpose: the two cards are not the same silicon, TP=2 runs at the pace of
the slower one, and 229 W / 243 W is the split where their clocks meet (+13.4% prefill over an
even split at similar total power; the sweep is documented inside
`/etc/systemd/system/nvidia-powerlimit.service`).

Same container, same minute, cap lifted to the stock limits and then restored through that
service (2026-09-20; three 90K prefills, and the greedy bench for decode):

| | capped 229 / 243 W | uncapped 350 / 370 W | |
|---|---|---|---|
| prefill @ 90K | 1 861 tok/s | **2 224** tok/s | **+19.5%** |
| decode, reasoning prose | 124.4 tok/s | **132.7** tok/s | **+6.7%** |
| decode, code | 256–274 tok/s | 285 ±12 tok/s | ≈ +5%, inside that bench's noise |
| SM clock under load | 1 370 / 1 464 MHz | 1 792 / 1 820 MHz | |
| power drawn | 224 / 238 W | 326 / 330 W | **+42%** |
| temperature (max) | 69 / 56 °C | 76 / 63 °C | |

The club-3090 `bench.sh` harness, run the same way with the cap lifted
([raw output](tests/bench/resultados/bench-dflash2-sin-cap-350-370w-20260920.txt), `INTEGRITY: OK`),
against the capped run in [Measured performance](#measured-performance):

| `bench.sh` | capped 229 / 243 W | uncapped 350 / 370 W | |
|---|---|---|---|
| prefill @ 10K | 2 808 tok/s (CV 1.3%) | **3 368** (CV 1.6%) | **+20%** |
| prefill @ 90K | 1 838 tok/s (CV 0.7%) | **2 221** (CV 0.3%) | **+21%** |
| decode, narrative | 134 tok/s (CV 5.3%) | **140** (CV 2.4%) | +4.3% |
| decode, code | 257 tok/s (CV 8.7%) | **285** (CV 6.1%) | +11%, one run per arm at 6–9% CV: "somewhat up", not a number |

The two harnesses agree where they overlap: 2 221 vs 2 224 tok/s at 90K, and a single-digit
gain on decode. The code-vs-narrative gap stays at ~2× with or without the cap — it comes from
the drafter's acceptance rate, not from power.

Prefill is compute-bound and follows the clock. Decode is bound by memory traffic and barely
moves — which is the same thing the [decode profile](#where-a-decode-step-actually-goes) says
from the other side. So the cap trades a fifth of the prefill for 42% less power, and costs
decode almost nothing.

**The cap also makes measurements drift.** The driver holds the power limit by moving the
clock, so the clock wanders with temperature: within a single run of five identical 90K
prefills GPU0 went from 1 220 to 1 162 MHz, and throughput from 1 896 to 1 819 tok/s. A
single run per arm can therefore "find" a ±5% effect that is not there — it happened in this
README, see the threshold table under
[Admission and time to first token](#admission-and-time-to-first-token).

### A/B on a greedy bench, 2026-09-20

`bench.sh` samples at `temperature=0.6`, so its decode spread hides anything under ~5%. For
A/B work there is [`tests/bench/medicion/ab_greedy.py`](tests/bench/medicion/ab_greedy.py):
`temperature=0`, 1 500 tokens, 1 warm-up + 3 runs, acceptance from the `/metrics` deltas, and a
foreign-traffic check. Two loads: reasoning prose, and code with thinking off. One fresh
container per arm.

| arm | prose tok/s (accept) | code tok/s (accept) | verdict |
|---|---|---|---|
| baseline, two boots | 118.5 / 117.9 (2.79) | 274.7 / 265.0 (6.58 / 6.38) | — |
| **PN139** — `lm_head` int4 instead of fp8 | **124.4 / 123.9** (2.86) | 273.6 / 256.5 | **+5% prose**, reproduced; on |
| drafter **without** Hadamard rotation | 123.3 (2.92) | **239.8** (5.79) | −7…13% code; rotation stays |
| PN92 + PN108 off (no fp8 left) | 125.2 | 273.9 | neutral; off |

Prose is tight (±0.3 tok/s, reproduces across boots). Code is **not**: the same config gave
256–275 across four boots, because the generation-based healthcheck lands inside some runs and
moves the acceptance. Under ~5% on code, one boot per arm proves nothing.

Two things fall out besides the verdicts. Steps/s is ~42 with or without the drafter rotation,
so the whole effect is acceptance — **fusing or optimising the FWHT kernel would buy nothing**.
And the bench is short-context only; K and the attention kernel were tuned at 50K+, and a
long-context arm is still missing.

### Behavioural quality

`quality-test.sh --quick` from the same project (ToolCall-15 +
InstructFollow-15), sampled at `temperature=0.6` — so expect run-to-run spread:

| run | stack | score | failing scenarios |
|---|---|---|---|
| 2026-09-19 | before today's changes | **27/30** | IF-04, TC-05, TC-09 |
| 2026-09-19 | same | **27/30** | IF-04, IF-10, TC-05 |
| 2026-09-20 | int8 + PN126 | **25/30** | IF-04, IF-10, TC-05, TC-09, **TC-11** |

Read the scenarios, not the total. **IF-04 and TC-05 fail in all three runs** —
those are stable failures, not noise. IF-10 and TC-09 each fail in one of the
two older runs, so they are the flaky ones, and today they simply happened to
fail together. Only **TC-11** is new.

That is one run against the current stack, and the 27/30 baseline predates
today's changes, so the two-point difference is **not** attributable to any
single patch yet. A clean A/B — PN126 on and off, several runs each, nothing
else moved — is pending.

### KV format: capacity and throughput

Same harness, same depths, one full run per format, everything else at the
production configuration:

| | `auto` (fp16) | `int8_per_token_head` |
|---|---|---|
| **KV capacity** | 328 160 tok · 1.25× | **566 314 tok · 2.16×** |
| prefill @ 10K | 2681 (CV 5.1%) | 2648 (CV 13.3%) |
| prefill @ 90K | 1690 (CV 1.3%) | **1838** (CV 0.7%) |
| decode, narrative | 151 (CV **23.3%**) | 134 (CV 3.3%) |
| decode, code | 250 (CV **16.2%**) | 247 (CV 9.4%) |

**int8 gives +73% more KV capacity than fp16** — 1.25× concurrency becomes
2.16× at full context. On throughput it wins clearly at 90K prefill (+8.8%);
everywhere else the means are close but the *stability* is not. fp16 runs the
generic kernel and swings 16–23% between runs; the integer kernel stays at
3–9%. Compare means with the CV beside them, not alone.

### Where a decode step actually goes

Kernel-by-kernel profile of one decode step, so the table above has a shape
behind it. Two HTML reports live in `docs/`:
[radiography](docs/informe-radiografia-decode-2026-09-16.html) (every kernel:
where it lives, µs/step, %, what it does) and
[remaining headroom](docs/informe-margen-por-seccion-2026-09-16.html) (eight
fronts ranked by prize, with the literature for each).

| section | share of the step |
|---|---|
| Marlin linears | **43%** (of which `lm_head` alone: 12.6%) |
| Attention — own PTX kernel | 19.2% |
| TP all-reduce | 11.2% |
| GDN (linear-attention layers) | 9% |
| Draft model, running fp16 | 7.6% |
| everything else | 10% |

Prefill has a different shape entirely: NCCL 40%, Marlin 38%, attention 8.6% —
which is why PN120 (int8 all-reduce) pays off in prefill and barely registers in
decode.

⚠️ **This profile is outdated in two ways that matter.** It was taken
2026-09-16 at 57K context with **int4 KV and the MTP drafter at K=4**;
production now runs **int8 KV and DFlash2 at K=8**. Expect the attention and
drafter shares in particular to have moved. It is kept because the *ranking* has
held up — Marlin linears dominate, `lm_head` is a surprisingly large single
item, and the drafter running in fp16 is free money left on the table (PN133
would fix it and is written but not enabled). Re-profiling against the current
configuration is on the list.

---

## What this fork adds

The dispatcher holds **214 entries**; **76 are applied** in the
idiotSavant container (boot log of 2026-10-06). Everything from PN120 up is this fork's work. ✅ marks what is
actually running in production, taken from the dispatcher's boot log rather than
from the code defaults.

### The integer pipeline

| | | |
|---|---|---|
| ✅ | **PN130** | Own Marlin W4A8 with **signed int16 scales** (vllm#48905). Upstream's Marlin produced silent garbage on AutoRound checkpoints with negative scales |
| ✅ | **PN125** | Positive-scale fallback for Marlin W4A8-INT8, kept as a safety net |
| ✅ | **PN131** | **SK-18h**: attention decode in hand-written PTX, integer throughout, reading `int8_per_token_head` KV with no dequant |
| ✅ | **PN120** | TP all-reduce compressed to int8 during prefill |
| ✅ | **PN124** | Fast TRITON_ATTN on Ampere for head dim 256 |
| ✅ | **PN126** | Hadamard rotation of q/k after RoPE — halves the int8 KV error at layer 35, costs ~1.4% prefill |
| | PN134-137 | int8 activation quantization fused into the RMSNorm; group smoothing folded into the weights (exact SmoothQuant) |
| | PN133 | Draft-model linears in W8A8 — the 7.6% above |
| ✅ | **PN139** | `lm_head` in per-group int4 (g128), target and drafter — replaces the fp8 of PN77, which stays on only as the hook |

### KV cache and capacity

| | | |
|---|---|---|
| ✅ | **PN145** | Sliding-window block aligned with primary attention |
| ✅ | **PN146** | Selectable KV group size — upstream's heuristic picks badly on hybrid models |
| ✅ | **PN122** | Speculative rollback on GDN without speculative blocks. Needs its own hooks in the v2 model runner, which DFlash2 forces — see [open problems](#status-and-open-problems) |
| ✅ | **PN127** | `MambaManager` honours `drop_eagle_block` (vllm#48375) |
| ✅ | **PN121** | Preemption-cascade guard with deferred frees |
| ✅ | **PN168** | Sparse GDN checkpoints that actually work with speculative decoding: the next turn looks for the recurrent state at block P−3, vLLM kept it at P−1. Real session 61% → 70% cached |
| ✅ | **PN170** | L2 offload hits at all on this hybrid + DFlash model: the GDN groups no longer count as EAGLE groups in the external lookup (they demanded two consecutive states that sparse retention never has, so every lookup returned 0) |
| ✅ | **PN171** | L2 keeps only the drafter-window chunks a lookup can ask for (−13% of L2 per conversation) |
| | PN172 | Eviction by role: a sub-agent never evicts a main thread's blocks. Off: ties with L2, loses without it |

PN145 + PN146, giving the drafter int8 KV, and the int4 `lm_head` took capacity from 178 823 to
**583 790 tokens (+226%)**. The mechanism is counter-intuitive and worth
knowing: on a hybrid model `bytes_per_block` is the **max** across groups, not
the sum, so *smaller* groups yield *more* total blocks.

### Speculative decoding

| | | |
|---|---|---|
| ✅ | **PN142** | DFlash2 usable on 0.29.0 (vllm#51581 + port of `fa5017a5`) |
| ✅ | **PN144** | Scales the DFlash2 drafter's residual so it fits fp16. The drafter ships in bf16 and overflowed at 50 080 against fp16's 65 504 ceiling, accepting **0%** while producing perfectly coherent text. Watch the `eps` trap: RMSNorm is scale-invariant, `eps` is not |
| ✅ | **PN128** | Async input-prep waits for spec-decode post-processing (GDN+MTP) |
| | PN132 | Trimmed vocabulary for the drafter (FR-Spec) |

### Infrastructure

| | | |
|---|---|---|
| ✅ | **PN143** | Genesis hooks into *every* vLLM process via `load_general_plugins()` |
| ✅ | **PN83** | The engine explains its own memory layout and risks at boot |
| ✅ | **PN81/PN88** | Disk quota and Prometheus metrics for the KV tiers — vLLM ships neither |
| ✅ | **PN169** | Conversation identity per request: session, parent, root and agent from `X-Genesis-*` headers set by an [opencode plugin](#conversation-identity-from-opencode), copied into `kv_transfer_params` so the scheduler sees them |
| ✅ | **PN165** | Prefix-cache diagnosis per request: common prefix with earlier requests, actual hit, and the exact message/token where it diverged (numbers only, never text) |
| | PN166 | Captures large chat requests as they arrive, to replay real sessions on the test instance (`tests/bench/medicion/sesion_real.sh`) |

PN143 exists because `apply_all` runs as a separate process and then `exec`s
`vllm serve`: anything registered in memory is lost across that boundary. This
bit us silently once — the server booted, logged "applied", and ran the generic
kernel anyway. Only a one-time notice inside the kernel's `forward` revealed it.

### Admission and time to first token

| | | |
|---|---|---|
| ✅ | **PN115** | Admission control in the scheduler: KV-headroom gate, **one prefill at a time**, and three ways past that queue — *forced* priority (the client says so), a short prompt, or little left to compute after the prefix-cache hit |
| ✅ | **PN173** | Dynamic prefill chunk: 2 640 tokens while nobody is generating, 880 while someone is, so a turn that is decoding is not slowed to the pace of someone else's long prefill |
| ✅ | **PN89** | Request tracker and the HTTP endpoints described under [Client-facing controls](#client-facing-controls) |

With an idle server TTFT is simply prefill: 82 ms at 100 tokens, 0.54 s at 1.5K, 2.5 s at 7.4K,
8.1 s at 22K, 18.8 s at 44K. The two problems worth writing down were both about *scheduling*,
and both were invisible in a throughput benchmark.

**A short request waited for the whole long prefill.** PN115 serialises prefills on purpose:
the prefill already runs at 80–85% of the compute roofline, so N concurrent prefills each get
1/N of it and *none* finishes until all do. Six 50K sub-agents are all usable at 200 s that
way, versus the first one at 33 s when serialised. That reasoning holds between comparable
prompts and is pure loss for a 300-token request stuck behind a 59K one — measured at
**26 s**, against 108 ms on an idle engine. The fix is the automatic priority:
`GENESIS_PN115_PROMPT_CORTO_TOKENS=2000` lets a prompt of up to 2 000 tokens skip the
serialisation — and *only* that; it still respects the KV-headroom gate and never preempts
anyone.

**The prefix cache only hit in 7 920-token steps.** On this hybrid model the GDN recurrent
state is saved at the end of each prefill chunk, so the chunk *is* the cache granularity.
With `--long-prefill-token-threshold` equal to `--max-num-batched-tokens` (8192 → aligned
chunks of 9 × 880 tokens) a repeated 7.4K prompt cached **nothing**.

One flag moves both, because a smaller chunk also leaves room in each step for the short
request that was just let through. Measured, one clean boot per arm
([`tests/bench/medicion/ttft_ab.py`](tests/bench/medicion/ttft_ab.py)):

| `--long-prefill-token-threshold` | repeated 7.4K prompt | repeated 22K prompt | 90K prefill, idle server | short request under a 59K prefill |
|---|---|---|---|---|
| 8192 (before) | 2.49 s · 0 cached | 2.61 s · 15 840 cached | 1 867 tok/s | **26 s** |
| 3520 | 1.67 s · 3 520 | 1.95 s · 17 600 | not re-measured ² | 3.6–6.6 s ¹ |
| **1760** ← in production until 2026-10-06 | **0.91 s** · 5 280 | **1.28 s** · 19 360 | 1 861 (−0.3%, noise) ² | **1.6–3.3 s** |

¹ measured with forced priority, before the automatic one existed; the 1760 row is with plain,
unmarked requests. KV capacity is identical in all three arms.

² **Retraction.** This table first said the smaller chunk cost −5.5% of long-prefill
throughput (1 920 → 1 814 tok/s), from one 90K run per arm. Repeated properly — fresh boot, one
90K warm-up, five 90K runs per arm — it is 1 867 (range 1 819–1 896) against 1 861 (1 852–1 879):
no measurable cost. The kernel profile agrees: at 18K the 1760 arm is actually *faster*
(6.47 s vs 6.86 s, Marlin is more efficient at the smaller batch), and what grows with more
steps — idle gaps, the offload's device-to-host copies, PN131's per-step KV dequantisation —
adds up to tens of milliseconds. The original gap was clock drift under the power cap.

The threshold alone does **not** fix the 26 s — with PN115 serialising, the short request is
never admitted, whatever the chunk size. And the priority alone is not enough either: at 8192
an admitted short request still waits 7.5–13 s, because the long prefill takes the whole step.
It takes both. Long prompts still run one at a time: three 30K prompts launched together get
their first token at 16 / 28 / 40 s, with short requests in between answered in 0.5–3 s.

"Short" used to be judged on the total prompt length, so a 100K turn with 96K already cached
counted as long. Since 2026-10-06 PN115 also looks up the local prefix-cache hit at the gate and
lets a request through when what is **left to compute** fits in one batch
(`GENESIS_PN115_FALTAN_TOKENS=8192`).

**A decoding turn crawled during someone else's long prefill.** The first token was fine (5 s);
the rest went at one step per ~1–1.4 s, because every step also carried a prefill chunk. The
chunk sets the step time, so it trades the long prefill against everyone else's decode — measured
with a 125K prefill and, during it, a cached turn generating 64 tokens:

| prefill chunk | 125K prefill alone | the decoding turn |
|---|---|---|
| 880 | 83.7 s | **16 s** |
| 1760 (before) | 76.3 s | 31 s |
| 2640 | **74.0 s** | 49 s |
| **dynamic, PN173** ← in production | 74.6 s | ~20 s |

PN173 picks 2 640 while nobody is decoding and 880 as soon as someone is (or has only a little
prefill left). On the replayed real session it ties overall, and the first turns of two big
threads that start together pay +6–9%; it is on by choice, for responsiveness.

### Tiered KV cache: L2 in RAM

When VRAM fills, vLLM **discards** old prefix blocks and recomputes them later. The offload
connector copies blocks to RAM (L2) as they are computed and brings them back over PCIe when a
later turn needs them. **Since 2026-10-06 it runs as L2 only — 12 GiB of RAM, no disk.**

**Until then it never hit on this model, at any size.** Two bugs stacked, both specific to a
hybrid (GDN + attention) model with speculative decoding:

1. **The reuse point had no state (PN168).** With speculation, the lookup drops one block in
   attention and another in the GDN manager, so the next turn looks for the recurrent state at
   block **P−3** (P = full blocks of the previous prompt). With sparse checkpoints vLLM kept it at
   P−1, and the prefill chunk left P−3 in the middle of a chunk, with no state. This also broke
   the *GPU* prefix cache, not just L2.
2. **The GDN groups were treated as draft groups (PN170).** With DFlash and no group marked as the
   drafter's, the connector declares *every* group EAGLE. For the GDN groups that adds the EAGLE
   extra window: the lookup demands **two consecutive** states, which sparse retention never has.
   The GDN group returned 0, and the lookup needs every group to hit, so the external hit was 0
   every time — with 5 GB of correct attention KV sitting in L2.

Measured after both fixes, on the test instance:

| | |
|---|---|
| 100K turn, after emptying the GPU prefix cache | **95 920 tokens from L2, TTFT 55.9 s → 4.3 s** |
| turn resuming the same conversation halfway (segment checkpoint) | 56 320 of 62 813 tokens from L2, 4.2 s |
| exactness | per-block checksums of all 9 KV groups, both GPUs: what comes back is byte-identical to what was stored |
| real session, GPU KV cut to 0.82 to force eviction | **without L2: 22% cached, TTFT sum 786 s · with L2: 70%, 66 s** — the same as with the full GPU |
| partial hits | a turn takes what is still in VRAM and only the rest from L2 (e.g. 5 280 local + 117 040 from L2) |
| size | ~4.3 GB of L2 per 100K-token conversation (PN171 stores only the drafter chunks a lookup can ask for) |

Eviction from VRAM is already block by block: a finished request's blocks go back tail first, LRU
across requests, so the start of a conversation is the last thing to leave.

Output after an L2 hit is not token-identical to an L1 hit: the prefill that follows is split
into different chunks, which moves the rounding (an L1 hit differs from a cold run in the same
way). The bytes are exact; use logprobs or token ids, not streamed tool-call text, to compare.

**L3 (NVMe) is off.** It is still in the code (PN81 quota, PN90 per-agent write gating), but on
this rig every hit came from RAM, and the disk only added writes. Before turning it back on: the
`persist_disk` flag a client sends overrides the server-side allow-list, and if PN90 ever fails
to apply on a new vLLM, upstream's default is to write everything through to disk.

The older write-up, from when L2 + L3 were sized for the `noon` checkpoint, is in
[docs/KV-OFFLOADING.md](docs/KV-OFFLOADING.md); its sizing rule predates the fixes above.

## Client-facing controls

Everything a client can send, or call, that changes how this server behaves. None of it is
stock vLLM except the `priority` field itself.

### Conversation identity from opencode

[`clients/opencode/genesis-sesion.ts`](clients/opencode/genesis-sesion.ts) is a ~50-line opencode
plugin (copy it to `~/.config/opencode/plugin/`). Through opencode's `chat.headers` hook it adds
four headers to every request sent to a provider whose id matches `GENESIS_PROVEEDORES` (a regex,
`^llm_saitama` by default) and to no other provider:

| header | value |
|---|---|
| `X-Genesis-Sesion` | the opencode session making the request |
| `X-Genesis-Padre` | the session that launched it (empty for a main thread) |
| `X-Genesis-Raiz` | the main thread at the root of the tree |
| `X-Genesis-Agente` | the opencode agent (`build`, `agi_explore`, …) |

PN169 copies them into `kv_transfer_params`, so the scheduler knows which requests belong to the
same conversation and which are sub-agents. PN165 logs them next to each request's prefix-cache
diagnosis, PN166 keeps them in captures, and PN172 uses them for its eviction-by-role rule. A
request without the headers is served exactly as before. The ids are opaque; the chat template
never renders them, so they do not affect the prefix cache.

### Request fields

They travel in the body of `/v1/chat/completions` (or `/v1/completions`). `kv_transfer_params`
is an existing vLLM field that reaches the scheduler and the offloading tiers untouched, which
is why it is used as the carrier: no plumbing patch is needed.

```jsonc
{
  "model": "qwen3.8",
  "messages": [...],
  "priority": -8,                       // optional; forced priority
  "kv_transfer_params": {
    "genesis_agent": "primary",         // who is asking
    "persist_disk": true,               // may this request's KV reach the NVMe tier?
    "name": "refactor-auth"             // display name in the request history
  }
}
```

| field | read by | effect |
|---|---|---|
| `priority` (int) | PN115, scheduler | Below `high_prio_threshold` (default `0`) it is **forced priority**: never waits behind a prefill, skips the KV-headroom gate, and if every slot is taken it may **preempt** a lower-priority request. It also orders the waiting queue (`--scheduling-policy priority`). Lower = more urgent. |
| `kv_transfer_params.priority` | PN115 | Same, used only when the top-level `priority` is `0` or absent. |
| `kv_transfer_params.genesis_agent` (alias `agent`) | PN115, PN90, PN100, PN88, PN101 | One tag, three jobs — see the table below. |
| `kv_transfer_params.persist_disk` (bool) | PN90, PN100 | Overrides the agent allowlist: `true` lets this request's blocks be demoted to the NVMe tier, `false` forbids it. |
| `kv_transfer_params.name` | PN89 | Cosmetic: the name shown in `/v1/kv-offload/requests`. |
| request id (`X-Request-Id` header) | PN115 | Last-resort fallback: an id starting with `coach-`, `primary_`… takes that agent's priority. Auto-generated ids never match. |
| *(nothing — prompt length)* | PN115 | **Automatic priority**: a prompt of up to `GENESIS_PN115_PROMPT_CORTO_TOKENS` tokens skips the prefill queue. It gets nothing else. |

Priority is resolved in that order: explicit `priority`, then `kv_transfer_params.priority`,
then the agent tag, then the request id.

**What `genesis_agent` decides:**

| agent | priority (PN115) | may write to NVMe (PN90) |
|---|---|---|
| `coach` | −10 | yes |
| `primary`, `primary_high` | −8 | yes |
| `planner` | −5 | yes |
| `coder`, `verifier` | −5 | no |
| `primary_low`, `primary_nothink`, `planner_nothink` | 0 | yes |
| `build`, `plan` | 0 | yes |
| `utility` | 5 | no |
| `explorer`, `vision`, `art` | 10 | no |
| *(no tag)* | 0 | **no** |

The third job is labelling: the tag becomes the `agent` label on the `kv_tier_*` Prometheus
series. Names outside `GENESIS_KV_AGENTS` collapse into `other`, so a typo in a client config
cannot create unbounded series. The disk allowlist is `GENESIS_KV_DISK_WRITERS`.

The reason behind the disk column: sub-agents produce single-use context. Letting it reach the
NVMe tier only evicts the long thread's prefix, which is the one thing worth keeping.

In opencode this goes in each agent's `extraBody`:

```jsonc
"extraBody": { "kv_transfer_params": { "genesis_agent": "coach", "persist_disk": true } }
```

### HTTP endpoints

| endpoint | key | what it does |
|---|---|---|
| `GET /v1/kv-offload/requests` | API | Recent requests: id, name, agent, priority, prompt / cached / output tokens, TTFT, prefill and decode tok/s, KV hit rate. No prompt text. |
| `GET /v1/genesis/pid` (alias `/v1/kv-offload/pid`) | API | PN115 state: settings, free KV blocks, and the counters `gated_by_prefill_total`, `short_prompt_bypass_total`, `priority_bypass_total`, `emergency_preemptions_total`. |
| `POST /v1/genesis/pid` (alias `/v1/kv-offload/pid`) | **admin** | Retunes PN115 live, no restart. JSON body, validated — see below. |
| `POST /v1/kv-offload/reset` | **admin** | Empties L1 (GPU), L2 (RAM) and L3 (NVMe), **terminates in-flight requests**, zeroes the metrics. Query flags, all default `true`: `force`, `notify_clients`, `clear_l1`, `clear_l2`, `clear_l3`, `clear_metrics`, `clear_history`. |

Keys accepted by `POST /v1/genesis/pid`. Anything else, or a wrong type or range, is rejected
whole with `400` and never reaches the control file:

| key | type | range | meaning |
|---|---|---|---|
| `enabled`, `kv_gating`, `latency_pid` | bool | | master switch · KV-headroom gate · latency PID (forced off under async scheduling) |
| `max_concurrent_prefills` | int | 0–64 | `0` = no serialisation, `1` = one prefill at a time |
| `short_prompt_tokens` | int | 0–1 000 000 | automatic-priority limit; `0` = off |
| `high_prio_threshold` | int | −1000–1000 | priorities *below* this are "forced" |
| `headroom_ratio` | float | 0–0.9 | share of GPU blocks never committed |
| `min_concurrency`, `max_concurrency` | int | 1–1024 · 0–1024 | never gate below · cap |
| `target_step_ms`, `kp`, `kd` | float | ≥ 0 | latency PID tuning |

```bash
curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://HOST:8320/v1/genesis/pid
curl -s -X POST -H "Authorization: Bearer $GENESIS_ADMIN_API_KEY" \
     -H 'Content-Type: application/json' -d '{"short_prompt_tokens": 1000}' \
     http://HOST:8320/v1/genesis/pid
```

### Security model

**Every Genesis route is authenticated, by construction.** vLLM's own middleware only guards
paths that start with `/v1`; everything else is open by design (`/health`, `/metrics`). Genesis
used to mount aliases outside that prefix — `/kv-offload/requests`, `/kv-offload/reset`,
`/reset_prefix_cache` — and they were open without anyone having decided so: on 2026-09-20,
`GET /kv-offload/requests` with no key returned `200`, and the same route family could wipe
all three cache tiers and cut every request in flight. The fix is structural, not a path
rename: authentication is a dependency of the **router** (`vllm/_genesis/api_auth.py`), so a
route added tomorrow is born protected whatever its prefix — and a test walks every route of
the router and asserts `401` without a key. The unprefixed aliases were removed; nothing used
them, and `/reset_prefix_cache` shadowed one of vLLM's own development endpoints.

**Two keys, two levels.** `VLLM_API_KEY` reads. Routes that *change* the server take the admin
level: with `GENESIS_ADMIN_API_KEY` set they demand that key, so an inference client cannot
retune admission or flush the caches; unset, the normal key is accepted. The admin key also
reads. With no API key configured at all the server is open — same behaviour as vLLM.

**What is *not* protected, and cannot be with one shared key: the request fields.** They are
self-declared. Anyone holding the inference key can tag itself `coach` and get top priority
with the right to preempt, or send `persist_disk: true`. The damage is bounded — preemption
only happens when every slot is taken, and the NVMe tier is capped by PN81's quota
(`GENESIS_KV_DISK_MAX_GB`) — but it is a trust decision, not an access control. It fits a
single-user rig. If the key is ever shared, put a proxy in front that strips or rewrites
`priority` and `kv_transfer_params` per caller.

---

## Getting started

### 1. Credentials

Every compose reads its API key from the environment. **Nothing secret is
versioned** — `.env` is gitignored, only `.env.example` is tracked.

```bash
cp compose/.env.example compose/.env
openssl rand -hex 32          # generate a real key, paste it into compose/.env
```

`GENESIS_ADMIN_API_KEY` is optional: set it and the routes that change the server
(cache reset, admission config) stop accepting the inference key — see
[Security model](#security-model).

Benchmark and diagnostic scripts read the same `VLLM_API_KEY` **with no
default**: a script run without it fails loudly instead of sending a stale key.

### 2. Run

```bash
# the model and its drafter (19 + 1.2 GB) into the models cache the compose mounts
hf download BlairQ/qwen3.8_27b_idiotSavant_sm_86 --local-dir ../models-cache/qwen3.8_27b_idiotSavant_sm_86
hf download BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2 --local-dir ../models-cache/qwen3.8_27b_idiotSavant_sm_86_dflash2

cd compose
docker compose -f docker-compose.qwen38-27b-idiotsavant-sm86.yml up -d
docker inspect genesis-27b-idiotsavant --format '{{.State.Health.Status}}'
```

Boot takes ~4 minutes: patches, `torch.compile` (cached after the first time), CUDA graphs and
pinning the 12 GiB of L2. Check the volume paths at the top of the compose against your layout.
For opencode, add the [conversation-identity plugin](#conversation-identity-from-opencode).

The healthcheck **generates a token** rather than pinging `/health`. A server
that boots but produces garbage is reported unhealthy — that has caught real
failures here.

### 3. Verify

```bash
# every patch decision, with its reason
docker logs genesis-27b-idiotsavant 2>&1 | grep "Genesis Dispatcher"

# KV capacity actually obtained
docker logs genesis-27b-idiotsavant 2>&1 | grep "GPU KV cache size"

# the integer attention backend actually took over
docker logs genesis-27b-idiotsavant 2>&1 | grep "PN131"
```

That third one matters: a patch can report `applied` and still not run. See
PN143 above.

### Before bumping vLLM

Anchor patching **fails silently when upstream moves a line**: the patch reports
`SKIPPED`, the server boots without it, and the only symptom is lower
throughput. Run:

```bash
python3 -m vllm._genesis.preflight_anclajes --bajar v0.29.0
```

It checks every anchor against the target tree without downloading an image, and
reports which patches would break *and actually run in this configuration*.

---

## Status and open problems

Kept deliberately honest — these are the things you would otherwise discover the
hard way:

- **int4 KV is the open goal, and quality is the only blocker.** The kernel,
  the hardware path and a working compose all exist; what is missing is a
  quantization process built around per-group scales along the head dimension.
  See [Next: int4 in the KV cache](#next-int4-in-the-kv-cache).
- **L2 is at 0.21× L1 and wants 9.7 GiB to do its job.** The tiered KV cache
  works and the rescue is reproducible; what limits it here is host RAM, not the
  mechanism. See [Size L2 relative to L1](#size-l2-relative-to-l1--this-is-the-whole-game).
- **PN122 silently corrupted long generations with DFlash2 — fixed 2026-09-20, and worth
  knowing how it hid.** DFlash2 forces vLLM's v2 model runner, which migrates GDN state across
  block boundaries from `mamba_hybrid.py`; PN122's materialisation hooks lived only on the v1
  path, while its "skip the biased copy" edit sat in a kernel both runners share. Every
  880-token block boundary left the recurrent state stale: output degenerated (loops, zeros,
  EOS mid-word) after ~1 000 tokens. `quality-test --quick` scored 27/30 throughout, because
  none of its answers cross a boundary. **The oracle for this patch is a >2 000-token
  generation, single and concurrent** — not a short-answer suite.
- **A second bug the same day:** the new integer FWHT reads 16-bit words, and PN131's
  reference-scale code handed it fp32. Signature: `ek=11` on *every* attention layer in the boot
  log, where a healthy boot shows `ek=0`. Identical values across layers are the tell.
- **The disk KV tier used to mix blocks from different KV formats — fixed 2026-09-21, inside
  PN81.** vLLM names the offload directory from the model, the parallelism, `tokens_per_hash`,
  the *model* dtype and the KV groups. Nothing in that says what the bytes **mean**:
  `--kv-cache-dtype`, whether `k` is stored rotated (PN126 / PN131), which kernel wrote it
  (PN131 stores with its own reference exponents), PN122, the drafter's KV. Change any of them
  and the directory stays the same — and because the tier is on disk, the blocks survive a
  container recreate and the next boot restores them under the new meaning. Measured: a 37K
  prompt repeated across the arms of an A/B came back as `"\n"` + EOS (2 tokens) in 1.8 s; a
  fresh prompt of the same length, **in the same boot**, gave 400 correct tokens in 16.7 s.
  It is the long-context face of "emits two tokens and stops", and the healthcheck never sees
  it: its six-word prompt does not fill a block, so it never touches the tier.
  PN81 now adds a fingerprint of the KV byte format (`vllm/_genesis/kv_formato.py`) to the
  hashed fields: another config, another directory. `K` is deliberately *not* in it — it does
  not change what is stored. The sub-patch is not best-effort: if its anchor drifts, PN81 fails
  as a whole, because a disk tier that mixes formats is worse than no tier. Old directories
  are left unused and PN81's orphan purge removes them after `GENESIS_KV_DISK_ORPHAN_DAYS`.
  **If you A/B anything that changes the KV format, use a different prompt per arm and read
  the output text, not just tok/s** — a whole afternoon of long-context results here had to be
  thrown away for exactly this reason.
- **The decode profile is a configuration behind.** Re-profile against int8 KV +
  DFlash2 K=8 before acting on the percentages above.
- **The drafter still runs in fp16** — 7.6% of every decode step. PN133 is
  written and not enabled.
- **About 8.5% of boots on 0.29.0 came up broken** — the model emits two tokens
  and stops. The generation-based healthcheck catches it. Root cause still open.
- **vLLM 0.29.0 is validated here but is not on the inherited pin allowlist.**
  Boot logs a pin-gate warning; that is expected, not a fault.
- **PCIe links negotiate gen4 ×8, not ×16** on this machine. Every number above
  was measured under that constraint.
- **The 3090s are power-capped — 229 W and 243 W, set by `nvidia-powerlimit.service`** (the
  kernel-level bandwidth notes elsewhere in this repo were taken at the earlier, even 220 W
  cap: sustained SM clock of 810 MHz, real roof 640 GB/s instead of 730). Two consequences:
  every number here is ~20% short of what the silicon does uncapped on prefill
  ([measured](#what-the-power-cap-costs)), and the clock drifts with temperature, so a burst
  overestimates and a single run per arm is not evidence.

### Dead ends

Recorded so they are not repeated. Both came from single runs and both had to be
retracted:

- **"The draft group vetoes the other groups' hit, remove the veto."** A patch
  (PN147) was written, applied, and measured with a successful rescue — n=1, and
  it did not reproduce. With PN91's budget raised and PN147 *off*, the same group
  hits 19/19. The data was always there. PN147 was deleted.
- **"PN100's staging ring wastes 44% of L2, disabling it gains capacity."**
  Worse: the attention prefix went from 12/20 to 0/20 hits. Bounding ephemeral
  traffic is what protects the long thread's prefix.

This subsystem has genuine run-to-run variance. Take every result with a clean
disk, a restart, and at least two repetitions.

---

## Repository layout

```
vllm/_genesis/        the patches, kernels and dispatcher — the project itself
  ├─ wiring/          one module per patch, grouped by subsystem
  ├─ kernels/         PTX and Triton kernels (integer attention, Marlin, GDN)
  └─ tests/           unit tests for the patch machinery
compose/              the two engine composes + .env.example
tests/
  ├─ bench/medicion/  one script per question (acceptance, KV blocks, concurrency)
  ├─ bench/resultados/raw benchmark output, with the config that produced it
  └─ repro/           standalone harnesses that reproduce upstream bugs without a GPU
docs/                 subsystem write-ups, hardware notes, decode profiles
benchmarks/           historical measurement campaigns
```

Each patch states in its own docstring *why* it exists and what was measured.
Patches carry `upstream_drift_markers` and retire themselves once they detect
that upstream merged the underlying fix.

---

## Credits

Built on [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches).
Several fixes and the whole measurement methodology come from
[noonghunna/club-3090](https://github.com/noonghunna/club-3090), whose benchmark
and quality harnesses are used here directly.

Apache 2.0 — see [LICENSE](LICENSE).
