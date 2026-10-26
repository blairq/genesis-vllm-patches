---
license: apache-2.0
base_model: BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2
tags: [speculative-decoding, dflash, dcoco, draft-model]
---

# qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco

The DFlash2 drafter of **[qwen3.8_27b_idiotSavant_sm_86](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86)**
([dflash2](https://huggingface.co/BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2)) with a **DCOCO** corrector on top of its
candidate selector.

DFlash2 proposes 16 candidates per draft position and scores the transitions between them with a low-rank bilinear
selector. DCOCO ([Eisenstadt et al., arXiv 2610.12570](https://arxiv.org/abs/2610.12570),
[code](https://github.com/two-delta/DCOCO)) adds a small MLP (~2-3 M parameters) that scores each transition from a
*pair state* of adjacent draft hidden states and per-candidate features (target embedding + slot context). Here it is a
**residual** correction of the selector's table: with the head zeroed the drafter is exactly the original one. The chain
and the 8-node draft tree both use the corrected table.

- **Layers and weights:** the DFlash2 drafter is unchanged (W4A16, same int4 values as `dflash2`, unless trained jointly);
  the head is stored as `dcoco.*` in bf16, its hyperparameters in `dflash_config` (`dcoco_emb_dim`, ...).
- **Serving:** needs the corrector in vLLM: [vllm-idiotsavant](https://github.com/blairq/vllm-idiotsavant) with
  `"dcoco": true` in `--speculative-config`, or [genesis-vllm-patches](https://github.com/blairq/genesis-vllm-patches)
  with `GENESIS_ENABLE_PN178_DCOCO=1`. A stock vLLM cannot load the `dcoco.*` tensors.
- **Training:** `entrenamiento/dflash2-dcoco/` in genesis-vllm-patches, on the same agent captures as the drafter
  (teacher-forced transition loss, weighted e^{-(p-1)/4}).

## Results

To be filled from `registro.json` (held-out requests) and the vLLM A/B of `dflash2_dcoco.sh`:

| | accepted length, chain | accepted length, 8-node tree | tok/s (agent bench) |
|---|---|---|---|
| dflash2 (selector) | | | |
| dflash2_dcoco | | | |

## Usage (vllm-idiotsavant)

```bash
--speculative-config '{"method":"dflash","model":"BlairQ/qwen3.8_27b_idiotSavant_sm_86_dflash2_dcoco",
  "num_speculative_tokens":8,"tree_drafting":true,"draft_vocab_size":65536,"draft_residual_scale":32,"dcoco":true}'
```
