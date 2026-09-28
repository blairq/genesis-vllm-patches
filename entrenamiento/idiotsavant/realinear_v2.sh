#!/bin/bash
# idiotSavant v1 -> v2 (o_proj/out_proj con Hadamard por cabeza, in_proj_a/b en W4) con las Hessianas de la
# etapa A (cuant-cache/A, attn_in sobre x = g n). Corre en la imagen de vLLM (tiene torch/transformers).
P=/home/usuario/Proyectos
docker run --rm --name idiotsavant-realinear --gpus '"device=0"' --memory 20g --memory-swap 20g \
  -v $P/genesis-vllm-patches:/repo:ro -v $P/models-cache:/models -v $P/cuant-cache/A:/hess:ro \
  --entrypoint python3 vllm/vllm-openai:v0.29.0 /repo/entrenamiento/idiotsavant/idiotsavant.py realinear \
  --bf16 /models/orcarouter-qwen3.8-27b-uncensored-bf16 --modelo /models/qwen3.8_27b_idiotSavant_sm_86 \
  --hessianas /hess --h_sobre_x --trabajo /models/trabajo_idiotsavant_v2 \
  --salida /models/${SALIDA:-qwen3.8_27b_idiotSavant_sm_86_v2} "$@"
echo "codigo $?"
