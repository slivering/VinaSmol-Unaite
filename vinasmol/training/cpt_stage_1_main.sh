#!/bin/bash

# FIXME: possible problems with different architecture
# TODO: fork litgpt to support EEVE
litgpt pretrain HuggingFaceTB/SmolLM2-360M \
  --tokenizer_dir "../tokenization/data/optimized" \
  --initial_checkpoint_dir "../tokenization/data/smollm_extended" \
  --data LitData \
  --data.path "./dataset/data/tokenized" \
  --data.seed 20250829 \
  --train.max_tokens 10_000 \
  --out_dir data/checkpoints/vinasmol_stage_1