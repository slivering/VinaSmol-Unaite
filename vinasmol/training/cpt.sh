#!/bin/bash

# FIXME: possible problems with different architecture
# TODO: include English and Vietnamese
# TODO: add SmolLM2 vocab extension code
# TODO: fork litgpt to support EEVE
litgpt pretrain HuggingFaceTB/SmolLM2-360M \
  --tokenizer_dir "../tokenization/data/merged" \
  --initial_checkpoint_dir "../tokenization/data/smollm_extended" \
  --data TextFiles \
  --data.train_data_path "./dataset/data/filtering/vi-all/deduped" \
  --data.seed 42 \
  --train.max_tokens 10_000 \
  --out_dir data/vinasmol_stage_1