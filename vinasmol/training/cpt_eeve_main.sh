#!/bin/bash

# HACK: use different seeds for each stage so that on average,
# every example is seen once during the whole training. 

# FIXME: manual intervention needed if we want to train until convergence

set -e

pretrain_eeve=../../scripts/eeve/pretrain.py
eeve_inter_dir=./checkpoints/VinaSmol/eeve
eeve_final_dir=./checkpoints/VinaSmol

python $pretrain_eeve \
    --config ./cpt_eeve_main.yml \
    --eeve_stage 0 \
    --train.max_tokens 100000000 \
    --train.micro_batch_size 4 \
    --seed 2025090500 \
    --out_dir $eeve_inter_dir/VinaSmol_stage_1

litgpt convert_pretrained_checkpoint \
    $eeve_inter_dir/VinaSmol_stage_1/final \
    $eeve_final_dir/VinaSmol_eeve_stage_1

python $pretrain_eeve \
    --config ./cpt_eeve_main.yml \
    --eeve_stage 1 \
    --train.max_tokens 200000000 \
    --train.micro_batch_size 4 \
    --seed 2025090501 \
    --initial_checkpoint_dir $eeve_final_dir/VinaSmol_eeve_stage_1 \
    --out_dir $eeve_inter_dir/VinaSmol_stage_2

litgpt convert_pretrained_checkpoint \
    $eeve_inter_dir/VinaSmol_stage_2/final \
    $eeve_final_dir/VinaSmol_eeve_stage_2

python $pretrain_eeve \
    --config ./cpt_eeve_main.yml \
    --eeve_stage 2 \
    --train.max_tokens 500000000 \
    --train.micro_batch_size 2 \
    --seed 2025090502 \
    --initial_checkpoint_dir $eeve_final_dir/VinaSmol_eeve_stage_2 \
    --out_dir $eeve_inter_dir/VinaSmol_stage_3

litgpt convert_pretrained_checkpoint \
    $eeve_inter_dir/VinaSmol_stage_3/final \
    $eeve_final_dir/VinaSmol_eeve_stage_3

python $pretrain_eeve \
    --config ./cpt_eeve_main.yml \
    --eeve_stage 3 \
    --train.max_tokens 500000000 \
    --train.micro_batch_size 2 \
    --seed 2025090503 \
    --initial_checkpoint_dir $eeve_final_dir/VinaSmol_eeve_stage_3 \
    --out_dir $eeve_inter_dir/VinaSmol_stage_4

litgpt convert_pretrained_checkpoint \
    $eeve_inter_dir/VinaSmol_stage_4/final \
    $eeve_final_dir/VinaSmol_eeve_stage_4

# For compatibility with regular training (cpt_stage_1_main.sh)
cp $eeve_final_dir/VinaSmol_eeve_stage_4 $eeve_final_dir/VinaSmol_stage_1