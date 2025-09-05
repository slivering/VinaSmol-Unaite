#!/bin/bash

ft_dir=vinasmol/finetuning/checkpoints/VinaSmol
out_dir=vinasmol/finetuning/hf_checkpoints/VinaSmol

for name in $(ls $ft_dir); do
    litgpt convert_from_litgpt $ft_dir/$name/final $out_dir/$name
    python scripts/pth_to_safetensors.py $out_dir/$name $out_dir/$name
done