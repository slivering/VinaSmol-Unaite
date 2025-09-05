#!/bin/bash

oldsize=49152
newsize=55936
model_name=SmolLM2-360M-Instruct
dir=.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -o|--old-vocab-size) oldsize="$2"; shift ;;
        -n|--new-vocab-size) newsize="$2"; shift ;;
        --model_name) model_name="$2"; shift ;;
        *) dir="$1"; shift ;;
    esac
    shift
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo "Converting $dir to LitGPT and copying configuration from $model_name"
litgpt convert_to_litgpt --model_name $model_name $dir
echo "Replacing *vocab_size: $oldsize with *vocab_size: $newsize in model_config.yaml"
sed -i "s/$oldsize/$newsize/g" "$dir/model_config.yaml"