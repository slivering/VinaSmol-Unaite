# How to use LitGPT

We use [LitGPT](https://github.com/Lightning-AI/litgpt) as a training and finetuning framework for VinaSmol. We **highly recommend** to read the documentation of LitGPT, in particular some of the [tutorials](https://github.com/Lightning-AI/litgpt/tree/main/tutorials) can be helpful.

Nevertheless, LitGPT uses its own model format and only "supports" [a specific set of model architectures](https://github.com/Lightning-AI/litgpt/tree/main?tab=readme-ov-file#all-models), which can create some confusion at first. In particular, using VinaSmol requires small additional steps as described below.

## Convert a HuggingFace Transformers checkpoint to a LitGPT checkpoint

[Reference](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/convert_hf_checkpoint.md)

In order to run continued pretraining or finetuning, a HF checkpoint needs to be converted to the LitGPT format. For **SmolLM2-360M-Instruct**, use the following command:

```bash
litgpt convert_to_litgpt --model_name SmolLM2-360M-Instruct checkpoint_dir
```

This should create a `lit_model.pth` and a `model_config.yaml` file.

For **VinaSmol**, you will need to edit the `model_config.yaml` file so it matches the actual vocabulary size after [extension and embedding resizing](../vinasmol/tokenization/README.md#extend-smollms-vocabulary-with-vietnamese).

```yml
# Replace 49152 with the actual vocabulary size of the merged tokenizer
padded_vocab_size: 55936
...
vocab_size: 55936
```

Otherwise, you can run the two steps above in a single command:

```bash
bash scripts/convert_vinasmol_to_litgpt.sh --old_vocab_size 49152 --new_vocab_size 55936 checkpoint_dir
```

## Convert a LitGPT checkpoint to a HuggingFace Transformers checkpoint

[Reference](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/convert_lit_models.md)

```bash
litgpt convert_from_litgpt checkpoint_dir converted_dir
```

This will create a large, fp32 `model.pth` file. In order to convert to a compressed `.safetensors` format, you can run the following command in the root directory.

```bash
python scripts/pth_to_safetensors.py converted_dir converted_dir
```

If you want to use the PyTorch model format, you can just rename `model.pth` to `pytorch_model.bin` for compatibility with HF Transformers.