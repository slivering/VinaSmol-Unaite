# Finetuning

> [!NOTE]
>  This page is a work in progress.

We finetune VinaSmol on Vietnamese conversation and instruction data in order to improve its response quality. We also perform English finetuning again as continued pretraining on billions of tokens significantly deteriorates instruction capabilities.

## Run

### Requirements

You need [a converted LitGPT pre-training checkpoint](../training/README.md#checkpoint-conversion).

If you have a HuggingFace checkpoint, make sure to follow the instructions [here](../../docs/litgpt_help.md#convert-a-huggingface-transformers-checkpoint-to-a-litgpt-checkpoint).

### Finetuning runs

Finetune the base model on multiple chat and instruction datasets:

```bash
cd vinasmol/finetuning
litgpt finetune_lora --config ./vi_alpaca.yml
litgpt finetune_lora --config ./vi_multi_turn_alpaca.yml
litgpt finetune_lora --config ./lima.yml
```

All of these finetuning steps are independent and can be done in parallel or on different machines.

### Conversion to HuggingFace Transformers

```bash
cd scripts
bash ./lit_to_hf
```

### Merge

TODO: mergekit and link to [merging/](../merging/README.md)

## Recipe

### ORPO

We use [ORPO](https://arxiv.org/abs/2403.07691) to perform supervised fine-tuning without a base model.

ORPO is integrated in the [transformers](https://huggingface.co/docs/trl/main/en/orpo_trainer) library.

Alternatively, [QDoRa](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html#dora) can be an efficient finetuning method.

### Merging

Details [here](../merging/README.md).

Alternative: train from SmolLM2-360M base model and add Instruct residual like suggested by [Jindal et al.](https://arxiv.org/abs/2410.10739v1). May not be applicable given that adding a whole new language affects the model much more than domain-specific adaptation.

## Datasets

### Vietnamese datasets

- [MURI-IT](https://huggingface.co/datasets/akoksal/muri-it-language-split): Instruction-tuning with reverse instructions generated from [MADLAD-400-3B-MT](https://huggingface.co/google/madlad400-3b-mt), 25k rows
- [vi_alpaca_clean](https://huggingface.co/datasets/tsdocode/vi_alpaca_clean) (CC-BY, 28k rows), [alternative](https://huggingface.co/datasets/Chat-Error/Vietnamese_x_Alpaca)
- [Vietnamese Multi-turn Chat Alpaca](https://huggingface.co/datasets/lamhieu/alpaca_multiturns_dialogue_vi) (MIT, Apache 2.0)
- [Vietnamese CosmosQA Google Translated](https://huggingface.co/datasets/5CD-AI/Vietnamese-cosmos-qa-gg-translated), a translated version of [CosmosQA](https://huggingface.co/datasets/allenai/cosmos_qa) (CC-BY)

We are also considering the synthetic math datasets below.

- [Vietnamese OpenMathInstruct-1 Google Translated](https://huggingface.co/datasets/5CD-AI/Vietnamese-nvidia-OpenMathInstruct-1-50k-gg-translated) ([original from Nvidia](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1), from Mixtral-8x7B outputs, permissive Nvidia license)
- [Vietnamese MetaMathQA Google Translated](https://huggingface.co/datasets/5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated) ([original from Meta](https://huggingface.co/datasets/meta-math/MetaMathQA), GPT-3.5 outputs, MIT)
- [Vietnamese ORCA Math World Problems 200k Google Translated](https://huggingface.co/datasets/5CD-AI/Vietnamese-microsoft-orca-math-word-problems-200k-gg-translated) ([original from Microsoft](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k), GPT-4 outputs, MIT)
- https://huggingface.co/datasets/ontocord/viet4all (based on [OpenHermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5), GPT-4 outputs)

### Related Collections

- https://huggingface.co/5CD-AI/collections
- https://ai.stanford.edu/~sttruong/villm/
- https://nlp.uit.edu.vn/datasets

### Other miscellaneous datasets considered

- https://huggingface.co/datasets/SEACrowd/wikihow_gosc: Wikihow goal-oriented script
- https://huggingface.co/datasets/SEACrowd/vimmrc : Machine comprehension for common-sense reasoning, aimed at 1-5th graders
- https://huggingface.co/datasets/vilm/OpenOrca-Viet : Reasoning
- https://huggingface.co/datasets/SEACrowd/belebele : Reading comprehension with multiple choices

- https://huggingface.co/datasets/PaDaS-Lab/webfaq : FAQ (unclean)
- https://huggingface.co/datasets/VTSNLP/base_normal_quality : Toxicity classification
- https://github.com/ThanhChinhBK/vietnews : News summarizations

### English datasets

We use [Smol-SmalTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) which was used to finetune SmolLM2-360M.


