# Finetuning

> [!NOTE]
>  This page is a work in progress.

Finetuning aims to improve the capabilities, the response quality and the safety of LLMs.

## Recipe

### ORPO

We use [ORPO](https://arxiv.org/abs/2403.07691) to perform supervised fine-tuning without a base model.

ORPO is integrated in the [transformers](https://huggingface.co/docs/trl/main/en/orpo_trainer) library.

Alternatively, [QDoRa](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html#dora) can be an efficient finetuning method.

### Merging

Details [here](../merging/README.md).

## Datasets

We are considering the finetuning datasets below.

### Collections

- https://huggingface.co/5CD-AI/collections
- https://ai.stanford.edu/~sttruong/villm/
- https://nlp.uit.edu.vn/datasets

### Misc

- https://huggingface.co/akoksal/muri-it-language-split : Instruction-tuning with reverse instructions
- https://huggingface.co/datasets/vilm/OpenOrca-Viet : Reasoning
- https://huggingface.co/PaDaS-Lab/webfaq : FAQ (unclean)
- https://huggingface.co/SEACrowd/bactrian_x : Instruction-response pairs (Google Translated)
- https://huggingface.co/SEACrowd/xquad : Cross-lingual question answering
- https://huggingface.co/SEACrowd/belebele : Reading comprehension with multiple choices
- https://huggingface.co/SEACrowd/wikihow_gosc: Wikihow goal-oriented script
- https://huggingface.co/SEACrowd/vimmrc : Machine comprehension for common-sense reasoning, aimed at 1-5th graders
- https://huggingface.co/VTSNLP/base_normal_quality : Toxicity classification
- https://github.com/ThanhChinhBK/vietnews : News summarizations
