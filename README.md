# VinaSmol: Extending SmolLM with Vietnamese

VinaSmol is a proof-of-concept project that aims to add Vietnamese language capabilities and knowledge to [SmolLM 360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).

Our approach aims to demonstrate that efficient pretraining methods on a medium-sized Vietnamese dataset (less than 10B tokens) can effectively integrate a new language into an LLM that was not trained on it.

Future plans include extending these techniques to [Lucie 7B](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1).

## Disclaimer

This model is still in heavy development and is not ready for general use.

## Development

Install [uv](https://docs.astral.sh/uv/) and set up the virtual environment:

```bash
uv lock
uv sync
```

## Training methods

We employ a comprehensive pipeline to integrate the Vietnamese language into SmolLM. We outline the main steps below:

1. [Extend an existing model’s tokenizer with new Vietnamese tokens](./vinasmol/tokenization/README.md).
2. [Continued pre-training of the model on Vietnamese](./vinasmol/training/README.md)
3. [Fine-tuning for instructions, capabilities and safety](./vinasmol/finetuning/README.md)
4. [Model merging to combine the finetuned checkpoints](./vinasmol/merging/README.md)
5. [Evaluation of the final model in both Vietnamese and English](./vinasmol/evaluation/README.md)

We combine state-of-the-art techniques used for LLMs targeting South-East Asian languages, prioritizing efficiency and results within reasonable compute resources. The goal is to enable replication of the VinaSmol training process on a single node and in reasonable time.

## Related work

Existing state-of-the-art Vietnamese models are developed by organizations with larger resources, often targeting South-East Asian languages as a whole. Few models have been developed with frugal training in mind.

Furthermore, there is a lack of fully open-source and performant Vietnamese models. Indeed, most of the available models are based on Qwen, Llama or Gemma which do not disclose their training datasets. The authors of other base Vietnamese models often remain vague on the provenance of their training data.

## Future directions

We are looking forward further enhancements for VinaSmol:

- Better dataset curation
- Improved reasoning abilities
- Safety-aware pretraining
