# Continued Pre-Training

> [!NOTE]
>  This page is a work in progress.

## Training data

Details [here](./dataset/README.md).

## Framework

We use [litgpt](https://github.com/Lightning-AI/litgpt) for continued pretraining, which supports SmolLM2-360M.

In order to follow the multi-stage training of EEVE, we customize the [continued pretraining recipe of litgpt](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/pretrain.md#continued-pretraining-on-custom-data) by freezing the adequate parameters before the training starts.

## Recipe

### Vocabulary extension

We use the [Efficient and Effective Vocabulary Extension](https://arxiv.org/abs/2402.14714v1) method (EEVE), which encompasses both tokenization and training.

#### Tokenizer training

We train a new tokenizer on Vietnamese corpora and extend the base tokenizer with the new tokens. Details for running tokenizer training, merging, vocabulary extension and embedding initialization can be found [here](../tokenization/README.md).

#### Embedding initialization

For the input embeddings of the new Vietnamese tokens, we use the average embeddings of their subword tokens (default in the `tokenizers` library, as in [Hewitt](https://nlp.stanford.edu/~johnhew/vocab-expansion.html)). Whenever possible, we use a convex combination of the initialized embedding and the embedding of their translation using [EnViT5-base](https://huggingface.co/VietAI/envit5-translation).

Since SmolLM2-360M has tied embeddings due to its size, we simply propagate the input embeddings initialization to the output embeddings. This differs from the output embeddings initialization in EEVE.

<details>
<summary>Read more...</summary>
For the output embeddings of the new tokens, Kim et al. suggest to initialize them with the embeddings of the first subword token. This harmonization approach works if the base model has not its embeddings tied and has already some Vietnamese completion capabilities. Furthermore, it would be harmful for an English/Vietnamese model since most of the Vietnamese tokens start with a Latin consonant, which is already in the English alphabet. Single-consonant token embeddings already have a value in SmolLM2 and most of their information is useful for English completion, not Vietnamese completion.
</details>

### Multi-stage training

#### Requirements

- A SmolLM2-360M checkpoint with extended vocabulary ([instructions here](../tokenization/README.md#extend-smollms-vocabulary-with-vietnamese))
- The prepared Vietnamese, English and code datasets ([instructions here](./dataset/README.md#prepare-data-for-training))

Edit the `cpt_stage_1_main.yml` and update the model architecture:

```yml
model_config:
  ...
  vocab_size: <vocabulary size of SmolLM2-360 extended here>
  padded_vocab_size: <vocabulary size of SmolLM2-360 extended here>
```

The training pipeline can then be started with the following commands:

```bash
# Convert the SmolLM2-360M weights & tokenizer to a litgpt checkpoint
litgpt convert_to_litgpt vinasmol/tokenization/data/smollm_extended

cd vinasmol/training
# Start initial continued pretraining with sequences of length 2048
bash cpt_stage_1_main.sh
```

The different training stages used in EEVE are depicted below.

![EEVE pipeline](https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0/resolve/main/EEVE_figure.png)

Refer to the [original paper](https://arxiv.org/abs/2402.14714v1) for more information.

Since SmolLM2-360M has tied embeddings, we only perform stages 3, 4, 6, 7 from the original EEVE pipeline.

### ReLoRA

In order to pretrain the model on even more tokens with limited resources, [ReLoRA](https://arxiv.org/abs/2307.05695) is an interesting technique that drastically cuts down on the compute and memory requirements.

We can use ReLoRA during stages 6 and 7 of the multi-stage training, where the number of trainable parameters increases with the transformer layers.

While ReLoRA may be unnecessary for very small LLMs such as [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), such an approach would be crucial for training larger models such as [Lucie-7B](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1).

Implementations (not reviewed yet):
- https://github.com/ElleLeonne/Lightning-ReLoRA
- https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/llama-2/relora.yml

How to integrate both EEVE and ReLoRA into existing training frameworks remains very unclear. Custom training code is very likely to be necessary, therefore a PyTorch Lightning-based solution could be more suitable for customization.

### Hyperparameters

Refer to https://huggingface.co/blog/smollm and the [SmolLM2 paper](https://arxiv.org/abs/2502.02737v1) for hyperparameter tuning.

Similar to [Sailor 7B](https://arxiv.org/abs/2404.03608), we adjust the language mixture proportions and the learning rate based on initial experiments.

### Context extension

We continued the pretraining of SmolLM2 on sequence lengths of 2048 to save costs. However, this caused the model to lose its context length extension of 8192. Therefore we rerun context length extension for the final VinaSmol model.

We follow the procedure outlined by [Gao et al.](https://arxiv.org/abs/2410.02660), keeping the same data mixture but upsampling long documents within each dataset.

### Later stages and annealing

Following the approach used by [SmolLM2](https://arxiv.org/abs/2502.02737v1), we add high-quality content, technical content, textbooks, medium-sized and [instruction](https://magazine.sebastianraschka.com/p/instruction-pretraining-llms#%C2%A7pretraining-with-instruction-data) datasets during the annealing phase in order to maximize their impact.

The new  high-quality datasets are added for annealing:
- CCVJ (Vietnamese)
- OlmOCR-PeS2o, OpenWebMath, StackMathQA (English)

## Further improvements

### Larger models

Using a larger base model such as [Lucie 7B](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1) may improve the model capabilities and degree of multilinguality.

Techniques such as [depth upscaling](https://planetbanatt.net/articles/modelmerging.html#orgf613f37) used by [SOLAR 10.7B](https://arxiv.org/abs/2312.15166) could be used to increase the size of the model and scale to a higher number of parameters. However it's unclear whether depth upscaling coud prove to be useful for very small models.

## Citations

- [litgpt, 2023](https://github.com/Lightning-AI/litgpt)