# Continued Pre-Training

> [!NOTE]
>  This page is a work in progress.

## Training data

Details [here](./dataset/README.md).

## Framework

TODO: depends on the machine and the base model. 3D parallelism is crucial if the model is large. Possibly DeepSpeed-ZERO for a whole cluster, or something simpler for a single-node setup.

EEVE only requires parameter freezing, but ReLoRA might be harder to integrate with existing distributed frameworks.

## Recipe

### Vocabulary extension

We use the [Efficient and Effective Vocabulary Extension](https://arxiv.org/abs/2402.14714v1) method (EEVE), which encompasses both tokenization and training.

#### Tokenizer training

We train a new tokenizer on Vietnamese corpora and extend the base tokenizer with the new tokens. [Details](../tokenization/README.md).

#### Embedding initialization

For the input embeddings of the new Vietnamese tokens, we use the average embeddings of their subword tokens. Alternatively, we could initialize them to the embedding of their English translation (possibly a sentence), using an auxiliary translation model such as [PhoBERT](https://huggingface.co/vinai/phobert-base), plus a random normal noise.

For the output embeddings of the new tokens, the EEVE method suggests to initialize them with the embeddings of the first subword token. While this approach makes sense when the base model was trained on a bit of Vietnamese, SmolLM has about zero Vietnamese completion capabilities, and therefore such harmonization strategy might not be helpful.

Initial experiments should guide us towards the best initialization method.

### Multi-stage training

The different training stages used in EEVE are depicted below.

![EEVE pipeline](https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0/resolve/main/EEVE_figure.png)

Refer to the [original paper](https://arxiv.org/abs/2402.14714v1) for more information.

### ReLoRA

In order to pretrain the model on even more tokens with limited resources, [ReLoRA](https://arxiv.org/abs/2307.05695) is an interesting technique that drastically cuts down on the compute and memory requirements.
Diff lora

We can use ReLoRA during stages 6 and 7 of the multi-stage training, where the number of trainable parameters increases with the transformer layers.

While ReLoRA may be unnecessary for very small LLMs such as [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M), such an approach would be crucial for training larger models such as [Lucie-7B](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1).

Implementation (not reviewed yet):
https://github.com/ElleLeonne/Lightning-ReLoRA

### Hyperparameters

Refer to https://huggingface.co/blog/smollm and https://arxiv.org/abs/2502.02737v1 for experimental tuning

Similar to [Sailor 7B](https://arxiv.org/abs/2404.03608), we adjust the language mixture proportions and the learning rate based on initial experiments.

### Later stages and annealing

Following the approach used by [SmolLM2](https://arxiv.org/abs/2502.02737v1), we add high-quality content, technical content, textbooks, medium-sized and instruction datasets during the annealing phase in order to maximize their impact.

### Context extension

We follow the procedure outlined by [Gao et al.](https://arxiv.org/abs/2410.02660), keeping the same data mixture but upsampling long documents within each dataset.

## Further improvements

### Larger models

Using a larger base model such as [Lucie 7B](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1) may improve the model capabilities and degree of multilinguality.

Techniques such as [depth upscaling](https://planetbanatt.net/articles/modelmerging.html#orgf613f37) used by [SOLAR 10.7B](https://arxiv.org/abs/2312.15166) could be used to increase the size of the model and scale to a higher number of parameters. However it's unclear whether depth upscaling coud prove to be useful for very small models.
