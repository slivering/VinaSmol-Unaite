# Tokenization

## Get started

### Prerequisites

The Vietnamese datasets need to have been preprocessed, filtered and deduplicated. Follow the steps [here](../training/dataset/README.md).

### Extend SmolLM's vocabulary with Vietnamese

```bash
python -m vinasmol.tokenization.training \
    vinasmol/training/dataset/data/deduped/vi-all \
    --tokenizer-out-dir vinasmol/tokenization/data \
    --vietnamese-vocab-size 10000
python -m vinasmol.tokenization.vocab_extension \
    vinasmol/tokenization/data/merged \
    vinasmol/tokenization/data/smollm_extended
```

The merged tokenizer and the SmolLM weights with extended vocabulary can be found in the
directory specified by `--tokenizer-out-dir`.

> [!NOTE] Terminology
> 
> We refer as *new* the tokenizer trained anew on Vietnamese corpora, with of vocabulary size specified by `--vietnamese-vocab-size`.
> We refer as *merged* the tokenizer with the previous English vocabulary and the added Vietnamese vocabulary from the *new* tokenizer. The number of added tokens is **lower** than `--vietnamese-vocab-size`.

---

## Rationale

SmolLM is a monolingual trained on English. Therefore, it has no Vietnamese tokens (see [this Jupyter notebook](./language_exploration_smollm.ipynb) for experiments and results visualization).

Adapting the base tokenizer for Vietnamese yields the following improvements:
1. A lower tokenizer fertility largely contributes to the reduction of training computational costs.
2. A language-specific tokenizer improves language understanding and task accuracy ([arXiv](https://arxiv.org/abs/2502.12560v2)).

## Recipe

The SmolLM tokenizer is trained on English with a vocabulary size of 49152 and a window size of 8192.

We train a new tokenizer on a subset of our training Vietnamese corpora ([details](../training/dataset/README.md)). We add about 10000 new tokens to the base tokenizer, filter rare tokens and add accentuated letters if not present.

Vietnamese tokenization differs from Latin language tokenization since space is not word separator in Vietnamese. Therefore, we use a SentencePiece tokenizer, which is language-agnostic. Additionally, we enable BPE-Dropout to improve robustness against different input formats.

## Validity of our approach

We believe our approach is valid for the following reasons:

1. SmolLM has no cross-lingual alignment between English and Vietnamese, so the negative impacts of vocabulary extension outlined by [Zhao et al.](https://arxiv.org/abs/2401.01055) and in [Sailor](https://arxiv.org/abs/2404.03608) are unlikely to apply.
2. We employ the [EEVE](https://arxiv.org/abs/2402.14714v1) method for vocabulary extension, which is designed to avoid friction with the introduction fo a new language in the model training.

## Adjustments

The EEVE methodology requires several modifications for VinaSmol.

1. There is some overlap between Vietnamese tokens and English tokens (e.g "can", "a"...), which can be determined with a Vietnamese dictionary, for instance. Therefore, all of the Vietnamese token embeddings should be trained. This could include unaccentuated tokens.
2. The embedding initialization will differ from EEVE-Korean. [More details](../training/README.md#embedding-initialization).

## Citations

- [Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models](https://arxiv.org/abs/2402.14714v1)
- [MTet: Multi-domain Translation for English and Vietnamese](https://doi.org/10.48550/arxiv.2210.05610)
- [vietnamese-wordlist](https://github.com/duyet/vietnamese-wordlist)
