# Training data

This folder contains code for downloading and processing pretraining datasets for VinaSmol.

## Build the pretraining dataset

First, activate a virtual environment shell and log in to HuggingFace:

```bash
uv run bash
hf auth login
```
Your HuggingFace account needs to accept agreements for the following datasets:
- [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)
- [MathPile Commercial](https://huggingface.co/datasets/GAIR/MathPile_Commercial)

The next commands should be executed in the root folder of the repository.


## Download data

Download and preprocess the Vietnamese and English datasets by executing:

```bash
python -m vinasmol.training.dataset.datasets
```

Around 10 GB of data will be downloaded from HuggingFace. The preprocessed datasets will be stored in `./data`.



## Filter and deduplicate data

Filtering, redaction and deduplication can be run with the following command:

```bash
uv run python -m vinasmol.training.dataset.filtering
```

For now, filtering is only implemented for the Vietnamese datasets. The final results are dumped into `./data/vi-all/deduped`. Information about removal will be written into `./data/vi-all/removed`.


## Recipe

### English and code datasets

Adding English and code datasets to the continued pretraining mixture avoids catastrophic forgetting of the acquired language proficiency, capabilities and knowledge of the base model. 

We can use [Lucie Training Dataset](https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset) in order to introduce new content to SmolLM, keeping only the English and code subsets. The upsampling coefficients used by Lucie 7B may be tweaked to increase the focus on high-quality sources.

The [SmolLM Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus): is a well-filtered pretraining corpus used by the SmolLM models. Retraining on the SmolLM corpus might not be as beneficial as training on another dataset, however the SmolLM curation methods serve as inspiration for our Vietnamese training datasets.

### Vietnamese datasets

- [Wikipedia](https://huggingface.co/datasets/omarkamali/wikipedia-monthly) (Vietnamese)
- [epfml/FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ): General, high-quality web dataset.
- [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX): General web dataset.
- [madlad-400_vi](https://huggingface.co/datasets/Symato/madlad-400_vi): General web dataset. Clean Vietnamese subset of [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400)
- [BKAI News Corpus](bkai-foundation-models/BKAINewsCorpus): large news dataset.
- [phongmt184172/mtet](https://huggingface.co/phongmt184172/mtet): Parallel Vietnamese/English pairs.
- [doanhieung/vbpl](https://huggingface.co/doanhieung/vbpl): Vietnam’s official law texts.

### Training budget

We plan to continue the pretraining of SmolLM2 on around 2B training tokens, as for [EEVE-Korean](https://arxiv.org/abs/2402.14714). Even if EEVE-Korean has 10B parameters, its base model ([SOLAR 10.7B](https://arxiv.org/abs/2312.15166)) was likely pretrained on a little Korean before, so both factors might compensate.


### Missing resources

- **Vietnamese literature with a permissive license** (ebooks, fiction works, public domain works…) The [National Library of Vietnam](http://nlv.gov.vn/) has a [digital library](http://dlib.nlv.gov.vn/) which used to host such books + doctoral theses, but it is not accessible via Internet right now…
    - [vnthuquan.net](http://vnthuquan.net) is the underlying source of the 10000 Vietnamese books dataset on [HuggingFace](https://huggingface.co/datasets/thailevann/10000_Vietnamese_Books) and on [archive.org](https://archive.org/details/vnthuquan), but copyright is not entirely clear for editions of the website before 2006. We could filter works that are guaranteed to be in public domain using the publication date and possibly the author's date of death if famous.
- **Research publications in Vietnamese** exist and can be found on open-access websites such as [Vietnam Journals OnLine](https://vjol.info.vn/) (only publications in Vietnamese language) and [SemanticScholar](https://www.semanticscholar.org/) (can’t filter by language easily).
    - There is no dataset for VJOL and while the site’s archives can be scraped, this represents significant work.
    - SemanticScholar provides an English-only dataset ([S2ORC](https://huggingface.co/datasets/sentence-transformers/s2orc)). Reproducing an equivalent dataset for Vietnamese will require access to the [API](https://api.semanticscholar.org/api-docs/graph) and clever search techniques for language filtering.
    - PDF preprocessing can be done with a tool such a https://github.com/allenai/science-parse


## Data preparation

Our data preparation pipeline uses [datatrove](https://github.com/huggingface/datatrove). We take inspiration from [Sailcraft](https://github.com/sail-sg/sailcraft) for Vietnamese-specific filters.

### Formatting

The Vietnamese Wikipedia is not properly formatted and there are lots of sections to be removed. We can fork [wikiplaintext](https://github.com/OpenLLM-France/wikiplaintext) to adapt it for Vietnamese, process the Vietnamese Wikipedia separately, save it locally and use it instead. This separation can help with licensing.

In the meantime, [VietGPT's processed Wikipedia](https://huggingface.co/vietgpt/wikipedia_vi) is a good resource for prototyping as it is significantly cleaner than other sources.

### Normalization

We normalize Unicode punctuation and fix text encoding using [ftfy](https://ftfy.readthedocs.io/en/latest/).

### Deduplication

We use [Rensa](https://github.com/beowolx/rensa), a high-performing text deduplication that implements three variants of the MinHash algorithm.

### Filtering

#### Quality filtering

Since the final training dataset is going to be small, around 2B tokens, we can filter data using even more precise quality signals. We can reuse the data curation method used for [FineWebEdu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), by training a fastText language classifier on either:
1. Wikipedia as the high-quality source and Binhvq news corpus as the lower-quality source.
2. Annotations generated by a Vietnamese LLM such as [SEA-LIONv3 9B](https://huggingface.co/aisingapore/Gemma-SEA-LION-v3-9B-IT). Inspired by [Tournesol](https://tournesol.app/) ([Hoang et al., 2021](https://arxiv.org/abs/2107.07334)), we use multiple criteria and scale the annotations according to adjectives such as recommendable, reliable, pedagogical, important, engaging. For consistency and due to limited resources, we use only one judge for the annotations.

The average quality indicators can be used for upsampling some datasets.

Visualization is a good way to ensure the data cleaning process preserves content diversity and overall document length distribution. [This article](https://developer.nvidia.com/blog/processing-high-quality-vietnamese-language-data-with-nvidia-nemo-curator/) provides some code samples used by [Viettel Solutions](https://huggingface.co/VTSNLP/Llama3-ViettelSolutions-8B).

Useful tools for visualization include:
- [text-clustering](https://github.com/huggingface/text-clustering)

### PII removal

We may use:
- [scrubadub](https://scrubadub.readthedocs.io/en/stable/), uses spaCy under the hood. Should work provided the spaCy Vietnamese model is good enough. Tests required.

### Data composition

Small scale experiments following the Sailor paper methodology may help find the right dataset sampling and the learning rate.

## Compromises

### Limiting filtering to reduce bias

Both methods described above are likely to introduce bias ([Dodge et al., 2021](https://arxiv.org/pdf/2104.08758)).

Possible mitigations:
- Use website whitelists and blacklists (keep quality content and filter adult / malicious websites)
- Use more lenient rules in data filtering

### Limiting anonymization to avoid false positives

Regex rules such as for phone numbers are likely to trigger false positives.

### Copyright

The underlying sources of Binhvq news corpus are not in the public domain.

## Further improvements

### Better unbiased data curation

In order to improve generalization capabilities, possibly use toxic identification instead of too much toxicity filtering ([Longpre et al., 2023](https://arxiv.org/abs/2305.13169)).

### Overcome temporal misalignment

The age difference between pretraining and finetuning datasets results in degraded performance ([Longpre et al., 2023](https://arxiv.org/abs/2305.13169)) due to misaligned language use and knowledge.

For our cases, the Binhvq news corpus (BKAI version), CulturaX and MADLAD-400 date back from 2023, FineWeb2-HQ was compiled in 2025. A good metric is to report the temporal distribution of the training dataset.

### Benchmark data decontamination

Removing test samples from the training data would make it possible to compute fair benchmark scores.

### Document-level code-switching

Using a Vietnamese-to-English dictionary for translation, code-switching as described in [Sailor](https://arxiv.org/abs/2404.03608) could help the model align its word representations between English and Vietnamese.

## Citations

TODO: grow the list to all of the pretraining datasets

- [The Lucie-7B LLM and the Lucie Training Dataset: Open resources for multilingual language generation.](https://arxiv.org/abs/2503.12294)
- [CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages](https://aclanthology.org/2024.lrec-main.377)
- [MADLAD-400 (Multilingual Audited Dataset: Low-resource And Document-level)](https://arxiv.org/abs/2309.04662)
- [Towards Comprehensive Vietnamese Retrieval-Augmented Generation and Large Language Models](https://arxiv.org/abs/2403.01616)
- [MTet: Multi-domain Translation for English and Vietnamese](https://arxiv.org/abs/2210.05610)
- [vbpl.vn - The National Database of Legal Documents, Ministry of Justice, Vietnam](https://vbpl.vn)
- [Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](https://arxiv.org/abs/2502.10361)
- [SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model](https://arxiv.org/abs/2502.02737v1)
- [MathPile: A Billion-Token-Scale Pretraining Corpus for Math](https://openreview.net/forum?id=RSvhU69sbG)
- [Sailor: Open Language Models for South-East Asia](https://arxiv.org/abs/2404.03608)
- [Penedo et al. (2024). DataTrove: large scale data processing.](https://github.com/huggingface/datatrove)
- [Robyn Speer. (2019). ftfy (Version 5.5). Zenodo.](http://doi.org/10.5281/zenodo.2591652)
- [vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords)
