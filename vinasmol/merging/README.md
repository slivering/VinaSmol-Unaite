# Model merging

We use [model merging](https://planetbanatt.net/articles/modelmerging.html) in order to combine the finetuning subtasks into a single model. This techniques mitigates catastrophic forgetting and improves multi-task learning after finetuning (multilinguality, safety, alignment). It is better than mere CPT with mixed language data [(Aakanksha et al.)](https://arxiv.org/abs/2410.10801).

Here is an example [multi-stage merging](https://github.com/arcee-ai/mergekit/blob/main/docs/multimerge.md) for Lucie-7B-Instruct inspired by the [SEA-LION](https://arxiv.org/pdf/2504.05747) methodology:

1. Lucie-7B-Instruct | *CPT language mix* -> **Lucie-VI-Base**
2. Lucie-VI-Base | *IFT EN/FR* -> **Lucie-Stage-1**
3. Lucie-Stage-1 | *IFT VI* -> **Lucie Stage 2**
4. Lucie-VI-Base + Lucie-Stage-1 + Lucie-Stage-2 | *DARE TIES merging* -> **Lucie-Merge-1**
5. Lucie-7B-Instruct + ... + *Lucie-Merge-1* | *Consensus TA merging* -> **Lucie-Merge-2**
6. Lucie-Merge-2 | *Preference Alignment (helpfulness, safety, internal merges...)* -> **Lucie-Aligned**
7. Lucie-7B-Instruct + Lucie-Aligned | *DELLA Linear merging* -> **VinaLucie-Instruct**

## References

- [Arcee's MergeKit: A Toolkit for Merging Large Language Models](https://aclanthology.org/2024.emnlp-industry.36)
