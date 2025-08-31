from functools import partial
from pathlib import Path

from datasets import IterableDataset, load_dataset, interleave_datasets
import numpy as np
import typer

from . import DATA_DIR, DATA_DIR_EN_ANNEALING
from .datasets import SEED, estimate_dataset_size, to_sharded_parquet
from .preprocessing import DatasetNames

ANNEALING_OUT_DIR = DATA_DIR / "deduped" / "annealing-all"

def subset_from_filtered_data(data_dir: Path, subset_name: str) -> IterableDataset:
    """Retrieve a specific subset from the filtered and deduped corpus.

    Args:
        data_dir (Path): The directory of the corpus.
        subset_name (str): The name of the subset, i.e. `document['metadata']['origin']`.

    Returns:
        IterableDataset: the processed subset.
    """
    corpus = load_dataset(path=data_dir, split='train')
    return corpus.filter(
        lambda row: row['metadata']['origin'] == subset_name,
        input_columns='metadata',
    )

def strip_metadata_to(row: dict, fields_to_keep: list[str]) -> dict:
    """Retain only specific metadata fields in order to normalize the row format.

    Args:
        row (dict): A dataset row.
        fields_to_keep (list[str], optional): The list of metadata fields to keep.
    """
    row['metadata'] = {
        key: row['metadata'].get(key)
        for key in fields_to_keep
    }
    return row

def main(
    out_dir: Path = ANNEALING_OUT_DIR,
):
    vi_dir = DATA_DIR / "deduped" / "vi-all"
    en_dir = DATA_DIR / "deduped" / "en-all"


    wikipedia_en = subset_from_filtered_data(en_dir, DatasetNames.wikipedia_en)
    gutenberg_en = subset_from_filtered_data(en_dir, DatasetNames.gutenberg_en)
    wikipedia_vi = subset_from_filtered_data(vi_dir, DatasetNames.wikipedia_vi)
    binhvq_news_corpus = subset_from_filtered_data(vi_dir, DatasetNames.binhvq_news_corpus)

    # We don't filter those datasets as they are already high-quality and rather clean
    finemath_4plus = load_dataset(
        DATA_DIR_EN_ANNEALING / "finemath_4plus",
        split='train',
        streaming=True,
    )
    stackmathqa = load_dataset(
        DATA_DIR_EN_ANNEALING / "stackmathqa",
        split='train',
        streaming=True,
    )

    # We assume that the datasets are (mostly) disjoint and already shuffled
    all_subsets: list[IterableDataset] = [
        subset.map(partial(strip_metadata_to, fields_to_keep=['origin', 'url']))
        for subset in [
            wikipedia_en,
            gutenberg_en,
            wikipedia_vi,
            binhvq_news_corpus,
            finemath_4plus,
            stackmathqa,
        ]
    ]
    proportions = [
        0.1,
        0.1,
        0.2,
        0.3,
        0.2,
        0.1,
    ]

    probabilities = np.zeros(len(proportions))
    for i in range(len(all_subsets)):
        probabilities[i] = proportions[i] / estimate_dataset_size(all_subsets[i])
    probabilities /= probabilities.sum()

    annealing_mix: IterableDataset = interleave_datasets(
        all_subsets,
        probabilities=probabilities,
        seed=SEED,
        stopping_strategy='all_exhausted',
    )
    # FIXME: this will convert to a list eagerly
    to_sharded_parquet(annealing_mix.take(2_000_000), out_dir)

    # TODO: combine all of these datasets with good proportions

if __name__ == '__main__':
    typer.run(main)