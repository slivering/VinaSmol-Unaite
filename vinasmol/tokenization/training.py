import numpy as np
from pathlib import Path
import random
from typing_extensions import Annotated

from datasets import Dataset, load_dataset
from loguru import logger
from tokenizers.implementations import SentencePieceBPETokenizer
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)
import typer

from vinasmol.hfmodel import BASE_MODEL # SMOLLM2
from vinasmol.vietnamese import LETTERS as VIETNAMESE_LETTERS

from . import DATA_DIR, number_of_added_tokens

# Default parameters
# FIXME: adjust this because the actual number of added tokens will be smaller
VIETNAMESE_VOCAB_SIZE = 10_000
BATCH_SIZE = 5_000
DROPOUT_RATE = 0.1
LIMIT_VIETNAMESE_ALPHABET = 500
SEED = 20250821


def load_vietnamese_corpus(data_dirs: list[Path]) -> list[Dataset]:
    """Load the data files for the Vietnamese tokenizer training corpus.

    Args:
        data_dirs (list[Path]): The directories that contains the dataset files.

    Returns:
        list[Dataset]: The list of loaded datasets.
    """
    datasets = [
        load_dataset(f"{dir}", split='train')
        for dir in data_dirs
    ]
    return datasets

def generate_training_corpus(datasets: list[Dataset], batch_size: int, seed):
    """Generate the examples for the tokenizer training corpus.

    Args:
        data_dir (Path): The directory that contains the dataset files.
        seed: The random seed for shuffling examples.

    Yields:
        samples (list[str]): a batch of text examples.
    """
    random.seed(seed)
    lengths = np.array([len(dataset) for dataset in datasets])
    total = lengths.sum()

    num_batches = total // batch_size
    step_sizes = batch_size * lengths // total
    for i in range(num_batches):
        samples = [
            text
            for dataset, step in zip(datasets, step_sizes)
            for text in dataset[i : i + step]['text']
        ]
        random.shuffle(samples)
        yield samples



# TODO: one sentence = one example?
def train_vietnamese_tokenizer(
        corpus,
        vocab_size: int = VIETNAMESE_VOCAB_SIZE,
        dropout: float = DROPOUT_RATE,
        limit_alphabet: int = LIMIT_VIETNAMESE_ALPHABET,
        min_frequency: int = 5,
    ) -> PreTrainedTokenizerBase:
    tokenizer = SentencePieceBPETokenizer(
        unk_token="<unk>",
        add_prefix_space=True,
        dropout=dropout, # Sailor paper: only during "finetuning"???
        fuse_unk=False,
    )
    tokenizer.train_from_iterator(
        corpus,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        # We don't need to add the special tokens since only the new tokens
        # will be added to SmolLM's tokenizer
        special_tokens=["<unk>"],
        initial_alphabet=VIETNAMESE_LETTERS,
        limit_alphabet=limit_alphabet,
        show_progress=True,
    )
    return PreTrainedTokenizerFast(tokenizer)

def merge_tokenizers(
        base_tokenizer: PreTrainedTokenizerBase,
        new_tokenizer: PreTrainedTokenizerBase,
    ) -> int:
    """Merge two tokenizer into a base tokenizer.

    Args:
        base_tokenizer (PreTrainedTokenizerBase): The pretrained tokenizer to extend.
        new_tokenizer (PreTrainedTokenizerBase): The tokenizer trained on the new language.

    Returns:
        int: The number of new tokens.
    """
    return base_tokenizer.add_tokens(new_tokenizer.get_vocab())


def initialize_new_embeddings(
        model: PreTrainedModel,
        updated_tokenizer: PreTrainedTokenizerBase,
    ):
    # Hewitt initialization by default: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
    model.resize_token_embeddings(len(updated_tokenizer), mean_resizing=True)



def main(
        dataset_dirs: Annotated[
            list[Path],
            typer.Argument(help="The directories that contains data files of the prepared Vietnamese corpora")
        ],
        tokenizer_out_dir: Annotated[
            Path,
            typer.Option(help="The directory to save the merged tokenizer and vocabulary")
        ] = DATA_DIR,
        vietnamese_vocab_size: Annotated[
            int,
            typer.Option(help="The vocabulary size of the new Vietnamese tokenizer")
        ] = VIETNAMESE_VOCAB_SIZE,
        batch_size: int = BATCH_SIZE,
        dropout_rate: float = DROPOUT_RATE,
        limit_vietnamese_alphabet: int = LIMIT_VIETNAMESE_ALPHABET,
        seed: int = SEED,
):
    """Extend the vocabulary of SmolLM using the prepared Vietnamese corpus."""
    datasets = load_vietnamese_corpus(dataset_dirs)
    lengths = [len(dataset) for dataset in datasets]
    logger.info("Loaded {} datasets with lengths {}", len(datasets), lengths)

    corpus = generate_training_corpus(datasets, batch_size=batch_size, seed=seed)

    logger.info("Starting tokenizer training")
    # TODO: save intermediate tokenizer states
    new_tokenizer = train_vietnamese_tokenizer(
        corpus,
        vocab_size=vietnamese_vocab_size,
        dropout=dropout_rate,
        limit_alphabet=limit_vietnamese_alphabet,
    )
    logger.info("New tokenizer vocabulary size: {}", len(new_tokenizer))
    new_tokenizer.save_pretrained(
        save_directory=tokenizer_out_dir,
        filename_prefix="vietnamese",
    )

    base_tokenizer = BASE_MODEL.load_tokenizer()
    logger.info("Base tokenizer vocabulary size: {}", len(base_tokenizer))

    # Merge into base_tokenizer
    n_added_tokens = merge_tokenizers(base_tokenizer, new_tokenizer)
    assert n_added_tokens == number_of_added_tokens(BASE_MODEL.load_tokenizer(), base_tokenizer)

    logger.info("Added {} new tokens", n_added_tokens)

    # FIXME: will save_pretrained save the newly added tokens???
    base_tokenizer.save_vocabulary(
        save_directory=tokenizer_out_dir,
        filename_prefix="merged",
    )

    logger.info("Saved tokenizer states in {}", tokenizer_out_dir)

if __name__ == '__main__':
    typer.run(main)