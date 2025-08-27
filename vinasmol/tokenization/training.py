import copy
import json
from pathlib import Path
import random
from typing_extensions import Annotated

from datasets import Dataset, load_dataset
from loguru import logger
import numpy as np
from tokenizers.models import BPE
from tokenizers.implementations import BaseTokenizer, SentencePieceBPETokenizer
from transformers import GPT2TokenizerFast, PreTrainedTokenizerBase
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
        min_frequency: int = 1000,
    ) -> BaseTokenizer:
    tokenizer = SentencePieceBPETokenizer(
        # The unknown token from SmolLM2
        # TODO: don't hardcode this
        unk_token="<|endoftext|>",
        replacement="Ġ",
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
        special_tokens=["<|endoftext|>"],
        initial_alphabet=VIETNAMESE_LETTERS,
        limit_alphabet=limit_alphabet,
        show_progress=True,
    )
    return tokenizer


def merge_tokenizers(
        base_tokenizer: PreTrainedTokenizerBase,
        new_tokenizer: BaseTokenizer,
    ) -> int:
    """Merge two tokenizers into a base tokenizer.

    Args:
        base_tokenizer (PreTrainedTokenizerBase): The pretrained tokenizer to extend.
        new_tokenizer (BaseTokenizer): The tokenizer trained on the new language.

    Returns:
        int: The number of new tokens.
    """
    new_vocab = [
        token.replace('Ġ', ' ')
        for token in new_tokenizer.get_vocab()
        if not token.isnumeric()
    ]
    new_tokens = set(new_vocab) - set(base_tokenizer.get_vocab())
    return base_tokenizer.add_tokens(list(new_tokens))

# https://gucci-j.github.io/post/en/vocab-expansion/
# Haven't made this work yet
def merge_tokenizers_sp_into_gpt2(
        base_tokenizer: PreTrainedTokenizerBase,
        new_tokenizer: BaseTokenizer,
    ) -> int:
    """Merge two tokenizers into a new tokenizer.

    Args:
        base_tokenizer (PreTrainedTokenizerBase): The pretrained tokenizer to extend.
        new_tokenizer (BaseTokenizer): The tokenizer trained on the new language.

    Returns:
        int: The number of new tokens.
    """
    # Only add the new tokens
    # TODO: don't add numeric tokens with 2+ digits
    base_tokenizer_json = json.loads(base_tokenizer._tokenizer.to_str())
    base_vocab = base_tokenizer_json['model']['vocab']
    base_merges = base_tokenizer_json['model']['merges']

    new_tokenizer_json = json.loads(new_tokenizer._tokenizer.to_str())
    new_vocab = new_tokenizer_json['model']['vocab']
    new_merges = new_tokenizer_json['model']['merges']

    num_new_tokens = 0
    ret_vocab = copy.copy(base_vocab)
    ret_merges = []
    old_merges = copy.copy(base_merges)

    for token in new_vocab:
        if token not in ret_vocab:
            ret_vocab[token] = len(ret_vocab)
            num_new_tokens += 1

    for merge in new_merges:
        # Add vocab
        [token_1, token_2] = merge
        token = token_1 + token_2

        # Add merges
        if merge in old_merges:
            old_merges.remove(merge)
            ret_merges.append(merge)
        elif token in ret_vocab and token_1 in ret_vocab and token_2 in ret_vocab:
            ret_merges.append(merge)
        else:
            raise RuntimeError
    
    merges = ret_merges + old_merges
    vocab = ret_vocab
    new_tokenizer.model = BPE(
        vocab=vocab,
        merges=[(merge[0], merge[1]) for merge in merges],
        fuse_unk=False,
        #byte_fallback=True
    )

    return num_new_tokens

    

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
    logger.info("New tokenizer vocabulary size: {}", new_tokenizer.get_vocab_size())
    new_tokenizer_path = tokenizer_out_dir / "new_vietnamese_tokenizer.json"
    tokenizer_out_dir.mkdir(parents=True, exist_ok=True)
    new_tokenizer.save(f"{new_tokenizer_path}")

    base_tokenizer = BASE_MODEL.load_tokenizer()
    logger.info("Base tokenizer vocabulary size: {}", len(base_tokenizer))

    # Merge into base_tokenizer
    n_added_tokens = merge_tokenizers(base_tokenizer, new_tokenizer)
    effective_n_added_tokens = number_of_added_tokens(BASE_MODEL.load_tokenizer(), base_tokenizer)
    assert n_added_tokens == effective_n_added_tokens

    logger.info("Added {} new tokens", n_added_tokens)

    logger.debug("Test:")

    s = "Việt Nam, quốc hiệu đầy đủ là Cộng hòa xã hội chủ nghĩa Việt Nam, là một quốc gia nằm ở cực Đông của bán đảo Đông Dương thuộc khu vực Đông Nam Á, giáp với Lào, Campuchia, Trung Quốc, biển Đông và vịnh Thái Lan"
    logger.debug(s)
    logger.debug(
        "{}",
        [base_tokenizer.decode(token) for token in base_tokenizer(s)['input_ids']]
    )

    # TODO: make the added tokens like part of the vocabulary
    # Performance impact on tokenizer loaded with GPT2TokenizerFast.from_pretrained??
    base_tokenizer.save_pretrained(f"{tokenizer_out_dir / 'merged'}")

    logger.info("Saved tokenizer states in {}", tokenizer_out_dir)

if __name__ == '__main__':
    typer.run(main)
