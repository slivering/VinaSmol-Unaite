from functools import partial
import os
from pathlib import Path
from typing import Annotated

from datasets import load_dataset
from litdata import optimize, TokensLoader
from loguru import logger
import pyarrow.parquet as pq
import torch
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
import typer

from . import DATA_DIR
from .datasets import to_sharded_parquet
from vinasmol.tokenization import DATA_DIR as TOKENIZER_DATA_DIR

SEED = 20250827

DATA_DIRS = [
    DATA_DIR / "deduped" / "en-all",
    DATA_DIR / "deduped" / "vi-all",
    DATA_DIR / "deduped" / "code-all",
]

MAIN_SPLITS_DIR = DATA_DIR / "deduped" / "splits"

def tokenize_examples(
        examples: dict,
        tokenizer: PreTrainedTokenizerFast,
        add_bos_eos: bool = True,
    ) -> list[list[int]]:
    """Tokenize a batch of examples.

    Args:
        tokenizer (PreTrainedTokenizerFast): The VinaSmol merged tokenizer.
        examples (dict): A batch of examples from the dataset.
        add_bos_eos (bool, optional): Add BOS and EOS tokens. Defaults to True.

    Returns:
        list[list[int]]: The batch token ids.
    
    Adding BOS and EOS tokens helps avoid spurious continuations between unrelated examples.
    It is necessary since SmolLM2 was trained with them.
    """
    # No need to return as np/pt since we append and concatenate the whole sequence
    # TODO: suppress warnings about sequence length by splitting and recombining long examples
    tokens_batch = tokenizer(
        examples['text'],
        add_special_tokens=False,
        return_attention_mask=False,
    )['input_ids']

    if add_bos_eos:
        assert tokenizer.bos_token_id is not None
        assert tokenizer.eos_token_id is not None

        for tokens in tokens_batch:
            assert tokens[0] != tokenizer.bos_token_id
            assert tokens[-1] != tokenizer.eos_token_id
            tokens.insert(0, tokenizer.bos_token_id)
            tokens.append(tokenizer.eos_token_id)
    
    return {'input_ids': tokens_batch}

def tokenize_to_parquet_splits(
        data_dir: Path,
        out_dir: Path,
        tokenizer: PreTrainedTokenizerFast,
    ):
    ds = load_dataset('parquet', split='train', data_dir=f"{data_dir}")
    logger.info("Starting tokenization")
    ds = ds.map(
        partial(tokenize_examples, tokenizer=tokenizer),
        batched=True,
        num_proc=16,
        remove_columns=ds.column_names,
    )
    train_split = ds.train_test_split(
        test_size=0.2,
        seed=SEED,
    )
    ds_train = train_split['train']

    test_split = train_split['test'].train_test_split(test_size=0.5)
    ds_val = test_split['train']
    ds_test = test_split['test']

    logger.info("Saving splits to sharded Parquet files")
    to_sharded_parquet(ds_train, out_dir / 'train', main_column='input_ids')
    to_sharded_parquet(ds_val, out_dir / 'val', main_column='input_ids')
    to_sharded_parquet(ds_test, out_dir / 'test', main_column='input_ids')


def process(filepath):
    parquet_file = pq.ParquetFile(filepath)
    # reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=8192, columns=['input_ids']):
        for tokens in batch.to_pandas()['input_ids']:
            yield torch.tensor(tokens, dtype=torch.int)
    parquet_file.close()

def main(
        data_dirs: Annotated[list[Path], typer.Option()] = [],
        out_dir: Path = DATA_DIR / "tokenized",
        main_splits_dir: Path = MAIN_SPLITS_DIR,
        block_size: int = 2048 + 1,
        chunk_bytes: str = "64MB",
):
    """Tokenize the training datasets, split them into train/val/test subsets
    and optimize them with litdata.
    
    For the block size, we recommend to use sequence lengths of 2048 + 1
    instead of 8192 + 1 in order to speed up pretraining by a factor of 10-15.

    If you are GPU rich and want to avoid doing context length extension again,
    you may consider setting `block_size = 8192 + 1`.
    """
    # TODO: enable custom data mixture with resampling

    if not data_dirs:
        data_dirs = DATA_DIRS

    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DATA_DIR / 'merged')

    # train, val, test
    splits_dirs = [
        main_splits_dir / data_dir.name
        for data_dir in data_dirs
    ]
    
    for data_dir, tokenized_dir in zip(data_dirs, splits_dirs):
        logger.info("Tokenizing Parquet dataset at {}", data_dir)
        tokenize_to_parquet_splits(data_dir, out_dir=tokenized_dir, tokenizer=tokenizer)

    for splits_dir in splits_dirs:
        corpus_name = splits_dir.name

        for split in list(splits_dir.iterdir()):
            split_out_dir = out_dir / corpus_name / split.name
            logger.info("Optimizing Parquet split at {} to {}", split, split_out_dir)

            #index_parquet_dataset(f"{split}")
            inputs = [str(p.absolute()) for p in split.glob("*.parquet")]
            optimize(
                fn=process,
                inputs=inputs,
                output_dir=f"{split_out_dir}",
                item_loader=TokensLoader(block_size=block_size),
                chunk_bytes=chunk_bytes,
                num_workers=(os.cpu_count() - 2),
            )


if __name__ == '__main__':
    typer.run(main)
