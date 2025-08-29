import os
from pathlib import Path
from typing import Annotated

from datasets import load_dataset
from litdata import optimize, TokensLoader
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
import typer

from . import DATA_DIR
from .datasets import to_sharded_parquet
from vinasmol.tokenization import DATA_DIR as TOKENIZER_DATA_DIR

SEED = 20250827

DATA_DIRS = [
    DATA_DIR / "deduped" / "en-all",
    #DATA_DIR / "deduped" / "vi-all",
    #DATA_DIR / "deduped" / "code-all",
]

MAIN_SPLITS_DIR = DATA_DIR / "deduped" / "splits"

def tokenize_examples(
        tokenizer: PreTrainedTokenizerFast,
        examples: dict,
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
    tokens_batch = tokenizer(examples['text'], add_special_tokens=False)['input_ids']

    if add_bos_eos:
        if tokenizer.bos_token_id != tokenizer.eos_token_id:
            raise ValueError(
                f"BOS == {tokenizer.bos_token_id} "
                f"but EOS == {tokenizer.eos_token_id}."
                " Expected BOS == EOS for SmolLM2's tokenizer."
            )
        # No need to add BOS for the first example since the last example
        # of the previous batch already ends with EOS.
        # litdata.optimize concatenates and chunks the whole token sequence
        for tokens in tokens_batch:
            # Only interleave one separator.
            tokens.append(tokenizer.eos_token_id)
    
    return {'input_ids': tokens_batch}

def tokenize_to_parquet_splits(
        data_dir: Path,
        out_dir: Path,
        tokenizer: PreTrainedTokenizerFast,
    ):
    ds = load_dataset('parquet', split='train', data_dir=f"{data_dir}")
    # TODO: take the full dataset
    ds = ds.take(10000).map(
        tokenize_examples,
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
    to_sharded_parquet(ds_train, out_dir / 'train', main_column='input_ids')
    to_sharded_parquet(ds_val, out_dir / 'val', main_column='input_ids')
    to_sharded_parquet(ds_test, out_dir / 'test', main_column='input_ids')


def main(
        data_dirs: Annotated[list[Path], typer.Argument()] = DATA_DIRS,
        out_dir: Path = DATA_DIR / "tokenized",
        main_splits_dir: Path = MAIN_SPLITS_DIR,
        block_size: int = 2048 + 1,
        chunk_bytes: str = "64MB",
):
    """Tokenize the training datasets and optimize them with litdata.
    
    For the block size, we recommend to use sequence lengths of 2048 + 1
    instead of 8192 + 1 in order to speed up pretraining by a factor of 10-15.

    If you are GPU rich and want to avoid doing context length extension again,
    you may consider setting `block_size = 8192 + 1`.
    """
    # Smaller sequence length -> larger batch size -> faster
    # TODO: enable custom data mixture with resampling

    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DATA_DIR / 'merged')

    # train, val, test
    splits_dirs = [
        main_splits_dir / data_dir.name
        for data_dir in data_dirs
    ]
    
    for data_dir, out_dir in zip(data_dirs, splits_dirs):
        tokenize_to_parquet_splits(data_dir, out_dir=out_dir, tokenizer=tokenizer)

    for splits_dir in splits_dirs:
        corpus_name = splits_dir.name

        for split in splits_dir.iterdir():
            split_out_dir = out_dir / corpus_name / split.name

            #index_parquet_dataset(f"{split}")
            ds = load_dataset('parquet', data_dir=f'{split}')
            optimize(
                fn=lambda x: x['input_ids'],
                inputs=ds,
                output_dir=f"{split_out_dir}",
                item_loader=TokensLoader(block_size=block_size),
                chunk_bytes=chunk_bytes,
                num_workers=os.cpucount(), #Useful if CUDA available?
            )


if __name__ == '__main__':
    typer.run(main)