from pathlib import Path

from datasets import load_dataset
from litdata import index_parquet_dataset
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
import typer

from . import DATA_DIR
from .datasets import to_sharded_parquet
from vinasmol.tokenization import DATA_DIR as TOKENIZER_DATA_DIR

SEED = 20250827


def tokenize_to_parquet_splits(
        data_dir: Path,
        out_dir: Path,
        tokenizer: PreTrainedTokenizerFast,
    ):
    ds = load_dataset('parquet', split='train', data_dir=f"{data_dir}")
    ds = ds.map(
        lambda examples: tokenizer(examples['text'], return_tensors='np'),
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
    to_sharded_parquet(ds_train, out_dir / 'train')
    to_sharded_parquet(ds_val, out_dir / 'val')
    to_sharded_parquet(ds_test, out_dir / 'test')


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DATA_DIR / 'merged')

    data_dirs = [
        DATA_DIR / "deduped" / "en-all",
        #DATA_DIR / "deduped" / "vi-all",
        #DATA_DIR / "deduped" / "code-all",
    ]
    splits_dirs = [
        data_dir.parent / "splits" / data_dir.name
        for data_dir in data_dirs
    ]
    
    for data_dir, out_dir in zip(data_dirs, splits_dirs):
        tokenize_to_parquet_splits(data_dir, out_dir=out_dir, tokenizer=tokenizer)

    for splits_dir in splits_dirs:
        for split in splits_dir.iterdir():
            index_parquet_dataset(f"{split}")


if __name__ == '__main__':
    typer.run(main)