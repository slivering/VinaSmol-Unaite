from functools import partial
from pathlib import Path

from datasets import load_dataset
from litdata import optimize
import pyarrow.parquet as pq
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
import typer

from . import DATA_DIR
from vinasmol.tokenization import DATA_DIR as TOKENIZER_DATA_DIR


SEED = 20250827

def tokenize_fn(filepath: str, tokenizer: PreTrainedTokenizerFast):
    parquet_file = pq.ParquetFile(filepath)
    # Process per batch to reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=8192, columns=['text']):
        for text in batch.to_pandas()['text']:
            # SmolLM does not have BOS by default?
            yield tokenizer.encode(text, add_special_tokens=True)

def split_parquet(data_dir: Path, out_dir: Path):
    ds = load_dataset('parquet', data_dir=f"{data_dir}")
    train_split = ds.train_test_split(
        test_size=0.2,
        seed=SEED
    )
    ds_train = train_split['train']

    test_split = train_split['test'].train_test_split(test_size=0.5)
    ds_val = test_split['train']
    ds_test = test_split['test']
    ds_train.to_parquet(f"{out_dir / 'train.parquet'}")
    ds_val.to_parquet(f"{out_dir / 'val.parquet'}")
    ds_test.to_parquet(f"{out_dir / 'test.parquet'}")


def main():
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DATA_DIR / 'merged')

    data_dirs = [
        DATA_DIR / "en-all" / "deduped",
        DATA_DIR / "vi-all" / "deduped",
        DATA_DIR / "code-all" / "deduped",
    ]

    
    for data_dir in data_dirs:
        out_dir = data_dir.parent / "splits"
        split_parquet(data_dir, out_dir=out_dir)

    for data_dir in data_dirs:
        inputs = [str(file) for file in Path(data_dir).rglob("*splits/*.parquet")]

        outputs = optimize(
            fn=partial(tokenize_fn, tokenizer=tokenizer),
            inputs=inputs,
            output_dir=f"{data_dir.parent}/optimized",
            # sequence length of 2048 before context length extension
            chunk_size=(2049 * 127),
        )

        # TODO: restructure file hierarchy : corpus / split / file

if __name__ == '__main__':
    typer.run(main)