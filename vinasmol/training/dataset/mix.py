from functools import partial
import json
from pathlib import Path

from datasets import load_dataset, Dataset
from loguru import logger
import pyarrow.parquet as pq
from litdata import (
    index_parquet_dataset, merge_datasets, optimize,
    StreamingDataset, CombinedStreamingDataset,
    StreamingDataLoader, TokensLoader,
)
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2TokenizerFast, PreTrainedTokenizerFast
import typer

from . import DATA_DIR, DATA_DIR_CODE, DATA_DIR_EN, DATA_DIR_VI
from .preprocessing import DatasetNames
from vinasmol.tokenization import DATA_DIR as TOKENIZER_DATA_DIR


SEED = 20250827
CHAR_BUDGET = 6_000_000_000 # 6B chars ~ 2B tokens

TARGET_WEIGHTS_EN = {
    DatasetNames.fineweb_edu_dedup: 0.7,
    DatasetNames.gutenberg_en: 0.2,
    DatasetNames.wikipedia_en: 0.1,
}

TARGET_WEIGHTS_CODE = {
    DatasetNames.starcoder_python_edu: 1.0,
}

TARGET_WEIGHTS_EN_ANNEALING = {
    DatasetNames.cosmopedia_v2: 0.55,
    DatasetNames.olmocr_pes2o: 0.2,
    DatasetNames.flan_v2: 0.1,
    DatasetNames.stackmathqa: 0.05,
    DatasetNames.open_web_math: 0.05,
    DatasetNames.mathpile_commercial: 0.05,
}

TARGET_WEIGHTS_VI = {
    DatasetNames.wikipedia_vi: 0.1,
    DatasetNames.fineweb2_hq: 0.25,
    DatasetNames.culturax: 0.25,
    DatasetNames.madlad400: 0.15,
    DatasetNames.binhvq_news_corpus: 0.2,
    DatasetNames.mtet: 0.03,
    DatasetNames.vbpl: 0.02,
    #DatasetNames.ccvj: # Keep for annealing
}

def compute_weights(
        dataset_folders: list[Path],
        target_weights: dict[str, float],
        char_budget: int = CHAR_BUDGET,
    ) -> dict[Path, float]:
    weights = {}
    for folder in dataset_folders:
        ds = load_dataset(data_dir=f"{folder}", split='train', num_proc=4)
        total_chars = sum(len(row['text']) for row in ds)
        origin = ds['metadata'][0]['origin']
        current_weight = total_chars / char_budget
        target_weight = target_weights[origin] / sum(target_weights.values())
        weights[folder] = target_weight / current_weight
    return weights
    
def tokenize_fn(filepath: str, tokenizer: PreTrainedTokenizerFast):
    parquet_file = pq.ParquetFile(filepath)
    # Process per batch to reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=8192, columns=['text']):
        for text in batch.to_pandas()['text']:
            # SmolLM does not have BOS by default?
            yield tokenizer.encode(text, add_special_tokens=True)


def main(
        char_budget: int = CHAR_BUDGET,
):
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_DATA_DIR / 'merged')

    data_dirs = [
        DATA_DIR / "en-all" / "deduped",
        DATA_DIR / "vi-all" / "deduped",
        DATA_DIR / "code-all" / "deduped",
    ]

    for data_dir in data_dirs:
        inputs = [str(file) for file in Path(data_dir).rglob("*.parquet")]

        outputs = optimize(
            fn=partial(tokenize_fn, tokenizer=tokenizer),
            inputs=inputs,
            output_dir=f"{data_dir.parent}/optimized",
            # sequence length of 2048 before context length extension
            chunk_size=(2049 * 128),
        )

if __name__ == '__main__':
    typer.run(main)