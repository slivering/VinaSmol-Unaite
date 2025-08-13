from datasets import enable_progress_bars, load_dataset
from pathlib import Path

from ...hfmodel import BASE_MODEL, SMOLLM2, LUCIE
from .preprocessing import (
    format_vbpl_md, convert_mediawiki_to_md, NormalizeCols,
)

DATA_DIR = (Path(__file__).parent / "data").resolve()
DATA_DIR_EN = DATA_DIR / "english"
DATA_DIR_VI = DATA_DIR / "vietnamese"

LOAD_KWARGS = dict(split="train", streaming=True)
# TODO: download the sampled subset as a Dataset and then map in parallel
NUM_PROC = 16
SEED = 20250801

MIN_PYTHON_EDU_SCORE = 3

# FIXME: datasets shards are too large and target data is smaller than one shard
# TODO: possibly perform map after shuffle/take/Dataset.from_generator?

def download_english_datasets(data_dir: Path):
    if BASE_MODEL == SMOLLM2:
        # 28B tokens, 39M rows
        cosmopedia_v2 = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", **LOAD_KWARGS)
        (cosmopedia_v2
            .map(NormalizeCols.cosmopedia_v2, remove_columns=cosmopedia_v2.column_names)
            .shuffle(seed=SEED, buffer_size=1000)
            .take(1_000_000)
            .to_parquet(data_dir / "cosmopedia_v2.parquet")
        )

        # ~400 GB, 190M rows
        fineweb_edu_dedup = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", **LOAD_KWARGS)
        (fineweb_edu_dedup
            .map(NormalizeCols.fineweb_edu_dedup, remove_columns=fineweb_edu_dedup.column_names)
            .shuffle(seed=SEED, buffer_size=5_000)
            .take(1_000_000)
            .to_parquet(data_dir / "fineweb_edu.parquet")
        )

        # 5 GB, 3M rows
        # Alternative: "meryyllebr543/stack-edu-huggingface", "python" (25M rows), one shard
        starcoder_python_edu = load_dataset(
            "JanSchTech/starcoderdata-python-edu-lang-score",
            **LOAD_KWARGS,
            columns=[
                'max_stars_repo_path',
                'max_stars_repo_name',
                'id',
                'language', # TODO: use language=='vi' for finetuning (around 100 examples)
                'content_cleaned',
                'edu_score',
            ]
        )
        (starcoder_python_edu
            .filter(lambda row: round(row['edu_score']) >= MIN_PYTHON_EDU_SCORE)
            .map(NormalizeCols.starcoder_python_edu, remove_columns=starcoder_python_edu.column_names)
            .shuffle(seed=SEED, buffer_size=5_000)
            .take(500_000)
            .to_parquet(data_dir / "starcoder_python_edu.parquet")
        )

        #lucie_python = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "code-python", revision="v1.2", split='train', streaming=True)
        
        # Datasets used in the annealing phase of Lucie

        # 27 GB, 6M rows
        openwebmath = load_dataset("open-web-math/open-web-math", **LOAD_KWARGS)
        (openwebmath
            .map(NormalizeCols.open_web_math, remove_columns=openwebmath.column_names)
            .shuffle(seed=SEED, buffer_size=1000)
            .take(100_000)
            .to_parquet(data_dir / "openwebmath.parquet")
        )

        # 42B tokens, 40M rows
        # Higher-quality and up-to-date pes2o dataset with proper formatting
        olmocr_pes2o = load_dataset("allenai/olmOCR-pes2o-0225", **LOAD_KWARGS)
        (olmocr_pes2o
            .map(NormalizeCols.olmocr_pes2o, remove_columns=olmocr_pes2o.column_names)
            .shuffle(seed=SEED, buffer_size=1000)
            .take(500_000)
            .to_parquet(data_dir / "pes2o.parquet")
        )

        # TODO: CLI download as per HuggingFace documentation
        # 10B tokens, 900M rows
        # Not shuffled, many sources in different shards
        #mathpile_commercial = load_dataset("GAIR/MathPile_Commercial", token=True, **LOAD_KWARGS)
        #(mathpile_commercial
        #    .map(NormalizeCols.mathpile_commercial, remove_columns=mathpile_commercial.column_names)
        #    .shuffle(seed=SEED, buffer_size=10_000)
        #    .take(10_000_000)
        #    .to_parquet(data_dir / "mathpile_commercial.parquet")
        #)

        # ~200MB, 200k rows
        stackmathqa = load_dataset("math-ai/StackMathQA", "stackmathqa200k", split="train")
        (stackmathqa
            .map(NormalizeCols.stackmathqa, remove_columns=stackmathqa.column_names)
            .shuffle(seed=SEED)
            .to_parquet(data_dir / "stackmathqa.parquet")
        )

        # ~400GB, ~300M rows
        # Temporarily excluded because of the large download
        #flan_v2 = load_dataset("SirNeural/flan_v2", **LOAD_KWARGS)
        #(flan_v2
        #    .map(NormalizeCols.flan_v2, remove_columns=flan_v2.column_names)
        #    .shuffle(seed=SEED, buffer_size=1000)
        #    .take(500_000)
        #    .to_parquet(data_dir / "flan_v2.parquet")
        #)

        # ~20GB, 7.03M rows
        wikipedia_en = load_dataset("omarkamali/wikipedia-monthly", "20250702.en", **LOAD_KWARGS)
        (wikipedia_en
            .map(convert_mediawiki_to_md, fn_kwargs=dict(lang='en'))
            .map(NormalizeCols.wikipedia_en, remove_columns=wikipedia_en.column_names)
            .shuffle(seed=SEED, buffer_size=5_000)
            .take(1_000_000)
            .to_parquet(data_dir / "wikipedia_en.parquet")
        )

        # Datasets used in the pretraining phase of Lucie (mostly orthogonal with SmolLM-Corpus)

        # 1B tokens, 1M rows, a bit low quality...
        #claire_en = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "Claire-en", revision="v1.2", **LOAD_KWARGS)
        
        # 5.5B tokens, 56k rows
        # Needs public domain filtering
        #gutenberg_en = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "Gutenberg-en", revision="v1.2", **LOAD_KWARGS)
        
        # 70M tokens, 10k rows
        #europarl_en = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "Europarl-en", revision="v1.2", **LOAD_KWARGS)
        
        # Avoid training a very small model on too many programming languages
        #lucie_training_code = load_dataset("OpenLLM-France/Lucie-Training-Dataset", "code", revision="v1.2", **load_kwargs)
    
    elif BASE_MODEL == LUCIE:
        #config_names = list(load_dataset_builder("OpenLLM-France/Lucie-Training-Dataset").builder_configs)
        # Downsample RedPajamaV2
        #lucie_training = load_dataset("OpenLLM-France/Lucie-Training-Dataset", revision="v1.2", **LOAD_KWARGS)
        raise NotImplementedError(BASE_MODEL)
    else:
        raise NotImplementedError(BASE_MODEL)



def download_vietnamese_datasets(data_dir: Path):
    # ~ 1 GB, 1.3M rows
    wikipedia_vi = load_dataset("omarkamali/wikipedia-monthly", "20250702.vi", split="train")
    (wikipedia_vi
        .map(convert_mediawiki_to_md, fn_kwargs=dict(lang='vi'))
        .map(NormalizeCols.wikipedia_vi, remove_columns=wikipedia_vi.column_names, num_proc=NUM_PROC)
        .shuffle(seed=SEED)
        .to_parquet(data_dir / "wikipedia_vi.parquet")
    )

    # 59 GB, 4M rows
    fineweb2_hq = load_dataset("epfml/FineWeb2-HQ", "vie_Latn", **LOAD_KWARGS)
    (fineweb2_hq
        .map(NormalizeCols.fineweb2_hq, remove_columns=fineweb2_hq.column_names)
        .shuffle(seed=SEED, buffer_size=1000)
        .take(200_000)
        .to_parquet(data_dir / "fineweb2_hq.parquet")
    )

    # 55 B tokens, 58M rows
    # Possibly add https://huggingface.co/datasets/ontocord/CulturaY (data from Internet Archive)
    culturax = load_dataset("uonlp/CulturaX", "vi", token=True, **LOAD_KWARGS)
    (culturax
        .map(NormalizeCols.culturax, remove_columns=culturax.column_names)
        .shuffle(seed=SEED, buffer_size=10_000)
        .take(1_000_000)
        .to_parquet(data_dir / "cultura_x.parquet")
    )

    # 49 B tokens, 93M rows
    madlad400 = load_dataset("Symato/madlad-400_vi", split="train", streaming=True)
    (madlad400
        .map(NormalizeCols.madlad400, remove_columns=madlad400.column_names)
        .shuffle(seed=SEED, buffer_size=10_000)
        .take(1_000_000)
        .to_parquet(data_dir / "madlad_400.parquet")
    )

    # 29 GB, 17M rows
    # Apache license for binhvq but underlying copyright unclear
    # BKAINewsCorpus doesn't have a license, only suggests citation
    bkai_news = load_dataset("bkai-foundation-models/BKAINewsCorpus", **LOAD_KWARGS)
    (bkai_news
        .map(NormalizeCols.bkai_news, remove_columns=bkai_news.column_names)
        .shuffle(seed=SEED, buffer_size=10_000)
        .take(1_000_000)
        .to_parquet(data_dir / "bkai_news_corpus.parquet")
    )

    # 800 MB, 8M rows
    mtet = load_dataset("phongmt184172/mtet", split="train")
    (mtet
        .map(NormalizeCols.mtet, remove_columns=mtet.column_names, num_proc=NUM_PROC)
        .shuffle(seed=SEED)
        .to_parquet(data_dir / "mtet.parquet")
    )

    # Muennighoff/flores200
    # ~ 100k words
    # Aligned professional translation data (useful if the base model is multilingual)


    # Official law texts (version: June 2025)
    # ~ 300 MB, 62k rows, mostly clean
    # Don't use streaming mode in order to avoid Arrow column type inference errors
    vbpl = load_dataset("doanhieung/vbpl", split="train")
    (vbpl
        .map(format_vbpl_md, num_proc=NUM_PROC)
        .map(NormalizeCols.vbpl, remove_columns=vbpl.column_names)
        .shuffle(seed=SEED)
        .to_parquet(data_dir / "vbpl.parquet")
    )


if __name__ == "__main__":
    enable_progress_bars()
    DATA_DIR_VI.mkdir(parents=True, exist_ok=True)
    DATA_DIR_EN.mkdir(parents=True, exist_ok=True)
    #download_vietnamese_datasets(DATA_DIR_VI)
    download_english_datasets(DATA_DIR_EN)