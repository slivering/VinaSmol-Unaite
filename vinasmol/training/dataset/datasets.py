from datasets import enable_progress_bars, load_dataset
from pathlib import Path

from ...hfmodel import BASE_MODEL, SMOLLM2, LUCIE
from .preprocessing import (
    _set_text_as_md_in_vbpl, _set_text_as_md_in_wiki, NormalizeCols,
)

DATA_DIR = (Path(__file__).parent / "data").resolve()
DATA_DIR_EN = DATA_DIR / "english"
DATA_DIR_VI = DATA_DIR / "vietnamese"

LOAD_KWARGS = dict(split="train", streaming=True)
SEED = 20250801

# FIXME: datasets shards are too large and target data is smaller than one shard
# TODO: possibly perform map after shuffle/take/Dataset.from_generator?

def download_english_datasets(data_dir: Path):
    if BASE_MODEL == SMOLLM2:
        # 28B tokens, 39M rows
        cosmopedia_v2 = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", **LOAD_KWARGS)
        (cosmopedia_v2
            .map(NormalizeCols.cosmopedia_v2)
            .shuffle(seed=SEED, buffer_size=1000)
            .take(1_000_000)
            .to_parquet(data_dir / "cosmopedia_v2.parquet")
        )

        # ~400 GB, 190M rows
        fineweb_edu_dedup = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", **LOAD_KWARGS)
        (fineweb_edu_dedup
            .map(NormalizeCols.fineweb_edu_dedup)
            .shuffle(seed=SEED, buffer_size=5_000)
            .take(1_000_000)
            .to_parquet(data_dir / "fineweb_edu.parquet")
        )

        # 4B tokens, 8M rows
        python_edu = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", **LOAD_KWARGS)
        (python_edu
            .map(NormalizeCols.python_edu)
            .shuffle(seed=SEED, buffer_size=5_000)
            .take(1_000_000)
            .to_parquet(data_dir / "python_edu.parquet")
        )
        
        # Datasets used in the annealing phase of Lucie

        # 27 GB, 6M rows
        openwebmath = load_dataset("open-web-math/open-web-math", **LOAD_KWARGS)
        (openwebmath
            .map(NormalizeCols.open_web_math)
            .shuffle(seed=SEED, buffer_size=1000)
            .take(100_000)
            .to_parquet(data_dir / "openwebmath.parquet")
        )

        # 42B tokens, 40M rows
        # Higher-quality and up-to-date pes2o dataset with proper formatting
        olmocr_pes2o = load_dataset("allenai/olmOCR-pes2o-0225", **LOAD_KWARGS)
        (olmocr_pes2o
            .map(NormalizeCols.olmocr_pes2o)
            .shuffle(seed=SEED, buffer_size=1000)
            .take(500_000)
            .to_parquet(data_dir / "pes2o.parquet")
        )

        # TODO: CLI download as per HuggingFace documentation
        # 10B tokens, 900M rows
        # Not shuffled, many sources in different shards
        #mathpile_commercial = load_dataset("GAIR/MathPile_Commercial", **LOAD_KWARGS)
        #(mathpile_commercial
        #    .map(NormalizeCols.mathpile_commercial)
        #    .shuffle(seed=SEED, buffer_size=10_000)
        #    .take(10_000_000)
        #    .to_parquet(data_dir / "mathpile_commercial.parquet")
        #)

        # ~200MB, 200k rows
        stackmathqa = load_dataset("math-ai/StackMathQA", "stackmathqa200k", **LOAD_KWARGS)
        (stackmathqa
            .map(NormalizeCols.stackmathqa)
            .shuffle(seed=SEED, buffer_size=1000)
            .take(500_000)
            .to_parquet(data_dir / "stackmathqa.parquet")
        )

        # ~400GB, ~300M rows
        #flanv2 = load_dataset("SirNeural/flan_v2", **LOAD_KWARGS)
        #(flanv2
        #    .map(NormalizeCols.flan_v2)
        #    .shuffle(seed=SEED, buffer_size=1000)
        #    .take(500_000)
        #    .to_parquet(data_dir / "flan_v2.parquet")
        #)

        # ~20GB, 7.03M rows
        wikipedia_en = load_dataset("omarkamali/wikipedia-monthly", "20250702.en", **LOAD_KWARGS)
        (wikipedia_en
            .map(_set_text_as_md_in_wiki, fn_kwargs=dict(lang='en'))
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
    wikipedia_vi = load_dataset("omarkamali/wikipedia-monthly", "20250702.vi", **LOAD_KWARGS)
    (wikipedia_vi
        .map(_set_text_as_md_in_wiki, fn_kwargs=dict(lang='vi'))
        .map(NormalizeCols.wikipedia_vi)
        .shuffle(seed=SEED, buffer_size=10_000)
        .to_parquet(data_dir / "wikipedia_vi.parquet")
    )

    # 59 GB, 4M rows
    fineweb2_hq = load_dataset("epfml/FineWeb2-HQ", "vie_Latn", **LOAD_KWARGS)
    (fineweb2_hq
        .map(NormalizeCols.fineweb2_hq)
        .shuffle(seed=SEED, buffer_size=1000)
        .take(500_000)
        .to_parquet(data_dir / "fineweb2_hq.parquet")
    )

    # 55 B tokens, 58M rows
    # Possibly add https://huggingface.co/datasets/ontocord/CulturaY (data from Internet Archive)
    culturax = load_dataset("uonlp/CulturaX", "vi", token=True, **LOAD_KWARGS)
    (culturax
        .map(NormalizeCols.culturax)
        .shuffle(seed=SEED, buffer_size=10_000)
        .take(500_000)
        .to_parquet(data_dir / "cultura_x.parquet")
    )

    # 49 B tokens, 93M rows
    madlad400 = load_dataset("allenai/madlad-400", "vi", split="clean", streaming=True, trust_remote_code=True)
    (madlad400
        .map(NormalizeCols.madlad400)
        .shuffle(seed=SEED, buffer_size=10_000)
        .take(1_000_000)
        .to_parquet(data_dir / "madlad_400.parquet")
    )

    # 29 GB, 17M rows
    # Apache license for binhvq but underlying copyright unclear
    # BKAINewsCorpus doesn't have a license, only suggests citation
    bkai_news = load_dataset("bkai-foundation-models/BKAINewsCorpus", **LOAD_KWARGS)
    (bkai_news
        .map(NormalizeCols.bkai_news)
        .shuffle(seed=SEED, buffer_size=10_000)
        .take(1_000_000)
        .to_parquet(data_dir / "bkai_news_corpus.parquet")
    )

    # 800 MB, 8M rows
    mtet = load_dataset("phongmt184172/mtet", **LOAD_KWARGS)
    (mtet
        .map(NormalizeCols.mtet)
        .shuffle(seed=SEED, buffer_size=10_000)
        .to_parquet(data_dir / "mtet.parquet")
    )

    # Muennighoff/flores200
    # ~ 100k words
    # Aligned professional translation data (useful if the base model is multilingual)


    # Official law texts (version: June 2025)
    # ~ 300 MB, 62k rows, mostly clean
    vbpl = load_dataset("doanhieung/vbpl", **LOAD_KWARGS)
    (vbpl
        .map(_set_text_as_md_in_vbpl)
        .map(NormalizeCols.vbpl)
        .shuffle(seed=SEED, buffer_size=10_000)
        .to_parquet(data_dir / "vbpl.parquet")
    )


if __name__ == "__main__":
    enable_progress_bars()
    DATA_DIR_VI.mkdir(parents=True, exist_ok=True)
    DATA_DIR_EN.mkdir(parents=True, exist_ok=True)
    download_vietnamese_datasets(DATA_DIR_VI)
    download_english_datasets(DATA_DIR_EN)