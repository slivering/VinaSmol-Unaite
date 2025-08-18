import re

# TODO: use Slurm when scaling becomes necessary
from datatrove.executor.local import LocalPipelineExecutor

from datatrove.data import Document
from datatrove.pipeline.filters import (
    C4BadWordsFilter,
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    #FastTextClassifierFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.dedup import ESDatasetToSequence, ESMergeSequences, ESRangeRemover
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.text import Languages

from ...hfmodel import SEA_LION_BERT
from . import DATA_DIR, DATA_DIR_VI
from .constants import (
    STOP_WORDS,
    FLAGGED_WORDS_SAILCRAFT,
)
from .deduplication import RensaBuildIndex, RensaDeduplicate
from .normalization import Formatter

VIETNAMESE_TOKENIZER = SEA_LION_BERT.tokenizer

FILTERING_OUTPUT_DIR = (DATA_DIR / "filtering").resolve()
ES_DIR = DATA_DIR / "es"
MAIN_OUTPUT_DIR = DATA_DIR
DUMP_TO_PROCESS = "vi-all"
SEED = 20250801

class FlaggedWordsThresholdFilter(C4BadWordsFilter):
    """Filter documents that contain too many flagged words.
    
    Contrary to the C4 one, this filter tolerates a proportion of flagged words.
    """

    name = "⛰ Flagged Words Threshold"

    def __init__(
        self,
        language: str,
        language_flagged_words_override: list[str] = None,
        flagged_thr: float = 0.1,
        keep_fraction: float = 0.1,
        seed: int = None,
        exclusion_writer = None,
    ):
        """Initialize the filter.

        Args:
            language (str): the default language for the badwords filter.
            language_flagged_words_override (list[str], optional):
                Flagged words for the specified language.
            flagged_thr (float, optional): Maximum proportion of flagged words.
                The ratio is computed by counting the number of syllables.
            keep_fraction (float, optional): Proportion of filtered documents to keep.
        """
        super().__init__(
            keep_fraction=keep_fraction,
            seed=seed,
            default_language=Languages.vietnamese,
            exclusion_writer=exclusion_writer,
        )
        if language_flagged_words_override is not None:
            words = [re.escape(w.lower()) for w in language_flagged_words_override]
            flagged_re = re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))
            self._badwords_regex[language] = flagged_re

        self.flagged_thr = flagged_thr

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        lang: str = doc.metadata.get("language", self.default_language)
        
        badwords_regex = self._get_badwords(lang)
        if badwords_regex is None:
            self.stat_update("missing_badwords_lang", f"missing_badwords_lang_{lang}")
            return True
        
        num_flagged = len(badwords_regex.findall(doc.text.lower()))
        num_syllables = len(doc.text.split())
        ratio = num_flagged / num_syllables
        if ratio > self.flagged_thr:
            self.stat_update(f"documents_with_many_flagged_words_{lang}")
            if self.keep_fraction > 0.0 and self.uniform() < self.keep_fraction:
                self.stat_update(f"document_kept_with_many_flagged_words_{lang}")
                return True
            self.stat_update(f"document_removed_with_flagged_words_{lang}")
            return False, f"document_removed_with_many_flagged_words_{lang}"
        return True
    

output_intermediate_1 = f"{FILTERING_OUTPUT_DIR}/output_1/{DUMP_TO_PROCESS}"
output_intermediate_2 = f"{FILTERING_OUTPUT_DIR}/output_2/{DUMP_TO_PROCESS}"
output_intermediate_3 = f"{FILTERING_OUTPUT_DIR}/output_3/{DUMP_TO_PROCESS}"
output_intermediate_4 = f"{FILTERING_OUTPUT_DIR}/output_4/{DUMP_TO_PROCESS}"
es_dir_vi = f"{ES_DIR}/{DUMP_TO_PROCESS}"

main_processing_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            str(DATA_DIR_VI),
            default_metadata={"dump": DUMP_TO_PROCESS},
        ),
        # URL filtering for malicious, toxic and adult websites
        # Mainly for CulturaX, MADLAD-400 (FIXME: no source for the latter...)
        # The default list includes 361 / 4510795 .vn websites, which is fairly incomplete
        # CulturaX has already undergone such filtering. Possibly remove MADLAD-400
        # TODO: add domain and word additional lists
        # TODO: ignore absent URL fields
        URLFilter(
            extra_domains=None,
            extra_urls=None,
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_DIR}/removed/1_url/{DUMP_TO_PROCESS}"),
        ),
        Formatter(),
        LanguageFilter(
            Languages.vietnamese,
            language_threshold=0.65,
            backend="ft176",
            exclusion_writer=JsonlWriter(
                f"{FILTERING_OUTPUT_DIR}/2_non_english/",
                output_filename="${language}/" + DUMP_TO_PROCESS + "/${rank}.jsonl.gz",
            ),
        ),

        # FIXME: unfortunately there's a lot of redundant work due to repeated tokenization

        GopherRepetitionFilter(
            language=Languages.vietnamese,
            # TODO: audit removed samples due to ngram filters
            # Poems might be affected and coefficients might be different for Vietnamese
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_DIR}/removed/3_gopher_rep/{DUMP_TO_PROCESS}"),
        ),
        GopherQualityFilter(
            min_doc_words=100, # Tokens counted by spaCy's Vietnamese tokenizer 
            max_doc_words=300_000, # TODO: audit for bias against long, high-quality documents
            min_avg_word_length=2,
            max_avg_word_length=10, # High for mixed language document (e.g. parallel translation)
            max_bullet_lines_ratio=0.9,
            max_ellipsis_lines_ratio=0.3,
            min_stop_words=2,
            stop_words=STOP_WORDS, # Some words in this list have two Spacy tokens. This is ok.
            language=Languages.vietnamese,
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_DIR}/removed/4_gopher_qual/{DUMP_TO_PROCESS}"),
        ),
        C4QualityFilter(
            filter_no_terminal_punct=False,
            min_num_sentences=3,
            min_words_per_line=1,
            # FIXME: filter=True ?? would that hurt understanding of programming blogs?
            filter_javascript=False,
            filter_curly_bracket=False,
            language=Languages.vietnamese,
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_DIR}/removed/5_c4/{DUMP_TO_PROCESS}"),
        ),
        FineWebQualityFilter(
            line_punct_thr=0.1, # TODO: audit for bias
            line_punct_exclude_zero=False,
            short_line_thr=0.67,
            short_line_length=20, # TODO: audit for bias (e.g. poem exclusion)
            char_duplicates_ratio=0.1, # TODO: try other values
            new_line_ratio=0.3,
            language=Languages.vietnamese,
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_DIR}/removed/6_fineweb_qual/{DUMP_TO_PROCESS}")
        ),
        # TODO: audit for bias (e.g. medical content, Wikipedia...)
        FlaggedWordsThresholdFilter(
            language=Languages.vietnamese,
            language_flagged_words_override=FLAGGED_WORDS_SAILCRAFT,
            flagged_thr=0.1,
            keep_fraction=0.1,
            seed=SEED,
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_DIR}/removed/7_c4_badwords/{DUMP_TO_PROCESS}")
        ),

        # Other filters: FastTextClassifierFilter (probably biased), SamplerFilter

        JsonlWriter(output_intermediate_1),
    ],
    tasks=64,
    workers=16,
    logging_dir=f"{MAIN_OUTPUT_DIR}/logs/base_processing/{DUMP_TO_PROCESS}",
)


rensa_index = RensaBuildIndex(
    num_perm=128,
    seed=SEED,
    lsh_threshold=0.8,
    num_bands=16,
    final_jaccard_threshold=0.85,
)

document_dedup_stage = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(output_intermediate_1),
        rensa_index,
        RensaDeduplicate(
            rensa_index=rensa_index,
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_DIR}/removed/8_dedup/{DUMP_TO_PROCESS}")
        ),
        JsonlWriter(output_intermediate_2),
    ],
    logging_dir=f"{MAIN_OUTPUT_DIR}/logs/minhash/{DUMP_TO_PROCESS}",
    depends=main_processing_executor,
)

sentence_dedup_stage_1 = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(output_intermediate_2),
        ESDatasetToSequence(output_folder=es_dir_vi),
        JsonlWriter(output_intermediate_3),
    ],
    workers=4,
    tasks=4,
    logging_dir=f"{MAIN_OUTPUT_DIR}/logs/es/1/{DUMP_TO_PROCESS}",
    depends=main_processing_executor,
)

sentence_dedup_stage_2 = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(output_intermediate_3),
        ESMergeSequences(
            data_folder=es_dir_vi,
            tasks_stage_1=4,
        ),
        JsonlWriter(output_intermediate_4),
    ],
    logging_dir=f"{MAIN_OUTPUT_DIR}/logs/es/2/{DUMP_TO_PROCESS}",
    depends=main_processing_executor,
)

final_stage = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(output_intermediate_4),
        ESRangeRemover(
            min_doc_words=50,
            sequence_folder=es_dir_vi,
        ),
        TokensCounter(VIETNAMESE_TOKENIZER),

        # TODO: (here) check final URL distribution

        # FIXME: performance/security issues?
        # Possibly use scrubadub for more in-depth cleaning (beware of performance)
        PIIFormatter(),
        JsonlWriter(f"{MAIN_OUTPUT_DIR}/{DUMP_TO_PROCESS}/deduped"),
    ],
    logging_dir=f"{MAIN_OUTPUT_DIR}/logs/es/3/{DUMP_TO_PROCESS}",
    depends=main_processing_executor,
)


if __name__ == "__main__":
    main_processing_executor.run()
    final_stage.run()
    

