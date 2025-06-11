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
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.text import Languages

from rensa import RMinHash

from ...hfmodel import SEA_LION_BERT
from . import DATA_DIR, DATA_DIR_VI
from .constants import (
    STOP_WORDS,
    FLAGGED_WORDS_SAILCRAFT as FLAGGED_WORDS,
)
from .deduplication import RensaDeduplication
from .normalization import Formatter

VIETNAMESE_TOKENIZER = SEA_LION_BERT.tokenizer

FILTERING_OUTPUT_DIR = (DATA_DIR / "filtering").resolve()
MAIN_OUTPUT_DIR = DATA_DIR
DUMP_TO_PROCESS = "vi-all"
SEED = 20250801

class VietnameseFlaggedWordsFilter(C4BadWordsFilter):
    """Filter documents that contain too many flagged words.
    
    Contrary to the C4 one, this filter tolerates a proportion of flagged words.
    """
    def __init__(
        self,
        flagged_words_vi: list[str] = None,
        flagged_thr: float = 0.1,
        keep_fraction: float = 0.1,
        seed: int = None,
        exclusion_writer = None,
    ):
        """Initialize the filter.

        Args:
            flagged_words_vi (list[str], optional): Flagged words for Vietnamese.
                Defaults to the flagged word list used by Sailcraft.
            flagged_thr (float, optional): Maximum proportion of flagged words.
                The ratio is computed by counting the number of syllables.
            keep_fraction (float, optional): Proportion of filtered documents to keep.
        """
        super().__init__(
            keep_fraction=keep_fraction,
            seed=seed,
            exclusion_writer=exclusion_writer,
        )
        if flagged_words_vi is None:
            flagged_words_vi = FLAGGED_WORDS
        words = [re.escape(w.lower()) for w in flagged_words_vi]
        flagged_re = re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(words)))
        self._badwords_regex[Languages.vietnamese] = flagged_re

        self.flagged_thr = flagged_thr

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        lang: str = doc.metadata.get("language", self.default_language)
        if lang != Languages.vietnamese:
            return True
        
        badwords_regex = self._get_badwords(lang)
        if badwords_regex is None:
            self.stat_update("missing_badwords_lang", f"missing_badwords_lang_{lang}")
            return True
        
        num_flagged = len(badwords_regex.findall(doc.text.lower()))
        num_syllables = len(doc.text.split())
        ratio = num_flagged / num_syllables
        if ratio > self.flagged_thr:
            self.stat_update("documents_with_many_flagged_words_vi")
            if self.keep_fraction > 0.0 and self.uniform() < self.keep_fraction:
                self.stat_update("document_kept_with_many_flagged_words_vi")
                return True
            self.stat_update("document_removed_with_flagged_words_vi")
            return False, "document_removed_with_many_flagged_words_vi"
        return True
    


main_processing_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            DATA_DIR_VI,
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
        VietnameseFlaggedWordsFilter(
            keep_fraction=0.1,
            seed=SEED,
            language=Languages.vietnamese,
            exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_DIR}/removed/7_c4_badwords/{DUMP_TO_PROCESS}")
        ),

        # Other filters: FastTextClassifierFilter (probably biased), SamplerFilter

        JsonlWriter(f"{FILTERING_OUTPUT_DIR}/output/{DUMP_TO_PROCESS}"),
    ],
    tasks=64,
    workers=16,
    logging_dir=f"{MAIN_OUTPUT_DIR}/logs/base_processing/{DUMP_TO_PROCESS}",
)

INPUT_READER = JsonlReader(
    f"{FILTERING_OUTPUT_DIR}/output/{DUMP_TO_PROCESS}"
)

final_stage = LocalPipelineExecutor(
    pipeline=[
        INPUT_READER,
        RensaDeduplication(RMinHash, num_perm=128, seed=SEED),

        TokensCounter(VIETNAMESE_TOKENIZER),

        # TODO: (here) check final URL distribution

        # FIXME: performance/security issues?
        # Possibly use scrubadub for more in-depth cleaning (beware of performance)
        PIIFormatter(),
        JsonlWriter(f"{MAIN_OUTPUT_DIR}/{DUMP_TO_PROCESS}/deduped"),
    ],
    logging_dir=f"{MAIN_OUTPUT_DIR}/logs/deduplication/{DUMP_TO_PROCESS}",
    depends=main_processing_executor,
)


if __name__ == "__main__":
    main_processing_executor.run()
    final_stage.run()
    

