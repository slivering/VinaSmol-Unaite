import re

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
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
from datatrove.pipeline.stats import (
    DocStats, LineStats, ParagraphStats,
    WordStats, SentenceStats, TokenStats, 
    #CCNetPerplexityStats,
    TopKConfig,
)
from datatrove.pipeline.dedup import ESDatasetToSequence, ESMergeSequences, ESRangeRemover
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.readers.parquet import ParquetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.text import Languages

from tldextract import TLDExtract

from vinasmol.training.dataset.preprocessing import DatasetNames

from ...hfmodel import VINALLAMA_7B
from . import DATA_DIR, DATA_DIR_VI
from .constants import (
    STOP_WORDS,
    FLAGGED_WORDS_SAILCRAFT,
)
from .deduplication import ESComputeRangesExternal, RensaBuildIndex, RensaDeduplicate
from .normalization import Formatter

VIETNAMESE_TOKENIZER = VINALLAMA_7B.tokenizer

top_k_config = TopKConfig(top_k_groups=["fqdn"], top_k=10_000)

MAIN_OUTPUT_DIR = DATA_DIR
FILTERING_OUTPUT_DIR = (MAIN_OUTPUT_DIR / "filtering").resolve()
FILTERING_REMOVED_DIR = (FILTERING_OUTPUT_DIR / "removed").resolve()
ES_DIR = MAIN_OUTPUT_DIR / "es"
LOGGING_DIR = MAIN_OUTPUT_DIR / "logs"
STATS_DIR = MAIN_OUTPUT_DIR / "stats"

DUMP_TO_PROCESS = "vi-all"
SEED = 20250801

DOMAIN_WHITELIST = [
    "wikipedia.org",
    "wikihow.com",
]
DOMAIN_WHITELIST += DatasetNames.PLACEHOLDER_URLS


class DomainWhitelistMixin(BaseFilter):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.name = f"{cls.name} (with 📋 whitelist)"

    def __init__(
            self,
            *args,
            domain_whitelist: list[str] = None,
            allow_no_url: bool = False,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.domain_whitelist = domain_whitelist or []
        # TODO: allow domain suffix wildcard
        self.allow_no_url = allow_no_url

        if not hasattr(self, 'tldextractor'):
            self.tldextractor = TLDExtract()
    
    def filter(self, document: Document):
        url = document.metadata.get("url")
        if not url:
            if self.allow_no_url:
                # Accept sources which have no URL by design
                return True
            raise ValueError("Missing document 'url' field")

        url_info = self.tldextractor(url)
        if url_info.top_domain_under_public_suffix in self.domain_whitelist:
            return True

        return super().filter(document)

class URLFilterWithWhitelist(DomainWhitelistMixin, URLFilter):
    """Perform filtering on URLs, whith a domain whitelist."""


class LanguageFilterWithWhitelist(DomainWhitelistMixin, LanguageFilter):
    """Perform filtering on URLs, whith a domain whitelist."""



DEFAULT_SCORE = 3

class FlaggedWordsThresholdFilter(C4BadWordsFilter):
    """Filter documents that contain too many flagged words.
    
    Contrary to the C4 one, this filter tolerates a proportion of flagged words.
    """

    name = "🚩 Flagged Words Threshold"

    def __init__(
        self,
        language: str,
        language_flagged_words_override: list[str] | dict[str, int] = None,
        flagged_thr: float = 0.1,
        keep_fraction: float = 0.1,
        seed: int = None,
        exclusion_writer = None,
    ):
        """Initialize the filter.

        Args:
            language (str): the default language for the badwords filter.
            language_flagged_words_override (list[str] | dict [str, int], optional):
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
        flagged = language_flagged_words_override
        if flagged is not None:
            if isinstance(flagged, list):
                scores = {w: DEFAULT_SCORE for w in flagged}
            elif isinstance(flagged, dict):
                scores = flagged
            else:
                raise TypeError("Wrong type for scores:", type(scores))
            escaped_words = [re.escape(w.lower()) for w in flagged]
            # Must span over complete syllables
            flagged_re = re.compile(r"(?:\W|^)({})(?:\W|$)".format("|".join(escaped_words)))
            self._badwords_regex[language] = flagged_re
            self._flagged_words_scores = scores

        self.flagged_thr = flagged_thr

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        lang: str = doc.metadata.get("language", self.default_language)
        
        badwords_regex = self._get_badwords(lang)
        if badwords_regex is None:
            self.stat_update("missing_badwords_lang", f"missing_badwords_lang_{lang}")
            return True
        
        flagged = badwords_regex.findall(doc.text.lower())
        scores = [
            self._flagged_words_scores.get(word) or DEFAULT_SCORE
            for word in flagged
        ]
        # Exponential weight scale in (0, 1]
        weights = [2 ** (x - 5) for x in scores]
        total_flagged_weight = sum(
            weight * word.count(' ')
            for weight, word in zip(weights, flagged)
        )
        num_syllables = len(doc.text.split())
        ratio = total_flagged_weight / num_syllables
        if ratio > self.flagged_thr:
            self.stat_update(f"over_flagged_words_threshold_{lang}")
            if self.keep_fraction > 0.0 and self.uniform() < self.keep_fraction:
                return True
            return False, f"too_many_flagged_words_{lang}"
        return True
    

output_intermediate_1 = f"{FILTERING_OUTPUT_DIR}/output_1/{DUMP_TO_PROCESS}"
output_intermediate_2 = f"{FILTERING_OUTPUT_DIR}/output_2/{DUMP_TO_PROCESS}"
es_dir_vi = f"{ES_DIR}/{DUMP_TO_PROCESS}"

# TODO: make English pipeline

main_processing_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            str(DATA_DIR_VI),
            recursive=True,
            limit=10_000, # TODO: remove, for debugging
            default_metadata={"dump": DUMP_TO_PROCESS},
        ),
        # URL filtering for malicious, toxic and adult websites
        # Mainly for CulturaX, MADLAD-400 (FIXME: no source for the latter...)
        # The default list includes 361 .vn / 4510795 banned websites, which is fairly incomplete
        # CulturaX has already undergone such filtering. Possibly remove MADLAD-400
        # TODO: add domain and word additional lists
        # TODO: ignore absent URL fields
        URLFilterWithWhitelist(
            # TODO: add other domains if useful
            extra_domains=None,
            extra_urls=None,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/1_url/{DUMP_TO_PROCESS}"),
            domain_whitelist=DOMAIN_WHITELIST,
            allow_no_url=True,
        ),
        Formatter(
            strip_whitespace=False, # TODO: only False for code and code excerpts
            normalize_punctuation=False,
        ),
        LanguageFilterWithWhitelist(
            # It's okay to have English data in the corpus if it's about Vietnam.
            # Since it's a minority, it won't affect much of the proportions.
            # However, for consistency with the next filters, we restrict to only one language
            # as English.
            # TODO: exclude mtet and add it in the end, split VJOL depending on language
            ['vie_Latn'], # GlotLID -> vie_Latn, FT176LID -> vi
            language_threshold=0.65,
            backend="glotlid", # FIXME: is the LID duplicated across workers?
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/2_non_vietnamese"),
            domain_whitelist=[DatasetNames.mtet.placeholder_url],
            allow_no_url=True,
        ),

        # FIXME: unfortunately there's a lot of redundant work due to repeated tokenization
        # TODO: cache token spans

        GopherRepetitionFilter(
            # default parameters
            dup_line_frac=0.3,
            dup_para_frac=0.3,
            dup_line_char_frac=0.2,
            dup_para_char_frac=0.2,
            top_n_grams=((2, 0.2), (3, 0.18), (4, 0.16)),
            dup_n_grams=((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.1)),
            language=Languages.vietnamese,
            # TODO: audit removed samples due to ngram filters
            # Poems might be affected and coefficients might be different for Vietnamese
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/3_gopher_rep/{DUMP_TO_PROCESS}"),
        ),
        GopherQualityFilter(
            min_doc_words=50, # Tokens counted by spaCy's Vietnamese tokenizer 
            max_doc_words=200_000, # Around 50 PDF pages
            min_avg_word_length=2,
            max_avg_word_length=10, # High for mixed language document (e.g. parallel translation)
            max_bullet_lines_ratio=0.9,
            max_ellipsis_lines_ratio=0.3,
            min_stop_words=2,
            stop_words=STOP_WORDS, # Some words in this list have two Spacy tokens. This is ok.
            language=Languages.vietnamese,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/4_gopher_qual/{DUMP_TO_PROCESS}"),
        ),
        C4QualityFilter(
            filter_no_terminal_punct=False,
            min_num_sentences=3,
            #min_words_per_line=1,
            # FIXME: filter=True ?? would that hurt understanding of programming blogs?
            filter_javascript=False,
            filter_curly_bracket=False,
            filter_policy=True,
            remove_citations=True,
            language=Languages.vietnamese,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/5_c4/{DUMP_TO_PROCESS}"),
        ),
        FineWebQualityFilter(
            # Keep the lower bound low in order to tolerate many newlines in paragraph formatting
            line_punct_thr=0.1,
            line_punct_exclude_zero=False,
            # Improvement: merge short adjacent examples
            short_line_thr=0.67,
            short_line_length=20, # TODO: audit for bias (e.g. poem exclusion)
            char_duplicates_ratio=0.1, # TODO: try other values
            new_line_ratio=0.3,
            language=Languages.vietnamese,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/6_fineweb_qual/{DUMP_TO_PROCESS}")
        ),
        # TODO: audit for bias (e.g. medical content, Wikipedia...)
        FlaggedWordsThresholdFilter(
            language=Languages.vietnamese,
            language_flagged_words_override=FLAGGED_WORDS_SAILCRAFT,
            flagged_thr=0.01,
            keep_fraction=0.1,
            seed=SEED,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/7_c4_badwords/{DUMP_TO_PROCESS}")
        ),

        # Other filters: FastTextClassifierFilter (probably biased), SamplerFilter

        # TODO: add Decontamination filters from datatrove (requires lighteval)

        JsonlWriter(output_intermediate_1),
    ],
    tasks=64,
    workers=8,
    logging_dir=f"{LOGGING_DIR}/base_processing/{DUMP_TO_PROCESS}",
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
        JsonlReader(output_intermediate_1),
        RensaDeduplicate(
            rensa_index=rensa_index,
            exclusion_writer=JsonlWriter(f"{FILTERING_REMOVED_DIR}/8_dedup/{DUMP_TO_PROCESS}")
        ),
        JsonlWriter(output_intermediate_2),
    ],
    logging_dir=f"{LOGGING_DIR}/minhash/{DUMP_TO_PROCESS}",
    depends=main_processing_executor,
)

tasks_sequence_dedup = 64

sequence_dedup_stage_1 = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(output_intermediate_2),
        ESDatasetToSequence(
            output_folder=es_dir_vi,
            tokenizer_name_or_path=VIETNAMESE_TOKENIZER,
        ),
    ],
    workers=16,
    tasks=tasks_sequence_dedup,
    logging_dir=f"{LOGGING_DIR}/es/1/{DUMP_TO_PROCESS}",
    depends=document_dedup_stage,
)

sequence_dedup_stage_2 = LocalPipelineExecutor(
    pipeline=[
        ESMergeSequences(
            data_folder=es_dir_vi,
            tasks_stage_1=tasks_sequence_dedup,
        ),
    ],
    logging_dir=f"{LOGGING_DIR}/es/2/{DUMP_TO_PROCESS}",
    depends=sequence_dedup_stage_1,
)

external_dedup_stage_3 = LocalPipelineExecutor(
    pipeline=[
        ESComputeRangesExternal(
            length_threshold=100,
            data_folder=es_dir_vi,
            num_threads=16,
        ),
    ],
    logging_dir=f"{LOGGING_DIR}/es/2/{DUMP_TO_PROCESS}",
    depends=sequence_dedup_stage_2,
)

final_stage = LocalPipelineExecutor(
    pipeline=[
        JsonlReader(output_intermediate_2),
        ESRangeRemover(
            min_doc_words=50,
            sequence_folder=es_dir_vi,
            tokenizer_name_or_path=VIETNAMESE_TOKENIZER,
            language=Languages.vietnamese,
        ),
        DocStats(
            f"{STATS_DIR}/docs",
            top_k_config=top_k_config,
        ),
        LineStats(
            f"{STATS_DIR}/lines",
            top_k_config=top_k_config,
        ),
        ParagraphStats(
            f"{STATS_DIR}/paragraphs",
            top_k_config=top_k_config,
        ),
        # Not very meaningful to compute word length (includes spaces!).
        # TODO: count syllables
        WordStats(
            f"{STATS_DIR}/words",
            stop_words=STOP_WORDS,
            language=Languages.vietnamese,
            top_k_config=top_k_config,
        ),
        SentenceStats(
            f"{STATS_DIR}/sentences",
            language=Languages.vietnamese,
            top_k_config=top_k_config,
        ),
        TokenStats(
            f"{STATS_DIR}/tokens",
            tokenizer_name_or_path=VIETNAMESE_TOKENIZER,
            top_k_config=top_k_config,
        ),
        # Expensive, disabled
        #CCNetPerplexityStats(
        #    f"{STATS_DIR}/perplexity",
        #    model_dataset="oscar", # https://huggingface.co/edugp/kenlm/tree/main
        #    language=Languages.vietnamese,
        #    top_k_config=top_k_config,
        #),

        # FIXME: performance/security issues?
        # Possibly use scrubadub for more in-depth cleaning (beware of performance)
        PIIFormatter(),
        JsonlWriter(f"{MAIN_OUTPUT_DIR}/{DUMP_TO_PROCESS}/deduped"),
    ],
    tasks=64,
    workers=16,
    logging_dir=f"{LOGGING_DIR}/es/3/{DUMP_TO_PROCESS}",
    depends=external_dedup_stage_3,
)



def main():
    main_processing_executor.run()
    document_dedup_stage.run()
    sequence_dedup_stage_1.run()
    sequence_dedup_stage_2.run()
    external_dedup_stage_3.run()
    final_stage.run()

if __name__ == "__main__":
    main()
