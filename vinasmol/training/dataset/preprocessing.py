#import asyncio
from datetime import datetime
from enum import StrEnum
#from goodwiki import GoodwikiClient
import json
import random
import re

from datatrove.utils.text import Languages
import pypandoc
from vinasmol.hfmodel import LUCIE, SMOLLM2

def filter_keys(d: dict, keys: list[str]) -> dict:
    return {k: d[k] for k in keys}

def format_en_wiki_to_md(title: str, mediawiki: str) -> str:
    """Format an English Wikipedia page to Markdown.

    Args:
        mediawiki (str): the page content, in MediaWiki format.

    Returns:
        str: The formatted content as Github-Flavored Markdown.
    """
    #client = GoodwikiClient()
    # This function is not really async since we're directly processing the page content
    # page = asyncio.run(client.get_page_from_wikitext(
    #     mediawiki,
    #     title,
    #     pageid=0,
    #     revid=0,
    #     with_styling=False,
    # ))
    # return page.markdown
    return mediawiki # FIXME: goodwiki is outdated

def format_vi_wiki_to_md(title: str, mediawiki: str) -> str:
    """Format a Vietnamese Wikipedia page to Markdown.

    Args:
        mediawiki (str): the page content, in MediaWiki format.

    Returns:
        str: The formatted content as Github-Flavored Markdown.
    """
    # TODO: fork GoodWiki to add Vietnamese support
    return format_en_wiki_to_md(title, mediawiki)

def parse_publication_date(publish: str) -> datetime:
    """Parse the publication date of a Vietnamese new article.

    Args:
        publish (str): the value of the "publish" field in the BKAI News Corpus.
            It should have the following format: `{"$date": "2023-10-16502:16:00.000Z"}`

    Returns:
        datetime: the parsed publication date.
    """
    date = json.loads(publish)["$date"]
    return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%z")

_DISABLED_PANDOC_MD_EXTENSIONS = [
    'raw_html',
    'fenced_divs',
    'native_divs',
    'link_attributes',
    'markdown_attribute',
    'bracketed_spans',
    'footnotes',
]

_MD_LINK = re.compile(r"\[([^\[\]]+)\]\(([^)]+)\)")
_MD_LINK_WITH_SQUARE_BRACKETS = re.compile(r"\[\\\[([^\[\]]+)\\\]\]\(([^)]+)\)")

_ABSTRACT_RE = re.compile(r"^Abstract", flags=re.MULTILINE)
_REFERENCES_RE = re.compile(r"^References|^\*\*References", flags=re.MULTILINE)

def format_olmocr_pes2o(text: str) -> str:
    """Format olmOCR-pes2o-0225 to Markdown and remove superflous content if possible.
    
    - Author list
    - Reference list
    - Appendix
    """
    title = text.splitlines()[0]
    
    abstract_idx = len(title) + 1
    abstract_match = _ABSTRACT_RE.search(text)
    if abstract_match is not None:
        abstract_idx = abstract_match.start()
    
    references_idx = len(text)
    references_match = _REFERENCES_RE.search(text)
    if references_match is not None:
        references_idx = references_match.start()

    # TODO: also remove inline references (hard)

    # This intentionally excludes the appendix, which avoids too long documents
    return title + "\n" + text[abstract_idx:references_idx]



def _set_text_as_md_in_wiki(row: dict, lang: str = 'en') -> dict:
    # TODO:
    # match lang:
    #     case 'en':
    #         fmt = format_en_wiki_to_md
    #     case 'vi':
    #         fmt = format_vi_wiki_to_md
    # row['text'] = fmt(row['title'], row['raw_mediawiki'])
    del row['raw_mediawiki']
    return row

def _set_text_as_md_in_vbpl(row: dict) -> dict:
    # TODO: check that markdown content is faithfully formatted
    row['text'] = row['markdown_content']
    del row['html_content']
    return row


class DatasetNames(StrEnum):
    _ignore_ = [
        '_IDS', '_ID_COUNTERS', '_GLOBAL_ID_COUNTER',
        'ENABLED', 'ENGLISH', 'VIETNAMESE', 'CODE',
    ]

    # SmolLM corpus
    cosmopedia_v2 = "HuggingFaceTB/smollm-corpus:cosmopedia-v2"
    fineweb_edu_dedup = "HuggingFaceTB/smollm-corpus:fineweb-edu-dedup"
    python_edu = "HuggingFaceTB/smollm-corpus:python-edu"

    # Lucie annealing datasets
    open_web_math = "open-web-math/open-web-math"
    olmocr_pes2o = "allenai/olmOCR-pes2o-0225"
    stackmathqa = "math-ai/StackMathQA:stackmathqa200k"
    flan_v2 = "SirNeural/flan_v2"
    mathpile_commercial = "GAIR/MathPile_Commercial"
    wikipedia_en = "omarkamali/wikipedia-monthly:20250702.en"

    # Lucie pretraining datasets (excerpt)
    claire_en = "OpenLLM-France/Lucie-Training-Dataset:Claire-en"
    gutenberg_en = "OpenLLM-France/Lucie-Training-Dataset:Gutenberg-en"
    europarl_en = "OpenLLM-France/Lucie-Training-Dataset:Europarl-en"

    lucie_training_code = "OpenLLM-France/Lucie-Training-Dataset:code"

    # Vietnamese datasets
    wikipedia_vi = "omarkamali/wikipedia-monthly:20250702.vi"
    culturax = "uonlp/CulturaX:vi"
    madlad400 = "allenai/MADLAD-400:vi"
    fineweb2_hq = "epfml/FineWeb2-HQ:vie_Latn"
    bkai_news_corpus = "bkai-foundation-models/BKAINewsCorpus"
    vbpl = "doanhieung/vbpl"
    mtet = "phongmt184172/mtet"

    _IDS = {}
    _GLOBAL_ID_COUNTER = 0
    _ID_COUNTERS = {}
    _ENABLED: dict[str, set["DatasetNames"]] = {}
    ENGLISH = set()
    VIETNAMESE = set()
    CODE = set()


    @classmethod
    def _init_cls_vars(cls):
        """Initialize class member variables."""
        cls._IDS = {
            name: i + 1
            for i, name in enumerate(cls.__members__)
        }
        cls._ID_COUNTERS = {name: 0 for name in cls._IDS}
        cls._GLOBAL_ID_COUNTER = 0
        cls.ENABLED = {
            SMOLLM2.name: {
                cls.cosmopedia_v2,
                cls.fineweb_edu_dedup,
                cls.python_edu,
                cls.open_web_math,
                cls.olmocr_pes2o,
                cls.stackmathqa,
                cls.wikipedia_en,

                cls.wikipedia_vi,
                cls.culturax,
                cls.fineweb2_hq,
                cls.bkai_news_corpus,
                cls.vbpl,
                cls.mtet,
            },
            LUCIE.name: set(cls.__members__.values())
        }
        cls.ENGLISH = {
            cls.cosmopedia_v2,
            cls.fineweb_edu_dedup,
            cls.python_edu,
            cls.open_web_math,
            cls.olmocr_pes2o,
            cls.stackmathqa,
            cls.flan_v2,
            cls.mathpile_commercial,
            cls.wikipedia_en,
            cls.claire_en,
            cls.gutenberg_en,
            cls.europarl_en,
        }
        cls.VIETNAMESE = {
            cls.wikipedia_vi,
            cls.culturax,
            cls.madlad400,
            cls.fineweb2_hq,
            cls.bkai_news_corpus,
            cls.vbpl,
            cls.mtet,
        }
        cls.CODE = {
            cls.lucie_training_code,
        }
    
    @property
    def language(self) -> str:
        cls = type(self)
        if self in cls.ENGLISH:
            return Languages.english
        elif self in cls.VIETNAMESE:
            return Languages.vietnamese
        elif self in cls.CODE:
            return "code"
        raise RuntimeError(f"Unknown language: {self}")


    @property
    def _id(self) -> int:
        """The dataset identifier."""
        return DatasetNames._IDS[self.name]
    
    @property
    def _counter(self) -> int:
        return DatasetNames._ID_COUNTERS[self.name]
    
    def origin_metadata(self, id: int | str = None) -> dict[str, object]:
        """Add additional metadata about the used source dataset.

        Args:
            id (int, optional): The original id of the dataset.

        Returns:
            dict[str, object]: additional metadata about this dataset.
        
        **Note:** this function should always be called after [`generate_row_id`].
        """
        metadata = dict(origin=self.value)
        if id is None:
            metadata['dataset_generated_id'] = self._counter
        else:
            metadata['original_id'] = id
        
        return metadata

    def generate_row_id(self) -> int:
        """Generate a new globally-unique id for a row.

        Args:
            id (int, optional): The original id used by the dataset source,
                to be registered and mapped to the new generated row id.

        Returns:
            int: A globally unique row id.
        
        Warning
        -----
            This function cannot be called concurrently as it mutates a global shared variable.
        """
        DatasetNames._ID_COUNTERS[self.name] += 1
        DatasetNames._GLOBAL_ID_COUNTER += 1
        return DatasetNames._GLOBAL_ID_COUNTER

DatasetNames._init_cls_vars()


class NormalizeCols:
    """Organize the dataset columns according to a unified format.
    
    Warning
    -----
        The static methods provided by this class cannot be called concurrently
        as they mutate global shared state.
    """

    @staticmethod
    def format_prompt_response(prompt: str, response: str) -> str:
        """Return a text example using a randomized prompt-response template."""
        # TODO: experiment with best prompt templates, possibly use Vietnamese templates
        return random.choices(
            population=[
                f"{prompt}\n{response}",
                f"User: {prompt}\nAssistant: {response}",
                f"User: {prompt} Assistant: {response}",
                f"{prompt} {response}",
            ],
            weights=[
                0.75,
                0.15,
                0.05,
                0.05,
            ]
        )[0]
    
    @staticmethod
    def cosmopedia_v2(row: dict) -> dict:
        return dict(
            id=DatasetNames.cosmopedia_v2.generate_row_id(),
            # Don't include prompt
            text=row['text'],
            metadata=dict(
                **filter_keys(row, ['audience', 'format', 'seed_data']),
                **DatasetNames.cosmopedia_v2.origin_metadata(),
            ),
        )
    
    @staticmethod
    def fineweb_edu_dedup(row: dict) -> dict:
        # text, id(string), metadata[dump, url, date, file_path, score, int_score]
        return dict(
            id=DatasetNames.fineweb_edu_dedup.generate_row_id(),
            text=row['text'],
            metadata=dict(
                **filter_keys(
                    row['metadata'],
                    ['dump', 'url', 'date', 'file_path', 'score', 'int_score'],
                ),
                **DatasetNames.fineweb_edu_dedup.origin_metadata(row['id']),
            ),
        )

    @staticmethod
    def python_edu(row: dict) -> dict:
        return dict(
            id=DatasetNames.python_edu.generate_row_id(),
            text=row['text'],
            metadata=dict(
                **filter_keys(row, ['blob_id', 'repo_name', 'path', 'score']),
                **DatasetNames.python_edu.origin_metadata(),
            ),
        )
    
    @staticmethod
    def open_web_math(row: dict) -> dict:
        # url, text, date, metadata['warc_path']
        return dict(
            id=DatasetNames.open_web_math.generate_row_id(),
            text=row['text'],
            metadata=dict(
                warc_path=row['metadata']['warc_path'],
                **filter_keys(row, ['url', 'date']),
                **DatasetNames.open_web_math.origin_metadata(),
            ),
        )
    
    @staticmethod
    def stackmathqa(row: dict) -> dict:
        return dict(
            id=DatasetNames.stackmathqa.generate_row_id(),
            text=NormalizeCols.format_prompt_response(row['Q'], row['A']),
            metadata=dict(
                **filter_keys(
                    row,
                    ['url', 'timestamp', 'source', 'answer_count', 'answer_id', 'question_score'],
                ),
                **DatasetNames.stackmathqa.origin_metadata(),
            ),
        )
    
    @staticmethod
    def olmocr_pes2o(row: dict) -> dict:
        return dict(
            id=DatasetNames.olmocr_pes2o.generate_row_id(),
            text=format_olmocr_pes2o(row['text']),
            metadata=dict(
                **filter_keys(row['metadata'], ['pdf-total-pages', 'fieldofstudy']),
                **DatasetNames.olmocr_pes2o.origin_metadata(row['id']),
            ),
        )
    
    @staticmethod
    def flan_v2(row: dict) -> dict:
        return dict(
            id=DatasetNames.flan_v2.generate_row_id(),
            text=NormalizeCols.format_prompt_response(row['inputs'], row['targets']),
            metadata=dict(
                **filter_keys(
                    row,
                    ['task'],
                ),
                **DatasetNames.flan_v2.origin_metadata(),
            ),
        )
    
    @staticmethod
    def mathpile_commercial(row: dict) -> dict:
        # TODO: more preprocessing for arXiv and Wikipedia, for example
        md_text = pypandoc.convert_text(
            row['text'],
            'markdown-' + '-'.join(_DISABLED_PANDOC_MD_EXTENSIONS),
            format='latex',
            extra_args=["--wrap=none"],
        )
        # Replace links by their texts
        for pat in [
                _MD_LINK,
                _MD_LINK_WITH_SQUARE_BRACKETS,
            ]:
            md_text = pat.sub(lambda m: m.groups()[0], md_text)
        
        return dict(
            id=DatasetNames.mathpile_commercial.generate_row_id(),
            text=md_text,
            metadata=dict(
                **filter_keys(row, ['subset']),
                **DatasetNames.mathpile_commercial.origin_metadata(row['meta']['id']),
            ),
        )

    @staticmethod
    def claire_en(row: dict) -> dict:
        return dict(
            id=DatasetNames.claire_en.generate_row_id(),
            text=row['text'],
            metadata=dict(
                subset=row['extra']['subset'],
                **DatasetNames.claire_en.origin_metadata(row['id']['idx_row']),
            ),
        ) 
    
    @staticmethod
    def wikipedia_en(row: dict) -> dict:
        row['metadata'] = dict(
            **filter_keys(row, ['url', 'title']),
            **DatasetNames.wikipedia_en.origin_metadata(row['id']),
        )
        del row['url']
        del row['title']
        row['id'] = DatasetNames.wikipedia_en.generate_row_id()
        return row

    @staticmethod
    def wikipedia_vi(row: dict) -> dict:
        row['metadata'] = dict(
            **filter_keys(row, ['url', 'title']),
            **DatasetNames.wikipedia_vi.origin_metadata(row['id']),
        )
        del row['url']
        del row['title']
        row['id'] = DatasetNames.wikipedia_vi.generate_row_id()
        return row

    @staticmethod
    def culturax(row: dict) -> dict:
        return dict(
            id=DatasetNames.culturax.generate_row_id(),
            text=row['text'],
            metadata=dict(
                **filter_keys(row, ['url', 'timestamp', 'source']),
                **DatasetNames.culturax.origin_metadata(row['id']),
            )
        )

    @staticmethod
    def madlad400(row: dict) -> dict:
        return dict(
            id=DatasetNames.madlad400.generate_row_id(),
            text=row['text'],
            metadata=DatasetNames.madlad400.origin_metadata(),
        )

    @staticmethod
    def fineweb2_hq(row: dict) -> dict:
        return dict(
            id=DatasetNames.fineweb2_hq.generate_row_id(),
            text=row['text'],
            metadata=dict(
                # How is quality score computed?
                **filter_keys(row, ['url', 'date', 'dump', 'language_score', 'quality_score']),
                **DatasetNames.fineweb2_hq.origin_metadata(row['id']),
            )
        )

    @staticmethod
    def bkai_news(row: dict) -> dict:
        row['metadata'] = dict(
            url=row['link'],
            date=parse_publication_date(row['publish']).strftime("%Y-%m-%dT%H:%M:%S%z"),
            **DatasetNames.bkai_news_corpus.origin_metadata(row['id'])
        )
        del row['link']
        del row['publish']
        row['id'] = DatasetNames.bkai_news_corpus.generate_row_id()
        return row

    @staticmethod
    def vbpl(row: dict) -> dict:
        return dict(
            id=DatasetNames.vbpl.generate_row_id(),
            text=row['text'],
            metadata=dict(
                title=row["Tiêu đề"],
                scope=row["Phạm vi"],
                industry=row["Ngành"],
                sector=row["Lĩnh vực"],
                effective_date=row["Ngày có hiệu lực"],
                validity_status=row["Tình trạng hiệu lực"],
                **DatasetNames.vbpl.origin_metadata(row['id'])
            )
        )

    @staticmethod
    def mtet(row: dict) -> dict:
        return dict(
            id=DatasetNames.mtet.generate_row_id(),
            text=NormalizeCols.format_prompt_response(row['prompt'], row['response']),
            metadata=DatasetNames.mtet.origin_metadata(),
        )
