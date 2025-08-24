"""
Constants for the Vietnamese language.
"""

from pathlib import Path
import requests

from diskcache import Cache
from loguru import logger

DATA_DIR = Path(__file__).parent / "data" / "vietnamese"

LATN_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LATN_LOWERCASE = LATN_UPPERCASE.lower()
VI_UPPERCASE = "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
VI_LOWERCASE = VI_UPPERCASE.lower()
LETTERS = list(LATN_LOWERCASE + LATN_UPPERCASE + VI_LOWERCASE + VI_UPPERCASE)

cache = Cache(DATA_DIR)

@cache.memoize(tag="vietnamese-words")
def download_vietnamese_words(url: str) -> list[str]:
    logger.info("Downloading Vietnamese words at {}", url)
    r = requests.get(url)
    return r.text.splitlines()

WORDS_URL = "https://raw.githubusercontent.com/duyet/vietnamese-wordlist/refs/heads/master/Viet22K.txt"
WORDS = download_vietnamese_words(WORDS_URL)

SYLLABLES = set(syllable for word in WORDS for syllable in word.split())

@cache.memoize(tag="vietnamese-stopwords")
def download_vietnamese_stopwords(url: str) -> list[str]:
    logger.info("Downloading Vietnamese stop words at {}", url)
    r = requests.get(url)
    return r.text.splitlines()

# FIXME: list possibly too long?
STOP_WORDS_URL = "https://github.com/stopwords/vietnamese-stopwords/raw/a453d389e1b52e20748ca83ddd8b0faebb04f5aa/vietnamese-stopwords.txt"
STOP_WORDS = download_vietnamese_stopwords(STOP_WORDS_URL)
