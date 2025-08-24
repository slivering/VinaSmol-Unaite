"""
Constants for the Vietnamese language.
"""

import json
import requests

from diskcache import Cache
from loguru import logger

from . import DATA_DIR

_cache = Cache(DATA_DIR / "vietnamese-words")

@_cache.memoize(tag="vietnamese-stopwords")
def download_vietnamese_stopwords(url: str) -> list[str]:
    logger.info("Downloading Vietnamese stop words at {}", url)
    r = requests.get(url)
    return r.text.splitlines()

# FIXME: list possibly too long?
STOP_WORDS_URL = "https://github.com/stopwords/vietnamese-stopwords/raw/a453d389e1b52e20748ca83ddd8b0faebb04f5aa/vietnamese-stopwords.txt"
STOP_WORDS = download_vietnamese_stopwords(STOP_WORDS_URL)


@_cache.memoize(tag="vietnamese-flagged-words")
def download_vietnamese_flagged_words(url: str) -> dict[str, int]:
    logger.info("Downloading Vietnamese flagged words from {}", url)
    r = requests.get(url)
    flagged_words = json.loads(r.text)
    assert isinstance(flagged_words, dict)
    return flagged_words

# TODO: find words that identify adult websites very strongly
# Adjust weights, threshold and audit unnecessary words in the list
FLAGGED_WORDS_URL = "https://gist.githubusercontent.com/slivering/4ab824e9957fa3663ee5ef5d7c7e4315/raw/02de0eb6d7387e2c3bad95729bdb806f099e859e/vietnamese_flagged_words.json"
FLAGGED_WORDS_SAILCRAFT = download_vietnamese_flagged_words(FLAGGED_WORDS_URL)
