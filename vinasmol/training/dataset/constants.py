"""
Constants for Vietnamese dataset preparation.
"""

import json
import requests

from loguru import logger

from vinasmol.vietnamese import STOP_WORDS, cache # noqa

@cache.memoize(tag="vietnamese-flagged-words")
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
