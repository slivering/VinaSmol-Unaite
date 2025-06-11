"""
Constants for the Vietnamese language.
"""

import importlib
import requests

def download_vietnamese_stopwords(url: str) -> list[str]:
    r = requests.get(url)
    return r.text.splitlines()

# FIXME: list possibly too long?
STOP_WORDS_URL = "https://github.com/stopwords/vietnamese-stopwords/raw/a453d389e1b52e20748ca83ddd8b0faebb04f5aa/vietnamese-stopwords.txt"
STOP_WORDS = download_vietnamese_stopwords(STOP_WORDS_URL)

def compile_module(name: str, source: str):
    """Create a module from a source string."""
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(source, module.__dict__)
    return module

def download_vietnamese_flagged_words(url: str) -> list[str]:
    r = requests.get(url)
    flagged_words = compile_module('flagged_words', r.text)
    return flagged_words.flagged_words['vi'] # noqa

# TODO: find words that identify adult websites very strongly
# Weigh words by "cursedness" and audit unnecessary words in the list
FLAGGED_WORDS_URL = "https://github.com/sail-sg/sailcraft/raw/315d002711dea7b2541bdab2a322c45bb2e197fa/code/data_cleaning/flagged_words.py"
FLAGGED_WORDS_SAILCRAFT = download_vietnamese_flagged_words(FLAGGED_WORDS_URL)
