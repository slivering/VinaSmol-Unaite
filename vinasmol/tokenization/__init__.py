from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    PreTrainedModel, PreTrainedTokenizerBase,
)

from ..hfmodel import ENVIT5
from ..vietnamese import SYLLABLES

DATA_DIR = (Path(__file__).parent / "data").resolve()

def number_of_added_tokens(
        base_tokenizer: PreTrainedTokenizerBase,
        merged_tokenizer: PreTrainedTokenizerBase,
    ) -> int:
    """Returns the number of added tokens in the merged tokenizer
    compared to the base tokenizer."""
    return len(merged_tokenizer) - len(base_tokenizer)

def initialize_new_embeddings(
        model: PreTrainedModel,
        base_tokenizer: PreTrainedTokenizerBase,
        # FIXME: Tokenizer.from_file is maybe not fast and doesn't inherit from PreTrainedTokenizerBase
        merged_tokenizer: PreTrainedTokenizerBase,
        translation_weight: float = 0.9,
    ) -> int:
    """Initialize new token embeddings with meaningful defaults."""
    assert 0.0 <= translation_weight <= 1.0
    t = translation_weight

    # Hewitt initialization by default: https://nlp.stanford.edu/~johnhew/vocab-expansion.html
    model.resize_token_embeddings(len(merged_tokenizer), mean_resizing=True)
   
    translations = vietnamese_vocabulary_to_english(
        merged_tokenizer,
        min_token_id=len(base_tokenizer),
    )
    tokenized_translations = base_tokenizer(
        list(translations),
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        translation_embeddings: list = [
            # We take the mean token embedding for the translation
            torch.mean(torch.cat(
                [model.lm_head[token_id] for token_id in tokenized],
                dim=0,
            ))
            for tokenized in tokenized_translations
        ]
        for token_id, embedding in zip(translations.keys(), translation_embeddings):
            init_embed = model.lm_head[token_id].clone()
            trans_embed = embedding.clone()
            model.lm_head[token_id] = (1 - t) * init_embed + t * trans_embed

            # Alternative: decouple added token embeddings?
            # https://github.com/huggingface/transformers/issues/1413#issuecomment-2404291317

    return len(translations)


def is_vietnamese_token(decoded_token: str) -> bool:
    """Returns whether a token is made out of Vietnamese syllables."""
    # TODO: get the list of words that contain the token -> embeddings -> average
    # possibly use notion of distance (but test before implementing that)
    return all(syllable in SYLLABLES for syllable in decoded_token.split())

def find_vietnamese_tokens(
        tokenizer: PreTrainedTokenizerBase,
        min_token_id: int = 0,
    ) -> dict[int, str]:
    """Determine all of the tokens of a tokenizer that have a meaning in Vietnamese."""
    special_ids = tokenizer.all_special_ids
    vietnamese_tokens = {}

    for token_id in tokenizer.get_vocab().values():
        if token_id < min_token_id:
            continue
        if token_id in special_ids:
            continue

        decoded = tokenizer.decode(token_id, skip_special_tokens=True)
        if is_vietnamese_token(decoded):
            vietnamese_tokens[token_id] = decoded
    
    return vietnamese_tokens

def vietnamese_to_english(inputs: list[str]) -> list[str]:
    """Translate a list of Vietnamese inputs to English."""
    tokenizer = ENVIT5.load_tokenizer()
    model = AutoModelForSeq2SeqLM.from_pretrained(ENVIT5.name)
    inputs = [f"vi: {input}" for input in inputs]
    tokenized = tokenizer(inputs, return_tensors="pt", padding=True)
    # TODO: cache translations
    outputs = model.generate(tokenized.input_ids.to(model.device), max_length=32)
    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )

def vietnamese_vocabulary_to_english(
        tokenizer: PreTrainedTokenizerBase,
        min_token_id: int = 0,
    ) -> dict[int, str]:
    """Convert the Vietnamese vocabulary of a tokenizer to English.

    Args:
        tokenizer (PreTrainedTokenizerBase): The base tokenizer.
        min_token_id (int, optional): The minimum id for the tokens to analyze. This option
            is useful to skip the previous English vocabulary of a merged tokenizer.

    Returns:
        translations (dict[int, str]): A mapping from a token id to its English translation.
    """
    vietnamese_tokens = find_vietnamese_tokens(tokenizer, min_token_id=min_token_id)
    translations = vietnamese_to_english(list(vietnamese_tokens.values()))
    return {
        token_id: translation
        for token_id, translation in zip(vietnamese_tokens.keys(), translations)
    }
