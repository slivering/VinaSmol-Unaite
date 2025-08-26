from pathlib import Path
from typing import Annotated

from loguru import logger
import torch
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    GPT2TokenizerFast,
    PreTrainedModel, PreTrainedTokenizerBase,
)
from transformers.models import LlamaForCausalLM
import typer

from . import DATA_DIR

from ..hfmodel import BASE_MODEL, ENVIT5
from ..vietnamese import SYLLABLES

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
    translations = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )
    for in_, tr in zip(inputs, translations):
        if not tr.startswith("en: "):
            logger.warning(
                "EnViT5 English translation did not start with 'en: '. "
                "model generated '{}' from '{}'", in_, tr,
            )

    return [tr.removeprefix("en: ") for tr in translations]

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
    # A few subword translations won't make sense, but this does not matter.
    translations = vietnamese_to_english(list(vietnamese_tokens.values()))

    token_translations_file = DATA_DIR / "token_translations.txt"
    token_translations_file.write_text("\n".join(
        f"{token}\t{translation}"
        for token, translation in zip(vietnamese_tokens.values(), translations)
    ))
    logger.info("Saved token translations to {}", token_translations_file)
    return {
        token_id: translation
        for token_id, translation in zip(vietnamese_tokens.keys(), translations)
    }


def initialize_new_embeddings(
        model: PreTrainedModel,
        base_tokenizer: PreTrainedTokenizerBase,
        # FIXME: Tokenizer.from_file is maybe not fast and doesn't inherit from PreTrainedTokenizerBase
        merged_tokenizer: PreTrainedTokenizerBase,
        translation_weight: float = 0.9,
        tie_weights: bool = True,
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
    tokenized_translations = base_tokenizer(list(translations.values()))['input_ids']
    with torch.no_grad():
        translation_embeddings: list = [
            # We take the mean token embedding for the translation
            torch.mean(torch.cat(
                [model.lm_head.weight[token_id, :] for token_id in tokenized],
                dim=0,
            ))
            for tokenized in tokenized_translations
        ]
        for token_id, embedding in zip(translations.keys(), translation_embeddings):
            init_embed = model.lm_head.weight[token_id, :]
            tr_embed = embedding.to(model.device)
            model.lm_head.weight[token_id, :] = (1 - t) * init_embed + t * tr_embed
            # Alternative: decouple added token embeddings?
            # https://github.com/huggingface/transformers/issues/1413#issuecomment-2404291317

        if tie_weights:
            model.tie_weights()

    return len(translations)

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
        merged_tokenizer_dir: Annotated[
            Path,
            typer.Argument(help="The directory of the tokenizer with merged vocabulary")
        ],
        out_model_dir: Annotated[
            Path,
            typer.Argument(help="The directory to save the model with extended vocabulary")
        ],
        translation_weight: Annotated[
            float,
            typer.Option(help="The embedding weight of the English translations of the Vietnamese tokens")
        ] = 0.9):
    
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(BASE_MODEL.name)
    base_tokenizer = BASE_MODEL.load_tokenizer()
    merged_tokenizer = GPT2TokenizerFast.from_pretrained(merged_tokenizer_dir)
    initialize_new_embeddings(
        model,
        base_tokenizer,
        merged_tokenizer,
        translation_weight=translation_weight,
    )
    model.to(torch.bfloat16).save_pretrained(out_model_dir)



if __name__ == '__main__':
    app()