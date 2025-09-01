from pathlib import Path

from transformers import PreTrainedTokenizerBase

DATA_DIR = (Path(__file__).parent / "checkpoints").resolve()

def number_of_added_tokens(
        base_tokenizer: PreTrainedTokenizerBase,
        merged_tokenizer: PreTrainedTokenizerBase,
    ) -> int:
    """Returns the number of added tokens in the merged tokenizer
    compared to the base tokenizer."""
    return len(merged_tokenizer) - len(base_tokenizer)
