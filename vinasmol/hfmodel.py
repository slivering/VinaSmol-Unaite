from dataclasses import dataclass
from transformers import AutoTokenizer, PreTrainedTokenizerBase

@dataclass
class HFModel:
    # The HuggingFace identifier of the base model weights
    name: str

    # The HuggingFace identifier of the base model tokenizer
    tokenizer: str

    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(self.name)
    
    @property
    def friendly_name(self) -> str:
        return self.name.split('/')[1].split('-')[0]


LUCIE = HFModel(
    name="OpenLLM-France/Lucie-7B-Instruct-v1.1",
    tokenizer="OpenLLM-France/Lucie-7B"
)

SMOLLM2 = HFModel(
    name="HuggingFaceTB/SmolLM2-360M",
    tokenizer="HuggingFaceTB/SmolLM2-360M",
)

# For Vietnamese tokenization when processing the dataset
SAILOR_2_8B = HFModel(
    name="sail/Sailor2-8B-Chat",
    tokenizer="sail/Sailor2-8B-Chat",
)

BASE_MODEL = SMOLLM2