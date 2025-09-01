from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from torch.utils.data import DataLoader

from litgpt.data import DataModule, SFTDataset, Alpaca, Deita, deita
from litgpt.tokenizer import Tokenizer

@dataclass
class VinaSmolData(DataModule):
    """A mix of Vietnamese, English and code datasets with training and validation dataloaders."""

    data_path: Union[str, Path] = Path("data/")
    annealing: bool = False
    seed: int = 20250828
    num_workers: int = 4

    batch_size: int = field(init=False, repr=False, default=1)
    seq_length: int = field(init=False, repr=False, default=2048)

    def __post_init__(self):
        super().__init__()
        self.data_path = Path(self.data_path)
        self.vi_train = self.data_path / "vi-all/train"
        self.vi_val = self.data_path / "vi-all/val"
        self.en_train = self.data_path / "en-all/train"
        self.en_val = self.data_path / "en-all/val"
        self.code_train = self.data_path / "code-all/train"
        self.code_val = self.data_path / "code-all/val"
        self.required_paths = [
            self.vi_train, self.vi_val,
            self.en_train, self.en_val,
            self.code_train, self.code_val,
        ]
        if self.annealing:
            self.annealing_train = self.data_path / "annealing/train"
            self.annealing_val = self.data_path / "annealing/val"

    def connect(
        self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        self.batch_size = batch_size
        # Increase by one because we need the next token as well
        self.seq_length = max_seq_length + 1

    def prepare_data(self) -> None:
        for path in self.required_paths:
            if not path.is_dir():
                raise FileNotFoundError(
                    "The data path for SmolLM must be a directory with these subdirectories:"
                    " `en-all/train`, `en-all/val`, `vi-all/train`, `vi-all/val`, `code-all/train`, `code-all/val`."
                    f" The directory {path} does not exist."
                    " Set it via `--data.data_path=...`"
                )

    def train_dataloader(self) -> DataLoader:
        from litdata import TokensLoader
        from litdata.streaming import (
            CombinedStreamingDataset,
            StreamingDataLoader, StreamingDataset,
        )

        vi_train_data = StreamingDataset(
            input_dir=self.vi_train,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )
        en_train_data = StreamingDataset(
            input_dir=self.en_train,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )
        code_train_data = StreamingDataset(
            input_dir=self.code_train,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )
        train_datasets = [
            vi_train_data,
            en_train_data,
            code_train_data,
        ]

        if self.annealing:
            annealing_data = StreamingDataset(
                input_dir=self.annealing_train,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            )
            train_datasets.append(annealing_data)
            # TODO: make these weights configurable
            weights = (0.32, 0.25, 0.03, 0.4)
        else:
            weights = (0.55, 0.4, 0.05)


        train_data = CombinedStreamingDataset(
            datasets=train_datasets,
            seed=self.seed,
            weights=weights,
            iterate_over_all=False,
        )

        train_dataloader = StreamingDataLoader(
            train_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata import TokensLoader
        from litdata.streaming import (
            StreamingDataLoader, CombinedStreamingDataset, StreamingDataset,
        )

        vi_val_data = StreamingDataset(
            input_dir=self.vi_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
        )
        en_val_data = StreamingDataset(
            input_dir=self.en_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
        )
        code_val_data = StreamingDataset(
            input_dir=self.code_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
        )
        val_datasets = [
            vi_val_data,
            en_val_data,
            code_val_data,
        ]
        if self.annealing:
            annealing_data = StreamingDataset(
                input_dir=self.annealing_val,
                item_loader=TokensLoader(block_size=self.seq_length),
                shuffle=True,
                drop_last=True,
            )
            val_datasets.append(annealing_data)

        val_data = CombinedStreamingDataset(
            datasets=val_datasets,
            seed=self.seed,
            iterate_over_all=False,
        )
        val_dataloader = StreamingDataLoader(
            val_data,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return val_dataloader

@dataclass
class AlpacaVi(Alpaca):
    file_url: str = "https://huggingface.co/datasets/tsdocode/vi_alpaca_clean/resolve/main/vi_alpaca_data.json"

    file_name: str = "vi_alpaca_data.json"

    download_dir: Path = Path("./data/vi_alpaca_clean")

@dataclass
class MultiTurnAlpacaVi(Deita):
    """Vietnamese multi-turn Alpaca data module for supervised finetuning."""

    download_dir: Path = Path("./data/vi_multi_turn_alpaca")

    repo_id: str = "lamhieu/alpaca_multiturns_dialogue_vi"

    val_split_fraction: float = 0.078758 # Get exactly 1000 examples

    def prepare_data(self) -> None:
        from datasets import load_dataset

        load_dataset(self.repo_id, split='train', cache_dir=self.download_dir)

    def setup(self, stage: str = "") -> None:
        from datasets import load_dataset

        train_val_split = load_dataset(self.repo_id, split='train').train_test_split(
            test_size=self.val_split_fraction,
            seed=self.seed,
        )
        train_data = deita.format_dataset(
            train_val_split['train'],
            self.include_multiturn_conversations,
        )
        test_data = deita.format_dataset(
            train_val_split['test'],
            self.include_multiturn_conversations,
        )

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )