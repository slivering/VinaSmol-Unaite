#!uv run python3

from pathlib import Path
from typing import Annotated, Optional

import torch
from transformers import AutoModel
import typer

def main(model_dir: Path, out_dir: Annotated[Optional[Path], typer.Argument()] = None):
    if out_dir is None:
        out_dir = model_dir
    
    state_dict = torch.load(model_dir / "model.pth")
    model = AutoModel.from_pretrained(
        model_dir, local_files_only=True, state_dict=state_dict
    )
    model.save_pretrained(out_dir)

if __name__ == '__main__':
    typer.run(main)