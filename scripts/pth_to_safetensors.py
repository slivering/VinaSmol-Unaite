#!uv run python3

import os
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
from transformers import AutoModel
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(model_dir: Path, out_dir: Annotated[Optional[Path], typer.Argument()] = None):
    if out_dir is None:
        out_dir = model_dir
    
    #state_dict = torch.load(model_dir / "model.pth")
    model_file = model_dir / "model.pth"
    if model_file.exists():
        logger.info("Moving model.pth to pytorch_model.bin")
        os.rename(model_file, model_dir / "pytorch_model.bin")
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model = model.bfloat16()
    model.save_pretrained(out_dir)

if __name__ == '__main__':
    app()