#!uv run python3

import os
from pathlib import Path
import shutil
from typing import Annotated, Optional

from loguru import logger
from transformers import AutoModel
import typer

ROOT_DIR = Path(__file__).parent.parent.resolve()

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(model_dir: Path, out_dir: Annotated[Optional[Path], typer.Argument()] = None):
    if out_dir is None:
        out_dir = model_dir
    
    #state_dict = torch.load(model_dir / "model.pth")
    model_file = model_dir / "model.pth"
    config_json = ROOT_DIR / "vinasmol/training/hf_checkpoints/VinaSmol/config.json"
    config_json_out = model_dir / config_json.name

    if model_file.exists():
        logger.info("Moving model.pth to pytorch_model.bin")
        os.rename(model_file, model_dir / "pytorch_model.bin")
    
    if not config_json_out.exists():
        logger.info("Copying {}", config_json)
        shutil.copy2(config_json, config_json_out)
        
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model = model.bfloat16()
    model.save_pretrained(out_dir)

if __name__ == '__main__':
    app()