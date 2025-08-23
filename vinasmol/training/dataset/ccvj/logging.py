from pathlib import Path
import sys

from loguru import Logger
from tqdm import tqdm

def setup_logging(logger: Logger, logfile: Path):
    logger.level(name="DEBUG", color="<fg 30>")
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
    )
    logger.configure(handlers=[dict(sink=lambda msg: tqdm.write(msg, end=''), colorize=True)])
    logger.add(logfile, format="{time} {level} {message}", level="DEBUG", rotation="100 MB")