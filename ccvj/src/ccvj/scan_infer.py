#!/usr/bin/python

from enum import StrEnum
import json
from multiprocessing.pool import ThreadPool
from pathlib import Path
import subprocess


from loguru import logger
from tqdm import tqdm
import typer


class PDFTextInfo(StrEnum):
    TEXT = "text"
    SCANNED = "scanned"
    UNKNOWN = "unknown"

def find_substring(file: Path, substring: str) -> bool:
    return subprocess.run(
        ["grep", "-aqs", substring, f"{file}"],
    ).returncode == 0

def get_pdf_text_info(pdf_file: Path) -> tuple[Path, PDFTextInfo]:
    pdf_file = pdf_file.resolve()
    has_text = find_substring(pdf_file, "/Text/")
    if has_text:
        return pdf_file, PDFTextInfo.TEXT
    has_image = find_substring(pdf_file, "/Image/")
    if has_image:
        return pdf_file, PDFTextInfo.SCANNED
    return pdf_file, PDFTextInfo.UNKNOWN

def setup_logging(logging_dir: Path):
    logging_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        logging_dir / "scan_infer.log",
        format="{time} {level} {message}",
        level="DEBUG",
        rotation="10 MB",
    )
    logger.configure(handlers=[dict(sink=lambda msg: tqdm.write(msg, end=''), colorize=True)])

def pdfs_to_infer(dir: Path):
    for pdf_file in dir.glob("./**/*.pdf"):
        pdf_file = pdf_file.resolve()
        if pdf_file.name.endswith(".pdf.bkp"):
            continue
        yield pdf_file

def save_results(infos: dict[str, str], logging_dir: Path):
    infos_file = logging_dir / "pdf_text_info.json"
    infos_file.write_text(json.dumps(infos, indent=2))

def main(
        pdf_dir: Path,
        logging_dir: Path = Path("./logs"),
    ):
    setup_logging(logging_dir)

    results = {}
    pool = ThreadPool()

    pdfs = list(pdfs_to_infer(pdf_dir))

    results = iter(tqdm(
        pool.imap_unordered(get_pdf_text_info, pdfs),
        desc="Get PDF text infos",
        total=len(pdfs)
    ))
    infos = {}
    while True:
        try:
            pdf_file, info = next(results)
            infos[f"{pdf_file.resolve()}"] = f"{info}"
        except subprocess.CalledProcessError as e:
            logger.exception("Failed get info for {}", pdf_file)
            logger.error(f"\tStdout:\n{e.stdout.decode('utf-8')}")
            logger.error(f"\tStderr:\n{e.stderr.decode('utf-8')}")
        except StopIteration:
            break

    infos = dict(sorted(infos.items()))
    save_results(infos, logging_dir)


if __name__ == '__main__':
    typer.run(main)
