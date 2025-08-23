import json
from multiprocessing.pool import ThreadPool
from pathlib import Path
import subprocess

from loguru import logger
from tqdm import tqdm
import typer

def convert_pdf(pdf_file: Path, pdf_settings: str = '/screen') -> tuple[Path, Path]:
    file_path = f"{pdf_file}"
    output_file = f"{file_path.removesuffix('.pdf')}_compressed.pdf"
    command = [
        "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4", f"-dPDFSETTINGS={pdf_settings}",
        "-dNOPAUSE", "-dQUIET", "-dBATCH", f"-sOutputFile={output_file}", file_path,
    ]
    subprocess.check_output(command)
    return pdf_file, output_file

def pdfs_to_compress(dir: Path, size_threshold: int, text_info_file: Path | None):
    text_info = {}
    if text_info_file is not None:
        assert text_info_file.name.endswith(".json")
        text_info = json.loads(text_info_file.read_text())

    for pdf_file in dir.glob("./**/*.pdf"):
        pdf_file = pdf_file.resolve()
        #if pdf_file.name.endswith("_compressed.pdf"):
        #    continue
        if Path(f"{pdf_file[:-4]}_compressed.pdf").exists():
            continue
        if pdf_file.stat().st_size < size_threshold:
            continue

        file_path = pdf_file.as_posix()
        if file_path not in text_info or text_info[file_path] == "text":
            yield pdf_file

def setup_logging(logging_dir: Path):
    logging_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        logging_dir / "compress_pdfs.log",
        format="{time} {level} {message}",
        level="DEBUG",
        rotation="10 MB",
    )

def save_results(compressions: dict[str, str], logging_dir: Path):
    conversions_file = logging_dir / "pdf_compressions.json"
    conversions_file.write_text(json.dumps(compressions, indent=2))

def main(
        dir: Path = Path("."),
        *,
        size_threshold: int = 1_000_000,
        pdf_settings: str = "/screen",
        logging_dir: Path = Path("./logs"),
        exclude_scanned: bool = True,
    ):
    """Compress PDF files in bulk with GhostScript.

    Args:
        dir (Path, optional): PDFs will be found recursively in this directory.
        size_threshold (int, optional): The minimum size to perform compression.
        pdf_settings (str, optional): The PDF settings provided to GhostScript.
        logging_dir (Path, optional): The directory to save logs and results.
        exclude_scanned (bool, optional): Do not compress scanned PDFs.
    """
    logging_dir = logging_dir.resolve()
    setup_logging(logging_dir)
    
    results = {}
    pool = ThreadPool()

    text_info_file = None
    if exclude_scanned:
        text_info_file = logging_dir / "pdf_text_info.json"
        if not text_info_file.exists():
            subprocess.check_output([
                "./scan_infer.py",
                f"{dir}",
                "--logging-dir", f"{logging_dir}",
            ])
    
    to_compress = list(pdfs_to_compress(dir, size_threshold, text_info_file))

    results = iter(tqdm(
        pool.imap_unordered(lambda file: convert_pdf(file, pdf_settings), to_compress),
        desc="Compress PDFs",
        total=len(to_compress),
    ))

    compressions = {}
    while True:
        try:
            pdf_file, output_file = next(results)
            compressions[f"{pdf_file}"] = f"{output_file}"
        except subprocess.CalledProcessError as e:
            logger.exception(f"Failed to compress {pdf_file}")
            logger.error(e.stdout.decode('utf-8'))
            logger.error(e.stderr.decode('utf-8'))
        except StopIteration:
            break

    save_results(compressions, logging_dir)


if __name__ == '__main__':
    typer.run(main)
