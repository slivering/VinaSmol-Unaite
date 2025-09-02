import json
from multiprocessing.pool import ThreadPool
from pathlib import Path
import subprocess

from loguru import logger
from tqdm import tqdm
import typer

def compress_pdf(pdf_file: Path, pdf_settings: str = '/screen') -> tuple[Path, Path]:
    file_path = f"{pdf_file}"
    backup_file = f"{file_path.removesuffix('.pdf')}.pdf.bkp"
    pdf_file.rename(backup_file)
    command = [
        "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.4", f"-dPDFSETTINGS={pdf_settings}",
        "-dNOPAUSE", "-dQUIET", "-dBATCH", f"-sOutputFile={file_path}", backup_file,
    ]
    subprocess.check_output(command)
    return pdf_file, backup_file

def pdfs_to_compress(dir: Path, size_threshold: int, text_info_file: Path | None):
    text_info = {}
    if text_info_file is not None:
        assert text_info_file.name.endswith(".json")
        text_info = json.loads(text_info_file.read_text())

    for pdf_file in dir.glob("./**/*.pdf"):
        pdf_file = pdf_file.resolve()
        if pdf_file.stat().st_size < size_threshold:
            continue
        yield pdf_file, text_info.get(pdf_file.as_posix(), "unknown")

def setup_logging(logging_dir: Path):
    logging_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        logging_dir / "compress_pdfs.log",
        format="{time} {level} {message}",
        level="DEBUG",
        rotation="10 MB",
    )
    logger.configure(handlers=[dict(sink=lambda msg: tqdm.write(msg, end=''), colorize=True)])


def save_results(compressions: dict[str, str], logging_dir: Path):
    conversions_file = logging_dir / "pdf_compressions.json"
    conversions_file.write_text(json.dumps(compressions, indent=2))

def remove_backups_files(dir: Path = Path(".")):
    for backup_file in dir.glob("**/*.pdf.bkp"):
        backup_file.unlink(missing_ok=True)

def main(
        dir: Path = Path("."),
        *,
        size_threshold: int = 1_000_000,
        text_pdf_settings: str = "/screen",
        image_pdf_settings: str = "/ebook",
        unknown_pdf_settings: str | None = "/ebook",
        logging_dir: Path = Path("./logs"),
        remove_backups: bool = False,
    ):
    """Compress PDF files in bulk with GhostScript.

    Args:
        dir (Path, optional): PDFs will be found recursively in this directory.
        size_threshold (int, optional): The minimum size to perform compression.
        pdf_settings (str, optional): The PDF settings provided to GhostScript.
        logging_dir (Path, optional): The directory to save logs and results.
        remove_backups (bool, optional): Remove old backup files.
    """
    def maybe_compress(pdf_file: Path, pdf_text_type: str):
        match pdf_text_type:
            case "text":
                return compress_pdf(pdf_file, text_pdf_settings)
            case "scanned":
                return compress_pdf(pdf_file, image_pdf_settings)
            case "unknown":
                if unknown_pdf_settings:
                    return compress_pdf(pdf_file, unknown_pdf_settings)
            case _:
                raise RuntimeError(f"Invalid PDF text type: {pdf_text_type}")

    dir = dir.resolve()
    logging_dir = logging_dir.resolve()
    setup_logging(logging_dir)
    
    results = {}
    pool = ThreadPool()

    text_info_file = logging_dir / "pdf_text_info.json"
    if not text_info_file.exists():
        subprocess.check_output([
            "python", f"{Path(__file__).parent / 'scan_infer.py'}",
            f"{dir}",
            "--logging-dir", f"{logging_dir}",
        ])
    
    to_compress = list(pdfs_to_compress(dir, size_threshold, text_info_file))

    results = iter(tqdm(
        pool.imap_unordered(lambda args: maybe_compress(args[0], args[1]), to_compress),
        desc="Compress PDFs",
        total=len(to_compress),
    ))

    compressions = {}
    while True:
        try:
            pdf_file, backup_file = next(results)
            compressions[f"{pdf_file}"] = f"{backup_file}"
        except subprocess.CalledProcessError as e:
            logger.exception(f"Failed to compress {pdf_file}")
            logger.error(e.stdout.decode('utf-8'))
            logger.error(e.stderr.decode('utf-8'))
        except StopIteration:
            break

    save_results(compressions, logging_dir)
    if remove_backups:
        remove_backups_files(dir)


if __name__ == '__main__':
    typer.run(main)
