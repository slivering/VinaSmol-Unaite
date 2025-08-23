import asyncio
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetBuilder, load_dataset_builder
from loguru import logger
from tqdm import tqdm
import typer

from . import DATA_DIR, PDF_DOWNLOAD_DIR, JournalId, RecordMetadata
from .resource import ListResource

OUT_DATASET_DIR = DATA_DIR / "vjol" / "parquet"
REPO_ID = "vjol"

@dataclass
class ConversionResult:
    input_pdf_file: Path
    output_md_file: Path
    success: bool

class PaperProcessor:
    def __init__(self):
        self.results: list[ConversionResult] = []
    
    def postprocess_md(self, md_content: str) -> str:
        """Clean the markdown content to remove superfluous items."""
        raise NotImplementedError

    async def convert_pdf(
            self,
            pdf_file: str | Path,
            output_md_file: str | Path,
        ) -> ConversionResult:
        """Process a PDF paper using scienceparse.

        Args:
            pdf_file (str | Path): the path to the downloaded paper as a PDF file.
            output_md_file (str | Path): the file path to write Markdown content.
        Returns:
            ConversionResult: whether the PDF file was successfully converted.
        """
        # TODO: conversion should depend on whether the PDF is a whole issue or a single article
        # issues: remove title page with TOC
        raise NotImplementedError

    async def convert_papers(self, download_dir: str | Path) -> list[ConversionResult]:
        """Process PDF papers in bulk using scienceparse.

        Args:
            download_dir (str | Path): the directory that contains the downloaded PDFs.

        Returns:
            results (list[ConversionResult]): the conversion results.
        """
        download_dir = Path(download_dir)

        # TODO: multiprocessing with pool
        tasks = []
        for pdf_file in download_dir.glob("*.pdf"):
            md_file = download_dir / pdf_file.name.replace(".pdf", ".md")
            task = self.convert_pdf(pdf_file, md_file)
            tasks.append(task)

        results: list[ConversionResult] = await asyncio.gather(*tasks)
        self.results.extend(results)

        return results

def compile_dataset(
        records: dict[JournalId, ListResource[RecordMetadata]],
        markdown_files: dict[Path, Path],
        dataset_dir: Path,
    ) -> DatasetBuilder:
    """Compile a Parquet dataset into a directory.

    Args:
        markdown_files (dict[Path, Path]): mapping from PDF files to Markdown files.
        dataset_dir (Path): the output directory for the Parquet files.
    
    Returns:
        DatasetBuilder: a builder of the compiled Parquet dataset files.
    """

    def gen_journal_records(journal: JournalId, journal_records: list[RecordMetadata]):
        for record in journal_records:
            if 'local_pdf_file' not in record:
                # PDF was not downloaded yet
                continue
            local_pdf = record['local_pdf_file']
            if local_pdf not in markdown_files:
                # PDF was not converted yet
                continue
            record['local_md_file'] = markdown_files[local_pdf]
            record['journal_id'] = journal
            record['text'] = record['local_md_file'].read_text()
            #del record['local_pdf_file']
            #del record['local_md_file']
            yield record

    for journal, journal_records in tqdm(records.items(), desc="Compile journals"):
        journal_ds = Dataset.from_generator(
            gen_journal_records,
            gen_kwargs=dict(journal=journal, journal_records=journal_records),
        )
        journal_ds.to_parquet(dataset_dir / f"{journal}.parquet")

    return load_dataset_builder(
        'text',
        data_files=OUT_DATASET_DIR / "*.parquet",
    )

    

def process_papers(processor: PaperProcessor, pdf_dir: Path) -> dict[Path, Path]:
    # TODO: multiprocessing instead
    results = asyncio.run(processor.convert_papers(pdf_dir))

    pdf_to_md = {}
    successes = []
    failures = []
    for result in results:
        if result.success:
            pdf_to_md[result.input_pdf_file] = result.output_md_file
            successes.append(result)
        else:
            failures.append(result)
    
    logger.info("{} PDFs successfully converted to Markdown.", len(successes))
    logger.warning("{} PDFs couldn't be converted.", len(failures))
    return pdf_to_md

def compile_and_load_vjol(
        records: dict[JournalId, ListResource[RecordMetadata]],
        pdf_to_md: dict[Path, Path],
        dataset_dir: Path = OUT_DATASET_DIR,
        streaming: bool = True,
    ):
    builder = compile_dataset(records, pdf_to_md, dataset_dir=dataset_dir)
    if streaming:
        return builder.as_streaming_dataset(split='train')
    else:
        return builder.as_dataset(split='train')


app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
        pdf_dir: Path = PDF_DOWNLOAD_DIR,
        #out_dataset_dir: Path = OUT_DATASET_DIR,
        #repo_id: str = REPO_ID,
    ):
      
    processor = PaperProcessor()
    pdf_to_md = process_papers(processor, pdf_dir)

    # TODO: upload to HuggingFace if not created, otherwise load_dataset the uploaded version
    #downloader = VjolDownloader(...)
    #dataset = compile_and_load_vjol(downloader.records, pdf_to_md, streaming=True)
    #dataset.push_to_hub(repo_id, private=True)