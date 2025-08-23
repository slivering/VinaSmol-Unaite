from dataclasses import dataclass
from urllib.parse import urlparse, urlsplit
import aiohttp
import asyncio
import logging
from loguru import logger
import os
from pathlib import Path
import random
import re
import sys
import time

from datasets import Dataset, DatasetBuilder, load_dataset_builder
import requests
from requests import Session
import rich
from sickle import Sickle
from sickle.app import Record
from sickle.oaiexceptions import NoRecordsMatch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import typer

from . import DATA_DIR

from .util import (
    DictResource,
    ListResource,
    LicenseResult,
    cc_right_in_licenses,
    auto_patch_ssl_verification,
    auto_patch_ssl_verification_async,
    DownloadResult,
    DownloadTracker,
)

Url = str
JournalId = str
RecordMetadata = dict[str, list]

SIMULTANEOUS_DOWNLOADS = 4
USER_AGENT = "vinasmol-training"

VJOL_BASE_URL = "https://vjol.info.vn/index.php"
LOGFILE = DATA_DIR / "vjol" / "vjol.log"
RECORDS_DOWNLOAD_DIR = DATA_DIR / "vjol" / "records"
PDF_DOWNLOAD_DIR = DATA_DIR / "vjol" / "pdf"
OUT_DATASET_DIR = DATA_DIR / "vjol" / "parquet"
REPO_ID = "vjol"

_LICENSE_RE = re.compile(
    r'"(https?://creativecommons\.org/(?:licenses|publicdomain)/(?:[a-z0-9/\.-])+)"'
)


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
    
def _Sickle_request(self, kwargs):
    """Monkeypatched function for Sickle._request."""
    if not hasattr(self, 'session'):
        logger.warning("Missing session attribute, creating new session for Sickle")
        self.session = Session()
    if self.http_method == 'GET':
        return self.session.get(self.endpoint, params=kwargs, **self.request_args)
    return self.session.post(self.endpoint, data=kwargs, **self.request_args)


class VjolDownloader:
    def __init__(
            self,
            base_url: str,
            pdf_dir: Path,
            records_dir: Path = RECORDS_DOWNLOAD_DIR,
            simultaneous_downloads: int = SIMULTANEOUS_DOWNLOADS,
            user_agent: str = "vinasmol-training",
            force_refresh_records: bool = False,
        ):
        self.base_url = base_url

        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir = Path(records_dir)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        
        self.simultaneous_downloads = simultaneous_downloads
        self.user_agent = user_agent
        self.force_refresh_records = force_refresh_records

        self.api_urls: DictResource[JournalId, Url] = DictResource(
            file_path = self.records_dir / "api_urls.json"
        )
        self.records: dict[JournalId, ListResource[RecordMetadata]] = {}
        self.pdfs: list[Path] = []
        self.licenses: DictResource[Url, list[Url]] = DictResource(
            file_path = self.records_dir / "licenses.json"
        )
    
    def _new_session(self) -> Session:
        session = Session()
        session.headers['User-Agent'] = self.user_agent
        return session

    def _new_async_session(self) -> aiohttp.ClientSession:
        session = aiohttp.ClientSession()
        session.headers['User-Agent'] = self.user_agent
        return session

    def _sample_ratelimit_duration(self) -> float:
        return random.uniform(0.25, 1.0)
    
    def _scrape_journal_api_urls(self, session: Session) -> dict[JournalId, Url]:
        """List the journals made available by VJOL and participying publishers.
        
        Args:
            base_url (str): the VJOL base URL.
            session (Session): a requests.Session object.

        Returns:
            dict[str, str]: a dictionary that maps journal identifiers to their OAI API URLs,
                for every journal served by VJOL or participying publishers.
        """
        # NOTE: unfortunately GET https://vjol.info.vn/api/v1/issues/ is broken,
        # so we have to scrape the list
        with auto_patch_ssl_verification(self.base_url, session):
            r = session.get(self.base_url)
        
        html_content = r.text

        # https://vjol.info.vn/index.php/index/oai is a superset of all the journals on VJOL
        # FIXME: this creates a mega-journal which should be split later on
        vjol_apis = {
            "vjol.info.vn_index": f"{self.base_url}/index/oai"
        }
        
        # Sometimes VJOL links to other websites with a different journal identifier
        other_website_re = r'<a href="([^\"]+)/issue/archive" target="_blank">'
        external_links = re.findall(other_website_re, html_content)

        def extract_journal_identifier(url: str) -> str:
            split = urlsplit(url)
            domain = split.netloc
            components = split.path.split('/')[1:]
            match components:
                case ["en" | "vi" | "vn"] | ["index.php", "en" | "vi" | "vn"] | ["index.php"]:
                    return domain
                case [identifier] | ["index.php", identifier]:
                    return f"{domain}_{identifier}"
                case _:
                    return domain

        external_apis = {
            extract_journal_identifier(url): f"{url}/oai"
            for url in external_links
        }
        external_domains = [urlsplit(url).netloc for url in external_links]
        logger.info(
            "Will make requests to additional domains: {}",
            ', '.join(external_domains),
        )

        return vjol_apis | external_apis

    def download_api_urls(self) -> DictResource[JournalId, Url]:
        """Download the VJOL journal API urls to the records directory as `api_urls.json`.
        
        Returns:
            api_urls (JSONDictResource): the OAI-PMH API urls for each VJOL journal.
        """
        fp = self.api_urls.file_path
        if self.api_urls and not self.force_refresh_records:
            logger.info("API URLs already downloaded at {}", fp)
            return self.api_urls

        rich.print(
            f"Creating or refreshing {fp.name}. This script will retrieve all journals "
            "from https://vjol.info.vn which is an open-access Vietnamese library provider.",
            "[red]However, some journals on VJOL do not specify an explicit CC license.\n",
            "Consider running the notebook ./doaj_vn.ipynb first to download "
            "API urls of [green]permissively-licensed journals[/green] "
            "indexed by https://doaj.org ."
        )
        if not typer.confirm("Continue?"):
            raise SystemExit
        
        with self._new_session() as session:
            logger.debug("Scraping journal API urls")
            self.api_urls.update(self._scrape_journal_api_urls(session=session))
        
        logger.info("Found {} journals available with an OAI API", len(self.api_urls))
        self.api_urls.save()
        return self.api_urls

    @classmethod
    def get_records(cls, oai_api_url: Url, session: Session) -> list[RecordMetadata]:
        """Get the metadata for every journal record in VJOL.

        Args:
            oai_api_url (Url): the OAI base URL.
            session (Session): a requests.Session object.

        Returns:
            RecordMetadata (list[dict[str, list]]): the list of metadata associated to the records.
        """
        sickle = Sickle(oai_api_url)
        sickle.session = session
        Sickle._request = _Sickle_request

        # NOTE: context manager is safe since get_records is not executed async
        with auto_patch_ssl_verification(oai_api_url, session):
            try:
                #journal_name = sickle.Identify().repositoryName
                records_iter = sickle.ListRecords(metadataPrefix='oai_dc', ignore_deleted=True)
            except NoRecordsMatch:
                logger.error(
                    "No matching records in {}.",
                    oai_api_url
                )
                return []
            except requests.exceptions.RequestException:
                logger.exception(
                    "Failed to list records at {}, see traceback below.",
                    oai_api_url
                )
                return []

            oai_records: list[Record] = list(tqdm(records_iter, f"{oai_api_url} records"))

        metadata = []
        file_missing = []
        for record in oai_records:
            meta = record.metadata
            record_id = meta['identifier'][0]

            if 'relation' not in meta:
                file_missing.append(record_id.split('/')[-1])
                continue
            meta['relation'] = [
                url for url in meta['relation']
                if "downloadSuppFile" not in url
            ]

            if 'language' not in meta:
                # Language attribute is not reliable, so don't use it for filtering
                logger.warning("Missing language attribute for {}", record_id)
            if 'format' not in meta:
                logger.warning("Missing format attribute for {}", record_id)
            elif "application/pdf" not in meta['format']:
                # Word and HTML are rare so we discard them for simplicity
                logger.info(
                    "Dropping record {} due to unhandled format: {}",
                    record_id,
                    meta['format'],
                )
                continue
            elif len(meta['format']) != len(meta['relation']) and len(meta['format']) > 1:
                logger.error(
                    "Record {} is invalid due to mismatching length: "
                    "format is {} but relation is {}",
                    record_id,
                    meta['format'], meta['relation'],
                )
                continue

            # Take the latest revision in priority, if it's a URL
            # Sometimes meta['relation'] contains the DOI
            for i in range(len(meta['relation']) - 1, -1, -1):
                if 'view' in meta['relation'][i]:
                    meta['view_url'] = meta['relation'][i]
                    meta['download_url'].replace('view', 'download')
                    break
            if 'download_url' not in meta:
                logger.error(
                    "Record {} has no valid PDF download link in {}",
                    record_id, meta['relation'],
                )
                continue

            # if "vjol" in oai_api_url:
            #     journal = meta['identifier'][0].split('/')[-4]

            metadata.append(meta)
        
        if file_missing:
            logger.warning(
                "Dropped {} records without a download link: {}",
                len(file_missing), file_missing,
            )
        
        num_skipped = len(oai_records) - len(metadata)
        if num_skipped > 0:
            logger.info(
                "Skipped {} records due to incomplete or invalid metadata or invalid format",
                num_skipped,
            )

        return metadata

    def download_records_metadata(self) -> dict[JournalId, ListResource[RecordMetadata]]:
        """Get all of the records available on VJOL and downloads their metadata.

        Returns:
            records (dict[JournalId, list[RecordMetadata]]): the records by journal.
        """
        with self._new_session() as session:
            # TODO: get records in parallel
            for journal, api_url in tqdm(
                    self.api_urls.items(),
                    desc="Get all records for each journal",
                ):
                self.records[journal] = ListResource(
                    file_path = self.records_dir / journal / "records.json"
                )

                if self.records[journal] and not self.force_refresh_records:
                    logger.info(
                        "Records for {} already downloaded at {}",
                        journal, self.records[journal].file_path,
                    )
                else:
                    time.sleep(self._sample_ratelimit_duration())
                    
                    journal_records = VjolDownloader.get_records(api_url, session=session)
                    self.records[journal].load_with(journal_records)
                    self.records[journal].save()

                if not self.records[journal]:
                    logger.info(
                        "Journal {} won't be included in the dataset as it has no records",
                        journal,
                    )
                    del self.records[journal]
                else:
                    logger.info(
                        "Found {} records for journal {}",
                        len(self.records[journal]), journal,
                    )
        
        num_total_records = sum(len(part) for part in self.records.values())
        num_journals = len(self.records)
        logger.info(
            "Found a total of {} records across all {} journals",
            num_total_records, num_journals,
        )
        return self.records

    async def _find_licenses(
            self,
            view_url: str,
            session: aiohttp.ClientSession,
            semaphore: asyncio.BoundedSemaphore,
        ) -> LicenseResult:
        """Scrape a CreativeCommons license from a publication page.

        Args:
            view_url (str): the article view URL, ends with `/article/view/xxx/yyy`.

        Returns:
            LicenseResult: the list of extracted licenses from the HTML page.
        """
        async with semaphore:
            await asyncio.sleep(self._sample_ratelimit_duration())
            logger.debug("Checking record license at {}", view_url)
            async with session.get(view_url) as response:
                response.raise_for_status()
                html_content = await response.text()
                licenses = _LICENSE_RE.findall(html_content)
                return LicenseResult(url=view_url, licenses=licenses)
        
    async def _download_pdf(
            self,
            download_url: str,
            output_file: str | Path,
            session: aiohttp.ClientSession,
            semaphore: asyncio.BoundedSemaphore,
        ) -> DownloadResult:
        output_file = Path(output_file)
        if output_file.exists() and os.stat(output_file).st_size > 0:
            # Skip item when resuming downloads
            return DownloadResult(
                success=True,
                url=download_url,
                already_downloaded=True,
                file_path=output_file,
            )

        async with semaphore:
            await asyncio.sleep(self._sample_ratelimit_duration())
            logger.debug("Downloading {}", download_url)
            async with session.get(download_url) as response:
                response.raise_for_status()
                
                content_type = response.headers.get("Content-Type")
                if content_type != "application/pdf":
                    logger.error(
                        "Specified content type for {} is not application/pdf but {}",
                        download_url, content_type,
                    )
                    return DownloadResult(
                        success=False,
                        url=download_url,
                        content_type=content_type,
                    )

                output_file.parent.mkdir(exist_ok=True, parents=True)

                content_length = 0
                try:
                    with open(output_file, mode="wb") as file:
                        for i in range(1_000_000):
                            chunk = await response.content.read()
                            if not chunk:
                                break
                            if i == 0 and not chunk.startswith(b"%PDF"):
                                logger.error(
                                    "Expected PDF content for {} but got invalid file header: {},"
                                    " aborting download",
                                    download_url, chunk[:4],
                                )
                                break
                            file.write(chunk)
                            content_length += len(chunk)
                
                except Exception as e:
                    # Delete the partially downloaded file so it will be re-downloaded after
                    output_file.unlink()
                    raise e

                if i == 0:
                    # Download aborted, remove file
                    output_file.unlink()

        return DownloadResult(
            success=True,
            url=download_url,
            content_type=content_type,
            file_path=output_file,
            content_length=content_length,
        )

    async def get_available_licenses(self) -> dict[Url, LicenseResult]:
        """Determine the licenses for all records.

        Returns:
            list[LicenseResult]: the list of license results.

        # Notes

        The CC license for an article is scraped from its HTML webpage. If it's not found,
        the journal's license (which is compatible if indexed by DOAJ) is assumed to hold.
        """
        for journal, journal_records in tqdm(
                self.records.items(),
                desc="Checking individual paper licenses for each journal",
                unit="journals",
            ):
            async with self._new_async_session() as download_session:
                sample_url: Url = self.api_urls[journal]
                domain = urlparse(sample_url).hostname
                async with auto_patch_ssl_verification_async(sample_url, download_session):
                    # Async downloads for a single journal
                    tasks = []
                    semaphore = asyncio.BoundedSemaphore(self.simultaneous_downloads)
                    for record in journal_records:
                        article_view_url: str = record['view_url']
                        if article_view_url in self.licenses:
                            # License is already downloaded
                            continue
                        tasks.append(self._find_licenses(
                            article_view_url,
                            download_session,
                            semaphore,
                        ))

                    # In case user hits Ctrl+C
                    logging.getLogger('asyncio').addFilter(
                        lambda record: not record.getMessage().startswith(
                            "Task was destroyed but it is pending!"
                        )
                    )
                    results = await DownloadTracker.gather(
                        *tasks,
                        return_exceptions=True,
                        desc=f"Checking individual paper licenses from {journal} ({domain})",
                        unit="papers",
                    )
                    for result in results:
                        if isinstance(result, LicenseResult):
                            self.licenses[result.url] = result.licenses
                        else:
                            assert isinstance(result, Exception)
                            logger.error("Error scraping license:\n {}", result)
                    
                    for record in tqdm(self.records[journal], desc="Merging licenses"):
                        licenses = self.licenses.get(record['download_url']) or []

                        # Find additional licenses in metadata
                        cc_in_meta = []
                        if 'description' in record and record['description'][0]:
                            if cc_desc := _LICENSE_RE.search(record['description'][0]):
                                cc_in_meta.append(cc_desc)
                        
                        for right in record.get('rights', []):
                            if right.startswith('http'):
                                cc_in_meta.append(right)
                        
                        for adt_license in cc_in_meta:
                            if not cc_right_in_licenses(adt_license, licenses):
                                licenses.append(adt_license)
                        
                        self.licenses[record['download_url']] = licenses
                        record['license'] = licenses
                        self.records[journal].save()
                        self.licenses.save()
            
            self.licenses.save()

        num_incompatible = sum(
            not LicenseResult(url, license).is_acceptable()
            for url, license in self.licenses.items()
        )
        logger.info("Found {} incompatible licenses in total", num_incompatible)
        return self.licenses

    async def download_records_as_pdf(self) -> list[Path]:
        """Downloads all of the PDFs from the downloaded VJOL records metadata.

        Returns:
            list[Path]: the list of the downloaded PDF files.
        """
        self.pdfs = []
        for journal, journal_records in tqdm(
                self.records.items(),
                desc="Download journal PDFs",
                unit="journals",
            ):
            async with self._new_async_session() as download_session:
                sample_url = self.api_urls[journal]
                domain = urlparse(sample_url).hostname
                async with auto_patch_ssl_verification_async(sample_url, download_session):
                    # Async downloads for a single journal
                    tasks = []
                    semaphore = asyncio.BoundedSemaphore(self.simultaneous_downloads)
                    for record in journal_records:
                        view_url: str = record['identifier'][0]
                        license: Url | None = record['license']
                        if license and not LicenseResult(view_url, license).is_acceptable():
                            logger.debug(
                                "Not downloading article {} due to incompatible license: {}",
                                view_url, license,
                            )
                            continue
                        
                        article_number = view_url.split('/')[-1]
                        output_file = self.pdf_dir / journal / f"{article_number}.pdf"
                        record['local_pdf_file'] = output_file
                        tasks.append(self._download_pdf(
                            record['download_url'],
                            output_file,
                            download_session,
                            semaphore,
                        ))

                    # In case user hits Ctrl+C
                    logging.getLogger('asyncio').addFilter(
                        lambda record: not record.getMessage().startswith(
                            "Task was destroyed but it is pending!"
                        )
                    )
                    results = await DownloadTracker.gather(
                        *tasks,
                        return_exceptions=True,
                        desc=f"Downloading from {journal} ({domain})",
                        unit="papers",
                    )
                    self._collect_pdfs(results, journal)

        return self.pdfs
    
    def _collect_pdfs(self, results: list[DownloadResult | Exception], journal: JournalId):
        """Collect the successfully downloaded PDFs into `self.pdfs`."""
        exceptions = []
        not_exceptions = []
        already_downloaded = []
        not_success = []
        num_success = 0
        for result in results:
            if isinstance(result, Exception):
                exceptions.append(result)
            elif isinstance(result, DownloadResult):
                not_exceptions.append(result)
                if result.already_downloaded:
                    already_downloaded.append(result)
                if not result.success:
                    not_success.append(result)
                else:
                    assert result.file_path is not None
                    self.pdfs.append(result.file_path)
                    num_success += 1
            else:
                raise RuntimeError(f"Invalid result: {type(result)}")
        
        logger.info("{} PDFs downloaded for journal {}", num_success, journal)
        if exceptions:
            logger.error("{} exceptions below:", len(exceptions))
            for (i, exception) in enumerate(exceptions):
                logger.error("Exception #{}: {}", i, exception)
        if already_downloaded:
            logger.info("{} PDFs already downloaded", len(already_downloaded))
        if not_success:
            logger.error("{} files with incorrect content type", len(not_success))
    
    def download_dataset_pdfs(self) -> list[Path]:
        """Downloads all of the PDFs of the VJOL records.

        Returns:
            list[Path]: the list of the downloaded PDF files.
        """
        logger.info("Starting paper harvesting.")
        self.download_api_urls()
        self.download_records_metadata()
        asyncio.run(self.get_available_licenses())
        # TODO: re-save self.records to each journal records file
        return asyncio.run(self.download_records_as_pdf())

    def compile_dataset(
            self,
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

        for journal, journal_records in tqdm(self.records.items(), desc="Compile journals"):
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
        downloader: VjolDownloader,
        pdf_to_md: dict[Path, Path],
        dataset_dir: Path = OUT_DATASET_DIR,
        streaming: bool = True,
    ):
    builder = downloader.compile_dataset(pdf_to_md, dataset_dir=dataset_dir)
    if streaming:
        return builder.as_streaming_dataset(split='train')
    else:
        return builder.as_dataset(split='train')

def setup_logging(logfile: Path):
    logger.level(name="DEBUG", color="<fg 30>")
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format="<green>{time}</green> <level>{message}</level>",
    )
    logger.configure(handlers=[dict(sink=lambda msg: tqdm.write(msg, end=''), colorize=True)])
    logger.add(logfile, format="{time} {level} {message}", level="DEBUG", rotation="100 MB")


app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
        pdf_dir: Path = PDF_DOWNLOAD_DIR,
        records_dir: Path = RECORDS_DOWNLOAD_DIR,
        out_dataset_dir: Path = OUT_DATASET_DIR,
        repo_id: str = REPO_ID,
        simultaneous_downloads: int = SIMULTANEOUS_DOWNLOADS,
        logfile: Path = LOGFILE,
        user_agent: str = USER_AGENT,
        force_refresh_records: bool = False,
    ):
    setup_logging(logfile)

    downloader = VjolDownloader(
        base_url=VJOL_BASE_URL,
        pdf_dir=pdf_dir,
        records_dir=records_dir,
        simultaneous_downloads=simultaneous_downloads,
        user_agent=user_agent,
        force_refresh_records=force_refresh_records,
    )
    with logging_redirect_tqdm():
        downloader.download_dataset_pdfs()

        processor = PaperProcessor()
        pdf_to_md = process_papers(processor, pdf_dir)

    # TODO: upload to HuggingFace if not created, otherwise load_dataset the uploaded version
    dataset = compile_and_load_vjol(downloader, pdf_to_md, streaming=True)
    #dataset.push_to_hub(repo_id, private=True)

if __name__ == '__main__':
    app()
