import dataclasses
from urllib.parse import urlsplit
import aiohttp
import asyncio
import json
import logging
from pathlib import Path
import random
import re
import time

from datasets import Dataset, IterableDataset, DatasetBuilder, load_dataset_builder
import requests
from requests import Session
from sickle import Sickle
from sickle.app import Record
from sickle.oaiexceptions import NoRecordsMatch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from . import DATA_DIR

# VJOL's SSL certificate is not recognized by Python HTTPS client
from .util import no_ssl_verification

Url = str
JournalId = str
RecordMetadata = dict[str, list]

MAX_SIMULTANEOUS_DOWNLOADS = 4

VJOL_BASE_URL = "https://vjol.info.vn/index.php"
LOGFILE = DATA_DIR / "vjol" / "vjol.log"
RECORDS_DOWNLOAD_DIR = DATA_DIR / "vjol" / "records"
PDF_DOWNLOAD_DIR = DATA_DIR / "vjol" / "pdf"
DATASET_DIR = DATA_DIR / "vjol" / "parquet"
REPO_ID = "vjol"

logger = logging.getLogger(__name__)

class PaperProcessor:
    def __init__(self):
        self.pdf_to_md: dict[Path, Path] = {}
    
    def postprocess_md(self, md_content: str) -> str:
        """Clean the markdown content to remove superfluous items."""
        raise NotImplementedError

    async def convert_pdf(self, pdf_file: str | Path, output_md_file: str | Path):
        """Process a PDF paper using scienceparse.

        Args:
            pdf_file (str | Path): the path to the downloaded paper as a PDF file.
            output_md_file (str | Path): the file path to write Markdown content.
        """
        raise NotImplementedError

    async def convert_papers(self, download_dir: str | Path) -> dict[Path, Path]:
        """Process PDF papers in bulk using scienceparse.

        Args:
            download_dir (str | Path): The directory that contains the downloaded PDFs.

        Returns:
            dict[Path, Path]: a mapping from the original PDF to its processed Markdown content.
        """
        download_dir = Path(download_dir)
        # TODO: in bulk instead of per file:
        # java -Xmx6g -jar ./science-parse-cli-assembly-2.0.3.jar download_dir

        # TODO: multiprocessing with pool
        tasks = []
        for pdf_file in download_dir.glob("*.pdf"):
            md_file = download_dir / pdf_file.name.replace(".pdf", ".md")
            self.pdf_to_md[pdf_file] = md_file

            task = self.convert_pdf(pdf_file, md_file)
            tasks.append(task)

        await asyncio.gather(*tasks)
        return self.pdf_to_md
    
def _Sickle_request(self, kwargs):
    """Monkeypatched function for Sickle._request."""
    if not hasattr(self, 'session'):
        logger.warning("Missing session attribute, creating new session for Sickle")
        self.session = Session()
    if self.http_method == 'GET':
        return self.session.get(self.endpoint, params=kwargs, **self.request_args)
    return self.session.post(self.endpoint, data=kwargs, **self.request_args)

@dataclasses.dataclass()
class DownloadResult:
    success: bool
    file_path: Path | None = None
    content_type: str | None = None
    already_downloaded: bool = False

class VjolDownloader:
    def __init__(
            self,
            base_url: str,
            pdf_dir: Path,
            records_dir: Path | None = None,
            user_agent: str = "vinasmol-training",
            force_refresh_records: bool = False,
        ):
        self.base_url = base_url

        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        if records_dir is None:
            self.records_dir = self.pdf_dir / "records"
        else:
            self.records_dir = Path(records_dir)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        
        self.user_agent = user_agent
        self.force_refresh_records = force_refresh_records

        self.api_urls: dict[JournalId, Url] = {}
        self.records: dict[JournalId, list[RecordMetadata]] = {}
        self.pdfs: list[Path] = []
    
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

        r = session.get(self.base_url)
        html_content = r.text

        # https://vjol.info.vn/index.php/index/oai is a superset of all the journals on VJOL
        vjol_apis = {
            "index": f"{self.base_url}/index/oai"
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
            "Will make requests to additional domains: %s",
            ', '.join(external_domains),
        )

        return vjol_apis | external_apis

    def download_api_urls(self) -> dict:
        """Download the VJOL journal API urls to the records directory as `api_urls.json`.
        
        Returns:
            api_urls (dict): the OAI API urls for each VJOL journal.
        """
        # NOTE: context manager is safe since download_api_urls is not executed async
        with no_ssl_verification():
            with self._new_session() as session:
                logger.debug("Scraping journal API urls")
                self.api_urls = self._scrape_journal_api_urls(session=session)
        
        logger.info("Found %d journals available with an OAI API", len(self.api_urls))

        api_urls_json = self.records_dir / "api_urls.json"
        api_urls_json.write_text(json.dumps(self.api_urls, indent=2))

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
        with no_ssl_verification():
            try:
                #journal_name = sickle.Identify().repositoryName
                records_iter = sickle.ListRecords(metadataPrefix='oai_dc', ignore_deleted=True)
            except NoRecordsMatch:
                logger.error(
                    "No matching records in %s.",
                    oai_api_url
                )
                return []
            except requests.exceptions.RequestException:
                logger.exception(
                    "Failed to list records at %s, see traceback below.",
                    oai_api_url
                )
                return []

            records: list[Record] = list(tqdm(records_iter, f"{oai_api_url} records"))

        metadata = []
        file_missing = []
        for record in records:
            meta = record.metadata
            record_id = meta['identifier'][0]

            if 'relation' not in meta:
                file_missing.append(record_id.split('/')[-1])
                continue
            if 'language' not in meta:
                # Language attribute is not reliable, so don't use it for filtering
                logger.warning("Missing language attribute for %s", record_id)
            if 'format' not in meta:
                logger.warning("Missing format attribute for %s", record_id)
            elif "application/pdf" not in meta['format']:
                # Word and HTML are rare so we discard them for simplicity
                logger.warning(
                    "Dropping record %s due to unhandled format: %s",
                    record_id,
                    meta['format'],
                )
                continue
            elif len(meta['format']) != len(meta['relation']) and len(meta['format']) > 1:
                logger.error(
                    "Record %s is invalid due to mismatching length: "
                    "format is %s but relation is %s",
                    record_id,
                    meta['format'], meta['relation'],
                )
                continue

            # Take the latest revision
            meta['pdf_url'] = meta['relation'][-1].replace('view', 'download')

            metadata.append(meta)
        
        logger.warning(
            "Dropped %d records without a download link: %s",
            len(file_missing), file_missing,
        )
        
        num_skipped = len(records) - len(metadata)
        if num_skipped > 0:
            logger.warning(
                "Skipped %d records due to incomplete or invalid metadata or invalid format",
                num_skipped,
            )

        return metadata

    def download_records_metadata(self) -> dict[JournalId, list[RecordMetadata]]:
        """Get all of the records available on VJOL and downloads their metadata.

        Returns:
            records (dict[JournalId, list[RecordMetadata]]): the records by journal.
        """
        with self._new_session() as session:
            for journal, api_url in tqdm(
                    self.api_urls.items(),
                    desc="Get all records for each journal",
                ):
                journal_records_json = self.records_dir / journal / "records.json"

                if journal_records_json.exists() and not self.force_refresh_records:
                    logger.info(
                        "Records for %s already downloaded, loading %s",
                        journal, journal_records_json,
                    )
                    self.records[journal] = json.loads(journal_records_json.read_text())
                else:
                    time.sleep(self._sample_ratelimit_duration())
                    self.records[journal] = VjolDownloader.get_records(api_url, session=session)
                    journal_records_json.parent.mkdir(parents=True, exist_ok=True)
                    journal_records_json.write_text(json.dumps(self.records[journal], indent=2))

                if not self.records[journal]:
                    logger.info(
                        "Journal %s won't be included in the dataset as it has no records",
                        journal,
                    )
                    del self.records[journal]
                else:
                    logger.info(
                        "Found %d records for journal %s",
                        len(self.records[journal]), journal,
                    )
        
        num_total_records = sum(len(part) for part in self.records.values())
        num_journals = len(self.records)
        logger.info(
            "Found a total of %d records across all %d journals",
            num_total_records, num_journals,
        )
        return self.records

    async def _download_pdf(
            self,
            pdf_url: str,
            output_file: str | Path,
            session: aiohttp.ClientSession,
            semaphore: asyncio.BoundedSemaphore,
        ) -> DownloadResult:
        output_file = Path(output_file)
        if output_file.exists():
            # Skip item when resuming downloads
            return DownloadResult(success=True, already_downloaded=True, file_path=output_file)

        async with semaphore:
            async with session.get(pdf_url) as response:
                content_type = response.headers.get("Content-Type")
                if content_type != "application/pdf":
                    logger.error(
                        "Content type is not application/pdf but %s, this should not happen",
                        content_type,
                    )
                    return DownloadResult(success=False, content_type=content_type)

                with open(output_file, mode="wb") as file:
                    while True:
                        chunk = await response.content.read()
                        if not chunk:
                            break
                        file.write(chunk)

        return DownloadResult(success=True, content_type=content_type, file_path=output_file)

    async def _download_worker(self, queue: asyncio.Queue):
        pdf_url, output_file, session = await queue.get()
        await asyncio.sleep(self._sample_ratelimit_duration())
        result = await self._download_pdf(pdf_url, output_file, session)
        return result

    async def download_records_as_pdf(self) -> list[Path]:
        """Downloads all of the PDFs from the downloaded VJOL records metadata.

        Returns:
            list[Path]: the list of the downloaded PDF files.
        """
        self.pdfs = []
        async with self._new_async_session() as download_session:
            for journal, journal_records in tqdm(
                    self.records.items(),
                    desc="Download journal PDFs",
                ):

                # Async downloads for a single journal
                tasks = []
                semaphore = asyncio.BoundedSemaphore(MAX_SIMULTANEOUS_DOWNLOADS)
                for record in journal_records:
                    article_url: str = record['identifier'][0]
                    article_id = article_url.split('/')[-1]
                    output_file = self.pdf_dir / journal / f"{article_id}.pdf"
                    record['local_pdf_file'] = output_file
                    tasks.append(self._download_pdf(
                        record['pdf_url'],
                        output_file,
                        download_session,
                        semaphore,
                    ))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                self.collect_pdfs(results, journal)

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
        
        logger.info("%d PDFs downloaded for journal %s", num_success, journal)
        if exceptions:
            logger.error("%d exceptions below:", len(exceptions))
            for (i, exception) in enumerate(exceptions):
                logger.error("Exception #%d: %s", i, exception)
        if already_downloaded:
            logger.info("%d PDFs already downloaded", len(already_downloaded))
        if not_success:
            logger.error("%d files with incorrect content type", len(not_success))
    
    def download_dataset_pdfs(self) -> list[Path]:
        """Downloads all of the PDFs of the VJOL records.

        Returns:
            list[Path]: the list of the downloaded PDF files.
        """
        logger.warning(
            "Starting paper harvesting. This will make insecure requests to %s.",
            self.base_url,
        )
        self.download_api_urls()
        self.download_records_metadata()
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
                local_pdf = record['local_pdf_file']
                if local_pdf not in markdown_files:
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
                gen_kwargs=dict(journal=journal, jounal_records=journal_records),
            )
            journal_ds.to_parquet(dataset_dir / f"{journal}.parquet")

        return load_dataset_builder(
            'text',
            data_files=DATASET_DIR / "*.parquet",
        )


def create_and_load_vjol(base_url: str, streaming: bool = True) -> Dataset | IterableDataset:
    downloader = VjolDownloader(
        base_url,
        pdf_dir=PDF_DOWNLOAD_DIR,
        records_dir=RECORDS_DOWNLOAD_DIR,
    )
    with logging_redirect_tqdm():
        downloader.download_dataset_pdfs()

    processor = PaperProcessor()
    markdown_files = asyncio.run(processor.convert_papers(PDF_DOWNLOAD_DIR))
    builder = downloader.compile_dataset(markdown_files, dataset_dir=DATASET_DIR)
    if streaming:
        return builder.as_streaming_dataset(split='train')
    else:
        return builder.as_dataset(split='train')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename=LOGFILE)
    # TODO: upload to HuggingFace if not created, otherwise load_dataset the uploaded version
    dataset = create_and_load_vjol(base_url=VJOL_BASE_URL)
    #dataset.push_to_hub(REPO_ID, private=True)