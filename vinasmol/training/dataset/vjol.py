import aiohttp
import asyncio
import json
import logging
from pathlib import Path
import random
import re
import time

from datasets import Dataset, IterableDataset, DatasetBuilder, load_dataset_builder
from requests import Session
from sickle import Sickle
from sickle.app import Record
from tqdm import tqdm

from . import DATA_DIR
# VJOL's SSL certificate is not recognized by Python HTTPS client
# https://forum.pkp.sfu.ca/t/oai-pmh-errors-and-ssl-certificate/72584/3
from .util import no_ssl_verification

Url = str
JournalId = str
RecordMetadata = dict[str, list]

MAX_CONCURRENT_DOWNLOADS = 4

VJOL_BASE_URL = "https://vjol.info.vn/index.php"
PDF_DOWNLOAD_DIR = DATA_DIR / "vjol" / "pdf"
DATASET_DIR = DATA_DIR / "vjol" / "parquet"
REPO_ID = "vjol"

logger = logging.getLogger(__name__)

class PaperProcessor:
    def __init__(self):
        self.pdf_to_md: dict[Path, Path] = {}

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
        self.session = Session()
    if self.http_method == 'GET':
        return self.session.get(self.endpoint, params=kwargs, **self.request_args)
    return self.session.post(self.endpoint, data=kwargs, **self.request_args)

class VjolDownloader:
    def __init__(
            self,
            base_url: str,
            download_dir: Path,
            records_dir: Path | None = None,
            user_agent = "vinasmol-training",
        ):
        self.base_url = base_url

        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        if records_dir is None:
            self.records_dir = self.download_dir / "records"
        else:
            self.records_dir = Path(records_dir)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        
        self.user_agent = user_agent

        self.api_urls: dict[JournalId, Url] = {}
        self.records: dict[JournalId, list[RecordMetadata]] = {}
        self.pdfs: list[Path] = []

        self.download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    
    def _new_session(self) -> Session:
        session = Session()
        session.headers['User-Agent'] = self.user_agent
        return session

    def _new_async_session(self) -> aiohttp.ClientSession:
        session = aiohttp.ClientSession()
        session.headers['User-Agent'] = self.user_agent
        return session

    def _sample_ratelimit_duration(self) -> float:
        return random.uniform(1.0, 3.0)
    
    def _scrape_journal_api_urls(self, session: Session) -> dict[JournalId, Url]:
        """List the journals made available by VJOL.
        
        Args:
            base_url (str): the VJOL base URL.
            session (Session): a requests.Session object.

        Returns:
            dict[str, str]: a dictionary that maps journal identifiers to their OAI API URLs,
                for every journal served by VJOL.
        """
        # NOTE: unfortunately GET https://vjol.info.vn/api/v1/issues/ is broken,
        # so we have to scrape the list

        r = session.get(self.base_url)
        html_content = r.text

        vjol_link_format = f'<a href="{self.base_url}/_journal_/issue/current">'
        vjol_journal_re = re.escape(vjol_link_format).replace(r'_journal_', r'(\w+)')
        vjol_journals = re.findall(vjol_journal_re, html_content)
        vjol_apis = {
            journal: f"{self.base_url}/index.php/{journal}/oai"
            for journal in vjol_journals
        }

        #other_website_re = r'<a href="([^\"]+)/issue/archive" target="_blank">'
        #external_links = re.findall(other_website_re, html_content)
        #external_apis = [f"{external_links}/oai"]

        return vjol_apis

    def download_api_urls(self) -> dict:
        """Download the VJOL journal API urls to the records directory as `api_urls.json`.
        
        Returns:
            api_urls (dict): the OAI API urls for each VJOL journal.
        """
        with no_ssl_verification():
            with self._new_session() as session:
                logger.debug("Scraping journal API urls")
                self.api_urls = self._scrape_journal_api_urls(session=session)
        
        logger.info(f"Found {len(self.api_urls)} journals available with an OAI API")

        api_urls_json = self.records_dir / "api_urls.json"
        api_urls_json.write_text(json.dumps(self.api_urls))

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

        with no_ssl_verification():
            records_iter = sickle.ListRecords(metadataPrefix='oai_dc', ignore_deleted=True)
            records: list[Record] = list(tqdm(records_iter, f"{oai_api_url} records"))

        metadata = []
        for record in records:
            m = record.metadata

            if m['language'] != ['vie']:
                # Only keep articles in Vietnamese
                logger.warning(f"Unhandled language: {m['language']}")
                continue
            if m['format'] != ['application/pdf']:
                logger.warning(f"Unhandled format: {m['format']}")
                continue

            pdf_view_url: str = m['relation']
            m['pdf_url'] = pdf_view_url.replace('view', 'download')
            metadata.append(m)
        
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
                time.sleep(self._sample_ratelimit_duration())

                self.records[journal] = VjolDownloader.get_records(api_url, session=session)
                # Backup the journal record list
                journal_records_json = self.records_dir / journal / "records.json"
                journal_records_json.parent.mkdir(parents=True, exist_ok=True)
                journal_records_json.write_text(json.dumps(self.records[journal]))
                logger.info(f"Found {len(self.records)} records for journal {journal}")
        
        return self.records

    async def _download_pdf(
            self,
            pdf_url: str,
            output_file: str | Path,
            session: aiohttp.ClientSession,
        ):
        async with self.download_semaphore:
            async with session.get(pdf_url) as response:
                with open(output_file, mode="wb") as file:
                    while True:
                        chunk = await response.content.read()
                        if not chunk:
                            break
                        file.write(chunk)

    async def download_records_as_pdf(self) -> list[Path]:
        """Downloads all of the PDFs from the downloaded VJOL records metadata.

        Returns:
            list[Path]: the list of the downloaded PDF files.
        """
        self.pdfs = []
        async with self._new_async_session() as download_session:
            tasks = []
            for journal, journal_records in tqdm(
                    self.records.items(),
                    desc="Download journal PDFs",
                ):
                for record in journal_records:
                    await asyncio.sleep(self._sample_ratelimit_duration())

                    article_url: str = record['identifier'][0]
                    article_id = article_url.split('/')[-1]
                    output_file = self.download_dir / journal / f"{article_id}.pdf"
                    self.pdfs.append(output_file)

                    record['local_pdf_file'] = output_file
                    task = self._download_pdf(record['pdf_url'], output_file, session=download_session)
                    tasks.append(task)
            
            # TODO: handle a few exceptions gracefully and log errors
            # TODO: resume from journal records JSON file

            await asyncio.gather(*tasks)
        
        return self.pdfs
    
    def download_dataset_pdfs(self) -> list[Path]:
        """Downloads all of the PDFs of the VJOL records.

        Returns:
            list[Path]: the list of the downloaded PDF files.
        """
        self.download_api_urls()
        self.download_records_metadata()
        return self.download_records_as_pdf()

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
                record['local_md_file'] = markdown_files[record['local_pdf_file']]
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
    downloader = VjolDownloader(base_url, PDF_DOWNLOAD_DIR)
    records = asyncio.run(downloader.download_dataset_pdfs())

    processor = PaperProcessor()
    markdown_files = asyncio.run(processor.convert_papers(PDF_DOWNLOAD_DIR))
    builder = downloader.compile_dataset(records, markdown_files, dataset_dir=DATASET_DIR)
    if streaming:
        return builder.as_streaming_dataset(split='train')
    else:
        return builder.as_dataset(split='train')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # TODO: upload to HuggingFace if not created, otherwise load_dataset the uploaded version
    dataset = create_and_load_vjol(base_url=VJOL_BASE_URL)
    #dataset.push_to_hub(REPO_ID, private=True)