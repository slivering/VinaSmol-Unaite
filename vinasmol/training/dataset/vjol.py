import aiohttp
import asyncio
import json
import logging
from pathlib import Path
import re

from datasets import Dataset, load_dataset
from requests import Session
from sickle import Sickle
from sickle.app import Record
from tqdm import tqdm

from . import DATA_DIR
# VJOL's SSL certificate is not recognized by Python HTTPS client
# https://forum.pkp.sfu.ca/t/oai-pmh-errors-and-ssl-certificate/72584/3
from .util import no_ssl_verification

PDF_DOWNLOAD_DIR = DATA_DIR / "vjol" / "pdf"
VJOL_DATASET_DIR = DATA_DIR / "vjol" / "parquet"

def list_journal_apis(base_url: str, session: Session) -> dict[str, str]:
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

    r = session.get(base_url)
    html_content = r.text

    vjol_link_format = r'<a href="https://vjol.info.vn/index.php/journal/issue/current">'
    vjol_journal_re = re.escape(vjol_link_format).replace(r'journal', r'(\w+)')
    vjol_journals = re.findall(vjol_journal_re, html_content)
    vjol_apis = {
        journal: f"https://vjol.info.vn/index.php/{journal}/oai"
        for journal in vjol_journals
    }

    #other_website_re = r'<a href="([^\"]+)/issue/archive" target="_blank">'
    #external_links = re.findall(other_website_re, html_content)
    #external_apis = [f"{external_links}/oai"]

    return vjol_apis

def _Sickle_request(self, kwargs):
    """Monkeypatched function for Sickle._request."""
    if not hasattr(self, 'session'):
        self.session = Session()
    if self.http_method == 'GET':
        return self.session.get(self.endpoint, params=kwargs, **self.request_args)
    return self.session.post(self.endpoint, data=kwargs, **self.request_args)

def get_records(oai_api_url: str, session: Session) -> list[dict[str, list]]:
    """Get the metadata for every journal record in VJOL.

    Args:
        oai_api_url (str): the OAI base URL.
        session (Session): a requests.Session object.

    Returns:
        list[dict[str, list]]: the list of metadata associated to the records.
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
            logging.warning(f"Unhandled language: {m['language']}")
            continue
        if m['format'] != ['application/pdf']:
            logging.warning(f"Unhandled format: {m['format']}")
            continue

        pdf_view_url: str = m['relation']
        m['pdf_url'] = pdf_view_url.replace('view', 'download')
        metadata.append(m)
    return metadata

async def download_pdf(pdf_url: str, output_file: str | Path):
    async with aiohttp.ClientSession() as session:
        async with session.get(pdf_url) as response:
            with open(output_file, mode="wb") as file:
                while True:
                    chunk = await response.content.read()
                    if not chunk:
                        break
                    file.write(chunk)

async def process_pdf(pdf_file: str | Path, output_md_file: str | Path):
    """Process a PDF paper using scienceparse.

    Args:
        pdf_file (str | Path): the path to the downloaded paper as a PDF file.
        output_md_file (str | Path): the file path to write Markdown content.
    """
    raise NotImplementedError

async def process_papers(download_dir: str | Path) -> dict[Path, Path]:
    """Process PDF papers in bulk using scienceparse.

    Args:
        download_dir (str | Path): The directory that contains the downloaded PDFs.

    Returns:
        dict[Path, Path]: a mapping from the original PDF to its processed Markdown content.
    """
    download_dir = Path(download_dir)

    raise NotImplementedError
    # java -Xmx6g -jar ./science-parse-cli-assembly-2.0.3.jar download_dir
    # TODO: output_file?

async def download_dataset(base_url: str, download_dir: str | Path):
    download_dir = Path(download_dir)
    record_dir = download_dir / "records"
    session = Session()
    user_agent = "vinasmol-training"
    session.headers['User-Agent'] = user_agent

    api_urls = list_journal_apis(base_url, session=session)
    # Backup the API URLs
    api_urls_json = record_dir / "api_urls.json"
    api_urls_json.write_text(json.dumps(api_urls))

    records = {}
    for journal, api_url in tqdm(api_urls.keys(), desc="Get all records for each journal"):
        records[journal] = get_records(api_url, session=session)
        # Backup the journal record list
        journal_records_json = record_dir / journal / "records.json"
        journal_records_json.write_text(json.dumps(records[journal]))
        logging.info(f"Found {len(records)} records for journal {journal}")

    with aiohttp.ClientSession() as download_session:
        tasks = []
        download_session.headers['User-Agent'] = user_agent
        for journal, journal_records in records.items():
            for record in journal_records:
                article_url: str = record['identifier'][0]
                article_id = article_url.split('/')[-1]
                output_file = download_dir / journal / f"{article_id}.pdf"

                record['local_pdf_file'] = output_file
                task = download_pdf(record['pdf_url'], output_file)
                tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    return records

def compile_dataset(records: dict, markdown_files: dict[Path, Path], data_dir: Path):
    def gen_journal_records(journal: str, journal_records: list):
        for record in journal_records:
            record['local_md_file'] = markdown_files[record['local_pdf_file']]
            record['journal_id'] = journal
            record['text'] = record['local_md_file'].read_text()
            yield record

    for journal, journal_records in records.items():
        journal_ds = Dataset.from_generator(
            gen_journal_records,
            gen_kwargs=dict(journal=journal, jounal_records=journal_records),
        )
        journal_ds.to_parquet(data_dir / f"{journal}.parquet")

def create_and_load_vjol(base_url: str) -> Dataset:
    # TODO: upload to HuggingFace if not created, otherwise load_dataset the uploaded version
    records = asyncio.run(download_dataset(base_url, PDF_DOWNLOAD_DIR))
    markdown_files = asyncio.run(process_papers(PDF_DOWNLOAD_DIR))
    compile_dataset(records, markdown_files, data_dir=VJOL_DATASET_DIR)
    
    load_dataset('text', data_files=VJOL_DATASET_DIR / "*.parquet")
