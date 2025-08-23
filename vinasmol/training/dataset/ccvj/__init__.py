from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RECORDS_DOWNLOAD_DIR = DATA_DIR / "ccvj" / "records"
PDF_DOWNLOAD_DIR = DATA_DIR / "ccvj" / "pdf"

Url = str
JournalId = str
RecordMetadata = dict[str, list]