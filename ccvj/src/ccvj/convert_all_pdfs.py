from functools import partial
from pathlib import Path, PurePath
import re
from typing_extensions import override
from multiprocessing.pool import ThreadPool
import unicodedata

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from docling_core.transforms.serializer.base import (
    BaseDocSerializer,
    SerializationResult,
)
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownListSerializer,
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc.document import DoclingDocument, ListGroup, TableItem

from loguru import logger
import typer

class FrugalMarkdownTableSerializer(MarkdownTableSerializer):
    """A Markdown table serializer that skips large tables."""

    _CELL_PADDING_RE = re.compile(r"\|([^\|]*)")
    _HEADER_SEP_RE = re.compile(r"--(-+)")

    def __init__(self, max_table_rows: int, max_table_cols: int, max_md_len: int):
        super().__init__()
        self.max_table_rows = max_table_rows
        self.max_table_cols = max_table_cols
        self.max_md_len = max_md_len
    
    @classmethod
    def strip_tabulation_spaces(cls, text: str) -> str:
        """Remove unnecessary spaces and separators from a Github-formatted Markdown table."""
        text = cls._CELL_PADDING_RE.sub(lambda m: "| " + m.group().strip() + " ", text)
        text = cls._HEADER_SEP_RE.sub(r"---", text)
        return text

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        if item.data.num_cols > self.max_table_cols:
            # Table with many columns are difficult to understand
            return create_ser_result(text="", span_source=item)
        
        old_num_rows = item.data.num_rows
        if old_num_rows > self.max_table_rows:
            # Skip when there are too many rows
            item.data.num_rows = self.max_table_rows
            item.data.table_cells = [
                cell for cell in item.data.table_cells
                if cell.end_row_offset_idx < self.max_table_rows
            ]
        # TODO: possibly exclude overly complex tables

        result = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            **kwargs
        )
        item.data.num_rows = old_num_rows
        
        # centered text is sometimes split across multiple lines inside a cell
        result.text = result.text.replace("<br>", " ")
        result.text = FrugalMarkdownTableSerializer.strip_tabulation_spaces(result.text)
        
        if len(result.text) > self.max_md_len:
            # Long tables clutter the context, discard
            result.text = ""
                
        return result

class FrugalMarkdownListSerializer(MarkdownListSerializer):

    max_items: int    

    @override
    def serialize(
            self, *,
            item: ListGroup,
            doc_serializer: BaseDocSerializer,
            doc: DoclingDocument,
            list_level = 0,
            is_inline_scope = False,
            visited = None,
            **kwargs,
        ):
        old_children = item.children
        item.children = old_children[:self.max_items]
        result = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            list_level=list_level,
            is_inline_scope=is_inline_scope,
            visited=visited,
            **kwargs
        )
        item.children = old_children
        return result


def strip_accents(s: str) -> str:
   # Alternative: from pyvi.ViUtils import remove_accents
   return ''.join(
       c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

class MarkdownPostProcessor:

    _VI_UPPER = r"A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
    _VI_LOWER = r"a-zàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    _VI_ALPHA = _VI_LOWER + _VI_UPPER

    _PUNCTUATION = ".,;:!?" # NOT a Regex
    _WORD_OR_PUNCT_RE = r"[\w\)\(\?!\.,;:/\"']"

    # Unlikely to interfere with programming
    # Possibly some rare issues with LaTeX with four-term multiplication
    _BAD_SPACE_IN_TEXT = re.compile(r"[ \t\xa0](?:[%s][ \t\xa0]){4,}" % _VI_ALPHA)

    _BIBLIOGRAPHY_RE_CAREFUL = re.compile(
        r"TAI\s{1,3}LIEU\s{1,3}THAM\s{1,3}KHAO",
        flags=re.IGNORECASE
    )
    _BIBLIOGRAPHY_SECTION_RE = re.compile(
        r"[\*# ]{,3}"
        r"(?:TAI\s{1,3}LIEU\s{1,3}THAM\s{1,3}KHAO|Tài\s{1,3}liệu\s{1,3}tham\s{1,3}khảo)"
        r"[^#]+",
        flags=re.IGNORECASE
    )

    _THANKS = re.compile(
        r"[#\*]{0,4}\s?LOI\s{1,3}CAM\s{1,3}ON(:.+\n|\n[^#]+)",
        flags=re.IGNORECASE | re.MULTILINE
    )

    _INLINE_REF_RE = [
        re.compile(r"\s\[\s{0,3}\d{1,3}(,\s{0,3}\d{1,3})*\s{0,3}\]"), # [ 999, 42]
        re.compile(r"\s\([^),]+,\s{0,2}\d{4}[^)]*\)") # (Doe et al., 2019; John, 2023)
    ]

    _UNPARSED_FORMULA = re.compile("^.+=\n", flags=re.MULTILINE)

    _FIG_LEGEND_RE = re.compile(r"^[#\*]*(?:Bảng|Hình).+(\n)+", flags=re.MULTILINE)

    _RULER_RE = re.compile(r"(\\_)+\n\n", flags=re.MULTILINE)

    _EXCESS_SPACE_RE = [
        # Space before punctuation (disallowed for Vietnamese and English typography)
        (re.compile(r"(\s+)([,\.:;?!'])"), r"\2"),
        # Space between word and dash
        # FIXME: false positive with long dash converted to dash
        (re.compile(r"(\w)(\s+)-"), r"\1-"),
        # Space before closing parenthese
        (re.compile(r"(\s+)\)"), r")"),
        # Space after opening parenthese
        (re.compile(r"\((\s+)"), r"("),
        # Two spaces between words
        (re.compile(fr"({_WORD_OR_PUNCT_RE}\s)(\s+)({_WORD_OR_PUNCT_RE})"), r"\1\3"),
    ]

    _CONSECUTIVE_BLANKLINES_RE = re.compile(r"\n\n\n+", flags=re.MULTILINE)

    _LINE_WITH_EMAIL_RE = re.compile("^.+@.+\n", flags=re.MULTILINE)

    _PUBLISHING_LINE_PREFIXES = [
        "THÔNG TIN BÀI BÁO", "PUBLISHING INFORMATION",
        "Tác giả:", "Author:", "Authors:",
        "Quá trình:", "Received:",
        "Trang điện tử:", "Website:",
        r"\*?Tác giả liên hệ:", r"\*?Corresponding author:",
        r"E-mail(?: address)?:", r"Email(?: address)?:",
        "DOI:",
        "©",
        "Ngày nhận bản thảo:", r"(:?Ngày)? nhận bài báo", "Ngày bài báo", "Date of manuscript",
        r"Ngày nộp bài (:?sửa)?:", r"Received (in Revised Form)?:",
        "Ngày nhận bài phản biện", # Received for review
        "Ngày chấp nhận:", "Accepted:",
        "Nhận bài", "Chấp nhận", "Đăng online",
    ]
    # FIXME: sometimes the value is on the line below
    _PUBLISHING_INFO_RE = re.compile(
        rf"^({'|'.join(_PUBLISHING_LINE_PREFIXES)}).+\n",
        flags=re.MULTILINE,
    )

    _MULTILINE_TITLE_RE = [
        (re.compile(fr"(#{{{i}}} .+)\n\n#{{{i}}} (.+)"), r"\1 \2")
        for i in (1, 2)
    ]

    _MULTILINE_SENTENCE_RE = re.compile(
        rf"(^[^#\n].+?[0-9{_VI_ALPHA}])\s\s+?([0-9{_VI_LOWER}])",
        flags=re.MULTILINE,
    )

    def __init__(
            self,
            heading_max_num_syllables: int = 20,
            careful_bibliography_identification: bool = False,
            max_num_bad_space_in_text: int = 30,
            max_freq_bad_space_in_text: float = 0.05,
        ):
        self.max_num_bad_space_in_text = max_num_bad_space_in_text
        self.max_freq_bad_space_in_text = max_freq_bad_space_in_text
        self.careful_bibliography_identification = careful_bibliography_identification
        self.heading_max_num_syllables = heading_max_num_syllables
        

    def should_rescan_with_ocr(self, md_content: str) -> bool:
        num_bad = len(MarkdownPostProcessor._BAD_SPACE_IN_TEXT.findall(md_content))
        return num_bad > 10 and (
            num_bad > self.max_num_bad_space_in_text
            or num_bad / len(md_content) > 8 * self.max_freq_bad_space_in_text
        )

    def remove_page_headers(self, md_content: str) -> str:
        """Remove duplicate lines that are preceded and followed by blank lines.
        
        This is also an heuristic for removing page headers.
        """
        lines = [line.strip() for line in md_content.splitlines()]
        indices = {}
        for i, line in enumerate(lines):
            if line not in indices:
                indices[line] = []
            indices[line].append(i)
        duplicate_indices = {
            index
            for line_indices in indices.values()
                for index in line_indices
                    if len(line_indices) > 1 and lines[index]
        }
        to_remove = set()
        for i in duplicate_indices:
            if 1 <= i < len(lines) - 1:
                if not lines[i - 1] and not lines[i + 1]:
                    # The line is preceded and followed by blanks
                    to_remove.add(i - 1)
                    to_remove.add(i)
        return "\n".join(line for i, line in enumerate(lines) if i not in to_remove)

    def remove_bibliography(self, md_content: str) -> str:
        """Remove the bibliography."""
        if not self.careful_bibliography_identification:
            # Remove the whole section
            return MarkdownPostProcessor._BIBLIOGRAPHY_SECTION_RE.sub("", md_content)
        
        def is_reference(line: str) -> bool:
            # A more systematic tool is https://github.com/inukshuk/anystyle-cli
            # Requires Ruby interpreter, so usage is not planned
            line = line.strip()
            if line.startswith("-"):
                # Bullet points can be part of the bibliography
                return True
            if line == "":
                # Spaces occur when the bibliography spreads across multiple lines,
                # and after some element removal
                return True
            
            num_points = line.count(".")
            num_commas = line.count(",")
            num_capitalized = sum(c.isupper() for c in line)
            num_digits = sum(c.isdigit() for c in line)
            if (line.endswith(".")
                    and num_commas >= 2
                    and num_points >= 5
                    and num_capitalized >= 5
                    and num_digits >= 4):
                return True

            return False

        lines = md_content.splitlines()
        
        part_of_bibliography = [False for _ in range(len(lines))]
        for (i, line) in enumerate(lines):
            normalized = strip_accents(line)
            if MarkdownPostProcessor._BIBLIOGRAPHY_RE_CAREFUL.search(normalized):
                part_of_bibliography[i] = True
                for j in range(i + 1, len(lines)):
                    if is_reference(lines[j]):
                        part_of_bibliography[j] = True
                    else:
                        break
        
        return "\n".join(line for (i, line) in enumerate(lines) if not part_of_bibliography[i])
    
    def remove_acknowledgments(self, md_content: str) -> str:
        normalized = strip_accents(md_content)
        thanks = MarkdownPostProcessor._THANKS.search(normalized)
        if thanks:
            start = thanks.start()
            end = thanks.end()
            md_content = md_content[:start] + md_content[end:]
        return md_content

    def remove_inline_references(self, md_content: str) -> str:
        for pat in MarkdownPostProcessor._INLINE_REF_RE:
            md_content = pat.sub("", md_content)
        return md_content
    
    def remove_excess_space(self, md_content: str) -> str:
        lines = md_content.splitlines()
        for i in range(len(lines)):
            for pat, repl in MarkdownPostProcessor._EXCESS_SPACE_RE:
                lines[i] = pat.sub(repl, lines[i])
        return "\n".join(lines)
    
    def remove_figure_legend(self, md_content: str) -> str:
        md_content = MarkdownPostProcessor._FIG_LEGEND_RE.sub("", md_content)
        return md_content
    
    def remove_rulers(self, md_content: str) -> str:
        return MarkdownPostProcessor._RULER_RE.sub("", md_content)
    
    def remove_contact_info(self, md_content: str) -> str:
        return MarkdownPostProcessor._LINE_WITH_EMAIL_RE.sub("", md_content)

    def remove_publishing_info(self, md_content: str) -> str:
        # TODO: remove Vietnamese names with name list
        return MarkdownPostProcessor._PUBLISHING_INFO_RE.sub("", md_content)

    def format_free_headings(self, md_content: str) -> str:
        lines = md_content.splitlines()
        for (i, line) in enumerate(lines):
            if (line
                    and line[0].isupper()
                    and line[-1] not in MarkdownPostProcessor._PUNCTUATION
                    and line.count(" ") < self.heading_max_num_syllables
                ):
                # Sometimes these free standing lines are not headings
                # but metadata or incomplete lines. In that case they would be better removed,
                # but at least marking them with bold helps
                lines[i] = f"**{line}**"
        return "\n".join(lines)
    
    def remove_unparsed_formulas(self, md_content: str) -> str:
        return MarkdownPostProcessor._UNPARSED_FORMULA.sub("", md_content)
    
    def merge_multiline_title(self, md_content: str) -> str:
        for pat, repl in MarkdownPostProcessor._MULTILINE_TITLE_RE:
            # Applies to the Vietnamese and the English title at most
            md_content = pat.sub(repl, md_content, count=2)
        return md_content
    
    def recombine_newlines(self, md_content: str) -> str:
        return MarkdownPostProcessor._CONSECUTIVE_BLANKLINES_RE.sub("\n\n", md_content)
    
    def merge_multiline_sentences(self, md_content: str) -> str:
        return MarkdownPostProcessor._MULTILINE_SENTENCE_RE.sub(
            r"\1 \2",
            md_content,
        )
    
    def format(self, md_content: str) -> str:
        md_content = unicodedata.normalize("NFC", md_content)
        md_content = self.remove_page_headers(md_content)
        md_content = self.remove_acknowledgments(md_content)
        md_content = self.remove_bibliography(md_content)
        md_content = self.remove_inline_references(md_content)
        md_content = self.remove_unparsed_formulas(md_content)
        md_content = self.remove_excess_space(md_content)
        md_content = self.remove_figure_legend(md_content)
        md_content = self.remove_rulers(md_content)
        md_content = self.remove_contact_info(md_content)
        md_content = self.remove_publishing_info(md_content)
        md_content = self.merge_multiline_title(md_content)
        md_content = self.merge_multiline_sentences(md_content)
        md_content = self.recombine_newlines(md_content)
        md_content = self.format_free_headings(md_content)

        # TODO: escape asterisks
        # use layout to determine page break

        # TODO: extract abstract from table
        # remove authorship information (name, university, contact info, journal copyright)
        # --disable_image_extraction
        # --extract_images False
        # --max_table_rows 10
        return md_content


class MarkdownConversionWriter:
    def __init__(self, file: PurePath):
        assert file.name.endswith(".pdf")

        self.file = file
        self.dir = file.parent
        self.file_name = file.name.removesuffix(".pdf")
    
    def _write(self, pat: str, content: str) -> Path:
        f = Path(self.dir / pat.format(self.file_name))
        f.write_text(content)


def make_md_doc_serializer(document: DoclingDocument) -> MarkdownDocSerializer:
    return MarkdownDocSerializer(
        doc=document,
        table_serializer=FrugalMarkdownTableSerializer(
            max_table_rows=5,
            max_table_cols=5,
            max_md_len=256,
        ),
        # TODO:
        #list_serializer=FrugalMarkdownListSerializer(max_items=10),
        params=MarkdownParams(
            include_formatting=True,
            include_hyperlinks=False,
            image_placeholder="",
            #enable_chart_tables=False, # Possibly keep small tables with custom table serializer
            include_annotations=False,
            escape_html=False,
        ),
    )

def format_trace(result: ConversionResult) -> str:
    trace = ""
    for error in result.errors:
        trace += error.error_message + "\n"
    return trace

# TODO: https://docling-project.github.io/docling/examples/rapidocr_with_custom_models/
# needs rapidocr_onnxruntime
# Setup RapidOcrOptions for English detection
# For multi-language:
# https://rapidai.github.io/RapidOCRDocs/blog/2022/09/28/%E6%94%AF%E6%8C%81%E8%AF%86%E5%88%AB%E8%AF%AD%E8%A8%80/

pipeline_options = PdfPipelineOptions(
    do_table_structure=True,
    #do_formula_enrichment=True,
    do_code_enrichment=False,
    do_ocr=True,
    ocr_options=EasyOcrOptions(lang=["en", "vi"]),
    #TesseractOcrOptions(lang=["en", "vi"])
)
# uses text cells predicted from table structure model
#pipeline_options.table_structure_options.do_cell_matching = False
# pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
# pipeline_options.accelerator_options.num_threads = 8
# pipeline_options.accelerator_options.cuda_use_flash_attention2 = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
        )
    }
)

def convert_all_pdfs(sources: list[str], force_full_ocr: bool = False):
    if not sources:
        logger.info("Finished!")
        return
    logger.debug("Starting conversion job with {} sources", len(sources))

    if force_full_ocr:
        converter.format_to_options[InputFormat.PDF].pipeline_options.ocr_options.force_full_page_ocr = True
        logger.info("Starting conversion with forced full OCR")
    else:
        logger.info("Starting conversion")
    
    # TODO: decide max_num_pages based on average number of pages
    results = converter.convert_all(sources, raises_on_error=False, max_num_pages=20)

    redo_with_ocr = []

    for (source, result) in zip(sources, results):
        writer = MarkdownConversionWriter(result.input.file)
        name = writer.file_name
        logger.info("Got result for file {}", name)

        if result.status == ConversionStatus.SUCCESS:
            document = result.document
            confidence = result.confidence

            score = confidence.low_score
            if score < 0.8:
                logger.warning("Low-quality conversion with score {}", score)
                # TODO: discard the document
            else:
                logger.info("Conversion with score {}", score)
        else:
            logger.error("Document {} could not be converted.", result.input.file)
            writer._write(".{}.md.error", format_trace(result))
            continue
        
        default_md = result.document.export_to_markdown(include_annotations=False)
        writer._write(".{}.md.default", default_md)

        serializer = make_md_doc_serializer(document)
        ser_result = serializer.serialize()
        ser_text = ser_result.text
        writer._write(".{}.md.serialized", ser_result.text)

        postprocessor = MarkdownPostProcessor()

        if postprocessor.should_rescan_with_ocr(ser_text):
            if force_full_ocr:
                err_msg = f"Failed twice to cleanly convert {name}, discarding"
                logger.warning(err_msg)
                writer._write(".{}.md.error", err_msg)
                continue
            else:
                logger.info("PDF {} has bad space in text, will redo conversion with OCR", name)
                # FIXME: result.input.file loses information about parent folders
                redo_with_ocr.append(source)
        else:
            # Write the final result
            preprocessed_text = postprocessor.format(ser_text)
            writer._write("{}.md", preprocessed_text)
    
    convert_all_pdfs(redo_with_ocr, force_full_ocr=True)



app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
        pdf_dir: Path,
        num_proc: int | None = None,
        batch_size: int | None = None
    ):
    sources = list(pdf_dir.rglob("*.pdf"))
    logger.info("Starting conversion of {} PDFs", len(sources))


    if num_proc is None:
        convert_all_pdfs(sources, force_full_ocr=False)
    else:
        pool = ThreadPool(num_proc)
        batched = [sources[i : i + batch_size] for i in range(0, len(sources), batch_size)]
        pool.map(
            partial(convert_all_pdfs, force_full_ocr=False),
            batched,
        )


if __name__ == '__main__':
    app()

