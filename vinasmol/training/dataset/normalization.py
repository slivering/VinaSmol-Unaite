import re

from datatrove.pipeline.formatters.base import BaseFormatter
from datatrove.utils.perplexity import KenlmModel
import ftfy

FTFY_CONFIG = ftfy.TextFixerConfig(
    unescape_html="auto",
    remove_terminal_escapes=True,
    fix_encoding=True,
    restore_byte_a0=True,
    replace_lossy_sequences=True,
    decode_inconsistent_utf8=True,
    fix_c1_controls=True,
    fix_latin_ligatures=False,
    fix_character_width=False,
    uncurl_quotes=False,
    fix_line_breaks=False,
    fix_surrogates=True,
    remove_control_chars=True,
    normalization=None,
)

# TODO: audit and exclude some of these items
UNICODE_PUNCTUATION: dict[str, str] = KenlmModel.unicode_punct

# Only strip if it's not Markdown list indentation
WS_LEADING_RE = re.compile(r"^([ \r\t\xa0]{4,})([^*+\- \r\t\xa0])")
WS_TRAILING_RE = re.compile(r"([ \r\t\xa0]{4,})$")

class Formatter(BaseFormatter):

    name = "ftfy & Punctuation Normalizer"

    def __init__(
            self,
            ftfy_config: ftfy.TextFixerConfig = FTFY_CONFIG,
            strip_whitespace=True,
            normalize_punctuation=False,
        ):
        """Initialize the formatter

        Args:
            ftfy_config (ftfy.TextFixerConfig, optional): The ftfy configuration to fix
                and normalize encoding. Defaults to a suitable configuration for textual data.
            strip_whitespace (bool, optional): Remove leading and trailing spaces
                of length four or more. Defaults to True.
            normalize_punctuation (bool, optional): Replace some common Unicode punctuation
                by their ASCII equivalents. Defaults to False.
        """
        super().__init__()
        self.config = ftfy_config
        self.strip_whitespace = strip_whitespace
        self.normalize_punctuation = normalize_punctuation
    
    def format(self, text: str) -> str:
        fixed = ftfy.fix_text(text, self.config)
        if fixed != text:
            self.stat_update("ftfy_modified_docs")

        if self.strip_whitespace:
            fixed = WS_LEADING_RE.sub(r"\2", fixed)
            fixed = WS_TRAILING_RE.sub(r"", fixed)
        
        fixed_with_punct = text
        if self.normalize_punctuation:
            table = {
                ord(punct): replacement
                for punct, replacement in UNICODE_PUNCTUATION.items()
            }
            fixed_with_punct = fixed.translate(table)
            if fixed_with_punct != fixed:
                self.stat_update("punct_normalized_docs")

        return fixed_with_punct