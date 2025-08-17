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

UNICODE_PUNCTUATION: dict[str, str] = KenlmModel.unicode_punct


class Formatter(BaseFormatter):
    def __init__(
            self,
            ftfy_config: ftfy.TextFixerConfig = FTFY_CONFIG,
            strip_whitespace=True,
            normalize_punctuation=True,
        ):
        self.config = ftfy_config
        self.strip_whitespace = strip_whitespace
        self.normalize_punctuation = normalize_punctuation
    
    def format(self, text: str) -> str:
        text = ftfy.fix_text(text, self.config)

        if self.strip_whitespace:
            text = text.strip()
        
        if self.normalize_punctuation:
            table = {
                ord(punct): replacement
                for punct, replacement in UNICODE_PUNCTUATION.items()
            }
            text = text.translate(table)

        return text