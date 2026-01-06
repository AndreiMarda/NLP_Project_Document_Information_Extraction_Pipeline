import re

def basic_clean(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace.
    """
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# def basic_clean_keep_newlines(text: str) -> str:
#     # normalize Windows newlines
#     text = text.replace("\r\n", "\n").replace("\r", "\n")
#
#     # collapse spaces/tabs but keep newlines
#     text = re.sub(r"[ \t]+", " ", text)
#
#     # collapse 3+ newlines to just 2 (optional, keeps paragraph breaks)
#     text = re.sub(r"\n{3,}", "\n\n", text)
#
#     return text.strip()
import re

def clean_preserve_newlines(text: str) -> str:
    """
    Cleans text while preserving paragraph breaks (\n).

    - Normalizes Windows/Mac newlines to \n
    - Trims each line
    - Collapses multiple spaces/tabs inside a line
    - Collapses 2+ blank lines into a single blank line
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        line = re.sub(r"[ \t]+", " ", line)  # do NOT touch \n
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{2,}", "\n", text)  # optional: avoid many empty paragraphs
    return text.strip()

import re

SENT_END = re.compile(r'[.!?]["\')\]]?$')
UPPER_START = re.compile(r'^[A-Z]')

def normalize_pdf_paragraphs(raw_text: str) -> str:
    """
    Turn PDF 'layout newlines' into spaces, while preserving real paragraph breaks.
    Returns text where paragraphs are separated by '\n'.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [ln.strip() for ln in text.split("\n")]

    paras = []
    buf = []

    def flush_buf():
        if buf:
            paras.append(" ".join(buf).strip())
            buf.clear()

    for i, line in enumerate(lines):
        if line == "":
            # blank line => definitely a new paragraph
            flush_buf()
            continue

        # If previous buffer ended with a hyphen, join without space and drop hyphen
        if buf and buf[-1].endswith("-"):
            buf[-1] = buf[-1][:-1] + line
        else:
            # Decide if this line starts a new paragraph (strong signal)
            prev = buf[-1] if buf else ""
            if buf and SENT_END.search(prev) and UPPER_START.search(line):
                flush_buf()
                buf.append(line)
            else:
                buf.append(line)

    flush_buf()

    # final tidy: collapse extra spaces inside each paragraph
    paras = [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]
    return "\n".join(paras)
