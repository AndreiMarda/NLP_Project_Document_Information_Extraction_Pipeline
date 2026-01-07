import os
from dataclasses import dataclass
from typing import List, Tuple

from extractors.pdf_extractor import extract_text_from_pdf
from extractors.docx_extractor import extract_text_from_docx
from extractors.html_extractor import extract_text_from_url

from nlp.preprocessing import normalize_pdf_paragraphs, clean_preserve_newlines

SUPPORTED_EXTS: Tuple[str, ...] = (".pdf", ".docx")


@dataclass(frozen=True)
class LoadedDoc:
    doc_id: str   # filename or url
    source: str   # full path or url
    ext: str      # ".pdf", ".docx", ".html"
    text: str     # prepared text (paragraphs preserved with \n)


def prepare_text(raw_text: str, ext: str) -> str:
    if ext == ".pdf":
        raw_text = normalize_pdf_paragraphs(raw_text)
    return clean_preserve_newlines(raw_text)


def load_file(path: str) -> LoadedDoc:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        raw = extract_text_from_pdf(path)
    else:
        raw = extract_text_from_docx(path)

    return LoadedDoc(
        doc_id=os.path.basename(path),
        source=path,
        ext=ext,
        text=prepare_text(raw, ext),
    )


def load_url(url: str) -> LoadedDoc:
    raw = extract_text_from_url(url)
    return LoadedDoc(
        doc_id=url,
        source=url,
        ext=".html",
        text=prepare_text(raw, ".html"),
    )


def list_folder(folder: str, exts: Tuple[str, ...] = SUPPORTED_EXTS) -> List[str]:
    paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not paths:
        raise FileNotFoundError(f"No documents found in {folder} with extensions {exts}")
    return paths


def load_folder(folder: str, exts: Tuple[str, ...] = SUPPORTED_EXTS) -> List[LoadedDoc]:
    return [load_file(p) for p in list_folder(folder, exts)]
