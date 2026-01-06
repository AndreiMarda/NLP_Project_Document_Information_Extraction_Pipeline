import argparse
import os
import spacy
from typing import Optional, Dict, Any, Tuple, List

from extractors.pdf_extractor import extract_text_from_pdf
from extractors.docx_extractor import extract_text_from_docx
from extractors.html_extractor import extract_text_from_url

from nlp.preprocessing import (
    clean_preserve_newlines,
    normalize_pdf_paragraphs,
)
from nlp.segmentation import segment_text
from nlp.query_understanding import detect_intent
from nlp.intent_executor import execute_intent

# spaCy loaded once
nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------
# Extraction helpers
# --------------------------------------------------

def extract_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".docx":
        return extract_text_from_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")


def list_documents(folder: str, exts: Tuple[str, ...] = (".pdf", ".docx")) -> List[str]:
    docs = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not docs:
        raise SystemExit(f"No {exts} files found in folder: {folder}")
    return docs


# --------------------------------------------------
# Text preparation
# --------------------------------------------------

def prepare_text(raw_text: str, ext: str) -> str:
    """
    Prepare text for NLP while preserving paragraph boundaries.
    """
    if ext == ".pdf":
        raw_text = normalize_pdf_paragraphs(raw_text)
    return clean_preserve_newlines(raw_text)


# --------------------------------------------------
# Output
# --------------------------------------------------

def print_sentences(doc_name: str, text: str) -> None:
    _, sentences = segment_text(text, nlp)
    for s in sentences:
        line = " ".join(s.text.split())
        print(f"[DOC={doc_name} | P{s.paragraph_id} | S{s.sentence_id}] {line}")


def print_results(results: Dict[str, Any]) -> None:
    print("\n=== RESULTS ===")
    for key, value in results.items():
        print(f"\n[{key.upper()}]")
        if isinstance(value, list):
            if not value:
                print("  (none found)")
            else:
                for v in sorted(set(value)):
                    print("  -", v)
        else:
            print(value if len(value) < 1000 else value[:1000] + "...")


# --------------------------------------------------
# Modes
# --------------------------------------------------

def run_folder_mode(folder: str) -> None:
    first = True
    for path in list_documents(folder, exts=(".pdf", ".docx")):
        if not first:
            print()
        first = False

        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            raw = extract_text_from_pdf(path)
        elif ext == ".docx":
            raw = extract_text_from_docx(path)
        else:
            continue  # or raise

        # IMPORTANT: preprocessing depends on the source type
        text = prepare_text(raw, ext)  # uses normalize_pdf_paragraphs only for .pdf

        print_sentences(os.path.basename(path), text)

    print("\nDone.\n")



def run_single_mode(url: Optional[str], file_path: Optional[str]) -> None:
    if url:
        raw = extract_text_from_url(url)
        label = url
        ext = ".html"
    elif file_path:
        raw = extract_from_path(file_path)
        label = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
    else:
        raise SystemExit("Provide --url or --file or --folder")

    text = prepare_text(raw, ext)

    # Sentence listing
    print_sentences(label, text)

    # Interactive NLP
    print("\nNow type what you want to know, e.g.:")
    print("  - e-mails")
    print("  - phone numbers")
    print("  - person names")
    print("  - organisations")
    print("  - locations")
    print("  - dates")


    query = input("Your query: ")
    intent = detect_intent(query)
    results = execute_intent(text, intent)
    print_results(results)

    print("\nDone.\n")


# --------------------------------------------------
# CLI
# --------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NLP document analyzer for PDF / DOCX / HTML"
    )
    parser.add_argument("--url", help="HTML page URL", required=False)
    parser.add_argument("--file", help="Path to PDF or DOCX file", required=False)
    parser.add_argument("--folder", help="Folder containing PDF files", required=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.folder:
        run_folder_mode(args.folder)
    else:
        run_single_mode(args.url, args.file)


if __name__ == "__main__":
    main()
