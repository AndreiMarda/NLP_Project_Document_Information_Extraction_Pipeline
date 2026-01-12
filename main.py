import spacy
import os
import hashlib
import re

from utils.document_io import load_folder, load_file, load_url
from utils.actions import print_indexed_sentences, extract_info_from_query, qa_on_doc
from retrieval.semantic_search import SemanticCorpusIndex


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

def _cache_key_from_url(url: str) -> str:
    # stable short key for URLs
    return "url_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]

def choose_source():
    print("\nChoose input source:")
    print("1) Folder (PDF/DOCX)")
    print("2) Single file (PDF/DOCX)")
    print("3) URL (HTML)")
    choice = input("Your choice: ").strip()

    if choice == "1":
        folder = input("Folder path (e.g. pdf): ").strip()
        docs = load_folder(folder)
        # corpus key based on folder path
        cache_key = "folder_" + os.path.abspath(folder)
        return docs, cache_key

    if choice == "2":
        path = input("File path (e.g. pdf/biology.pdf): ").strip()
        docs = [load_file(path)]
        # corpus key based on absolute file path
        cache_key = "file_" + os.path.abspath(path)
        return docs, cache_key

    if choice == "3":
        url = input("URL: ").strip()
        docs = [load_url(url)]
        cache_key = _cache_key_from_url(url)
        return docs, cache_key

    print("Invalid choice.")
    return [], ""


def menu():
    nlp = spacy.load("en_core_web_sm")

    while True:
        print("\n=== Document Analyzer ===")
        print("1) Print indexed sentences ")
        print("2) Extract info (emails, phones, persons, orgs, locations, dates, full_text)")
        print("3) Ask a question ")
        print("4) Semantic search across multiple documents (topic search)")
        print("0) Exit")

        opt = input("Select option: ").strip()

        if opt == "0":
            return

        docs, cache_key = choose_source()
        if not docs:
            continue

        if opt == "1":
            for i, d in enumerate(docs):
                if i > 0:
                    print()
                print_indexed_sentences(d, nlp)

        elif opt == "2":
            print("\nExamples:")
            print("  Show me all emails and phone numbers")
            print("  List all persons and organizations mentioned")
            print("  Give me all dates and locations")
            print("  Show me full text\n")
            q = input("Your query: ").strip()

            for i, d in enumerate(docs):
                if i > 0:
                    print()
                results = extract_info_from_query(d, q, nlp)
                print(f"[DOC={d.doc_id}]")
                for k, v in results.items():
                    print(f"  {k}:")
                    if isinstance(v, list):
                        if not v:
                            print("    (none)")
                        else:
                            for item in v:
                                print("   -", item)
                    else:
                        print("   ", v if len(v) < 800 else v[:800] + "...")

        elif opt == "3":
            question = input("Your question: ").strip()
            for i, d in enumerate(docs):
                if i > 0:
                    print()
                ans = qa_on_doc(d, question)
                print(f"[DOC={d.doc_id}] {ans}")


        elif opt == "4":
            query = input("Topic / query: ").strip()
            top_k = input("Top K results (default 5): ").strip()
            top_k = int(top_k) if top_k.isdigit() else 5

            # per-corpus cache file
            cache_path = f"corpus_index__{safe_name(cache_key)}.pkl"

            index = SemanticCorpusIndex()

            try:
                index.load(cache_path)
                print(f"Loaded cached index from {cache_path}")
            except Exception:
                print("Building index (first run or cache missing)...")
                index.build_from_docs(docs, min_par_len=50)
                index.save(cache_path)
                print(f"Saved index to {cache_path}")
            results = index.search(query, top_k=top_k)

            print("\n=== Results ===")
            for score, chunk in results:
                snippet = chunk.text if len(chunk.text) < 300 else chunk.text[:300] + "..."
                print(f"- score={score:.3f} | DOC={chunk.doc_id} | P{chunk.paragraph_id}")
                print(f"  {snippet}\n")


        else:
            print("Invalid option.")


if __name__ == "__main__":
    menu()
