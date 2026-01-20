import spacy
import os
import hashlib
import re
import json
from datetime import datetime

from utils.document_io import load_folder, load_file, load_url
from utils.actions import print_indexed_sentences, extract_info_from_query
from retrieval.semantic_search import SemanticCorpusIndex

from qa.qa import answer_question


def safe_name(s: str) -> str:
    """
    Replaces any run of non-alphanumeric characters (except ., _, -) with a single underscore
    """
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

def _cache_key_from_url(url: str) -> str:
    """
    It generates a stable, short cache key for a URL by SHA-1 hashing the URL
    string and taking the first 12 hex characters, prefixed with "url_"
    """
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

def save_iteration_jsonl(log_path: str, record: dict) -> None:
    """
    Appends one JSON record per line (JSONL format).
    """
    record = dict(record)  # shallow copy to avoid surprises
    record["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log_path_for_option(opt: str) -> str:
    # One log file per menu option
    mapping = {
        "1": "log_opt1_indexed_sentences.jsonl",
        "2": "log_opt2_extract_info.jsonl",
        "3": "log_opt3_qa.jsonl",
        "4": "log_opt4_semantic_search.jsonl",
    }
    return mapping.get(opt, "log_unknown_option.jsonl")


def menu():
    nlp = spacy.load("en_core_web_sm")

    # JSONL log file name (created/appended in the same folder you run the script from)
    iteration = 0

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

        LOG_PATH = log_path_for_option(opt)

        docs, cache_key = choose_source()
        if not docs:
            continue

        # One iteration per successful menu run (i.e., after selecting option + choosing a source)
        iteration += 1

        if opt == "1":
            for i, d in enumerate(docs):
                if i > 0:
                    print()
                print_indexed_sentences(d, nlp)

                # Save a small record (this option primarily prints)
                save_iteration_jsonl(LOG_PATH, {
                    "iteration": iteration,
                    "option": opt,
                    "method": "Print indexed sentences",
                    "cache_key": cache_key,
                    "doc_id": d.doc_id,
                    "source": d.source,
                })

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

                # Save the extracted results
                save_iteration_jsonl(LOG_PATH, {
                    "iteration": iteration,
                    "option": opt,
                    "method": "Extract info",
                    "cache_key": cache_key,
                    "doc_id": d.doc_id,
                    "source": d.source,
                    "user_query": q,
                    "results": results,
                })

        elif opt == "3":
            question = input("Your question: ").strip()
            for i, d in enumerate(docs):
                if i > 0:
                    print()

                ans = answer_question(question, d.text)
                print(f"[DOC={d.doc_id}] {ans}")

                # Save question + answer
                save_iteration_jsonl(LOG_PATH, {
                    "iteration": iteration,
                    "option": opt,
                    "method": "Question answering",
                    "cache_key": cache_key,
                    "doc_id": d.doc_id,
                    "source": d.source,
                    "question": question,
                    "answer": ans,
                })

        elif opt == "4":
            query = input("Topic / query: ").strip()
            top_k = input("Top K results (default 5): ").strip()
            top_k = int(top_k) if top_k.isdigit() else 5

            # per-corpus cache file
            cache_path = f"corpus_index__{safe_name(cache_key)}.pkl"

            index = SemanticCorpusIndex()

            used_cache = False
            try:
                index.load(cache_path)
                print(f"Loaded cached index from {cache_path}")
                used_cache = True
            except Exception:
                print("Building index (first run or cache missing)...")
                index.build_from_docs(docs, min_par_len=50)
                index.save(cache_path)
                print(f"Saved index to {cache_path}")

            results = index.search(query, top_k=top_k)

            print("\nResults: ")
            for score, chunk in results:
                snippet = chunk.text if len(chunk.text) < 300 else chunk.text[:300] + "..."
                print(f"- score={score:.3f} | DOC={chunk.doc_id} | P{chunk.paragraph_id}")
                print(f"  {snippet}\n")

            # Save semantic search results (JSON-friendly)
            save_iteration_jsonl(LOG_PATH, {
                "iteration": iteration,
                "option": opt,
                "method": "Semantic search",
                "cache_key": cache_key,
                "index_cache_path": cache_path,
                "index_loaded_from_cache": used_cache,
                "query": query,
                "top_k": top_k,
                "results": [
                    {
                        "score": score,
                        "doc_id": chunk.doc_id,
                        "paragraph_id": chunk.paragraph_id,
                        "text": chunk.text,
                    }
                    for score, chunk in results
                ],
            })

        else:
            print("Invalid option.")


if __name__ == "__main__":
    menu()
