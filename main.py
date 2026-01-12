import spacy

from utils.document_io import load_folder, load_file, load_url
from utils.actions import print_indexed_sentences, extract_info_from_query, qa_on_doc


def choose_source():
    print("\nChoose input source:")
    print("1) Folder (PDF/DOCX)")
    print("2) Single file (PDF/DOCX)")
    print("3) URL (HTML)")
    choice = input("Your choice: ").strip()

    if choice == "1":
        folder = input("Folder path (e.g. pdf): ").strip()
        return load_folder(folder)

    if choice == "2":
        path = input("File path (e.g. pdf/biology.pdf): ").strip()
        return [load_file(path)]

    if choice == "3":
        url = input("URL: ").strip()
        return [load_url(url)]

    print("Invalid choice.")
    return []


def menu():
    nlp = spacy.load("en_core_web_sm")

    while True:
        print("\n=== Document Analyzer ===")
        print("1) Print indexed sentences ")
        print("2) Extract info (emails, phones, persons, orgs, locations, dates, full_text)")
        print("3) Ask a question ")
        print("0) Exit")

        opt = input("Select option: ").strip()

        if opt == "0":
            return

        docs = choose_source()
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

        else:
            print("Invalid option.")


if __name__ == "__main__":
    menu()
