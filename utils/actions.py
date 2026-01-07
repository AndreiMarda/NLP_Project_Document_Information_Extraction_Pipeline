from typing import Dict, Any, List
from nlp.segmentation import segment_text
from nlp.intent_executor import execute_intent
from nlp.query_understanding import detect_intent

from qa.qa import answer_question  # if you keep QA


def print_indexed_sentences(doc, nlp) -> None:
    _, sentences = segment_text(doc.text, nlp)
    for s in sentences:
        line = " ".join(s.text.split())
        print(f"[DOC={doc.doc_id} | P{s.paragraph_id} | S{s.sentence_id}] {line}")


def extract_info_from_query(doc, user_query: str, nlp) -> Dict[str, Any]:
    intent = detect_intent(user_query)
    return execute_intent(doc.text, intent, nlp)


def qa_on_doc(doc, question: str) -> str:
    return answer_question(question, doc.text)
