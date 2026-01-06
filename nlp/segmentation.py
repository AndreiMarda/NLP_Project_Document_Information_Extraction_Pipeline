from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class Paragraph:
    paragraph_id: int
    text: str

@dataclass(frozen=True)
class Sentence:
    paragraph_id: int
    sentence_id: int   # sentence index WITHIN the paragraph
    text: str

def segment_paragraphs(text: str, min_len: int = 1) -> List[Paragraph]:
    raw_paras = (p.strip() for p in text.split("\n"))
    paras = [p for p in raw_paras if len(p) >= min_len]
    return [Paragraph(i, p) for i, p in enumerate(paras)]

def segment_sentences(
    text: str,
    nlp,
    paragraphs: Optional[List[Paragraph]] = None,
    min_len: int = 1,
) -> List[Sentence]:

    if paragraphs is None:
        paragraphs = segment_paragraphs(text, min_len=min_len)

    # Ensure sentence boundaries exist
    if not any(p in nlp.pipe_names for p in ("parser", "senter")):
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    sentences: List[Sentence] = []

    for p in paragraphs:
        doc = nlp(p.text)
        sent_id = 0
        for sent in doc.sents:
            s = sent.text.strip()
            if len(s) < min_len:
                continue
            sentences.append(
                Sentence(
                    paragraph_id=p.paragraph_id,
                    sentence_id=sent_id,
                    text=s,
                )
            )
            sent_id += 1

    return sentences

def segment_text(text: str, nlp, paragraph_min_len: int = 1, sentence_min_len: int = 1):
    paragraphs = segment_paragraphs(text, min_len=paragraph_min_len)
    sentences = segment_sentences(text, nlp=nlp, paragraphs=paragraphs, min_len=sentence_min_len)
    return paragraphs, sentences
