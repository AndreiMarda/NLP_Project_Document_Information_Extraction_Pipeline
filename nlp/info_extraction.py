import re
import spacy

nlp = spacy.load("en_core_web_sm")

# Regex patterns
EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_REGEX = r"\+?\d[\d\s\-()]{7,}"


def extract_emails(text: str):
    """Return list of all email addresses in the text."""
    return re.findall(EMAIL_REGEX, text)


def extract_phone_numbers(text: str):
    """Return list of phone numbers."""
    return re.findall(PHONE_REGEX, text)


def extract_named_entities(text: str):
    """
    Extract named entities using spaCy NER.
    Returns list of dicts: {text, label, start_char, end_char}
    """
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })
    return ents
