import re

# Regex patterns
EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_REGEX = r"\+?\d[\d\s\-()]{7,}"


def extract_emails(text: str):
    """Return list of all email addresses in the text."""
    return re.findall(EMAIL_REGEX, text)


def extract_phone_numbers(text: str):
    """Return list of phone numbers."""
    return re.findall(PHONE_REGEX, text)


def extract_named_entities(text: str, nlp):
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]