from dataclasses import dataclass
from typing import List


@dataclass
class Intent:
    name: str                 # "info_extraction" or "qa"
    targets: List[str]        # e.g. ["emails", "phones"]
    query: str = ""           # original user query


def is_question(text: str) -> bool:
    q = text.strip().lower()
    return (
        q.endswith("?") or
        q.startswith(("who", "what", "when", "where", "why", "how"))
    )


def detect_intent(query: str) -> Intent:
    """
    Identify what the user wants based on natural-language input.
    """
    q = query.lower()
    targets = []

    # Detect common extraction commands
    if "email" in q or "e-mail" in q:
        targets.append("emails")

    if "phone" in q or "number" in q or "telephone" in q:
        targets.append("phones")

    if "person" in q or "people" in q or "name" in q:
        targets.append("persons")

    if "org" in q or "company" in q or "organisation" in q:
        targets.append("orgs")

    if "location" in q or "city" in q or "country" in q:
        targets.append("locations")

    if "date" in q:
        targets.append("dates")

    if "everything" in q or "full text" in q or "all text" in q:
        targets.append("full_text")

    # Detect Q&A questions
    if is_question(q):
        return Intent(
            name="qa",
            targets=["qa"],
            query=query
        )

    # Default fallback if nothing detected general NER info
    if not targets:
        targets = ["persons", "orgs", "locations", "dates"]

    return Intent(name="info_extraction", targets=targets, query=query)
