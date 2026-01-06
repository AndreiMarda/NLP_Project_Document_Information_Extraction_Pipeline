from typing import Dict, Any

from .info_extraction import (
    extract_emails,
    extract_phone_numbers,
    extract_named_entities,
)
from qa.qa import answer_question


def execute_intent(text: str, intent) -> Dict[str, Any]:
    """
    Runs the appropriate extraction logic based on what the user asked.
    """
    results: Dict[str, Any] = {}

    # # QA mode
    if intent.name == "qa":
        answer = answer_question(intent.query, text)
        results["answer"] = answer
        return results

    if "emails" in intent.targets:
        results["emails"] = list(set(extract_emails(text)))

    if "phones" in intent.targets:
        results["phones"] = list(set(extract_phone_numbers(text)))

    if any(t in intent.targets for t in ["persons", "orgs", "locations", "dates"]):
        ents = extract_named_entities(text)

        if "persons" in intent.targets:
            results["persons"] = [e["text"] for e in ents if e["label"] == "PERSON"]

        if "orgs" in intent.targets:
            results["orgs"] = [e["text"] for e in ents if e["label"] == "ORG"]

        if "locations" in intent.targets:
            results["locations"] = [
                e["text"] for e in ents if e["label"] in ("GPE", "LOC")
            ]

        if "dates" in intent.targets:
            results["dates"] = [e["text"] for e in ents if e["label"] == "DATE"]

    if "full_text" in intent.targets:
        results["full_text"] = text

    return results
