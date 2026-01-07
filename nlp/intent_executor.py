from typing import Dict, Any
from nlp.info_extraction import extract_emails, extract_phone_numbers, extract_named_entities

def execute_intent(text: str, intent, nlp) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    if "emails" in intent.targets:
        results["emails"] = sorted(set(extract_emails(text)))

    if "phones" in intent.targets:
        results["phones"] = sorted(set(extract_phone_numbers(text)))

    if any(t in intent.targets for t in ("persons", "orgs", "locations", "dates")):
        ents = extract_named_entities(text, nlp)

        if "persons" in intent.targets:
            results["persons"] = sorted([e["text"] for e in ents if e["label"] == "PERSON"])
        if "orgs" in intent.targets:
            results["orgs"] = sorted([e["text"] for e in ents if e["label"] == "ORG"])
        if "locations" in intent.targets:
            results["locations"] = sorted([e["text"] for e in ents if e["label"] in ("GPE", "LOC")])
        if "dates" in intent.targets:
            results["dates"] = sorted([e["text"] for e in ents if e["label"] == "DATE"])

    if "full_text" in intent.targets:
        results["full_text"] = text

    return results
