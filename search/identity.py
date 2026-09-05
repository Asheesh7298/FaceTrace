"""
search/identity.py
==================
Extract a person's NAME from a Google Lens `ai_overview` block.

Google Lens returns a natural-language identity summary such as:
    "This image appears to show acclaimed actor Tom Hanks."
    "The person in the image is Emma Watson, ..."
or a refusal:
    "I'm not able to identify the person in this image."

We turn that prose into a clean name using spaCy PERSON NER (primary) with a
regex heuristic fallback, so the pipeline still works if the spaCy model is
unavailable. Movie titles / character names in later sentences are avoided by
scoring the FIRST paragraph most heavily.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Phrases that mean "Google could not identify the person"
_REFUSAL = re.compile(
    r"(can'?t|cannot|not able to|unable to|couldn'?t|no way to)\s+.*identif",
    re.IGNORECASE,
)

# Words that look like names (Capitalized) but aren't people
_STOPWORDS = {
    "The", "This", "That", "He", "She", "It", "They", "Academy", "Awards",
    "Award", "Best", "Actor", "Actress", "American", "Indian", "British",
    "Would", "You", "Image", "Photo", "Their", "His", "Her",
}

_nlp = None
_nlp_tried = False


def _get_nlp():
    global _nlp, _nlp_tried
    if _nlp_tried:
        return _nlp
    _nlp_tried = True
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy NER loaded for identity extraction")
    except Exception as e:
        logger.warning(f"spaCy unavailable ({e}); using regex name fallback")
        _nlp = None
    return _nlp


def _blocks_to_text(ai_overview: dict) -> tuple[str, str]:
    """Return (first_paragraph, full_text) from an ai_overview dict."""
    if not ai_overview:
        return "", ""
    first_para = ""
    parts = []
    for block in ai_overview.get("text_blocks", []) or []:
        if block.get("type") == "paragraph":
            snip = (block.get("snippet") or "").strip()
            if snip:
                parts.append(snip)
                if not first_para:
                    first_para = snip
        elif block.get("type") == "list":
            for item in block.get("list", []) or []:
                snip = (item.get("snippet") or "").strip()
                if snip:
                    parts.append(snip)
    return first_para, " ".join(parts)


def _person_via_spacy(text: str) -> Optional[str]:
    nlp = _get_nlp()
    if not nlp or not text:
        return None
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if len(name.split()) >= 2 and name.split()[0] not in _STOPWORDS:
                return name
    return None


def _person_via_regex(text: str) -> Optional[str]:
    """Grab the first plausible 'Firstname Lastname' sequence."""
    if not text:
        return None
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text):
        name = m.group(1)
        if name.split()[0] in _STOPWORDS:
            continue
        return name
    return None


def extract_person_name(ai_overview: dict) -> Optional[str]:
    """
    Return the identified person's name, or None if Lens could not identify one.
    """
    first_para, full_text = _blocks_to_text(ai_overview)

    if _REFUSAL.search(full_text):
        logger.info("Lens ai_overview refused to identify the person")
        return None

    # First paragraph carries the identity; try it first, then the whole text.
    for scope in (first_para, full_text):
        name = _person_via_spacy(scope) or _person_via_regex(scope)
        if name:
            logger.info(f"Identity extracted: {name}")
            return name
    return None


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    ao = data.get("ai_overview_expanded") or data.get("ai_overview")
    print("NAME:", extract_person_name(ao))
