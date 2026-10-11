"""Deterministic source visitation, never an assertion of semantic atomicity."""

import re


# Punctuation conventions, not topic or dataset-specific vocabulary.
_ABBREVIATIONS = frozenset({"sr", "sra", "srta", "dr", "dra", "prof", "profa",
    "eng", "mr", "mrs", "ms", "jr", "st", "vs", "etc", "ex", "p", "pp"})
_TERMINAL = re.compile(r"[.!?][\"'“”‘’«»)\]}]*(?=\s|$)")


def source_sentence_spans(text):
    """Cover every non-whitespace character with original source offsets.

    Conservative punctuation boundaries preserve decimals, initials and common
    abbreviations. Final unpunctuated text is visited too. A span can be filler,
    a fragment, an atom or a compound; only semantic validation decides.
    """
    boundaries = []
    for match in _TERMINAL.finditer(text):
        if match.group()[0] == ".":
            preceding = re.search(r"([\w.]+)$", text[:match.start()])
            token = preceding.group(1) if preceding else ""
            if token.casefold() in _ABBREVIATIONS or (len(token) == 1 and token.isalpha() and token.isupper()):
                continue
            if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", token):
                continue
        boundaries.append(match.end())
    result, start = [], 0
    for end in [*boundaries, len(text)]:
        while start < end and text[start].isspace():
            start += 1
        trimmed = end
        while trimmed > start and text[trimmed - 1].isspace():
            trimmed -= 1
        if start < trimmed:
            result.append((start, trimmed))
        start = end
    return tuple(result)


def source_sentence_proposal(segment_id, text, start, end):
    """Copy source text as an unverified hypothesis with empty proposed metadata."""
    quote = text[start:end]
    return {"text": quote, "unit_type": "assertion",
            "qualifiers": {"attribution": None, "negated": False, "modality": None,
                           "quantities": [], "conditions": []},
            "evidence": [{"segment_id": segment_id, "quote": quote,
                          "start_char": start, "end_char": end, "role": "assertion"}],
            "question": None, "answer": None, "unresolved_references": []}
