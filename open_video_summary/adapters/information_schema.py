"""Strict generation schema for transcript-grounded information candidates."""

from open_video_summary.contracts import OutputSpec
from open_video_summary.errors import InvalidResponseError


UNIT_TYPES = (
    "assertion",
    "definition",
    "procedure",
    "conditional",
    "opinion",
    "recommendation",
    "hypothesis",
)
QUALIFIER_FIELDS = ("attribution", "negated", "modality", "quantities", "conditions")
CANDIDATE_FIELDS = (
    "text",
    "unit_type",
    "qualifiers",
    "evidence",
    "question",
    "answer",
    "unresolved_references",
)


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def information_schema(spec: OutputSpec) -> dict:
    text = {"type": "string", "minLength": 1}
    strings = {"type": "array", "items": text}
    identifier = dict(text)
    if spec.segment_ids:
        identifier["enum"] = list(spec.segment_ids)
    evidence = _object(
        {
            "segment_id": identifier,
            "quote": text,
            "start_char": {"type": "integer", "minimum": 0},
            "end_char": {"type": "integer", "minimum": 1},
            "role": {"type": "string", "enum": ["assertion", "context"]},
        }
    )
    qa = text if spec.kind == "information_qa" else {"type": ["string", "null"]}
    candidate = _object(
        {
            "text": text,
            "unit_type": {"type": "string", "enum": list(UNIT_TYPES)},
            "qualifiers": _object(
                {
                    "attribution": {"type": ["string", "null"]},
                    "negated": {"type": "boolean"},
                    "modality": {"type": ["string", "null"]},
                    "quantities": strings,
                    "conditions": strings,
                }
            ),
            "evidence": {"type": "array", "items": evidence, "minItems": 1},
            "question": qa,
            "answer": qa,
            "unresolved_references": strings,
        }
    )
    candidates = {"type": "array", "items": candidate}
    if spec.max_items is not None:
        candidates["maxItems"] = spec.max_items
    issue = _object(
        {
            "kind": text,
            "detail": text,
            "segment_ids": {"type": "array", "items": identifier},
        }
    )
    return _object(
        {"candidates": candidates, "issues": {"type": "array", "items": issue}}
    )


def validate_information(value, spec: OutputSpec):
    """Validate types locally even when the provider promises a JSON schema."""

    def invalid():
        raise InvalidResponseError(
            "The language model returned invalid information candidates."
        )

    def string(item):
        return isinstance(item, str) and bool(item.strip())

    def strings(item):
        return isinstance(item, list) and all(string(part) for part in item)

    if not isinstance(value, dict) or set(value) != {"candidates", "issues"}:
        invalid()
    candidates, issues = value["candidates"], value["issues"]
    if not isinstance(candidates, list) or not isinstance(issues, list):
        invalid()
    if spec.max_items is not None and len(candidates) > spec.max_items:
        invalid()
    for item in candidates:
        if not isinstance(item, dict) or set(item) != set(CANDIDATE_FIELDS):
            invalid()
        if not string(item["text"]) or item["unit_type"] not in UNIT_TYPES:
            invalid()
        qualifiers = item["qualifiers"]
        if not isinstance(qualifiers, dict) or set(qualifiers) != set(QUALIFIER_FIELDS):
            invalid()
        if not isinstance(qualifiers["negated"], bool):
            invalid()
        if any(
            qualifiers[key] is not None and not string(qualifiers[key])
            for key in ("attribution", "modality")
        ):
            invalid()
        if not all(strings(qualifiers[key]) for key in ("quantities", "conditions")):
            invalid()
        for key in ("question", "answer"):
            if spec.kind == "information_qa":
                if not string(item[key]):
                    invalid()
            elif item[key] is not None and not string(item[key]):
                invalid()
        if not strings(item["unresolved_references"]):
            invalid()
        if not isinstance(item["evidence"], list) or not item["evidence"]:
            invalid()
        for evidence in item["evidence"]:
            if not isinstance(evidence, dict) or set(evidence) != {
                "segment_id",
                "quote",
                "start_char",
                "end_char",
                "role",
            }:
                invalid()
            if not string(evidence["segment_id"]) or not string(evidence["quote"]):
                invalid()
            if spec.segment_ids and evidence["segment_id"] not in spec.segment_ids:
                invalid()
            if (
                type(evidence["start_char"]) is not int
                or type(evidence["end_char"]) is not int
            ):
                invalid()
            if not 0 <= evidence["start_char"] < evidence["end_char"]:
                invalid()
            if not isinstance(evidence["role"], str) or evidence["role"] not in {
                "assertion",
                "context",
            }:
                invalid()
    for issue in issues:
        if not isinstance(issue, dict) or set(issue) != {
            "kind",
            "detail",
            "segment_ids",
        }:
            invalid()
        if (
            not string(issue["kind"])
            or not string(issue["detail"])
            or not strings(issue["segment_ids"])
        ):
            invalid()
        if spec.segment_ids and any(
            identifier not in spec.segment_ids for identifier in issue["segment_ids"]
        ):
            invalid()
    return value
