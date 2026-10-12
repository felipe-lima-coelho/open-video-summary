"""Source-grounded initial questions without another discovery pass."""

import json
from dataclasses import replace

from open_video_summary.core.summarizers.information_contracts import (
    InformationQuestion, canonical_json,
)
from open_video_summary.core.summarizers.information_evaluation import (
    passes_probability_cutoff,
)


def with_information_questions(report, threshold=0.85):
    """Retain validated QA; cover other accepted units with explicit templates.

    The template quotes the complete accepted proposition rather than guessing a
    natural-language question or stripping its uncertainty. It needs no generator
    request, so disabling independent QA discovery retains its existing meaning.
    """
    if report.questions:
        return report
    records = {record.id: record for record in report.candidates}
    rows = []
    uncovered = []
    for unit in report.units:
        occurrences = tuple(item for item in report.occurrences
                            if item.unit_id == unit.id and item.assertion_evidence)
        if not occurrences:
            uncovered.append(unit.id)
            continue
        proposals = {}
        for identifier in unit.candidate_ids:
            record = records.get(identifier)
            if record is None or record.validation != "accepted" or record.candidate is None:
                continue
            candidate = record.candidate
            signals = dict(record.signals)
            if (not candidate.question or not candidate.answer
                    or not all(key in signals and passes_probability_cutoff(signals[key], threshold)
                               for key in ("qa_anchor", "qa_consistency"))):
                continue
            key = (candidate.question, candidate.answer)
            proposals.setdefault(key, []).append(record)
        if not proposals:
            occurrence = occurrences[0]
            rows.append(InformationQuestion(
                f"q{len(rows)}", unit.id,
                f"What does the source communicate in this proposition: “{unit.text}”?",
                unit.text, unit.candidate_ids, tuple(item.id for item in occurrences),
                occurrence.assertion_evidence, occurrence.context_evidence,
                "templated_proposition",
            ))
            continue
        for (question, answer), candidates in proposals.items():
            evidence = candidates[0].candidate.evidence
            rows.append(InformationQuestion(
                f"q{len(rows)}", unit.id, question, answer,
                tuple(record.id for record in candidates),
                tuple(item.id for item in occurrences),
                tuple(item for item in evidence if item.role == "assertion"),
                tuple(item for item in evidence if item.role == "context"),
                "validated_qa", tuple((key, dict(candidates[0].signals)[key])
                                       for key in ("qa_anchor", "qa_consistency")),
            ))
    metadata = json.loads(report.metadata_json)
    metadata["question_inventory"] = {
        "scope": "accepted_initial_units",
        "validated_qa_questions": sum(row.origin == "validated_qa" for row in rows),
        "templated_proposition_questions": sum(row.origin == "templated_proposition" for row in rows),
        "uncovered_unit_ids": uncovered,
        "counts_as_information": False,
        "completeness_proven": False,
        "semantics": "Validated source QA is retained with its complete answer and literal evidence. "
                     "Direct-only units use an explicitly labeled query quoting their complete "
                     "accepted proposition. Queries preserve communicated certainty and do not "
                     "add units, assert external truth or establish complete source coverage.",
    }
    return replace(report, schema_version=max(11, report.schema_version),
                   questions=tuple(rows), metadata_json=canonical_json(metadata))
