"""Literal, scoped questions for the non-generative information evaluator.

Keep source text and proposed claims readable. Source identities, timestamps and
generation instructions belong in the report, not in the semantic comparison.
Qualifier fidelity and annotation correctness are separate decisions; an empty
annotation never disables a source-based check.
"""

import json


EVALUATION_TEMPLATE_VERSION = "literal-comparisons-v2"
VALIDATION_QUESTIONS = {
    "support": "Does the assertion evidence support the candidate claim? Use reference context only to resolve references. Evaluate what was communicated, not external truth.",
    "conditions": "Does the candidate preserve conditional rules and exceptions that apply to its own assertion? If there are none and none are added, answer yes. Other independent assertions in the evidence need not be included.",
    "negation": "Does the candidate claim preserve negation and its scope in the assertion evidence? If both are affirmative, answer yes.",
    "quantities": "Does the candidate claim preserve the quantities relevant to it in the assertion evidence? If none apply to this claim and none are added, answer yes.",
    "modality": "Does the candidate claim preserve the level of certainty in the assertion evidence?",
    "attribution": "Does the candidate claim preserve the speaker attribution in the assertion evidence? If no speaker is named, answer yes.",
}
ANNOTATION_ABSENCE_QUESTIONS = {
    "conditions_annotation": "Is the candidate claim free of explicit conditional rules and exceptions?",
    "quantities_annotation": "Is the candidate claim free of quantities, dates, ranks and comparisons of amount?",
    "modality_annotation": "Does the candidate communicate a definite assertion, including conditional rules, rather than uncertainty, possibility, obligation or speculation?",
    "attribution_annotation": "Is the candidate a narration without quoting or reporting a named source's words or opinion?",
}
ANNOTATION_VALUE_QUESTIONS = {
    "conditions_annotation": "Does {value} accurately describe the conditional rules or exceptions in the candidate claim?",
    "quantities_annotation": "Does {value} accurately describe the quantitative values, ranks or comparisons in the candidate claim?",
    "modality_annotation": "Does {value} accurately describe the modal or claim-status wording in the candidate claim?",
    "attribution_annotation": "Is {value} named as the speaker or claimant in the candidate claim?",
}
NEGATION_ANNOTATION_QUESTIONS = {
    True: "Does the candidate claim contain explicit negation, such as not, never, without, não, nem or sem?",
    False: "Is the candidate claim affirmative, without explicit negation such as not, never, without, não, nem or sem?",
}
GRANULARITY_CRITERIA = {
    "atomic": "One contextual proposition. A condition and its consequence form one proposition; an attributed claim includes its reporting source.",
    "compound": "Two or more independently meaningful propositions, such as backup frequency and retention duration.",
    "not_information": "No communicated propositional content, such as a greeting or filler.",
    "needs_context": "An unresolved reference prevents understanding the proposition using the provided evidence and reference context.",
}
RELATION_CRITERIA = {
    "equivalent": "The claims express the same complete meaning in both directions, including entities, attribution, quantities, negation, modality and conditions.",
    "complementary": "The claims express different compatible information; neither fully entails the other.",
    "more_specific_left": "The left claim entails the right, but the right omits a detail from the left.",
    "more_specific_right": "The right claim entails the left, but the left omits a detail from the right.",
    "contradiction": "The claims communicate incompatible assertions about the same entities and scope.",
    "correction_left": "The left claim explicitly corrects the right claim in the transcript.",
    "correction_right": "The right claim explicitly corrects the left claim in the transcript.",
    "uncertain": "The relationship cannot be established from the cited evidence and reference context.",
}
COVERAGE_QUESTIONS = {
    "missing": "Does the original target communicate a proposition or necessary qualifier not represented by the listed claims? A citation alone does not represent every proposition in it.",
    "missing_occurrence": "Is an asserted occurrence in the original target missing from the represented assertion evidence spans? Duplicate routes at the same span count once; separate repeated utterances need separate spans.",
}
QA_QUESTIONS = {
    "qa_anchor": "Is the candidate question answerable from the assertion evidence and reference context?",
    "qa_consistency": "Does the candidate answer communicate the same complete proposition as the candidate claim?",
}


def _evidence_text(candidate, role):
    return "\n".join(
        f"[{item.segment_id}] {item.quote}"
        for item in candidate.evidence
        if item.role == role
    ) or "None"


def validation_spec(candidate, candidate_id, target_id):
    """Return the exact state and all independent acceptance questions."""
    blocks = [
        f"Candidate id: {candidate_id}\nTarget id: {target_id}",
        "Assertion evidence:\n" + _evidence_text(candidate, "assertion"),
        "Reference context:\n" + _evidence_text(candidate, "context"),
        "Candidate claim:\n" + candidate.text,
    ]
    questions = dict(VALIDATION_QUESTIONS)
    qualifiers = candidate.qualifiers
    for field in ("conditions", "quantities", "modality", "attribution"):
        identifier = field + "_annotation"
        value = getattr(qualifiers, field)
        questions[identifier] = (
            ANNOTATION_VALUE_QUESTIONS[identifier].format(
                value=json.dumps("; ".join(value) if isinstance(value, tuple) else value, ensure_ascii=False)
            )
            if value else ANNOTATION_ABSENCE_QUESTIONS[identifier]
        )
    questions["negation_annotation"] = NEGATION_ANNOTATION_QUESTIONS[
        qualifiers.negated
    ]
    if candidate.question is not None:
        blocks += [
            "Candidate question:\n" + candidate.question,
            "Candidate answer:\n" + candidate.answer,
        ]
        questions.update(QA_QUESTIONS)
    return "\n\n".join(blocks), questions, {
        "granularity": ("Classify the candidate claim.", GRANULARITY_CRITERIA)
    }


def coverage_spec(target, context, accepted, markers):
    represented = []
    for record in accepted:
        candidate = record.candidate
        spans = ", ".join(
            f"{item.segment_id} [{item.start_char},{item.end_char})"
            for item in candidate.evidence if item.role == "assertion"
        )
        represented.append(
            f"{record.id}: {candidate.text}\nAssertion spans: {spans}\n"
            + "Assertion evidence: " + _evidence_text(candidate, "assertion")
        )
    blocks = [
        f"Original target [{target.id}]:\n{target.content}",
        "Current reference context:\n" + (
            "\n".join(f"[{item.id}] {item.content}" for item in context) or "None"
        ),
        "Represented claims:\n" + ("\n\n".join(represented) or "None"),
    ]
    questions = dict(COVERAGE_QUESTIONS)
    for index, (marker, start, end) in enumerate(markers):
        questions[f"focus{index}"] = (
            f"Is a relation or qualifier involving {json.dumps(marker, ensure_ascii=False)} "
            f"at original-target span [{start},{end}) missing from the represented claims? "
            "A marker alone is not a proposition."
        )
    return "\n\n".join(blocks), questions


def relation_state(left, right):
    blocks = []
    for label, record in (("Left", left), ("Right", right)):
        candidate = record.candidate
        blocks.extend((
            f"{label} candidate id: {record.id}",
            f"{label} assertion evidence:\n" + _evidence_text(candidate, "assertion"),
            f"{label} reference context:\n" + _evidence_text(candidate, "context"),
            f"{label} claim:\n" + candidate.text,
        ))
    return "\n\n".join(blocks)


def evaluation_templates():
    """Templates are persisted as hashes alongside generator prompt hashes."""
    return {
        "validation_questions": VALIDATION_QUESTIONS,
        "annotation_absence": ANNOTATION_ABSENCE_QUESTIONS,
        "annotation_values": ANNOTATION_VALUE_QUESTIONS,
        "negation_annotations": {str(key): value for key, value in NEGATION_ANNOTATION_QUESTIONS.items()},
        "granularity_criteria": GRANULARITY_CRITERIA,
        "coverage_questions": COVERAGE_QUESTIONS,
        "qa_questions": QA_QUESTIONS,
        "relation_criteria": RELATION_CRITERIA,
    }
