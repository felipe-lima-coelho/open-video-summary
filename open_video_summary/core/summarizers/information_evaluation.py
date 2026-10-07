"""Literal, scoped questions for the non-generative information evaluator.

Keep source text and proposed claims readable. Source identities, timestamps and
generation instructions belong in the report, not in the semantic comparison.
Qualifier fidelity and annotation correctness are separate decisions; an empty
annotation never disables a source-based check.
"""

import json
import math


EVALUATION_TEMPLATE_VERSION = "source-scope-fidelity-v6"
SCOPE_INSTRUCTION = "The markers select exact source occurrences, not complete propositions. Interpret each marked assertion with its governing wording in the whole original source. Other independent assertions are not support for the candidate. Marked reference context resolves references only."
ANCHOR_BINDING_QUESTION = "Does the candidate claim refer to the property or event expressed by the selected assertion wording? Governing conditions, attribution, modality and negation may be outside the selected wording and are checked separately."
VALIDATION_QUESTIONS = {
    "support": "Does the original source support the candidate claim at the selected anchor, including governing wording outside that anchor? Reference context may resolve references but may not supply a new assertion. Evaluate communicated meaning, not external truth.",
    "conditions": "Does the candidate preserve conditions and exceptions governing its anchored assertion in the original source scope? If none govern this assertion and none are added, answer yes. Unrelated assertions need not be included.",
    "negation": "Does the candidate claim preserve the polarity of this assertion in the original source, including negation outside the selected anchor? If both are affirmative, answer yes.",
    "quantities": "Does the candidate claim preserve this assertion's quantities, ranks and quantitative comparisons in the original source? If none apply and none are added, answer yes. Ignore quantities of other independent assertions.",
    "modality": "Does the candidate claim preserve the source assertion's level of certainty? If both make a definite assertion, answer yes.",
    "attribution": "Does the candidate claim preserve who reports or claims this assertion in the original source? If no reporting source is present or added, answer yes.",
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
    "qa_anchor": "Is the candidate question answerable from the selected assertion in its original source and cited reference context?",
    "qa_consistency": "Does the candidate answer communicate the same complete proposition as the candidate claim?",
}
RELATION_ADJUDICATION_QUESTIONS = {
    "left_entails_right": "Does the complete Left claim entail every part of the Right claim, including entities, scope, attribution, quantities, polarity, certainty, conditions and exceptions? Mere topical similarity is not entailment. Compare communicated content, not external truth.",
    "right_entails_left": "Does the complete Right claim entail every part of the Left claim, including entities, scope, attribution, quantities, polarity, certainty, conditions and exceptions? Mere topical similarity is not entailment. Compare communicated content, not external truth.",
    "incompatible": "Do the claims make incompatible assertions about the same entities, event or property and temporal scope? Different compatible or unrelated properties are not a contradiction. Preserve modal possibilities as possibilities.",
    "correction_left": "Does the original source explicitly present the Left assertion as a correction of the Right assertion? Chronological order or different quantities alone do not establish a correction.",
    "correction_right": "Does the original source explicitly present the Right assertion as a correction of the Left assertion? Chronological order or different quantities alone do not establish a correction.",
    "same_complete_meaning": "After resolving references using only each claim's cited evidence, do the Left and Right claims state exactly the same complete informational proposition? Each must entail every detail expressed by the other. Preserve who makes an allegation, the asserted action and its actor, quantities, negation, modality, conditions, time and scope. A fact present only in surrounding source and not expressed by a claim is not represented by that claim. Sharing an underlying allegation or topic is insufficient when one claim additionally states an actor, reporting source or action. Synonymous wording may be equivalent; additional communicated details are not.",
}
RELATION_SUBTYPE_CHECKS = ("left_entails_right", "right_entails_left", "incompatible",
                           "correction_left", "correction_right")
RELATION_BATCH_QUESTION = (
    "Evaluate only {pair_id}, Left candidate {left_id} and Right candidate {right_id}. "
    "Only their own marked source scopes and cited reference context may be used; "
    "assertions in other pairs supply no evidence. "
)


def passes_probability_cutoff(probability, threshold):
    """Include finite-precision boundary equality, within two representation steps."""
    return probability >= threshold or math.isclose(probability, threshold,
        rel_tol=0.0, abs_tol=2 * max(math.ulp(probability), math.ulp(threshold)))


def resolve_equivalence(signals, threshold):
    """Separate mutual-entailment certainty from descriptive relation subtypes."""
    yes = lambda key: key in signals and passes_probability_cutoff(signals[key], threshold)
    no = lambda key: key in signals and passes_probability_cutoff(1 - signals[key], threshold)
    mutual = yes("left_entails_right") and yes("right_entails_left")
    negative = [key for key in ("left_entails_right", "right_entails_left") if no(key)]
    if ((negative and yes("same_complete_meaning"))
            or (mutual and no("same_complete_meaning"))
            or ((mutual or yes("same_complete_meaning"))
                and any(yes(key) for key in ("incompatible", "correction_left", "correction_right")))):
        return "uncertain", None, "conflicting_adjudication_signals"
    if negative:
        return "distinct", max(1 - signals[key] for key in negative), "directional_non_entailment"
    if no("same_complete_meaning"):
        return "distinct", 1 - signals["same_complete_meaning"], "complete_meaning_check"
    if mutual:
        return "equivalent", min(signals["left_entails_right"], signals["right_entails_left"]), "mutual_entailment"
    if yes("same_complete_meaning"):
        return "equivalent", signals["same_complete_meaning"], "complete_meaning_check"
    return "uncertain", None, "unresolved_adjudication_signals"


def resolve_relation(signals, threshold):
    """Require all defining checks for a subtype, independently of equivalence."""
    yes = lambda key: passes_probability_cutoff(signals[key], threshold)
    no = lambda key: passes_probability_cutoff(1 - signals[key], threshold)
    if yes("correction_left") and no("correction_right"):
        label = "correction_left"
        defining = ("correction_left", "correction_right")
    elif yes("correction_right") and no("correction_left"):
        label = "correction_right"
        defining = ("correction_left", "correction_right")
    elif no("correction_left") and no("correction_right"):
        defining = RELATION_SUBTYPE_CHECKS
        if yes("incompatible") and no("left_entails_right") and no("right_entails_left"):
            label = "contradiction"
        elif no("incompatible"):
            if yes("left_entails_right") and yes("right_entails_left"):
                label = "equivalent"
            elif yes("left_entails_right") and no("right_entails_left"):
                label = "more_specific_left"
            elif no("left_entails_right") and yes("right_entails_left"):
                label = "more_specific_right"
            elif no("left_entails_right") and no("right_entails_left"):
                label = "complementary"
            else:
                return "uncertain", None
        else:
            return "uncertain", None
    else:
        return "uncertain", None
    strength = min(max(signals[key], 1 - signals[key]) for key in defining)
    return label, strength


def _evidence_text(candidate, role):
    return "\n".join(
        f"[{item.segment_id} [{item.start_char},{item.end_char})] {item.quote}"
        for item in candidate.evidence
        if item.role == role
    ) or "None"


def _marked_source(segment, evidence):
    """Mark exact positions so repeated quotations keep their distinct scopes."""
    events = {}
    for index, item in enumerate(evidence):
        if item.segment_id != segment.id:
            continue
        events.setdefault(item.start_char, []).append((1, -item.end_char, f"<source-{item.role}-{index}>"))
        events.setdefault(item.end_char, []).append((0, -item.start_char, f"</source-{item.role}-{index}>"))
    parts, previous = [], 0
    for position in sorted(events):
        parts.append(segment.content[previous:position])
        parts.extend(event[2] for event in sorted(events[position]))
        previous = position
    parts.append(segment.content[previous:])
    return "".join(parts)


def anchor_binding_spec(candidate):
    """Do not expose unrelated source assertions to the anchor-binding decision."""
    blocks = [
        "Selected assertion wording:\n" + "\n".join(
            item.quote for item in candidate.evidence if item.role == "assertion"
        ),
        "Candidate claim:\n" + candidate.text,
    ]
    if any(item.role == "context" for item in candidate.evidence):
        blocks.append("Cited literal reference context (resolves references only; cannot supply a new assertion):\n"
                      + _evidence_text(candidate, "context"))
    return "\n\n".join(blocks), {"anchor_binding": ANCHOR_BINDING_QUESTION}


def _source_scope(candidate, target, context):
    """Use whole source-owned segments, including surrounding cited context.

    The envelope governs interpretation of the anchored assertion. Its unrelated
    propositions do not become assertion evidence for this candidate.
    """
    cited = {item.segment_id for item in candidate.evidence if item.role == "context"}
    return [
        f"Original target scope [{target.id}]:\n{_marked_source(target, candidate.evidence)}",
        "Original reference scope:\n" + (
            "\n".join(f"[{item.id}] {_marked_source(item, candidate.evidence)}" for item in context if item.id in cited)
            or "None"
        ),
    ]


def validation_spec(candidate, candidate_id, target, context=()):
    """Return source-scope fidelity checks and a separate annotation audit."""
    blocks = [
        f"Candidate id: {candidate_id}\nTarget id: {target.id}",
        SCOPE_INSTRUCTION,
        *_source_scope(candidate, target, context),
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


def relation_state(left, right, segments):
    blocks = []
    for label, record in (("Left", left), ("Right", right)):
        candidate = record.candidate
        blocks.extend((
            f"{label} candidate id: {record.id}",
            SCOPE_INSTRUCTION,
            *(
                f"{label} {block}" for block in _source_scope(
                    candidate, segments[record.target_segment_id],
                    tuple(segments[identifier] for identifier in dict.fromkeys(
                        item.segment_id for item in candidate.evidence if item.role == "context"
                    )),
                )
            ),
            f"{label} claim:\n" + candidate.text,
        ))
    return "\n\n".join(blocks)


def relation_batch_spec(pairs, segments, instruction):
    """Format the same pair/source scopes for individual and keyed evaluation."""
    if len(pairs) == 1:
        return relation_state(*pairs[0], segments), {
            "relation": (instruction, RELATION_CRITERIA)
        }
    blocks, choices = [], {}
    for index, (left, right) in enumerate(pairs):
        identifier = f"pair{index}"
        blocks.append(f"BEGIN {identifier}\n{relation_state(left, right, segments)}\nEND {identifier}")
        choices[identifier] = (RELATION_BATCH_QUESTION.format(
            pair_id=identifier, left_id=left.id, right_id=right.id) + instruction,
            RELATION_CRITERIA)
    return "\n\n".join(blocks), choices


def evaluation_templates():
    """Templates are persisted as hashes alongside generator prompt hashes."""
    return {
        "validation_questions": VALIDATION_QUESTIONS,
        "source_scope_instruction": SCOPE_INSTRUCTION,
        "anchor_binding_question": ANCHOR_BINDING_QUESTION,
        "annotation_absence": ANNOTATION_ABSENCE_QUESTIONS,
        "annotation_values": ANNOTATION_VALUE_QUESTIONS,
        "negation_annotations": {str(key): value for key, value in NEGATION_ANNOTATION_QUESTIONS.items()},
        "granularity_criteria": GRANULARITY_CRITERIA,
        "coverage_questions": COVERAGE_QUESTIONS,
        "qa_questions": QA_QUESTIONS,
        "relation_criteria": RELATION_CRITERIA,
        "relation_adjudication_questions": RELATION_ADJUDICATION_QUESTIONS,
        "relation_batch_question": RELATION_BATCH_QUESTION,
    }
