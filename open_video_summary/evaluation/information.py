"""Evaluate frozen information reports against source-grounded references.

Semantic alignments and literal anchors are different evidence. Only reviewed
alignments or identical complete text with identical source anchors contribute
to coverage. Unmatched paraphrases remain unresolved, rather than becoming false
negatives. Provider decisions are predictions, never the reference labels.
"""

import hashlib
import json
import math
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from open_video_summary.utils.paths import portable_path, project_path


EVALUATION_SCHEMA_VERSION = 1
UNIT_RELATIONS = {
    "equivalent",
    "split",
    "merged",
    "compound_plus_components",
    "partial",
    "unresolved",
}
OCCURRENCE_RELATIONS = {
    "equivalent",
    "split_anchor",
    "merged_occurrences",
    "unresolved",
}
QUALIFIER_FIELDS = {"attribution", "negated", "modality", "quantities", "conditions"}


def evaluation_fingerprint(value):
    """Hash canonical JSON content, independent of indentation and key order."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value):
    # Preserve case, punctuation, quantities and polarity. This is not a semantic
    # similarity heuristic and deliberately does not remove Portuguese accents.
    return " ".join(unicodedata.normalize("NFC", value).split())


def _index(rows, label):
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be an array.")
    indexed = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
        ):
            raise ValueError(f"{label} records require nonempty string ids.")
        if row["id"] in indexed:
            raise ValueError(f"{label} contains duplicate id {row['id']}.")
        indexed[row["id"]] = row
    return indexed


def _ids(row, key, known, *, required=True):
    values = row.get(key, [])
    if not isinstance(values, list) or any(
        not isinstance(item, str) for item in values
    ):
        raise ValueError(f"{key} must contain string ids.")
    if len(values) != len(set(values)) or not set(values) <= set(known):
        raise ValueError(f"{key} contains duplicate or unknown ids.")
    if required and not values:
        raise ValueError(f"{key} must not be empty.")
    return values


def _source(snapshot, label):
    if not isinstance(snapshot, dict):
        raise ValueError(f"{label}.snapshot must be an object.")
    videos = _index(snapshot.get("source"), f"{label}.source")
    segments = {}
    for video in videos.values():
        for identifier, segment in _index(
            video.get("segments"), f"{label}.segments"
        ).items():
            if identifier in segments or not isinstance(segment.get("content"), str):
                raise ValueError(
                    f"{label} has duplicate segments or missing transcript text."
                )
            if segment.get("video_id", video["id"]) != video["id"]:
                raise ValueError(f"{label} segment belongs to the wrong video.")
            segments[identifier] = segment
    order = snapshot.get("current_order", list(segments))
    if (
        not isinstance(order, list)
        or len(set(order)) != len(order)
        or not set(order) <= set(segments)
    ):
        raise ValueError(f"{label}.current_order has duplicate or unknown segments.")
    return segments, order


def _evidence_errors(evidence, segments, default_segment, owner):
    errors = []
    if not isinstance(evidence, list) or not evidence:
        return [{"owner": owner, "error": "missing_assertion_evidence"}]
    for index, anchor in enumerate(evidence):
        segment_id = (
            anchor.get("segment_id", default_segment)
            if isinstance(anchor, dict)
            else None
        )
        segment = segments.get(segment_id)
        if segment is None:
            errors.append(
                {"owner": owner, "anchor_index": index, "error": "unknown_segment"}
            )
            continue
        start, end, quote = (
            anchor.get("start_char"),
            anchor.get("end_char"),
            anchor.get("quote"),
        )
        if (
            type(start) is not int
            or type(end) is not int
            or not isinstance(quote, str)
            or not quote
            or not 0 <= start < end <= len(segment["content"])
            or segment["content"][start:end] != quote
        ):
            errors.append(
                {
                    "owner": owner,
                    "anchor_index": index,
                    "error": "nonliteral_or_invalid_span",
                }
            )
        identity = anchor.get("source_identity")
        if identity is not None and identity != segment.get("source_identity"):
            errors.append(
                {
                    "owner": owner,
                    "anchor_index": index,
                    "error": "wrong_source_identity",
                }
            )
        for field in ("segment_start", "segment_end"):
            source_field = "start" if field == "segment_start" else "end"
            if field in anchor and anchor[field] != segment.get(source_field):
                errors.append(
                    {
                        "owner": owner,
                        "anchor_index": index,
                        "error": "wrong_segment_timestamp",
                    }
                )
    return errors


def _anchors(occurrence, field):
    return frozenset(
        (
            anchor.get("segment_id", occurrence["segment_id"]),
            anchor["start_char"],
            anchor["end_char"],
        )
        for anchor in occurrence.get(field, [])
    )


def _same_occurrence(reference, predicted):
    if reference["segment_id"] != predicted["segment_id"]:
        return False
    if _anchors(reference, "assertion_evidence") != _anchors(
        predicted, "assertion_evidence"
    ):
        return False
    # Governing reference context must also be traceable in the predicted record.
    context = _anchors(predicted, "context_evidence") | _anchors(
        predicted, "assertion_evidence"
    )
    return all(
        any(
            segment == other and start >= left and end <= right
            for other, left, right in context
        )
        for segment, start, end in _anchors(reference, "context_evidence")
    )


def _overlapping_occurrence(reference, predicted):
    return reference["segment_id"] == predicted["segment_id"] and any(
        segment == other and max(start, left) < min(end, right)
        for segment, start, end in _anchors(reference, "assertion_evidence")
        for other, left, right in _anchors(predicted, "assertion_evidence")
    )


def _reviewed(row, provenance):
    """Keep model-generated mappings unverified until an independent review."""
    if row.get("review_status") != "verified":
        return False
    if provenance.get("reviewer_status") != "reviewed" or not provenance.get(
        "reviewer"
    ):
        return False
    if provenance.get("annotator_type") not in {"agent", "human", "synthetic"}:
        return False
    if provenance.get("model_generated") and not provenance.get(
        "independently_reviewed"
    ):
        return False
    return True


def _qualifier_errors(row):
    errors = row.get("qualifier_errors", [])
    if not isinstance(errors, list):
        raise ValueError("qualifier_errors must be an array.")
    for error in errors:
        if (
            not isinstance(error, dict)
            or error.get("field") not in QUALIFIER_FIELDS
            or error.get("kind") not in {"content", "annotation"}
            or not isinstance(error.get("detail"), str)
        ):
            raise ValueError("Qualifier errors require field, kind and detail.")
    return errors


def _validate_link(row, reference, predicted, relations, occurrence=False):
    ref_key = "reference_occurrence_ids" if occurrence else "reference_unit_ids"
    out_key = "report_occurrence_ids" if occurrence else "report_unit_ids"
    refs, outs = _ids(row, ref_key, reference), _ids(row, out_key, predicted)
    relation = row.get("relation")
    if relation not in relations:
        raise ValueError(f"Unknown alignment relation {relation!r}.")
    if relation == "equivalent" and (len(refs) != 1 or len(outs) != 1):
        raise ValueError(
            "Equivalent links must align one complete unit or occurrence to one."
        )
    if relation in {"split", "split_anchor"} and (len(refs) != 1 or len(outs) < 2):
        raise ValueError(
            "Split links require one reference and multiple report records."
        )
    if relation in {"merged", "merged_occurrences"} and (
        len(refs) < 2 or len(outs) != 1
    ):
        raise ValueError(
            "Merged links require multiple references and one report record."
        )
    if relation == "compound_plus_components":
        parents = _ids(row, "parent_report_unit_ids", predicted)
        components = _ids(row, "component_report_unit_ids", predicted)
        if (
            len(refs) < 2
            or len(components) < 2
            or set(parents) & set(components)
            or set(parents + components) != set(outs)
        ):
            raise ValueError(
                "Compound links must identify disjoint counted parents and components."
            )
        if row.get("decomposition_verified") is not True:
            raise ValueError(
                "Compound-plus-components errors require an explicitly verified decomposition."
            )
    if not occurrence:
        errors = _qualifier_errors(row)
        if relation in {
            "equivalent",
            "split",
            "merged",
            "compound_plus_components",
        } and any(error["kind"] == "content" for error in errors):
            raise ValueError(
                "A content qualifier error cannot be complete equivalence; use partial."
            )
    return refs, outs


def _decisions(rows, key, known, allowed, provenance):
    verified = {}
    unverified = []
    for row in rows:
        identifier = row.get(key)
        if identifier not in known or row.get("state") not in allowed:
            raise ValueError(f"Invalid reviewed decision for {key}.")
        if _reviewed(row, provenance):
            if identifier in verified:
                raise ValueError(f"Conflicting duplicate decisions for {identifier}.")
            verified[identifier] = row
        else:
            unverified.append(row)
    return verified, unverified


def _fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def _coverage(states):
    counts = Counter(states)
    total = len(states)
    unresolved = counts["unresolved"]
    return {
        "reference_total": total,
        "covered": counts["covered"],
        "confirmed_omitted": counts["omitted"],
        "confirmed_partial": counts["partial"],
        "unresolved": unresolved,
        "reviewed_fraction": _fraction(total - unresolved, total),
        "coverage_lower_bound": _fraction(counts["covered"], total),
        "coverage_upper_bound": _fraction(counts["covered"] + unresolved, total),
        "recall_on_resolved_references": _fraction(
            counts["covered"], total - unresolved
        ),
        "bounds_assumption": "The upper bound allows every unresolved reference to have a full match; partial and omitted decisions are exhaustive reviewed nonmatches.",
    }


def _integrity(report, units, occurrences, candidates):
    errors = []
    counts = report.get("counts", {})
    actual = {
        "candidates": len(candidates),
        "accepted_candidates": sum(
            row.get("validation") == "accepted" for row in candidates.values()
        ),
        "unique_units": len(units),
        "occurrences": len(occurrences),
    }
    for name, value in actual.items():
        if name in counts and counts[name] != value:
            errors.append(
                {
                    "error": "aggregate_count_mismatch",
                    "field": name,
                    "recorded": counts[name],
                    "actual": value,
                }
            )
    for name, scope_field in (("by_segment", "segment_id"), ("by_video", "video_id")):
        segments, _ = _source(report["snapshot"], "report")
        grouped = defaultdict(list)
        for occurrence in occurrences.values():
            segment = segments.get(occurrence["segment_id"], {})
            scope = (
                occurrence["segment_id"]
                if scope_field == "segment_id"
                else segment.get("video_id")
            )
            grouped[scope].append(occurrence)
        for row in counts.get(name, []):
            scoped = grouped[row["id"]]
            for field, value in (
                ("occurrences", len(scoped)),
                ("unique_units", len({item["unit_id"] for item in scoped})),
            ):
                if row.get(field) != value:
                    errors.append(
                        {
                            "error": "scope_count_mismatch",
                            "scope": row["id"],
                            "field": field,
                            "recorded": row.get(field),
                            "actual": value,
                        }
                    )
    return errors


def _criteria(report, alignment, provenance, candidates):
    labels, skipped = [], []
    seen = set()
    threshold = (
        report.get("metadata", {}).get("settings", {}).get("acceptance_threshold")
    )
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Recorded acceptance threshold must be a finite probability.")
    state_confusion = Counter()
    granularity_confusion = Counter()
    observations = defaultdict(list)
    for row in alignment.get("candidate_labels", []):
        identifier = row.get("candidate_id")
        if identifier not in candidates or identifier in seen:
            raise ValueError(
                "Candidate labels contain unknown or duplicate candidate ids."
            )
        seen.add(identifier)
        if not _reviewed(row, provenance):
            skipped.append(identifier)
            continue
        expected = row.get("expected_validation")
        if expected not in {"accepted", "rejected", "uncertain"}:
            raise ValueError(
                "Candidate reference validation must be accepted, rejected or uncertain."
            )
        candidate = candidates[identifier]
        predicted = candidate.get("validation", "not_evaluated")
        state_confusion[(expected, predicted)] += 1
        label = {
            "candidate_id": identifier,
            "expected": expected,
            "predicted": predicted,
            "selection_basis": row.get("selection_basis", "unspecified"),
        }
        expected_granularity = row.get("expected_granularity")
        if expected_granularity is not None:
            granularity_confusion[
                (expected_granularity, candidate.get("granularity") or "unknown")
            ] += 1
        signals = dict(candidate.get("signals", []))
        signals.update(dict(candidate.get("annotation_signals", [])))
        gate_labels = row.get("gates", {})
        if not isinstance(gate_labels, dict):
            raise ValueError("Candidate gate reference labels must be an object.")
        for gate, expected_yes in gate_labels.items():
            if expected_yes is not None and type(expected_yes) is not bool:
                raise ValueError(
                    "Gate labels must be true, false or null for unresolved."
                )
            probability = signals.get(gate)
            if probability is not None and (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Recorded gate signals must be finite probabilities.")
            observations[gate].append((probability, expected_yes, identifier))
        labels.append(label)
    gates = {}
    for gate, rows in sorted(observations.items()):
        cells = Counter()
        bins = defaultdict(list)
        scored = []
        unavailable = []
        for probability, expected, identifier in rows:
            if probability is None or expected is None or threshold is None:
                unavailable.append(identifier)
                continue
            yes = probability >= threshold
            cells[(expected, yes)] += 1
            scored.append((probability, expected))
            bins[min(int(probability * 10), 9)].append((probability, expected))
        reliability = []
        for index, values in sorted(bins.items()):
            mean = sum(value for value, _ in values) / len(values)
            rate = sum(label for _, label in values) / len(values)
            reliability.append(
                {
                    "lower": index / 10,
                    "upper": (index + 1) / 10,
                    "upper_inclusive": index == 9,
                    "count": len(values),
                    "mean_probability": mean,
                    "observed_yes_fraction": rate,
                }
            )
        gates[gate] = {
            "recorded_threshold": threshold,
            "scored": len(scored),
            "true_positive": cells[(True, True)],
            "false_positive": cells[(False, True)],
            "true_negative": cells[(False, False)],
            "false_negative": cells[(True, False)],
            "unresolved_or_unavailable_candidate_ids": unavailable,
            "reliability_bins": reliability,
            "brier_score": _fraction(
                sum((probability - expected) ** 2 for probability, expected in scored),
                len(scored),
            ),
            "sample_expected_calibration_error": _fraction(
                sum(
                    item["count"]
                    * abs(item["mean_probability"] - item["observed_yes_fraction"])
                    for item in reliability
                ),
                len(scored),
            ),
        }

    def confusion(cells):
        return [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(cells.items())
        ]

    return {
        "reviewed_candidate_count": len(labels),
        "unverified_candidate_ids": skipped,
        "validation_confusion_matrix": confusion(state_confusion),
        "granularity_confusion_matrix": confusion(granularity_confusion),
        "gates": gates,
        "candidate_results": labels,
        "thresholds_calibrated": False,
        "sampling": alignment.get(
            "sampling", "unspecified; selection bias cannot be ruled out"
        ),
        "interpretation": "These are descriptive tables for explicitly reviewed labels. Targeted controls, small samples and agent labels do not establish population accuracy or calibration. No threshold is tuned.",
    }


def _experiment(report):
    metadata = report.get("metadata", {})
    settings = metadata.get("settings", {})
    mode = metadata.get("experimental_mode", settings.get("experimental_mode"))
    inferred = False
    if mode is None:
        inferred = True
        mode = (
            "hybrid_coverage"
            if settings.get("qa_enabled", True)
            and settings.get("max_coverage_rounds", 0)
            else "hybrid" if settings.get("qa_enabled", True) else "direct"
        )
    if mode == "hybrid_recovery":
        mode = "hybrid_coverage"
    return {
        "mode": mode,
        "mode_inferred_from_legacy_settings": inferred,
        "generator": metadata.get("generator"),
        "evaluator": metadata.get("evaluator"),
        "evaluator_provider": metadata.get("evaluator_provider"),
        "settings": settings,
        "prompt_template_hashes": metadata.get("prompt_template_hashes"),
        "evaluation_template_version": metadata.get("evaluation_template_version"),
    }


def _cost(report):
    metadata = report.get("metadata", {})
    totals = defaultdict(
        lambda: {
            "logical_calls": 0,
            "completed_calls": 0,
            "physical_attempts": 0,
            "reported_input_tokens": 0,
            "reported_output_tokens": 0,
            "calls_with_reported_input_tokens": 0,
            "calls_with_reported_output_tokens": 0,
        }
    )
    for call in report.get("calls", []):
        group = totals[call.get("provider", "unknown")]
        group["logical_calls"] += 1
        group["completed_calls"] += call.get("status") == "completed"
        usage = call.get("metadata", {})
        group["physical_attempts"] += usage.get("attempts", 0) or 0
        for field in ("input_tokens", "output_tokens"):
            value = usage.get(field)
            if type(value) is int and value >= 0:
                group[f"reported_{field}"] += value
                group[f"calls_with_reported_{field}"] += 1
    return {
        "duration_seconds": metadata.get("duration_seconds"),
        "logical_calls": metadata.get("logical_calls", len(report.get("calls", []))),
        "physical_attempts": metadata.get("physical_attempts"),
        "wait_seconds": metadata.get("wait_seconds"),
        "by_provider": dict(totals),
        "monetary_cost": None,
        "monetary_cost_note": "No pricing assumptions; missing token usage is not zero cost.",
    }


def evaluate_information_report(
    report, reference, alignment=None, *, allow_protocol_mismatch=False
):
    """Return auditable metrics from JSON objects; never call a model or service.

    A verified alignment must bind both canonical document fingerprints. It must
    carry reviewer provenance distinct from the report's evaluator output.
    Explicit omission and partial decisions mean that the current report has
    been exhaustively inspected for that reference, not merely lexically searched.
    """
    if hasattr(report, "to_dict"):
        report = report.to_dict()
    if not isinstance(report, dict) or not isinstance(reference, dict):
        raise ValueError("Report and reference must be JSON objects.")
    if reference.get("schema_version") != 1:
        raise ValueError("Unsupported reference schema_version; expected 1.")
    protocol = reference.get("protocol", {})
    if not isinstance(protocol, dict) or not all(
        protocol.get(key) for key in ("id", "version", "granularity")
    ):
        raise ValueError(
            "Reference protocol requires id, version and explicit granularity rules."
        )
    provenance = reference.get("provenance", {})
    if (
        not isinstance(provenance, dict)
        or not provenance.get("annotator_type")
        or not provenance.get("reviewer_status")
    ):
        raise ValueError("Reference annotation provenance is required.")
    matched_protocol = protocol["id"] == report.get("protocol_version")
    if not matched_protocol and not allow_protocol_mismatch:
        raise ValueError(
            "Reference and report protocols differ; explicitly allow a diagnostic cross-protocol evaluation."
        )
    ref_segments, scope = _source(reference.get("snapshot"), "reference")
    out_segments, out_scope = _source(report.get("snapshot"), "report")
    if not set(scope) <= set(out_scope):
        raise ValueError(
            "The report's current input does not include the reference scope."
        )
    if not reference["snapshot"].get("source_fingerprint") or reference["snapshot"].get(
        "source_fingerprint"
    ) != report["snapshot"].get("source_fingerprint"):
        raise ValueError(
            "Source fingerprints differ; references cannot be aligned to another source."
        )
    for identifier in scope:
        fields = ("content", "start", "end", "source_identity", "video_id")
        if any(
            ref_segments[identifier].get(field) != out_segments[identifier].get(field)
            for field in fields
        ):
            raise ValueError(
                f"Reference source segment {identifier} differs from the report."
            )
    all_ref_units = _index(reference.get("units"), "reference.units")
    units = _index(report.get("units"), "report.units")
    all_occurrences = _index(report.get("occurrences"), "report.occurrences")
    candidates = _index(report.get("candidates", []), "report.candidates")
    occurrences = {}
    literal_errors = []
    invalid_occurrences = set()
    for identifier, occurrence in all_occurrences.items():
        if (
            occurrence.get("unit_id") not in units
            or occurrence.get("segment_id") not in out_segments
        ):
            raise ValueError("Report occurrences contain unknown unit or source ids.")
        errors = _evidence_errors(
            occurrence.get("assertion_evidence"),
            out_segments,
            occurrence["segment_id"],
            identifier,
        )
        if any(
            anchor.get("segment_id", occurrence["segment_id"])
            != occurrence["segment_id"]
            for anchor in occurrence.get("assertion_evidence", [])
            if isinstance(anchor, dict)
        ):
            errors.append(
                {"owner": identifier, "error": "assertion_outside_occurrence_target"}
            )
        if occurrence.get("context_evidence"):
            errors += _evidence_errors(
                occurrence["context_evidence"],
                out_segments,
                occurrence["segment_id"],
                identifier,
            )
        literal_errors.extend(errors)
        if errors:
            invalid_occurrences.add(identifier)
        if occurrence["segment_id"] in scope:
            occurrences[identifier] = occurrence
    scoped_unit_ids = {row["unit_id"] for row in occurrences.values()}
    # A report unit without occurrences is an integrity error and is retained in
    # the scope when the reference covers the complete input.
    orphan_units = set(units) - {row["unit_id"] for row in all_occurrences.values()}
    if set(scope) == set(out_scope):
        scoped_unit_ids |= orphan_units
    scoped_units = {
        identifier: units[identifier]
        for identifier in units
        if identifier in scoped_unit_ids
    }
    all_ref_occurrences = {}
    for unit in all_ref_units.values():
        if not isinstance(unit.get("text"), str) or not unit["text"].strip():
            raise ValueError("Reference units require complete proposition text.")
        if not unit.get("occurrences"):
            raise ValueError("Each reference proposition requires a source occurrence.")
        for identifier, occurrence in _index(
            unit.get("occurrences"), "reference.occurrences"
        ).items():
            if (
                identifier in all_ref_occurrences
                or occurrence.get("segment_id") not in scope
            ):
                raise ValueError(
                    "Reference occurrences have duplicate ids or lie outside the evaluation scope."
                )
            occurrence = dict(occurrence, unit_id=unit["id"])
            errors = _evidence_errors(
                occurrence.get("assertion_evidence"),
                ref_segments,
                occurrence["segment_id"],
                identifier,
            )
            if any(
                anchor.get("segment_id", occurrence["segment_id"])
                != occurrence["segment_id"]
                for anchor in occurrence.get("assertion_evidence", [])
                if isinstance(anchor, dict)
            ):
                errors.append(
                    {
                        "owner": identifier,
                        "error": "assertion_outside_occurrence_target",
                    }
                )
            if occurrence.get("context_evidence"):
                errors += _evidence_errors(
                    occurrence["context_evidence"],
                    ref_segments,
                    occurrence["segment_id"],
                    identifier,
                )
            if errors:
                raise ValueError(f"Reference evidence is not literal: {errors[0]}.")
            all_ref_occurrences[identifier] = occurrence
    ref_units = {
        identifier: row
        for identifier, row in all_ref_units.items()
        if row.get("scorable", True) is not False
    }
    ref_occurrences = {
        identifier: row
        for identifier, row in all_ref_occurrences.items()
        if row["unit_id"] in ref_units and row.get("scorable", True) is not False
    }
    excluded_units = [
        {
            "reference_unit_id": identifier,
            "text": row["text"],
            "state": "excluded",
            "exclusion_reason": row.get(
                "exclusion_reason", "Reference marks this proposition non-scorable."
            ),
        }
        for identifier, row in all_ref_units.items()
        if identifier not in ref_units
    ]
    excluded_occurrences = [
        {
            "reference_occurrence_id": identifier,
            "reference_unit_id": row["unit_id"],
            "segment_id": row["segment_id"],
            "state": "excluded",
            "exclusion_reason": row.get(
                "exclusion_reason", "The unit or occurrence is marked non-scorable."
            ),
        }
        for identifier, row in all_ref_occurrences.items()
        if identifier not in ref_occurrences
    ]
    for unit in units.values():
        if not isinstance(unit.get("text"), str):
            raise ValueError("Report units require proposition text.")
    alignment = alignment or {}
    if not isinstance(alignment, dict):
        raise ValueError("Alignment must be a JSON object.")
    alignment_provenance = alignment.get("provenance", {})
    if alignment:
        if alignment.get("schema_version") != 1:
            raise ValueError("Unsupported alignment schema_version; expected 1.")
        if alignment.get("report_fingerprint") != evaluation_fingerprint(
            report
        ) or alignment.get("reference_fingerprint") != evaluation_fingerprint(
            reference
        ):
            raise ValueError(
                "Alignment fingerprints do not bind these frozen report and reference documents."
            )
    links, unverified_links = [], []
    for index, row in enumerate(alignment.get("unit_links", [])):
        _validate_link(row, all_ref_units, scoped_units, UNIT_RELATIONS)
        row = dict(
            row, id=row.get("id", f"reviewed-unit-{index}"), method="reviewed_alignment"
        )
        if not set(row["reference_unit_ids"]) <= set(ref_units):
            unverified_links.append(
                dict(
                    row,
                    exclusion_reason="Contains a non-scorable reference; no match or count error is scored.",
                )
            )
            continue
        if _reviewed(row, alignment_provenance) and row["relation"] != "unresolved":
            links.append(row)
        else:
            unverified_links.append(row)
    ref_by_text = defaultdict(list)
    for identifier, unit in ref_units.items():
        ref_by_text[_text(unit["text"])].append(identifier)
    explicit_pairs = {
        (ref, out)
        for row in links
        if row["relation"] != "compound_plus_components"
        for ref in row["reference_unit_ids"]
        for out in row["report_unit_ids"]
    }
    for out_id, unit in scoped_units.items():
        matches = ref_by_text.get(_text(unit["text"]), [])
        if len(matches) != 1 or (matches[0], out_id) in explicit_pairs:
            continue
        ref_id = matches[0]
        anchored = any(
            _same_occurrence(ref_occurrence, out_occurrence)
            for ref_occurrence in ref_occurrences.values()
            if ref_occurrence["unit_id"] == ref_id
            for out_occurrence in occurrences.values()
            if out_occurrence["unit_id"] == out_id
            and out_occurrence["id"] not in invalid_occurrences
        )
        if anchored:
            links.append(
                {
                    "id": f"exact-unit-{ref_id}-{out_id}",
                    "reference_unit_ids": [ref_id],
                    "report_unit_ids": [out_id],
                    "relation": "equivalent",
                    "review_status": "verified",
                    "method": "identical_complete_text_and_source_anchors",
                    "qualifier_errors": [],
                }
            )
    equivalent_references = defaultdict(set)
    for link in links:
        if link["relation"] == "equivalent":
            equivalent_references[link["report_unit_ids"][0]].update(
                link["reference_unit_ids"]
            )
    if any(len(values) > 1 for values in equivalent_references.values()):
        raise ValueError(
            "One report unit cannot be equivalent to distinct reference units; use an explicit merged alignment."
        )
    occurrence_links, unverified_occurrence_links = [], []
    for index, row in enumerate(alignment.get("occurrence_links", [])):
        _validate_link(
            row, all_ref_occurrences, occurrences, OCCURRENCE_RELATIONS, True
        )
        row = dict(
            row,
            id=row.get("id", f"reviewed-occurrence-{index}"),
            method="reviewed_alignment",
        )
        if not set(row["reference_occurrence_ids"]) <= set(ref_occurrences):
            unverified_occurrence_links.append(
                dict(
                    row,
                    exclusion_reason="Contains a non-scorable reference occurrence.",
                )
            )
            continue
        if _reviewed(row, alignment_provenance) and row["relation"] != "unresolved":
            if set(row["report_occurrence_ids"]) & invalid_occurrences:
                raise ValueError(
                    "A reviewed occurrence match cannot use nonliteral source evidence."
                )
            for ref_id in row["reference_occurrence_ids"]:
                for out_id in row["report_occurrence_ids"]:
                    if not _overlapping_occurrence(
                        ref_occurrences[ref_id], occurrences[out_id]
                    ):
                        raise ValueError(
                            "Reviewed occurrence matches must identify overlapping assertion anchors in the same utterance."
                        )
                    pair = (
                        ref_occurrences[ref_id]["unit_id"],
                        occurrences[out_id]["unit_id"],
                    )
                    if not any(
                        pair[0] in unit_link["reference_unit_ids"]
                        and pair[1] in unit_link["report_unit_ids"]
                        and unit_link["relation"] != "partial"
                        for unit_link in links
                    ):
                        raise ValueError(
                            "Occurrence links require a verified complete unit alignment."
                        )
            occurrence_links.append(row)
        else:
            unverified_occurrence_links.append(row)
    explicit_occurrence_pairs = {
        (ref, out)
        for row in occurrence_links
        for ref in row["reference_occurrence_ids"]
        for out in row["report_occurrence_ids"]
    }
    for link in links:
        if link["relation"] != "equivalent":
            continue
        for ref_occurrence in ref_occurrences.values():
            if ref_occurrence["unit_id"] != link["reference_unit_ids"][0]:
                continue
            for out_occurrence in occurrences.values():
                pair = ref_occurrence["id"], out_occurrence["id"]
                if (
                    out_occurrence["unit_id"] == link["report_unit_ids"][0]
                    and out_occurrence["id"] not in invalid_occurrences
                    and pair not in explicit_occurrence_pairs
                    and _same_occurrence(ref_occurrence, out_occurrence)
                ):
                    occurrence_links.append(
                        {
                            "id": f"exact-occurrence-{pair[0]}-{pair[1]}",
                            "reference_occurrence_ids": [pair[0]],
                            "report_occurrence_ids": [pair[1]],
                            "relation": "equivalent",
                            "review_status": "verified",
                            "method": "identical_source_anchors_under_verified_unit_alignment",
                        }
                    )
    ref_decisions, unverified_ref_decisions = _decisions(
        alignment.get("reference_decisions", []),
        "reference_unit_id",
        all_ref_units,
        {"omitted", "partial", "unresolved"},
        alignment_provenance,
    )
    ref_decisions = {
        identifier: row
        for identifier, row in ref_decisions.items()
        if identifier in ref_units
    }
    out_decisions, unverified_out_decisions = _decisions(
        alignment.get("report_decisions", []),
        "report_unit_id",
        scoped_units,
        {"unsupported", "partial", "duplicate", "supported_unreferenced", "unresolved"},
        alignment_provenance,
    )
    ref_occ_decisions, unverified_ref_occ_decisions = _decisions(
        alignment.get("reference_occurrence_decisions", []),
        "reference_occurrence_id",
        all_ref_occurrences,
        {"omitted", "partial", "unresolved"},
        alignment_provenance,
    )
    ref_occ_decisions = {
        identifier: row
        for identifier, row in ref_occ_decisions.items()
        if identifier in ref_occurrences
    }
    out_occ_decisions, unverified_out_occ_decisions = _decisions(
        alignment.get("report_occurrence_decisions", []),
        "report_occurrence_id",
        occurrences,
        {"unsupported", "duplicate", "unresolved"},
        alignment_provenance,
    )
    ref_to_links, out_to_links = defaultdict(list), defaultdict(list)
    granularity_errors = []
    qualifier_errors = []
    compound_parents = set()
    for link in links:
        for identifier in link["reference_unit_ids"]:
            ref_to_links[identifier].append(link)
        for identifier in link["report_unit_ids"]:
            out_to_links[identifier].append(link)
        if link["relation"] in {"split", "merged", "compound_plus_components"}:
            excess = len(link["report_unit_ids"]) - len(link["reference_unit_ids"])
            if link["relation"] == "compound_plus_components":
                compound_parents.update(link["parent_report_unit_ids"])
                excess = len(link["parent_report_unit_ids"])
            granularity_errors.append(
                {
                    "alignment_id": link["id"],
                    "kind": link["relation"],
                    "reference_unit_ids": link["reference_unit_ids"],
                    "report_unit_ids": link["report_unit_ids"],
                    "count_excess": max(0, excess),
                    "count_deficit": max(0, -excess),
                }
            )
        for error in link.get("qualifier_errors", []):
            qualifier_errors.append(
                dict(
                    error,
                    alignment_id=link["id"],
                    reference_unit_ids=link["reference_unit_ids"],
                    report_unit_ids=link["report_unit_ids"],
                )
            )
    complete = {"equivalent", "split", "merged", "compound_plus_components"}
    ref_results = []
    for identifier, unit in ref_units.items():
        aligned = ref_to_links[identifier]
        covered = any(link["relation"] in complete for link in aligned)
        decision = ref_decisions.get(identifier, {})
        if covered and decision.get("state") in {"omitted", "partial"}:
            raise ValueError(
                "A reference cannot be fully covered and explicitly omitted or partial."
            )
        state = "covered" if covered else decision.get("state", "unresolved")
        ref_results.append(
            {
                "reference_unit_id": identifier,
                "text": unit["text"],
                "state": state,
                "report_unit_ids": sorted(
                    {out for link in aligned for out in link["report_unit_ids"]}
                ),
                "alignment_ids": [link["id"] for link in aligned],
                "has_partial_alignment": any(
                    link["relation"] == "partial" for link in aligned
                ),
                "review_note": decision.get("reason"),
                "qualifier_errors": [
                    error
                    for error in qualifier_errors
                    if identifier in error["reference_unit_ids"]
                ],
                "granularity_errors": [
                    error
                    for error in granularity_errors
                    if identifier in error["reference_unit_ids"]
                ],
            }
        )
    out_results = []
    for identifier, unit in scoped_units.items():
        aligned = out_to_links[identifier]
        supported = any(link["relation"] in complete for link in aligned)
        decision = out_decisions.get(identifier, {})
        if supported and decision.get("state") in {"unsupported", "partial"}:
            raise ValueError(
                "A fully supported alignment conflicts with an unsupported or partial report decision."
            )
        state = "supported" if supported else decision.get("state", "unresolved")
        if state == "duplicate":
            # Output equivalence alone cannot prove that its content is supported.
            state = "unresolved"
        out_results.append(
            {
                "report_unit_id": identifier,
                "text": unit["text"],
                "state": state,
                "reference_unit_ids": sorted(
                    {ref for link in aligned for ref in link["reference_unit_ids"]}
                ),
                "alignment_ids": [link["id"] for link in aligned],
                "review_note": decision.get("reason"),
                "compound_parent_extra": identifier in compound_parents,
            }
        )
    ref_occ_to_links, out_occ_to_links = defaultdict(list), defaultdict(list)
    for link in occurrence_links:
        for identifier in link["reference_occurrence_ids"]:
            ref_occ_to_links[identifier].append(link)
        for identifier in link["report_occurrence_ids"]:
            out_occ_to_links[identifier].append(link)
    ref_occ_results = []
    for identifier, occurrence in ref_occurrences.items():
        aligned = ref_occ_to_links[identifier]
        decision = ref_occ_decisions.get(identifier, {})
        if aligned and decision.get("state") in {"omitted", "partial"}:
            raise ValueError(
                "An occurrence cannot be covered and explicitly omitted or partial."
            )
        ref_occ_results.append(
            {
                "reference_occurrence_id": identifier,
                "reference_unit_id": occurrence["unit_id"],
                "segment_id": occurrence["segment_id"],
                "state": "covered" if aligned else decision.get("state", "unresolved"),
                "report_occurrence_ids": sorted(
                    {out for link in aligned for out in link["report_occurrence_ids"]}
                ),
                "alignment_ids": [link["id"] for link in aligned],
                "review_note": decision.get("reason"),
            }
        )
    out_occ_results = []
    for identifier, occurrence in occurrences.items():
        aligned = out_occ_to_links[identifier]
        decision = out_occ_decisions.get(identifier, {})
        if aligned and decision.get("state") == "unsupported":
            raise ValueError(
                "A matched occurrence conflicts with an unsupported occurrence decision."
            )
        state = "supported" if aligned else decision.get("state", "unresolved")
        if state == "duplicate":
            state = "unresolved"
        if identifier in invalid_occurrences:
            state = "invalid_evidence"
        out_occ_results.append(
            {
                "report_occurrence_id": identifier,
                "report_unit_id": occurrence["unit_id"],
                "segment_id": occurrence["segment_id"],
                "state": state,
                "reference_occurrence_ids": sorted(
                    {
                        ref
                        for link in aligned
                        for ref in link["reference_occurrence_ids"]
                    }
                ),
                "alignment_ids": [link["id"] for link in aligned],
                "review_note": decision.get("reason"),
            }
        )
    duplicate_groups = []
    duplicate_ids = set()
    for ref_id, aligned in ref_to_links.items():
        equivalent = sorted(
            {
                out
                for link in aligned
                if link["relation"] == "equivalent"
                for out in link["report_unit_ids"]
            }
        )
        if len(equivalent) > 1:
            duplicate_groups.append(
                {
                    "reference_unit_id": ref_id,
                    "report_unit_ids": equivalent,
                    "extra_count": len(equivalent) - 1,
                }
            )
            duplicate_ids.update(equivalent[1:])
    for identifier, decision in out_decisions.items():
        if decision["state"] == "duplicate":
            other = decision.get("duplicate_of")
            if other not in scoped_units or other == identifier:
                raise ValueError(
                    "A duplicate decision must identify another counted report unit."
                )
            duplicate_ids.add(identifier)
    duplicate_ids |= compound_parents
    duplicate_occurrence_groups = []
    duplicate_occurrence_ids = set()
    for ref_id, aligned in ref_occ_to_links.items():
        equivalent = sorted(
            {
                out
                for link in aligned
                if link["relation"] == "equivalent"
                for out in link["report_occurrence_ids"]
            }
        )
        if len(equivalent) > 1:
            duplicate_occurrence_groups.append(
                {
                    "reference_occurrence_id": ref_id,
                    "report_occurrence_ids": equivalent,
                    "extra_count": len(equivalent) - 1,
                }
            )
            duplicate_occurrence_ids.update(equivalent[1:])
    for link in occurrence_links:
        if link["relation"] == "split_anchor":
            duplicate_occurrence_ids.update(sorted(link["report_occurrence_ids"])[1:])
    compound_occurrence_parents = set()
    for link in links:
        if link["relation"] != "compound_plus_components":
            continue
        for identifier, occurrence in occurrences.items():
            if occurrence["unit_id"] not in link["parent_report_unit_ids"]:
                continue
            represented = {
                ref
                for item in out_occ_to_links[identifier]
                for ref in item["reference_occurrence_ids"]
            }
            if {ref_occurrences[ref]["unit_id"] for ref in represented} == set(
                link["reference_unit_ids"]
            ) and all(
                any(
                    occurrences[out]["unit_id"] in link["component_report_unit_ids"]
                    for item in ref_occ_to_links[ref]
                    for out in item["report_occurrence_ids"]
                )
                for ref in represented
            ):
                compound_occurrence_parents.add(identifier)
    duplicate_occurrence_ids |= compound_occurrence_parents
    for identifier, decision in out_occ_decisions.items():
        if decision["state"] == "duplicate":
            other = decision.get("duplicate_of")
            if (
                other not in occurrences
                or other == identifier
                or not _overlapping_occurrence(
                    occurrences[identifier], occurrences[other]
                )
            ):
                raise ValueError(
                    "An occurrence duplicate must identify overlapping anchors in the same utterance."
                )
            duplicate_occurrence_ids.add(identifier)
    out_states = Counter(row["state"] for row in out_results)
    faithful = (
        out_states["supported"]
        + out_states["duplicate"]
        + out_states["supported_unreferenced"]
    )
    unresolved = out_states["unresolved"]
    occ_states = Counter(row["state"] for row in out_occ_results)
    faithful_occurrences = occ_states["supported"] + occ_states["duplicate"]
    integrity = _integrity(report, units, all_occurrences, candidates)
    integrity.extend(
        {"error": "unit_without_occurrence", "report_unit_id": identifier}
        for identifier in sorted(orphan_units)
    )
    uncertainty = report.get("metadata", {})
    findings = {
        "unit_coverage": _coverage([row["state"] for row in ref_results]),
        "occurrence_coverage": _coverage([row["state"] for row in ref_occ_results]),
        "source_fidelity": {
            "report_total": len(scoped_units),
            "verified_supported": faithful,
            "confirmed_false_positive": out_states["unsupported"],
            "confirmed_partial": out_states["partial"],
            "unresolved": unresolved,
            "fidelity_lower_bound": _fraction(faithful, len(scoped_units)),
            "fidelity_upper_bound": _fraction(faithful + unresolved, len(scoped_units)),
            "precision_on_resolved_units": _fraction(
                faithful, len(scoped_units) - unresolved
            ),
            "interpretation": "Source fidelity concerns complete communicated content. Supported split/merged/duplicate units can still violate canonical granularity; those count errors are reported separately.",
        },
        "occurrence_fidelity": {
            "report_total": len(occurrences),
            "verified_supported": faithful_occurrences,
            "confirmed_false_positive": occ_states["unsupported"],
            "invalid_evidence": occ_states["invalid_evidence"],
            "unresolved": occ_states["unresolved"],
            "fidelity_lower_bound": _fraction(faithful_occurrences, len(occurrences)),
            "fidelity_upper_bound": _fraction(
                faithful_occurrences + occ_states["unresolved"], len(occurrences)
            ),
        },
        "consolidation": {
            "verified_duplicate_extra_units": len(duplicate_ids),
            "duplicate_report_unit_ids": sorted(duplicate_ids),
            "equivalent_duplicate_groups": duplicate_groups,
            "compound_plus_components_extra_units": len(compound_parents),
            "unnecessary_split_groups": sum(
                row["kind"] == "split" for row in granularity_errors
            ),
            "improper_merge_groups": sum(
                row["kind"] == "merged" for row in granularity_errors
            ),
            "granularity_errors": granularity_errors,
            "duplicate_occurrence_extra_count": len(duplicate_occurrence_ids),
            "duplicate_report_occurrence_ids": sorted(duplicate_occurrence_ids),
            "compound_occurrence_parent_extra_count": len(compound_occurrence_parents),
            "split_occurrence_extra_count": sum(
                len(row["report_occurrence_ids"]) - 1
                for row in occurrence_links
                if row["relation"] == "split_anchor"
            ),
            "merged_occurrence_deficit_count": sum(
                len(row["reference_occurrence_ids"]) - 1
                for row in occurrence_links
                if row["relation"] == "merged_occurrences"
            ),
            "duplicate_occurrence_groups": duplicate_occurrence_groups,
            "occurrence_alignment_errors": [
                link for link in occurrence_links if link["relation"] != "equivalent"
            ],
        },
        "qualifier_errors": {
            "total": len(qualifier_errors),
            "by_field": dict(Counter(error["field"] for error in qualifier_errors)),
            "by_kind": dict(Counter(error["kind"] for error in qualifier_errors)),
            "details": qualifier_errors,
        },
        "traceability": {
            "reference_occurrences": len(ref_occurrences),
            "report_occurrences_in_scope": len(occurrences),
            "report_occurrences_with_literal_errors": len(
                invalid_occurrences & set(occurrences)
            ),
            "literal_errors": literal_errors,
            "integrity_errors": integrity,
        },
        "alignment": {
            "verified_unit_links": len(links),
            "automatic_unit_links": sum(
                row["method"] != "reviewed_alignment" for row in links
            ),
            "reviewed_unit_links": sum(
                row["method"] == "reviewed_alignment" for row in links
            ),
            "verified_occurrence_links": len(occurrence_links),
            "unverified_unit_links": unverified_links,
            "unverified_occurrence_links": unverified_occurrence_links,
            "unverified_decisions": unverified_ref_decisions
            + unverified_out_decisions
            + unverified_ref_occ_decisions
            + unverified_out_occ_decisions,
            "unit_links": links,
            "occurrence_links": occurrence_links,
        },
        "reference_exclusions": {
            "annotation_unit_records": len(all_ref_units),
            "scorable_units": len(ref_units),
            "annotation_occurrence_records": len(all_ref_occurrences),
            "scorable_occurrences": len(ref_occurrences),
            "units": excluded_units,
            "occurrences": excluded_occurrences,
            "gaps": reference.get("gaps", []),
        },
        "report_uncertainty": {
            "status": report.get("status"),
            "recorded_counts": report.get("counts"),
            "uncertain_equivalence_relations": uncertainty.get(
                "equivalence_uncertain_relations"
            ),
            "unexamined_pair_count": uncertainty.get("unexamined_pair_count"),
            "coverage_focus_states": dict(
                Counter(
                    row.get("state", "unknown")
                    for row in report.get("coverage_foci", [])
                )
            ),
            "decomposition_states": dict(
                Counter(
                    row.get("state", "unknown")
                    for row in report.get("decompositions", [])
                )
            ),
        },
        "criteria_validation": _criteria(
            report, alignment, alignment_provenance, candidates
        ),
    }
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_kind": "offline_reference_inventory",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "report_fingerprint": evaluation_fingerprint(report),
            "reference_fingerprint": evaluation_fingerprint(reference),
            "alignment_fingerprint": (
                evaluation_fingerprint(alignment) if alignment else None
            ),
            "report_run_id": report.get("run_id"),
            "source_fingerprint": report["snapshot"].get("source_fingerprint"),
            "reference_annotation": provenance,
            "alignment_annotation": alignment_provenance,
            "reference_protocol": protocol,
            "report_protocol": report.get("protocol_version"),
            "matched_protocol": matched_protocol,
            "scope_segment_ids": scope,
            "diagnostic_only": not matched_protocol
            or provenance.get("annotator_type") != "human"
            or provenance.get("reviewer_status") != "reviewed",
            "interpretation": "Results are relative to the frozen annotated reference and reviewed alignment. Agent annotations and unverified model output are not human gold or measured scientific accuracy.",
        },
        "experiment": _experiment(report),
        "operational_cost": _cost(report),
        "metrics": findings,
        "reference_unit_results": ref_results + excluded_units,
        "report_unit_results": out_results,
        "reference_occurrence_results": ref_occ_results + excluded_occurrences,
        "report_occurrence_results": out_occ_results,
    }


def evaluate_information_files(
    report_path, reference_path, alignment_path=None, *, allow_protocol_mismatch=False
):
    """Load repository-relative files with their byte hashes retained in output."""
    loaded, files = [], {}
    for label, path in (
        ("report", report_path),
        ("reference", reference_path),
        ("alignment", alignment_path),
    ):
        if path is None:
            loaded.append(None)
            continue
        resolved = project_path(path)
        raw = resolved.read_bytes()
        loaded.append(json.loads(raw.decode("utf-8-sig")))
        files[label] = {
            "path": portable_path(resolved),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    result = evaluate_information_report(
        *loaded, allow_protocol_mismatch=allow_protocol_mismatch
    )
    result["provenance"]["files"] = files
    return result


def save_information_evaluation(result, path):
    """Publish a complete new JSON artifact under ignored outputs/ atomically."""
    destination = project_path(path).resolve()
    if (
        not destination.is_relative_to(project_path("outputs").resolve())
        or destination.suffix.lower() != ".json"
    ):
        raise ValueError("Evaluation outputs must be JSON files under outputs/.")
    if destination.exists():
        raise ValueError("The evaluation output already exists; choose a new path.")
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=".evaluation-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def compare_information_evaluations(evaluations):
    """Tabulate ablations only when protocol, source, scope and reference agree."""
    if not evaluations:
        raise ValueError("At least one evaluation is required.")
    anchor = evaluations[0]["provenance"]
    for value in evaluations:
        provenance = value["provenance"]
        if not provenance.get("matched_protocol"):
            raise ValueError(
                "Cross-protocol diagnostics cannot enter a controlled ablation comparison."
            )
        for field in (
            "reference_fingerprint",
            "source_fingerprint",
            "scope_segment_ids",
            "report_protocol",
        ):
            if provenance.get(field) != anchor.get(field):
                raise ValueError(f"Ablation evaluations differ in {field}.")
    rows = []
    for value in evaluations:
        metrics = value["metrics"]
        rows.append(
            {
                "run_id": value["provenance"].get("report_run_id"),
                "experiment": value["experiment"],
                "unit_coverage": metrics["unit_coverage"],
                "occurrence_coverage": metrics["occurrence_coverage"],
                "source_fidelity": metrics["source_fidelity"],
                "consolidation": metrics["consolidation"],
                "qualifier_error_count": metrics["qualifier_errors"]["total"],
                "operational_cost": value["operational_cost"],
                "report_status": metrics["report_uncertainty"]["status"],
                "reviewed_unit_links": metrics["alignment"]["reviewed_unit_links"],
            }
        )
    return {
        "schema_version": 1,
        "comparison_kind": "same_reference_ablation_table",
        "reference_fingerprint": anchor["reference_fingerprint"],
        "rows": rows,
        "interpretation": "Compare coverage bounds, review completeness, fidelity, count errors and cost together. Unequal review coverage or a partial run can confound a mode/provider comparison; no ranking or threshold tuning is performed.",
    }
