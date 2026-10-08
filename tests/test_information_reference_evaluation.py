"""Offline regressions for reviewed reference alignment and count diagnostics."""

import copy
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from open_video_summary.evaluation import (
    compare_information_evaluations,
    evaluate_information_files,
    evaluate_information_report,
    evaluation_fingerprint,
    save_information_evaluation,
)
from open_video_summary.evaluation.__main__ import main
from open_video_summary.utils.paths import project_path


def fixtures():
    first = "No Alfa, os backups automáticos são feitos a cada 24 horas e guardados por 30 dias."
    conditional = "Se a conexão cair, o backup automático não é realizado."
    repeated = "No Alfa, os backups automáticos são feitos a cada 24 horas."
    texts = [first, conditional, repeated + " " + repeated]
    segments = [
        {
            "id": f"v0:s{i}",
            "video_id": "v0",
            "source_identity": f"source:v0:s{i}",
            "content": text,
            "start": i * 10,
            "end": i * 10 + 9,
        }
        for i, text in enumerate(texts)
    ]
    snapshot = {
        "source_fingerprint": "source",
        "current_order": [row["id"] for row in segments],
        "source": [{"id": "v0", "name": "Alfa", "segments": segments}],
    }

    def anchor(index, quote, offset=0):
        start = texts[index].index(quote, offset)
        return {
            "segment_id": f"v0:s{index}",
            "start_char": start,
            "end_char": start + len(quote),
            "quote": quote,
        }

    assertions = [
        anchor(0, first[: first.index(" e guardados")]),
        anchor(0, "guardados por 30 dias"),
        anchor(1, conditional),
        anchor(2, repeated),
        anchor(2, repeated, len(repeated)),
    ]
    reference = {
        "schema_version": 1,
        "protocol": {
            "id": "contextual-propositions-v2",
            "version": "2",
            "granularity": "One contextual proposition; condition and consequence remain together; independently meaningful properties are separate.",
        },
        "provenance": {
            "annotator_type": "synthetic",
            "reviewer_status": "reviewed",
            "reviewer": "test fixture",
        },
        "snapshot": snapshot,
        "units": [
            {"id": "r0", "text": repeated, "occurrences": []},
            {
                "id": "r1",
                "text": "No Alfa, os backups automáticos são guardados por 30 dias.",
                "occurrences": [],
            },
            {"id": "r2", "text": conditional, "occurrences": []},
        ],
    }
    order = [0, 1, 2, 0, 0]
    occurrences = []
    for index, (unit_index, evidence) in enumerate(zip(order, assertions)):
        occurrence = {
            "id": f"ro{index}",
            "segment_id": evidence["segment_id"],
            "assertion_evidence": [evidence],
            "context_evidence": [],
        }
        reference["units"][unit_index]["occurrences"].append(occurrence)
        occurrences.append(
            dict(copy.deepcopy(occurrence), id=f"o{index}", unit_id=f"u{unit_index}")
        )
    report = {
        "schema_version": 7,
        "protocol_version": "contextual-propositions-v2",
        "run_id": "fixture",
        "status": "completed",
        "snapshot": copy.deepcopy(snapshot),
        "units": [
            {"id": f"u{i}", "text": row["text"], "candidate_ids": [f"c{i}"]}
            for i, row in enumerate(reference["units"])
        ],
        "occurrences": occurrences,
        "candidates": [
            {
                "id": f"c{i}",
                "validation": "accepted",
                "granularity": "atomic",
                "signals": [["support", 0.95], ["conditions", 0.95]],
            }
            for i in range(3)
        ],
        "counts": {
            "candidates": 3,
            "accepted_candidates": 3,
            "unique_units": 3,
            "occurrences": 5,
        },
        "calls": [],
        "metadata": {
            "experimental_mode": "hybrid_coverage",
            "evaluator_provider": "fixture",
            "settings": {"acceptance_threshold": 0.85},
            "thresholds_calibrated": False,
        },
    }
    return report, reference


def reviewed(report, reference, **values):
    return {
        "schema_version": 1,
        "report_fingerprint": evaluation_fingerprint(report),
        "reference_fingerprint": evaluation_fingerprint(reference),
        "provenance": {
            "annotator_type": "agent",
            "reviewer_status": "reviewed",
            "reviewer": "independent fixture review",
        },
        **values,
    }


def link(refs, outputs, relation="equivalent", **values):
    return {
        "reference_unit_ids": refs,
        "report_unit_ids": outputs,
        "relation": relation,
        "review_status": "verified",
        **values,
    }


class InformationReferenceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.network = patch(
            "socket.socket",
            side_effect=AssertionError("Network is disabled for offline evaluation."),
        )
        self.network.start()
        self.addCleanup(self.network.stop)
        self.report, self.reference = fixtures()

    def test_exact_complete_text_and_anchors_cover_units_and_separate_utterances(self):
        result = evaluate_information_report(self.report, self.reference)
        self.assertEqual(3, result["metrics"]["unit_coverage"]["covered"])
        self.assertEqual(5, result["metrics"]["occurrence_coverage"]["covered"])
        self.assertEqual(
            0, result["metrics"]["consolidation"]["duplicate_occurrence_extra_count"]
        )
        self.assertEqual(
            1, result["metrics"]["source_fidelity"]["fidelity_lower_bound"]
        )
        self.assertTrue(result["provenance"]["diagnostic_only"])

    def test_lexical_paraphrase_miss_is_unresolved_not_a_false_negative(self):
        self.report["units"][0][
            "text"
        ] = "O intervalo dos backups automáticos do Alfa é de 24 horas."
        result = evaluate_information_report(self.report, self.reference)
        coverage = result["metrics"]["unit_coverage"]
        self.assertEqual(
            (2, 0, 1),
            (
                coverage["covered"],
                coverage["confirmed_omitted"],
                coverage["unresolved"],
            ),
        )
        self.assertEqual(1, coverage["coverage_upper_bound"])
        self.assertEqual(3, result["metrics"]["occurrence_coverage"]["unresolved"])
        aligned = reviewed(
            self.report, self.reference, unit_links=[link(["r0"], ["u0"])]
        )
        result = evaluate_information_report(self.report, self.reference, aligned)
        self.assertEqual(3, result["metrics"]["unit_coverage"]["covered"])
        self.assertEqual(5, result["metrics"]["occurrence_coverage"]["covered"])

    def test_broad_quote_does_not_cover_retention_by_itself_or_resolve_anchor_overlap(
        self,
    ):
        self.report["units"][1][
            "text"
        ] = "Os backups automáticos do Alfa ficam retidos por 30 dias."
        self.report["occurrences"][1]["assertion_evidence"] = [
            {
                "segment_id": "v0:s0",
                "start_char": 0,
                "end_char": len(
                    self.report["snapshot"]["source"][0]["segments"][0]["content"]
                ),
                "quote": self.report["snapshot"]["source"][0]["segments"][0]["content"],
            }
        ]
        aligned = reviewed(
            self.report, self.reference, unit_links=[link(["r1"], ["u1"])]
        )
        result = evaluate_information_report(self.report, self.reference, aligned)
        self.assertEqual(3, result["metrics"]["unit_coverage"]["covered"])
        self.assertEqual(1, result["metrics"]["occurrence_coverage"]["unresolved"])
        aligned["occurrence_links"] = [
            {
                "reference_occurrence_ids": ["ro1"],
                "report_occurrence_ids": ["o1"],
                "relation": "equivalent",
                "review_status": "verified",
            }
        ]
        self.assertEqual(
            5,
            evaluate_information_report(self.report, self.reference, aligned)[
                "metrics"
            ]["occurrence_coverage"]["covered"],
        )

    def test_same_segment_different_repeated_assertion_is_not_the_same_occurrence(self):
        self.report["occurrences"].pop()
        self.report["counts"]["occurrences"] = 4
        result = evaluate_information_report(self.report, self.reference)
        self.assertEqual(
            (4, 1),
            (
                result["metrics"]["occurrence_coverage"]["covered"],
                result["metrics"]["occurrence_coverage"]["unresolved"],
            ),
        )
        aligned = reviewed(
            self.report,
            self.reference,
            occurrence_links=[
                {
                    "reference_occurrence_ids": ["ro4"],
                    "report_occurrence_ids": ["o3"],
                    "relation": "equivalent",
                    "review_status": "verified",
                }
            ],
        )
        with self.assertRaisesRegex(ValueError, "overlapping"):
            evaluate_information_report(self.report, self.reference, aligned)

    def test_reviewed_omission_and_unsupported_unit_are_not_inferred_from_no_match(
        self,
    ):
        self.report["units"][2]["text"] = "Os backups funcionam mesmo sem conexão."
        aligned = reviewed(
            self.report,
            self.reference,
            reference_decisions=[
                {
                    "reference_unit_id": "r2",
                    "state": "omitted",
                    "review_status": "verified",
                    "reason": "Exhaustive report review: the governing condition and negation are absent.",
                }
            ],
            report_decisions=[
                {
                    "report_unit_id": "u2",
                    "state": "unsupported",
                    "review_status": "verified",
                    "reason": "This reverses the source's stated condition.",
                }
            ],
            reference_occurrence_decisions=[
                {
                    "reference_occurrence_id": "ro2",
                    "state": "omitted",
                    "review_status": "verified",
                }
            ],
            report_occurrence_decisions=[
                {
                    "report_occurrence_id": "o2",
                    "state": "unsupported",
                    "review_status": "verified",
                }
            ],
        )
        result = evaluate_information_report(self.report, self.reference, aligned)
        self.assertEqual(1, result["metrics"]["unit_coverage"]["confirmed_omitted"])
        self.assertEqual(
            1, result["metrics"]["source_fidelity"]["confirmed_false_positive"]
        )
        self.assertEqual(
            1, result["metrics"]["occurrence_fidelity"]["confirmed_false_positive"]
        )
        self.assertAlmostEqual(
            2 / 3, result["metrics"]["unit_coverage"]["coverage_upper_bound"]
        )

    def test_model_generated_unreviewed_links_are_never_gold(self):
        self.report["units"][0]["text"] = "Periodicidade: 24 horas."
        aligned = reviewed(
            self.report, self.reference, unit_links=[link(["r0"], ["u0"])]
        )
        aligned["provenance"].update(model_generated=True, independently_reviewed=False)
        result = evaluate_information_report(self.report, self.reference, aligned)
        self.assertEqual(1, result["metrics"]["unit_coverage"]["unresolved"])
        self.assertEqual(
            1, len(result["metrics"]["alignment"]["unverified_unit_links"])
        )

    def test_verified_duplicate_units_and_duplicate_routes_do_not_add_reference_information(
        self,
    ):
        duplicate = dict(copy.deepcopy(self.report["units"][0]), id="u3")
        self.report["units"].append(duplicate)
        self.report["occurrences"].append(
            dict(copy.deepcopy(self.report["occurrences"][0]), id="o5", unit_id="u3")
        )
        self.report["counts"].update(unique_units=4, occurrences=6)
        result = evaluate_information_report(self.report, self.reference)
        self.assertEqual(3, result["metrics"]["unit_coverage"]["covered"])
        self.assertEqual(
            1, result["metrics"]["consolidation"]["verified_duplicate_extra_units"]
        )
        self.assertEqual(
            1, result["metrics"]["consolidation"]["duplicate_occurrence_extra_count"]
        )

    def test_compound_parent_plus_components_requires_verified_decomposition(self):
        self.report["units"].append(
            {
                "id": "up",
                "text": "No Alfa, os backups são feitos a cada 24 horas e guardados por 30 dias.",
            }
        )
        self.report["occurrences"].append(
            dict(copy.deepcopy(self.report["occurrences"][0]), id="op", unit_id="up")
        )
        self.report["counts"].update(unique_units=4, occurrences=6)
        before = evaluate_information_report(self.report, self.reference)
        self.assertEqual(
            0,
            before["metrics"]["consolidation"]["compound_plus_components_extra_units"],
        )
        grouping = link(
            ["r0", "r1"],
            ["u0", "u1", "up"],
            "compound_plus_components",
            parent_report_unit_ids=["up"],
            component_report_unit_ids=["u0", "u1"],
            decomposition_verified=True,
        )
        aligned = reviewed(self.report, self.reference, unit_links=[grouping])
        result = evaluate_information_report(self.report, self.reference, aligned)
        self.assertEqual(
            1,
            result["metrics"]["consolidation"]["compound_plus_components_extra_units"],
        )
        self.assertEqual(
            1, result["metrics"]["consolidation"]["verified_duplicate_extra_units"]
        )
        grouping["decomposition_verified"] = False
        with self.assertRaisesRegex(ValueError, "verified decomposition"):
            evaluate_information_report(self.report, self.reference, aligned)

    def test_known_split_and_merge_are_count_errors_not_unmatched_false_positives(self):
        self.report["units"].append({"id": "ux", "text": "São 24 horas entre backups."})
        self.report["occurrences"].append(
            dict(copy.deepcopy(self.report["occurrences"][0]), id="ox", unit_id="ux")
        )
        self.report["units"][0]["text"] = "Backups automáticos do Alfa."
        self.report["counts"].update(unique_units=4, occurrences=6)
        aligned = reviewed(
            self.report,
            self.reference,
            unit_links=[link(["r0"], ["u0", "ux"], "split")],
        )
        result = evaluate_information_report(self.report, self.reference, aligned)
        self.assertEqual(
            1, result["metrics"]["consolidation"]["unnecessary_split_groups"]
        )
        self.assertEqual(
            0, result["metrics"]["source_fidelity"]["confirmed_false_positive"]
        )
        self.assertEqual(
            1,
            result["metrics"]["consolidation"]["granularity_errors"][0]["count_excess"],
        )
        self.report, self.reference = fixtures()
        self.report["units"][0]["text"] += " A retenção é de 30 dias."
        self.report["units"].pop(1)
        self.report["occurrences"] = [
            row for row in self.report["occurrences"] if row["unit_id"] != "u1"
        ]
        self.report["counts"].update(unique_units=2, occurrences=4)
        aligned = reviewed(
            self.report,
            self.reference,
            unit_links=[link(["r0", "r1"], ["u0"], "merged")],
        )
        result = evaluate_information_report(self.report, self.reference, aligned)
        self.assertEqual(1, result["metrics"]["consolidation"]["improper_merge_groups"])
        self.assertEqual(
            1,
            result["metrics"]["consolidation"]["granularity_errors"][0][
                "count_deficit"
            ],
        )

    def test_qualifier_content_errors_keep_reference_partial_and_annotations_separate(
        self,
    ):
        self.report["units"][2]["text"] = "O backup automático não é realizado."
        aligned = reviewed(
            self.report,
            self.reference,
            unit_links=[
                link(
                    ["r2"],
                    ["u2"],
                    "partial",
                    qualifier_errors=[
                        {
                            "field": "conditions",
                            "kind": "content",
                            "detail": "Dropped the connection condition.",
                        }
                    ],
                )
            ],
            reference_decisions=[
                {
                    "reference_unit_id": "r2",
                    "state": "partial",
                    "review_status": "verified",
                }
            ],
            report_decisions=[
                {
                    "report_unit_id": "u2",
                    "state": "partial",
                    "review_status": "verified",
                }
            ],
        )
        result = evaluate_information_report(self.report, self.reference, aligned)
        self.assertEqual(1, result["metrics"]["qualifier_errors"]["by_kind"]["content"])
        self.assertEqual(1, result["metrics"]["unit_coverage"]["confirmed_partial"])
        self.assertEqual(0, result["metrics"]["unit_coverage"]["unresolved"])

    def test_gate_false_positives_false_negatives_and_reliability_use_explicit_labels(
        self,
    ):
        self.report["candidates"][1]["validation"] = "rejected"
        self.report["candidates"][1]["signals"] = [["support", 0.2]]
        self.report["counts"]["accepted_candidates"] = 2
        aligned = reviewed(
            self.report,
            self.reference,
            sampling="targeted_controls",
            candidate_labels=[
                {
                    "candidate_id": "c0",
                    "expected_validation": "rejected",
                    "review_status": "verified",
                    "gates": {"support": False},
                },
                {
                    "candidate_id": "c1",
                    "expected_validation": "accepted",
                    "review_status": "verified",
                    "gates": {"support": True},
                },
                {
                    "candidate_id": "c2",
                    "expected_validation": "uncertain",
                    "review_status": "unverified",
                    "gates": {"support": True},
                },
            ],
        )
        result = evaluate_information_report(self.report, self.reference, aligned)
        tables = result["metrics"]["criteria_validation"]
        self.assertEqual(2, tables["reviewed_candidate_count"])
        self.assertEqual(
            (1, 1),
            (
                tables["gates"]["support"]["false_positive"],
                tables["gates"]["support"]["false_negative"],
            ),
        )
        self.assertEqual(2, len(tables["gates"]["support"]["reliability_bins"]))
        self.assertFalse(tables["thresholds_calibrated"])
        self.assertEqual("targeted_controls", tables["sampling"])

    def test_gate_boundary_uses_core_two_ulp_decision_semantics(self):
        threshold = 0.85
        below = math.nextafter(threshold, 0)
        two_below = math.nextafter(below, 0)
        three_below = math.nextafter(two_below, 0)
        for probability, accepted in (
            (threshold, True),
            (below, True),
            (two_below, True),
            (three_below, False),
            (math.nextafter(threshold, 1), True),
        ):
            for expected_yes in (True, False):
                with self.subTest(probability=probability, expected_yes=expected_yes):
                    report = copy.deepcopy(self.report)
                    report["candidates"][0]["signals"] = [["support", probability]]
                    alignment = reviewed(
                        report,
                        self.reference,
                        candidate_labels=[
                            {
                                "candidate_id": "c0",
                                "expected_validation": "accepted",
                                "review_status": "verified",
                                "gates": {"support": expected_yes},
                            }
                        ],
                    )
                    gate = evaluate_information_report(
                        report, self.reference, alignment
                    )["metrics"]["criteria_validation"]["gates"]["support"]
                    self.assertEqual(threshold, gate["recorded_threshold"])
                    self.assertEqual(
                        int(expected_yes and accepted), gate["true_positive"]
                    )
                    self.assertEqual(
                        int(expected_yes and not accepted), gate["false_negative"]
                    )
                    self.assertEqual(
                        int(not expected_yes and accepted), gate["false_positive"]
                    )
                    self.assertEqual(
                        int(not expected_yes and not accepted), gate["true_negative"]
                    )

    def test_faithful_compound_repaired_for_atomicity_is_not_a_source_false_negative(
        self,
    ):
        source = self.report["snapshot"]["source"][0]["segments"][0]["content"]
        repaired_child = copy.deepcopy(self.report["candidates"][0])
        repaired_child["id"] = "c3"
        self.report["candidates"].append(repaired_child)
        self.report["units"][0]["candidate_ids"] = ["c3"]
        self.report["candidates"][0].update(
            validation="needs_repair",
            granularity="compound",
            inventory_role="unit_candidate",
            candidate={
                "text": source,
                "evidence": [
                    {
                        "segment_id": "v0:s0",
                        "quote": source,
                        "start_char": 0,
                        "end_char": len(source),
                        "role": "assertion",
                    }
                ],
            },
        )
        self.report["candidates"][0]["signals"] = [
            ["support", 0.99],
            ["conditions", 0.99],
        ]
        self.report["counts"].update(candidates=4, accepted_candidates=3)
        for expected_validation in ("rejected", None):
            with self.subTest(expected_validation=expected_validation):
                alignment = reviewed(
                    self.report,
                    self.reference,
                    candidate_labels=[
                        {
                            "candidate_id": "c0",
                            "expected_validation": expected_validation,
                            "expected_source_fidelity": "supported",
                            "expected_granularity": "compound",
                            "expected_count_role": "requires_decomposition",
                            "review_status": "verified",
                            "gates": {"support": True, "conditions": True},
                        }
                    ],
                )
                tables = evaluate_information_report(
                    self.report, self.reference, alignment
                )["metrics"]["criteria_validation"]
                self.assertEqual(
                    (1, 0),
                    (
                        tables["gates"]["support"]["true_positive"],
                        tables["gates"]["support"]["false_negative"],
                    ),
                )
                self.assertEqual(
                    [{"expected": "compound", "predicted": "compound", "count": 1}],
                    tables["granularity_confusion_matrix"],
                )
                self.assertEqual(
                    "supported",
                    tables["candidate_results"][0]["expected_source_fidelity"],
                )
                self.assertEqual([], tables["candidate_results"][0]["counted_unit_ids"])
                self.assertFalse(
                    any(
                        row["predicted"] == "accepted"
                        for row in tables["validation_confusion_matrix"]
                    )
                )
                if expected_validation is None:
                    self.assertEqual(0, tables["overall_validation_labeled_count"])
                    self.assertEqual(
                        ["c0"], tables["overall_validation_unlabeled_candidate_ids"]
                    )
                    self.assertEqual([], tables["validation_confusion_matrix"])
                else:
                    self.assertEqual(
                        [
                            {
                                "expected": "rejected",
                                "predicted": "needs_repair",
                                "count": 1,
                            }
                        ],
                        tables["validation_confusion_matrix"],
                    )

    def test_invalid_gate_probabilities_and_thresholds_cannot_enter_tables(self):
        for value in (-0.01, 1.01, True, float("nan"), float("inf"), -float("inf")):
            with self.subTest(probability=value):
                report = copy.deepcopy(self.report)
                report["candidates"][0]["signals"] = [["support", value]]
                with self.assertRaises(ValueError):
                    alignment = reviewed(
                        report,
                        self.reference,
                        candidate_labels=[
                            {
                                "candidate_id": "c0",
                                "expected_validation": "accepted",
                                "review_status": "verified",
                                "gates": {"support": True},
                            }
                        ],
                    )
                    evaluate_information_report(report, self.reference, alignment)
            with self.subTest(threshold=value):
                report = copy.deepcopy(self.report)
                report["metadata"]["settings"]["acceptance_threshold"] = value
                with self.assertRaises(ValueError):
                    evaluate_information_report(report, self.reference)

    def test_count_integrity_detects_bad_aggregate_even_when_semantic_matches_are_correct(
        self,
    ):
        self.report["counts"].update(unique_units=99, occurrences=99)
        result = evaluate_information_report(self.report, self.reference)
        errors = result["metrics"]["traceability"]["integrity_errors"]
        self.assertEqual(
            {"unique_units", "occurrences"}, {row["field"] for row in errors}
        )
        self.assertEqual(3, result["metrics"]["source_fidelity"]["report_total"])

    def test_source_protocol_reference_spans_and_alignment_fingerprints_are_enforced(
        self,
    ):
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            evaluate_information_report(
                self.report,
                self.reference,
                reviewed(self.report, self.reference, unit_links=[])
                | {"report_fingerprint": "wrong"},
            )
        bad = copy.deepcopy(self.reference)
        bad["units"][0]["occurrences"][0]["assertion_evidence"][0][
            "quote"
        ] = "Invented quote"
        with self.assertRaisesRegex(ValueError, "not literal"):
            evaluate_information_report(self.report, bad)
        bad = copy.deepcopy(self.reference)
        bad["snapshot"]["source"][0]["segments"][0]["content"] += "changed"
        with self.assertRaisesRegex(ValueError, "differs"):
            evaluate_information_report(self.report, bad)
        self.report["protocol_version"] = "contextual-propositions-v1"
        with self.assertRaisesRegex(ValueError, "protocols differ"):
            evaluate_information_report(self.report, self.reference)
        diagnostic = evaluate_information_report(
            self.report, self.reference, allow_protocol_mismatch=True
        )
        self.assertFalse(diagnostic["provenance"]["matched_protocol"])
        with self.assertRaisesRegex(ValueError, "Cross-protocol"):
            compare_information_evaluations([diagnostic])

    def test_empty_denominators_are_unknown_not_perfect_recall(self):
        self.reference["units"] = []
        self.report["units"] = []
        self.report["occurrences"] = []
        self.report["candidates"] = []
        self.report["counts"].update(
            candidates=0, accepted_candidates=0, unique_units=0, occurrences=0
        )
        result = evaluate_information_report(self.report, self.reference)
        self.assertIsNone(result["metrics"]["unit_coverage"]["coverage_lower_bound"])
        self.assertIsNone(result["metrics"]["source_fidelity"]["fidelity_lower_bound"])

    def test_non_scorable_reference_records_are_retained_outside_denominators(self):
        self.reference["units"][0].update(
            scorable=False, exclusion_reason="Unresolved identity."
        )
        result = evaluate_information_report(self.report, self.reference)
        self.assertEqual(2, result["metrics"]["unit_coverage"]["reference_total"])
        self.assertEqual(2, result["metrics"]["occurrence_coverage"]["reference_total"])
        self.assertEqual(1, len(result["metrics"]["reference_exclusions"]["units"]))
        self.assertEqual(3, len(result["reference_unit_results"]))

    def test_changed_claimant_child_does_not_make_its_supported_parent_redundant(self):
        source = "Na semana passada, Donald Trump assinou um decreto alegando emergência nacional."
        snapshot = copy.deepcopy(self.report["snapshot"])
        snapshot["current_order"] = ["v0:s0"]
        snapshot["source"][0]["segments"] = snapshot["source"][0]["segments"][:1]
        snapshot["source"][0]["segments"][0]["content"] = source

        def occurrence(identifier, unit_id, start, end, contexts=()):
            return {
                "id": identifier,
                "unit_id": unit_id,
                "segment_id": "v0:s0",
                "assertion_evidence": [
                    {"quote": source[start:end], "start_char": start, "end_char": end}
                ],
                "context_evidence": [
                    {
                        "quote": source[left:right],
                        "start_char": left,
                        "end_char": right,
                        "segment_id": "v0:s0",
                    }
                    for left, right in contexts
                ],
            }

        reference = copy.deepcopy(self.reference)
        reference["snapshot"] = snapshot
        reference["units"] = [
            {
                "id": "r0",
                "text": "Na semana passada, Donald Trump assinou um decreto.",
                "occurrences": [occurrence("ro0", "r0", 0, 50)],
            },
            {
                "id": "r1",
                "text": "Donald Trump alegou emergência nacional ao assinar o decreto.",
                "occurrences": [occurrence("ro1", "r1", 51, 79, ((19, 50),))],
            },
        ]
        report = copy.deepcopy(self.report)
        report["snapshot"] = copy.deepcopy(snapshot)
        report["units"] = [
            {"id": "u0", "text": reference["units"][0]["text"]},
            {"id": "u1", "text": "O decreto alegava emergência nacional."},
            {"id": "up", "text": source},
        ]
        report["occurrences"] = [
            occurrence("o0", "u0", 0, 50),
            occurrence("o1", "u1", 40, 79),
            occurrence("op", "up", 0, len(source)),
        ]
        report["counts"].update(occurrences=3)
        alignment = reviewed(
            report,
            reference,
            unit_links=[
                link(["r0"], ["u0"]),
                link(
                    ["r1"],
                    ["u1"],
                    "partial",
                    qualifier_errors=[
                        {
                            "field": "attribution",
                            "kind": "content",
                            "detail": "The decree replaced Donald Trump as claimant; authorship does not establish mutual entailment.",
                        }
                    ],
                ),
                link(["r0", "r1"], ["up"], "merged"),
                link(
                    ["r0", "r1"],
                    ["up", "u0", "u1"],
                    "unresolved",
                    review_status="unverified",
                    parent_report_unit_ids=["up"],
                    component_report_unit_ids=["u0", "u1"],
                    decomposition_verified=False,
                ),
            ],
            report_decisions=[
                {
                    "report_unit_id": "u1",
                    "state": "partial",
                    "review_status": "verified",
                }
            ],
            occurrence_links=[
                {
                    "reference_occurrence_ids": ["ro0", "ro1"],
                    "report_occurrence_ids": ["op"],
                    "relation": "merged_occurrences",
                    "review_status": "verified",
                }
            ],
        )
        result = evaluate_information_report(report, reference, alignment)
        self.assertEqual(2, result["metrics"]["unit_coverage"]["covered"])
        self.assertEqual(2, result["metrics"]["occurrence_coverage"]["covered"])
        self.assertEqual(1, result["metrics"]["source_fidelity"]["confirmed_partial"])
        self.assertEqual(
            1, result["metrics"]["qualifier_errors"]["by_field"]["attribution"]
        )
        self.assertEqual(
            0,
            result["metrics"]["consolidation"]["compound_plus_components_extra_units"],
        )
        self.assertEqual(
            0,
            result["metrics"]["consolidation"][
                "compound_occurrence_parent_extra_count"
            ],
        )
        self.assertEqual(
            1, len(result["metrics"]["alignment"]["unverified_unit_links"])
        )

    def test_verified_compound_occurrence_parent_counts_extra_only_with_matching_components(
        self,
    ):
        self.report["units"].append(
            {"id": "up", "text": "Frequency and retention parent."}
        )
        self.report["occurrences"].append(
            dict(copy.deepcopy(self.report["occurrences"][0]), id="op", unit_id="up")
        )
        source = self.report["snapshot"]["source"][0]["segments"][0]["content"]
        self.report["occurrences"][-1]["assertion_evidence"] = [
            {
                "segment_id": "v0:s0",
                "quote": source,
                "start_char": 0,
                "end_char": len(source),
            }
        ]
        self.report["counts"].update(unique_units=4, occurrences=6)
        grouping = link(
            ["r0", "r1"],
            ["u0", "u1", "up"],
            "compound_plus_components",
            parent_report_unit_ids=["up"],
            component_report_unit_ids=["u0", "u1"],
            decomposition_verified=True,
        )
        aligned = reviewed(
            self.report,
            self.reference,
            unit_links=[grouping],
            occurrence_links=[
                {
                    "reference_occurrence_ids": ["ro0", "ro1"],
                    "report_occurrence_ids": ["op"],
                    "relation": "merged_occurrences",
                    "review_status": "verified",
                }
            ],
        )
        result = evaluate_information_report(self.report, self.reference, aligned)
        self.assertEqual(
            1,
            result["metrics"]["consolidation"][
                "compound_occurrence_parent_extra_count"
            ],
        )
        self.assertEqual(
            1, result["metrics"]["consolidation"]["duplicate_occurrence_extra_count"]
        )

    def test_duplicate_claim_alone_does_not_establish_source_fidelity(self):
        self.report["units"][0]["text"] = "O Alfa guarda tudo para sempre."
        self.report["units"][1]["text"] = "O Alfa guarda tudo para sempre."
        aligned = reviewed(
            self.report,
            self.reference,
            report_decisions=[
                {
                    "report_unit_id": "u0",
                    "state": "unsupported",
                    "review_status": "verified",
                },
                {
                    "report_unit_id": "u1",
                    "state": "duplicate",
                    "duplicate_of": "u0",
                    "review_status": "verified",
                },
            ],
        )
        result = evaluate_information_report(self.report, self.reference, aligned)
        self.assertEqual(1, result["metrics"]["source_fidelity"]["verified_supported"])
        self.assertEqual(1, result["metrics"]["source_fidelity"]["unresolved"])
        self.assertEqual(
            1, result["metrics"]["consolidation"]["verified_duplicate_extra_units"]
        )

    def test_file_command_is_offline_portable_and_never_overwrites_output(self):
        directory = Path(
            tempfile.mkdtemp(
                prefix="reference-evaluation-", dir=project_path("outputs")
            )
        )
        self.addCleanup(shutil.rmtree, directory)
        report_path, reference_path = (
            directory / "report.json",
            directory / "reference.json",
        )
        report_path.write_text(
            json.dumps(self.report, ensure_ascii=False), encoding="utf-8"
        )
        reference_path.write_text(
            json.dumps(self.reference, ensure_ascii=False), encoding="utf-8"
        )
        output_path = directory / "evaluation.json"
        with patch("sys.stdout"):
            self.assertEqual(
                0,
                main(
                    [
                        "--report",
                        str(report_path),
                        "--reference",
                        str(reference_path),
                        "--output",
                        str(output_path),
                    ]
                ),
            )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(
            evaluation_fingerprint(self.report),
            result["provenance"]["report_fingerprint"],
        )
        self.assertTrue(
            result["provenance"]["files"]["report"]["path"].startswith("outputs/")
        )
        with self.assertRaisesRegex(ValueError, "already exists"):
            save_information_evaluation(result, output_path)
        with self.assertRaisesRegex(ValueError, "under outputs"):
            save_information_evaluation(result, project_path("data/evaluation.json"))
        evaluation = evaluate_information_files(report_path, reference_path)
        comparison = compare_information_evaluations([evaluation, evaluation])
        self.assertEqual(2, len(comparison["rows"]))


if __name__ == "__main__":
    unittest.main()
