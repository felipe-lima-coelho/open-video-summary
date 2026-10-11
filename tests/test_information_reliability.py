"""Adversarial Portuguese controls; these fixtures do not measure model accuracy."""

import copy
import json
import unittest
from dataclasses import asdict, replace
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import patch

from test_information_analysis import ScriptedGenerator, SyntheticEvaluator, candidate, videos
from test_information_inventory import InventoryEvaluator
from test_information_pairs_parallel import make_run
import test_information_recovery as recovery_helpers
from open_video_summary.__main__ import build_parser
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer, _AnalysisRun
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import CoverageRecord, capture_snapshot
from open_video_summary.core.summarizers.information_evaluation import RELATION_ADJUDICATION_QUESTIONS, anchor_binding_spec
from open_video_summary.core.summarizers.information_reliability import analysis_manifest, compare_information_reports
from open_video_summary.errors import ConfigurationError


class InformationReliabilityTests(unittest.TestCase):
    def inventory(self, source, proposals, *, components=(), evaluator=None, **settings):
        generator = ScriptedGenerator(callback=lambda data, route:
            list(components if route == "decompose_candidate" else proposals if route == "direct" else ()))
        options = dict(qa_enabled=False, experimental_mode="direct", concurrency=1, max_calls=200,
                       direct_window_chars=4000, max_literal_repairs=0, max_coverage_foci=0)
        options.update(settings)
        originals = videos([source])
        frozen = copy.deepcopy(asdict(originals[0]))
        report = InformationAnalyzer(generator, evaluator or InventoryEvaluator(),
            InformationAnalysisConfig(**options)).analyze(capture_snapshot(originals))
        self.assertEqual(frozen, asdict(originals[0]))
        return report, generator

    def test_lone_false_atomic_compound_requires_verified_complete_decomposition(self):
        source = "No Alfa, os backups são feitos a cada 24 horas e guardados por 30 dias."
        parent = candidate("v0:s0", source, quantities=("24 horas", "30 dias"))
        parts = [candidate("v0:s0", source, text="No Alfa, os backups são feitos a cada 24 horas.", quantities=("24 horas",)),
                 candidate("v0:s0", source, text="No Alfa, os backups são guardados por 30 dias.", quantities=("30 dias",))]
        evaluator = InventoryEvaluator(splits={source: tuple(item["text"] for item in parts)})
        report, _ = self.inventory(source, [parent], components=parts, evaluator=evaluator)
        self.assertEqual(("accepted", "atomic", "compound", "decomposed_parent"),
            (report.candidates[0].validation, report.candidates[0].granularity,
             report.candidates[0].atomicity_state, report.candidates[0].inventory_role))
        self.assertEqual("verified", report.decompositions[-1].state)
        self.assertEqual(2, report.counts.unique_units)
        self.assertFalse(report.counts.counts_provisional)
        self.assertTrue(all(record.atomicity_state == "atomic" for record in report.candidates[1:]))

    def test_incomplete_decomposition_retains_parent_and_provisional_granularity(self):
        source = "O backup é diário e a retenção dura 30 dias."
        evaluator = InventoryEvaluator(splits={source: ("O backup é diário.", "A retenção dura 30 dias.")})
        report, _ = self.inventory(source, [candidate("v0:s0", source)], evaluator=evaluator)
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual("unit_candidate", report.candidates[0].inventory_role)
        self.assertEqual("provisional_granularity", report.units[0].inventory_state)
        self.assertTrue(report.counts.counts_provisional)

    def test_confirmed_late_compound_gets_joint_review_before_atomic_overlaps(self):
        source = "No Alfa, os backups são feitos a cada 24 horas e guardados por 30 dias."
        parts = [candidate("v0:s0", source, text="No Alfa, os backups são feitos a cada 24 horas.", quantities=("24 horas",)),
                 candidate("v0:s0", source, text="No Alfa, os backups são guardados por 30 dias.", quantities=("30 dias",))]
        parent = candidate("v0:s0", source, quantities=("24 horas", "30 dias"))
        evaluator = InventoryEvaluator(splits={source: tuple(item["text"] for item in parts)})
        report, _ = self.inventory(source, [*parts, parent], evaluator=evaluator, max_granularity_checks=1)
        self.assertEqual([item["text"] for item in [*parts, parent]],
                         [record.candidate.text for record in report.candidates])
        self.assertEqual("decomposed_parent", report.candidates[2].inventory_role)
        verified = [item for item in report.decompositions if item.state == "verified"]
        self.assertEqual(["c2"], [item.parent_candidate_id for item in verified])
        self.assertEqual(("c0", "c1"), verified[0].component_candidate_ids)
        parent_reviews = [state for state in evaluator.inventory_states if "parent" in state]
        self.assertEqual([source, source], [state["parent"]["candidate"]["text"] for state in parent_reviews])

    def test_condition_attribution_negation_and_uncertainty_are_not_separate_facts(self):
        source = "Segundo Ana, se a rede cair, o backup talvez não seja realizado."
        proposal = candidate("v0:s0", source, attribution="Ana", conditions=("se a rede cair",),
                             negated=True, modality="talvez", unit_type="conditional")
        report, _ = self.inventory(source, [proposal])
        self.assertEqual("atomic", report.candidates[0].atomicity_state)
        self.assertEqual((), report.decompositions)
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(source, report.units[0].text)
        self.assertEqual(("Ana", True, "talvez", ("se a rede cair",)),
            (report.units[0].qualifiers.attribution, report.units[0].qualifiers.negated,
             report.units[0].qualifiers.modality, report.units[0].qualifiers.conditions))

    def test_atomicity_budget_retains_unreviewed_claims_as_visible_provisional_units(self):
        sentences = [f"A regra {index} foi publicada." for index in range(10)]
        source = " ".join(sentences)
        proposals = [candidate("v0:s0", source, quote=text) for text in sentences]
        report, _ = self.inventory(source, proposals, max_granularity_checks=1)
        self.assertEqual(10, report.counts.unique_units)
        self.assertEqual(8, sum(item.atomicity_state == "atomic" for item in report.candidates))
        self.assertEqual(2, report.counts.provisional_granularity_units)
        self.assertIn("atomicity_review_budget", {item.kind for item in report.issues})

    def pair_report(self, signals, *, followups=1):
        run = make_run(["O backup é diário", "O backup ocorre todo dia"])
        run.config = replace(run.config, max_relation_adjudications=followups)
        helper = recovery_helpers.InformationRecoveryTests()
        run.evaluator, requests = helper.evaluator(labels={("c0", "c1"): "equivalent"}, signals=signals)
        return run.run(), requests

    def test_confident_primary_equivalence_still_needs_both_directions(self):
        signals = dict.fromkeys(RELATION_ADJUDICATION_QUESTIONS, .01) | {
            "left_entails_right": .99, "right_entails_left": .99, "same_complete_meaning": .99}
        report, requests = self.pair_report(signals)
        self.assertEqual(2, len(requests))
        self.assertEqual("equivalent", report.relations[0].equivalence_state)
        self.assertTrue(report.relations[0].merged)
        self.assertEqual(signals, dict(report.relations[0].adjudication_signals))

    def test_source_scope_non_entailment_or_gray_direction_blocks_merge(self):
        for right in (.01, .5):
            with self.subTest(right=right):
                signals = dict.fromkeys(RELATION_ADJUDICATION_QUESTIONS, .01) | {
                    "left_entails_right": .99, "right_entails_left": right, "same_complete_meaning": .99}
                report, _ = self.pair_report(signals)
                self.assertEqual("uncertain", report.relations[0].equivalence_state)
                self.assertFalse(report.relations[0].merged)
                self.assertEqual(2, report.counts.unique_units)
                self.assertTrue(report.counts.counts_provisional)

    def test_followup_disabled_never_authorizes_a_confident_primary_merge(self):
        report, requests = self.pair_report({}, followups=0)
        self.assertEqual(1, len(requests))
        self.assertEqual("confirmation_pending", report.relations[0].equivalence_origin)
        self.assertEqual(2, report.counts.unique_units)
        self.assertEqual(1, json.loads(report.metadata_json)["reliability_diagnostics"]["unconfirmed_equivalence_pairs"])

    def test_equivalence_confirmation_precedes_uncertain_pairs_across_waves(self):
        run = make_run(["A primeira propriedade", "A segunda propriedade", "A terceira propriedade"])
        run.config = replace(run.config, max_calls=8, max_relation_adjudications=1, pair_batch_size=1, pair_concurrency=1)
        run.budget.limit = 8
        run._candidate_pairs = lambda records, limit: ([], list(combinations(records, 2))[:limit])
        signals = dict.fromkeys(RELATION_ADJUDICATION_QUESTIONS, .01) | {
            "left_entails_right": .99, "right_entails_left": .99, "same_complete_meaning": .99}
        helper = recovery_helpers.InformationRecoveryTests()
        run.evaluator, requests = helper.evaluator(labels={("c0", "c1"): "uncertain", ("c1", "c2"): "equivalent"}, signals=signals)
        report = run.run()
        confirmed = next(item for item in report.relations if (item.left_candidate_id, item.right_candidate_id) == ("c1", "c2"))
        self.assertEqual("equivalent", confirmed.equivalence_state)
        self.assertEqual(1, json.loads(report.metadata_json)["relation_adjudications"])
        followup = next(request for request in requests if "left_entails_right" in request["questions"])
        self.assertIn("Left candidate id: c1", followup["state"])

    def test_anchor_binding_receives_local_subject_and_condition_but_no_other_sentence(self):
        source = "Se a rede cair, os backups não são realizados. A retenção dura 30 dias."
        analyzer = InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator())
        run = _AnalysisRun(analyzer, capture_snapshot(videos([source])))
        target = run.snapshot.current_segments[0]
        claim, _ = run._literal_candidate(candidate(target.id, source, quote="não são realizados",
            text="Se a rede cair, os backups não são realizados."), target, (target.id,))
        state, _ = anchor_binding_spec(claim, target)
        self.assertIn("Se a rede cair, os backups <selected-assertion>não são realizados</selected-assertion>", state)
        self.assertNotIn("A retenção", state)

    def test_quoted_sentence_boundary_isolates_binding_and_atomicity_source(self):
        source = 'Ana afirmou: "Não haverá aumento." A decisão entra em vigor amanhã.'
        raw = candidate("v0:s0", source, quote="Não haverá aumento.", text="Segundo Ana, não haverá aumento.",
                        attribution="Ana", negated=True)
        evaluator = InventoryEvaluator()
        report, _ = self.inventory(source, [raw], evaluator=evaluator)
        review = next(state for state in evaluator.inventory_states if "atomicity_review" in state)
        marked = review["atomicity_review"][0]["assertion_scopes"][0]["marked_text"]
        self.assertIn("<selected-assertion>Não haverá aumento.</selected-assertion>", marked)
        self.assertNotIn("A decisão", marked)
        state, _ = anchor_binding_spec(report.candidates[0].candidate, report.snapshot.current_segments[0])
        self.assertNotIn("A decisão", state)
        self.assertEqual(source, report.snapshot.current_segments[0].content)

    def test_exact_recovery_packing_drops_optional_context_and_preserves_cited_context(self):
        source = ["O prazo permanece igual.", "O prazo é 30 dias.", "Contexto não citado. " * 600]
        run = _AnalysisRun(InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator(),
            InformationAnalysisConfig(max_context_chars=6000)), capture_snapshot(videos(source)))
        target, cited, optional = run.snapshot.current_segments
        raw = candidate(target.id, target.content, contexts=({"segment_id": cited.id, "quote": cited.content,
            "start_char": 0, "end_char": len(cited.content), "role": "context"},))
        run._validate(raw, target, "direct", 0, (target.id, cited.id))
        audit = CoverageRecord(target.id, 0, ("c0",), (), (), "gap_detected")
        prompt, spec, data = run._extraction_request(target, "recovery", audit, context=(cited, optional))
        self.assertLessEqual(len(prompt), run.config.max_context_chars)
        self.assertEqual([cited.id], [item["id"] for item in data["context"]])
        self.assertEqual(target.content, data["target"]["text"])
        self.assertIn(cited.id, spec.segment_ids)
        self.assertEqual(raw["text"], data["accepted"][0]["text"])

    def test_saturated_source_window_subdivides_without_dropping_original_proposals(self):
        sentences = ["O backup ocorre todo dia.", "A retenção dura 30 dias.",
                     "A rede usa conexão local.", "O arquivo é criptografado."]
        source = " ".join(sentences)
        def discover(data, route):
            window = data["discovery_window"]
            return [candidate("v0:s0", source, quote=text, qa=True) for text in sentences
                    if window["start_char"] <= source.index(text) < window["end_char"]]
        generator = ScriptedGenerator(callback=discover)
        report = InformationAnalyzer(generator, SyntheticEvaluator(), InformationAnalysisConfig(
            experimental_mode="qa", concurrency=1, qa_window_chars=4000, max_qa_windows=8,
            max_granularity_checks=0, max_literal_repairs=0, max_calls=200)).analyze(capture_snapshot(videos([source])))
        windows = json.loads(report.metadata_json)["qa_source_windows"]
        self.assertEqual("saturated", windows[0]["status"])
        self.assertTrue(all(item["status"] == "completed" for item in windows[1:]))
        self.assertEqual(sentences, [record.candidate.text for record in report.candidates[:4]])
        self.assertEqual(4, report.counts.unique_units)
        self.assertEqual(4, report.counts.occurrences)
        self.assertEqual(source, report.snapshot.current_segments[0].content)

    def test_essential_oversized_source_stays_explicit_and_is_never_sent_truncated(self):
        source = "A regra precisa de toda esta descrição. " * 200
        generator = ScriptedGenerator()
        report = InformationAnalyzer(generator, SyntheticEvaluator(), InformationAnalysisConfig(
            experimental_mode="direct", concurrency=1, max_context_chars=6000,
            max_granularity_checks=0, max_literal_repairs=0)).analyze(capture_snapshot(videos([source])))
        self.assertEqual([], generator.requests)
        self.assertEqual(source, report.snapshot.current_segments[0].content)
        self.assertIn("context_budget_exceeded", {item.kind for item in report.issues})
        self.assertTrue(report.counts.counts_provisional)

    def test_fresh_reproducible_runs_have_stable_manifest_and_digest_without_replay(self):
        source = "O backup é diário."
        reports, generators = [], []
        for _ in range(2):
            report, generator = self.inventory(source, [candidate("v0:s0", source)], reproducible=True)
            reports.append(report.to_dict())
            generators.append(generator)
        self.assertNotEqual(reports[0]["run_id"], reports[1]["run_id"])
        self.assertTrue(all(len(generator.requests) == 1 for generator in generators))
        self.assertEqual(reports[0]["metadata"]["content_digest"], reports[1]["metadata"]["content_digest"])
        self.assertTrue(compare_information_reports(reports)["matching_manifests"])
        changed = copy.deepcopy(reports[1])
        changed["units"][0]["text"] = "O backup é semanal."
        comparison = compare_information_reports([reports[0], changed])
        self.assertEqual(0.0, comparison["pairs"][0]["exact_unit_wording"]["jaccard"])
        self.assertEqual(reports[0]["counts"]["unique_units"], changed["counts"]["unique_units"])

    def test_reproducible_profile_validation_cli_and_endpoint_redaction(self):
        settings = InformationAnalysisConfig(reproducible=True, concurrency=8, pair_concurrency=8, pair_batch_size=8)
        self.assertEqual((1, 1, 1), (settings.target_workers, settings.pair_workers, settings.pair_batch_items))
        with self.assertRaises(ConfigurationError):
            InformationAnalysisConfig(reproducible="true")
        args = build_parser().parse_args(["analyze-information", "--dataset", "data/processed/bebe_real.json", "--information-reproducible"])
        self.assertTrue(args.information_reproducible)
        adapter = SimpleNamespace(config=SimpleNamespace(base_url="https://user:secret@example.org/v1?key=secret#secret"))
        manifest = analysis_manifest(capture_snapshot(videos(["Texto." ])), settings, {}, adapter, adapter, lambda _: {})
        self.assertNotIn("secret", json.dumps(manifest))
        self.assertEqual("https://example.org/v1", manifest["manifest"]["generator"]["endpoint"])


if __name__ == "__main__":
    unittest.main()
