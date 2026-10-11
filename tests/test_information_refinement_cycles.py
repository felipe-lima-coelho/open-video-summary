"""Offline graph invariants; scripted semantic scores are not accuracy evidence."""

import json
import socket
import unittest
from dataclasses import replace
from itertools import permutations, product
from unittest.mock import patch

from test_information_analysis import ScriptedGenerator, candidate, videos
from test_information_stability import RefinementEvaluator
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer, _AnalysisRun
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import capture_snapshot
from open_video_summary.core.summarizers.information_inventory import DECOMPOSITION_QUESTIONS


class InformationRefinementCycleTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(socket, "create_connection", side_effect=AssertionError("Network disabled"))
        network.start()
        self.addCleanup(network.stop)
        self.parent = "O backup e diario e a retencao dura 30 dias e a rede e privada."
        self.intermediate = "O backup e diario e a retencao dura 30 dias."
        self.backup, self.retention, self.network = "O backup e diario.", "A retencao dura 30 dias.", "A rede e privada."
        self.signals = tuple(dict.fromkeys(DECOMPOSITION_QUESTIONS, .99).items())

    def public_report(self, source, splits, order, reverse=False):
        def generate(data, route):
            texts = order if route == "direct" else splits.get(data["parent_claim_text"], ()) if route == "decompose_candidate" else ()
            if reverse and route == "decompose_candidate":
                texts = tuple(reversed(texts))
            return [candidate("v0:s0", source, text=text) for text in texts]
        return InformationAnalyzer(ScriptedGenerator(callback=generate),
            RefinementEvaluator(atomic_intermediates=set(splits), splits=splits),
            InformationAnalysisConfig(experimental_mode="direct", concurrency=1, max_calls=256,
                max_source_sentences=0, max_literal_repairs=0, max_direct_windows=1,
                direct_window_chars=4000, pair_batch_size=1, max_candidates_per_route=12,
                max_granularity_checks=8)).analyze(capture_snapshot(videos([source])))

    def assert_public_cycle(self, report, source, cyclic_parents, dependent_parents=()):
        records = {item.id:item for item in report.candidates}
        proofs = {records[item.parent_candidate_id].candidate.text:item for item in report.decompositions
                  if item.superseded_by is None}
        self.assertEqual("partial", report.status)
        self.assertTrue(report.counts.counts_provisional)
        self.assertIn(source, {item.text for item in report.units})
        self.assertEqual(source, report.snapshot.current_segments[0].content)
        self.assertIn("decomposition_cycle", {item.kind for item in report.issues})
        self.assertFalse(any("failed" in item.kind for item in report.issues))
        for text in cyclic_parents | set(dependent_parents):
            proof = proofs[text]
            self.assertEqual("conflicting", proof.state)
            self.assertEqual("cyclic_decomposition" if text in cyclic_parents else "component_refinement_unresolved", proof.reason)
            self.assertTrue(all(dict(proof.signals)[name] == .99 for name in DECOMPOSITION_QUESTIONS))
            aliases = [item for item in report.candidates if item.candidate.text == text]
            self.assertTrue(all(item.inventory_role == "unit_candidate" and proof.id in item.decomposition_ids for item in aliases))
            self.assertTrue(all(item.inventory_state == "provisional_granularity" for item in report.units if item.text == text))
        metadata = json.loads(report.metadata_json)
        self.assertLessEqual(metadata["logical_calls"], 256)
        self.assertLessEqual(metadata["atomicity_review_requests"], 8)
        self.assertLessEqual(metadata["granularity_checks"], 8)
        return sorted((item.text, item.inventory_state) for item in report.units)

    def test_public_self_and_two_family_cycles_restore_content_in_all_52_orders(self):
        cases = (({self.parent:(self.parent, self.backup)}, (self.parent, self.backup)),
                 ({self.parent:(self.intermediate, self.network), self.intermediate:(self.parent, self.backup)},
                  (self.parent, self.intermediate, self.network, self.backup)))
        runs = 0
        for splits, texts in cases:
            observations = []
            for order, reverse in product(permutations(texts), (False, True)):
                with self.subTest(order=order, reverse=reverse):
                    report = self.public_report(self.parent, splits, order, reverse)
                    observations.append(self.assert_public_cycle(report, self.parent, set(splits)))
                    self.assertTrue(any(item.route == "decomposition" and item.candidate.text == self.parent
                                        for item in report.candidates))
                    runs += 1
            self.assertTrue(all(item == observations[0] for item in observations))
        self.assertEqual(52, runs)

    def test_public_three_family_cycle_restores_dependent_ancestor_and_aliases(self):
        source = self.parent + " Existe uma copia externa."
        external = "Existe uma copia externa."
        third = "O backup e diario e a rede e privada."
        splits = {source:(self.parent, external), self.parent:(self.intermediate, self.network),
                  self.intermediate:(third, self.retention), third:(self.parent, self.backup)}
        texts = (source, self.parent, self.intermediate, third, self.backup, self.retention, self.network, external, source)
        observations = []
        for index, reverse in product(range(len(texts)), (False, True)):
            order = texts[index:] + texts[:index]
            if reverse:
                order = tuple(reversed(order))
            with self.subTest(index=index, reverse=reverse):
                report = self.public_report(source, splits, order, reverse)
                observations.append(self.assert_public_cycle(report, source, set(splits) - {source}, (source,)))
                self.assertGreaterEqual(sum(item.candidate.text == source for item in report.candidates), 2)
        self.assertTrue(all(item == observations[0] for item in observations))

    def seeded_run(self, texts, relation=None):
        source = " ".join(texts)
        run = _AnalysisRun(InformationAnalyzer(ScriptedGenerator(),
            RefinementEvaluator(atomic_intermediates=texts, relation=relation),
            InformationAnalysisConfig(concurrency=1, max_source_sentences=0, max_literal_repairs=0,
                max_calls=256, pair_batch_size=1)), capture_snapshot(videos([source])))
        target = run.snapshot.current_segments[0]
        records = {text:run._validate(candidate(target.id, source, text=text), target, "direct", 0, (target.id,)) for text in texts}
        run._review_atomicity(target)
        return run, records

    @staticmethod
    def counters(run):
        return len(run.calls), run.budget.remaining, run.atomicity_review_requests, run.granularity_checks

    def test_circular_newcomer_cannot_supersede_grounded_alternative_or_its_ancestor(self):
        outer = "The complete service configuration was recorded."
        extra = "The service is available."
        for order in permutations(("parent_cycle", "intermediate_cycle")):
            with self.subTest(order=order):
                run, rows = self.seeded_run((outer, self.parent, self.intermediate, self.backup, self.retention, extra))
                grounded = run._record_decomposition(rows[self.parent], [rows[self.backup], rows[self.retention]], "verified", self.signals)
                ancestor = run._record_decomposition(rows[outer], [rows[self.parent], rows[extra]], "verified", self.signals)
                before = self.counters(run)
                for kind in order:
                    parent, children = ((self.parent, (self.intermediate, extra)) if kind == "parent_cycle" else
                                        (self.intermediate, (self.parent, self.backup)))
                    run._record_decomposition(rows[parent], [rows[text] for text in children], "verified", self.signals)
                proofs = {item.id:item for item in run.decompositions}
                self.assertEqual(("verified", None), (proofs[grounded.id].state, proofs[grounded.id].superseded_by))
                self.assertEqual(("verified", None), (proofs[ancestor.id].state, proofs[ancestor.id].superseded_by))
                rejected = [item for item in proofs.values() if item.id not in {grounded.id, ancestor.id}]
                self.assertTrue(all(item.state == "conflicting" and item.reason == "cyclic_decomposition" for item in rejected))
                self.assertTrue(all(item.signals == self.signals for item in proofs.values()))
                self.assertEqual(before, self.counters(run))
                self.assertFalse(run._inherit_decomposition_state())
                self.assertEqual(before, self.counters(run))

    def test_distinct_acyclic_alternatives_coexist_and_need_component_equivalence_for_resolved_count(self):
        backup_alias, retention_alias = "Backups occur daily.", "Retention lasts thirty days."
        variants = ((self.backup, self.retention), (backup_alias, retention_alias))
        for equivalent, order in product((False, True), permutations(variants)):
            with self.subTest(equivalent=equivalent, order=order):
                relations = {frozenset((self.backup, backup_alias)):"equivalent",
                             frozenset((self.retention, retention_alias)):"equivalent"} if equivalent else {}
                run, rows = self.seeded_run((self.parent, self.backup, self.retention, backup_alias, retention_alias), relations)
                for parts in order:
                    run._record_decomposition(rows[self.parent], [rows[text] for text in parts], "verified", self.signals)
                self.assertEqual(2, sum(item.state == "verified" and item.superseded_by is None for item in run.decompositions))
                units, _ = run._consolidate()
                run._finalize_semantic_inventory(units)
                self.assertEqual(2 if equivalent else 4, len(units))
                self.assertTrue(all(item.inventory_state == ("validated" if equivalent else "provisional_granularity") for item in units))
                self.assertEqual(not equivalent, "decomposition_alternatives_unresolved" in {item.kind for item in run.issues})

    def test_superseded_cycle_and_unverified_generation_ancestry_are_not_graph_edges(self):
        run, rows = self.seeded_run((self.parent, self.backup, self.retention))
        before = self.counters(run)
        cyclic = run._record_decomposition(rows[self.parent], [rows[self.parent], rows[self.backup]], "verified", self.signals)
        self.assertEqual(("conflicting", "cyclic_decomposition"), (cyclic.state, cyclic.reason))
        target = run.snapshot.current_segments[0]
        alias = run._validate(candidate(target.id, target.content, text=self.parent), target, "decomposition", 0, (target.id,))
        run._inherit_decomposition_state()
        alias = next(item for item in run.candidates if item.id == alias.id)
        self.assertEqual("unit_candidate", alias.inventory_role)
        self.assertIn(cyclic.id, alias.decomposition_ids)
        current = run._record_decomposition(rows[self.parent], [rows[self.backup], rows[self.retention]], "verified", self.signals)
        self.assertEqual("verified", current.state)
        self.assertEqual(current.id, next(item for item in run.decompositions if item.id == cyclic.id).superseded_by)
        self.assertTrue(all(item.inventory_role == "decomposed_parent" for item in run.candidates if item.candidate.text == self.parent))
        # Raw ancestry and rejected witnesses must not resurrect a graph edge.
        run.candidates = [replace(item, parent_candidate_ids=(rows[self.parent].id, rows[self.backup].id)) for item in run.candidates]
        run._record_decomposition(rows[self.backup], [rows[self.parent], rows[self.retention]], "conflicting",
            tuple((name, .01 if name == "parent_covers_components" else value) for name, value in self.signals))
        self.assertEqual("verified", next(item for item in run.decompositions if item.id == current.id).state)
        self.assertFalse(run._inherit_decomposition_state())
        self.assertEqual(before, self.counters(run))
        # Reordering retained proof history cannot change the current certificate.
        run.decompositions.reverse()
        self.assertFalse(run._inherit_decomposition_state())
        self.assertEqual("verified", next(item for item in run.decompositions if item.id == current.id).state)

    def test_fresh_identical_scores_can_revalidate_after_a_cycle_is_broken_without_erasing_history(self):
        run, rows = self.seeded_run((self.parent, self.intermediate, self.network, self.backup, self.retention))
        old_parent = run._record_decomposition(rows[self.parent], [rows[self.intermediate], rows[self.network]], "verified", self.signals)
        old_child = run._record_decomposition(rows[self.intermediate], [rows[self.parent], rows[self.backup]], "verified", self.signals)
        self.assertTrue(all(item.state == "conflicting" for item in run.decompositions))
        run._record_decomposition(rows[self.intermediate], [rows[self.backup], rows[self.retention]], "verified", self.signals)
        before = self.counters(run)
        fresh = run._verify_decomposition(run.snapshot.current_segments[0], rows[self.parent],
                                          [rows[self.intermediate], rows[self.network]])
        self.assertEqual("verified", fresh.state)
        self.assertNotEqual(old_parent.id, fresh.id)
        self.assertEqual(old_parent.signals, fresh.signals)
        prior = next(item for item in run.decompositions if item.id == old_parent.id)
        self.assertEqual(("conflicting", "cyclic_decomposition", fresh.id), (prior.state, prior.reason, prior.superseded_by))
        self.assertEqual("conflicting", next(item for item in run.decompositions if item.id == old_child.id).state)
        after = self.counters(run)
        self.assertEqual((before[0] + 1, before[1] - 1, *before[2:]), after)


if __name__ == "__main__":
    unittest.main()
