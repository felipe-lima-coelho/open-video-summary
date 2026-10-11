"""Causal inventory regressions; scripted judges are not accuracy evidence."""

import json
import unittest
from dataclasses import replace

from test_information_analysis import ScriptedGenerator, candidate, videos
from test_information_inventory import InventoryEvaluator
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer, _AnalysisRun
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import CoverageFocus, capture_snapshot
from open_video_summary.core.summarizers.information_evaluation import exact_proposition_key
from open_video_summary.core.summarizers.information_sources import source_sentence_spans


class InformationStabilityTests(unittest.TestCase):
    def run_for(self, source, *, evaluator=None, generator=None, **settings):
        options = dict(concurrency=1, qa_enabled=False, max_calls=256,
                       max_literal_repairs=0, pair_batch_size=1)
        options.update(settings)
        return _AnalysisRun(InformationAnalyzer(generator or ScriptedGenerator(),
            evaluator or InventoryEvaluator(), InformationAnalysisConfig(**options)),
            capture_snapshot(videos([source])))

    def validate(self, run, raw, **kwargs):
        target = run.snapshot.current_segments[0]
        return run._validate(raw, target, "direct", 0, (target.id,), **kwargs)

    def test_validated_missing_focus_is_preserved_without_regeneration(self):
        source = "Segundo o ministério, a acusação é uma informação falsa."
        run = self.run_for(source)
        raw = candidate("v0:s0", source, attribution="o ministério")
        destination = []
        proposal = self.validate(run, raw, destination=destination, identifier="focus:test")
        run.coverage_foci.append(CoverageFocus("focus:test", "v0:s0", proposal.candidate, proposal,
                                             state="missing"))
        target = run.snapshot.current_segments[0]
        promoted = run._promote_validated_foci(target)
        run._refresh_foci(target)
        self.assertEqual(("c0",), promoted)
        self.assertEqual([], run.generator.requests)
        self.assertEqual(("coverage_promotion", "focus:test", "focus:test", "atomic"),
            (run.candidates[0].route, run.candidates[0].validation_reused_from,
             run.candidates[0].recovery_focus_id, run.candidates[0].atomicity_state))
        self.assertEqual(raw, json.loads(run.candidates[0].raw_json))
        self.assertEqual("covered", run.coverage_foci[0].state)
        self.assertEqual((), run._promote_validated_foci(target))

    def test_unvalidated_focus_and_decomposed_parent_cannot_be_promoted(self):
        source = "O backup é diário."
        run = self.run_for(source)
        record = self.validate(run, candidate("v0:s0", source))
        run.candidates[0] = replace(record, inventory_role="decomposed_parent")
        for index, proposal in enumerate((record, replace(record, validation="needs_review"))):
            run.coverage_foci.append(CoverageFocus(f"f{index}", "v0:s0", proposal.candidate, proposal))
        self.assertEqual((), run._promote_validated_foci(run.snapshot.current_segments[0]))
        self.assertEqual(1, len(run.candidates))

    def test_type_annotation_does_not_duplicate_identical_qualified_source_claim(self):
        source = "Segundo Ana, talvez o backup não seja diário."
        run = self.run_for(source, max_granularity_checks=0)
        a = self.validate(run, candidate("v0:s0", source, attribution="Ana", negated=True,
                                        modality="talvez", unit_type="assertion"))
        b = self.validate(run, candidate("v0:s0", source, attribution="Ana", negated=True,
                                        modality="talvez", unit_type="opinion"))
        before = len(run.calls)
        units, occurrences = run._consolidate()
        self.assertEqual(1, len(units))
        self.assertEqual(1, len(occurrences))
        self.assertEqual(before, len(run.calls))
        self.assertEqual({a.id, b.id}, set(units[0].candidate_ids))
        self.assertEqual({"assertion", "opinion"}, {row.candidate.unit_type for row in run.candidates})
        self.assertEqual(source, units[0].text)

    def test_exact_identity_preserves_occurrence_context_and_complete_meaning(self):
        source = "A regra mudou. A regra mudou."
        run = self.run_for(source, max_granularity_checks=0)
        first = self.validate(run, candidate("v0:s0", source, quote="A regra mudou."))
        evidence = first.candidate.evidence[0]
        second = replace(first, candidate=replace(first.candidate, evidence=(
            replace(evidence, start_char=14, end_char=28),)))
        different = replace(first, candidate=replace(first.candidate, text="Talvez a regra mudou."))
        context = replace(first, candidate=replace(first.candidate, evidence=(
            evidence, replace(evidence, role="context"))))
        reordered = replace(context, candidate=replace(context.candidate,
                                                     evidence=tuple(reversed(context.candidate.evidence))))
        for other in (second, different, context):
            self.assertNotEqual(exact_proposition_key(first), exact_proposition_key(other))
        self.assertEqual(exact_proposition_key(context), exact_proposition_key(reordered))

    def test_gray_splittability_can_propose_but_only_complete_witnesses_replace(self):
        source = "O backup é diário e a retenção dura 30 dias."
        parts = [candidate("v0:s0", source, text="O backup é diário."),
                 candidate("v0:s0", source, text="A retenção dura 30 dias.", quantities=("30 dias",))]
        for preservation, expected in ((.99, "verified"), (.5, "uncertain"), (.01, "conflicting")):
            with self.subTest(preservation=preservation):
                evaluator = InventoryEvaluator(splits={source: tuple(item["text"] for item in parts)},
                    joint={source: {"splittable": .6}},
                    verification={source: {"qualifiers_preserved": preservation}})
                generator = ScriptedGenerator(callback=lambda data, route:
                    parts if route == "decompose_candidate" else [])
                run = self.run_for(source, evaluator=evaluator, generator=generator)
                parent = self.validate(run, candidate("v0:s0", source, quantities=("30 dias",)))
                target = run.snapshot.current_segments[0]
                run._reconcile_granularity(target)
                self.assertEqual(expected, run.decompositions[-1].state)
                self.assertEqual("decomposed_parent" if expected == "verified" else "unit_candidate",
                                 run.candidates[0].inventory_role)
                self.assertEqual(parent.raw_json, run.candidates[0].raw_json)
                self.assertEqual(.6, dict(run.decompositions[-1].signals)["splittable"])
                sent = len(generator.requests)
                run._reconcile_granularity(target)
                self.assertEqual(sent, len(generator.requests))

    def test_grouping_and_representative_are_independent_of_proposal_order(self):
        source = "O backup é diário. A retenção dura 30 dias."
        proposals = [candidate("v0:s0", source, quote="O backup é diário."),
                     candidate("v0:s0", source, quote="A retenção dura 30 dias.", quantities=("30 dias",))]
        observed = []
        for ordered in (proposals, list(reversed(proposals))):
            run = self.run_for(source, max_granularity_checks=0)
            for raw in ordered:
                self.validate(run, raw)
            units, _ = run._consolidate()
            observed.append([(unit.id, unit.text, unit.unit_type) for unit in units])
        self.assertEqual(observed[0], observed[1])

    def test_default_pipeline_visits_omitted_source_without_counting_slots(self):
        source = "O serviço parou. A equipe chegou."
        generator = ScriptedGenerator()
        report = InformationAnalyzer(generator, InventoryEvaluator()).analyze(capture_snapshot(videos([source])))
        rows = json.loads(report.metadata_json)["source_sentence_hypotheses"]
        self.assertEqual(64, InformationAnalysisConfig().max_source_sentences)
        self.assertEqual(["O serviço parou.", "A equipe chegou."], [row["quote"] for row in rows])
        self.assertEqual(["accepted", "accepted"], [row["status"] for row in rows])
        self.assertEqual({"source_sentence"}, {record.route for record in report.candidates})
        self.assertEqual(2, report.counts.unique_units)
        self.assertTrue(all(record.atomicity_state == "atomic" for record in report.candidates))
        self.assertEqual(source, report.snapshot.current_segments[0].content)
        operations = [call.operation for call in report.calls]
        self.assertLess(operations.index("coverage_audit"), operations.index("validate_candidate"))

    def test_source_hypotheses_still_fail_fidelity_references_and_compound_gates(self):
        source = "Segundo Ana, o serviço talvez não retorne."
        for evaluator in (InventoryEvaluator(validation={source: .01}),
                          InventoryEvaluator(granularity={source: "needs_context"}),
                          InventoryEvaluator(granularity={source: "compound"})):
            with self.subTest(evaluator=evaluator.granularity):
                report = InformationAnalyzer(ScriptedGenerator(), evaluator).analyze(capture_snapshot(videos([source])))
                self.assertEqual(0, report.counts.unique_units)
                self.assertNotEqual("accepted", report.candidates[0].validation)
                self.assertEqual(source, report.candidates[0].candidate.text)
                self.assertEqual("literal_identity", report.candidates[0].anchor_binding_origin)
                self.assertTrue(any(call.operation == "validate_candidate" for call in report.calls))

    def test_sentence_visitation_handles_abbreviations_unicode_numbers_and_final_fragment(self):
        source = '  Dr. João mede 3.14 m. "A ação mudou!" Sr. A. Silva chegou. sem pontuação  '
        spans = source_sentence_spans(source)
        self.assertEqual(['Dr. João mede 3.14 m.', '"A ação mudou!"',
                          'Sr. A. Silva chegou.', 'sem pontuação'], [source[a:b] for a, b in spans])
        visited = {i for start, end in spans for i in range(start, end)}
        self.assertTrue(all(i in visited for i, char in enumerate(source) if not char.isspace()))
        self.assertTrue(all(a < b <= c < d for (a, b), (c, d) in zip(spans, spans[1:])))

    def test_source_cap_and_never_started_targets_leave_visible_pending_spans(self):
        source = "O serviço parou. A equipe chegou."
        limited = InformationAnalyzer(ScriptedGenerator(), InventoryEvaluator(),
            InformationAnalysisConfig(max_source_sentences=1)).analyze(capture_snapshot(videos([source])))
        rows = json.loads(limited.metadata_json)["source_sentence_hypotheses"]
        self.assertEqual(("pending", "source_sentence_limit"), (rows[1]["status"], rows[1]["reason"]))
        self.assertTrue(limited.counts.counts_provisional)
        exhausted = InformationAnalyzer(ScriptedGenerator(), InventoryEvaluator(),
            InformationAnalysisConfig(max_calls=1)).analyze(capture_snapshot(videos([source, source])))
        rows = json.loads(exhausted.metadata_json)["source_sentence_hypotheses"]
        self.assertEqual(4, len(rows))
        self.assertTrue(all(row["status"] == "pending" for row in rows))
        self.assertTrue(any(row["reason"] == "source_target_unvisited" for row in rows))
        self.assertEqual(1, json.loads(exhausted.metadata_json)["logical_calls"])
        self.assertTrue(exhausted.counts.counts_provisional)

    def test_source_supplement_reuses_open_parent_adapters_after_target_forks_close(self):
        evaluators, generators = [], []

        class ForkEvaluator(InventoryEvaluator):
            def __init__(self):
                super().__init__()
                self.closed = False
                evaluators.append(self)
            def fork(self):
                return ForkEvaluator()
            def close(self):
                self.closed = True
            def evaluate(self, *args, **kwargs):
                if self.closed:
                    raise AssertionError("Closed evaluator reused")
                return super().evaluate(*args, **kwargs)

        class ForkGenerator(ScriptedGenerator):
            def __init__(self):
                super().__init__()
                self.closed = False
                generators.append(self)
            def fork(self):
                return ForkGenerator()
            def close(self):
                self.closed = True
            def generate(self, *args, **kwargs):
                if self.closed:
                    raise AssertionError("Closed generator reused")
                return super().generate(*args, **kwargs)

        generator, evaluator = ForkGenerator(), ForkEvaluator()
        report = InformationAnalyzer(generator, evaluator).analyze(
            capture_snapshot(videos(["O serviço parou.", "A equipe chegou."])))
        self.assertFalse(generator.closed)
        self.assertFalse(evaluator.closed)
        self.assertTrue(all(item.closed for item in generators[1:]))
        self.assertTrue(all(item.closed for item in evaluators[1:]))
        self.assertTrue(all(row["status"] == "accepted"
                           for row in json.loads(report.metadata_json)["source_sentence_hypotheses"]))
        self.assertFalse(any(item.kind == "source_sentence_analysis_failed" for item in report.issues))

    def test_source_supplement_does_not_reset_spent_atomicity_cap(self):
        source = "O serviço parou. A equipe chegou."
        proposals = [candidate("v0:s0", source, quote="O serviço parou.")]
        generator = ScriptedGenerator(callback=lambda data, route: proposals if route == "direct" else [])
        report = InformationAnalyzer(generator, InventoryEvaluator(),
            InformationAnalysisConfig(max_granularity_checks=1)).analyze(capture_snapshot(videos([source])))
        new = next(record for record in report.candidates if record.candidate.text == "A equipe chegou.")
        self.assertEqual("not_reviewed", new.atomicity_state)
        unit = next(unit for unit in report.units if new.id in unit.candidate_ids)
        self.assertEqual("provisional_granularity", unit.inventory_state)
        self.assertEqual(1, json.loads(report.metadata_json)["atomicity_review_requests"])
        self.assertIn("atomicity_review_budget", {issue.kind for issue in report.issues})

    def test_source_qualified_child_must_still_be_a_part_of_the_explicit_parent(self):
        source = "O sistema faz o telefone funcionar e permite usar aplicativos."
        parent = candidate("v0:s0", source, text="O sistema faz o telefone funcionar.")
        child = candidate("v0:s0", source, text="O sistema permite usar aplicativos.")
        evaluator = InventoryEvaluator(splits={parent["text"]: (child["text"],)},
            joint={parent["text"]: {"splittable": .6}},
            verification={parent["text"]: {"parent_covers_components": .01}})
        generator = ScriptedGenerator(callback=lambda data, route:
            [child] if route == "decompose_candidate" else [])
        run = self.run_for(source, evaluator=evaluator, generator=generator)
        self.validate(run, parent)
        run._reconcile_granularity(run.snapshot.current_segments[0])
        self.assertEqual("accepted", run.candidates[1].validation)
        self.assertEqual("conflicting", run.decompositions[-1].state)
        self.assertEqual("unit_candidate", run.candidates[0].inventory_role)
        self.assertEqual(.01, dict(run.decompositions[-1].signals)["parent_covers_components"])

    def test_source_span_identity_does_not_depend_on_generator_order(self):
        source = "O serviço parou. A equipe chegou."
        proposals = [candidate("v0:s0", source, quote="O serviço parou."),
                     candidate("v0:s0", source, quote="A equipe chegou.")]
        ledgers = []
        for ordered in (proposals, list(reversed(proposals))):
            generator = ScriptedGenerator(callback=lambda data, route:
                ordered if route == "direct" else [])
            report = InformationAnalyzer(generator, InventoryEvaluator()).analyze(capture_snapshot(videos([source])))
            ledgers.append([(row["id"], row["start_char"], row["end_char"], row["quote"])
                            for row in json.loads(report.metadata_json)["source_sentence_hypotheses"]])
        self.assertEqual(ledgers[0], ledgers[1])

    def test_source_sentence_configuration_and_cli_ablation_are_explicit(self):
        from open_video_summary.__main__ import build_parser
        from open_video_summary.errors import ConfigurationError
        for command in ("analyze-information", "summarize"):
            args = build_parser().parse_args([command, "--information-source-sentences", "0"])
            self.assertEqual(0, args.information_source_sentences)
        for value in (-1, 257, True):
            with self.assertRaises(ConfigurationError):
                InformationAnalysisConfig(max_source_sentences=value)

    def decomposed_alias_report(self, *, parent_count=1, source_sentences=64, conflict=False):
        source = "O backup é diário e a retenção dura 30 dias."
        parent = candidate("v0:s0", source, quantities=("30 dias",))
        parts = [candidate("v0:s0", source, text="O backup é diário."),
                 candidate("v0:s0", source, text="A retenção dura 30 dias.", quantities=("30 dias",))]
        proposals = [*[parent for _ in range(parent_count)], *parts]
        generator = ScriptedGenerator(callback=lambda data, route: proposals if route == "direct" else [])
        evaluator = InventoryEvaluator(splits={source: tuple(item["text"] for item in parts)},
            relation={frozenset(item["text"] for item in parts): "equivalent"} if conflict else None)
        report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
            qa_enabled=False, max_source_sentences=source_sentences)).analyze(capture_snapshot(videos([source])))
        return report, generator, source, parts

    def test_verified_partition_suppresses_existing_and_late_exact_parent_aliases(self):
        for parent_count in (1, 2):
            for source_sentences in (0, 64):
                with self.subTest(parents=parent_count, supplement=source_sentences):
                    report, generator, source, parts = self.decomposed_alias_report(
                        parent_count=parent_count, source_sentences=source_sentences)
                    parents = [record for record in report.candidates if record.candidate.text == source]
                    self.assertEqual(parent_count + bool(source_sentences), len(parents))
                    self.assertTrue(all(record.inventory_role == "decomposed_parent" for record in parents))
                    self.assertEqual({item["text"] for item in parts}, {unit.text for unit in report.units})
                    self.assertEqual(2, report.counts.unique_units)
                    self.assertEqual(len(parents), report.counts.decomposed_parent_candidates)
                    verified = [item for item in report.decompositions
                                if item.state == "verified" and item.superseded_by is None]
                    self.assertEqual(1, len(verified))
                    self.assertTrue(all(verified[0].id in record.decomposition_ids for record in parents))
                    children = [record for record in report.candidates
                                if record.id in verified[0].component_candidate_ids]
                    self.assertTrue(all({record.id for record in parents} <= set(child.parent_candidate_ids)
                                        for child in children))
                    self.assertEqual(1, sum(call.operation == "verify_decomposition" for call in report.calls))
                    self.assertEqual(1, sum(call.operation == "review_candidate_atomicity" for call in report.calls))
                    self.assertFalse(any('"analysis_task":"decompose_candidate"' in request.prompt
                                         for request in generator.requests))
                    self.assertTrue(all(json.loads(record.raw_json)["text"] == source for record in parents))

    def test_partition_conflict_restores_all_exact_aliases_as_one_parent_unit(self):
        for source_sentences in (0, 64):
            with self.subTest(supplement=source_sentences):
                report, _, source, _ = self.decomposed_alias_report(
                    parent_count=2, source_sentences=source_sentences, conflict=True)
                parents = [record for record in report.candidates if record.candidate.text == source]
                self.assertTrue(all(record.inventory_role == "unit_candidate" for record in parents))
                self.assertEqual(0, report.counts.decomposed_parent_candidates)
                active = [item for item in report.decompositions if item.superseded_by is None]
                self.assertEqual(["conflicting"], [item.state for item in active])
                self.assertTrue(all(active[0].id in record.decomposition_ids for record in parents))
                parent_units = [unit for unit in report.units if unit.text == source]
                self.assertEqual(1, len(parent_units))
                self.assertEqual({record.id for record in parents}, set(parent_units[0].candidate_ids))
                self.assertEqual("provisional_granularity", parent_units[0].inventory_state)
                self.assertEqual(3, report.counts.unique_units)
                self.assertTrue(report.counts.counts_provisional)
                self.assertEqual(1, sum(call.operation == "verify_decomposition" for call in report.calls))

    def test_partition_inheritance_never_crosses_qualified_text_or_reference_scope(self):
        source = "Segundo Ana, o backup é diário e a retenção dura 30 dias."
        parent = candidate("v0:s0", source, attribution="Ana", quantities=("30 dias",))
        children = [candidate("v0:s0", source, text=text, attribution="Ana") for text in
                    ("Segundo Ana, o backup é diário.", "Segundo Ana, a retenção dura 30 dias.")]
        attributed = candidate("v0:s0", source,
            text="O backup é diário e a retenção dura 30 dias.", quantities=("30 dias",))
        other_scope_raw = candidate("v0:s0", source, attribution="Ana", quantities=("30 dias",))
        other_scope_raw["evidence"].append({"segment_id": "v0:s0", "quote": "Ana",
                                          "start_char": 8, "end_char": 11, "role": "context"})
        for alternative in (attributed, other_scope_raw):
            with self.subTest(alternative=alternative):
                proposals = [parent, *children, alternative]
                generator = ScriptedGenerator(callback=lambda data, route:
                    proposals if route == "direct" else [])
                # The scripted judge deliberately accepts the alternatives;
                # this tests proof scope, not their actual semantic correctness.
                evaluator = InventoryEvaluator(splits={source: tuple(item["text"] for item in children)})
                report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
                    qa_enabled=False, max_granularity_checks=1)).analyze(capture_snapshot(videos([source])))
                records = {record.id: record for record in report.candidates}
                proof = next(item for item in report.decompositions if item.state == "verified")
                self.assertEqual("decomposed_parent", records["c0"].inventory_role)
                self.assertEqual("unit_candidate", records["c3"].inventory_role)
                self.assertNotIn(proof.id, records["c3"].decomposition_ids)
                self.assertEqual(3, report.counts.unique_units)
                self.assertEqual(1, sum(call.operation == "verify_decomposition" for call in report.calls))

    def test_late_alias_of_invalidated_partition_retains_conflict_provenance(self):
        source = "O backup é diário e a retenção dura 30 dias."
        run = self.run_for(source)
        parent = self.validate(run, candidate("v0:s0", source))
        children = [self.validate(run, candidate("v0:s0", source, text=text))
                    for text in ("O backup é diário.", "A retenção dura 30 dias.")]
        proof = run._record_decomposition(parent, children, "verified")
        run.decompositions[0] = replace(proof, state="conflicting",
                                        reason="component_equivalence_conflicts_with_joint_distinctness")
        late = self.validate(run, candidate("v0:s0", source, unit_type="definition"))
        calls = len(run.calls)
        run._inherit_decomposition_state()
        records = {record.id: record for record in run.candidates}
        self.assertEqual(calls, len(run.calls))
        for identifier in (parent.id, late.id):
            self.assertEqual("unit_candidate", records[identifier].inventory_role)
            self.assertIn(proof.id, records[identifier].decomposition_ids)


if __name__ == "__main__":
    unittest.main()
