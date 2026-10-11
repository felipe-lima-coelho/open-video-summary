"""Causal inventory regressions; scripted judges are not accuracy evidence."""

import json
import unittest
from dataclasses import replace
from itertools import permutations, product

from test_information_analysis import ScriptedGenerator, candidate, videos
from test_information_inventory import InventoryEvaluator
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer, _AnalysisRun
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import CoverageFocus, InformationRelation, capture_snapshot
from open_video_summary.core.summarizers.information_evaluation import exact_proposition_key
from open_video_summary.core.summarizers.information_inventory import DECOMPOSITION_QUESTIONS
from open_video_summary.core.summarizers.information_sources import source_sentence_spans


class GranularityEvaluator(InventoryEvaluator):
    """Control granularity independently from all source-fidelity decisions."""

    def __init__(self, *, choice_probability=.99, atomicity=None, failed_gate=None, **kwargs):
        super().__init__(**kwargs)
        self.choice_probability, self.atomicity, self.failed_gate = choice_probability, atomicity, failed_gate

    def evaluate(self, context, noul=None, choice=None):
        result = super().evaluate(context, noul, choice)
        if choice and "granularity" in choice:
            result = replace(result, choice=tuple(replace(item,
                probabilities=tuple((key, self.choice_probability if key == item.selected else
                    (1 - self.choice_probability) / (len(item.probabilities) - 1)) for key, _ in item.probabilities),
                confidence=self.choice_probability) for item in result.choice))
            if self.failed_gate:
                result = replace(result, noul=tuple(replace(item, probability=.5)
                    if item.id == self.failed_gate else item for item in result.noul))
        if self.atomicity is not None and context.startswith('{') and '"claim0_text"' in context:
            state = json.loads(context)
            # Parents with an explicit witness remain compounds in the fixture.
            result = replace(result, noul=tuple(replace(item,
                probability=self.atomicity if item.id.endswith("_atomic") else 1 - self.atomicity)
                if not self.splits.get(state[item.id.rsplit("_", 1)[0] + "_text"]) else item
                for item in result.noul))
        return result


class RefinementEvaluator(InventoryEvaluator):
    """Expose a disagreement between an early atomic label and later parts."""

    def __init__(self, *, atomic_intermediates, **kwargs):
        super().__init__(**kwargs)
        self.atomic_intermediates = set(atomic_intermediates)

    def evaluate(self, context, noul=None, choice=None):
        result = super().evaluate(context, noul, choice)
        if context.startswith('{') and '"claim0_text"' in context:
            state = json.loads(context)
            result = replace(result, noul=tuple(replace(item,
                probability=.99 if item.id.endswith("_atomic") else .01)
                if state[item.id.rsplit('_', 1)[0] + '_text'] in self.atomic_intermediates else item
                for item in result.noul))
        return result


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

    def test_source_hypotheses_still_fail_fidelity_and_reference_gates(self):
        source = "Segundo Ana, o serviço talvez não retorne."
        for evaluator in (InventoryEvaluator(validation={source: .01}),
                          InventoryEvaluator(granularity={source: "needs_context"}),
                          InventoryEvaluator(granularity={source: "not_information"})):
            with self.subTest(evaluator=evaluator.granularity):
                report = InformationAnalyzer(ScriptedGenerator(), evaluator).analyze(capture_snapshot(videos([source])))
                self.assertEqual(0, report.counts.unique_units)
                self.assertNotEqual("accepted", report.candidates[0].validation)
                self.assertEqual(source, report.candidates[0].candidate.text)
                self.assertEqual("literal_identity", report.candidates[0].anchor_binding_origin)
                self.assertTrue(any(call.operation == "validate_candidate" for call in report.calls))

    def test_supported_gray_or_compound_claim_remains_provisional_with_original_decisions(self):
        source = "A regra permite o serviço por 90 dias."
        for label, probability in (("atomic", .72), ("compound", .99)):
            with self.subTest(label=label):
                evaluator = GranularityEvaluator(granularity={source: label},
                                                  choice_probability=probability, atomicity=.5)
                raw = candidate("v0:s0", source, quantities=("90 dias",))
                report = InformationAnalyzer(ScriptedGenerator({("v0:s0", "direct"): [raw]}), evaluator,
                    InformationAnalysisConfig(qa_enabled=False, max_granularity_checks=0,
                                              max_source_sentences=0)).analyze(capture_snapshot(videos([source])))
                record = report.candidates[0]
                self.assertEqual("supported_pending_granularity", record.validation)
                self.assertEqual(("granularity",), record.reasons)
                self.assertEqual((label, probability, probability),
                    (record.granularity, record.granularity_probability, record.granularity_confidence))
                self.assertEqual(raw, json.loads(record.raw_json))
                self.assertTrue(all(value >= .85 for _, value in record.signals))
                self.assertEqual((0, 1, 1), (report.counts.accepted_candidates,
                    report.counts.supported_pending_granularity_candidates, report.counts.unique_units))
                self.assertEqual("provisional_granularity", report.units[0].inventory_state)
                self.assertTrue(report.counts.counts_provisional)
                exported = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
                self.assertEqual(11, exported["schema_version"])
                self.assertEqual(record.validation, exported["candidates"][0]["validation"])
                self.assertEqual(1, exported["counts"]["supported_pending_granularity_candidates"])

    def test_supported_gray_source_receives_independent_audit_and_focus_promotion(self):
        source = "A regra permite o serviço por 90 dias."
        evaluator = GranularityEvaluator(choice_probability=.72)
        report = InformationAnalyzer(ScriptedGenerator(), evaluator).analyze(capture_snapshot(videos([source])))
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual("supported_pending_granularity", report.candidates[0].validation)
        self.assertEqual("atomic", report.candidates[0].atomicity_state)
        self.assertTrue(any(call.operation == "review_candidate_atomicity" for call in report.calls))
        run = self.run_for(source, evaluator=GranularityEvaluator(choice_probability=.72, atomicity=.5),
                           max_granularity_checks=0)
        proposal = self.validate(run, candidate("v0:s0", source), destination=[], identifier="f:gray")
        run.coverage_foci.append(CoverageFocus("f:gray", "v0:s0", proposal.candidate, proposal, state="missing"))
        target = run.snapshot.current_segments[0]
        self.assertEqual(("c0",), run._promote_validated_foci(target))
        run._refresh_foci(target)
        units, _ = run._consolidate()
        self.assertEqual("covered", run.coverage_foci[0].state)
        self.assertEqual("supported_pending_granularity", run.candidates[0].validation)
        self.assertEqual("provisional_granularity", units[0].inventory_state)

    def test_supported_status_never_bypasses_content_or_reference_failure(self):
        source = "Segundo Ana, a regra talvez não mude por 90 dias."
        for defect in ("support", "conditions", "negation", "quantities", "modality", "attribution",
                       "unresolved_references", "needs_context", "not_information"):
            with self.subTest(defect=defect):
                raw = candidate("v0:s0", source)
                if defect == "unresolved_references":
                    raw["unresolved_references"] = ["a regra"]
                label = defect if defect in {"needs_context", "not_information"} else "atomic"
                evaluator = GranularityEvaluator(granularity={source: label}, choice_probability=.72,
                    failed_gate=defect if defect in {"support", "conditions", "negation", "quantities", "modality", "attribution"} else None)
                run = self.run_for(source, evaluator=evaluator)
                record = self.validate(run, raw)
                self.assertFalse(record.content_supported)
                self.assertEqual([], run._accepted())
                self.assertNotEqual("supported_pending_granularity", record.validation)

    def test_gray_or_compound_children_cannot_replace_parent_without_atomic_evidence(self):
        source = "O backup é diário e a retenção dura 30 dias."
        parts = [candidate("v0:s0", source, text="O backup é diário."),
                 candidate("v0:s0", source, text="A retenção dura 30 dias.")]
        for atomicity, expected in ((.5, False), (.99, True)):
            with self.subTest(atomicity=atomicity):
                evaluator = GranularityEvaluator(choice_probability=.72, atomicity=atomicity,
                    splits={source: tuple(item["text"] for item in parts)})
                generator = ScriptedGenerator(callback=lambda data, route:
                    [candidate("v0:s0", source)] if route == "direct" else parts if route == "decompose_candidate" else [])
                report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
                    qa_enabled=False, max_source_sentences=0)).analyze(capture_snapshot(videos([source])))
                self.assertEqual(expected, any(item.state == "verified" for item in report.decompositions))
                self.assertEqual("decomposed_parent" if expected else "unit_candidate",
                                 report.candidates[0].inventory_role)
                self.assertEqual(2 if expected else 3, report.counts.unique_units)
                self.assertTrue(all(record.validation == "supported_pending_granularity" for record in report.candidates))
                origins = json.loads(report.metadata_json)["decomposition_only_units"]
                self.assertEqual(2, len(origins))
                self.assertTrue(all(row["inventory_state"] == "provisional_granularity"
                                    for row in origins) if not expected else True)

    def test_supported_parent_aliases_inherit_partition_across_default_source_route(self):
        source = "O backup é diário e a retenção dura 30 dias."
        parent = candidate("v0:s0", source)
        parts = [candidate("v0:s0", source, text="O backup é diário."),
                 candidate("v0:s0", source, text="A retenção dura 30 dias.")]
        generator = ScriptedGenerator(callback=lambda data, route:
            [parent, parent, *parts] if route == "direct" else [])
        evaluator = GranularityEvaluator(choice_probability=.72,
            splits={source: tuple(item["text"] for item in parts)})
        report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
            qa_enabled=False)).analyze(capture_snapshot(videos([source])))
        parents = [record for record in report.candidates if record.candidate.text == source]
        self.assertEqual(3, len(parents))
        self.assertTrue(all(record.validation == "supported_pending_granularity" for record in parents))
        self.assertTrue(all(record.inventory_role == "decomposed_parent" for record in parents))
        self.assertEqual(2, report.counts.unique_units)
        proof = next(item for item in report.decompositions if item.state == "verified")
        self.assertTrue(all(proof.id in record.decomposition_ids for record in parents))
        self.assertTrue(all({record.id for record in parents} <= set(child.parent_candidate_ids)
                            for child in report.candidates if child.id in proof.component_candidate_ids))
        self.assertEqual(1, sum(call.operation == "verify_decomposition" for call in report.calls))

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

    def test_final_atomicity_batch_includes_source_supplement_under_one_cap(self):
        source = "O serviço parou. A equipe chegou."
        proposals = [candidate("v0:s0", source, quote="O serviço parou.")]
        generator = ScriptedGenerator(callback=lambda data, route: proposals if route == "direct" else [])
        report = InformationAnalyzer(generator, InventoryEvaluator(),
            InformationAnalysisConfig(max_granularity_checks=1)).analyze(capture_snapshot(videos([source])))
        new = next(record for record in report.candidates if record.candidate.text == "A equipe chegou.")
        self.assertEqual("atomic", new.atomicity_state)
        unit = next(unit for unit in report.units if new.id in unit.candidate_ids)
        self.assertEqual("validated", unit.inventory_state)
        self.assertEqual(1, json.loads(report.metadata_json)["atomicity_review_requests"])
        self.assertNotIn("atomicity_review_budget", {issue.kind for issue in report.issues})

    def test_final_atomicity_cap_still_exposes_unaudited_source_claims(self):
        source = " ".join(f"A máquina {index} parou." for index in range(9))
        report = InformationAnalyzer(ScriptedGenerator(), InventoryEvaluator(),
            InformationAnalysisConfig(max_granularity_checks=1)).analyze(capture_snapshot(videos([source])))
        self.assertEqual(1, json.loads(report.metadata_json)["atomicity_review_requests"])
        pending = [record for record in report.candidates if record.atomicity_state == "not_reviewed"]
        self.assertEqual(1, len(pending))
        self.assertTrue(any(pending[0].id in issue.candidate_ids for issue in report.issues
                            if issue.kind == "atomicity_review_budget"))
        self.assertTrue(any(pending[0].id in unit.candidate_ids and
                            unit.inventory_state == "provisional_granularity" for unit in report.units))

    def test_late_whole_source_parent_precedes_speculative_short_parent_reviews(self):
        source = "O sistema faz o telefone funcionar e permite usar aplicativos."
        parts = [candidate("v0:s0", source, quote="O sistema faz o telefone funcionar",
                           text="O sistema faz o telefone funcionar."),
                 candidate("v0:s0", source, quote="permite usar aplicativos",
                           text="O sistema permite usar aplicativos.")]
        short = [candidate("v0:s0", source, text=text) for text in
                 ("O telefone funciona com o sistema.", "O sistema permite o funcionamento do telefone.",
                  "Usar aplicativos é possível com o sistema.", "O sistema viabiliza o uso de aplicativos.")]
        evaluator = InventoryEvaluator(
            splits={source: tuple(item["text"] for item in parts)})
        generator = ScriptedGenerator(callback=lambda data, route: [*short, *parts] if route == "direct" else [])
        report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
            qa_enabled=False, max_granularity_checks=1)).analyze(capture_snapshot(videos([source])))
        parent = next(record for record in report.candidates if record.candidate.text == source)
        self.assertEqual("source_sentence", parent.route)
        self.assertEqual("decomposed_parent", parent.inventory_role)
        reviewed = [state["parent_claim_text"] for state, questions in evaluator.inventory_calls
                    if "splittable" in questions]
        self.assertEqual([source], reviewed)
        self.assertEqual(1, json.loads(report.metadata_json)["granularity_checks"])
        self.assertFalse(any('"analysis_task":"decompose_candidate"' in request.prompt
                             for request in generator.requests))
        self.assertTrue(any(item.reason == "joint_review_budget" for item in report.decompositions))
        self.assertTrue(all(unit.text in {item["text"] for item in [*parts, *short]} for unit in report.units))

    def test_generated_exact_aliases_do_not_displace_fixed_parent_agenda_or_peers(self):
        early = "O backup é diário e a retenção dura 30 dias."
        whole = "O sistema faz o telefone funcionar e permite usar aplicativos."
        source = early + " " + whole
        parent = candidate("v0:s0", source, quote=early, quantities=("30 dias",))
        late = candidate("v0:s0", source, quote=whole)
        parts = [candidate("v0:s0", source, quote="O sistema faz o telefone funcionar",
                           text="O sistema faz o telefone funcionar."),
                 candidate("v0:s0", source, quote="permite usar aplicativos",
                           text="O sistema permite usar aplicativos.")]
        captures = []
        for aliases in ([], [late, *parts, late, *parts]):
            def generate(data, route):
                if route == "direct":
                    return [parent, *parts]
                if route == "decompose_candidate" and data["parent_claim_text"] == early:
                    return aliases
                return []
            evaluator = InventoryEvaluator(splits={early: ("O backup é diário.", "A retenção dura 30 dias."),
                whole: tuple(item["text"] for item in parts)},
                verification={early: {"parent_covers_components": .01}})
            generator = ScriptedGenerator(callback=generate)
            report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
                concurrency=1, qa_enabled=False, max_coverage_foci=0, max_literal_repairs=0,
                max_granularity_checks=2, max_calls=256)).analyze(capture_snapshot(videos([source])))
            reviews = [(state, questions) for state, questions in evaluator.inventory_calls if "splittable" in questions]
            self.assertEqual([early, whole], [state["parent_claim_text"] for state, _ in reviews])
            later_review = reviews[1]
            self.assertEqual([item["text"] for item in parts], later_review[0]["component_claim_texts"])
            self.assertEqual(2, json.loads(report.metadata_json)["granularity_checks"])
            self.assertEqual(1, sum('"analysis_task":"decompose_candidate"' in item.prompt for item in generator.requests))
            parents = [row for row in report.candidates if row.candidate.text == whole]
            self.assertTrue(all(row.inventory_role == "decomposed_parent" for row in parents))
            self.assertTrue(all(row.atomicity_state == "compound" for row in parents))
            self.assertEqual(len(aliases), sum(row.route == "decomposition" for row in report.candidates))
            captures.append((later_review, [unit.text for unit in report.units]))
        self.assertEqual(captures[0], captures[1])

    def test_exact_alias_replays_current_atomicity_before_component_eligibility(self):
        source = "O backup é diário e a retenção dura 30 dias."
        raw = candidate("v0:s0", source, quantities=("30 dias",))
        for state, expected in (("compound", False), ("atomic", True), ("uncertain", False)):
            with self.subTest(state=state):
                evaluator = GranularityEvaluator(choice_probability=.7,
                    atomicity=.5 if state == "uncertain" else None,
                    splits={source:("O backup é diário.", "A retenção dura 30 dias.")} if state == "compound" else {})
                run = self.run_for(source, evaluator=evaluator)
                original = self.validate(run, raw)
                target = run.snapshot.current_segments[0]
                run._review_atomicity(target)
                before = len(run.calls)
                alias = run._validate(raw, target, "decomposition", 0, (target.id,), parent_candidate_ids=(original.id,))
                self.assertEqual(before, len(run.calls))
                self.assertEqual("supported_pending_granularity", alias.validation)
                self.assertEqual(original.signals, alias.signals)
                self.assertEqual(raw, json.loads(alias.raw_json))
                self.assertEqual((state, original.id, original.id),
                    (alias.atomicity_state, alias.atomicity_review_origin, alias.validation_reused_from))
                self.assertEqual(expected, run._component_eligible(alias))
                # A later authoritative family state takes precedence over the
                # stale immutable alias passed by an earlier caller.
                run.atomicity_reviews[exact_proposition_key(alias)] = (
                    "uncertain", (("atomic", .5), ("compound", .5)), "new-audit-origin")
                self.assertFalse(run._component_eligible(alias))
                refreshed = run._with_known_atomicity(alias)
                self.assertEqual(("uncertain", "new-audit-origin"),
                                 (refreshed.atomicity_state, refreshed.atomicity_review_origin))

    def test_unaudited_alias_and_duplicate_children_never_fabricate_partition_proof(self):
        source = "O backup é diário e a retenção dura 30 dias."
        run = self.run_for(source, evaluator=GranularityEvaluator(choice_probability=.7))
        parent = self.validate(run, candidate("v0:s0", source))
        raw = candidate("v0:s0", source, text="O backup é diário.")
        first, alias = self.validate(run, raw), self.validate(run, raw)
        self.assertEqual(("not_reviewed", None), (alias.atomicity_state, alias.atomicity_review_origin))
        self.assertFalse(run._component_eligible(alias))
        # Even two independently accepted aliases are only one distinct meaning.
        children = [replace(row, validation="accepted", granularity="atomic", atomicity_state="atomic")
                    for row in (first, alias)]
        signals = (("splittable", .99), ("proper_part:" + first.id, .99))
        before = (len(run.calls), run.budget.remaining, run.granularity_checks)
        proof = run._verify_decomposition(run.snapshot.current_segments[0], parent, children, signals)
        self.assertEqual(before, (len(run.calls), run.budget.remaining, run.granularity_checks))
        self.assertEqual(("partial", "insufficient_distinct_components", signals),
                         (proof.state, proof.reason, proof.signals))
        self.assertEqual((first.id, alias.id), proof.component_candidate_ids)
        self.assertTrue(all(row.inventory_role == "unit_candidate" for row in run.candidates))

    def test_known_compound_alias_cannot_supply_second_eligible_component(self):
        source = "O backup é diário e a retenção dura 30 dias. A rede mudou."
        compound = candidate("v0:s0", source, quote="O backup é diário e a retenção dura 30 dias.")
        other = candidate("v0:s0", source, quote="A rede mudou.")
        run = self.run_for(source, evaluator=InventoryEvaluator(splits={compound["text"]:("O backup é diário.", "A retenção dura 30 dias.")}))
        parent = self.validate(run, candidate("v0:s0", source))
        original, atom = self.validate(run, compound), self.validate(run, other)
        target = run.snapshot.current_segments[0]
        run._review_atomicity(target)
        alias = run._validate(compound, target, "decomposition", 0, (target.id,))
        before = len(run.calls)
        proof = run._verify_decomposition(target, parent, [alias, atom])
        self.assertEqual(before, len(run.calls))
        self.assertEqual("insufficient_distinct_components", proof.reason)
        self.assertEqual("compound", alias.atomicity_state)
        self.assertEqual(original.id, alias.atomicity_review_origin)
        self.assertEqual((alias.id, atom.id), proof.component_candidate_ids)

    def test_generated_choice_atoms_require_independent_audit_before_parent_replacement(self):
        source = "Backup is daily, retention lasts 30 days and the network is private."
        compound = "Backup is daily and retention lasts 30 days."
        atom = "The network is private."
        for outcome, cap in (("compound", 2), ("uncertain", 2), ("not_reviewed", 1)):
            with self.subTest(outcome=outcome):
                class ChildEvaluator(InventoryEvaluator):
                    def evaluate(self, context, noul=None, choice=None):
                        result = super().evaluate(context, noul, choice)
                        if outcome == "uncertain" and context.startswith('{') and '"claim0_text"' in context:
                            state = json.loads(context)
                            result = replace(result, noul=tuple(replace(item, probability=.5)
                                if state[item.id.rsplit('_', 1)[0] + '_text'] == compound else item
                                for item in result.noul))
                        return result
                evaluator = ChildEvaluator(splits={source:(compound, atom),
                    compound:("Backup is daily.", "Retention lasts 30 days.")})
                generator = ScriptedGenerator(callback=lambda data, route:
                    [candidate("v0:s0", source), candidate("v0:s0", source)] if route == "direct" else
                    [candidate("v0:s0", source, text=text) for text in (compound, atom)]
                    if route == "decompose_candidate" else [])
                report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
                    experimental_mode="direct", concurrency=1, max_source_sentences=0,
                    max_literal_repairs=0, max_granularity_checks=cap, max_calls=128)).analyze(
                        capture_snapshot(videos([source])))
                parents = [row for row in report.candidates if row.candidate.text == source]
                child = next(row for row in report.candidates if row.candidate.text == compound)
                self.assertEqual("accepted", child.validation)
                self.assertEqual("atomic", child.granularity)
                self.assertEqual(outcome, child.atomicity_state)
                self.assertEqual(2, len(parents))
                self.assertTrue(all(row.inventory_role == "unit_candidate" for row in parents))
                self.assertFalse(any(proof.state == "verified" for proof in report.decompositions))
                self.assertFalse(any(call.operation == "verify_decomposition" for call in report.calls))
                proof = next(row for row in report.decompositions if row.parent_candidate_id == parents[0].id)
                self.assertEqual("insufficient_distinct_components", proof.reason)
                self.assertIn(child.id, proof.component_candidate_ids)
                self.assertEqual(3, report.counts.unique_units)
                metadata = json.loads(report.metadata_json)
                self.assertLessEqual(metadata["atomicity_review_requests"], cap)
                self.assertLessEqual(metadata["granularity_checks"], cap)
                self.assertLessEqual(metadata["logical_calls"], 128)
                if outcome == "not_reviewed":
                    self.assertIn("atomicity_review_budget", {item.kind for item in report.issues})

    def test_later_child_atomicity_invalidates_active_proof_and_restores_all_parent_aliases(self):
        source = "O backup é diário e a retenção dura 30 dias."
        parts = ("O backup é diário.", "A retenção dura 30 dias.")
        for changed_state in ("compound", "uncertain"):
            with self.subTest(changed_state=changed_state):
                class RevisingEvaluator(InventoryEvaluator):
                    revised = False
                    def evaluate(self, context, noul=None, choice=None):
                        result = super().evaluate(context, noul, choice)
                        if self.revised and context.startswith('{') and '"claim0_text"' in context:
                            state = json.loads(context)
                            result = replace(result, noul=tuple(replace(item,
                                probability=(.5 if changed_state == "uncertain" else
                                             .01 if item.id.endswith("_atomic") else .99))
                                if state[item.id.rsplit('_', 1)[0] + '_text'] == parts[0] else item
                                for item in result.noul))
                        return result
                evaluator = RevisingEvaluator(splits={source:parts})
                run = self.run_for(source, evaluator=evaluator, max_granularity_checks=8)
                parents = [self.validate(run, candidate("v0:s0", source)) for _ in range(2)]
                children = [self.validate(run, candidate("v0:s0", source, text=text)) for text in parts]
                target = run.snapshot.current_segments[0]
                run._reconcile_granularity(target)
                first = next(item for item in run.decompositions if item.state == "verified")
                current = {item.id:item for item in run.candidates}
                second = run._verify_decomposition(target, current[parents[0].id],
                    [current[item.id] for item in children], (("second_concrete_review", .99),))
                self.assertEqual("verified", second.state)
                self.assertEqual(second.id, next(item for item in run.decompositions if item.id == first.id).superseded_by)
                # Simulate a genuinely new independent audit, retaining the
                # existing request counter and global budget instead of resetting.
                evaluator.revised = True
                del run.atomicity_reviews[exact_proposition_key(children[0])]
                before_requests, before_calls = run.atomicity_review_requests, len(run.calls)
                run._review_atomicity(target)
                self.assertEqual(before_requests + 1, run.atomicity_review_requests)
                self.assertEqual(before_calls + 1, len(run.calls))
                active = next(item for item in run.decompositions if item.id == second.id)
                self.assertEqual("conflicting" if changed_state == "compound" else "uncertain", active.state)
                self.assertEqual(second.signals, active.signals)
                self.assertTrue(all(row.inventory_role == "unit_candidate" for row in run.candidates
                                    if row.candidate.text == source))
                late = self.validate(run, candidate("v0:s0", source, unit_type="definition"))
                calls = len(run.calls)
                run._inherit_decomposition_state()
                self.assertEqual(calls, len(run.calls))
                for row in run.candidates:
                    if row.candidate.text == source:
                        self.assertEqual("unit_candidate", row.inventory_role)
                        self.assertIn(active.id, row.decomposition_ids)
                current = {item.id:item for item in run.candidates}
                self.assertEqual(changed_state, current[children[0].id].atomicity_state)
                self.assertTrue(all({item.id for item in parents} <= set(current[child.id].parent_candidate_ids)
                                    for child in children))
                self.assertIn(active.id, current[late.id].decomposition_ids)
                self.assertEqual("verified", next(item for item in run.decompositions if item.id == first.id).state)
                self.assertFalse(any(item.state == "verified" and item.superseded_by is None
                                     for item in run.decompositions))

    def test_capped_peer_set_is_invariant_to_route_order_and_exact_aliases(self):
        source = "Backup is daily and retention lasts 30 days. The network is private and access is paid."
        parent = "Backup is daily and retention lasts 30 days."
        parts = ["Backup is daily.", "Retention lasts 30 days."]
        others = ["The network is private.", "Access is paid."]
        observations = []
        for first, second, duplicate in ((parts, others, False), (others, parts, False),
                (list(reversed(parts)), list(reversed(others)), True),
                (list(reversed(others)), list(reversed(parts)), True)):
            evaluator = InventoryEvaluator(splits={parent:tuple(parts)})
            def generate(data, route):
                texts = [parent, *first] if route == "direct" else [*second, *([parent] if duplicate else [])] if route == "qa" else []
                return [candidate("v0:s0", source, text=text, qa=True) for text in texts]
            report = InformationAnalyzer(ScriptedGenerator(callback=generate), evaluator,
                InformationAnalysisConfig(concurrency=1, max_source_sentences=0, max_literal_repairs=0,
                    max_candidates_per_route=3, max_granularity_checks=1, max_calls=128,
                    max_direct_windows=1, max_qa_windows=1,
                    direct_window_chars=4000, qa_window_chars=4000)).analyze(
                        capture_snapshot(videos([source])))
            review = next((state, questions) for state, questions in evaluator.inventory_calls if "splittable" in questions)
            parents = [row for row in report.candidates if row.candidate.text == parent]
            self.assertEqual(2 if duplicate else 1, len(parents))
            self.assertTrue(all(row.inventory_role == "decomposed_parent" for row in parents))
            self.assertEqual(4, report.counts.unique_units)
            self.assertEqual(3, len(review[0]["component_claim_texts"]))
            self.assertTrue(set(parts) <= set(review[0]["component_claim_texts"]))
            self.assertEqual(1, json.loads(report.metadata_json)["granularity_checks"])
            observations.append((review, [unit.text for unit in report.units]))
        self.assertTrue(all(row == observations[0] for row in observations[1:]))

    def test_public_nested_refinement_keeps_valid_terminals_and_restores_conflicted_ancestors(self):
        smallest = "O backup e diario e a retencao dura 30 dias."
        middle = "O backup e diario e a retencao dura 30 dias e a rede e privada."
        largest = "O backup e diario e a retencao dura 30 dias e a rede e privada e o acesso e pago."
        backup, retention = "O backup e diario.", "A retencao dura 30 dias."
        network, access = "A rede e privada.", "O acesso e pago."
        for deep in (False, True):
            source = largest if deep else middle
            splits = {source:(middle, access), middle:(smallest, network), smallest:(backup, retention)} if deep else {
                middle:(smallest, network), smallest:(backup, retention)}
            texts = list(dict.fromkeys([source, *(text for pair in splits.values() for text in pair)]))
            leaves = set(texts) - set(splits)
            for conflict in (False, True):
                observations = []
                for reverse in (False, True):
                    with self.subTest(deep=deep, conflict=conflict, reverse=reverse):
                        # QA windows intentionally allow at most four rows;
                        # keep all hypotheses within the native route schemas.
                        direct, qa = texts[:-2], [*texts[-2:], source]
                        if reverse:
                            direct, qa = list(reversed(direct)), list(reversed(qa))
                        evaluator = RefinementEvaluator(atomic_intermediates=set(splits) - {source}, splits=splits,
                            relation={frozenset((backup, retention)):"equivalent"} if conflict else {})
                        generator = ScriptedGenerator(callback=lambda data, route: [
                            candidate("v0:s0", source, text=text, qa=True)
                            for text in direct if route == "direct"] if route == "direct" else [
                            candidate("v0:s0", source, text=text, qa=True)
                            for text in qa if route == "qa"] if route == "qa" else [])
                        report = InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
                            experimental_mode="hybrid", concurrency=1, max_calls=256,
                            max_source_sentences=0, max_literal_repairs=0,
                            max_direct_windows=1, max_qa_windows=1, direct_window_chars=4000,
                            qa_window_chars=4000, pair_batch_size=1, max_candidates_per_route=12,
                            max_granularity_checks=8)).analyze(capture_snapshot(videos([source])))
                        records = {item.id:item for item in report.candidates}
                        proofs = {records[item.parent_candidate_id].candidate.text:item for item in report.decompositions
                                  if item.superseded_by is None and records[item.parent_candidate_id].candidate.text in splits}
                        self.assertEqual(set(splits), set(proofs))
                        self.assertTrue(all(item.state == ("conflicting" if conflict else "verified")
                                            for item in proofs.values()))
                        for text, proof in proofs.items():
                            self.assertTrue(all(dict(proof.signals)[key] == .99 for key in DECOMPOSITION_QUESTIONS))
                            if conflict and text != smallest:
                                self.assertEqual("component_refinement_unresolved", proof.reason)
                        self.assertEqual(2, sum(item.candidate.text == source for item in report.candidates))
                        self.assertTrue(all(item.inventory_role == ("unit_candidate" if conflict else "decomposed_parent")
                            for item in report.candidates if item.candidate.text in splits))
                        self.assertEqual(leaves | (set(splits) if conflict else set()), {item.text for item in report.units})
                        self.assertEqual(len(leaves) + (len(splits) if conflict else 0), report.counts.unique_units)
                        self.assertEqual(conflict, report.counts.counts_provisional)
                        self.assertTrue(all(item.inventory_state == "provisional_granularity"
                            for item in report.units if conflict and item.text in splits))
                        self.assertFalse(any("failed" in item.kind for item in report.issues))
                        metadata = json.loads(report.metadata_json)
                        self.assertLessEqual(metadata["atomicity_review_requests"], 8)
                        self.assertLessEqual(metadata["granularity_checks"], 8)
                        self.assertLessEqual(metadata["logical_calls"], 256)
                        observations.append(([(item.text, item.inventory_state) for item in report.units],
                            sorted((text, item.state, item.reason) for text, item in proofs.items())))
                self.assertEqual(2, len(observations))
                self.assertEqual(observations[0], observations[1])

    def test_deep_conflict_reconciliation_reaches_same_fixed_point_and_restores_late_aliases(self):
        source = "Backup is daily, retention is long, the network is private and access is paid."
        middle = "Backup is daily, retention is long and the network is private."
        smallest = "Backup is daily and retention is long."
        backup, retention, network, access = "Backup is daily.", "Retention is long.", "The network is private.", "Access is paid."
        splits = {source:(middle, access), middle:(smallest, network), smallest:(backup, retention)}
        observations = {"conflicting": [], "uncertain": []}
        for outcome, order in product(observations, permutations(splits)):
            with self.subTest(order=order, outcome=outcome):
                run = self.run_for(source, evaluator=RefinementEvaluator(
                    atomic_intermediates=(middle, smallest), splits=splits))
                texts = list(dict.fromkeys([source, *(text for pair in splits.values() for text in pair)]))
                rows = {text:self.validate(run, candidate("v0:s0", source, text=text)) for text in texts}
                target = run.snapshot.current_segments[0]
                run._review_atomicity(target)
                for text in order:
                    run._record_decomposition(rows[text], [rows[child] for child in splits[text]], "verified",
                        dict.fromkeys(DECOMPOSITION_QUESTIONS, .99).items())
                original = {item.id:item.signals for item in run.decompositions}
                before = (len(run.calls), run.budget.remaining, run.atomicity_review_requests, run.granularity_checks)
                if outcome == "conflicting":
                    run.relations.append(InformationRelation(rows[backup].id, rows[retention].id, "equivalent", .99, .99))
                    self.assertTrue(run._reconcile_decomposition_relations())
                else:
                    run.atomicity_reviews[exact_proposition_key(rows[backup])] = (
                        "uncertain", (("atomic", .5), ("compound", .5)), "later-uncertain-audit")
                    self.assertTrue(run._inherit_decomposition_state())
                self.assertEqual(before, (len(run.calls), run.budget.remaining, run.atomicity_review_requests, run.granularity_checks))
                self.assertTrue(all(item.state == outcome for item in run.decompositions))
                self.assertEqual(original, {item.id:item.signals for item in run.decompositions})
                for text in splits:
                    late = self.validate(run, candidate("v0:s0", source, text=text))
                    run._inherit_decomposition_state()
                    current = {item.id:item for item in run.candidates}
                    self.assertEqual("unit_candidate", current[late.id].inventory_role)
                    own_proofs = {item.id for item in run.decompositions if item.parent_candidate_id == rows[text].id}
                    self.assertTrue(own_proofs <= set(current[late.id].decomposition_ids))
                    self.assertTrue(own_proofs <= set(current[rows[text].id].decomposition_ids))
                settled = list(run.decompositions)
                self.assertFalse(run._inherit_decomposition_state())
                self.assertEqual(settled, run.decompositions)
                self.assertEqual(before, (len(run.calls), run.budget.remaining, run.atomicity_review_requests, run.granularity_checks))
                observations[outcome].append(sorted((next(text for text, row in rows.items() if row.id == item.parent_candidate_id),
                    item.state, item.reason) for item in run.decompositions))
        for results in observations.values():
            self.assertTrue(all(item == results[0] for item in results))

    def test_rejected_and_superseded_refinements_do_not_poison_current_partition(self):
        source = "Backup is daily, retention is long and the network is private."
        intermediate = "Backup is daily and retention is long."
        backup, retention, network = "Backup is daily.", "Retention is long.", "The network is private."
        splits = {source:(intermediate, network), intermediate:(backup, retention)}
        run = self.run_for(source, evaluator=RefinementEvaluator(atomic_intermediates=(intermediate,), splits=splits))
        rows = {text:self.validate(run, candidate("v0:s0", source, text=text))
                for text in (source, intermediate, backup, retention, network)}
        run._review_atomicity(run.snapshot.current_segments[0])
        signals = tuple(dict.fromkeys(DECOMPOSITION_QUESTIONS, .99).items())
        outer = run._record_decomposition(rows[source], [rows[intermediate], rows[network]], "verified", signals)
        rejected = run._record_decomposition(rows[intermediate], [rows[backup], rows[retention]], "conflicting",
            tuple((key, .01 if key == "parent_covers_components" else value) for key, value in signals))
        self.assertEqual("verified", next(item for item in run.decompositions if item.id == outer.id).state)
        child = run._record_decomposition(rows[intermediate], [rows[backup], rows[retention]], "verified", signals)
        self.assertEqual(child.id, next(item for item in run.decompositions if item.id == rejected.id).superseded_by)
        run.relations.append(InformationRelation(rows[backup].id, rows[retention].id, "equivalent", .99, .99))
        run._reconcile_decomposition_relations()
        self.assertTrue(all(item.state == "conflicting" for item in run.decompositions if item.superseded_by is None))
        before = (len(run.calls), run.budget.remaining, run.atomicity_review_requests, run.granularity_checks)
        newer = (*signals, ("new_concrete_review", .99))
        replacement_child = run._record_decomposition(rows[intermediate], [rows[backup], rows[retention]], "verified", newer)
        replacement_outer = run._record_decomposition(rows[source], [rows[intermediate], rows[network]], "verified", newer)
        self.assertEqual(("verified", "verified"), (replacement_child.state, replacement_outer.state))
        self.assertEqual({replacement_child.id, replacement_outer.id},
            {item.id for item in run.decompositions if item.superseded_by is None})
        self.assertTrue(all(item.state == "conflicting" for item in run.decompositions if item.superseded_by is not None))
        self.assertTrue(all(item.inventory_role == "decomposed_parent" for item in run.candidates
                            if item.candidate.text in splits))
        self.assertFalse(run._reconcile_decomposition_relations())
        self.assertFalse(run._inherit_decomposition_state())
        # A sound refinement supplies independently audited terminal atoms even
        # when a newer audit correctly classifies the intermediate as compound.
        run.atomicity_reviews[exact_proposition_key(rows[intermediate])] = (
            "compound", (("atomic", .01), ("compound", .99)), "later-compound-audit")
        self.assertFalse(run._inherit_decomposition_state())
        self.assertTrue(all(item.state == "verified" for item in run.decompositions if item.superseded_by is None))
        self.assertEqual(before, (len(run.calls), run.budget.remaining, run.atomicity_review_requests, run.granularity_checks))

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
        self.assertEqual("partial", run.decompositions[-1].state)
        self.assertEqual("insufficient_distinct_components", run.decompositions[-1].reason)
        self.assertEqual("unit_candidate", run.candidates[0].inventory_role)
        self.assertNotIn("parent_covers_components", dict(run.decompositions[-1].signals))
        self.assertFalse(any(call.operation == "verify_decomposition" for call in run.calls))

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
