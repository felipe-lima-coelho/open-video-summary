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


if __name__ == "__main__":
    unittest.main()
