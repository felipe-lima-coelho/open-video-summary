"""Offline v2 inventory behavior; fixtures do not measure model accuracy."""

import copy
import io
import json
import socket
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from unittest.mock import patch

from test_information_analysis import ScriptedGenerator, SyntheticEvaluator, candidate, videos
from open_video_summary.adapters.typesafe import EvaluationMetadata, EvaluationResult, NoulResult
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer, _AnalysisRun
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import (
    CoverageFocus, CoverageMatch, InformationRelation, capture_snapshot,
)
from open_video_summary.core.summarizers.information_inventory import DECOMPOSITION_QUESTIONS
from open_video_summary.errors import AuthenticationError, ConfigurationError, InvalidResponseError, RequestCancelledError


class InventoryEvaluator(SyntheticEvaluator):
    def __init__(self, *, splits=None, joint=None, verification=None, matches=None, revision=.99, **kwargs):
        super().__init__(**kwargs)
        self.splits = splits or {}
        self.joint = joint or {}
        self.verification = verification or {}
        self.matches = matches or {}
        self.revision = revision
        self.inventory_states = []

    def evaluate(self, context, noul=None, choice=None):
        if not context.startswith("{"):
            return super().evaluate(context, noul, choice)
        state = json.loads(context)
        self.inventory_states.append(state)
        if "focus_revision" in state:
            values = dict.fromkeys(noul, self.revision)
        elif "parent" in state:
            text = state["parent"]["candidate"]["text"]
            if "splittable" in noul:
                components = self.splits.get(text, ())
                values = {"splittable": .99 if components else .01}
                values.update({f"component{index}": .99 if item["candidate"]["text"] in components else .01
                               for index, item in enumerate(state["components"])})
                values.update(self.joint.get(text, {}))
            else:
                values = dict.fromkeys(DECOMPOSITION_QUESTIONS, .99)
                values.update(self.verification.get(text, {}))
        else:
            values = {}
            for index, item in enumerate(state["claims"]):
                row = self.matches.get((state["focus"]["text"], item["candidate"]["text"]), (.01, .01, .99))
                values.update({f"match{index}_{name}": value for name, value in zip(("full", "partial", "anchor"), row)})
        metadata = EvaluationMetadata("fixture", "fixture", .001, 1, "success", 20, 8, provider="fixture")
        self.records.append(metadata)
        return EvaluationResult(tuple(NoulResult(key, values[key]) for key in noul), (), metadata)


class InformationInventoryTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket, "create_connection", side_effect=AssertionError("Network disabled"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.source = "No Alfa, os backups são feitos a cada 24 horas e guardados por 30 dias."
        self.parent = candidate("v0:s0", self.source, quantities=("24 horas", "30 dias"))
        self.frequency = candidate("v0:s0", self.source, text="No Alfa, os backups são feitos a cada 24 horas.", quantities=("24 horas",))
        self.retention = candidate("v0:s0", self.source, text="No Alfa, os backups são guardados por 30 dias.", quantities=("30 dias",))

    def run_inventory(self, *, direct=(), qa=(), foci=(), recovery=(), components=(), evaluator=None, texts=None, **config):
        def generate(data, route):
            return {"direct": direct, "qa": qa, "discover_coverage_foci": foci,
                    "recovery": recovery, "decompose_candidate": components}.get(route, ())
        generator = ScriptedGenerator(callback=lambda data, route: list(generate(data, route)))
        settings = dict(concurrency=1, qa_enabled=False, max_calls=256, max_literal_repairs=0,
                        pair_batch_size=1, max_relation_adjudications=0,
                        direct_window_chars=4000, qa_window_chars=4000)
        settings.update(config)
        source = videos(texts or [self.source])
        original = copy.deepcopy(asdict(source[0]))
        report = InformationAnalyzer(generator, evaluator or InventoryEvaluator(),
                                     InformationAnalysisConfig(**settings)).analyze(capture_snapshot(source))
        self.assertEqual(original, asdict(source[0]))
        return report, generator

    def split_evaluator(self, **kwargs):
        return InventoryEvaluator(splits={self.source: (self.frequency["text"], self.retention["text"])}, **kwargs)

    def test_joint_structure_replaces_misclassified_accepted_parent_with_verified_parts(self):
        report, _ = self.run_inventory(direct=[self.parent, self.frequency, self.retention],
                                      evaluator=self.split_evaluator(), experimental_mode="direct")
        self.assertEqual(2, report.counts.unique_units)
        self.assertEqual(1, report.counts.decomposed_parent_candidates)
        parent = report.candidates[0]
        self.assertEqual(("accepted", "atomic", .99, "decomposed_parent"),
                         (parent.validation, parent.granularity, parent.granularity_probability, parent.inventory_role))
        self.assertEqual("verified", report.decompositions[0].state)
        self.assertEqual(("c1", "c2"), report.decompositions[0].component_candidate_ids)
        self.assertTrue(all("c0" not in unit.candidate_ids for unit in report.units))
        self.assertTrue(all(record.parent_candidate_ids == ("c0",) for record in report.candidates[1:]))
        self.assertFalse(report.counts.counts_provisional)
        self.assertEqual("contextual-propositions-v2", report.protocol_version)
        self.assertEqual(9, report.schema_version)
        self.assertEqual(3, report.counts.accepted_candidates)

    def test_joint_structure_retains_each_component_source_span(self):
        frequency = candidate("v0:s0", self.source, text=self.frequency["text"],
                              quote="os backups são feitos a cada 24 horas", quantities=("24 horas",))
        retention = candidate("v0:s0", self.source, text=self.retention["text"],
                              quote="guardados por 30 dias", quantities=("30 dias",))
        report, _ = self.run_inventory(direct=[self.parent, frequency, retention],
                                      evaluator=self.split_evaluator(), experimental_mode="direct")
        self.assertEqual(2, report.counts.unique_units)
        self.assertEqual(2, report.counts.occurrences)
        self.assertEqual([frequency["evidence"][0]["start_char"], retention["evidence"][0]["start_char"]],
                         [item.assertion_evidence[0].start_char for item in report.occurrences])
        self.assertTrue(all(item.source_identity == report.snapshot.current_segments[0].source_identity
                            for item in report.occurrences))

    def test_partial_decomposition_keeps_parent_and_parts_as_visible_provisional_alternatives(self):
        for children in ([self.frequency], [self.frequency, self.retention]):
            with self.subTest(children=len(children)):
                evaluator = self.split_evaluator(verification={self.source: {"components_cover_parent": .01}})
                report, _ = self.run_inventory(direct=[self.parent, *children], evaluator=evaluator, experimental_mode="direct")
                self.assertEqual(len(children) + 1, report.counts.unique_units)
                self.assertTrue(report.counts.counts_provisional)
                self.assertEqual(len(children) + 1, report.counts.provisional_granularity_units)
                self.assertEqual("partial", report.decompositions[0].state)
                self.assertTrue(all(unit.inventory_state == "provisional_granularity" for unit in report.units))
                self.assertEqual("unit_candidate", report.candidates[0].inventory_role)

    def test_missing_condition_or_conflicting_split_never_suppresses_parent(self):
        for check in ("parent_covers_components", "qualifiers_preserved", "components_independent", "source_occurrence"):
            with self.subTest(check=check):
                report, _ = self.run_inventory(direct=[self.parent, self.frequency, self.retention],
                    evaluator=self.split_evaluator(verification={self.source: {check: .01}}), experimental_mode="direct")
                self.assertEqual(3, report.counts.unique_units)
                self.assertEqual("conflicting", report.decompositions[0].state)
                self.assertEqual(0, report.counts.decomposed_parent_candidates)

    def test_later_equivalence_conflict_restores_parent_and_marks_pairs_unresolved(self):
        evaluator = self.split_evaluator(relation={frozenset((self.frequency["text"], self.retention["text"])): "equivalent"})
        report, _ = self.run_inventory(direct=[self.parent, self.frequency, self.retention], evaluator=evaluator,
                                      experimental_mode="direct")
        self.assertEqual(3, report.counts.unique_units)
        self.assertEqual("conflicting", report.decompositions[0].state)
        self.assertEqual("unit_candidate", report.candidates[0].inventory_role)
        self.assertEqual("uncertain", report.relations[0].equivalence_state)
        self.assertTrue(report.counts.counts_provisional)

    def seeded_inventory(self, proposals, *, equivalent=True):
        relations = {frozenset((left["text"], right["text"])): "equivalent"
                     for left in proposals for right in proposals} if equivalent else {}
        analyzer = InformationAnalyzer(ScriptedGenerator(), InventoryEvaluator(relation=relations),
            InformationAnalysisConfig(experimental_mode="direct", concurrency=1,
                max_granularity_checks=0, max_coverage_foci=0, pair_batch_size=1,
                max_relation_adjudications=0, max_literal_repairs=0))
        run = _AnalysisRun(analyzer, capture_snapshot(videos([self.source])))
        target = run.snapshot.current_segments[0]
        for proposal in proposals:
            run._validate(proposal, target, "direct", 0, (target.id,))
        run.evaluated_targets.add(target.id)
        return run

    @staticmethod
    def finish_seeded_inventory(run):
        # Exercise actual consolidation, final focus membership and reporting
        # from already validated synthetic candidates; no discovery is needed.
        with patch.object(run, "_run_targets"):
            return run.run()

    def test_unconfirmed_decomposition_preserves_exact_and_probabilistic_equivalence(self):
        paraphrase = candidate("v0:s0", self.source,
            text="No Alfa, o intervalo entre backups é de 24 horas.", quantities=("24 horas",))
        for peer in (self.frequency, paraphrase):
            for state, signals in (
                ("uncertain", {"splittable": .73, "component0": .84}),
                ("uncertain", {"splittable": .42, "component0": .83}),
                ("partial", {"parent_compound": .42, "parent_covers_components": .8,
                             "components_cover_parent": .5, "components_distinct": .23}),
            ):
                with self.subTest(exact=peer is self.frequency, state=state):
                    run = self.seeded_inventory([self.frequency, peer])
                    decomposition = run._record_decomposition(run.candidates[0], run.candidates[1:],
                                                              state, signals.items())
                    proposal = replace(run.candidates[0], id="focus")
                    run.coverage_foci.append(CoverageFocus("focus", "v0:s0", proposal.candidate, proposal,
                        state="covered", candidate_ids=("c0",),
                        matches=(CoverageMatch("c0", "covered", origin="fixture"),)))
                    report = self.finish_seeded_inventory(run)
                    relation = report.relations[0]
                    self.assertEqual("equivalent", relation.equivalence_state)
                    self.assertTrue(relation.merged)
                    self.assertNotEqual("conflicting_joint_granularity", relation.equivalence_origin)
                    self.assertEqual(1, report.counts.unique_units)
                    self.assertEqual("validated", report.units[0].inventory_state)
                    self.assertEqual(0, report.counts.provisional_granularity_units)
                    self.assertNotIn("granularity_unresolved", [item.kind for item in report.issues])
                    self.assertEqual(decomposition, report.decompositions[0])
                    self.assertTrue(all(item.inventory_role == "unit_candidate" for item in report.candidates))
                    focus = report.coverage_foci[0]
                    self.assertEqual(("covered", ("u0",)), (focus.state, focus.unit_ids))
                    self.assertTrue(set(focus.candidate_ids) <= set(report.units[0].candidate_ids))

    def test_collapsed_atomic_copies_keep_real_compound_conflict_without_losing_equivalence(self):
        paraphrase = candidate("v0:s0", self.source,
            text="No Alfa, o intervalo entre backups é de 24 horas.", quantities=("24 horas",))
        run = self.seeded_inventory([self.frequency, paraphrase, self.frequency])
        # Concrete native failure shape: a rejected partition of equivalent
        # copies has positive compound evidence, but no verified proper part.
        signals = {"parent_compound": .92, "parent_covers_components": .8,
                   "components_cover_parent": .12, "components_independent": .54,
                   "components_distinct": .23, "source_occurrence": .89, "qualifiers_preserved": .58}
        run._record_decomposition(run.candidates[0], run.candidates[1:], "partial", signals.items())
        report = self.finish_seeded_inventory(run)
        self.assertTrue(all(item.equivalence_state == "equivalent" and item.merged for item in report.relations))
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual("provisional_granularity", report.units[0].inventory_state)
        self.assertEqual(1, report.counts.provisional_granularity_units)
        self.assertIn("granularity_unresolved", [item.kind for item in report.issues])
        self.assertEqual(0, report.counts.decomposed_parent_candidates)

    def test_joint_component_identity_survives_filtering_and_later_verification(self):
        evaluator = InventoryEvaluator(
            joint={self.source: {"splittable": .99, "component0": .1, "component1": .99}},
            verification={self.source: dict.fromkeys(DECOMPOSITION_QUESTIONS, .5)},
            relation={frozenset((self.source, self.retention["text"])): "equivalent",
                      frozenset((self.source, self.frequency["text"])): "equivalent"})
        report, _ = self.run_inventory(direct=[self.parent, self.retention, self.frequency],
                                      evaluator=evaluator, experimental_mode="direct")
        decomposition = next(item for item in report.decompositions if item.parent_candidate_id == "c0")
        self.assertEqual(("c2",), decomposition.component_candidate_ids)
        checks = dict(decomposition.signals)
        self.assertEqual((.1, .99, .1, .99),
                         tuple(checks[key] for key in ("component0", "component1", "proper_part:c1", "proper_part:c2")))
        relations = {frozenset((item.left_candidate_id, item.right_candidate_id)): item for item in report.relations}
        self.assertEqual("equivalent", relations[frozenset(("c0", "c1"))].equivalence_state)
        self.assertEqual("conflicting_joint_granularity", relations[frozenset(("c0", "c2"))].equivalence_origin)
        self.assertEqual("unit_candidate", report.candidates[0].inventory_role)

    def test_legacy_component_indices_never_fabricate_filtered_candidate_identity(self):
        run = self.seeded_inventory([self.parent, self.retention, self.frequency])
        run._record_decomposition(run.candidates[0], [run.candidates[2]], "uncertain",
                                  (("splittable", .99), ("component0", .1), ("component1", .99)))
        report = self.finish_seeded_inventory(run)
        self.assertTrue(all(item.equivalence_state == "equivalent" for item in report.relations))
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual("provisional_granularity", report.units[0].inventory_state)
        self.assertFalse(any(key.startswith("proper_part:") for key, _ in report.decompositions[0].signals))

    def test_identified_part_requires_positive_compound_check_at_the_same_cutoff(self):
        for splittable in (.43, .85):
            with self.subTest(splittable=splittable):
                run = self.seeded_inventory([self.parent, self.frequency])
                run._record_decomposition(run.candidates[0], run.candidates[1:], "uncertain",
                    (("splittable", splittable), ("component0", .85), ("proper_part:c1", .85)))
                report = self.finish_seeded_inventory(run)
                self.assertEqual("equivalent" if splittable == .43 else "uncertain",
                                 report.relations[0].equivalence_state)
                self.assertTrue(all(item.inventory_state == "provisional_granularity" for item in report.units))

    def test_legacy_failure_without_components_is_not_a_collapsed_alternative(self):
        run = self.seeded_inventory([self.frequency])
        run._record_decomposition(run.candidates[0], (), "uncertain", reason="InvalidResponseError")
        report = self.finish_seeded_inventory(run)
        self.assertEqual("provisional_granularity", report.units[0].inventory_state)
        self.assertIn("granularity_unresolved", [item.kind for item in report.issues])
        self.assertEqual("unit_candidate", report.candidates[0].inventory_role)

    def test_verified_residual_preserves_parent_part_conflict_but_uncertain_gates_do_not(self):
        for source_support in (.99, .8):
            with self.subTest(source_support=source_support):
                run = self.seeded_inventory([self.parent, self.frequency])
                signals = dict.fromkeys(DECOMPOSITION_QUESTIONS, .99)
                signals.update(components_cover_parent=.01, parent_covers_components=source_support)
                run._record_decomposition(run.candidates[0], run.candidates[1:], "partial", signals.items())
                report = self.finish_seeded_inventory(run)
                relation = report.relations[0]
                self.assertEqual("uncertain" if source_support == .99 else "equivalent", relation.equivalence_state)
                self.assertEqual(2 if source_support == .99 else 1, report.counts.unique_units)
                self.assertTrue(all(item.inventory_state == "provisional_granularity" for item in report.units))
                self.assertEqual("unit_candidate", report.candidates[0].inventory_role)

    def test_verified_partition_also_conflicts_with_parent_equivalence_and_restores_parent(self):
        run = self.seeded_inventory([self.parent, self.frequency, self.retention])
        run._record_decomposition(run.candidates[0], run.candidates[1:], "verified",
                                  dict.fromkeys(DECOMPOSITION_QUESTIONS, .99).items())
        run.relations = [InformationRelation("c0", "c1", "equivalent", .99, .99)]
        self.assertTrue(run._reconcile_decomposition_relations())
        self.assertEqual("uncertain", run.relations[0].equivalence_state)
        self.assertEqual("conflicting", run.decompositions[0].state)
        self.assertEqual("unit_candidate", run.candidates[0].inventory_role)

    def test_equivalent_children_do_not_clear_an_unrepresented_compound_parent(self):
        run = self.seeded_inventory([self.parent, self.frequency, self.frequency])
        run.candidates[0] = replace(run.candidates[0], validation="needs_repair", granularity="compound")
        run._record_decomposition(run.candidates[0], run.candidates[1:], "partial")
        report = self.finish_seeded_inventory(run)
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(("c1", "c2"), report.units[0].candidate_ids)
        self.assertEqual("provisional_granularity", report.units[0].inventory_state)
        self.assertIn("granularity_unresolved", [item.kind for item in report.issues])
        self.assertEqual("needs_repair", report.candidates[0].validation)

    def run_decomposed_focus_refresh(self, *, fail_refresh=True, exact_child=False,
                                     changed_anchor=False, conflicting_pair=False):
        focus_text = self.frequency["text"]
        child = copy.deepcopy(self.frequency)
        if changed_anchor:
            child = candidate("v0:s0", self.source, text=focus_text,
                              quote="os backups são feitos a cada 24 horas", quantities=("24 horas",))
        elif not exact_child:
            child["text"] = "No Alfa, o intervalo entre backups é de 24 horas."

        class RefreshEvaluator(InventoryEvaluator):
            def evaluate(self, context, noul=None, choice=None):
                if fail_refresh and context.startswith("{"):
                    state = json.loads(context)
                    if (state.get("focus", {}).get("text") == focus_text
                            and any(item["id"] != "c0" for item in state.get("claims", []))):
                        raise InvalidResponseError("Fixture interrupted the post-decomposition focus refresh.")
                return super().evaluate(context, noul, choice)

        evaluator = RefreshEvaluator(
            splits={self.source: (child["text"], self.retention["text"])},
            matches={(focus_text, self.source): (.99, .01, .99),
                     (focus_text, child["text"]): (.99, .01, .99)},
            coverage={"v0:s0": [.99, .01]},
            relation={frozenset((child["text"], self.retention["text"])): "equivalent"}
            if conflicting_pair else None)
        report, _ = self.run_inventory(direct=[self.parent], foci=[self.frequency, self.retention],
                                      recovery=[child, self.retention], evaluator=evaluator)
        return report

    def test_failed_refresh_cannot_leave_covered_focus_bound_only_to_inactive_parent(self):
        report = self.run_decomposed_focus_refresh()
        first, repaired = report.coverage_foci
        self.assertEqual("verified", report.decompositions[0].state)
        self.assertEqual("decomposed_parent", report.candidates[0].inventory_role)
        self.assertEqual(("uncertain", (), ()), (first.state, first.candidate_ids, first.unit_ids))
        self.assertEqual((), first.matches)
        self.assertEqual(("c0", "covered"),
                         (first.inactive_matches[0].candidate_id, first.inactive_matches[0].state))
        self.assertEqual("covered", repaired.state)
        self.assertEqual(("c2",), repaired.candidate_ids)
        self.assertEqual(("u1",), repaired.unit_ids)
        self.assertEqual("resolved", repaired.history[0].outcome)
        self.assertEqual("covered", repaired.history[0].resulting_state)
        self.assertEqual(("c1", "c2"), repaired.history[0].candidate_ids)
        self.assertTrue(any(item.candidate_id == "c2" and item.state == "covered"
                            for item in repaired.history[0].matches))
        self.assertEqual({"covered": 1, "missing": 0, "partial": 0, "uncertain": 1},
                         json.loads(report.metadata_json)["source_focus_states"])
        self.assertEqual(2, report.counts.unique_units)
        self.assertEqual("partial", report.status)
        self.assertTrue(report.counts.counts_provisional)
        issue = next(item for item in report.issues if item.kind == "coverage_focus_membership_unresolved")
        self.assertEqual(("c0",), issue.candidate_ids)
        self.assertIn("coverage_focus_unresolved", [item.kind for item in report.issues])
        self.assertTrue(any(item.operation == "verify_focus_correspondence"
                            and item.status == "InvalidResponseError" for item in report.calls))
        active = {identifier for unit in report.units for identifier in unit.candidate_ids}
        self.assertTrue(all(set(focus.candidate_ids) <= active for focus in report.coverage_foci))
        self.assertTrue(all(focus.unit_ids for focus in report.coverage_foci if focus.state == "covered"))
        exported = report.to_dict()
        self.assertEqual("uncertain", exported["coverage_foci"][0]["state"])
        self.assertEqual([], exported["coverage_foci"][0]["candidate_ids"])
        self.assertEqual("c0", exported["coverage_foci"][0]["inactive_matches"][0]["candidate_id"])
        self.assertEqual("resolved", exported["coverage_foci"][1]["history"][0]["outcome"])

    def test_successful_refresh_keeps_verified_child_binding_and_archives_parent_match(self):
        report = self.run_decomposed_focus_refresh(fail_refresh=False)
        first = report.coverage_foci[0]
        self.assertEqual("completed", report.status)
        self.assertEqual(("covered", ("c1",), ("u0",)),
                         (first.state, first.candidate_ids, first.unit_ids))
        self.assertEqual(("c0", "covered"),
                         (first.inactive_matches[0].candidate_id, first.inactive_matches[0].state))
        self.assertTrue(all(item.candidate_id != "c0" for item in first.matches))
        self.assertEqual(2, json.loads(report.metadata_json)["source_focus_states"]["covered"])
        self.assertFalse(report.counts.counts_provisional)

    def test_finalization_requires_full_evidence_identity_for_free_child_binding(self):
        for changed_anchor in (False, True):
            with self.subTest(changed_anchor=changed_anchor):
                report = self.run_decomposed_focus_refresh(exact_child=True, changed_anchor=changed_anchor)
                focus = report.coverage_foci[0]
                if changed_anchor:
                    self.assertEqual(("uncertain", (), ()), (focus.state, focus.candidate_ids, focus.unit_ids))
                else:
                    self.assertEqual(("covered", ("c1",), ("u0",)),
                                     (focus.state, focus.candidate_ids, focus.unit_ids))
                    match = next(item for item in focus.matches if item.candidate_id == "c1")
                    self.assertEqual("exact_validated_proposition", match.origin)
                self.assertEqual("decomposed_parent", report.candidates[0].inventory_role)
                self.assertTrue(any(item.candidate_id == "c0" for item in focus.inactive_matches))

    def test_pair_conflict_restores_prior_verified_parent_match_from_provenance(self):
        for fail_refresh in (False, True):
            with self.subTest(fail_refresh=fail_refresh):
                report = self.run_decomposed_focus_refresh(fail_refresh=fail_refresh, conflicting_pair=True)
                focus = report.coverage_foci[0]
                self.assertEqual("conflicting", report.decompositions[0].state)
                self.assertEqual("unit_candidate", report.candidates[0].inventory_role)
                self.assertEqual("covered", focus.state)
                self.assertIn("c0", focus.candidate_ids)
                self.assertTrue(any(item.candidate_id == "c0" and item.state == "covered" for item in focus.matches))
                self.assertFalse(any(item.candidate_id == "c0" for item in focus.inactive_matches))
                parent_unit = next(item.id for item in report.units if "c0" in item.candidate_ids)
                self.assertIn(parent_unit, focus.unit_ids)
                self.assertEqual(2, json.loads(report.metadata_json)["source_focus_states"]["covered"])
                self.assertTrue(report.counts.counts_provisional)
                self.assertEqual("resolved", report.coverage_foci[1].history[0].outcome)

    def test_failed_consolidation_finalizes_coverage_against_the_empty_active_inventory(self):
        with patch.object(_AnalysisRun, "_consolidate", side_effect=InvalidResponseError("Fixture consolidation failure")):
            report, _ = self.run_inventory(direct=[self.frequency], foci=[self.frequency])
        focus = report.coverage_foci[0]
        self.assertEqual("failed", report.status)
        self.assertEqual((), report.units)
        self.assertEqual(("uncertain", (), ()), (focus.state, focus.candidate_ids, focus.unit_ids))
        self.assertEqual("c0", focus.inactive_matches[0].candidate_id)
        self.assertEqual(0, json.loads(report.metadata_json)["source_focus_states"]["covered"])
        self.assertTrue(report.counts.counts_provisional)

    def test_specificity_and_a_conditional_rule_do_not_automatically_split(self):
        source = "Se a conexão cair, o backup não é realizado."
        full = candidate("v0:s0", source, unit_type="conditional", conditions=("Se a conexão cair",), negated=True)
        report, _ = self.run_inventory(direct=[full], texts=[source], experimental_mode="direct")
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual((), report.decompositions)
        self.assertEqual(("Se a conexão cair",), report.units[0].qualifiers.conditions)
        shorter = candidate("v0:s0", self.source, text="No Alfa, são feitos backups.")
        report, _ = self.run_inventory(direct=[self.frequency, shorter], experimental_mode="direct")
        self.assertEqual(2, report.counts.unique_units)
        self.assertEqual((), report.decompositions)

    def test_independent_source_focus_finds_content_hidden_inside_shared_quote_and_repairs_it(self):
        evaluator = InventoryEvaluator(coverage={"v0:s0": [.99, .01]})
        report, generator = self.run_inventory(direct=[self.frequency], foci=[self.frequency, self.retention],
                                              recovery=[self.retention], evaluator=evaluator)
        self.assertEqual(2, report.counts.unique_units)
        self.assertEqual(2, report.counts.candidates)
        self.assertEqual(2, len(report.coverage_foci))
        focus = next(item for item in report.coverage_foci if item.proposition.text == self.retention["text"])
        self.assertEqual(("covered", "resolved"), (focus.state, focus.history[0].outcome))
        self.assertEqual(("c1",), focus.candidate_ids)
        self.assertEqual(("u1",), focus.unit_ids)
        self.assertEqual(focus.id, report.candidates[1].recovery_focus_id)
        self.assertEqual("missing", focus.history[0].previous_state)
        self.assertEqual(["gap_detected", "no_gap_signaled"], [item.state for item in report.coverage])
        self.assertFalse(json.loads(report.metadata_json)["completeness_proven"])
        requests = [json.loads(item.prompt.split("\nInput:\n", 1)[1]) for item in generator.requests]
        discovery = next(item for item in requests if item.get("analysis_task") == "discover_coverage_foci")
        self.assertNotIn("accepted", discovery)
        self.assertNotIn("repair_candidates", discovery)
        repair = next(item for item in requests if "coverage_focus" in item)
        self.assertEqual(focus.id, repair["coverage_focus"]["id"])

    def test_reformulations_and_exact_duplicate_rows_do_not_close_a_different_gap(self):
        reformulation = candidate("v0:s0", self.source, text="Os backups do Alfa ocorrem a cada 24 horas.", quantities=("24 horas",))
        for recovery in ([self.frequency], [reformulation]):
            with self.subTest(recovery=recovery[0]["text"]):
                evaluator = InventoryEvaluator(coverage={"v0:s0": [.99, .01]})
                report, _ = self.run_inventory(direct=[self.frequency], foci=[self.retention],
                                              recovery=recovery, evaluator=evaluator)
                focus = report.coverage_foci[0]
                self.assertEqual("missing", focus.state)
                self.assertEqual("no_progress", focus.history[0].outcome)
                self.assertEqual((), focus.candidate_ids)
                self.assertEqual((), focus.unit_ids)
                self.assertEqual(1, len(report.coverage))
                self.assertTrue(report.counts.counts_provisional)
                self.assertIn("coverage_no_progress", [item.kind for item in report.issues])

    def test_partial_and_conflicting_correspondences_never_claim_covered(self):
        for signals, expected in (((.01, .99, .99), "partial"), ((.99, .99, .99), "uncertain"),
                                  ((.99, .01, .01), "missing")):
            with self.subTest(signals=signals):
                evaluator = InventoryEvaluator(matches={(self.retention["text"], self.frequency["text"]): signals})
                report, _ = self.run_inventory(direct=[self.frequency], foci=[self.retention], evaluator=evaluator,
                                              max_coverage_rounds=0)
                self.assertEqual(expected, report.coverage_foci[0].state)
                self.assertEqual((), report.coverage_foci[0].unit_ids)
                self.assertTrue(report.counts.counts_provisional)

    def test_open_audit_can_stay_pending_even_when_all_known_foci_are_covered(self):
        report, _ = self.run_inventory(direct=[self.frequency], foci=[self.frequency],
                                      evaluator=InventoryEvaluator(coverage={"v0:s0": [.99]}))
        self.assertEqual("covered", report.coverage_foci[0].state)
        self.assertIn("coverage_unidentified_gap", [item.kind for item in report.issues])
        self.assertTrue(report.counts.counts_provisional)

    def test_focus_defect_repair_preserves_gap_identity_and_original_hypothesis(self):
        source = "Se a conexão cair, o backup não é realizado."
        defective = candidate("v0:s0", source, text="O backup não é realizado.", negated=True)
        repaired = candidate("v0:s0", source, unit_type="conditional", negated=True,
                             conditions=("Se a conexão cair",))
        for revision in (.99, .01):
            with self.subTest(revision=revision):
                evaluator = InventoryEvaluator(validation={defective["text"]: .01},
                    coverage={"v0:s0": [.99, .01]}, revision=revision)
                report, generator = self.run_inventory(texts=[source], foci=[defective], recovery=[repaired],
                                                      evaluator=evaluator)
                focus = report.coverage_foci[0]
                request = next(json.loads(item.prompt.split("\nInput:\n", 1)[1]) for item in generator.requests
                               if '"coverage_focus"' in item.prompt)
                self.assertEqual(focus.id, request["coverage_focus"]["id"])
                if revision == .99:
                    self.assertEqual("covered", focus.state)
                    self.assertEqual(source, focus.proposition.text)
                    self.assertEqual(defective["text"], focus.proposal_history[0].candidate.text)
                    self.assertEqual("rejected", focus.proposal_history[0].validation)
                    self.assertEqual("resolved", focus.history[0].outcome)
                    self.assertEqual(focus.matches, focus.history[0].matches)
                else:
                    self.assertEqual("uncertain", focus.state)
                    self.assertEqual((), focus.proposal_history)
                    self.assertEqual("no_progress", focus.history[0].outcome)

    def test_focus_ids_and_source_identity_are_stable_and_occurrences_do_not_cross_repetitions(self):
        reports = [self.run_inventory(direct=[self.frequency], foci=[self.frequency])[0] for _ in range(2)]
        self.assertEqual(reports[0].coverage_foci[0].id, reports[1].coverage_foci[0].id)
        self.assertEqual(reports[0].snapshot, reports[1].snapshot)
        source = "A cópia é diária. A cópia é diária."
        one = candidate("v0:s0", source, quote="A cópia é diária.", offset=0)
        two = candidate("v0:s0", source, quote="A cópia é diária.", offset=18)
        report, _ = self.run_inventory(texts=[source], direct=[one], foci=[two], max_coverage_rounds=0)
        self.assertEqual("missing", report.coverage_foci[0].state)
        self.assertEqual(1, report.counts.occurrences)
        self.assertEqual(18, report.coverage_foci[0].proposition.evidence[0].start_char)

    def test_all_four_modes_share_validation_granularity_and_frozen_input(self):
        snapshots = []
        qa = copy.deepcopy(self.frequency)
        qa.update(question="Qual é a periodicidade?", answer=qa["text"])
        for mode, routes in (("direct", {"direct"}), ("qa", {"qa"}),
                             ("hybrid", {"direct", "qa"}), ("hybrid_coverage", {"direct", "qa"})):
            with self.subTest(mode=mode):
                report, generator = self.run_inventory(direct=[self.frequency], qa=[qa], foci=[self.frequency],
                                                      experimental_mode=mode)
                snapshots.append(report.snapshot)
                self.assertEqual(routes, {item.route for item in report.candidates})
                self.assertEqual(mode, json.loads(report.metadata_json)["experimental_mode"])
                self.assertEqual(1, report.counts.unique_units)
                self.assertEqual(mode == "hybrid_coverage", bool(report.coverage))
                self.assertEqual(mode == "hybrid_coverage", bool(report.coverage_foci))
        self.assertTrue(all(item == snapshots[0] for item in snapshots))

    def test_joint_and_focus_limits_are_visible_under_the_global_call_cap(self):
        report, _ = self.run_inventory(direct=[self.parent, self.frequency, self.retention],
                                      foci=[self.frequency, self.retention], evaluator=self.split_evaluator(),
                                      max_calls=8, max_granularity_checks=1)
        self.assertLessEqual(json.loads(report.metadata_json)["logical_calls"], 8)
        self.assertTrue(report.counts.counts_provisional)
        self.assertTrue(any(item.kind in {"granularity_unresolved", "focus_discovery_budget", "call_budget_exhausted"}
                            for item in report.issues))
        self.assertFalse(json.loads(report.metadata_json)["completeness_proven"])

    def test_focus_proposals_survive_validation_budget_exhaustion_without_becoming_units(self):
        report, _ = self.run_inventory(foci=[self.frequency, self.retention], max_calls=5, max_coverage_rounds=0)
        self.assertEqual(2, len(report.coverage_foci))
        self.assertEqual("not_evaluated", report.coverage_foci[1].proposal_record.validation)
        self.assertEqual("uncertain", report.coverage_foci[1].state)
        self.assertEqual(0, report.counts.candidates)
        self.assertEqual(0, report.counts.unique_units)
        self.assertEqual(1, len(report.coverage))
        self.assertFalse(report.counts.valid_zero)
        self.assertEqual(5, json.loads(report.metadata_json)["logical_calls"])

    def test_mode_and_limit_configuration_is_explicit_and_bounded_on_both_commands(self):
        from open_video_summary.__main__ import build_parser
        for command in ("analyze-information", "summarize"):
            args = build_parser().parse_args([command, "--information-mode", "qa", "--no-information-qa",
                "--information-granularity-checks", "3", "--information-coverage-foci", "9", "--information-gap-repairs", "1"])
            self.assertEqual(("qa", 3, 9, 1), (args.information_mode, args.information_granularity_checks,
                                               args.information_coverage_foci, args.information_gap_repairs))
        self.assertEqual(("qa",), InformationAnalysisConfig(experimental_mode="qa", qa_enabled=False).discovery_routes)
        for settings in ({"experimental_mode": "unknown"}, {"max_granularity_checks": -1},
                         {"max_coverage_foci": 257}, {"max_gap_repairs": True}):
            with self.subTest(settings=settings), self.assertRaises(ConfigurationError):
                InformationAnalysisConfig(**settings)

    def test_terminal_focus_generation_failure_stops_new_provider_requests(self):
        def generate(data, route):
            if route == "discover_coverage_foci":
                raise AuthenticationError("fixture authentication failure")
            return [self.frequency]
        generator = ScriptedGenerator(callback=generate)
        report = InformationAnalyzer(generator, InventoryEvaluator(), InformationAnalysisConfig(
            qa_enabled=False, max_literal_repairs=0)).analyze(capture_snapshot(videos([self.source])))
        self.assertEqual("AuthenticationError", json.loads(report.metadata_json)["permanent_provider_failure"])
        self.assertEqual("AuthenticationError", report.calls[-1].status)
        self.assertEqual(2, len(generator.requests))
        self.assertTrue(report.counts.counts_provisional)

    def test_cancellation_prevents_focus_and_structural_sends(self):
        generator = ScriptedGenerator()
        run = _AnalysisRun(InformationAnalyzer(generator, InventoryEvaluator()), capture_snapshot(videos([self.source])))
        run.budget.cancel()
        with self.assertRaises(RequestCancelledError):
            run._discover_foci(run.snapshot.current_segments[0])
        self.assertEqual([], generator.requests)
        self.assertEqual(0, run.call_count)

    def test_cli_surfaces_alternative_counts_and_source_focus_counts_separately(self):
        from open_video_summary.__main__ import _print_information_report
        report, _ = self.run_inventory(direct=[self.parent, self.frequency], foci=[self.retention],
            evaluator=self.split_evaluator(verification={self.source: {"components_cover_parent": .01}}),
            max_coverage_rounds=0)
        stream = io.StringIO()
        with redirect_stdout(stream):
            _print_information_report(report, None)
        self.assertIn("Granularity alternatives:", stream.getvalue())
        self.assertIn("Source coverage foci: 1 tracked; 1 pending", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
