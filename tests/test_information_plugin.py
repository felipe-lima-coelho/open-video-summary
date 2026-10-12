"""Offline stage-plugin contracts; synthetic decisions are not scientific evaluation."""

import copy
import json
import os
import tempfile
import unittest
import uuid
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import patch

from pandas import DataFrame

from open_video_summary.contracts import EvaluationMetadata, EvaluationResult, NoulResult
from open_video_summary.core.selection_criteria.base import SelectionCriteria
from open_video_summary.core.selection_criteria.quality import QualityPick
from open_video_summary.core.summarizers.base import Summarizer
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig, configured_information_analyzer
from open_video_summary.core.summarizers.information_contracts import (
    CandidateRecord, Evidence, InformationCandidate, InformationCounts,
    InformationOccurrence, InformationReport, InformationUnit, Qualifiers,
    ScopeCount, canonical_json, capture_snapshot,
)
from open_video_summary.core.summarizers.information_io import append_information_stage, save_information_report
from open_video_summary.core.summarizers.information_plugin import InformationAnalysisPlugin, analyze_discarded_questions
from open_video_summary.core.summarizers.information_questions import with_information_questions
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.errors import ConfigurationError
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.utils.config import PROJECT_DIR
from tests.test_information_analysis import ScriptedGenerator, SyntheticEvaluator, candidate, videos


def fixture_report(snapshot, *, qa=False, repeat_questions=False, context_id=None, status="completed"):
    records, units, occurrences, by_text = [], [], [], {}
    for segment in snapshot.current_segments:
        identifier = f"c{len(records)}"
        evidence = (Evidence(segment.id, segment.source_identity, segment.content,
                    0, len(segment.content), "assertion", segment.start, segment.end),)
        if context_id is not None and segment.id != context_id:
            context = next(item for item in snapshot.current_segments if item.id == context_id)
            evidence += (Evidence(context.id, context.source_identity, context.content, 0,
                len(context.content), "context", context.start, context.end),)
        text = segment.content
        qualifiers = Qualifiers(None, False, None, (), ())
        question = f"What is communicated at {segment.id}?" if qa else None
        claim = InformationCandidate(text, "assertion", qualifiers, evidence, question,
                                     text if qa else None, ())
        signals = (("qa_anchor", .99), ("qa_consistency", .99)) if qa else ()
        records.append(CandidateRecord(identifier, segment.id, "qa" if qa else "direct", 0,
                       claim, "{}", "accepted", signals=signals, annotation_state="verified"))
        if text not in by_text:
            unit_id = f"u{len(units)}"
            by_text[text] = len(units)
            units.append(InformationUnit(unit_id, text, "assertion", qualifiers, (identifier,), "verified"))
        else:
            index = by_text[text]
            units[index] = replace(units[index], candidate_ids=units[index].candidate_ids + (identifier,))
            unit_id = units[index].id
        occurrences.append(InformationOccurrence(f"o{len(occurrences)}", unit_id, segment.id,
            segment.source_identity, (identifier,), evidence[:1], evidence[1:], ("qa" if qa else "direct",)))
        if repeat_questions:
            records.append(replace(records[-1], id=f"{identifier}:variant", candidate=replace(claim,
                question=f"Which complete claim appears at {segment.id}?", answer=text),
                signals=(("qa_anchor", .99), ("qa_consistency", .99))))
            index = by_text[text]
            units[index] = replace(units[index], candidate_ids=units[index].candidate_ids + (records[-1].id,))
    counts = InformationCounts(len(records), len(records), len(occurrences), len(units),
        tuple(ScopeCount(segment.id, 1, 1) for segment in snapshot.current_segments),
        tuple(ScopeCount(video.id, len(video.segments), len({item.content for item in video.segments}))
              for video in snapshot.source), not units and status == "completed")
    return InformationReport(11, "fixture", uuid.uuid4().hex, status, snapshot, tuple(records),
                             tuple(units), tuple(occurrences), (), (), counts, (), (), "{}")


class StageEvaluator:
    def __init__(self, signals=None, error=None):
        self.signals, self.error = signals or {}, error
        self.contexts, self.records = [], []
        self.preflight_calls = 0
        self.config = SimpleNamespace(provider="fixture", model="jev-fixture")

    def preflight(self):
        self.preflight_calls += 1

    def evaluate(self, context, noul=None, choice=None):
        self.contexts.append(context)
        if self.error:
            raise self.error
        metadata = EvaluationMetadata("jev-fixture", "jev-fixture", 0, 1, "success",
                                      provider="fixture")
        self.records.append(metadata)
        return EvaluationResult(tuple(NoulResult(key, self.signals.get(key, .99))
                                      for key in noul), (), metadata)


class FixtureAnalyzer:
    def __init__(self, *, evaluator=None, qa=False, repeat_questions=False, context_id=None,
                 status="completed", **settings):
        self.generator, self.evaluator = SimpleNamespace(), evaluator or StageEvaluator()
        self.config = InformationAnalysisConfig(**settings)
        self.progress, self.progress_interval_seconds = None, 10
        self.qa, self.repeat_questions = qa, repeat_questions
        self.context_id, self.status = context_id, status
        self.snapshots = []

    def analyze(self, snapshot):
        self.snapshots.append(snapshot)
        return fixture_report(snapshot, qa=self.qa, repeat_questions=self.repeat_questions,
                              context_id=self.context_id, status=self.status)

    def analyze_discarded(self, report, stage):
        return analyze_discarded_questions(self, report, stage)


class Remove(SelectionCriteria):
    def __init__(self, indices=(0,)):
        super().__init__()
        self.indices = indices

    def evaluate(self, handler):
        for index in self.indices:
            self.discard(handler, handler.source[0].segments[index])
        return handler


class Keep(SelectionCriteria):
    def evaluate(self, handler):
        for segment in handler.source[0].segments:
            if segment not in handler.discard:
                self.output(handler, segment)
        return handler


def handler_for(source):
    handler = SummarySegmentHandler()
    handler.set_source_videos(source)
    return handler


class InformationQuestionTests(unittest.TestCase):
    def test_real_initial_qa_and_direct_only_questions_keep_evidence_and_certainty(self):
        text = "Talvez o prazo não seja fixo, segundo a palestrante."
        for qa in (False, True):
            with self.subTest(qa=qa):
                generator = ScriptedGenerator({("v0:s0", "qa" if qa else "direct"):
                    [candidate("v0:s0", text, qa=qa, negated=True, modality="Talvez", attribution="palestrante")]})
                analyzer = InformationAnalyzer(generator, SyntheticEvaluator(), InformationAnalysisConfig(
                    qa_enabled=qa, max_granularity_checks=0, max_coverage_foci=0, max_literal_repairs=0))
                report = analyzer.analyze(capture_snapshot(videos([text])))
                self.assertEqual(1, len(report.questions))
                query = report.questions[0]
                self.assertEqual(text, query.answer)
                self.assertEqual(report.units[0].id, query.unit_id)
                self.assertEqual((report.occurrences[0].id,), query.occurrence_ids)
                self.assertEqual(text, query.assertion_evidence[0].quote)
                self.assertEqual("validated_qa" if qa else "templated_proposition", query.origin)
                self.assertEqual(1, report.counts.unique_units)
                if not qa:
                    self.assertIn(text, query.question)
                    self.assertFalse(any(request.output.kind == "information_qa" for request in generator.requests))

    def test_rejected_qa_never_becomes_a_natural_question(self):
        report = fixture_report(capture_snapshot(videos(["A source statement."])), qa=True)
        record = replace(report.candidates[0], signals=(("qa_anchor", .6), ("qa_consistency", .99)))
        report = with_information_questions(replace(report, candidates=(record,)))
        self.assertEqual("templated_proposition", report.questions[0].origin)


class InformationPluginTests(unittest.TestCase):
    def test_pipeline_baseline_precedes_selection_stage_only_for_removal_and_deduplicates(self):
        source = videos(["Repeated information.", "Repeated information.", "Another fact."])
        analyzer = FixtureAnalyzer(qa=True, repeat_questions=True)
        summarizer = Summarizer([Remove((0,)), Keep()])
        initial = []
        output = summarizer.summarize(source, save_output=False, information_analyzer=analyzer,
            information_report_observer=lambda report, path: initial.append(report))
        self.assertEqual("before_introduction", initial[0].snapshot.stage_id)
        self.assertFalse(initial[0].stage_reports)
        report = summarizer.last_information_report
        self.assertEqual(initial[0].units, report.units)
        self.assertEqual(initial[0].questions, report.questions)
        self.assertEqual(1, len(report.stage_reports))
        stage = report.stage_reports[0]
        self.assertEqual(("v0:s0",), stage.discarded_segment_ids)
        self.assertEqual(("v0:s0",), stage.explicit_discard_action_ids)
        self.assertEqual(1, stage.loss.removed_unique_units)
        self.assertEqual(0, stage.loss.potentially_lost_unique_units)
        self.assertEqual(("u0", "u1"), stage.loss.retained_unit_ids)
        self.assertGreater(len(stage.matches), 1)
        self.assertIsNone(stage.loss.definitely_lost_unique_units)
        self.assertEqual(source[0].segments[1:], output.segments)
        self.assertEqual(1, len(analyzer.snapshots))

    def test_direct_future_inherited_and_constructor_without_super_receive_plugin(self):
        class Future(SelectionCriteria):
            def __init__(self, label):
                self.label = label

            def evaluate(self, current, mode="remove"):
                if mode == "remove":
                    self.discard(current, current.source[0].segments[0])
                return current

        class Inherited(Future):
            pass

        for cls in (Future, Inherited):
            analyzer = FixtureAnalyzer()
            criterion = cls("fixture", information_analyzer=InformationAnalysisPlugin(analyzer))
            handler = handler_for(videos(["A source statement."]))
            self.assertIs(handler, criterion.evaluate(current=handler, mode="remove"))
            self.assertEqual(1, len(criterion.last_information_report.stage_reports))
            self.assertEqual(1, len(analyzer.snapshots))
            self.assertEqual(1, len(analyzer.evaluator.contexts))

    def test_declared_constructor_analyzer_and_super_evaluate_do_not_double_observe(self):
        class Child(Remove):
            def __init__(self, label, information_analyzer):
                self.received = information_analyzer
                super().__init__()

            def evaluate(self, handler):
                return super().evaluate(handler)

        analyzer = FixtureAnalyzer()
        criterion = Child("fixture", information_analyzer=analyzer)
        self.assertIs(analyzer, criterion.received)
        criterion.evaluate(handler_for(videos(["A source statement."])))
        self.assertEqual(1, len(criterion.last_information_report.stage_reports))
        self.assertEqual(1, len(analyzer.evaluator.contexts))

    def test_duck_modules_default_attachment_optouts_and_environment_exclusions(self):
        class Duck:
            name = "Duck"

            def evaluate(self, handler):
                handler.discard_segment(handler.source[0].segments[0], self.name)
                return handler

        for exclusion in ("Duck", f"{Duck.__module__}.{Duck.__qualname__}"):
            analyzer = FixtureAnalyzer()
            with patch.dict(os.environ, {"OVS_INFORMATION_EXCLUDE_MODULES": exclusion}):
                summarizer = Summarizer([Duck()])
                summarizer.summarize(videos(["A source statement."]), save_output=False,
                                     information_analyzer=analyzer)
            self.assertFalse(summarizer.last_information_report.stage_reports)
            self.assertFalse(analyzer.evaluator.contexts)
            self.assertEqual(0, analyzer.evaluator.preflight_calls)
        for binding in (None, False):
            analyzer = FixtureAnalyzer()
            summarizer = Summarizer([Remove(information_analyzer=binding)])
            summarizer.summarize(videos(["A source statement."]), save_output=False,
                                 information_analyzer=analyzer)
            self.assertFalse(summarizer.last_information_report.stage_reports)
        analyzer = FixtureAnalyzer()
        summarizer = Summarizer([Duck()], information_analyzer=analyzer)
        summarizer.summarize(videos(["A source statement."]), save_output=False)
        self.assertEqual(1, len(summarizer.last_information_report.stage_reports))
        summarizer.summarize(videos(["Another statement."]), save_output=False, information_analyzer=None)
        self.assertIsNone(summarizer.last_information_report)

    def test_per_module_analyzer_override_uses_shared_baseline(self):
        primary, alternate = FixtureAnalyzer(), FixtureAnalyzer(evaluator=StageEvaluator({"support": .5}))
        module = Remove(information_analyzer=alternate)
        summarizer = Summarizer([module])
        summarizer.summarize(videos(["A source statement."]), save_output=False, information_analyzer=primary)
        self.assertEqual(1, len(primary.snapshots))
        self.assertFalse(primary.evaluator.contexts)
        self.assertFalse(alternate.snapshots)
        self.assertEqual("uncertain", summarizer.last_information_report.stage_reports[0].matches[0].state)

    def test_empty_no_discard_temporary_discard_and_picks_make_no_stage_requests(self):
        class Moves(SelectionCriteria):
            def evaluate(self, handler):
                if handler.source[0].segments:
                    segment = handler.source[0].segments[0]
                    self.discard(handler, segment)
                    self.output(handler, segment)
                    self.discard(handler, segment)
                    self.pick(handler, {segment})
                return handler

        for texts in ([], ["A source statement."]):
            analyzer = FixtureAnalyzer()
            summarizer = Summarizer([Moves()])
            summarizer.summarize(videos(texts), save_output=False, information_analyzer=analyzer)
            self.assertFalse(summarizer.last_information_report.stage_reports)
            self.assertFalse(analyzer.evaluator.contexts)

    def test_equal_valued_source_positions_preserved_without_duplicate_unit_count(self):
        segment = VideoSegment("A source statement.", 0, 5, 0)
        source = [Video("Source", "data/raw/fixture.mp4", segments=[segment, copy.deepcopy(segment)])]
        summarizer = Summarizer([Remove()])
        summarizer.summarize(source, save_output=False, information_analyzer=FixtureAnalyzer())
        stage = summarizer.last_information_report.stage_reports[0]
        self.assertEqual(("v0:s0", "v0:s1"), stage.discarded_segment_ids)
        self.assertEqual(("v0:s0",), stage.explicit_discard_action_ids)
        self.assertEqual(1, stage.loss.potentially_lost_unique_units)

    def test_uncertain_negation_and_quantity_gates_never_become_supported_answers(self):
        text = "Talvez o prazo não seja 30 dias."
        for gates, expected in (({"support": .5}, "uncertain"), ({"negation": .01}, "not_answerable"),
                                ({"quantities": .01}, "not_answerable"), ({"modality": .01}, "not_answerable")):
            analyzer = FixtureAnalyzer(evaluator=StageEvaluator(gates), status="partial")
            summarizer = Summarizer([Remove()])
            summarizer.summarize(videos([text]), save_output=False, information_analyzer=analyzer)
            stage = summarizer.last_information_report.stage_reports[0]
            self.assertEqual(expected, stage.matches[0].state)
            self.assertIsNone(stage.matches[0].answer)
            self.assertEqual(gates, {key: dict(stage.matches[0].signals)[key] for key in gates})
            self.assertTrue(stage.loss.counts_provisional)
            self.assertEqual(0, stage.loss.potentially_lost_unique_units)
            self.assertIn(text, analyzer.evaluator.contexts[0])

    def test_missing_context_is_not_imported_from_retained_provenance(self):
        analyzer = FixtureAnalyzer(context_id="v0:s0")
        summarizer = Summarizer([Remove((1,))])
        summarizer.summarize(videos(["The reference is a proposed deadline.", "It may be extended."]),
                             save_output=False, information_analyzer=analyzer)
        stage = summarizer.last_information_report.stage_reports[0]
        self.assertEqual("context_missing", stage.matches[0].state)
        self.assertFalse(analyzer.evaluator.contexts)
        self.assertEqual(["v0:s1"], json.loads(stage.metadata_json)["semantic_context_ids"])

    def test_bounded_failed_stage_does_not_change_selection_or_mask_criterion_failure(self):
        source = videos(["First statement.", "Second statement."])
        for evaluator, cap in ((StageEvaluator(), 1), (StageEvaluator(error=RuntimeError("fixture")), 128),
                               (StageEvaluator({"support": float("nan")}), 128)):
            analyzer = FixtureAnalyzer(evaluator=evaluator, stage_max_calls=cap)
            summarizer = Summarizer([Remove((0, 1))])
            self.assertFalse(summarizer.summarize(source, save_output=False, information_analyzer=analyzer).segments)
            stage = summarizer.last_information_report.stage_reports[0]
            self.assertNotEqual("completed", stage.status)
            self.assertTrue(stage.issues)
            self.assertLessEqual(json.loads(stage.metadata_json)["logical_calls"], cap)
        class Broken(Remove):
            def evaluate(self, handler):
                super().evaluate(handler)
                raise ValueError("criterion failure")
        with self.assertRaisesRegex(ValueError, "criterion failure"):
            Summarizer([Broken()]).summarize(source, save_output=False, information_analyzer=FixtureAnalyzer())

    def test_sequential_runs_and_module_instances_have_fresh_question_and_stage_state(self):
        analyzer = FixtureAnalyzer()
        first, other = Remove(information_analyzer=analyzer), Remove(information_analyzer=analyzer)
        for module, text in ((first, "First statement."), (other, "Second statement."), (first, "Third statement.")):
            module.evaluate(handler_for(videos([text])))
            report = module.last_information_report
            self.assertEqual(text, report.questions[0].answer)
            self.assertEqual(1, len(report.stage_reports))
        pipeline = Summarizer([first])
        pipeline.summarize(videos(["Fourth statement."]), save_output=False, information_analyzer=analyzer)
        prior = pipeline.last_information_report
        pipeline.summarize(videos(["Fifth statement."]), save_output=False, information_analyzer=analyzer)
        self.assertNotEqual(prior.run_id, pipeline.last_information_report.run_id)
        self.assertEqual(1, len(pipeline.last_information_report.stage_reports))

    def test_zero_stage_budget_never_preflights_or_sends_and_keeps_pending_work(self):
        analyzer = FixtureAnalyzer(stage_max_calls=0)
        criterion = Remove(information_analyzer=analyzer)
        criterion.evaluate(handler_for(videos(["A source statement."])))
        stage = criterion.last_information_report.stage_reports[0]
        self.assertEqual("partial", stage.status)
        self.assertEqual(0, analyzer.evaluator.preflight_calls)
        self.assertFalse(analyzer.evaluator.contexts)
        self.assertEqual(1, json.loads(stage.metadata_json)["pending_comparisons"])

    def test_failing_analyzer_extension_is_reported_without_selection_changes(self):
        analyzer = FixtureAnalyzer()
        analyzer.analyze_discarded = lambda report, stage: (_ for _ in ()).throw(RuntimeError("observer failure"))
        summarizer = Summarizer([Remove(), Keep()])
        output = summarizer.summarize(videos(["Discarded statement.", "Kept statement."]),
                                     save_output=False, information_analyzer=analyzer)
        self.assertEqual(["Kept statement."], [segment.content for segment in output.segments])
        stage = summarizer.last_information_report.stage_reports[0]
        self.assertEqual("failed", stage.status)
        self.assertEqual("stage_analysis_failed", stage.issues[0].kind)

    def test_excluded_direct_criterion_exception_is_never_retried(self):
        class Broken(Remove):
            calls = 0

            def evaluate(self, handler):
                self.calls += 1
                raise ValueError("criterion failure")
        analyzer = FixtureAnalyzer()
        criterion = Broken(information_analyzer=analyzer)
        with (patch.dict(os.environ, {"OVS_INFORMATION_EXCLUDE_MODULES": "Broken"}),
              self.assertRaisesRegex(ValueError, "criterion failure")):
            criterion.evaluate(handler_for(videos(["A source statement."])))
        self.assertEqual(1, criterion.calls)
        self.assertFalse(analyzer.snapshots)
        self.assertFalse(analyzer.evaluator.contexts)


class QualityInformationPluginTests(unittest.TestCase):
    def quality_run(self, *, enabled=True, top_n=1, prior_include=False, excluded=False):
        source = videos(["A winner in the first group.", "A winner in the second group.", "A unique loser."])
        first, second, loser = source[0].segments
        class Groups(SelectionCriteria):
            def evaluate(self, handler):
                self.pick(handler, {first, loser})
                self.pick(handler, {first, second, loser})
                if prior_include:
                    self.include(handler, loser)
                return handler
        quality = QualityPick("Groups", top_n_segments=top_n)
        calls = []
        def score(features):
            calls.append(features)
            values = {first: 2, second: 3, loser: 1}
            return DataFrame({0: [values[item] for item in features]}, index=list(features))
        summarizer = Summarizer([Groups(), quality])
        with (patch.object(quality, "extract_segments_visual_features", side_effect=lambda cluster: {item: [] for item in cluster}),
              patch.object(quality, "get_bovw_dataframe", side_effect=score),
              patch.dict(os.environ, {"OVS_INFORMATION_EXCLUDE_MODULES": "QualityPick" if excluded else ""})):
            output = summarizer.summarize(source, save_output=False, collect_audit=False,
                information_analyzer=FixtureAnalyzer() if enabled else None)
        return source, output, summarizer, quality

    def test_quality_nonwinners_overlap_audit_disabled_and_unchanged_output(self):
        source, output, summarizer, quality = self.quality_run()
        _, baseline, disabled, _ = self.quality_run(enabled=False)
        self.assertEqual(output.segments, baseline.segments)
        self.assertEqual(summarizer.last_handler.discard, disabled.last_handler.discard)
        self.assertEqual(summarizer.last_handler.agent_logs, disabled.last_handler.agent_logs)
        stage = summarizer.last_information_report.stage_reports[0]
        self.assertEqual("QualityPick", stage.module_name)
        self.assertEqual(("v0:s2",), stage.discarded_segment_ids)
        self.assertFalse(stage.explicit_discard_action_ids)
        self.assertEqual(["v0:s2"], json.loads(stage.metadata_json)["outcome_removed_segment_ids"])
        self.assertEqual(1, stage.loss.potentially_lost_unique_units)
        self.assertEqual(set(source[0].segments[:2]), set(output.segments))
        self.assertIsNone(quality.last_audit)

    def test_quality_guaranteed_include_exclusions_and_zero_all_winners(self):
        for settings in ({"prior_include": True}, {"top_n": 3}, {"excluded": True}):
            _, _, summarizer, _ = self.quality_run(**settings)
            self.assertFalse(summarizer.last_information_report.stage_reports)
        _, output, summarizer, _ = self.quality_run(top_n=0)
        self.assertFalse(output.segments)
        self.assertEqual(("v0:s0", "v0:s1", "v0:s2"),
                         summarizer.last_information_report.stage_reports[0].discarded_segment_ids)

    def test_excluded_quality_outcomes_affect_later_availability_only_within_that_run(self):
        source = videos(["First winner.", "Second winner.", "Quality loser."])
        first, second, loser = source[0].segments
        class Groups(SelectionCriteria):
            def evaluate(self, handler):
                self.pick(handler, {first, second, loser})
                return handler
        quality = QualityPick("Groups", top_n_segments=2, information_analyzer=False)
        summarizer = Summarizer([Groups(), quality, Remove((1,))])
        table = DataFrame({0: [3, 2, 1]}, index=[first, second, loser])
        with (patch.object(quality, "extract_segments_visual_features", return_value={first: [], second: [], loser: []}),
              patch.object(quality, "get_bovw_dataframe", return_value=table)):
            summarizer.summarize(source, save_output=False, information_analyzer=FixtureAnalyzer())
        stages = summarizer.last_information_report.stage_reports
        self.assertEqual(["Remove"], [stage.module_name for stage in stages])
        self.assertEqual(("v0:s0",), stages[0].available_segment_ids)
        self.assertEqual(("v0:s1",), stages[0].discarded_segment_ids)
        self.assertEqual(("v0:s1", "v0:s2"), stages[0].current_discarded_segment_ids)
        self.assertNotIn("u2", stages[0].loss.retained_unit_ids)
        # A separate call owns no QualityPick outcome state from the prior run.
        fresh = Remove(information_analyzer=FixtureAnalyzer())
        fresh.evaluate(handler_for(source))
        self.assertEqual(("v0:s1", "v0:s2"), fresh.last_information_report.stage_reports[0].available_segment_ids)


class InformationPluginPersistenceTests(unittest.TestCase):
    def setUp(self):
        (PROJECT_DIR / "outputs").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=PROJECT_DIR / "outputs", prefix="plugin-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = PROJECT_DIR / "outputs" / os.path.basename(temporary.name)

    def test_initial_json_receives_stages_and_an_existing_destination_is_preserved(self):
        path = self.directory / "inventory.json"
        seen = []
        summarizer = Summarizer([Remove((0,)), Keep()])
        summarizer.summarize(videos(["Discarded statement.", "Kept statement."]), save_output=False,
            information_analyzer=FixtureAnalyzer(), information_output_path=path,
            information_report_observer=lambda report, saved: seen.append(json.loads(saved.read_text(encoding="utf-8"))))
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(seen[0]["questions"])
        self.assertFalse(seen[0]["stage_reports"])
        self.assertEqual(seen[0]["units"], payload["units"])
        self.assertEqual(1, len(payload["stage_reports"]))
        self.assertEqual(payload, summarizer.last_information_report.to_dict())
        before = path.read_bytes()
        with self.assertRaises(ConfigurationError):
            save_information_report(summarizer.last_information_report, path)
        self.assertEqual(before, path.read_bytes())

    def test_externally_changed_initial_report_is_not_clobbered_and_failure_is_visible(self):
        path = self.directory / "inventory.json"
        summarizer = Summarizer([Remove()])
        def edit(report, saved):
            saved.write_text('{"external":"edit"}', encoding="utf-8")
        summarizer.summarize(videos(["Discarded statement."]), save_output=False,
            information_analyzer=FixtureAnalyzer(), information_output_path=path,
            information_report_observer=edit)
        self.assertEqual({"external": "edit"}, json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual("ConfigurationError", summarizer.last_information_error)
        self.assertEqual("stage_report_write_failed", summarizer.last_information_report.stage_reports[-1].issues[-1].kind)
        if summarizer.last_information_path:
            self.addCleanup(summarizer.last_information_path.unlink, missing_ok=True)

    def test_matching_content_replacement_is_not_treated_as_an_owned_report(self):
        path = self.directory / "inventory.json"
        summarizer = Summarizer([Remove()])
        def replace_saved(report, saved):
            replacement = self.directory / "replacement.json"
            replacement.write_bytes(saved.read_bytes())
            os.replace(replacement, saved)
        summarizer.summarize(videos(["Discarded statement."]), save_output=False,
            information_analyzer=FixtureAnalyzer(), information_output_path=path,
            information_report_observer=replace_saved)
        self.assertFalse(json.loads(path.read_text(encoding="utf-8"))["stage_reports"])
        self.assertEqual("ConfigurationError", summarizer.last_information_error)
        if summarizer.last_information_path:
            self.addCleanup(summarizer.last_information_path.unlink, missing_ok=True)

    def test_external_formatting_edit_is_preserved_even_with_same_json_content(self):
        path = self.directory / "inventory.json"
        summarizer = Summarizer([Remove()])
        expected = []
        def reformat(report, saved):
            text = json.dumps(json.loads(saved.read_text(encoding="utf-8")), ensure_ascii=False)
            expected.append(text)
            saved.write_text(text, encoding="utf-8")
        summarizer.summarize(videos(["Discarded statement."]), save_output=False,
            information_analyzer=FixtureAnalyzer(), information_output_path=path,
            information_report_observer=reformat)
        self.assertEqual(expected[0], path.read_text(encoding="utf-8"))
        self.assertEqual("ConfigurationError", summarizer.last_information_error)
        if summarizer.last_information_path:
            self.addCleanup(summarizer.last_information_path.unlink, missing_ok=True)

    def test_owned_append_refuses_changed_initial_inventory_and_reserved_artifacts(self):
        summarizer = Summarizer([Remove()])
        summarizer.summarize(videos(["Discarded statement."]), save_output=False, information_analyzer=FixtureAnalyzer())
        full = summarizer.last_information_report
        baseline = replace(full, stage_reports=())
        path = self.directory / "inventory.json"
        save_information_report(baseline, path)
        with self.assertRaises(ConfigurationError):
            append_information_stage(baseline, replace(full, units=()), path)
        with self.assertRaises(ConfigurationError):
            append_information_stage(baseline, full, path, reserved=(path,))
        self.assertFalse(any(self.directory.glob("*.tmp")))

    def test_two_removing_modules_append_once_each_with_shared_initial_inventory(self):
        class Later(Remove):
            pass
        path = self.directory / "two_stages.json"
        analyzer = FixtureAnalyzer()
        summarizer = Summarizer([Remove((0,)), Later((1,))])
        summarizer.summarize(videos(["First statement.", "Second statement."]), save_output=False,
            information_analyzer=analyzer, information_output_path=path)
        exported = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(["Remove", "Later"], [stage["module_name"] for stage in exported["stage_reports"]])
        self.assertEqual(["1:Remove", "2:Later"], [stage["stage_id"] for stage in exported["stage_reports"]])
        self.assertEqual(1, len(analyzer.snapshots))
        self.assertEqual(2, exported["counts"]["unique_units"])

    def test_env_config_exclusions_and_stage_budget_are_loaded_only_when_enabled(self):
        analyzer = configured_information_analyzer(environ={
            "OVS_INFORMATION_EXCLUDE_MODULES": " Introduction,QualityPick ",
            "OVS_INFORMATION_STAGE_MAX_CALLS": "5",
        }, env_file=self.directory / "absent.env")
        self.assertEqual(("Introduction", "QualityPick"), analyzer.config.excluded_modules)
        self.assertEqual(5, analyzer.config.stage_max_calls)


if __name__ == "__main__":
    unittest.main()
