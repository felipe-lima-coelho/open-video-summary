"""Offline protocol tests on synthetic Portuguese transcripts, not a benchmark."""

import copy
import importlib.util
import json
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_video_summary.adapters.llm import (
    DomainResponseInterpreter,
    OllamaAdapter,
    OpenAIAdapter,
)
from open_video_summary.adapters.typesafe import (
    ChoiceResult,
    EvaluationMetadata,
    EvaluationResult,
    NoulResult,
    TransportResponse,
    TypeSafeConfig,
    TypeSafeEvaluator,
)
from open_video_summary.contracts import GenerationResult, OutputSpec, ServiceMetadata
from open_video_summary.core.summarizers.base import Summarizer
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer
from open_video_summary.core.summarizers.information_config import (
    InformationAnalysisConfig,
)
from open_video_summary.core.summarizers.information_contracts import capture_snapshot
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.errors import InvalidResponseError, ServiceTimeoutError
from open_video_summary.utils.providers import LLMConfig


def videos(texts):
    return [
        Video(
            "Fonte",
            "data/raw/absent.mp4",
            topics=["Alfa"],
            segments=[
                VideoSegment(
                    text, index * 10, (index + 1) * 10, index, "Alfa", "Backup"
                )
                for index, text in enumerate(texts)
            ],
        )
    ]


def candidate(
    segment_id,
    source,
    *,
    text=None,
    quote=None,
    offset=None,
    role="assertion",
    qa=False,
    quantities=(),
    conditions=(),
    negated=False,
    attribution=None,
    modality=None,
    unit_type="assertion",
    contexts=(),
):
    quote = source if quote is None else quote
    offset = source.index(quote) if offset is None else offset
    return {
        "text": text or quote,
        "unit_type": unit_type,
        "qualifiers": {
            "attribution": attribution,
            "negated": negated,
            "modality": modality,
            "quantities": list(quantities),
            "conditions": list(conditions),
        },
        "evidence": [
            {
                "segment_id": segment_id,
                "quote": quote,
                "start_char": offset,
                "end_char": offset + len(quote),
                "role": role,
            }
        ]
        + list(contexts),
        "question": "Qual é a regra comunicada?" if qa else None,
        "answer": text or quote if qa else None,
        "unresolved_references": [],
    }


class ScriptedGenerator:
    def __init__(self, items=None, callback=None):
        self.items = items or {}
        self.callback = callback
        self.records = []
        self.requests = []
        self.preflight = Mock()
        self.config = LLMConfig(provider="ollama", model="fixture")

    def generate(self, request):
        self.requests.append(request)
        data = json.loads(request.prompt.split("\nInput:\n", 1)[1])
        route = (
            "qa"
            if request.output.kind == "information_qa"
            else "recovery" if "coverage_audit" in data else "direct"
        )
        if self.callback:
            value = self.callback(data, route)
        else:
            value = self.items.get((data["target"]["id"], route), [])
        if isinstance(value, Exception):
            raise value
        envelope = {"candidates": copy.deepcopy(value), "issues": []}
        text = json.dumps(envelope, ensure_ascii=False)
        interpreted = DomainResponseInterpreter().interpret(text, request.output)
        metadata = ServiceMetadata(
            "ollama",
            "fixture",
            returned_model="fixture-1",
            input_tokens=17,
            output_tokens=11,
        )
        self.records.append(metadata)
        return GenerationResult(text, interpreted, metadata)


class SyntheticEvaluator:
    """Explicit deterministic decisions, restricted to synthetic test fixtures."""

    def __init__(
        self,
        *,
        coverage=None,
        granularity=None,
        validation=None,
        relation=None,
        relation_probability=0.99,
        relation_confidence=0.99,
    ):
        self.coverage = coverage or {}
        self.granularity = granularity or {}
        self.validation = validation or {}
        self.relation = relation or {}
        self.relation_probability, self.relation_confidence = (
            relation_probability,
            relation_confidence,
        )
        self.coverage_calls = {}
        self.calls = []
        self.records = []
        self.preflight = Mock()
        self.config = TypeSafeConfig(api_key="fixture-secret")

    def evaluate(self, context, noul=None, choice=None):
        state = json.loads(context)
        noul, choice = noul or {}, choice or {}
        self.calls.append((state, dict(noul), dict(choice)))
        probability = 0.99
        if "original_target" in state:
            identifier = state["original_target"]["id"]
            index = self.coverage_calls.get(identifier, 0)
            self.coverage_calls[identifier] = index + 1
            sequence = self.coverage.get(identifier, [0.01])
            probability = sequence[min(index, len(sequence) - 1)]
        elif "candidate" in state:
            probability = self.validation.get(state["candidate"]["text"], 0.99)
        choices = []
        for identifier, (_, options) in choice.items():
            if identifier == "granularity":
                selected = self.granularity.get(state["candidate"]["text"], "atomic")
                mass, confidence = 0.99, 0.99
            else:
                pair = frozenset((state["left"]["text"], state["right"]["text"]))
                selected = self.relation.get(
                    pair,
                    (
                        "equivalent"
                        if state["left"]["text"] == state["right"]["text"]
                        else "complementary"
                    ),
                )
                mass, confidence = self.relation_probability, self.relation_confidence
            probabilities = tuple(
                (
                    option,
                    mass if option == selected else (1 - mass) / (len(options) - 1),
                )
                for option in options
            )
            choices.append(
                ChoiceResult(identifier, selected, probabilities, confidence)
            )
        metadata = EvaluationMetadata(
            "jev-1.13.0",
            "jev-1.13.0",
            0.001,
            1,
            "success",
            20,
            8,
            provider=self.config.provider,
        )
        self.records.append(metadata)
        return EvaluationResult(
            tuple(NoulResult(key, probability) for key in noul),
            tuple(choices),
            metadata,
        )


def analyze(source, generator=None, evaluator=None, **settings):
    return InformationAnalyzer(
        generator or ScriptedGenerator(),
        evaluator or SyntheticEvaluator(),
        InformationAnalysisConfig(**settings),
    ).analyze(capture_snapshot(source))


class SnapshotAndPipelineTests(unittest.TestCase):
    def test_positional_identity_immutability_hashes_and_stage_orders(self):
        segment = VideoSegment("Mesmo conteúdo.", 1, 2, 0)
        source = [
            Video(
                "Fonte", "data/raw/f.mp4", ["Tema"], [segment, copy.deepcopy(segment)]
            )
        ]
        snapshot = capture_snapshot(
            source, selection_stage_order=("Introduction", "QualityPick")
        )
        self.assertEqual(("v0:s0", "v0:s1"), snapshot.source_order)
        self.assertNotEqual(
            snapshot.current_segments[0].source_identity,
            snapshot.current_segments[1].source_identity,
        )
        self.assertEqual(snapshot.source_order, snapshot.current_order)
        self.assertEqual("before_introduction", snapshot.stage_id)
        self.assertEqual(
            ("Introduction", "QualityPick"), snapshot.selection_stage_order
        )
        with self.assertRaises(FrozenInstanceError):
            snapshot.current_segments[0].content = "Mutated"
        source[0].segments[0].content = "Changed after capture"
        self.assertEqual("Mesmo conteúdo.", snapshot.current_segments[0].content)
        reordered = capture_snapshot(source, current_order=("v0:s1", "v0:s0"))
        self.assertNotEqual(
            snapshot.current_input_fingerprint, reordered.current_input_fingerprint
        )

    def test_removed_source_is_provenance_only_and_not_available_context(self):
        source = videos(["O prazo é 30 dias.", "Esse prazo não é fixo."])
        snapshot = capture_snapshot(
            source, stage_id="hypothetical_current", current_order=("v0:s1",)
        )
        generator = ScriptedGenerator()
        report = InformationAnalyzer(generator, SyntheticEvaluator()).analyze(snapshot)
        self.assertEqual(2, len(report.snapshot.source_segments))
        data = json.loads(generator.requests[0].prompt.split("\nInput:\n", 1)[1])
        self.assertEqual([], data["context"])
        with self.assertRaises(RuntimeError):
            capture_snapshot(
                source, current_order=("v0:s1",), allowed_context_ids=("v0:s0", "v0:s1")
            )

    def test_before_introduction_once_with_complete_source_and_no_actions(self):
        events = []
        source = videos(["Uma regra.", "Outra regra."])
        original = asdict(source[0])

        class Introduction:
            name = "Introduction"

            def evaluate(self, handler):
                events.append("Introduction")
                self.assertions = (len(handler.source[0].segments), len(handler.output))
                handler.add_output_segment(handler.source[0].segments[1], self.name)
                handler.discard_segment(handler.source[0].segments[0], self.name)
                return handler

        criterion = Introduction()
        summarizer = Summarizer([criterion])

        class Observer:
            def analyze(self, snapshot):
                events.append("analysis")
                self.snapshot = snapshot
                return analyze(source, qa_enabled=False)

        observer = Observer()
        with_observer = summarizer.summarize(
            source,
            save_output=False,
            collect_audit=False,
            information_analyzer=observer,
        )
        with_logs = copy.deepcopy(
            asdict(summarizer.last_handler)["_SummarySegmentHandler__agent_log"]
        )
        without = summarizer.summarize(source, save_output=False, collect_audit=False)
        self.assertEqual(["analysis", "Introduction", "Introduction"], events)
        self.assertEqual((2, 0), criterion.assertions)
        self.assertEqual(original, asdict(source[0]))
        self.assertEqual(with_observer.segments, without.segments)
        self.assertEqual(
            with_logs,
            asdict(summarizer.last_handler)["_SummarySegmentHandler__agent_log"],
        )
        self.assertIsNone(summarizer.last_information_report)

    def test_same_hsm_singleton_twice_has_fresh_report_and_no_model_weights(self):
        from open_video_summary.core import summarizers

        old = summarizers.__dict__.pop("HSMVideoSumm", None)
        try:
            with patch("open_video_summary.classifiers.text.CrossEncoder"):
                hsm = summarizers.HSMVideoSumm
            effects = []
            for criterion in hsm.selection_criteria:

                def evaluate(handler, name=criterion.name):
                    effects.append(name)
                    if name == "Introduction":
                        self.assertEqual([], handler.output)
                        handler.add_output_segment(handler.source[0].segments[0], name)
                    return handler

                criterion.evaluate = Mock(side_effect=evaluate)
            analyzer = InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator())
            hsm.summarize(
                videos(["Olá."]), save_output=False, information_analyzer=analyzer
            )
            first = hsm.last_information_report
            hsm.summarize(
                videos(["Outro vídeo."]),
                save_output=False,
                information_analyzer=analyzer,
            )
            second = hsm.last_information_report
            self.assertNotEqual(first.run_id, second.run_id)
            self.assertNotEqual(
                first.snapshot.source_fingerprint, second.snapshot.source_fingerprint
            )
            self.assertEqual(2, len(analyzer.generator.requests) // 2)
            self.assertEqual("before_introduction", second.snapshot.stage_id)
            self.assertEqual(
                [criterion.name for criterion in hsm.selection_criteria] * 2, effects
            )
        finally:
            summarizers.__dict__.pop("HSMVideoSumm", None)
            if old is not None:
                summarizers.HSMVideoSumm = old

    def test_observer_exception_is_visible_and_selection_proceeds(self):
        source = videos(["Regra."])
        criterion = SimpleNamespace(
            name="Introduction", evaluate=lambda handler: handler
        )
        summarizer = Summarizer([criterion])
        with self.assertWarns(RuntimeWarning):
            summarizer.summarize(
                source,
                save_output=False,
                information_analyzer=SimpleNamespace(
                    analyze=Mock(side_effect=ServiceTimeoutError("No service"))
                ),
            )
        self.assertEqual("failed", summarizer.last_information_report.status)
        self.assertFalse(summarizer.last_information_report.counts.valid_zero)


class InformationProtocolTests(unittest.TestCase):
    def test_empty_and_noninformative_are_valid_zero_but_missing_jev_is_failure(self):
        empty = analyze([])
        filler = analyze(videos(["Bom dia, pessoal."]))
        self.assertEqual("completed", empty.status)
        self.assertEqual("completed", filler.status)
        self.assertTrue(empty.counts.valid_zero)
        self.assertTrue(filler.counts.valid_zero)
        generator = ScriptedGenerator()
        missing = InformationAnalyzer(
            generator, TypeSafeEvaluator(TypeSafeConfig())
        ).analyze(capture_snapshot(videos(["Regra."])))
        self.assertEqual("failed", missing.status)
        self.assertFalse(missing.counts.valid_zero)
        self.assertEqual([], generator.requests)

    def test_two_routes_duplicate_one_unit_one_occurrence_repeats_preserved(self):
        text = "O backup ocorre a cada 24 horas."
        source = videos([text, text])
        source[0].segments[1] = copy.deepcopy(source[0].segments[0])
        items = {
            (f"v0:s{index}", route): [
                candidate(
                    f"v0:s{index}", text, qa=route == "qa", quantities=("24 horas",)
                )
            ]
            for index in range(2)
            for route in ("direct", "qa")
        }
        generator = ScriptedGenerator(items)
        report = analyze(source, generator)
        self.assertEqual("completed", report.status)
        self.assertEqual(4, report.counts.accepted_candidates)
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(2, report.counts.occurrences)
        self.assertFalse(report.counts.occurrences_provisional)
        self.assertTrue(
            all(not count.occurrences_provisional for count in report.counts.by_segment)
        )
        self.assertEqual(
            [1, 1], [count.occurrences for count in report.counts.by_segment]
        )
        self.assertEqual(("direct", "qa"), report.occurrences[0].routes)
        qa_data = json.loads(generator.requests[1].prompt.split("\nInput:\n", 1)[1])
        self.assertNotIn("accepted", qa_data)
        self.assertNotIn("candidates", qa_data)

    def test_distinct_spans_disambiguate_repeats_within_same_segment(self):
        text = "O prazo é fixo. O prazo é fixo."
        phrase = "O prazo é fixo."
        raw = [
            candidate("v0:s0", text, quote=phrase, offset=offset)
            for offset in (0, len(phrase) + 1)
        ]
        report = analyze(
            videos([text]),
            ScriptedGenerator({("v0:s0", "direct"): raw}),
            qa_enabled=False,
        )
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(2, report.counts.occurrences)
        self.assertEqual("completed", report.status)
        self.assertFalse(report.counts.occurrences_provisional)
        self.assertEqual(
            [0, len(phrase) + 1],
            [item.assertion_evidence[0].start_char for item in report.occurrences],
        )

    def test_broad_route_evidence_cannot_bridge_two_precise_repeat_spans(self):
        phrase = "O prazo é fixo."
        text = phrase + " " + phrase
        broad = candidate("v0:s0", text, text=phrase)
        precise = [
            candidate("v0:s0", text, text=phrase, quote=phrase, offset=offset, qa=True)
            for offset in (0, len(phrase) + 1)
        ]
        report = analyze(
            videos([text]),
            ScriptedGenerator({("v0:s0", "direct"): [broad], ("v0:s0", "qa"): precise}),
        )
        self.assertEqual(3, report.counts.occurrences)
        self.assertEqual(1, report.counts.unique_units)
        self.assertTrue(report.counts.occurrences_provisional)
        self.assertEqual("partial", report.status)
        self.assertIn(
            "occurrence_alignment_uncertain", [issue.kind for issue in report.issues]
        )

    def test_intermediate_qa_span_cannot_expand_occurrence_identity(self):
        text = "Backup is daily. To make the schedule completely clear, the backup is performed daily."
        unit_text = "Backup is daily."
        direct = [
            candidate(
                "v0:s0", text, text=unit_text, quote=text[start:end], offset=start
            )
            for start, end in ((0, 16), (17, 86))
        ]
        qa = candidate(
            "v0:s0", text, text=unit_text, quote=text[0:37], offset=0, qa=True
        )
        report = analyze(
            videos([text]),
            ScriptedGenerator({("v0:s0", "direct"): direct, ("v0:s0", "qa"): [qa]}),
        )
        self.assertEqual(3, report.counts.accepted_candidates)
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(3, report.counts.occurrences)
        self.assertEqual(3, report.counts.by_segment[0].occurrences)
        self.assertEqual(3, report.counts.by_video[0].occurrences)
        self.assertTrue(report.counts.occurrences_provisional)
        self.assertTrue(report.counts.by_segment[0].occurrences_provisional)
        self.assertTrue(report.counts.by_video[0].occurrences_provisional)
        self.assertEqual("partial", report.status)
        self.assertIn(
            "occurrence_alignment_uncertain", [issue.kind for issue in report.issues]
        )
        containing = {
            identifier: occurrence.id
            for occurrence in report.occurrences
            for identifier in occurrence.candidate_ids
        }
        self.assertNotEqual(containing["c0"], containing["c1"])
        self.assertNotEqual(containing["c0"], containing["c2"])
        self.assertNotEqual(containing["c1"], containing["c2"])

    def test_occurrence_members_require_identical_anchors_across_span_chains(self):
        text = "Backup is daily. To make the schedule completely clear, the backup is performed daily."
        for spans in (
            ((0, 16), (0, 37), (17, 86)),
            ((70, 86), (49, 86), (0, 69)),
            ((0, 16), (10, 47), (37, 86)),
        ):
            with self.subTest(spans=spans):
                raw = [
                    candidate(
                        "v0:s0",
                        text,
                        text="Backup is daily.",
                        quote=text[start:end],
                        offset=start,
                    )
                    for start, end in spans
                ]
                raw.append(copy.deepcopy(raw[0]))
                report = analyze(
                    videos([text]),
                    ScriptedGenerator({("v0:s0", "direct"): raw}),
                    qa_enabled=False,
                )
                self.assertEqual(3, report.counts.occurrences)
                self.assertEqual(
                    4,
                    sum(
                        len(occurrence.candidate_ids)
                        for occurrence in report.occurrences
                    ),
                )
                self.assertEqual("partial", report.status)
                self.assertTrue(report.counts.occurrences_provisional)
                by_id = {
                    record.id: record.candidate.evidence[0]
                    for record in report.candidates
                }
                for occurrence in report.occurrences:
                    for index, left_id in enumerate(occurrence.candidate_ids):
                        for right_id in occurrence.candidate_ids[index + 1 :]:
                            left, right = by_id[left_id], by_id[right_id]
                            self.assertEqual(left, right)

    def test_overlapping_repeat_citations_are_not_conclusive_occurrence_identity(self):
        text = "Backup is daily. I repeat: backup is daily."
        raw = [
            candidate(
                "v0:s0",
                text,
                text="Backup is daily.",
                quote=text[start:end],
                offset=start,
            )
            for start, end in ((0, 26), (17, 43))
        ]
        report = analyze(
            videos([text]),
            ScriptedGenerator({("v0:s0", "direct"): raw}),
            qa_enabled=False,
        )
        self.assertEqual(2, report.counts.accepted_candidates)
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(2, report.counts.occurrences)
        self.assertEqual("partial", report.status)
        self.assertTrue(report.counts.occurrences_provisional)
        self.assertTrue(report.counts.by_segment[0].occurrences_provisional)
        self.assertTrue(report.counts.by_video[0].occurrences_provisional)
        self.assertEqual(
            ["unresolved_overlap", "unresolved_overlap"],
            [item.alignment_state for item in report.occurrences],
        )
        self.assertIn(
            "occurrence_alignment_uncertain", [issue.kind for issue in report.issues]
        )

    def test_contained_and_partially_overlapping_assertion_anchors_remain_provisional(
        self,
    ):
        text = "Backup is daily. This is the schedule."
        for spans in (((0, 16), (0, len(text))), ((0, 20), (10, len(text)))):
            with self.subTest(spans=spans):
                raw = [
                    candidate(
                        "v0:s0",
                        text,
                        text="Backup is daily.",
                        quote=text[start:end],
                        offset=start,
                    )
                    for start, end in spans
                ]
                report = analyze(
                    videos([text]),
                    ScriptedGenerator({("v0:s0", "direct"): raw}),
                    qa_enabled=False,
                )
                self.assertEqual(1, report.counts.unique_units)
                self.assertEqual(2, report.counts.occurrences)
                self.assertEqual("partial", report.status)
                self.assertTrue(report.counts.occurrences_provisional)
                self.assertTrue(
                    all(
                        item.alignment_state == "unresolved_overlap"
                        for item in report.occurrences
                    )
                )

    def test_exact_assertion_anchors_deduplicate_routes_despite_different_contexts(
        self,
    ):
        text = "Backup is daily. More context."
        other = "The policy was confirmed."
        assertion = "Backup is daily."
        direct = candidate(
            "v0:s0",
            text,
            quote=assertion,
            contexts=(
                {
                    "segment_id": "v0:s0",
                    "quote": "More context.",
                    "start_char": 17,
                    "end_char": len(text),
                    "role": "context",
                },
            ),
        )
        qa = candidate(
            "v0:s0",
            text,
            quote=assertion,
            qa=True,
            contexts=(
                {
                    "segment_id": "v0:s1",
                    "quote": other,
                    "start_char": 0,
                    "end_char": len(other),
                    "role": "context",
                },
            ),
        )
        # Reordered evidence and a duplicate assertion do not change its set.
        qa["evidence"].append(copy.deepcopy(qa["evidence"][0]))
        qa["evidence"].reverse()
        report = analyze(
            videos([text, other]),
            ScriptedGenerator({("v0:s0", "direct"): [direct], ("v0:s0", "qa"): [qa]}),
        )
        self.assertEqual("completed", report.status)
        self.assertEqual(2, report.counts.accepted_candidates)
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(1, report.counts.occurrences)
        self.assertFalse(report.counts.occurrences_provisional)
        self.assertEqual(1, len(report.occurrences[0].assertion_evidence))
        self.assertEqual(2, len(report.occurrences[0].context_evidence))
        self.assertEqual(("direct", "qa"), report.occurrences[0].routes)
        self.assertEqual("source_anchor_group", report.occurrences[0].alignment_state)
        self.assertEqual([], list(report.issues))

    def test_identical_text_in_different_context_still_receives_pair_evaluation(self):
        source = videos(["O limite do Alfa é fixo.", "O limite do Beta é fixo."])
        items = {
            (f"v0:s{index}", "direct"): [
                candidate(f"v0:s{index}", segment.content, text="O limite é fixo.")
            ]
            for index, segment in enumerate(source[0].segments)
        }
        evaluator = SyntheticEvaluator(
            relation={frozenset(("O limite é fixo.",)): "complementary"}
        )
        report = analyze(source, ScriptedGenerator(items), evaluator, qa_enabled=False)
        self.assertEqual(2, report.counts.unique_units)
        self.assertTrue(any("left" in state for state, _, _ in evaluator.calls))

    def test_hallucinated_quote_rejected_even_with_high_semantic_scores(self):
        text = "O prazo é de 30 dias."
        raw = candidate("v0:s0", text)
        raw["evidence"][0]["quote"] = "O prazo é de 40 dias."
        report = analyze(
            videos([text]),
            ScriptedGenerator({("v0:s0", "direct"): [raw]}),
            qa_enabled=False,
        )
        self.assertEqual("literal_rejected", report.candidates[0].validation)
        self.assertEqual(0, report.counts.unique_units)
        self.assertFalse(report.counts.valid_zero)
        self.assertIn(
            "literal_evidence_rejected", [issue.kind for issue in report.issues]
        )

    def test_context_evidence_not_occurrence_and_assertion_target_consistency(self):
        source = videos(
            [
                "No Alfa, o prazo é 30 dias.",
                "Esse prazo não se aplica às cópias manuais.",
            ]
        )
        context = {
            "segment_id": "v0:s0",
            "quote": source[0].segments[0].content,
            "start_char": 0,
            "end_char": len(source[0].segments[0].content),
            "role": "context",
        }
        raw = candidate(
            "v0:s1",
            source[0].segments[1].content,
            text="O prazo de 30 dias do Alfa não se aplica às cópias manuais.",
            negated=True,
            contexts=(context,),
        )
        report = analyze(
            source, ScriptedGenerator({("v0:s1", "direct"): [raw]}), qa_enabled=False
        )
        self.assertEqual(1, report.counts.occurrences)
        self.assertEqual(0, report.counts.by_segment[0].occurrences)
        self.assertEqual("v0:s0", report.occurrences[0].context_evidence[0].segment_id)
        wrong = copy.deepcopy(raw)
        wrong["evidence"][0]["role"] = "context"
        wrong["evidence"][1]["role"] = "assertion"
        bad = analyze(
            source, ScriptedGenerator({("v0:s1", "direct"): [wrong]}), qa_enabled=False
        )
        self.assertEqual("literal_rejected", bad.candidates[0].validation)
        only_context = copy.deepcopy(raw)
        only_context["evidence"][0]["role"] = "context"
        no_assertion = analyze(
            source,
            ScriptedGenerator({("v0:s1", "direct"): [only_context]}),
            qa_enabled=False,
        )
        self.assertEqual("literal_rejected", no_assertion.candidates[0].validation)

    def test_qualifiers_and_compound_are_semantically_checked_and_not_counted(self):
        text = "Se a conexão cair, o backup não é realizado."
        raw = candidate(
            "v0:s0",
            text,
            unit_type="conditional",
            negated=True,
            conditions=("a conexão cair",),
        )
        evaluator = SyntheticEvaluator()
        report = analyze(
            videos([text]),
            ScriptedGenerator({("v0:s0", "direct"): [raw]}),
            evaluator,
            qa_enabled=False,
        )
        self.assertTrue(report.units[0].qualifiers.negated)
        self.assertEqual(("a conexão cair",), report.units[0].qualifiers.conditions)
        validation = next(call for call in evaluator.calls if "candidate" in call[0])
        self.assertEqual(
            {
                "support",
                "conditions",
                "negation",
                "quantities",
                "modality",
                "attribution",
            },
            set(validation[1]),
        )
        compound = candidate(
            "v0:s0", text, text="O backup ocorre diariamente e é retido 30 dias."
        )
        rejected = analyze(
            videos([text]),
            ScriptedGenerator({("v0:s0", "direct"): [compound]}),
            SyntheticEvaluator(granularity={compound["text"]: "compound"}),
            qa_enabled=False,
        )
        self.assertEqual("needs_repair", rejected.candidates[0].validation)
        self.assertEqual(0, rejected.counts.unique_units)

    def test_coverage_reads_source_and_recovers_missing_property_bounded(self):
        text = "Os backups ocorrem a cada 24 horas e são guardados por 30 dias."
        first = candidate(
            "v0:s0",
            text,
            text="Os backups ocorrem a cada 24 horas.",
            quantities=("24 horas",),
        )
        second = candidate(
            "v0:s0",
            text,
            text="Os backups são guardados por 30 dias.",
            quantities=("30 dias",),
        )
        generator = ScriptedGenerator(
            {("v0:s0", "direct"): [first], ("v0:s0", "recovery"): [second]}
        )
        evaluator = SyntheticEvaluator(coverage={"v0:s0": [0.99, 0.01]})
        report = analyze(
            videos([text]),
            generator,
            evaluator,
            qa_enabled=False,
            max_coverage_rounds=1,
        )
        self.assertEqual("completed", report.status)
        self.assertEqual(2, report.counts.unique_units)
        self.assertEqual(
            ["gap_detected", "no_gap_signaled"],
            [item.state for item in report.coverage],
        )
        audit_call = next(
            call for call in evaluator.calls if "original_target" in call[0]
        )
        self.assertEqual(text, audit_call[0]["original_target"]["text"])
        self.assertGreater(len(audit_call[1]), 1)
        pending = analyze(
            videos([text]),
            ScriptedGenerator(),
            SyntheticEvaluator(coverage={"v0:s0": [0.99]}),
            qa_enabled=False,
            max_coverage_rounds=1,
        )
        self.assertEqual("partial", pending.status)
        self.assertEqual(3, json.loads(pending.metadata_json)["logical_calls"])
        self.assertIn("coverage_no_progress", [item.kind for item in pending.issues])

    def test_calls_context_pair_limits_and_provider_failure_are_not_zero_success(self):
        text = "O prazo é 30 dias."
        raw = candidate("v0:s0", text)
        limited = analyze(
            videos([text]), ScriptedGenerator({("v0:s0", "direct"): [raw]}), max_calls=1
        )
        self.assertEqual("failed", limited.status)
        self.assertFalse(limited.counts.valid_zero)
        oversized = analyze(videos(["x" * 5000]), max_context_chars=4000)
        self.assertEqual("failed", oversized.status)
        self.assertEqual(0, json.loads(oversized.metadata_json)["logical_calls"])
        generator = ScriptedGenerator(
            {("v0:s0", "direct"): ServiceTimeoutError("Fixture timeout")}
        )
        unavailable = analyze(videos([text]), generator)
        self.assertEqual("failed", unavailable.status)
        self.assertEqual("ServiceTimeoutError", unavailable.calls[0].status)

    def test_oversized_target_remains_pending_without_blocking_other_targets(self):
        text = "O prazo é 30 dias."
        generator = ScriptedGenerator({("v0:s1", "direct"): [candidate("v0:s1", text)]})
        report = analyze(
            videos(["x" * 10000, text]),
            generator,
            qa_enabled=False,
            max_context_chars=8000,
        )
        self.assertEqual("partial", report.status)
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(["v0:s1"], [record.segment_id for record in report.coverage])

    def test_equivalence_no_transitive_chaining_and_other_relations_retained(self):
        texts = ["Afirmação A.", "Afirmação B.", "Afirmação C."]
        items = {
            (f"v0:s{index}", "direct"): [candidate(f"v0:s{index}", text)]
            for index, text in enumerate(texts)
        }
        relationships = {
            frozenset(texts[:2]): "equivalent",
            frozenset(texts[1:]): "equivalent",
            frozenset((texts[0], texts[2])): "contradiction",
        }
        report = analyze(
            videos(texts),
            ScriptedGenerator(items),
            SyntheticEvaluator(relation=relationships),
            qa_enabled=False,
        )
        self.assertEqual(2, report.counts.unique_units)
        self.assertEqual(3, report.counts.occurrences)
        self.assertEqual(("c0", "c1"), report.units[0].candidate_ids)
        self.assertEqual("contradiction", report.relations[1].relation)
        self.assertFalse(report.relations[2].merged)
        self.assertEqual("partial", report.status)
        self.assertIn(
            "equivalence_inconsistent", [issue.kind for issue in report.issues]
        )
        budget = analyze(
            videos(texts),
            ScriptedGenerator(items),
            qa_enabled=False,
            max_pair_comparisons=0,
        )
        self.assertEqual("partial", budget.status)
        self.assertEqual(3, budget.counts.unique_units)
        uncertain = analyze(
            videos(texts[:2]),
            ScriptedGenerator(items),
            SyntheticEvaluator(
                relation={frozenset(texts[:2]): "equivalent"},
                relation_probability=0.6,
                relation_confidence=0.99,
            ),
            qa_enabled=False,
        )
        self.assertEqual(2, uncertain.counts.unique_units)
        self.assertEqual("uncertain", uncertain.relations[0].relation)

    def test_actual_typesafe_serializer_drives_analysis_and_usage_hashes(self):
        bodies = []

        def transport(url, *, headers, body, timeout):
            payload = json.loads(body)
            bodies.append(payload)
            answers = {}
            state = json.loads(payload["state"])
            for key, question in payload["questions"].items():
                if question["type"] == "noul":
                    answers[key] = {
                        "type": "noul",
                        "noul": 0.01 if "original_target" in state else 0.99,
                    }
                else:
                    options = tuple(question["criteria"])
                    selected = "atomic"
                    answers[key] = {
                        "type": "choice",
                        "choice": selected,
                        "probabilities": {
                            item: 1.0 if item == selected else 0.0 for item in options
                        },
                        "confidence": 1.0,
                    }
            return TransportResponse(
                200,
                {},
                json.dumps(
                    {
                        "model": "jev-1.13.0",
                        "answers": answers,
                        "usage": {"input_tokens": 5, "output_tokens": 3},
                    }
                ).encode(),
            )

        text = "O prazo é 30 dias."
        evaluator = TypeSafeEvaluator(
            TypeSafeConfig(api_key="fixture-secret"), transport=transport
        )
        report = analyze(
            videos([text]),
            ScriptedGenerator({("v0:s0", "direct"): [candidate("v0:s0", text)]}),
            evaluator,
            qa_enabled=False,
        )
        self.assertEqual("completed", report.status)
        self.assertEqual(2, len(bodies))
        exported = report.to_dict()
        self.assertNotIn("fixture-secret", json.dumps(exported))
        self.assertEqual("typesafe", exported["metadata"]["evaluator_provider"])
        self.assertEqual("typesafe", exported["metadata"]["evaluator"]["provider"])
        self.assertEqual(5, exported["calls"][1]["metadata"]["input_tokens"])
        self.assertEqual(
            "atomic", exported["calls"][1]["decisions"]["choice"][0]["selected"]
        )
        self.assertIsNone(exported["calls"][1]["decisions"]["noul"][0]["confidence"])
        self.assertEqual(1.0, exported["candidates"][0]["granularity_probability"])
        self.assertEqual(64, len(exported["calls"][1]["input_hash"]))


class InformationSchemaTests(unittest.TestCase):
    def test_strict_empty_numeric_negation_condition_and_malformed_qa(self):
        interpreter = DomainResponseInterpreter()
        spec = OutputSpec(kind="information_units", segment_ids=("v0:s0",))
        self.assertEqual(
            {"candidates": [], "issues": []},
            interpreter.interpret('{"candidates":[],"issues":[]}', spec),
        )
        text = "Se chover, não ocorre por 30 dias."
        raw = candidate(
            "v0:s0", text, negated=True, conditions=("chover",), quantities=("30 dias",)
        )
        value = {"candidates": [raw], "issues": []}
        self.assertTrue(
            interpreter.interpret(json.dumps(value), spec)["candidates"][0][
                "qualifiers"
            ]["negated"]
        )
        for mutate in (
            lambda item: item["qualifiers"].update(negated="true"),
            lambda item: item["qualifiers"].update(quantities=[30]),
            lambda item: item["evidence"][0].update(start_char=True),
            lambda item: item["evidence"][0].update(segment_id="unknown"),
            lambda item: item.update(extra="no"),
        ):
            invalid = copy.deepcopy(raw)
            mutate(invalid)
            with self.assertRaises(InvalidResponseError):
                interpreter.interpret(
                    json.dumps({"candidates": [invalid], "issues": []}), spec
                )
        with self.assertRaises(InvalidResponseError):
            interpreter.interpret(str(value), spec)
        with self.assertRaises(InvalidResponseError):
            interpreter.interpret(
                json.dumps(value), replace(spec, kind="information_qa")
            )

    def test_openai_and_ollama_real_generation_transports_use_structured_kind(self):
        envelope = '{"candidates":[],"issues":[]}'
        create = Mock(
            return_value={
                "status": "completed",
                "model": "fixture",
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": envelope}],
                    }
                ],
            }
        )
        openai = OpenAIAdapter(
            config=LLMConfig(provider="openai", model="fixture"),
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
        )
        from open_video_summary.contracts import GenerationRequest

        result = openai.generate(
            GenerationRequest(
                "fixture", OutputSpec(kind="information_units", segment_ids=("v0:s0",))
            )
        )
        schema = create.call_args.kwargs["text"]["format"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertIn(
            "evidence",
            schema["schema"]["properties"]["candidates"]["items"]["properties"],
        )
        self.assertEqual(3, result.metadata.input_tokens)
        generate = Mock(
            return_value={
                "response": envelope,
                "model": "fixture",
                "prompt_eval_count": 4,
                "eval_count": 2,
            }
        )
        ollama = OllamaAdapter(client=SimpleNamespace(generate=generate))
        result = ollama.generate(
            GenerationRequest("fixture", OutputSpec(kind="information_qa"))
        )
        self.assertEqual("json", generate.call_args.kwargs["format"])
        self.assertEqual(4, result.metadata.input_tokens)

    @unittest.skipUnless(
        importlib.util.find_spec("httpx2"),
        "Pinned OpenAI SDK transport is not installed.",
    )
    def test_installed_openai_sdk_serializes_both_information_schemas_offline(self):
        import httpx2
        from openai import OpenAI
        from open_video_summary.contracts import GenerationRequest

        sent = []

        def handle(request):
            sent.append(json.loads(request.content))
            response = {
                "id": "resp_information_offline",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "fixture",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"candidates":[],"issues":[]}',
                            }
                        ],
                    }
                ],
            }
            return httpx2.Response(200, json=response)

        with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
            with OpenAI(
                api_key="offline-synthetic-key",
                base_url="https://offline.example/v1",
                max_retries=0,
                http_client=transport,
            ) as client:
                adapter = OpenAIAdapter(
                    config=LLMConfig(provider="openai", model="fixture"), client=client
                )
                for kind in ("information_units", "information_qa"):
                    adapter.generate(
                        GenerationRequest(
                            "Synthetic fixture",
                            OutputSpec(kind=kind, segment_ids=("v0:s0",), max_items=2),
                        )
                    )
        for index, request in enumerate(sent):
            response_format = request["text"]["format"]
            self.assertTrue(response_format["strict"])
            schema = response_format["schema"]["properties"]["candidates"]
            self.assertEqual(2, schema["maxItems"])
            evidence_schema = schema["items"]["properties"]["evidence"]["items"]
            self.assertEqual(
                ["v0:s0"], evidence_schema["properties"]["segment_id"]["enum"]
            )
            self.assertIn(
                "negated", schema["items"]["properties"]["qualifiers"]["required"]
            )
            question_type = schema["items"]["properties"]["question"]["type"]
            self.assertEqual(
                ["string", "null"] if index == 0 else "string", question_type
            )
            self.assertFalse(request["store"])


if __name__ == "__main__":
    unittest.main()
