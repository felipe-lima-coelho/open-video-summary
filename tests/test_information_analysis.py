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


def evaluation_state(context):
    """Read the labels used by the deterministic evaluator test doubles."""
    def section(label):
        return context.split(label + ":\n", 1)[1].split("\n\n", 1)[0]

    if context.startswith("Candidate id:"):
        first = context.split("\n\n", 1)[0].splitlines()
        return {
            "candidate_id": first[0].split(": ", 1)[1],
            "target_id": first[1].split(": ", 1)[1],
            "candidate": {"text": section("Candidate claim")},
        }
    if context.startswith("Selected assertion wording:"):
        return {"candidate": {"text": section("Candidate claim")}, "anchor_binding": True}
    if context.startswith("Original target ["):
        identifier = context.split("]:", 1)[0].split("[", 1)[1]
        return {
            "original_target": {"id": identifier, "text": section(f"Original target [{identifier}]")}
        }
    if context.startswith("Left candidate id:"):
        return {"left": {"text": section("Left claim")}, "right": {"text": section("Right claim")}}
    raise AssertionError("Unexpected evaluator test state.")


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
        state = evaluation_state(context)
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
                "conditions_annotation",
                "quantities_annotation",
                "modality_annotation",
                "attribution_annotation",
                "negation_annotation",
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
        self.assertEqual(("c0", "t1:c0"), report.units[0].candidate_ids)
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
            state = evaluation_state(payload["state"])
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


class InformationEvaluationRegressionTests(unittest.TestCase):
    """Report-derived failures with explicit offline decisions, not a benchmark."""

    def evaluator(self, signals=None, *, granularity="atomic"):
        bodies = []

        def transport(url, *, headers, body, timeout):
            payload = json.loads(body)
            bodies.append(payload)
            audit = "missing" in payload["questions"]
            answers = {}
            for key, question in payload["questions"].items():
                if question["type"] == "noul":
                    probability = (0.01 if audit else signals(payload, key)
                                   if callable(signals) else (signals or {}).get(key, 0.99))
                    answers[key] = {"type": "noul", "noul": probability}
                else:
                    selected = granularity if key == "granularity" else "complementary"
                    answers[key] = {
                        "type": "choice", "choice": selected,
                        "probabilities": {label: float(label == selected) for label in question["criteria"]},
                        "confidence": 0.99,
                    }
            return TransportResponse(200, {}, json.dumps({"answers": answers}).encode())

        return TypeSafeEvaluator(TypeSafeConfig(api_key="offline-fixture"), transport=transport), bodies

    def test_unique_exact_quote_resolves_report_offset_drift_and_retains_raw(self):
        text = "O Google decidiu não atualizar mais os aplicativos dos celulares da Huawei, segunda maior fabricante de smartphones do mundo, atrás da sul-coreana Samsung."
        quote = "segunda maior fabricante de smartphones do mundo, atrás da sul-coreana Samsung."
        raw = candidate("v0:s0", text, quote=quote, quantities=("segunda maior",))
        raw["evidence"][0]["end_char"] -= 1
        report = analyze(videos([text]), ScriptedGenerator({("v0:s0", "direct"): [raw]}), qa_enabled=False)
        record = report.candidates[0]
        self.assertEqual("accepted", record.validation)
        self.assertEqual(154, json.loads(record.raw_json)["evidence"][0]["end_char"])
        self.assertEqual(text[76:155], record.candidate.evidence[0].quote)
        resolution = record.evidence_resolutions[0]
        self.assertEqual((76, 154, 76, 155), (resolution.supplied_start_char, resolution.supplied_end_char, resolution.resolved_start_char, resolution.resolved_end_char))
        self.assertEqual("unique_exact_quote", resolution.method)
        self.assertEqual(3, report.to_dict()["schema_version"])
        self.assertEqual(155, report.to_dict()["candidates"][0]["evidence_resolutions"][0]["resolved_end_char"])
        self.assertEqual(text, report.snapshot.current_segments[0].content)

    def test_repeated_quotes_require_matching_supplied_anchor_and_never_choose_first(self):
        text = "Huawei, Google, Huawei."
        good = candidate("v0:s0", text, quote="Huawei", offset=16)
        bad = copy.deepcopy(good)
        bad["evidence"][0].update(start_char=15, end_char=21)
        report = analyze(videos([text]), ScriptedGenerator({("v0:s0", "direct"): [good, bad]}), qa_enabled=False)
        self.assertEqual(["accepted", "literal_rejected"], [record.validation for record in report.candidates])
        self.assertEqual(16, report.occurrences[0].assertion_evidence[0].start_char)
        self.assertEqual((), report.candidates[0].evidence_resolutions)
        self.assertIn("more than once", report.candidates[1].reasons[0])

    def test_literal_resolution_never_fuzzes_accents_or_changes_source_or_role(self):
        source = videos(["A decisão foi anunciada.", "Outro trecho."])
        valid = candidate("v0:s0", source[0].segments[0].content)
        for change in (
            lambda raw: raw["evidence"][0].update(quote="A decisao foi anunciada."),
            lambda raw: raw["evidence"][0].update(segment_id="v0:s1"),
            lambda raw: raw["evidence"][0].update(role="context"),
        ):
            raw = copy.deepcopy(valid)
            change(raw)
            with self.subTest(raw=raw):
                report = analyze(source, ScriptedGenerator({("v0:s0", "direct"): [raw]}), qa_enabled=False)
                self.assertEqual("literal_rejected", report.candidates[0].validation)
                self.assertEqual(0, report.counts.accepted_candidates)
                self.assertEqual(source[0].segments[0].content, report.snapshot.current_segments[0].content)

    def test_report_derived_scores_and_production_annotations_remain_separate_guards(self):
        text = "O Android é o sistema operacional mais usado em smartphones ao redor do planeta."
        raw = candidate("v0:s0", text, quantities=("mais usado",))
        generator = ScriptedGenerator({("v0:s0", "direct"): [raw]})
        original_signals = {"support": 0.91, "conditions": 0.82, "negation": 0.90, "quantities": 0.94, "modality": 0.67, "attribution": 0.46}
        evaluator, _ = self.evaluator(original_signals)
        pending = analyze(videos([text]), generator, evaluator, qa_enabled=False, max_coverage_rounds=0)
        self.assertEqual(0, pending.counts.accepted_candidates)
        evaluator, bodies = self.evaluator({"support": 0.95, "conditions": 0.94, "quantities_annotation": 0.85})
        accepted = analyze(videos([text]), generator, evaluator, qa_enabled=False)
        self.assertEqual(1, accepted.counts.accepted_candidates)
        self.assertEqual(0.85, json.loads(accepted.metadata_json)["settings"]["acceptance_threshold"])
        payload = bodies[0]
        self.assertIn("<source-assertion-0>" + text + "</source-assertion-0>", payload["state"])
        self.assertNotIn("source_identity", payload["state"])
        self.assertNotIn("Python Unicode", payload["state"])
        self.assertEqual("One contextual proposition. A condition and its consequence form one proposition; an attributed claim includes its reporting source.", payload["questions"]["granularity"]["criteria"]["atomic"])
        for failed_signal in ("support", "conditions", "negation", "quantities", "modality", "attribution"):
            evaluator, _ = self.evaluator({failed_signal: 0.03})
            with self.subTest(signal=failed_signal):
                report = analyze(videos([text]), generator, evaluator, qa_enabled=False, max_coverage_rounds=0)
                self.assertEqual(0, report.counts.accepted_candidates)
                self.assertIn(failed_signal, report.candidates[0].reasons)

    def test_source_scope_is_visible_without_authorizing_uncited_assertions_and_qa_is_checked(self):
        quote = "O prazo é fixo."
        text = quote + " O valor secreto é 99 dias."
        raw = candidate("v0:s0", text, quote=quote, qa=True)
        evaluator, bodies = self.evaluator({"qa_consistency": 0.04})
        report = analyze(videos([text]), ScriptedGenerator({("v0:s0", "qa"): [raw]}), evaluator, max_coverage_rounds=0)
        payload = next(body for body in bodies if "support" in body["questions"])
        self.assertIn("valor secreto", payload["state"])
        assertion = payload["state"].split("<source-assertion-0>", 1)[1].split("</source-assertion-0>", 1)[0]
        self.assertNotIn("valor secreto", assertion)
        self.assertIn("may not supply a new assertion", payload["questions"]["support"]["instructions"])
        self.assertIn("qa_anchor", payload["questions"])
        self.assertIn("qa_consistency", report.candidates[0].reasons)
        self.assertEqual(0, report.counts.accepted_candidates)

    def test_recovery_receives_literal_failures_and_semantic_reasons(self):
        text = "O prazo é de 30 dias."
        raw = candidate("v0:s0", text, quantities=("30 dias",))
        bad = copy.deepcopy(raw)
        bad["evidence"][0]["quote"] = "O prazo é de 40 dias."
        inputs = []

        def generate(data, route):
            inputs.append((data, route))
            return [bad] if route == "direct" else [raw]

        report = analyze(videos([text]), ScriptedGenerator(callback=generate), SyntheticEvaluator(coverage={"v0:s0": [0.99, 0.01]}), qa_enabled=False)
        recovery = next(data for data, route in inputs if route == "recovery")
        repair = recovery["repair_candidates"][0]
        self.assertEqual("literal_rejected", repair["validation"])
        self.assertEqual(bad, repair["proposal"])
        self.assertIn("does not occur literally", repair["reasons"][0])
        self.assertEqual(1, report.counts.accepted_candidates)

    def test_cropped_condition_never_enters_inventory_even_after_qualified_recovery(self):
        source = "Se a conexão cair, o backup não é realizado."
        quote = "o backup não é realizado."
        bad = candidate("v0:s0", source, quote=quote, negated=True)
        repaired = candidate("v0:s0", source, negated=True, conditions=("a conexão cair",))
        evaluator = SyntheticEvaluator(coverage={"v0:s0": [0.99, 0.01]})
        original_evaluate = evaluator.evaluate
        states = []

        def evaluate(state, noul=None, choice=None):
            states.append(state)
            result = original_evaluate(state, noul, choice)
            if state.startswith("Candidate id:") and "Candidate claim:\n" + quote in state:
                self.assertIn("Se a conexão cair, <source-assertion-0>" + quote, state)
                return replace(result, noul=tuple(
                    replace(item, probability=0.02) if item.id == "conditions" else item
                    for item in result.noul
                ))
            return result

        evaluator.evaluate = evaluate
        report = analyze(
            videos([source]), ScriptedGenerator({("v0:s0", "direct"): [bad],
                                                 ("v0:s0", "recovery"): [repaired]}),
            evaluator, qa_enabled=False,
        )
        self.assertEqual(["needs_repair", "accepted"], [record.validation for record in report.candidates])
        self.assertEqual([source], [unit.text for unit in report.units])
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(1, report.counts.occurrences)
        self.assertNotIn("c0", report.units[0].candidate_ids)
        self.assertEqual(["gap_detected", "no_gap_signaled"], [audit.state for audit in report.coverage])
        unconditional = analyze(videos([quote]), ScriptedGenerator({("v0:s0", "direct"): [candidate("v0:s0", quote)]}), qa_enabled=False)
        self.assertEqual(1, unconditional.counts.accepted_candidates)

    def test_whole_cited_context_preserves_surrounding_qualifier_scope(self):
        context_source = "Segundo Maria, o prazo pode ser de 30 dias."
        target = "Esse prazo aplica-se às cópias manuais."
        context_quote = "o prazo pode ser de 30 dias."
        context = {"segment_id": "v0:s0", "quote": context_quote,
                   "start_char": context_source.index(context_quote), "end_char": len(context_source),
                   "role": "context"}
        raw = candidate("v0:s1", target, text="O prazo de 30 dias aplica-se às cópias manuais.", contexts=(context,))
        evaluator, bodies = self.evaluator({"attribution": 0.03, "modality": 0.03})
        report = analyze(videos([context_source, target]), ScriptedGenerator({("v0:s1", "direct"): [raw]}), evaluator, qa_enabled=False)
        state = next(body["state"] for body in bodies if "support" in body["questions"])
        self.assertIn("Original reference scope:\n[v0:s0] Segundo Maria, <source-context-1>" + context_quote, state)
        self.assertEqual(0, report.counts.accepted_candidates)

    def test_report_derived_faithful_content_is_not_blocked_by_annotation_uncertainty(self):
        for text, qualifiers, failed, probability in (
            ("O Google decidiu não atualizar mais os aplicativos dos celulares da Huawei.", {"negated": True}, "attribution_annotation", 0.57),
            ("A Huawei é a segunda maior fabricante de smartphones do mundo.", {"quantities": ("segunda maior",)}, "quantities_annotation", 0.49),
        ):
            with self.subTest(text=text):
                raw = candidate("v0:s0", text, **qualifiers)
                evaluator, _ = self.evaluator({failed: probability})
                report = analyze(videos([text]), ScriptedGenerator({("v0:s0", "direct"): [raw]}), evaluator, qa_enabled=False)
                record, unit = report.candidates[0], report.units[0]
                self.assertEqual("accepted", record.validation)
                self.assertEqual((), record.reasons)
                self.assertEqual("uncertain", record.annotation_state)
                self.assertEqual((failed,), record.annotation_reasons)
                self.assertIsNone(unit.qualifiers)
                self.assertEqual("unknown", unit.qualifier_state)
                self.assertEqual(record.id, unit.representative_candidate_id)
                self.assertEqual(1, report.counts.unique_units)
                self.assertIn("annotation_unresolved", [issue.kind for issue in report.issues])
                self.assertFalse(report.counts.valid_zero)

    def test_wrong_auxiliary_metadata_is_unknown_and_never_influences_other_operations(self):
        source = "O Google decidiu não atualizar os aplicativos. A Huawei desenvolve uma plataforma."
        first = candidate("v0:s0", source, quote="O Google decidiu não atualizar os aplicativos.",
                          attribution="INVENTED_REPORTER_731", negated=False)
        paraphrase = copy.deepcopy(first)
        paraphrase["text"] = "O Google decidiu parar de atualizar os aplicativos."
        second = candidate("v0:s0", source, quote="A Huawei desenvolve uma plataforma.")
        requests = []
        generator = ScriptedGenerator({("v0:s0", "direct"): [first, paraphrase], ("v0:s0", "recovery"): [second]})
        evaluator = SyntheticEvaluator(coverage={"v0:s0": [0.99, 0.01]})
        evaluate_original = evaluator.evaluate

        def evaluate(state, noul=None, choice=None):
            requests.append(state)
            result = evaluate_original(state, noul, choice)
            if state.startswith("Candidate id:"):
                return replace(result, noul=tuple(
                    replace(item, probability=0.02) if item.id == "attribution_annotation" else item
                    for item in result.noul
                ))
            return result

        evaluator.evaluate = evaluate
        report = analyze(videos([source]), generator, evaluator, qa_enabled=False)
        self.assertEqual(3, report.counts.accepted_candidates)
        self.assertTrue(all(unit.qualifiers is None for unit in report.units))
        self.assertNotIn("INVENTED_REPORTER_731", "\n".join(requests))
        recovery = json.loads(generator.requests[-1].prompt.split("\nInput:\n", 1)[1])
        self.assertTrue(all(item["qualifiers"] is None for item in recovery["accepted"]))
        self.assertTrue(all(item["annotation_state"] == "invalid" for item in recovery["accepted"]))
        exported = report.to_dict()
        self.assertNotIn("qualifiers", exported["candidates"][0]["candidate"])
        self.assertEqual("INVENTED_REPORTER_731", exported["candidates"][0]["candidate"]["proposed_qualifiers"]["attribution"])
        self.assertEqual(first, exported["candidates"][0]["raw"])

    def test_representative_does_not_borrow_metadata_from_an_equivalent_paraphrase(self):
        source = "O backup não é realizado."
        first = candidate("v0:s0", source, negated=True)
        second = candidate("v0:s0", source, text="O backup deixa de ser realizado.", negated=False)
        evaluator, _ = self.evaluator(lambda payload, key: 0.4 if key == "negation_annotation" and "Candidate claim:\n" + source in payload["state"] else 0.99)
        # The paraphrase has its own annotation audit; an uncertain representative
        # must not borrow its negative/affirmative form-dependent annotation.
        original = evaluator.evaluate
        def evaluate(state, noul=None, choice=None):
            result = original(state, noul, choice)
            if choice and "relation" in choice:
                return replace(result, choice=(replace(result.choice[0], selected="equivalent", probabilities=tuple((key, float(key == "equivalent")) for key in dict(result.choice[0].probabilities))),))
            return result
        evaluator.evaluate = evaluate
        report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"): [first, second]}), evaluator, qa_enabled=False)
        self.assertEqual(1, len(report.units))
        self.assertEqual("verified", report.candidates[1].annotation_state)
        self.assertEqual(source, report.units[0].text)
        self.assertIsNone(report.units[0].qualifiers)
        self.assertEqual("c0", report.units[0].representative_candidate_id)

    def test_exact_duplicates_are_processed_even_after_paid_budget_exhaustion(self):
        source = "O Android pertence ao Google. A Huawei desenvolve uma plataforma."
        first = candidate("v0:s0", source, quote="O Android pertence ao Google.")
        other = candidate("v0:s0", source, quote="A Huawei desenvolve uma plataforma.")
        duplicate = copy.deepcopy(first)
        duplicate["qualifiers"]["attribution"] = "Unverified proposal"
        evaluator, _ = self.evaluator({"attribution_annotation": 0.3})
        report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"): [first, other, duplicate]}), evaluator, qa_enabled=False, max_calls=5)
        self.assertEqual(5, json.loads(report.metadata_json)["logical_calls"])
        self.assertEqual(2, report.counts.unique_units)
        self.assertEqual(2, report.counts.occurrences)
        self.assertEqual("exact_validated_proposition", report.relations[0].origin)
        self.assertTrue(report.relations[0].merged)
        self.assertIn("pair_budget_exhausted", [issue.kind for issue in report.issues])

    def test_pair_priority_reaches_same_anchor_paraphrases_without_implying_merge(self):
        texts = ["A retenção é 30 dias.", "A frequência é 24 horas.", "O sistema operacional faz o smartphone funcionar."]
        items = {(f"v0:s{index}", "direct"): [candidate(f"v0:s{index}", text)] for index, text in enumerate(texts)}
        items[("v0:s2", "direct")].append(candidate("v0:s2", texts[2], text="O sistema operacional é o que faz o smartphone funcionar."))
        report = analyze(videos(texts), ScriptedGenerator(items), qa_enabled=False, max_pair_comparisons=1)
        self.assertEqual(("t2:c0", "t2:c1"), (report.relations[0].left_candidate_id, report.relations[0].right_candidate_id))
        self.assertEqual("complementary", report.relations[0].relation)
        self.assertFalse(report.relations[0].merged)
        self.assertEqual(4, report.counts.unique_units)
        self.assertIn("pair_budget_exhausted", [issue.kind for issue in report.issues])

    def test_anchor_binding_cannot_borrow_an_unrelated_source_assertion(self):
        source = "O backup é diário. A retenção é 30 dias."
        wrong = candidate("v0:s0", source, quote="O backup é diário.", text="A retenção é 30 dias.", quantities=("30 dias",))
        evaluator, bodies = self.evaluator({"anchor_binding": 0.09})
        report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"): [wrong]}), evaluator, qa_enabled=False)
        binding = next(body for body in bodies if "anchor_binding" in body["questions"])
        self.assertEqual("Selected assertion wording:\nO backup é diário.\n\nCandidate claim:\nA retenção é 30 dias.", binding["state"])
        self.assertNotIn(source, binding["state"])
        self.assertNotIn("quantities_annotation", binding["questions"])
        self.assertEqual("rejected", report.candidates[0].validation)
        self.assertEqual(("anchor_binding",), report.candidates[0].reasons)
        self.assertEqual(0, report.counts.unique_units)
        self.assertFalse(any("support" in body["questions"] for body in bodies))

    def test_exact_validation_reuse_is_traced_and_changed_content_is_not_cached(self):
        text = "O Google decidiu não atualizar os aplicativos."
        first = candidate("v0:s0", text, negated=True)
        repeat = copy.deepcopy(first)
        repeat["evidence"][0]["end_char"] -= 1
        changed = candidate("v0:s0", text, text="O Google decidiu atualizar os aplicativos.")
        generator = ScriptedGenerator({("v0:s0", "direct"): [first, repeat, changed]})
        evaluator, bodies = self.evaluator({"anchor_binding": 0.1})
        report = analyze(videos([text]), generator, evaluator, qa_enabled=False)
        self.assertEqual(["accepted", "accepted", "rejected"], [record.validation for record in report.candidates])
        self.assertEqual("c0", report.candidates[1].validation_reused_from)
        self.assertEqual(1, len(report.candidates[1].evidence_resolutions))
        self.assertIsNone(report.candidates[2].validation_reused_from)
        self.assertEqual(1, sum("support" in body["questions"] for body in bodies))
        self.assertEqual(1, sum("anchor_binding" in body["questions"] for body in bodies))
        self.assertEqual(1, report.counts.unique_units)
        self.assertEqual(1, report.counts.occurrences)

    def test_validation_reuse_requires_same_qa_annotations_evidence_and_source_scope(self):
        source = "O backup é diário. O backup é diário."
        first = candidate("v0:s0", source, quote="O backup é diário.", offset=0)
        for change in (
            lambda item: item.update(question="Qual é a frequência?", answer="É semanal."),
            lambda item: item["qualifiers"].update(attribution="Maria"),
            lambda item: item["evidence"][0].update(start_char=19, end_char=37),
        ):
            other = copy.deepcopy(first)
            change(other)
            with self.subTest(other=other):
                evaluator, bodies = self.evaluator()
                report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"): [first, other]}), evaluator, qa_enabled=False)
                self.assertTrue(all(record.validation_reused_from is None for record in report.candidates))
                self.assertEqual(2, sum("support" in body["questions"] for body in bodies))

    def test_literal_containment_proves_binding_only_at_the_selected_source_occurrence(self):
        source = "Se a conexão cair, o backup não é realizado. Em qualquer situação, o backup não é realizado."
        quote = "o backup não é realizado."
        text = "Se a conexão cair, " + quote
        good = candidate("v0:s0", source, text=text, quote=quote, offset=source.index(quote), negated=True)
        wrong_occurrence = candidate("v0:s0", source, text=text, quote=quote, offset=source.rindex(quote), negated=True)
        evaluator, bodies = self.evaluator({"anchor_binding": 0.1})
        report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"): [good, wrong_occurrence]}), evaluator, qa_enabled=False)
        self.assertEqual("literal_source_passage", report.candidates[0].anchor_binding_origin)
        self.assertEqual("accepted", report.candidates[0].validation)
        self.assertEqual("evaluator", report.candidates[1].anchor_binding_origin)
        self.assertEqual("rejected", report.candidates[1].validation)
        self.assertEqual(1, sum("anchor_binding" in body["questions"] for body in bodies))
        self.assertEqual(1, sum("support" in body["questions"] for body in bodies))

    def test_literal_containment_never_skips_source_fidelity_or_atomicity(self):
        source = "Se a conexão cair, o backup não é realizado."
        quote = "o backup não é realizado."
        for field in ("conditions", "negation", "attribution", "modality"):
            cropped = candidate("v0:s0", source, quote=quote)
            evaluator, bodies = self.evaluator({field: 0.03})
            with self.subTest(field=field):
                report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"): [cropped]}), evaluator, qa_enabled=False)
                self.assertEqual("literal_identity", report.candidates[0].anchor_binding_origin)
                self.assertEqual(0, report.counts.unique_units)
                self.assertTrue(any("support" in body["questions"] for body in bodies))
        compound_source = "O backup é diário. A retenção é 30 dias."
        raw = candidate("v0:s0", compound_source, quote="O backup é diário.")
        raw["text"] = compound_source
        evaluator, bodies = self.evaluator(granularity="compound")
        report = analyze(videos([compound_source]), ScriptedGenerator({("v0:s0", "direct"): [raw]}), evaluator, qa_enabled=False)
        self.assertEqual("literal_source_passage", report.candidates[0].anchor_binding_origin)
        self.assertEqual("needs_repair", report.candidates[0].validation)
        self.assertEqual(0, report.counts.unique_units)
        self.assertTrue(any("granularity" in body["questions"] for body in bodies))

    def test_literal_containment_cannot_combine_disjoint_passages(self):
        source = "O backup é diário. A retenção é 30 dias."
        raw = candidate("v0:s0", source, quote="O backup é diário.")
        raw["evidence"].append({"segment_id": "v0:s0", "quote": "A retenção é 30 dias.",
                                "start_char": 19, "end_char": len(source), "role": "assertion"})
        evaluator, bodies = self.evaluator({"anchor_binding": 0.1})
        report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"): [raw]}), evaluator, qa_enabled=False)
        self.assertEqual("evaluator", report.candidates[0].anchor_binding_origin)
        self.assertEqual(0, report.counts.unique_units)
        self.assertTrue(any("anchor_binding" in body["questions"] for body in bodies))


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
