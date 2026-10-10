"""Literal focus coordinates are mechanical; source acceptance remains separate."""

import copy
import importlib.util
import json
import unittest

from test_information_analysis import ScriptedGenerator, candidate, videos
from test_information_inventory import InventoryEvaluator
from open_video_summary.adapters.information_schema import information_schema, validate_information
from open_video_summary.adapters.llm import DomainResponseInterpreter, OpenAIAdapter
from open_video_summary.contracts import GenerationRequest, OutputSpec
from open_video_summary.core.summarizers.information_analysis import (
    InformationAnalyzer, PROTOCOL, _AnalysisRun,
)
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import capture_snapshot, fingerprint
from open_video_summary.core.summarizers.information_inventory import (
    DECOMPOSITION_INSTRUCTION, FOCUS_INSTRUCTION, focus_protocol,
)
from open_video_summary.errors import InvalidResponseError
from open_video_summary.utils.providers import LLMConfig


def without_offsets(raw):
    result = copy.deepcopy(raw)
    for evidence in result["evidence"]:
        evidence.update(start_char=None, end_char=None)
    return result


class FocusEvidenceOffsetTests(unittest.TestCase):
    def run_for(self, texts, *, proposals=(), evaluator=None):
        generator = ScriptedGenerator(callback=lambda data, route:
            copy.deepcopy(proposals) if route == "discover_coverage_foci" else [])
        analyzer = InformationAnalyzer(generator, evaluator or InventoryEvaluator(),
            InformationAnalysisConfig(concurrency=1, qa_enabled=False,
                max_coverage_rounds=0, max_literal_repairs=0,
                max_granularity_checks=0, max_relation_adjudications=0))
        return _AnalysisRun(analyzer, capture_snapshot(videos(texts))), generator

    def test_null_offsets_are_focus_only_and_must_be_a_pair(self):
        raw = without_offsets(candidate("v0:s0", "A decisão não mudou."))
        value = {"candidates": [raw], "issues": []}
        focus = OutputSpec("information_foci", segment_ids=("v0:s0",))
        self.assertEqual(value, validate_information(value, focus))
        for kind in ("information_units", "information_qa"):
            ordinary = copy.deepcopy(value)
            if kind == "information_qa":
                ordinary["candidates"][0].update(question="O que mudou?", answer="A decisão não mudou.")
            with self.subTest(kind=kind), self.assertRaises(InvalidResponseError):
                DomainResponseInterpreter().interpret(json.dumps(ordinary), OutputSpec(kind))
        for start, end in ((None, 3), (0, None), (True, 3), (0, False),
                           (0.0, 3), (0, 3.0), (-1, 3), (3, 3), (4, 3)):
            bad = copy.deepcopy(value)
            bad["candidates"][0]["evidence"][0].update(start_char=start, end_char=end)
            with self.subTest(start=start, end=end), self.assertRaises(InvalidResponseError):
                validate_information(bad, focus)
        for kind, expected in (("information_foci", ["integer", "null"]),
                               ("information_units", "integer"), ("information_qa", "integer")):
            evidence = information_schema(OutputSpec(kind))["properties"]["candidates"]["items"]["properties"]["evidence"]["items"]
            self.assertEqual(expected, evidence["properties"]["start_char"]["type"])
            self.assertEqual(expected, evidence["properties"]["end_char"]["type"])

    def test_unique_unicode_quotes_keep_canonical_evidence_and_raw_null_provenance(self):
        source = "🧪 A ação não mudou. Uma ação diferente foi proposta."
        raw = candidate("v0:s0", source, quote="A ação não mudou.", negated=True)
        null = without_offsets(raw)
        run, _ = self.run_for([source])
        target = run.snapshot.current_segments[0]
        numeric, numeric_resolutions = run._literal_candidate(raw, target, (target.id,))
        resolved, resolutions = run._literal_candidate(null, target, (target.id,), allow_null_offsets=True)
        self.assertEqual(numeric, resolved)
        self.assertEqual((), numeric_resolutions)
        self.assertEqual((None, None, 2, 19), (
            resolutions[0].supplied_start_char, resolutions[0].supplied_end_char,
            resolutions[0].resolved_start_char, resolutions[0].resolved_end_char))
        self.assertEqual("unique_exact_quote", resolutions[0].method)
        self.assertIsNone(null["evidence"][0]["start_char"])
        self.assertEqual(source, target.content)
        with self.assertRaises(ValueError):
            run._literal_candidate(null, target, (target.id,))

    def test_repeated_and_overlapping_quotes_require_the_supplied_occurrence(self):
        source = "Alfa. Alfa."
        run, _ = self.run_for([source])
        target = run.snapshot.current_segments[0]
        for offset in (0, 6):
            raw = candidate(target.id, source, quote="Alfa.", offset=offset)
            resolved, resolutions = run._literal_candidate(raw, target, (target.id,), allow_null_offsets=True)
            self.assertEqual(offset, resolved.evidence[0].start_char)
            self.assertEqual((), resolutions)
        for text, quote in ((source, "Alfa."), ("aaa", "aa")):
            run, _ = self.run_for([text])
            target = run.snapshot.current_segments[0]
            raw = without_offsets(candidate(target.id, text, quote=quote))
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "more than once"):
                run._literal_candidate(raw, target, (target.id,), allow_null_offsets=True)

    def test_missing_nonliteral_or_forbidden_evidence_never_changes_source_or_role(self):
        source = "A ação não mudou."
        run, _ = self.run_for([source, source])
        target = run.snapshot.current_segments[0]
        original = without_offsets(candidate(target.id, source))
        for change, permitted in (
            ({"quote": "A acao não mudou."}, ("v0:s0",)),
            ({"quote": "A ação não mudou."}, ("v0:s0",)),
            ({"quote": "A ação  não mudou."}, ("v0:s0",)),
            ({"segment_id": "v0:s1"}, ("v0:s0", "v0:s1")),
            ({"segment_id": "v0:s1", "role": "context"}, ("v0:s0",)),
            ({"role": "context"}, ("v0:s0",)),
            ({"end_char": 1}, ("v0:s0",)),
        ):
            raw = copy.deepcopy(original)
            raw["evidence"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                run._literal_candidate(raw, target, permitted, allow_null_offsets=True)
        contextual = copy.deepcopy(original)
        contextual["evidence"].append({**original["evidence"][0], "segment_id": "v0:s1", "role": "context"})
        resolved, _ = run._literal_candidate(contextual, target, ("v0:s0", "v0:s1"), allow_null_offsets=True)
        self.assertNotEqual(resolved.evidence[0].source_identity, resolved.evidence[1].source_identity)
        self.assertEqual(["assertion", "context"], [item.role for item in resolved.evidence])

    def test_fixed_proposals_have_identical_semantic_validation_inputs(self):
        source = "No Alfa, os backups são guardados por 30 dias."
        raw = candidate("v0:s0", source, quantities=("30 dias",))
        outputs = []
        for proposal in (raw, without_offsets(raw)):
            evaluator = InventoryEvaluator()
            run, _ = self.run_for([source], evaluator=evaluator)
            destination = []
            record = run._validate(proposal, run.snapshot.current_segments[0], "coverage_focus",
                0, ("v0:s0",), destination=destination, identifier="fixed-focus")
            outputs.append((record, evaluator.calls, evaluator.inventory_states))
        numeric, null = outputs
        self.assertEqual(numeric[0].candidate, null[0].candidate)
        self.assertEqual(numeric[0].validation, null[0].validation)
        self.assertEqual("accepted", null[0].validation)
        self.assertEqual(numeric[1:], null[1:])
        self.assertIsNone(json.loads(null[0].raw_json)["evidence"][0]["start_char"])
        self.assertEqual((), numeric[0].evidence_resolutions)
        self.assertEqual(1, len(null[0].evidence_resolutions))

    def test_ambiguous_null_focus_remains_visibly_unresolved(self):
        source = "Alfa. Alfa."
        raw = without_offsets(candidate("v0:s0", source, quote="Alfa."))
        evaluator = InventoryEvaluator()
        run, _ = self.run_for([source], proposals=[raw], evaluator=evaluator)
        run._discover_foci(run.snapshot.current_segments[0])
        focus = run.coverage_foci[0]
        self.assertIsNone(focus.proposition)
        self.assertEqual("literal_rejected", focus.proposal_record.validation)
        self.assertEqual("uncertain", focus.state)
        self.assertEqual((), focus.matches)
        self.assertTrue(any(item.kind == "literal_evidence_rejected" for item in run.issues))
        self.assertEqual([], evaluator.calls)

    def test_null_focus_still_requires_semantic_acceptance_and_records_failures(self):
        source = "O prazo é 30 dias."
        raw = without_offsets(candidate("v0:s0", source, quantities=("30 dias",)))
        evaluator = InventoryEvaluator(validation={source: .01})
        run, generator = self.run_for([source], proposals=[raw], evaluator=evaluator)
        run._discover_foci(run.snapshot.current_segments[0])
        focus = run.coverage_foci[0]
        self.assertNotEqual("accepted", focus.proposal_record.validation)
        self.assertEqual("uncertain", focus.state)
        self.assertEqual((), focus.candidate_ids)
        self.assertTrue(evaluator.calls)
        report = run.run()
        exported = report.to_dict()
        self.assertEqual(8, exported["schema_version"])
        self.assertEqual(fingerprint(focus_protocol()), exported["metadata"]["prompt_template_hashes"]["focus_protocol"])
        exported_focus = exported["coverage_foci"][0]
        self.assertIsNone(exported_focus["proposal_record"]["raw"]["evidence"][0]["start_char"])
        self.assertIsInstance(exported_focus["proposition"]["evidence"][0]["start_char"], int)

    def test_only_focus_wire_protocol_and_schema_change(self):
        run, _ = self.run_for(["Alfa.", "Contexto."])
        target = run.snapshot.current_segments[0]
        initial, spec = run._inventory_request(target, "discover_coverage_foci", FOCUS_INSTRUCTION)
        supplemental, other_spec = run._inventory_request(target, "discover_coverage_foci",
            FOCUS_INSTRUCTION, {"open_source_audit": {"round": 1}})
        self.assertEqual("information_foci", spec.kind)
        self.assertEqual(spec, other_spec)
        self.assertTrue(initial.startswith(focus_protocol()))
        self.assertTrue(supplemental.startswith(focus_protocol()))
        self.assertNotIn("Evidence is an exact substring at zero-based", initial)
        first_data = json.loads(initial.split("\nInput:\n")[1])
        later_data = json.loads(supplemental.split("\nInput:\n")[1])
        self.assertEqual(first_data, {key: value for key, value in later_data.items() if key != "open_source_audit"})
        for route in ("direct", "qa", "recovery"):
            prompt, other_spec, _ = run._extraction_request(target, route)
            self.assertTrue(prompt.startswith(PROTOCOL))
            self.assertNotEqual("information_foci", other_spec.kind)
            self.assertNotIn("application computes canonical", prompt)
        prompt, other_spec = run._inventory_request(target, "decompose_candidate", DECOMPOSITION_INSTRUCTION)
        self.assertTrue(prompt.startswith(PROTOCOL))
        self.assertEqual("information_units", other_spec.kind)


@unittest.skipUnless(importlib.util.find_spec("httpx2"), "Pinned OpenAI SDK transport is not installed.")
class InstalledFocusSchemaTests(unittest.TestCase):
    def test_actual_sdk_sends_nullable_focus_schema_and_keeps_generation_settings(self):
        import httpx2
        from openai import OpenAI
        from test_llm_adapters import response

        raw = without_offsets(candidate("v0:s0", "O prazo é 30 dias."))
        body = {"candidates": [raw], "issues": []}
        sent = []
        def handle(request):
            sent.append(json.loads(request.content))
            fixture = response(json.dumps(body), effort="high")
            fixture.update(id="resp_offline", object="response", created_at=0)
            return httpx2.Response(200, json=fixture)
        with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
            with OpenAI(api_key="synthetic-offline-key", http_client=transport) as client:
                adapter = OpenAIAdapter(config=LLMConfig(provider="openai",
                    model="gpt-6-luna", reasoning_effort="high"), client=client)
                result = adapter.generate(GenerationRequest("unchanged source", OutputSpec(
                    "information_foci", max_items=16, segment_ids=("v0:s0",))))
        self.assertEqual(body, result.value)
        self.assertEqual("unchanged source", sent[0]["input"])
        self.assertEqual({"effort": "high"}, sent[0]["reasoning"])
        self.assertEqual(16384, sent[0]["max_output_tokens"])
        self.assertFalse(sent[0]["store"])
        self.assertNotIn("temperature", sent[0])
        self.assertEqual("ovs_information_foci", sent[0]["text"]["format"]["name"])
        self.assertEqual(information_schema(OutputSpec("information_foci", max_items=16,
            segment_ids=("v0:s0",))), sent[0]["text"]["format"]["schema"])


if __name__ == "__main__":
    unittest.main()
