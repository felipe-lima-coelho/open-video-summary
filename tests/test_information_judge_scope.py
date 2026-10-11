"""Production payload contracts with scripted scores, not model-accuracy evidence."""

import json
import socket
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_information_analysis import ScriptedGenerator, candidate, videos
from test_information_inventory import InventoryEvaluator
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer, _AnalysisRun
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import CoverageFocus, capture_snapshot
from open_video_summary.core.summarizers.information_evaluation import (
    EVALUATION_TEMPLATE_VERSION, validation_spec, coverage_spec, relation_batch_spec,
)
from open_video_summary.core.summarizers.information_inventory import (
    ATOMICITY_QUESTIONS, DECOMPOSITION_INSTRUCTION, DECOMPOSITION_QUESTIONS,
    JOINT_COMPONENT_QUESTION, inventory_templates, joint_granularity_questions,
)


class InformationJudgeScopeTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket, "create_connection", side_effect=AssertionError("Network disabled"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def seeded_run(self, texts, proposals, *, target_index=0, evaluator=None, components=()):
        generator = ScriptedGenerator(callback=lambda data, route:
            list(components) if route == "decompose_candidate" else [])
        evaluator = evaluator or InventoryEvaluator()
        run = _AnalysisRun(InformationAnalyzer(generator, evaluator, InformationAnalysisConfig(
            concurrency=1, qa_enabled=False, max_literal_repairs=0, max_coverage_foci=0,
            max_calls=200)), capture_snapshot(videos(texts)))
        target = run.snapshot.current_segments[target_index]
        for proposal in proposals:
            record = run._validate(proposal, target, "direct", 0, run.snapshot.current_order)
            self.assertEqual("accepted", record.validation)
        return run, target, evaluator, generator

    def assert_claim_only_state(self, state):
        # No candidate envelopes or annotations are supplied as extra propositions.
        for field in ("parent", "components", "candidate", "target", "context",
                      "qualifiers", "unit_type", "unresolved_references"):
            self.assertNotIn('"' + field + '":', json.dumps(state))

    def test_identical_and_paraphrased_claims_have_no_source_import_or_proper_part_instruction(self):
        source = "O sistema permite usar aplicativos. A fabricante mantém a loja."
        claim = "O sistema permite usar aplicativos."
        for peer_text in (claim, "É possível usar aplicativos com o sistema."):
            with self.subTest(peer=peer_text):
                proposals = [candidate("v0:s0", source, text=claim),
                             candidate("v0:s0", source, quote=claim, text=peer_text)]
                run, target, evaluator, _ = self.seeded_run([source], proposals)
                run._reconcile_granularity(target)
                calls = [(state, questions) for state, questions in evaluator.inventory_calls
                         if "splittable" in questions]
                self.assertEqual(2, len(calls))
                state, questions = next((state, questions) for state, questions in calls
                                        if state["parent_claim_text"] == claim)
                self.assertEqual(claim, state["parent_claim_text"])
                self.assertEqual([peer_text], state["component_claim_texts"])
                self.assert_claim_only_state(state)
                self.assertIn("ONLY parent_claim_text", questions["splittable"])
                self.assertIn("component_claim_texts[0]", questions["component0"])
                self.assertIn("An identical statement or complete paraphrase is NOT", questions["component0"])
                self.assertIn("never add other source assertions", questions["component0"])
                self.assertEqual([], run.decompositions)
                self.assertEqual([proposal["text"] for proposal in proposals],
                                 [record.candidate.text for record in run.candidates])

    def test_single_identical_component_is_judged_and_cannot_bypass_minimum_partition_size(self):
        source = "A repercussão é mundial. Outra notícia foi publicada."
        proposal = candidate("v0:s0", source, text="A repercussão é mundial.")
        run, target, evaluator, _ = self.seeded_run([source], [proposal, proposal])
        record = run._verify_decomposition(target, run.candidates[0], run.candidates[1:])
        state, questions = evaluator.inventory_calls[-1]
        self.assertEqual(state["parent_claim_text"], state["component_claim_texts"][0])
        self.assertIn("An identical statement or complete paraphrase covers", questions["components_cover_parent"])
        # Deliberately inconsistent all-positive fixture scores still cannot replace
        # a parent with one child; scores remain visible rather than being rewritten.
        self.assertEqual(dict.fromkeys(DECOMPOSITION_QUESTIONS, .99), dict(record.signals))
        self.assertEqual("partial", record.state)
        self.assertEqual("unit_candidate", run.candidates[0].inventory_role)

    def test_true_compound_generation_and_verification_use_only_explicit_parent_meaning(self):
        claim = "O backup é diário e a retenção dura 30 dias."
        source = claim + " A rede foi ampliada."
        parent = candidate("v0:s0", source, text=claim, quantities=("30 dias",))
        parts = [candidate("v0:s0", source, quote="O backup é diário", text="O backup é diário."),
                 candidate("v0:s0", source, quote="a retenção dura 30 dias", text="A retenção dura 30 dias.",
                           quantities=("30 dias",))]
        evaluator = InventoryEvaluator(splits={claim: tuple(item["text"] for item in parts)})
        run, target, evaluator, generator = self.seeded_run([source], [parent],
            evaluator=evaluator, components=parts)
        run._reconcile_granularity(target)
        self.assertEqual("verified", run.decompositions[-1].state)
        state, questions = next((state, questions) for state, questions in evaluator.inventory_calls
                                if "components_cover_parent" in questions)
        self.assertEqual(claim, state["parent_claim_text"])
        self.assertEqual([item["text"] for item in parts], state["component_claim_texts"])
        self.assert_claim_only_state(state)
        for key, question in questions.items():
            if key != "parent_compound":
                self.assertIn("component_claim_texts", question)
        request = generator.requests[0]
        data = json.loads(request.prompt.split("\nInput:\n", 1)[1])
        self.assertIn(DECOMPOSITION_INSTRUCTION, request.prompt)
        self.assertEqual(claim, data["parent_claim_text"])
        self.assertNotIn("parent_candidate", data)
        self.assertEqual(source, data["target"]["text"])
        self.assertIn("Never decompose the whole target", request.prompt)
        self.assertEqual(parent, json.loads(run.candidates[0].raw_json))
        self.assertEqual(("30 dias",), run.candidates[0].candidate.qualifiers.quantities)

    def test_indexed_atomicity_retains_own_anaphora_attribution_condition_and_negation_scope(self):
        context = "O backup foi configurado. A outra máquina falhou."
        source = "Segundo Ana, se a rede cair, ele talvez não seja realizado. A retenção mudou."
        quote = "ele talvez não seja realizado"
        text = "Segundo Ana, se a rede cair, o backup talvez não seja realizado."
        reference = {"segment_id": "v0:s0", "quote": "O backup", "start_char": 0,
                     "end_char": len("O backup"), "role": "context"}
        proposal = candidate("v0:s1", source, quote=quote, text=text, attribution="Ana",
            conditions=("se a rede cair",), negated=True, modality="talvez", contexts=(reference,))
        other = candidate("v0:s1", source, quote="A retenção mudou.")
        run, target, evaluator, _ = self.seeded_run([context, source], [proposal, other], target_index=1)
        original_candidates = [record.candidate for record in run.candidates]
        run._review_atomicity(target)
        state, questions = evaluator.inventory_calls[-1]
        self.assertEqual(text, state["claim0_text"])
        self.assertEqual(other["text"], state["claim1_text"])
        self.assert_claim_only_state(state)
        for index in range(2):
            for key in ATOMICITY_QUESTIONS:
                self.assertIn(f"Classify ONLY claim{index}_text", questions[f"claim{index}_{key}"])
        scope = state["source_scopes"]["claim0_text"]
        self.assertIn("Segundo Ana, se a rede cair, <selected-assertion>" + quote,
                      scope["assertion_scopes"][0]["marked_text"])
        self.assertNotIn("A retenção", scope["assertion_scopes"][0]["marked_text"])
        self.assertIn("<source-context-1>O backup</source-context-1>",
                      scope["reference_scopes"][0]["marked_text"])
        self.assertEqual([], state["source_scopes"]["claim1_text"]["reference_scopes"])
        joint = run._joint_state(target, run.candidates[0], run.candidates[1:])
        self.assertEqual(scope["assertion_spans"], joint["source_scopes"]["parent_claim_text"]["assertion_spans"])
        self.assertIn(source[:source.index(quote)],
                      joint["source_scopes"]["parent_claim_text"]["original_source_scope"]["marked_text"])
        self.assertIn("conditions", DECOMPOSITION_QUESTIONS["qualifiers_preserved"])
        self.assertIn("source_scopes", DECOMPOSITION_QUESTIONS["qualifiers_preserved"])
        self.assertEqual(original_candidates, [record.candidate for record in run.candidates])

    def test_identical_text_at_different_occurrences_keeps_exact_distinguishing_anchors(self):
        phrase = "A repercussão é mundial."
        source = phrase + " " + phrase
        proposals = [candidate("v0:s0", source, quote=phrase, offset=offset)
                     for offset in (0, len(phrase) + 1)]
        run, target, _, _ = self.seeded_run([source], proposals)
        state = run._joint_state(target, run.candidates[0], run.candidates[1:])
        scopes = state["source_scopes"]
        self.assertEqual(0, scopes["parent_claim_text"]["assertion_spans"][0]["start_char"])
        self.assertEqual(len(phrase) + 1,
                         scopes["component_claim_texts"][0]["assertion_spans"][0]["start_char"])
        self.assertNotEqual(scopes["parent_claim_text"]["original_source_scope"]["marked_text"],
                            scopes["component_claim_texts"][0]["original_source_scope"]["marked_text"])
        self.assertIn("exact marked assertion spans", DECOMPOSITION_QUESTIONS["source_occurrence"])
        self.assertIn("separate repetition", DECOMPOSITION_QUESTIONS["source_occurrence"])

    def test_prompt_manifest_includes_indexed_part_question_and_construction_version(self):
        self.assertEqual("source-scope-fidelity-v10-no-record-ids", EVALUATION_TEMPLATE_VERSION)
        self.assertEqual(JOINT_COMPONENT_QUESTION, inventory_templates()["joint_component_question"])
        self.assertEqual({"splittable"}, set(joint_granularity_questions(0)))
        questions = joint_granularity_questions(2)
        self.assertEqual({"splittable", "component0", "component1"}, set(questions))
        self.assertIn("component_claim_texts[1]", questions["component1"])

    def test_candidate_identity_and_route_never_change_semantic_builder_inputs(self):
        source = "Segundo Ana, se a rede cair, o backup talvez não seja realizado. A retenção dura 30 dias."
        first = candidate("v0:s0", source, quote=source.split(". ")[0] + ".",
            attribution="Ana", conditions=("se a rede cair",), negated=True, modality="talvez")
        second = candidate("v0:s0", source, quote="A retenção dura 30 dias.", quantities=("30 dias",))
        run, target, _, _ = self.seeded_run([source], [first, second])
        records = run.candidates
        renamed = [replace(row, id=f"private-renamed-{index}", route="different-route",
            raw_json='{"provenance":"different"}', round=99) for index, row in enumerate(records)]
        def payloads(rows):
            return (
                validation_spec(rows[0].candidate, rows[0].id, target),
                coverage_spec(target, (), rows, ()),
                relation_batch_spec([(rows[0], rows[1]), (rows[1], rows[0])], run.segments, "Compare."),
                run._joint_state(target, rows[0], rows[1:]),
                run._atomicity_state(target, rows),
                run._focus_revision_data(rows[0]),
            )
        self.assertEqual(payloads(records), payloads(renamed))
        encoded = json.dumps(payloads(renamed), ensure_ascii=False)
        self.assertNotIn("private-renamed", encoded)
        self.assertNotIn("different-route", encoded)
        for value in ("Ana", "se a rede cair", "talvez", "30 dias", target.source_identity):
            self.assertIn(value, encoded)
        self.assertIn("v0:s0", encoded)
        self.assertEqual(first, json.loads(records[0].raw_json))

    def test_focus_ids_and_routes_stay_in_correlation_not_judge_state(self):
        source = "O backup é diário. A retenção dura 30 dias."
        proposals = [candidate("v0:s0", source, quote="O backup é diário."),
                     candidate("v0:s0", source, quote="A retenção dura 30 dias.")]
        captures = []
        for prefix, route in (("original", "direct"), ("renamed", "recovery")):
            run, target, evaluator, _ = self.seeded_run([source], proposals)
            run.candidates = [replace(record, id=f"{prefix}:{index}", route=route,
                raw_json=json.dumps({"route":route})) for index, record in enumerate(run.candidates)]
            first = run.candidates[0]
            proposal = replace(first, id=prefix + ":focus", candidate=replace(first.candidate,
                text="A frequência do backup é diária."))
            focus = CoverageFocus(prefix + ":focus", target.id, proposal.candidate, proposal)
            run._focus_matches(target, focus)
            damaged = replace(proposal, validation="needs_repair", reasons=("conditions",))
            run._revise_focus(target, replace(focus, proposal_record=damaged), run.candidates)
            calls = run.calls[-2:]
            captures.append((evaluator.inventory_calls, [call.input_hash for call in calls]))
            for call in calls:
                correlation = json.loads(call.metadata_json)["correlation"]
                self.assertEqual(prefix + ":focus", correlation["focus_id"])
            state = json.dumps(evaluator.inventory_calls)
            self.assertNotIn(prefix + ":", state)
            self.assertNotIn('"route"', state)
        self.assertEqual(captures[0], captures[1])

    def test_same_validation_payload_keeps_distinct_external_candidate_correlation(self):
        source = "O backup é diário."
        run, target, _, _ = self.seeded_run([source], [candidate("v0:s0", source)])
        claim = run.candidates[0].candidate
        for identifier in ("route-one:c91", "route-two:c3"):
            state, questions, choices = validation_spec(claim, identifier, target)
            run._evaluate("validate_candidate", state, questions, choices,
                          correlation={"candidate_id":identifier})
        one, two = run.calls[-2:]
        self.assertEqual(one.input_hash, two.input_hash)
        self.assertEqual("route-one:c91", json.loads(one.metadata_json)["correlation"]["candidate_id"])
        self.assertEqual("route-two:c3", json.loads(two.metadata_json)["correlation"]["candidate_id"])


if __name__ == "__main__":
    unittest.main()
