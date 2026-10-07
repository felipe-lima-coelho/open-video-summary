"""Offline source recovery and keyed comparisons; no scientific accuracy claim."""

import copy
import json
import socket
import unittest
from dataclasses import replace
from unittest.mock import patch

from test_information_analysis import (
    ScriptedGenerator, SyntheticEvaluator, analyze, candidate, videos,
)
from test_information_pairs_parallel import make_run
from open_video_summary.adapters.typesafe import TransportResponse, TypeSafeConfig, TypeSafeEvaluator
from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import capture_snapshot
from open_video_summary.core.summarizers.information_evaluation import (
    RELATION_ADJUDICATION_QUESTIONS, anchor_binding_spec, resolve_relation,
)
from open_video_summary.errors import ServiceTimeoutError


class InformationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket, "create_connection", side_effect=AssertionError("Network disabled"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def evaluator(self, *, low=False, signals=None, labels=None, malformed=False):
        requests = []
        def transport(url, *, headers, body, timeout):
            request = json.loads(body)
            requests.append(request)
            answers = {}
            for identifier, question in request["questions"].items():
                if question["type"] == "noul":
                    probability = (signals or {}).get(identifier, 0.01)
                    answers[identifier] = {"type": "noul", "noul": probability}
                    continue
                if identifier == "relation":
                    scope = request["state"]
                else:
                    scope = request["state"].split(f"BEGIN {identifier}\n", 1)[1].split(f"\nEND {identifier}", 1)[0]
                    self.assertIn(f"Evaluate only {identifier}", question["instructions"])
                left = scope.split("Left candidate id: ", 1)[1].split("\n", 1)[0]
                right = scope.split("Right candidate id: ", 1)[1].split("\n", 1)[0]
                label = (labels or {}).get((left, right), "complementary")
                mass = 0.89 if low else 0.99
                options = question["criteria"]
                answers[identifier] = {"type": "choice", "choice": label,
                    "probabilities": {option: mass if option == label else (1-mass)/(len(options)-1) for option in options},
                    "confidence": mass}
            if malformed:
                answers = {"wrong_pair": next(iter(answers.values()))}
            return TransportResponse(200, {}, json.dumps({"model": "jev-1.13.0", "answers": answers,
                "usage": {"input_tokens": 200, "output_tokens": 30}}).encode())
        evaluator = TypeSafeEvaluator(TypeSafeConfig(api_key="offline", max_attempts=1), transport=transport)
        return evaluator, requests

    def test_keyed_batches_match_individual_decisions_and_complete_link_groups(self):
        texts = ["Backup diário", "Backup todo dia", "Backup a cada 24 horas", "Retenção de 30 dias"]
        labels = {("c0", "c1"): "equivalent", ("c1", "c2"): "equivalent",
                  ("c0", "c2"): "contradiction", ("c2", "c3"): "more_specific_right"}
        reports = []
        requests_by_size = []
        for batch_size in (1, 4):
            run = make_run(texts, duplicates=(texts[0],))
            run.config = replace(run.config, pair_batch_size=batch_size)
            run.evaluator, requests = self.evaluator(labels=labels)
            reports.append(run.run())
            requests_by_size.append(requests)
        one, four = reports
        self.assertEqual([(r.left_candidate_id, r.right_candidate_id, r.relation) for r in one.relations],
                         [(r.left_candidate_id, r.right_candidate_id, r.relation) for r in four.relations])
        self.assertEqual([u.candidate_ids for u in one.units], [u.candidate_ids for u in four.units])
        self.assertEqual(3, four.counts.unique_units)
        self.assertEqual(6, len(requests_by_size[0]))
        self.assertEqual(2, len(requests_by_size[1]))
        self.assertEqual(10, len(four.relations))
        self.assertEqual(3, json.loads(four.metadata_json)["reused_pair_relations"])
        for call in four.calls:
            metadata = json.loads(call.metadata_json)
            self.assertEqual(len(json.loads(call.decisions_json)["choice"]), metadata["question_count"])
            self.assertGreaterEqual(metadata["base_estimated_input_tokens"], metadata["question_count"] * 128)
        self.assertIn("equivalence_inconsistent", [issue.kind for issue in four.issues])

    def test_batch_context_falls_back_to_individual_and_explicit_pair_cap_is_decisions(self):
        run = make_run(["Backup diário " + "x"*900, "Retenção 30 dias " + "y"*900, "Mais uma propriedade " + "z"*900])
        run.config = replace(run.config, pair_batch_size=4, max_context_chars=12000, max_pair_comparisons=2)
        run.evaluator, requests = self.evaluator()
        report = run.run()
        self.assertEqual(2, len(report.relations))
        self.assertEqual(2, json.loads(report.metadata_json)["paid_pair_comparisons"])
        self.assertEqual(2, len(requests))
        self.assertTrue(all(len(request["questions"]) == 1 for request in requests))
        self.assertEqual(1, json.loads(report.metadata_json)["unexamined_pair_count"])

    def test_wrong_batch_identifiers_never_create_relations_or_merge(self):
        run = make_run(["Backup diário", "Backup semanal", "Retenção 30 dias"])
        run.config = replace(run.config, pair_batch_size=4)
        run.evaluator, requests = self.evaluator(malformed=True)
        report = run.run()
        self.assertEqual(1, len(requests))
        self.assertEqual((), report.relations)
        self.assertEqual(3, report.counts.unique_units)
        self.assertEqual(3, json.loads(report.metadata_json)["unexamined_pair_count"])

    def test_low_flat_choice_gets_one_focused_followup_without_lowering_threshold(self):
        signals = dict.fromkeys(RELATION_ADJUDICATION_QUESTIONS, 0.01)
        run = make_run(["A porta é azul", "O backup é diário"])
        run.config = replace(run.config, max_relation_adjudications=1)
        run.evaluator, requests = self.evaluator(low=True, signals=signals)
        report = run.run()
        relation = report.relations[0]
        self.assertEqual("complementary", relation.relation)
        self.assertEqual("focused_entailment_adjudication", relation.origin)
        self.assertEqual("complementary", relation.initial_relation)
        self.assertEqual(0.89, relation.initial_probability)
        self.assertEqual(signals, dict(relation.adjudication_signals))
        self.assertIsNone(relation.probability)
        self.assertEqual(.99, relation.adjudication_strength)
        self.assertEqual(2, len(requests))
        self.assertEqual(5, len(requests[1]["questions"]))
        self.assertEqual(1, json.loads(report.metadata_json)["relation_adjudications"])

    def test_binary_relation_definitions_preserve_direction_scope_and_real_uncertainty(self):
        negative = dict.fromkeys(RELATION_ADJUDICATION_QUESTIONS, 0.01)
        cases = [({}, "complementary"), ({"left_entails_right": .99, "right_entails_left": .99}, "equivalent"),
                 ({"left_entails_right": .99}, "more_specific_left"), ({"right_entails_left": .99}, "more_specific_right"),
                 ({"incompatible": .99}, "contradiction"), ({"correction_left": .99}, "correction_left"),
                 ({"correction_right": .99}, "correction_right"), ({"left_entails_right": .45}, "uncertain"),
                 ({"left_entails_right": .99, "incompatible": .99}, "uncertain")]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                self.assertEqual(expected, resolve_relation(negative | changes, .90)[0])
        run = make_run(["A regra é temporária", "A regra foi publicada"])
        run.config = replace(run.config, max_relation_adjudications=1)
        run.evaluator, _ = self.evaluator(low=True, signals=negative | {"left_entails_right": .45})
        report = run.run()
        self.assertEqual("uncertain", report.relations[0].relation)
        self.assertTrue(report.counts.counts_provisional)
        self.assertEqual(2, report.counts.unique_units)

    def test_global_request_cap_bounds_batch_and_followup_dispatch(self):
        run = make_run(["A regra é azul", "O backup é diário", "A retenção é curta"])
        run.config = replace(run.config, pair_batch_size=4, max_calls=1, max_relation_adjudications=100)
        run.budget.limit = 1
        run.evaluator, requests = self.evaluator(low=True)
        report = run.run()
        self.assertEqual(1, len(requests))
        self.assertEqual(1, json.loads(report.metadata_json)["logical_calls"])
        self.assertEqual(0, json.loads(report.metadata_json)["relation_adjudications"])
        self.assertTrue(all(r.relation == "uncertain" for r in report.relations))

    def test_primary_wave_retains_followup_capacity_and_reclaims_unused_reserve(self):
        run = make_run(["A regra é azul", "O backup é diário", "A retenção é curta", "O prazo é fixo"])
        run.config = replace(run.config, max_calls=5, max_relation_adjudications=3)
        run.budget.limit = 5
        run.evaluator, requests = self.evaluator(low=True)
        report = run.run()
        self.assertEqual(5, len(requests))
        self.assertEqual(1, json.loads(report.metadata_json)["relation_adjudications"])
        self.assertEqual(5, len(requests[-1]["questions"]))
        self.assertEqual(2, json.loads(report.metadata_json)["unexamined_pair_count"])

        run = make_run(["A regra é azul", "O backup é diário", "A retenção é curta", "O prazo é fixo", "O suporte é local"])
        run.config = replace(run.config, pair_batch_size=2, max_calls=5, max_relation_adjudications=2)
        run.budget.limit = 5
        run.evaluator, requests = self.evaluator()
        report = run.run()
        self.assertEqual(5, len(requests))
        self.assertEqual(10, len(report.relations))
        self.assertEqual(0, json.loads(report.metadata_json)["unexamined_pair_count"])
        self.assertEqual(0, json.loads(report.metadata_json)["relation_adjudications"])

    def test_qa_timeout_decomposes_original_source_and_coverage_still_runs(self):
        sentences = ["A primeira regra estabelece backup automático diário com uma cópia no servidor.",
                     "A segunda regra exige retenção de trinta dias para cada cópia do servidor.",
                     "A terceira regra permite arquivar as cópias antigas em um sistema separado."]
        source = " ".join(sentences)
        failed = False
        def generate(data, route):
            nonlocal failed
            if route == "direct":
                return [candidate("v0:s0", source, quote=sentences[0])]
            window = data["discovery_window"]
            self.assertNotIn("accepted", data)
            self.assertNotIn("repair_candidates", data)
            self.assertEqual(source, data["target"]["text"])
            if not failed and window["end_char"]-window["start_char"] > 80:
                failed = True
                raise ServiceTimeoutError("offline")
            return [candidate("v0:s0", source, quote=sentence, qa=True)
                    for sentence in sentences if window["start_char"] <= source.index(sentence) < window["end_char"]]
        generator = ScriptedGenerator(callback=generate)
        report = analyze(videos([source]), generator, qa_window_chars=160, max_qa_windows=12)
        self.assertEqual("completed", report.status)
        self.assertEqual(1, len(report.coverage))
        self.assertEqual(3, report.counts.unique_units)
        self.assertIn("ServiceTimeoutError", [call.status for call in report.calls])
        windows = json.loads(report.metadata_json)["qa_source_windows"]
        self.assertEqual("failed", windows[0]["status"])
        self.assertTrue(any(row["status"] == "completed" for row in windows))
        for record in report.candidates:
            for evidence in record.candidate.evidence:
                self.assertEqual(evidence.quote, source[evidence.start_char:evidence.end_char])

    def test_qa_failure_limit_does_not_drop_direct_candidates_or_original_audit(self):
        source = "Uma regra comunicada sobre a cópia automática de todos os arquivos do servidor remoto."
        generator = ScriptedGenerator({("v0:s0", "direct"): [candidate("v0:s0", source)],
                                       ("v0:s0", "qa"): ServiceTimeoutError("offline")})
        report = analyze(videos([source]), generator, max_qa_windows=1)
        self.assertEqual(1, report.counts.accepted_candidates)
        self.assertEqual(1, len(report.coverage))
        self.assertEqual("partial", report.status)
        self.assertIn("qa_window_budget_exhausted", [issue.kind for issue in report.issues])

    def test_modal_anchor_binding_uses_only_literal_cited_reference_context(self):
        source = "Google e Huawei tinham uma parceria. Trump pode ter acabado com essa parceria. Outro assunto."
        quote = "Trump pode ter acabado com essa parceria."
        raw = candidate("v0:s0", source, quote=quote,
            text="Trump pode ter acabado com a parceria entre Google e Huawei.", modality="possibilidade",
            contexts=[{"segment_id": "v0:s0", "quote": "Google e Huawei tinham uma parceria.",
                       "start_char": 0, "end_char": len("Google e Huawei tinham uma parceria."), "role": "context"}])
        report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"): [raw]}), qa_enabled=False)
        record = report.candidates[0]
        state, _ = anchor_binding_spec(record.candidate)
        self.assertIn("Google e Huawei tinham uma parceria.", state)
        self.assertNotIn("Outro assunto.", state)
        self.assertIn("resolves references only", state)
        self.assertEqual("accepted", record.validation)
        # Source modality remains an independent gate even with resolved binding.
        bad = copy.deepcopy(raw)
        bad["text"] = "Trump acabou com a parceria entre Google e Huawei."
        signals = SyntheticEvaluator(validation={bad["text"]: .10})
        rejected = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"): [bad]}), signals, qa_enabled=False)
        self.assertEqual(0, rejected.counts.accepted_candidates)


if __name__ == "__main__":
    unittest.main()
