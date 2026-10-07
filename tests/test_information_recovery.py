"""Offline source recovery and keyed comparisons; no scientific accuracy claim."""

import copy
import json
import math
import socket
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

from test_information_analysis import (
    ScriptedGenerator, SyntheticEvaluator, analyze, candidate, videos,
)
from test_information_pairs_parallel import make_run
from open_video_summary.adapters.typesafe import TransportResponse, TypeSafeConfig, TypeSafeEvaluator
from open_video_summary.core.summarizers.information_analysis import (
    InformationAnalyzer, _AnalysisRun, _LimitReached, _ProviderStopped,
)
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import capture_snapshot
from open_video_summary.core.summarizers.information_evaluation import (
    RELATION_ADJUDICATION_QUESTIONS, RELATION_CRITERIA, anchor_binding_spec, resolve_relation, resolve_equivalence,
    passes_probability_cutoff,
    relation_family_probabilities, reconcile_equivalence,
)
from open_video_summary.errors import (
    AuthenticationError, ProviderConfigurationError, RequestCancelledError,
    RunStoppedError, ServiceTimeoutError,
)


class InformationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket, "create_connection", side_effect=AssertionError("Network disabled"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def evaluator(self, *, low=False, signals=None, labels=None, malformed=False, distribution=None):
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
                # A borderline equivalence class leaves the family genuinely open.
                label = (labels or {}).get((left, right), "equivalent" if low else "complementary")
                mass = 0.89 if low else 0.99
                options = question["criteria"]
                probabilities = distribution or {option: mass if option == label else (1-mass)/(len(options)-1) for option in options}
                if distribution:
                    label = max(distribution, key=distribution.get)
                answers[identifier] = {"type": "choice", "choice": label,
                    "probabilities": probabilities,
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
        self.assertEqual("equivalent", relation.initial_relation)
        self.assertEqual(0.89, relation.initial_probability)
        self.assertEqual(signals, dict(relation.adjudication_signals))
        self.assertIsNone(relation.probability)
        self.assertEqual(.99, relation.adjudication_strength)
        self.assertEqual(2, len(requests))
        self.assertEqual(6, len(requests[1]["questions"]))
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
        run.evaluator, _ = self.evaluator(low=True, signals=negative | {
            "left_entails_right": .45, "right_entails_left": .45, "same_complete_meaning": .45})
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
        self.assertEqual(6, len(requests[-1]["questions"]))
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

    def test_native_float_boundary_and_unknown_subtype_do_not_create_count_uncertainty(self):
        native_false_info = {"left_entails_right": .07, "right_entails_left": .06,
            "incompatible": .10000000000000001, "correction_left": .04, "correction_right": .04,
            "same_complete_meaning": .04}
        self.assertEqual("complementary", resolve_relation(native_false_info, .90)[0])
        self.assertEqual("distinct", resolve_equivalence(native_false_info, .90)[0])
        self.assertFalse(passes_probability_cutoff(.89999, .90))
        boundary_above = native_false_info | {"incompatible": .10001}
        self.assertEqual("uncertain", resolve_relation(boundary_above, .90)[0])

        native_opinion_detail = {"left_entails_right": .46, "right_entails_left": .08,
            "incompatible": .08, "correction_left": .03, "correction_right": .04,
            "same_complete_meaning": .03}
        self.assertEqual("uncertain", resolve_relation(native_opinion_detail, .90)[0])
        self.assertEqual("distinct", resolve_equivalence(native_opinion_detail, .90)[0])
        run = make_run(["O italiano acha a medida absurda", "O italiano usa um smartphone chinês"])
        run.config = replace(run.config, max_relation_adjudications=1)
        run.evaluator, requests = self.evaluator(low=True, signals=native_opinion_detail)
        report = run.run()
        self.assertEqual("uncertain", report.relations[0].relation)
        self.assertEqual("distinct", report.relations[0].equivalence_state)
        self.assertEqual("completed", report.status)
        self.assertFalse(report.counts.counts_provisional)
        self.assertEqual(2, report.counts.unique_units)
        self.assertNotIn("relation_uncertain", [issue.kind for issue in report.issues])
        self.assertEqual(1, json.loads(report.metadata_json)["subtype_uncertain_relations"])
        self.assertFalse(json.loads(report.metadata_json)["descriptive_relations_complete"])

    def test_genuine_native_ambiguity_and_conflicting_checks_remain_provisional(self):
        native = {"left_entails_right": .52, "right_entails_left": .74,
            "incompatible": .07, "correction_left": .04, "correction_right": .05,
            "same_complete_meaning": .33}
        self.assertEqual("uncertain", resolve_equivalence(native, .90)[0])
        conflicts = [native | {"left_entails_right": .04, "same_complete_meaning": .99},
                     native | {"left_entails_right": .99, "right_entails_left": .99,
                               "same_complete_meaning": .04},
                     native | {"same_complete_meaning": .99, "incompatible": .99}]
        for conflicting in conflicts:
            with self.subTest(signals=conflicting):
                self.assertEqual(("uncertain", None, "conflicting_adjudication_signals"),
                                 resolve_equivalence(conflicting, .90))
        run = make_run(["O decreto alegava emergência nacional", "Trump alegou emergência nacional ao assinar o decreto"])
        run.config = replace(run.config, max_relation_adjudications=1)
        run.evaluator, _ = self.evaluator(low=True, signals=native)
        report = run.run()
        self.assertEqual("uncertain", report.relations[0].equivalence_state)
        self.assertEqual("partial", report.status)
        self.assertTrue(report.counts.counts_provisional)
        self.assertEqual(native, dict(report.relations[0].adjudication_signals))

    def test_complete_meaning_resolves_gray_directional_checks_without_repeating_requests(self):
        signals = dict.fromkeys(RELATION_ADJUDICATION_QUESTIONS, .01) | {
            "left_entails_right": .50, "right_entails_left": .60, "same_complete_meaning": .94}
        run = make_run(["O presidente relata a regra", "Donald Trump relata a regra"])
        run.config = replace(run.config, max_relation_adjudications=1)
        run.evaluator, requests = self.evaluator(low=True, signals=signals)
        report = run.run()
        self.assertEqual(2, len(requests))
        self.assertEqual(6, len(requests[1]["questions"]))
        self.assertEqual("equivalent", report.relations[0].equivalence_state)
        self.assertEqual("complete_meaning_check", report.relations[0].equivalence_origin)
        self.assertEqual(1, report.counts.unique_units)

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

    def test_retained_complete_assertion_binds_without_bypassing_any_fidelity_gate(self):
        source = "A parceria existe. Trump pode ter acabado com essa parceria. A Google atua com a Huawei."
        quote = "Trump pode ter acabado com essa parceria."
        text = "Trump pode ter acabado com essa parceria entre Google e Huawei."
        raw = candidate("v0:s0", source, quote=quote, text=text, modality="possibilidade")
        for failed_gate in (None, "support", "conditions", "negation", "quantities", "modality", "attribution"):
            with self.subTest(failed_gate=failed_gate):
                evaluator = SyntheticEvaluator()
                original = evaluator.evaluate
                def evaluate(context, noul=None, choice=None):
                    result = original(context, noul, choice)
                    return replace(result, noul=tuple(replace(item, probability=.01)
                        if item.id == failed_gate else item for item in result.noul))
                evaluator.evaluate = evaluate
                report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"): [raw]}),
                                 evaluator, qa_enabled=False)
                record = report.candidates[0]
                self.assertEqual("literal_assertion_sentence", record.anchor_binding_origin)
                self.assertFalse(any("anchor_binding" in questions for _, questions, _ in evaluator.calls))
                self.assertTrue({"support", "conditions", "negation", "quantities", "modality", "attribution"}
                                <= set(dict(record.signals)))
                self.assertEqual(failed_gate is None, record.validation == "accepted")

    def test_retained_sentence_proof_excludes_fragments_modified_wording_and_context_only_facts(self):
        source = "Segundo a rádio, Trump pode ter acabado com essa parceria. A Google é dona do Android."
        target = capture_snapshot(videos([source])).current_segments[0]
        run = _AnalysisRun(InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator()), capture_snapshot(videos([source])))
        for quote, text in (
            ("Trump pode ter acabado com essa parceria.", "Trump pode ter acabado com essa parceria entre empresas."),
            ("Segundo a rádio, Trump pode ter acabado com essa parceria.", "Segundo a rádio, Trump acabou com essa parceria entre empresas."),
            ("Segundo a rádio, Trump pode ter acabado com essa parceria.", "A Google é dona do Android."),
            ("parceria.", "A parceria entre empresas acabou."),
        ):
            with self.subTest(quote=quote, text=text):
                parsed, _ = run._literal_candidate(candidate(target.id, source, quote=quote, text=text),
                                                   target, (target.id,))
                self.assertFalse(run._literal_sentence_retained(text, source, parsed.evidence))

    def test_direct_windows_preserve_original_offsets_and_decompose_only_failed_window(self):
        sentences = ["A primeira regra estabelece backup automático diário com uma cópia no servidor.",
                     "A segunda regra exige retenção de trinta dias para cada cópia do servidor.",
                     "A terceira regra permite arquivar as cópias antigas em um sistema separado."]
        source = " ".join(sentences)
        failed = False
        def generate(data, route):
            nonlocal failed
            self.assertEqual("direct", route)
            self.assertEqual(source, data["target"]["text"])
            self.assertNotIn("accepted", data)
            window = data["discovery_window"]
            if not failed:
                failed = True
                raise ServiceTimeoutError("offline")
            return [candidate("v0:s0", source, quote=sentence) for sentence in sentences
                    if window["start_char"] <= source.index(sentence) < window["end_char"]]
        report = analyze(videos([source]), ScriptedGenerator(callback=generate),
                         qa_enabled=False, direct_window_chars=160, max_direct_windows=12)
        self.assertEqual("completed", report.status)
        self.assertEqual(3, report.counts.unique_units)
        windows = json.loads(report.metadata_json)["direct_source_windows"]
        self.assertEqual("failed", windows[0]["status"])
        completed = [row for row in windows if row["status"] == "completed"]
        self.assertEqual(list(range(len(source))), [position for row in completed
            for position in range(row["start_char"], row["end_char"])])
        self.assertEqual(1, len(report.coverage))
        self.assertFalse(json.loads(report.metadata_json)["qa_source_windows"])
        for record in report.candidates:
            for evidence in record.candidate.evidence:
                self.assertEqual(evidence.quote, source[evidence.start_char:evidence.end_char])

    def test_direct_window_limit_and_global_budget_never_claim_complete_discovery(self):
        source = "Uma regra comunicada sobre a cópia automática de todos os arquivos do servidor remoto. " * 3
        for settings, expected in (({"max_direct_windows":1}, "direct_window_budget_exhausted"),
                                   ({"max_calls":1}, "call_budget_exhausted")):
            with self.subTest(settings=settings):
                report = analyze(videos([source]), ScriptedGenerator(), qa_enabled=False,
                                 direct_window_chars=80, **settings)
                self.assertEqual("failed" if "max_calls" in settings else "partial", report.status)
                self.assertIn(expected, [issue.kind for issue in report.issues])
                self.assertTrue(report.counts.counts_provisional)
                if "max_calls" in settings:
                    self.assertEqual(1, json.loads(report.metadata_json)["logical_calls"])
                else:
                    self.assertEqual(1, len(report.coverage))

    def test_known_distinct_family_excludes_uncertain_mass_and_retains_conflicts(self):
        distribution = {"equivalent":.02, "complementary":.30, "more_specific_left":.33,
                        "more_specific_right":.08, "contradiction":.12, "correction_left":.12,
                        "correction_right":.01, "uncertain":.02}
        signals = {"left_entails_right":.43, "right_entails_left":.18, "incompatible":.23,
                   "correction_left":.10, "correction_right":.06, "same_complete_meaning":.13}
        self.assertEqual((.02, .96, .02), relation_family_probabilities(distribution.items()))
        for checks, expected in ((signals, "distinct"),
                                 (signals | {"same_complete_meaning":.96}, "uncertain"),
                                 (signals | {"left_entails_right":.02,"same_complete_meaning":.98}, "uncertain")):
            with self.subTest(checks=checks):
                run = make_run(["Google anunciou que não ia atualizar o sistema.", "Google suspendeu o sistema."])
                run.config = replace(run.config, max_relation_adjudications=1)
                run.evaluator, requests = self.evaluator(distribution=distribution, signals=checks)
                report = run.run()
                relation = report.relations[0]
                self.assertEqual(expected, relation.equivalence_state)
                self.assertEqual(.96, relation.primary_distinct_probability)
                self.assertEqual(.02, relation.primary_uncertain_probability)
                self.assertEqual(checks, dict(relation.adjudication_signals))
                self.assertEqual(2, len(requests))
                if expected == "distinct":
                    self.assertEqual("uncertain", relation.relation)
                    self.assertEqual("primary_relation_family", relation.equivalence_origin)
                    self.assertFalse(report.counts.counts_provisional)
                else:
                    self.assertTrue(report.counts.counts_provisional)
        unknown = distribution | {"uncertain":.94, "more_specific_left":.01,
            "complementary":.01, "more_specific_right":.00,"contradiction":.00,
            "correction_left":.01,"correction_right":.01}
        equivalent, distinct, _ = relation_family_probabilities(unknown.items())
        self.assertEqual("uncertain", reconcile_equivalence(signals,.90,equivalent,distinct)[0])
        boundary = distribution | {"uncertain":.08,"more_specific_left":.27}
        self.assertTrue(passes_probability_cutoff(relation_family_probabilities(boundary.items())[1],.90))
        self.assertEqual("uncertain", reconcile_equivalence(signals,.90,.02,.89999)[0])

    def test_rounded_primary_scores_normalize_before_family_and_subtype_decisions(self):
        empty = dict.fromkeys(RELATION_CRITERIA, 0.0)
        rounded = {"equivalent":.05, "complementary":.30, "more_specific_left":.30,
                   "more_specific_right":.10, "contradiction":.10, "correction_left":.05,
                   "correction_right":.05, "uncertain":.06}
        cases = [
            (rounded, "uncertain"),
            (empty | {"complementary":.90, "uncertain":.11}, "uncertain"),
            (empty | {"equivalent":.90, "uncertain":.11}, "uncertain"),
            (empty | {"equivalent":.04, "complementary":.891, "uncertain":.059}, "distinct"),
            (empty | {"equivalent":.891, "complementary":.04, "uncertain":.059}, "equivalent"),
        ]
        for distribution, expected in cases:
            for batch_size in (1, 4):
                with self.subTest(distribution=distribution, batch_size=batch_size):
                    run = make_run(["A regra é temporária", "O backup é diário", "O prazo é curto"])
                    run.config = replace(run.config, pair_batch_size=batch_size, max_relation_adjudications=3)
                    run.evaluator, _ = self.evaluator(distribution=distribution,
                        signals=dict.fromkeys(RELATION_ADJUDICATION_QUESTIONS, .5))
                    report = run.run()
                    total = math.fsum(distribution.values())
                    equivalent, distinct, unknown = relation_family_probabilities(distribution.items())
                    for relation in report.relations:
                        self.assertEqual(expected, relation.equivalence_state)
                        self.assertEqual(total, relation.primary_probability_total)
                        self.assertEqual(max(distribution.values()), relation.initial_probability)
                        self.assertEqual(equivalent, relation.primary_equivalent_probability)
                        self.assertEqual(distinct, relation.primary_distinct_probability)
                        self.assertEqual(unknown, relation.primary_uncertain_probability)
                        self.assertAlmostEqual(1, equivalent + distinct + unknown)
                        if relation.probability is not None:
                            self.assertEqual(max(distribution.values()) / total, relation.probability)
                    self.assertEqual(expected == "uncertain", report.counts.counts_provisional)
                    self.assertEqual("partial" if expected == "uncertain" else "completed", report.status)
                    self.assertEqual(1 if expected == "equivalent" else 3, report.counts.unique_units)
                    for call in report.calls:
                        for choice in json.loads(call.decisions_json)["choice"]:
                            self.assertEqual(distribution, dict(choice["probabilities"]))
                    if distribution == rounded:
                        self.assertAlmostEqual(.90 / 1.01, distinct)
                        self.assertAlmostEqual(.06 / 1.01, unknown)

    def test_rounded_family_keeps_strong_conflicting_followup_uncertain(self):
        distribution = {"equivalent":.02, "complementary":.30, "more_specific_left":.33,
                        "more_specific_right":.08, "contradiction":.12, "correction_left":.12,
                        "correction_right":.01, "uncertain":.03}
        gray = dict.fromkeys(RELATION_ADJUDICATION_QUESTIONS, .5)
        for changes, origin in (({"same_complete_meaning":.96}, "conflicting_primary_adjudication_signals"),
                                ({"left_entails_right":.02, "same_complete_meaning":.98},
                                 "conflicting_adjudication_signals")):
            with self.subTest(changes=changes):
                run = make_run(["A regra foi anunciada", "A regra entrou em vigor"])
                run.config = replace(run.config, max_relation_adjudications=1)
                run.evaluator, _ = self.evaluator(distribution=distribution, signals=gray | changes)
                report = run.run()
                relation = report.relations[0]
                self.assertAlmostEqual(.96 / 1.01, relation.primary_distinct_probability)
                self.assertEqual("uncertain", relation.equivalence_state)
                self.assertEqual(origin, relation.equivalence_origin)
                self.assertTrue(report.counts.counts_provisional)

    def test_invalid_relation_distributions_never_create_family_decisions(self):
        valid = dict.fromkeys(RELATION_CRITERIA, 0.0) | {"equivalent":1.0}
        invalid = [valid | {"equivalent":value} for value in (True, -.01, 1.01, math.nan, math.inf, "1")]
        invalid += [dict.fromkeys(RELATION_CRITERIA, 0.0), {"equivalent":1.0}, valid | {"extra":0.0}]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    relation_family_probabilities(values.items())
        with self.assertRaises(ValueError):
            relation_family_probabilities(tuple(valid.items()) + (("equivalent", 1.0),))
        # The real adapter also rejects totals outside its rounding tolerance.
        for mass in (0.0, .5):
            run = make_run(["A regra é temporária", "O backup é diário"])
            run.evaluator, _ = self.evaluator(distribution=valid | {"equivalent":mass})
            report = run.run()
            self.assertEqual((), report.relations)
            self.assertTrue(report.counts.counts_provisional)
            self.assertIn("pair_evaluation_failed", [issue.kind for issue in report.issues])

    def test_recovery_projection_deduplicates_literal_evidence_and_preserves_meanings(self):
        source = "Se houver falha, a cópia não é feita. A retenção é de 30 dias."
        raw = candidate("v0:s0", source, quote="Se houver falha, a cópia não é feita.",
                        conditions=("Se houver falha",), negated=True)
        report = analyze(videos([source]), ScriptedGenerator({("v0:s0","direct"):[raw]}), qa_enabled=False)
        base = report.candidates[0]
        run = _AnalysisRun(InformationAnalyzer(ScriptedGenerator(),SyntheticEvaluator()),report.snapshot)
        run.candidates = [replace(base,id=f"a{i}") for i in range(10)]
        run.candidates += [replace(base,id=f"r{i}",validation="needs_review",reasons=("modality",)) for i in range(20)]
        target = run.segments["v0:s0"]
        projection = run._recovery_projection(target)
        self.assertEqual(1,len(projection["accepted"]))
        self.assertEqual(1,len(projection["repair_candidates"]))
        self.assertEqual(1,len(projection["evidence_table"]))
        accepted = projection["accepted"][0]
        self.assertEqual(raw["text"], accepted["text"])
        self.assertEqual(asdict(base.candidate.qualifiers), accepted["qualifiers"])
        self.assertEqual(10,len(accepted["ids"]))
        self.assertEqual(20,len(projection["repair_candidates"][0]["ids"]))
        evidence = projection["evidence_table"][accepted["evidence"][0]]
        self.assertEqual(raw["evidence"][0],evidence)
        self.assertEqual(source,target.content)
        self.assertLess(len(json.dumps(projection)),3000)

    def test_context_limit_has_operation_size_and_does_not_consume_call_budget(self):
        report = analyze(videos(["x"*5000]), ScriptedGenerator(), qa_enabled=False,
                         max_context_chars=4000, max_calls=99)
        self.assertEqual(0,json.loads(report.metadata_json)["logical_calls"])
        failures = [issue for issue in report.issues if issue.kind == "context_budget_exceeded"]
        self.assertTrue(any("extract_direct requires" in issue.detail and "limit is 4000" in issue.detail
                            and "99 logical calls" in issue.detail for issue in failures))
        self.assertTrue(all(issue.segment_ids == ("v0:s0",) for issue in failures))
        self.assertIn("target_context_limited",[issue.kind for issue in report.issues])
        self.assertNotIn("_LimitReached",[issue.detail for issue in report.issues])

    def test_model_window_mismatch_is_audited_without_replacing_trusted_ownership(self):
        source = "Uma informação importante sobre o prazo. Outra informação independente sobre a cópia."
        class EchoGenerator(ScriptedGenerator):
            def generate(self, request):
                result = super().generate(request)
                return replace(result, value=dict(result.value, issues=[{
                    "kind":"discovery_window_bounds_mismatch", "detail":"The model counted different offsets.",
                    "segment_ids":["v0:s0"]}]))
        def generate(data, route):
            window = data["discovery_window"]
            return [candidate("v0:s0", source, quote=sentence) for sentence in
                    ("Uma informação importante sobre o prazo.", "Outra informação independente sobre a cópia.")
                    if window["start_char"] <= source.index(sentence) < window["end_char"]]
        report = analyze(videos([source]), EchoGenerator(callback=generate),qa_enabled=False,direct_window_chars=80)
        self.assertEqual("completed",report.status)
        self.assertEqual(2,report.counts.accepted_candidates)
        rows=json.loads(report.metadata_json)["discovery_window_declarations"]
        self.assertEqual(2,len(rows))
        self.assertTrue(all(row["request_slice_verified"] for row in rows))
        self.assertTrue(all(row["declared_issues"][0]["kind"] == "discovery_window_bounds_mismatch" for row in rows))

    def test_actual_outside_window_retains_declared_and_resolved_offsets(self):
        first = "Uma informação importante sobre o prazo."
        second = "Outra informação independente sobre a cópia."
        source = first + " " + second
        raw = candidate("v0:s0",source,quote=second,offset=source.index(second)-1)
        def generate(data,route):
            return [raw] if data["discovery_window"]["start_char"] == 0 else []
        report=analyze(videos([source]),ScriptedGenerator(callback=generate),qa_enabled=False,
                       direct_window_chars=80,max_literal_repairs=4)
        record=report.candidates[0]
        self.assertEqual("needs_review",record.validation)
        self.assertEqual(("direct_window_evidence_outside",),record.reasons)
        self.assertEqual(source.index(second),record.candidate.evidence[0].start_char)
        self.assertEqual(source.index(second)-1,record.evidence_resolutions[0].supplied_start_char)
        self.assertEqual(source.index(second),record.evidence_resolutions[0].resolved_start_char)
        self.assertFalse(any(c.route == "literal_recovery" for c in report.candidates))
        self.assertEqual((record.id,),next(i for i in report.issues if i.kind == "direct_window_evidence_outside").candidate_ids)

    def test_literal_repairs_keep_originals_provenance_qualifiers_and_global_bounds(self):
        source = "Trump pode ter acabado com essa parceria. A empresa pode mudar a regra."
        raw = candidate("v0:s0",source,quote="Trump pode ter acabado com essa parceria.",
                        text="Trump pode ter acabado com a parceria entre as empresas.",modality="possibilidade")
        evaluator = SyntheticEvaluator(validation={raw["text"]:.5})
        other_context = copy.deepcopy(raw)
        context_quote = "A empresa pode mudar a regra."
        other_context["evidence"].append({"segment_id":"v0:s0","quote":context_quote,
            "start_char":source.index(context_quote),"end_char":len(source),"role":"context"})
        report = analyze(videos([source]),ScriptedGenerator({("v0:s0","direct"):[raw,raw,other_context]}),
                         evaluator,qa_enabled=False,max_literal_repairs=4)
        repairs = [record for record in report.candidates if record.route == "literal_recovery"]
        self.assertEqual(1,len(repairs))
        repair=repairs[0]
        self.assertEqual("c0",repair.literal_repair_of)
        self.assertEqual(raw["evidence"][0]["quote"],repair.candidate.text)
        self.assertEqual(raw, json.loads(report.candidates[0].raw_json))
        self.assertEqual("accepted",repair.validation)
        self.assertTrue({"support","conditions","negation","quantities","modality","attribution"} <= set(dict(repair.signals)))
        limited = analyze(videos([source]),ScriptedGenerator({("v0:s0","direct"):[raw]}),
                          SyntheticEvaluator(validation={raw["text"]:.5}),qa_enabled=False,
                          max_literal_repairs=4,max_calls=3)
        self.assertFalse(any(record.route == "literal_recovery" for record in limited.candidates))
        self.assertEqual(3,json.loads(limited.metadata_json)["logical_calls"])
        self.assertEqual(1,len(limited.coverage))

    def test_literal_repair_limit_does_not_bypass_atomicity_or_fragment_ownership(self):
        compound="Os backups são feitos a cada 24 horas e guardados por 30 dias."
        condition="Se a conexão cair, o backup não é realizado."
        source=compound+" "+condition
        expanded=candidate("v0:s0",source,quote=compound,text="O sistema faz backups diários.")
        fragment=candidate("v0:s0",source,quote="o backup não é realizado.",text="O sistema não realiza backup.")
        evaluator=SyntheticEvaluator(validation={expanded["text"]:.5,fragment["text"]:.5},
                                      granularity={compound:"compound"})
        report=analyze(videos([source]),ScriptedGenerator({("v0:s0","direct"):[expanded,fragment]}),
                       evaluator,qa_enabled=False,max_literal_repairs=1)
        repairs=[c for c in report.candidates if c.route == "literal_recovery"]
        self.assertEqual(1,len(repairs))
        self.assertEqual("needs_repair",repairs[0].validation)
        self.assertEqual("compound",repairs[0].granularity)
        self.assertEqual(0,report.counts.accepted_candidates)

    def test_literal_repair_context_limit_preserves_failed_candidate_and_runs_affordable_audit(self):
        source = "A regra estabelece " + ("conteudo " * 330) + "final."
        raw = candidate("v0:s0", source, text="Uma regra declarada.")
        report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"):[raw]}),
                         SyntheticEvaluator(validation={raw["text"]:.5}), qa_enabled=False,
                         direct_window_chars=4000, max_literal_repairs=1, max_context_chars=7136)
        original, repair = report.candidates
        self.assertEqual(raw, json.loads(original.raw_json))
        self.assertEqual("needs_review", original.validation)
        self.assertEqual("literal_recovery", repair.route)
        self.assertEqual(original.id, repair.literal_repair_of)
        self.assertEqual(source, repair.candidate.text)
        self.assertEqual("not_evaluated", repair.validation)
        self.assertEqual(("_ContextLimitReached",), repair.reasons)
        self.assertEqual(source, report.snapshot.current_segments[0].content)
        self.assertEqual(1, len(report.coverage))
        self.assertEqual("no_gap_signaled", report.coverage[0].state)
        self.assertEqual(3, json.loads(report.metadata_json)["logical_calls"])
        self.assertEqual(["extract_direct", "bind_candidate_anchor", "coverage_audit"],
                         [call.operation for call in report.calls])
        issues = {issue.kind:issue for issue in report.issues}
        self.assertIn("validate_candidate requires 8571 characters; limit is 7136", issues["context_budget_exceeded"].detail)
        self.assertIn("254 logical calls remain", issues["context_budget_exceeded"].detail)
        self.assertEqual((original.id,), issues["literal_repair_failed"].candidate_ids)
        self.assertNotIn("coverage_not_audited", issues)
        self.assertNotIn("target_context_limited", issues)
        self.assertNotIn("call_budget_exhausted", issues)
        self.assertEqual("partial", report.status)
        self.assertTrue(report.counts.counts_provisional)

    def test_literal_repair_still_propagates_run_stops_and_terminal_provider_errors(self):
        source = "A regra define a frequência de cópia."
        raw = candidate("v0:s0", source, text="Uma regra declarada.")
        report = analyze(videos([source]), ScriptedGenerator({("v0:s0", "direct"):[raw]}),
                         SyntheticEvaluator(validation={raw["text"]:.5}), qa_enabled=False)
        for error in (_LimitReached, _ProviderStopped, RequestCancelledError, RunStoppedError,
                      AuthenticationError, ProviderConfigurationError):
            with self.subTest(error=error):
                run = _AnalysisRun(InformationAnalyzer(ScriptedGenerator(), SyntheticEvaluator(),
                    InformationAnalysisConfig(qa_enabled=False, max_literal_repairs=1)), report.snapshot)
                run.candidates = list(report.candidates)
                target = report.snapshot.current_segments[0]
                with patch.object(run, "_direct_extract"), patch.object(run, "_validate", side_effect=error()), \
                        patch.object(run, "_audit") as audit:
                    with self.assertRaises(error):
                        run._target(target)
                    audit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
